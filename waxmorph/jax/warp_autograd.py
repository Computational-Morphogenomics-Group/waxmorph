"""JAX autodiff bridge for Warp physics kernels.

The PyTorch backend can hand tensors to Warp through ``torch.autograd.Function``
and replay a ``wp.Tape`` in backward.  JAX cannot use that mechanism directly,
so this module wraps small, shape-regular Warp kernels with Warp's experimental
JAX FFI bridge.  Topology stays frozen outside autodiff, while the per-pair
physics math runs through Warp and has a JAX VJP.
"""

from functools import cache

import jax
import jax.numpy as jnp
import warp as wp

from waxmorph.constants import EPS_DIST, EPS_NORM

# Emulator-specific force constants. Keep these in sync with waxmorph.emulator
# without importing its @wp.func helpers across modules.
K_REP = 2.0
K_ATT = 0.5


def _device_requires_cuda(device: str | wp.Device | None) -> str:
    """Validate that the Warp/JAX bridge can run on a CUDA-backed JAX device."""
    device_name = "cuda" if device is None else str(device)
    if not device_name.startswith("cuda"):
        raise RuntimeError(
            "Differentiable JAX/Warp physics requires a CUDA device. "
            f"Received device={device_name!r}."
        )

    try:
        wp_device = wp.get_device(device_name)
    except Exception as exc:  # pragma: no cover - depends on local CUDA setup.
        raise RuntimeError(f"Warp CUDA device {device_name!r} is not available.") from exc

    if not wp_device.is_cuda:
        raise RuntimeError(
            "Differentiable JAX/Warp physics requires a CUDA Warp device. "
            f"Resolved device={wp_device!r}."
        )

    try:
        gpu_devices = jax.devices("gpu")
    except Exception as exc:  # pragma: no cover - depends on local CUDA setup.
        raise RuntimeError(
            "Differentiable JAX/Warp physics requires a JAX GPU backend, "
            "but JAX could not initialize one."
        ) from exc

    if not gpu_devices:
        raise RuntimeError("Differentiable JAX/Warp physics requires a JAX GPU backend.")

    return device_name


def _sticky_pair_forces_kernel_impl(
    X_i: wp.array(dtype=wp.vec3f),
    X_j: wp.array(dtype=wp.vec3f),
    R_i: wp.array(dtype=wp.float32),
    R_j: wp.array(dtype=wp.float32),
    F_i: wp.array(dtype=wp.vec3f),
    F_j: wp.array(dtype=wp.vec3f),
):
    e = wp.tid()
    d = X_i[e] - X_j[e]
    dist = wp.length(d) + EPS_NORM
    u = d / dist

    rs = R_i[e] + R_j[e]
    drep = rs - EPS_DIST
    datr = rs + EPS_DIST

    f_rep = K_REP * wp.max(drep - dist, wp.float32(0.0))
    f_att = K_ATT * wp.max(datr - dist, wp.float32(0.0)) * wp.float32(dist > drep)

    f_ij = (f_rep - f_att) * u
    F_i[e] = f_ij
    F_j[e] = -f_ij


def _gene_pair_flux_kernel_impl(
    G_i: wp.array2d(dtype=wp.float32),
    G_j: wp.array2d(dtype=wp.float32),
    flux_i: wp.array2d(dtype=wp.float32),
    flux_j: wp.array2d(dtype=wp.float32),
):
    e, g = wp.tid()
    flux = G_j[e, g] - G_i[e, g]
    flux_i[e, g] = flux
    flux_j[e, g] = -flux


@cache
def _sticky_pair_forces_jax():
    from warp.jax_experimental.ffi import jax_kernel

    return jax_kernel(
        wp.kernel(_sticky_pair_forces_kernel_impl),
        num_outputs=2,
        enable_backward=True,
    )


@cache
def _gene_pair_flux_jax():
    from warp.jax_experimental.ffi import jax_kernel

    return jax_kernel(
        wp.kernel(_gene_pair_flux_kernel_impl),
        num_outputs=2,
        enable_backward=True,
    )


def warp_mech_step(
    X: jax.Array,
    R: jax.Array,
    pair_i: jax.Array,
    pair_j: jax.Array,
    dt: float,
    *,
    num_pairs: jax.Array | int | None = None,
    device: str | wp.Device | None = "cuda",
) -> jax.Array:
    """Apply one differentiable sticky-sphere mechanics step.

    ``pair_i`` and ``pair_j`` are frozen unordered neighbor pairs.  The force
    law itself runs as a Warp kernel wrapped in a JAX custom VJP; JAX performs
    the scatter aggregation back to particles.
    """
    _device_requires_cuda(device)

    pair_count = int(pair_i.shape[0])
    if pair_count == 0:
        return X
    if num_pairs is None:
        num_pairs = jnp.asarray(pair_count, dtype=jnp.int32)
    else:
        num_pairs = jnp.asarray(num_pairs, dtype=jnp.int32)

    pair_i = jnp.asarray(pair_i, dtype=jnp.int32)
    pair_j = jnp.asarray(pair_j, dtype=jnp.int32)

    force_i, force_j = _sticky_pair_forces_jax()(
        X[pair_i],
        X[pair_j],
        R[pair_i],
        R[pair_j],
    )
    real_pair_mask = jnp.arange(pair_count, dtype=jnp.int32) < num_pairs
    force_i = jnp.where(real_pair_mask[:, None], force_i, 0.0)
    force_j = jnp.where(real_pair_mask[:, None], force_j, 0.0)
    gx = jnp.zeros_like(X)
    gx = gx.at[pair_i].add(force_i)
    gx = gx.at[pair_j].add(force_j)
    return X + jnp.asarray(dt, dtype=X.dtype) * gx


def warp_diffusion_step(
    G: jax.Array,
    pair_i: jax.Array,
    pair_j: jax.Array,
    alpha: float,
    dt: float,
    *,
    num_pairs: jax.Array | int | None = None,
    device: str | wp.Device | None = "cuda",
) -> jax.Array:
    """Apply one differentiable graph-Laplacian gene diffusion step."""
    _device_requires_cuda(device)

    pair_count = int(pair_i.shape[0])
    if pair_count == 0:
        return G
    if num_pairs is None:
        num_pairs = jnp.asarray(pair_count, dtype=jnp.int32)
    else:
        num_pairs = jnp.asarray(num_pairs, dtype=jnp.int32)

    pair_i = jnp.asarray(pair_i, dtype=jnp.int32)
    pair_j = jnp.asarray(pair_j, dtype=jnp.int32)

    flux_i, flux_j = _gene_pair_flux_jax()(G[pair_i], G[pair_j])
    real_pair_mask = jnp.arange(pair_count, dtype=jnp.int32) < num_pairs
    flux_i = jnp.where(real_pair_mask[:, None], flux_i, 0.0)
    flux_j = jnp.where(real_pair_mask[:, None], flux_j, 0.0)
    lap = jnp.zeros_like(G)
    lap = lap.at[pair_i].add(flux_i)
    lap = lap.at[pair_j].add(flux_j)

    scale = jnp.asarray(dt * alpha, dtype=G.dtype)
    return jnp.maximum(G + scale * lap, jnp.asarray(0.0, dtype=G.dtype))


__all__ = ["warp_diffusion_step", "warp_mech_step"]
