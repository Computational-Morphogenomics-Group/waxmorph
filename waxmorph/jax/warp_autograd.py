"""JAX/Warp custom-VJP bridge for frozen-pair physics on CUDA.

Backward-enabled Warp FFI differentiates dedicated per-pair kernels; JAX owns endpoint
gathers, masking, scatter aggregation, and state updates. ``pair_i`` and ``pair_j`` are
constructed outside autodiff, so gradients treat topology as fixed. PyTorch records its
emulator kernels on a :class:`warp.Tape`; JAX uses per-pair FFI kernel objects.

Top-level JAX APIs are preferred; a lazy legacy fallback supports the declared Warp 1.10
minimum.
"""

from functools import cache

import jax
import jax.numpy as jnp
import warp as wp

from waxmorph.constants import EPS_DIST, EPS_NORM

# Separate JAX FFI kernels duplicate these emulator coefficients; keep them aligned.
K_REP = 2.0
K_ATT = 0.5


@cache
def _load_jax_kernel():
    jax_kernel = getattr(wp, "jax_kernel", None)
    if jax_kernel is None:
        from warp.jax_experimental.ffi import jax_kernel
    return jax_kernel


def _device_requires_cuda(device: str | wp.Device | None) -> str:
    """Require available CUDA-capable Warp and JAX backends.

    ``None`` selects ``"cuda"``. The bridge requires initialized CUDA backends and raises
    ``RuntimeError`` for other device configurations.
    """
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


def _molecule_pair_flux_kernel_impl(
    c_i: wp.array2d(dtype=wp.float32),
    c_j: wp.array2d(dtype=wp.float32),
    flux_i: wp.array2d(dtype=wp.float32),
    flux_j: wp.array2d(dtype=wp.float32),
):
    e, g = wp.tid()
    flux = c_j[e, g] - c_i[e, g]
    flux_i[e, g] = flux
    flux_j[e, g] = -flux


@cache
def _sticky_pair_forces_jax():
    return _load_jax_kernel()(
        wp.kernel(_sticky_pair_forces_kernel_impl),
        num_outputs=2,
        enable_backward=True,
    )


@cache
def _molecule_pair_flux_jax():
    return _load_jax_kernel()(
        wp.kernel(_molecule_pair_flux_kernel_impl),
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
    r"""Apply one differentiable sticky-sphere mechanics step.

    For a frozen pair :math:`(i,j)`, define raw displacement
    :math:`\mathbf d=\mathbf x_i-\mathbf x_j`, stabilized distance
    :math:`r=\lVert\mathbf d\rVert+\epsilon_n`, regularized direction
    :math:`\mathbf u=\mathbf d/r`, and :math:`s=R_i+R_j`:

    .. math::

        f_{rep} &= k_{rep}\,\max(s - \epsilon_d - r,\; 0) \\
        f_{att} &= k_{att}\,\max(s + \epsilon_d - r,\; 0)
                   \cdot \mathbb{1}[r > s - \epsilon_d] \\
        \mathbf{F}_{i} &= (f_{rep} - f_{att})\,\mathbf{u},
        \quad \mathbf{F}_{j} = -\mathbf{F}_{i}

    Here :math:`k_{rep}=2`, :math:`k_{att}=0.5`, :math:`\epsilon_d` is
    ``EPS_DIST``, and :math:`\epsilon_n` is ``EPS_NORM``. JAX scatter-adds endpoint forces
    and applies :math:`\mathbf x\leftarrow\mathbf x+\mathrm{dt}\,\mathbf F`.

    The custom VJP differentiates ``X`` and ``R`` through per-pair forces while pair
    selection remains a fixed forward input. ``max`` has zero derivative on inactive
    branches; the strict adhesion indicator is constant in backward. Values at their
    nonsmooth boundaries follow Warp's branch convention.

    ``pair_i`` and ``pair_j`` are unordered frozen endpoints. For a padded pair buffer,
    ``num_pairs`` masks indices beyond the real count; omitted means all pairs are real. An
    empty buffer returns ``X`` unchanged.

    Raises:
        RuntimeError: If CUDA-backed Warp/JAX device validation fails.
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
    f_net = jnp.zeros_like(X)
    f_net = f_net.at[pair_i].add(force_i)
    f_net = f_net.at[pair_j].add(force_j)
    return X + jnp.asarray(dt, dtype=X.dtype) * f_net


def warp_diffusion_step(
    c: jax.Array,
    pair_i: jax.Array,
    pair_j: jax.Array,
    D_emu: float,
    dt: float,
    *,
    num_pairs: jax.Array | int | None = None,
    device: str | wp.Device | None = "cuda",
) -> jax.Array:
    r"""Apply one differentiable graph-Laplacian signaling-molecule diffusion step.

    Each frozen pair contributes antisymmetric flux
    :math:`\phi_{ij,g}=c_{j,g}-c_{i,g}`. JAX scatter-adds these values into
    :math:`(Lc)_{i,g}=\sum_{j\in\mathcal N(i)}(c_{j,g}-c_{i,g})`, then applies

    .. math::

        c_{i,g} \leftarrow \max\!\big(c_{i,g}
            + \mathrm{dt}\,D\,(L c)_{i,g},\; 0\big)

    where :math:`D` is ``D_emu``. The custom VJP differentiates concentrations through the
    per-pair flux, while pair selection remains fixed. At the JAX output clamp, the derivative
    with respect to the pre-clamp value is zero below zero, one above zero, and one half at
    exactly zero.

    ``num_pairs`` masks padded indices beyond the real frozen-pair count; omitted means every
    pair is real. An empty pair buffer returns ``c`` unchanged.

    Raises:
        RuntimeError: If CUDA-backed Warp/JAX device validation fails.
    """
    _device_requires_cuda(device)

    pair_count = int(pair_i.shape[0])
    if pair_count == 0:
        return c
    if num_pairs is None:
        num_pairs = jnp.asarray(pair_count, dtype=jnp.int32)
    else:
        num_pairs = jnp.asarray(num_pairs, dtype=jnp.int32)

    pair_i = jnp.asarray(pair_i, dtype=jnp.int32)
    pair_j = jnp.asarray(pair_j, dtype=jnp.int32)

    flux_i, flux_j = _molecule_pair_flux_jax()(c[pair_i], c[pair_j])
    real_pair_mask = jnp.arange(pair_count, dtype=jnp.int32) < num_pairs
    flux_i = jnp.where(real_pair_mask[:, None], flux_i, 0.0)
    flux_j = jnp.where(real_pair_mask[:, None], flux_j, 0.0)
    lap = jnp.zeros_like(c)
    lap = lap.at[pair_i].add(flux_i)
    lap = lap.at[pair_j].add(flux_j)

    scale = jnp.asarray(dt * D_emu, dtype=c.dtype)
    return jnp.maximum(c + scale * lap, jnp.asarray(0.0, dtype=c.dtype))


__all__ = ["warp_diffusion_step", "warp_mech_step"]
