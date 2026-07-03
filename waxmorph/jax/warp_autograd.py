"""JAX autodiff bridge that makes Warp physics steps differentiable.

Mental model: Warp owns the per-pair physics math, JAX owns the gradient
graph and the scatter aggregation back to particles. The PyTorch backend
hands tensors to Warp through a :class:`torch.autograd.Function` and replays a
:class:`warp.Tape` in backward; JAX has no equivalent tape mechanism, so this
module instead wraps small, shape-regular Warp kernels with Warp's
experimental JAX FFI (``jax_kernel(..., enable_backward=True)``). That wrapper
makes each kernel a JAX primitive with a custom VJP: the forward pass records
the kernel as the primitive's forward rule and the custom VJP supplies the
backward rule, so ``jax.grad``/``jax.vjp`` differentiate straight through the
Warp math. Neighbor topology is frozen outside autodiff (precomputed
``pair_i`` / ``pair_j`` edge lists); only the per-edge physics is
differentiated, and JAX performs the scatter-add aggregation onto particles.

This is the JAX parity backend, reached only via explicit imports; it keeps
gradient flow equivalent to the PyTorch default.

Notes:
    Backend divergence: this path hard-requires a CUDA Warp/JAX device (see
    :func:`_device_requires_cuda`) and raises :class:`RuntimeError` otherwise.
    The PyTorch twin imposes no such restriction.

See Also:
    :mod:`waxmorph.torch.warp_autograd`: PyTorch default backend, which reaches
        the same Warp physics through a :class:`torch.autograd.Function` that
        records and replays a :class:`warp.Tape`.
"""

from functools import cache

import jax
import jax.numpy as jnp
import warp as wp

from waxmorph.constants import EPS_DIST, EPS_NORM

# Force constants duplicated from waxmorph.emulator (avoids importing its
# @wp.func helpers across modules); must stay in sync with that source.
K_REP = 2.0
K_ATT = 0.5


def _device_requires_cuda(device: str | wp.Device | None) -> str:
    """Validate that the Warp/JAX bridge can run on a CUDA-backed JAX device.

    The FFI bridge only differentiates Warp kernels on CUDA, so this guards
    every public step against silently running on a CPU device.

    Args:
        device: Target device. ``None`` is treated as ``"cuda"``. A string is
            matched by prefix (must start with ``"cuda"``); a
            :class:`warp.Device` is resolved and checked for CUDA support.

    Returns:
        The validated device name (e.g. ``"cuda"`` or ``"cuda:0"``).

    Raises:
        RuntimeError: If the requested device is not CUDA, the Warp CUDA device
            cannot be resolved, or JAX has no initialized GPU backend.
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
    e = wp.tid()  # one thread per neighbor pair (edge)
    d = X_i[e] - X_j[e]
    dist = wp.length(d) + EPS_NORM
    u = d / dist  # unit separation direction

    rs = R_i[e] + R_j[e]
    drep = rs - EPS_DIST  # repulsion onset (overlap)
    datr = rs + EPS_DIST  # adhesion cutoff

    f_rep = K_REP * wp.max(drep - dist, wp.float32(0.0))
    # adhesion only past contact (gated off while still overlapping)
    f_att = K_ATT * wp.max(datr - dist, wp.float32(0.0)) * wp.float32(dist > drep)

    # equal-and-opposite force on the pair endpoints
    f_ij = (f_rep - f_att) * u
    F_i[e] = f_ij
    F_j[e] = -f_ij


def _molecule_pair_flux_kernel_impl(
    c_i: wp.array2d(dtype=wp.float32),
    c_j: wp.array2d(dtype=wp.float32),
    flux_i: wp.array2d(dtype=wp.float32),
    flux_j: wp.array2d(dtype=wp.float32),
):
    e, g = wp.tid()  # per (edge, molecule channel)
    # concentration difference along the edge; antisymmetric across endpoints
    flux = c_j[e, g] - c_i[e, g]
    flux_i[e, g] = flux
    flux_j[e, g] = -flux


@cache
def _sticky_pair_forces_jax():
    # wrap the Warp kernel as a JAX primitive with a custom VJP (enable_backward)
    from warp.jax_experimental.ffi import jax_kernel

    return jax_kernel(
        wp.kernel(_sticky_pair_forces_kernel_impl),
        num_outputs=2,
        enable_backward=True,
    )


@cache
def _molecule_pair_flux_jax():
    # wrap the Warp kernel as a JAX primitive with a custom VJP (enable_backward)
    from warp.jax_experimental.ffi import jax_kernel

    return jax_kernel(
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

    Warp implementation of overdamped sticky-sphere mechanics: a short-range
    repulsion keeps particles from interpenetrating and a slightly longer-range
    adhesion holds contacting neighbors together. The per-edge force law runs
    as a Warp kernel wrapped in a JAX custom VJP (see the module docstring),
    while JAX scatter-adds the edge forces onto particles and applies the
    explicit Euler position update.

    For an edge between particles :math:`i` and :math:`j` with separation
    :math:`\mathbf{d} = \mathbf{x}_i - \mathbf{x}_j`, distance
    :math:`r = \lVert\mathbf{d}\rVert + \epsilon_n`, unit direction
    :math:`\hat{\mathbf{u}} = \mathbf{d}/r`, and summed radii
    :math:`s = R_i + R_j`:

    .. math::

        f_{rep} &= k_{rep}\,\max(s - \epsilon_d - r,\; 0) \\
        f_{att} &= k_{att}\,\max(s + \epsilon_d - r,\; 0)
                   \cdot \mathbb{1}[r > s - \epsilon_d] \\
        \mathbf{F}_{i} &= (f_{rep} - f_{att})\,\hat{\mathbf{u}},
        \quad \mathbf{F}_{j} = -\mathbf{F}_{i}

    where :math:`k_{rep}` is ``K_REP``, :math:`k_{att}` is ``K_ATT``,
    :math:`\epsilon_d` is ``EPS_DIST``, :math:`\epsilon_n` is ``EPS_NORM``,
    and :math:`\mathbb{1}[\cdot]` gates adhesion off while still overlapping.
    The net force is summed over edges and integrated as
    :math:`\mathbf{x}_i \leftarrow \mathbf{x}_i + \mathrm{dt}\,\mathbf{F}_i`.

    Gradient convention (the law is non-smooth): each ``max(., 0)`` contributes
    zero gradient on its clamped (inactive) branch, and the
    :math:`\mathbb{1}[r > s - \epsilon_d]` adhesion gate is treated as a
    constant indicator, so no gradient flows through the gate switch itself.

    Args:
        X: Particle positions, shape ``[N, 3]``.
        R: Particle radii, shape ``[N]``.
        pair_i: First endpoint index of each frozen neighbor edge, shape
            ``[P]``. Edges are unordered; ``(pair_i, pair_j)`` is precomputed
            outside autodiff and held fixed for the step.
        pair_j: Second endpoint index of each edge, shape ``[P]``.
        dt: Mechanics Euler step size.
        num_pairs: Number of *real* edges when the edge buffers are
            statically over-allocated to ``P`` for JIT shape stability. Edges
            at index ``>= num_pairs`` are padding and are masked to zero force
            so they do not perturb the update. Defaults to all ``P`` edges
            being real.
        device: Target device; validated to be CUDA-backed.

    Returns:
        Updated positions, shape ``[N, 3]``. Returned unchanged if there are
        no edges (``P == 0``).

    Raises:
        RuntimeError: If ``device`` is not a CUDA-backed Warp/JAX device.

    See Also:
        :class:`waxmorph.torch.warp_autograd.WarpMechStep`: PyTorch twin using
            a :class:`warp.Tape` instead of a JAX custom VJP.
        :func:`waxmorph.emulator.mech_step_sticky_differentiable`: the Warp
            tape-recording mechanics step underlying the PyTorch path.
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

    # gather pair endpoints, compute per-edge forces via the Warp VJP kernel
    force_i, force_j = _sticky_pair_forces_jax()(
        X[pair_i],
        X[pair_j],
        R[pair_i],
        R[pair_j],
    )
    # zero out padded edges (static pair buffer over-allocates to num_pairs)
    real_pair_mask = jnp.arange(pair_count, dtype=jnp.int32) < num_pairs
    force_i = jnp.where(real_pair_mask[:, None], force_i, 0.0)
    force_j = jnp.where(real_pair_mask[:, None], force_j, 0.0)
    # scatter-add edge forces back onto particles, then explicit Euler step
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

    Warp implementation of explicit (forward-Euler) diffusion on the contact
    graph: signaling-molecule concentrations relax toward those of contacting
    neighbors. The per-edge concentration difference runs as a Warp kernel
    wrapped in a JAX custom VJP (see the module docstring), while JAX
    scatter-adds those differences to form the graph Laplacian per particle and
    applies the Euler update.

    For an edge between particles :math:`i` and :math:`j` and molecule channel
    :math:`g`, the antisymmetric flux is
    :math:`\phi_{ij,g} = c_{j,g} - c_{i,g}`. Summing incident edge fluxes gives
    the graph Laplacian :math:`(L c)_{i,g} = \sum_{j \in \mathcal{N}(i)}
    (c_{j,g} - c_{i,g})`, and the update is:

    .. math::

        c_{i,g} \leftarrow \max\!\big(c_{i,g}
            + \mathrm{dt}\,D\,(L c)_{i,g},\; 0\big)

    where :math:`D` is ``D_emu``. The final :math:`\max(\cdot, 0)` clamps
    concentrations non-negative.

    Gradient convention (the clamp is non-smooth): the non-negativity
    :math:`\max(\cdot, 0)` contributes zero gradient wherever it is active
    (i.e. where the pre-clamp concentration is negative), passing gradient
    through unchanged otherwise.

    Args:
        c: Per-particle concentrations, shape ``[N, num_molecules]``.
        pair_i: First endpoint index of each frozen neighbor edge, shape
            ``[P]``. Edges are unordered; ``(pair_i, pair_j)`` is precomputed
            outside autodiff and held fixed for the step.
        pair_j: Second endpoint index of each edge, shape ``[P]``.
        D_emu: Diffusion coefficient :math:`D`.
        dt: Diffusion Euler step size.
        num_pairs: Number of *real* edges when the edge buffers are
            statically over-allocated to ``P`` for JIT shape stability. Edges
            at index ``>= num_pairs`` are padding and are masked to zero flux
            so they do not perturb the Laplacian. Defaults to all ``P`` edges
            being real.
        device: Target device; validated to be CUDA-backed.

    Returns:
        Updated concentrations, shape ``[N, num_molecules]``. Returned
        unchanged if there are no edges (``P == 0``).

    Raises:
        RuntimeError: If ``device`` is not a CUDA-backed Warp/JAX device.

    See Also:
        :class:`waxmorph.torch.warp_autograd.WarpDiffusionStep`: PyTorch twin
            using a :class:`warp.Tape` instead of a JAX custom VJP.
        :func:`waxmorph.emulator.diffusion_step_differentiable`: the Warp
            tape-recording diffusion step underlying the PyTorch path.
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

    # per-edge concentration differences via the Warp VJP kernel
    flux_i, flux_j = _molecule_pair_flux_jax()(c[pair_i], c[pair_j])
    # zero out padded edges (static pair buffer over-allocates to num_pairs)
    real_pair_mask = jnp.arange(pair_count, dtype=jnp.int32) < num_pairs
    flux_i = jnp.where(real_pair_mask[:, None], flux_i, 0.0)
    flux_j = jnp.where(real_pair_mask[:, None], flux_j, 0.0)
    # scatter-add fluxes to form the graph Laplacian per particle
    lap = jnp.zeros_like(c)
    lap = lap.at[pair_i].add(flux_i)
    lap = lap.at[pair_j].add(flux_j)

    # explicit Euler diffusion step, clamped non-negative
    scale = jnp.asarray(dt * D_emu, dtype=c.dtype)
    return jnp.maximum(c + scale * lap, jnp.asarray(0.0, dtype=c.dtype))


__all__ = ["warp_diffusion_step", "warp_mech_step"]
