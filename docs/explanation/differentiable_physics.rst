Emulator differentiability
==========================

The learned emulator differentiates its uniform sticky-sphere mechanics and
graph diffusion so their gradients compose with the GNS.

Frozen topology within each correction
--------------------------------------

Each learned update may be followed by ``mech_steps`` mechanics calls and
``diff_steps`` diffusion calls; zero selects the identity path for that
correction. ``dt_mech`` and ``dt_diff`` set their respective explicit step
sizes.

Pair discovery occurs outside autodiff. Topology is rebuilt between correction
calls as the state evolves, then held fixed while that call is differentiated.
Gradients cover continuous force, flux, and state arithmetic; neighbor topology
remains a discrete forward input. Learned graphs use a host
:class:`scipy.spatial.cKDTree`; prescribed emulator
corrections use a Warp HashGrid. Candidate and output work depends on local
density and can be quadratic for dense states.

Torch Tape and JAX custom VJP
-----------------------------

:mod:`waxmorph.torch.warp_autograd` wraps each mechanics or diffusion call in a
:class:`torch.autograd.Function` with a fresh :class:`warp.Tape`. Backward seeds
the output adjoint and replays the recorded emulator kernels. The mechanics
bridge exposes position gradients; the diffusion bridge exposes concentration
gradients.

:mod:`waxmorph.jax.warp_autograd` uses custom VJPs on CUDA, backed by dedicated
backward-enabled per-pair Warp FFI kernels. JAX owns endpoint gathers,
padding masks, scatter-adds, and state updates. Its mechanics bridge
differentiates positions and radii on the frozen pairs; diffusion differentiates
concentrations. Adding a correction therefore requires backend-specific bridge
work.

Simulator derivative implementations
====================================

The forward simulator provides its own derivative implementations.
``mech_step_sticky`` uses explicit pair derivatives.
``mech_step_sticky_implicit`` uses local :func:`warp.grad` evaluation for
polarity, thickness, mesenchymal alignment, and optional WNT terms; its
soft-sphere derivatives remain explicit. Both paths use atomic reductions, so
floating-point reduction order may vary on parallel devices.
