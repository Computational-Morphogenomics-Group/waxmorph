Architecture
============

waxMorph is a two-backend learning stack layered over a shared Warp physics
core, plus rendering. This page explains how those pieces fit together and why
PyTorch is the default while JAX is a parity backend.

The shared Warp physics core
----------------------------

Both modes rest on the same NVIDIA Warp kernels, which is what keeps simulation
and emulation physically consistent.

* :mod:`waxmorph.simulator` holds the explicit mechanochemical kernels for
  forward simulation: sticky-sphere mechanics, activator-inhibitor
  reaction-diffusion, growth, neighbor counting, and division. It generates
  cheap training data and runs prescribed mechanistic models, and it can grow
  the active particle count up to a preallocated ``max_particles``.
* :mod:`waxmorph.emulator` holds the Warp mechanics and graph-Laplacian
  diffusion kernels for the differentiable, non-growing path. Its
  differentiable entry points record pairwise kernels on a :class:`warp.Tape`
  while freezing neighbor topology for the step
  (:doc:`differentiable_physics`).

Two framework-agnostic helpers sit alongside the kernels:
:mod:`waxmorph._graph_core` builds the contact adjacency once and shares it
across backends, and :mod:`waxmorph.constants` holds the EPS values, hash-grid
dimension, and force constants the kernels read.

Two backends, one interface
---------------------------

The learning layer exists twice, at parity, over that core.

The top-level modules ``waxmorph/{gnn,graph,train,losses,mlp}.py`` are thin
re-exports of ``waxmorph/torch/*``. PyTorch is the default backend, so
``from waxmorph import GNS, train, build_graph`` resolves to the ``torch/``
implementations. To change learning behavior you edit
``waxmorph/torch/<module>.py`` — the top-level shim only forwards — and then
mirror the change in ``waxmorph/jax/<module>.py`` to keep the backends in step.

The JAX/Equinox backend is reached only through explicit
``from waxmorph.jax import ...`` imports. It uses an Optax optimizer and
static-shape compilation.

Why PyTorch is the default
--------------------------

The split follows from how each framework reaches the Warp physics. On the
PyTorch side, Warp's automatic differentiation integrates through
:class:`torch.autograd.Function`, so the physics tape plugs straight into the
surrounding graph. The JAX path requires compile-time array sizes and an upper
bound on the number of edges, so its graph builder pads edge arrays to a fixed
capacity and the path uses more memory. PyTorch is therefore the default; the
JAX backend is appropriate when downstream analysis already depends on JAX,
Equinox, or Optax, or when static-shape compilation is required (see
:doc:`../how_to/choose_backend`).

Rendering
---------

:mod:`waxmorph.render` provides three renderers over the same trajectory
arrays: ``MPLInterface`` for static views, ``PyVistaInterface`` for interactive
inspection, and ``WarpMovieRenderer`` for USD-stage or headless-OpenGL movie
export. ``write_frame_from_numpy`` renders learned rollouts;
``write_frame_from_state`` renders live Warp simulation state.
