Architecture
============

waxMorph is a biophysically constrained trajectory learning stack layered over a shared `Warp <https://nvidia.github.io/warp/stable/>`_ physics
core, plus rendering. This page explains how those pieces fit together.

The shared Warp physics core
----------------------------

The forward simulator, and the inverse `PyTorch <https://docs.pytorch.org/docs/2.12/index.html>`_ / `JAX <https://docs.jax.dev/en/latest/>`_ learning modules build on the same `Warp <https://nvidia.github.io/warp/stable/>`_ kernels, keeping simulation and emulation physically consistent, and the primitives transferable.

* :mod:`waxmorph.simulator` holds the explicit mechanochemical kernels for
  forward simulation, exemplified by sticky-sphere mechanics, activator-inhibitor
  reaction-diffusion, growth, neighbor counting, and division. It can generate
  cheap simulations by running prescribed biophysical models with cell division, growing
  the active particle count up to a preallocated ``max_particles``.
* :mod:`waxmorph.emulator` holds the mechanics and graph-Laplacian
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
``waxmorph/torch/<module>.py`` and then
mirror the change in ``waxmorph/jax/<module>.py`` to keep the backends in sync.

The `JAX <https://docs.jax.dev/en/latest/>`_/`Equinox <https://docs.kidger.site/equinox/>`_ backend is reached only through explicit
``from waxmorph.jax import ...`` imports. It uses an `Optax <https://optax.readthedocs.io/en/latest/>`_ optimizer and
static-shape array compilation to trigger recompiles minimally.

Rendering
---------

:mod:`waxmorph.render` provides three renderers over the same trajectory
arrays: ``MPLInterface`` for static views built over `Matplotlib <https://matplotlib.org/stable/>`_, ``PyVistaInterface`` for interactive
inspection built on `PyVista <https://pyvista.org/>`_, and ``WarpMovieRenderer`` using `Warp rendering primitives <https://nvidia.github.io/warp/stable/api_reference/warp_render.html>`_ for exporting `USD stages <https://openusd.org/release/index.html>`_ or headless `OpenGL <https://www.opengl.org/>`_ movies. ``write_frame_from_numpy`` renders learned rollouts; ``write_frame_from_state`` renders live simulation state.
