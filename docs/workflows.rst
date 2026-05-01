Workflows
=========

From a mesh to a non-growing emulator
-------------------------------------

This is the most direct workflow when the biological question is shape assembly
with a fixed number of spheroidal agents.

1. Prepare source and target meshes.
2. Use :func:`waxmorph.data.sample_mesh_pair` to normalize each mesh and sample
   matched-size interior point clouds.
3. Initialize radii, polarity vectors, and gene or morphogen channels.
4. Build a :class:`waxmorph.gnn.GNS` model whose output heads match the state
   variables you want to update.
5. Train with :func:`waxmorph.train.train` using a point-cloud loss.
6. Inspect ``TrainResult.log["best_traj_pos"]`` and render the trajectory.

The top-level :mod:`waxmorph.train`, :mod:`waxmorph.graph`, and
:mod:`waxmorph.gnn` modules are PyTorch-compatible convenience exports. Import
from :mod:`waxmorph.torch` explicitly when a project needs to make the backend
choice visible in code.

Training against multiple target frames
---------------------------------------

Single-target training supervises the final state of a rollout. Multi-target
training supervises intermediate states as well:

.. code-block:: python

   from waxmorph.train import TrainConfig, train
   from waxmorph.losses import chamfer_distance

   config = TrainConfig(t_rollout=50, n_epochs=500)

   result = train(
       model,
       optimizer,
       chamfer_distance,
       source_pos=source_pos,
       polarities=polarities,
       genes=genes,
       radii=radii,
       targets=[
           (9, target_pos_early),
           (24, target_pos_mid),
           (49, target_pos_final),
       ],
       config=config,
       device="cuda",
   )

Frame indices are zero-based rollout steps after updates. For example, frame
``9`` constrains the state after ten learned updates.

Choosing a backend
------------------

Use the PyTorch backend first unless there is a specific reason to use JAX. It
is exposed through the top-level imports and has the most direct path from
``build_graph`` to ``GNS`` to ``train``.

Use the JAX backend when the rest of an analysis stack already uses JAX,
Equinox, or Optax, or when static-shape compilation is important. The JAX graph
builder may pad edge arrays to a fixed capacity through ``max_edges`` so that
JIT-compiled functions can keep stable shapes.

Rendering trajectories
----------------------

:mod:`waxmorph.render` contains three complementary visualization paths:

``MPLInterface``
   Static Matplotlib rendering for simple inspection and notebooks.

``PyVistaInterface``
   Interactive PyVista/VTK rendering of spheroids, polarity arrows, morphogen
   colors, and optional cell-type colors.

``WarpMovieRenderer``
   Headless OpenGL video or USD stage export for particle trajectories.

For trained emulators, the arrays stored in ``TrainResult.log`` are already in
the format expected by the renderers: trajectory positions, polarities, and gene
states are stored as NumPy arrays with time as the first dimension.
