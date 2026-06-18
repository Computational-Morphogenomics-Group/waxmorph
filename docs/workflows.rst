Workflows
=========

From a mesh to a non-growing emulator
-------------------------------------

This is the most direct workflow when the biological question is shape assembly
with a fixed number of spheroidal agents.

1. Prepare source and target meshes.
2. :func:`waxmorph.data.sample_mesh_pair` normalizes each mesh and samples
   matched-size interior point clouds.
3. Initialize radii, polarity vectors, and signaling-molecule concentrations.
4. Construct a :class:`waxmorph.gnn.GNS` model whose output heads match the state
   variables to be updated.
5. Train with :func:`waxmorph.train.train` under a point-cloud loss.
6. Inspect ``TrainResult.log["best_traj_pos"]`` and render the trajectory.

The top-level :mod:`waxmorph.train`, :mod:`waxmorph.graph`, and
:mod:`waxmorph.gnn` modules are PyTorch-compatible convenience exports.
Importing from :mod:`waxmorph.torch` explicitly makes the backend choice visible
in code.

A single graph is built from the live state with :func:`waxmorph.build_graph`,
which takes the signaling-molecule concentration array ``c`` as its fifth
argument:

.. code-block:: python

   from waxmorph import build_graph

   node_features, edge_index, edge_features = build_graph(
       X, P, R, particle_count, c=c
   )

The GNS exposes one output head per updated state variable. For non-growing
shape assembly the heads predict position, polarity, and signaling-molecule
increments. The node- and edge-feature dimensions are taken from the graph:

.. code-block:: python

   from waxmorph import GNS

   model = GNS(
       node_feature_dim=node_features.shape[1],
       edge_feature_dim=edge_features.shape[1],
       output_dims={"dX": 3, "dP": 3, "dc": num_molecules},
   )

   out = model(node_features, edge_index, edge_features)
   # out["dX"], out["dP"], out["dc"] are the predicted local updates.

Training against multiple target frames
---------------------------------------

Single-target training supervises the final state of a rollout. Multi-target
training additionally supervises intermediate states. Both use the explicit
``targets`` argument; a final-only run passes a single
``(t_rollout - 1, target_pos)`` pair.

.. code-block:: python

   from waxmorph import TrainConfig, train, chamfer_distance

   config = TrainConfig(t_rollout=50, n_epochs=500)

   result = train(
       model,
       optimizer,
       chamfer_distance,
       source_pos=source_pos,
       polarities=polarities,
       c=c,
       radii=radii,
       targets=[
           (9, target_pos_early),
           (24, target_pos_mid),
           (49, target_pos_final),
       ],
       config=config,
       device="cuda",
   )

Frame indices are zero-based rollout steps after updates. Frame ``9`` therefore
constrains the state after ten learned updates.

The differentiable graph-Laplacian diffusion of the signaling-molecule
concentrations is governed by ``TrainConfig.D_emu``, and the squared-displacement
regularization weight is ``TrainConfig.lambda_reg``. The training log records the
per-epoch ``losses_total``, ``losses_shape``, and ``losses_l2``, together with
the best rollout trajectories ``best_traj_pos``, ``best_traj_pol``, and
``best_traj_c``.

Choosing a backend
------------------

The PyTorch backend is the default unless there is a specific reason to use JAX.
It is exposed through the top-level imports and provides the most direct path
from ``build_graph`` to ``GNS`` to ``train``.

The JAX backend applies when the surrounding analysis stack already uses JAX,
Equinox, or Optax, or when static-shape compilation is required. The JAX graph
builder pads edge arrays to a fixed capacity through ``max_edges`` so that
JIT-compiled functions retain stable shapes, and it returns the observed edge
count as a fourth value:

.. code-block:: python

   from waxmorph.jax import build_graph

   node_features, edge_index, edge_features, num_edges = build_graph(
       X, P, R, particle_count, c=c, max_edges=max_edges
   )

Rendering trajectories
----------------------

:mod:`waxmorph.render` contains three complementary visualization paths:

``MPLInterface``
   Static Matplotlib rendering for inspection and notebooks.

``PyVistaInterface``
   Interactive PyVista/VTK rendering of spheroids, polarity arrows,
   signaling-molecule colors, and optional cell-type colors.

``WarpMovieRenderer``
   Headless OpenGL video or USD stage export for particle trajectories.

For trained emulators, the arrays stored in ``TrainResult.log`` are already in
the format expected by the renderers: trajectory positions, polarities, and
signaling-molecule concentrations are stored as NumPy arrays with time as the
first dimension.
