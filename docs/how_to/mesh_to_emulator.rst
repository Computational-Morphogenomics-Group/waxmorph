.. _mesh-to-emulator:

Train the emulator from meshes
========================================

Take this path when your question is shape assembly with a fixed number of
spheroidal agents: you have a source morphology and a target morphology, and
you want to learn the local rollout rule that carries one into the other.

The workflow has six steps.

1. Prepare source and target meshes.
2. Sample matched-size interior point clouds with
   :func:`waxmorph.data.sample_mesh_pair`, which normalizes each mesh first.
3. Initialize radii, polarity vectors, and signaling-molecule concentrations.
4. Build a :class:`waxmorph.gnn.GNS` whose output heads match the state
   variables you want to update.
5. Train with :func:`waxmorph.train.train` under a density-based loss.
6. Inspect ``TrainResult.log["best_traj_pos"]`` and render the trajectory.

.. _gns-primer:

The build_graph → GNS → train primer
------------------------------------

Sample the source and target shapes. ``sample_mesh_pair`` normalizes each mesh
and returns matched-size point clouds together with a uniform radius:

.. code-block:: python

   from waxmorph.data import sample_mesh_pair

   pair = sample_mesh_pair(
       "source.stl",
       "target.stl",
       n_points=2000,
       target_extent=10.0,
       seed=1,
   )
   source_pos = pair["source_pos"]
   target_pos = pair["target_pos"]
   radii = pair["radius"]

Build the contact graph from the live state;
:doc:`../explanation/graphs_and_locality` derives the contact-adjacency rule.
Node features carry the signaling-molecule concentrations ``c``; edge features
carry the intercellular distance and the angle between polarity vectors:

.. code-block:: python

   from waxmorph import build_graph

   node_features, edge_index, edge_features = build_graph(
       X, P, R, particle_count, c=c
   )

Construct the GNS with one output head per updated state variable. For
shape assembly the heads predict position, polarity, and
signaling-molecule increments, and the node- and edge-feature dimensions come
straight from the graph:

.. code-block:: python

   from waxmorph import GNS

   model = GNS(
       node_feature_dim=node_features.shape[1],
       edge_feature_dim=edge_features.shape[1],
       output_dims={"dX": 3, "dP": 3, "dc": num_molecules},
   )

   out = model(node_features, edge_index, edge_features)
   # out["dX"], out["dP"], out["dc"] are the predicted per-cell increments.

The GNS predicts per-cell update fields. :func:`waxmorph.train.train` applies
those fields during a rollout and
interleaves the differentiable mechanics and diffusion corrections described in
:doc:`../explanation/differentiable_physics`.

Train and read back the result. Endpoint supervision passes a single target:

.. code-block:: python

   import torch
   from waxmorph import TrainConfig, train, chamfer_distance

   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   config = TrainConfig(t_rollout=100, n_epochs=2000)

   result = train(
       model,
       optimizer,
       chamfer_distance,
       source_pos=source_pos,
       polarities=polarities,
       c=c,
       radii=radii,
       targets=[(config.t_rollout - 1, target_pos)],
       config=config,
       device="cuda",
   )

   trajectory = result.log["best_traj_pos"]  # [time, particles, 3]

The training log records the per-epoch ``losses_total``, ``losses_shape``, and
``losses_l2``, alongside the best rollout trajectories ``best_traj_pos``,
``best_traj_pol``, and ``best_traj_c``.

Where to go next
----------------

* Supervise intermediate and endpoint morphologies:
  :doc:`multiple_targets`.
* Run the same workflow on JAX: :doc:`choose_backend`.
* Turn ``best_traj_pos`` into a movie: :doc:`render_trajectories`.
