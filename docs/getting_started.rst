Getting Started
===============

WaxMorph works with tissues as point clouds of spheroidal agents. A minimal
state contains positions ``X``, polarities ``P``, radii ``R``, and gene or
morphogen values ``G``. The learning interface converts this state into a
contact graph, runs a Graph Network-based Simulator (GNS), and predicts local
updates such as ``dX``, ``dP``, and ``dG``.

Installation
------------

For the base package:

.. code-block:: bash

   pip install waxmorph

For the usual simulation and PyTorch learning workflow:

.. code-block:: bash

   pip install "waxmorph[simulation,learning]"

Use the JAX extra only when you plan to run the JAX/Equinox backend:

.. code-block:: bash

   pip install "waxmorph[jax]"

A first graph-network pass
--------------------------

The example below starts from in-memory arrays. In a full workflow those arrays
may come from a Warp simulation, a mesh sampled by :mod:`waxmorph.data`, or a
previous learned rollout.

.. code-block:: python

   import torch

   from waxmorph.graph import build_graph
   from waxmorph.gnn import GNS

   n_cells = 32
   n_genes = 2

   X = torch.randn(n_cells, 3)                  # cell centers
   P = torch.nn.functional.normalize(X, dim=1)  # polarity vectors
   R = torch.full((n_cells,), 0.45)             # spheroid radii
   G = torch.rand(n_cells, n_genes)             # morphogen / gene state

   node_features, edge_index, edge_features = build_graph(
       X,
       P,
       R,
       particle_count=n_cells,
       G=G,
   )

   model = GNS(
       node_feature_dim=node_features.shape[1],
       edge_feature_dim=edge_features.shape[1],
       output_dims={"dX": 3, "dP": 3, "dG": n_genes},
   )

   updates = model(node_features, edge_index, edge_features)
   X_next = X + 1e-2 * updates["dX"]

What this example does
----------------------

``build_graph`` defines the local neighborhood used by the emulator. Two cells
are connected when their center distance is within the sum of their radii plus
a small contact buffer. Node features currently contain gene or morphogen
values. Edge features contain intercellular distance and the angle between
polarity vectors.

``GNS`` then performs encode-process-decode message passing. The model does not
move cells by itself; it predicts per-cell update fields. Training code in
:mod:`waxmorph.torch.train` or :mod:`waxmorph.jax.train` applies these learned
updates during rollouts and can interleave differentiable mechanics and
diffusion corrections.

Preparing target shapes from meshes
-----------------------------------

When target morphologies are available as meshes, :mod:`waxmorph.data` provides
utilities for normalizing the mesh and sampling an interior point cloud. This
is useful when training a non-growing emulator with fixed cell count.

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

For shape sequences, use :func:`waxmorph.data.sample_mesh_sequence` with
``(frame, path)`` pairs. Frames refer to rollout steps during training; frame
``0`` supervises the state after the first learned update.
