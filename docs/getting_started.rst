Getting Started
===============

waxMorph represents tissues as point clouds of three-dimensional spheroidal
agents. A minimal state comprises positions ``X``, polarities ``P``, radii
``R``, and signaling molecule concentrations ``c``. The learning interface
converts this state into a contact graph, evaluates a graph-network-based
simulator (GNS), and predicts local update fields ``dX``, ``dP``, and ``dc``.

Installation
------------

The base package is installed with:

.. code-block:: bash

   pip install waxmorph

The combined simulation and PyTorch learning workflow requires:

.. code-block:: bash

   pip install "waxmorph[simulation,learning]"

The JAX extra provides the JAX/Equinox backend on CPU:

.. code-block:: bash

   pip install "waxmorph[jax]"

JAX with CUDA 12 support is installed with:

.. code-block:: bash

   pip install "waxmorph[jax-cuda]"

A first graph-network pass
--------------------------

The example below starts from in-memory arrays. In a full workflow these arrays
may originate from a Warp simulation, a mesh sampled by :mod:`waxmorph.data`, or
a previous learned rollout.

.. code-block:: python

   import torch

   from waxmorph import GNS, build_graph

   n_cells = 32
   num_molecules = 2

   X = torch.randn(n_cells, 3)                  # cell centers
   P = torch.nn.functional.normalize(X, dim=1)  # polarity vectors
   R = torch.full((n_cells,), 0.45)             # spheroid radii
   c = torch.rand(n_cells, num_molecules)       # signaling molecule concentrations

   node_features, edge_index, edge_features = build_graph(
       X,
       P,
       R,
       particle_count=n_cells,
       c=c,
   )

   model = GNS(
       node_feature_dim=node_features.shape[1],
       edge_feature_dim=edge_features.shape[1],
       output_dims={"dX": 3, "dP": 3, "dc": num_molecules},
   )

   updates = model(node_features, edge_index, edge_features)
   X_next = X + 1e-2 * updates["dX"]

What this example does
----------------------

``build_graph`` defines the local neighborhood used by the emulator. Two cells
are connected when their center distance falls within the sum of their radii
plus a small contact buffer, so the adjacency graph reflects spatial proximity.
Node features carry the signaling molecule concentrations. Edge features carry
the intercellular distance and the angle between polarity vectors.

``GNS`` then performs encode-process-decode message passing. The model does not
displace cells directly; it predicts per-cell update fields. The training
routines in :mod:`waxmorph.torch.train` and :mod:`waxmorph.jax.train` apply
these learned updates during rollouts and can interleave differentiable
mechanics and diffusion corrections.

Preparing target shapes from meshes
-----------------------------------

When target morphologies are available as meshes, :mod:`waxmorph.data` provides
utilities for normalizing the mesh and sampling an interior point cloud. This
procedure applies when training a non-growing emulator at fixed cell count.

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

For morphology sequences, :func:`waxmorph.data.sample_mesh_sequence` accepts
``(frame, path)`` pairs. Frames index rollout steps during training; frame
``0`` supervises the state after the first learned update.
