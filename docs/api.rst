API Reference
=============

This section enumerates the public modules of waxMorph. The top-level modules
default to the PyTorch backend, exposing the graph-network-based simulator,
contact-graph construction, shape losses, and the training entry point. JAX
parity variants are reached through :mod:`waxmorph.jax`.

``waxmorph.data``
   Mesh loading, normalization, and volumetric point sampling. Provides helpers
   for source-target shape pairs and for intermediate target sequences used as
   supervision frames.

``waxmorph.graph``
   Construction of the contact graph induced by spatial proximity. Node
   features carry per-cell signaling-molecule concentrations; edge features
   encode pairwise distance and relative polarity. Re-exports the PyTorch
   implementation.

``waxmorph.gnn``
   The graph-network-based simulator (GNS) that learns local,
   neighbor-dependent update rules over the contact graph.

``waxmorph.train``
   The non-growing emulator training loop, which couples learned updates with
   differentiable physical constraints. Re-exports the PyTorch training API.

``waxmorph.losses``
   Point-cloud shape losses for assigned (squared) and unassigned (Chamfer,
   optimal-transport) target morphologies.

``waxmorph.simulator`` and ``waxmorph.emulator``
   Warp kernels for explicit mechanochemical forward simulation and for the
   differentiable non-growing physics corrections, respectively. The simulator
   integrates soft-sphere mechanics, reaction-diffusion, growth, and cell
   division; the emulator freezes neighbor topology within a step so gradients
   propagate through the physics.

``waxmorph.render``
   Static, interactive, and movie renderers for trajectories of spheroidal
   agents.

Backend-specific APIs live under :mod:`waxmorph.torch` and
:mod:`waxmorph.jax`.

State and notation
------------------

A tissue is represented as a population of three-dimensional spheroidal agents.
Each agent carries a position ``X`` (shape ``[N, 3]``), a polarity ``P``
(shape ``[N, 3]``), a radius ``R`` (shape ``[N]``), and a vector of signaling
molecule concentrations ``c`` (shape ``[N, num_molecules]``). The graph-network
processor predicts increments ``dX``, ``dP``, and ``dc`` over these fields,
while differentiable mechanical and diffusive corrections enforce biophysical
constraints.

Minimal emulation example
--------------------------

The following constructs a contact graph, instantiates a GNS over the molecular
and geometric state, and runs a short non-growing training rollout against a
single target morphology:

.. code-block:: python

   import numpy as np
   import torch
   from waxmorph import GNS, build_graph, train, TrainConfig, squared_loss

   num_molecules = 2
   N = 64

   # Source state: positions, polarities, radii, and molecule concentrations.
   X = torch.rand(N, 3)
   P = torch.nn.functional.normalize(torch.randn(N, 3), dim=-1)
   R = torch.full((N,), 0.05)
   c = torch.rand(N, num_molecules)

   # Contact graph: node features carry c, edge features carry distance/angle.
   node_features, edge_index, edge_features = build_graph(
       X, P, R, particle_count=N, c=c
   )

   model = GNS(
       node_feature_dim=node_features.shape[1],
       edge_feature_dim=edge_features.shape[1],
       output_dims={"dX": 3, "dP": 3, "dc": num_molecules},
   )
   out = model(node_features, edge_index, edge_features)
   out["dX"], out["dP"], out["dc"]  # predicted per-cell increments

   # Inverse design: fit local update rules to a target morphology.
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   config = TrainConfig(n_epochs=10, t_rollout=4, D_emu=0.1, lambda_reg=1e-3)

   result = train(
       model,
       optimizer,
       squared_loss,
       source_pos=X.numpy(),
       polarities=P.numpy(),
       c=c.numpy(),
       radii=R.numpy(),
       targets=[(config.t_rollout - 1, X.numpy())],
       config=config,
       device="cpu",
   )

   result.log["losses_total"]   # per-epoch total loss
   result.log["best_traj_pos"]  # best rollout positions
   result.log["best_traj_c"]    # best rollout molecule concentrations

The JAX parity backend mirrors this interface. Its ``build_graph`` additionally
returns the active edge count to support static-shape compilation:

.. code-block:: python

   from waxmorph.jax import build_graph as build_graph_jax

   node_features, edge_index, edge_features, num_edges = build_graph_jax(
       X_jax, P_jax, R_jax, particle_count=N, c=c_jax
   )

.. toctree::
   :maxdepth: 2

   api/training
   external_references

.. autosummary::
   :toctree: generated
   :recursive:

   waxmorph
