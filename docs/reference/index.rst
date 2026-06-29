Reference
=========

This section enumerates the public modules of waxMorph. The top-level modules
default to the PyTorch backend, exposing the graph-network-based simulator,
contact-graph construction, shape losses, and the training entry point. The JAX
parity variants live under :mod:`waxmorph.jax`.

Module map
----------

``waxmorph.data``
   Loads, normalizes, and volumetrically samples meshes. Provides helpers for
   source-target shape pairs and for the intermediate target sequences used as
   supervision frames.

``waxmorph.graph``
   Builds the contact graph induced by spatial proximity. Node features carry
   per-cell signaling-molecule concentrations; edge features encode pairwise
   distance and relative polarity. Re-exports the PyTorch implementation.

``waxmorph.gnn``
   The graph-network-based simulator (GNS) that learns local,
   neighbor-dependent update rules over the contact graph.

``waxmorph.train``
   Runs the non-growing emulator training loop, coupling learned updates with
   differentiable physical constraints. Re-exports the PyTorch training API.

``waxmorph.losses``
   Point-cloud shape losses for assigned (squared) and unassigned (Chamfer,
   optimal-transport) target morphologies.

``waxmorph.simulator`` and ``waxmorph.emulator``
   Warp kernels for explicit mechanochemical forward simulation and for the
   differentiable non-growing physics corrections. The simulator integrates
   soft-sphere mechanics, reaction-diffusion, growth, and division; the
   emulator freezes neighbor topology within a step so gradients propagate
   through the physics.

``waxmorph.render``
   Static, interactive, and movie renderers for trajectories of spheroidal
   agents.

The backend-specific APIs live under :mod:`waxmorph.torch` and
:mod:`waxmorph.jax`.

State and notation
------------------

A tissue is a population of three-dimensional spheroidal agents. Each agent
carries a position ``X`` (shape ``[N, 3]``), a polarity ``P`` (shape
``[N, 3]``), a radius ``R`` (shape ``[N]``), and a vector of signaling-molecule
concentrations ``c`` (shape ``[N, num_molecules]``). The graph-network
processor predicts increments ``dX``, ``dP``, and ``dc`` over these fields,
while differentiable mechanical and diffusive corrections enforce the
biophysical constraints.

For a worked, runnable version of the ``build_graph`` → ``GNS`` → ``train``
flow, see :ref:`gns-primer`. The JAX parity backend mirrors that interface; its
``build_graph`` additionally returns the active edge count to support
static-shape compilation:

.. code-block:: python

   from waxmorph.jax import build_graph as build_graph_jax

   node_features, edge_index, edge_features, num_edges = build_graph_jax(
       X_jax, P_jax, R_jax, particle_count=N, c=c_jax
   )

Training and external APIs
--------------------------

.. toctree::
   :maxdepth: 2

   training
   external_references

Module index
------------

.. autosummary::
   :toctree: ../generated
   :recursive:

   waxmorph
