Reference
=========

waxMorph's top-level modules default to the PyTorch backend and expose the
graph-network-based simulator, contact-graph construction, shape losses, and the
training entry point. Explicit JAX counterparts live under :mod:`waxmorph.jax`.

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
   Trains the emulator with a fixed agent count by coupling learned updates with
   differentiable physical constraints. Re-exports the PyTorch training API.

``waxmorph.losses``
   Point-cloud shape losses for assigned (squared) and unassigned (Chamfer,
   optimal-transport) target morphologies.

``waxmorph.simulator`` and ``waxmorph.emulator``
   Warp kernels for explicit mechanochemical forward simulation and for the
   differentiable physics corrections used in rollouts with a fixed agent
   count. The simulator integrates soft-sphere mechanics, reaction-diffusion,
   growth, and division; the emulator freezes neighbor topology within a step
   so gradients propagate through the physics.

``waxmorph.render``
   Static, interactive, and movie renderers for trajectories of spheroidal
   agents.

The backend-specific APIs live under :mod:`waxmorph.torch` and
:mod:`waxmorph.jax`.

State and notation
------------------

waxMorph represents a tissue as spheroidal agents carrying positions ``X``,
polarities ``P``, radii ``R``, and signaling-molecule concentrations ``c``; the
graph-network processor predicts increments ``dX``, ``dP``, and ``dc`` over these
fields. See :doc:`../explanation/cell_state` for the field shapes and index
conventions.

For a worked ``build_graph`` → ``GNS`` → ``train`` flow, see
:ref:`the mesh-to-emulator guide <gns-primer>`. JAX provides backend-specific
graph returns and padding, loss families, PRNG, optimizer and checkpoint
behavior, and CUDA execution for Warp physics; see
:doc:`../how_to/choose_backend`.

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
