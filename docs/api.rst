API Reference
=============

This section documents the public modules that most users will import. The
top-level modules favor the PyTorch backend for compatibility:

``waxmorph.data``
   Mesh loading, normalization, volumetric point sampling, and helpers for
   source-target shape pairs or target sequences.

``waxmorph.graph``
   Contact graph construction from positions, radii, polarities, and gene
   states. Re-exports the PyTorch implementation.

``waxmorph.gnn``
   Graph Network-based Simulator modules for learned local updates.

``waxmorph.train``
   Non-growing emulator training loop. Re-exports the PyTorch training API.

``waxmorph.losses``
   Point-cloud losses for assigned and unassigned target shapes.

``waxmorph.simulator`` and ``waxmorph.emulator``
   Warp kernels for explicit mechanochemical simulation and differentiable
   non-growing physics corrections.

``waxmorph.render``
   Static, interactive, and movie renderers for spheroidal trajectories.

Backend-specific APIs live under :mod:`waxmorph.torch` and
:mod:`waxmorph.jax`.

.. toctree::
   :maxdepth: 2

   api/training
   external_references

.. autosummary::
   :toctree: generated
   :recursive:

   waxmorph
