Training APIs
=============

Training a learned emulator optimizes neighbor-dependent local updates per agent,
evolving an initial volumetric sample across a rollout toward target shape volumes
throughout the trajectory.

The top-level :mod:`waxmorph.train` module is a convenience alias for the
default PyTorch training API. The backend-specific entry points remain available
from :mod:`waxmorph.torch.train` and :mod:`waxmorph.jax.train`.

The training loop expects:

* an initial particle state: ``source_pos``, ``polarities``, ``c`` (signaling-
  molecule concentrations), and ``radii``;
* a :class:`waxmorph.gnn.GNS` model or its backend-specific equivalent;
* an optimizer;
* a shape loss such as :func:`waxmorph.losses.chamfer_distance`;
* a list of ``(frame, positions)`` targets.

Target frames are zero-based rollout indices *after* updates: frame ``0`` supervises
the state after the first model update, not the unmodified source. See
:doc:`../how_to/multiple_targets`.

.. toctree::
   :maxdepth: 1

   waxmorph.train
   waxmorph.torch.train
   waxmorph.jax.train
