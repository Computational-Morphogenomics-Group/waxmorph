Training APIs
=============

Starting from an initial volumetric sample, training a learned emulator optimizes 
learning neighbor-dependent local updates per agent across a rollout
to reach target shape volumes throughout the trajectory.

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

Target frames are zero-based rollout indices *after* updates. A target at frame
``0`` supervises the state after the first model update, not the unmodified
source state. For the task-oriented recipe, see :doc:`../how_to/multiple_targets`.

.. toctree::
   :maxdepth: 1

   waxmorph.train
   waxmorph.torch.train
   waxmorph.jax.train
