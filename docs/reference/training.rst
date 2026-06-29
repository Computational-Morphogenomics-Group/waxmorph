Training APIs
=============

Training in waxMorph means optimizing a local emulator against one or more
target shapes. The current training APIs implement non-growing shape assembly:
the number of active spheroidal agents stays fixed during the rollout, while
positions, polarities, and signaling-molecule concentrations are updated.

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
