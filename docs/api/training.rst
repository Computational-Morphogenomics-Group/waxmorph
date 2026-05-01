Training APIs
=============

Training in WaxMorph means optimizing a local emulator against one or more
target shapes. The current training APIs implement non-growing shape assembly:
the number of active spheroidal agents is fixed during the rollout, while
positions, polarities, and gene or morphogen states are updated.

The top-level :mod:`waxmorph.train` module is a convenience alias for the
default PyTorch training API. Backend-specific entry points remain available
from :mod:`waxmorph.torch.train` and :mod:`waxmorph.jax.train`.

The training loop expects:

* an initial particle state: ``source_pos``, ``polarities``, ``genes``, and
  ``radii``;
* a :class:`waxmorph.gnn.GNS` model or backend-specific equivalent;
* an optimizer;
* a shape loss such as :func:`waxmorph.losses.chamfer_distance`;
* a list of ``(frame, positions)`` targets.

Target frames are zero-based rollout indices after updates. A target at frame
``0`` supervises the state after the first model update, not the unmodified
source state.

.. toctree::
   :maxdepth: 1

   waxmorph.train
   waxmorph.torch.train
   waxmorph.jax.train
