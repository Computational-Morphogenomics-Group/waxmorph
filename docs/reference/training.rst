Training APIs
=============

Training optimizes neighbor-dependent local updates so selected rollout frames
approach target point clouds.

The top-level :mod:`waxmorph.train` module is a convenience alias for the
default PyTorch training API. The backend-specific entry points remain available
from :mod:`waxmorph.torch.train` and :mod:`waxmorph.jax.train`.

The training loop expects:

* an initial particle state: ``source_pos``, ``polarities``, ``c`` (signaling-
  molecule concentrations), and ``radii``;
* a :class:`waxmorph.gnn.GNS` model or its backend-specific equivalent;
* a PyTorch optimizer, or an Optax transformation and state for JAX;
* a backend-compatible shape loss;
* a list of ``(frame, positions)`` targets.

Target frames are zero-based indices within a model evaluation. The source is
the trajectory's initial entry; frame ``0`` supervises the state after the first
complete learned, mechanics, and diffusion rollout step. See
:doc:`../how_to/multiple_targets`.

Post-update selection
---------------------

``n_epochs`` counts optimizer updates. Each history contains ``n_epochs``
entries for the resulting models; evaluating them requires one extra rollout.
The returned model, ``best_loss``, ``best_epoch``, and best trajectory describe
the same post-update epoch. The initial evaluation supplies the first gradient;
the ``n_epochs`` updated models form the selection candidates.

Checkpoint and backend contracts
--------------------------------

Model checkpoints contain model weights and configuration. During a new PyTorch
run, the supplied optimizer is restored with the selected model. Refining a
trusted, compatible checkpoint starts with fresh optimizer state, performs one
update, and preserves the existing files. JAX validates the saved configuration
and PyTree, replaces the model, reinitializes Optax state, and likewise performs
one refinement update while preserving existing files. ``TrainResult`` contains
the selected model and log; callers manage Optax state separately.

JAX training bounds directed edges and neighbor pairs with
``max_edges_factor``. Its Warp mechanics and diffusion run on CUDA; CPU runs use
``mech_steps=0`` and ``diff_steps=0``.

.. toctree::
   :maxdepth: 1

   waxmorph.train
   waxmorph.torch.train
   waxmorph.jax.train
