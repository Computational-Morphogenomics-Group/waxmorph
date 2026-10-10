API reference
=============

Top-level ``waxmorph`` and ``waxmorph.{gnn, graph, losses, mlp, train}`` re-export the PyTorch
backend :mod:`waxmorph.torch`. The JAX/Equinox backend :mod:`waxmorph.jax` is imported explicitly.

Modules
-------

``waxmorph.simulator``
   Forward simulator (Warp): cell-type-dependent mechanics, activator-inhibitor
   reaction-diffusion, growth, and division into preallocated capacity.
``waxmorph.emulator``
   Differentiable physics for learned rollouts with a fixed agent count: sticky-sphere mechanics
   and graph-Laplacian diffusion on frozen contact pairs.
``waxmorph.data``
   Mesh loading, normalization, and volume sampling: ``sample_mesh_pair`` for one target,
   ``sample_mesh_sequence`` for frame-tagged targets.
``waxmorph.graph``, ``waxmorph.gnn``, ``waxmorph.losses``, ``waxmorph.train``
   Contact graph, graph network simulator (GNS), shape losses, and training.
``waxmorph.render``
   Matplotlib, PyVista, and Warp (USD or OpenGL video) renderers.

.. _state-and-notation:

State and notation
------------------

``X`` ``[N, 3]`` positions, ``P`` ``[N, 3]`` unit polarities, ``R`` ``[N]`` radii, ``c``
``[N, C]`` signaling-molecule concentrations, and optional ``CT`` cell types (``0`` mesenchymal,
``1`` epithelial). The GNS predicts increments ``dX``, ``dP``, and ``dc``. All values are
nondimensional model units.

.. _backends:

Backends
--------

The JAX backend differs from PyTorch in these contracts:

* ``build_graph`` also returns the real-edge count, and pads to ``max_edges`` when given. Edge
  distances are :math:`\sqrt{\lVert x_i-x_j\rVert^2+\mathrm{EPS\_NORM}^2}` (PyTorch uses the
  exact norm), and padded self-edges have zero features.
* ``GNS`` takes an explicit PRNG ``key``; ``train`` takes an Optax optimizer and its state.
* The loss factory is ``make_sinkhorn_loss`` (OTT) instead of ``make_samples_loss`` (GeomLoss).
* Warp mechanics and diffusion require CUDA; CPU training sets ``mech_steps=0`` and
  ``diff_steps=0``. Training bounds edges and neighbor pairs by ``N * max_edges_factor``.
* Checkpoints are Equinox leaves plus ``<path>.json`` instead of one ``torch.save`` file.

.. toctree::
   :maxdepth: 1

   waxmorph.train
   waxmorph.torch.train
   waxmorph.jax.train
   Runtime and numerical contracts <../adr/0001-runtime-and-numerical-contracts>

.. autosummary::
   :toctree: ../generated
   :recursive:

   waxmorph
