Choose a learning backend
=========================

PyTorch is the default learning path. The JAX backend fits into an existing
JAX, Equinox, and Optax stack and supports static-shape compilation.

Use PyTorch (the default)
-------------------------

The top-level imports resolve to the PyTorch backend, so you get the most direct
path from ``build_graph`` to ``GNS`` to ``train``, as :ref:`the mesh-to-emulator
guide <gns-primer>` shows.
Warp's automatic differentiation integrates directly through
:class:`torch.autograd.Function`.

Use JAX when your stack already uses it
---------------------------------------

Switch to the JAX/Equinox backend when the surrounding analysis already depends
on JAX, Equinox, or Optax, or when you need static-shape compilation. Reach it
through explicit imports:

.. code-block:: python

   from waxmorph.jax.gnn import GNS
   from waxmorph.jax.graph import build_graph
   from waxmorph.jax.losses import make_sinkhorn_loss
   from waxmorph.jax.train import TrainConfig, train

JAX model construction uses an explicit PRNG key. Its graph builder always
returns the observed edge count as a fourth value. Supplying ``max_edges`` pads
the edge arrays to that capacity for stable compiled shapes; the default returns
edge arrays sized to the observed graph:

.. code-block:: python

   from waxmorph.jax import build_graph

   node_features, edge_index, edge_features, num_edges = build_graph(
       X, P, R, particle_count, c=c, max_edges=max_edges
   )

PyTorch training takes a mutable optimizer and restores it to the selected best
model. JAX training takes an Optax transformation and optimizer state;
``TrainResult`` contains the selected model and log, while callers manage Optax
state separately.

.. note::

   JAX training bounds directed edges and neighbor pairs by
   ``N * max_edges_factor`` and raises on overflow. Its Warp mechanics and
   diffusion custom VJPs run on CUDA; CPU training uses zero for both prescribed
   step counts. PyTorch accepts supported Torch/Warp CPU or CUDA devices.

PyTorch exposes GeomLoss Sinkhorn, MMD, and Hausdorff families. JAX exposes
debiased OTT Sinkhorn divergence. See :doc:`../explanation/losses`.
