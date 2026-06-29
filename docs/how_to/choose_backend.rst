Choose a learning backend
=========================

Reach for the PyTorch backend by default and the JAX backend only when your
surrounding stack calls for it.

Use PyTorch (the default)
-------------------------

The PyTorch backend is the default unless you have a specific reason to switch.
The top-level imports resolve to it, giving you the most direct path from
``build_graph`` to ``GNS`` to ``train`` shown in :ref:`gns-primer`. Warp's
automatic differentiation integrates through
:class:`torch.autograd.Function`, so the physics gradients flow without extra
plumbing.

Use JAX when your stack already does
------------------------------------

Switch to the JAX/Equinox backend when the surrounding analysis already depends
on JAX, Equinox, or Optax, or when you need static-shape compilation. Reach it
through explicit imports:

.. code-block:: python

   from waxmorph.jax.gnn import GNS
   from waxmorph.jax.graph import build_graph
   from waxmorph.jax.losses import make_sinkhorn_loss
   from waxmorph.jax.train import TrainConfig, train

Two interface differences follow from JAX's compile-time shape requirements.
The JAX graph builder pads edge arrays to a fixed capacity through
``max_edges`` so that JIT-compiled functions keep stable shapes, and it returns
the observed edge count as a fourth value:

.. code-block:: python

   from waxmorph.jax import build_graph

   node_features, edge_index, edge_features, num_edges = build_graph(
       X, P, R, particle_count, c=c, max_edges=max_edges
   )

The JAX training call takes an Optax optimizer and its optimizer state rather
than a PyTorch optimizer. Because the JAX path requires an upper bound on the
number of edges, it uses more memory than the PyTorch path; see
:doc:`../explanation/architecture` for the design rationale behind the
two-backend split.
