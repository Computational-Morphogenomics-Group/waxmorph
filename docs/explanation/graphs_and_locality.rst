Graphs and locality
===================

The central modelling assumption in waxMorph is locality: a cell influences
primarily its spatial neighbors. waxMorph evaluates every interaction — mechanics,
polarity potentials, and molecular diffusion — over a spatial adjacency graph
rather than over a global image grid, and it rebuilds that graph as the tissue
deforms.

How the contact graph is built
------------------------------

Graph construction lives in :mod:`waxmorph.graph`, :mod:`waxmorph.torch.graph`,
and :mod:`waxmorph.jax.graph`, all sharing the contact-adjacency core in
:mod:`waxmorph._graph_core` so the topology is identical across backends. The
graph induces a neighborhood from spatial proximity: it places one node per
cell and connects cells :math:`i` and :math:`j` when their centers fall within
the sum of their radii plus a small contact buffer,

.. math::

   \lVert x_i - x_j \rVert_2 \le r_i + r_j + \varepsilon.

Node features carry the per-cell signaling-molecule concentrations. Edge
features concatenate the center distance :math:`d_{ij} = \lVert x_i - x_j
\rVert_2` and the polarity angle :math:`\theta_{ij} = \arccos(p_i^\top p_j)`.
Because the graph is induced by cell positions rather than by a fixed mesh of
cell boundaries, neighborhoods change as cells move; rebuilding the graph each
step, combined with spatial hashing, keeps the cost :math:`O(N)` rather than
:math:`O(N^2)`.

Why the topology is frozen within a step
----------------------------------------

For PyTorch inputs, feature construction stays differentiable, but the edge
topology is built from a *detached* snapshot of positions and radii. Gradients
therefore flow through distances, angles, signaling-molecule concentrations, and
model parameters — but not through the discrete event of an edge appearing or
disappearing within a single step.

This is the standard approximation in differentiable particle systems: optimize
the continuous state while treating the contact graph as fixed over the current
local update. :doc:`differentiable_physics` covers how that approximation
interacts with the Warp autodiff that propagates gradients through the physics
corrections.
