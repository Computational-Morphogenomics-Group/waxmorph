Graphs and locality
===================

waxMorph restricts interactions to spatial neighbors. Learned graph features
use detached host topology; simulator and emulator Warp kernels perform their
own HashGrid neighbor searches.

Learned contact graphs
----------------------

:mod:`waxmorph.torch.graph` and :mod:`waxmorph.jax.graph` share
:mod:`waxmorph._graph_core`. Distinct cells :math:`i` and :math:`j` produce both
directed edges when

.. math::

   \lVert x_i-x_j\rVert_2 \le r_i+r_j+\varepsilon.

The default :math:`\varepsilon` is ``0.01`` model units. A host
:class:`scipy.spatial.cKDTree` first queries candidates within
``2 * max(radius) + eps_dist``; the variable-radius criterion then filters
them. Candidate and output work is density-sensitive; dense graphs realize
quadratic pair counts. The simulator independently uses HashGrid searches and local
``EPS_DIST=0.25`` for adjacency-gated polarity and chemistry terms.

Backend feature contracts
-------------------------

Node features are signaling-molecule concentrations. Edge features concatenate
distance and
:math:`\arccos(\operatorname{clip}(p_i^\top p_j))`. Polarities must be unit
vectors because both backends use the supplied dot products directly. Torch uses the exact
Euclidean edge norm and returns unpadded arrays. JAX regularizes real-edge
distance as
:math:`\sqrt{\lVert x_i-x_j\rVert_2^2+\mathrm{EPS\_NORM}^2}`, masks padded
self-edges, and always returns the real-edge count; ``max_edges`` enables fixed
capacity padding.

Gradient boundary
-----------------

Both backends copy positions and radii to the host to build topology. Node and
edge features remain differentiable after the edge set is fixed. Gradients
cover continuous feature arithmetic while edge appearance remains a discrete
forward choice. Each graph construction rebuilds topology from the current state.
