"""Framework-agnostic contact-adjacency topology from :mod:`numpy` arrays.

The neighborhood graph is induced by spheroidal-agent geometry alone: two
cells are neighbors when their spheres touch within a contact buffer. Sharing
one :class:`scipy.spatial.cKDTree` implementation here keeps that topology
identical across backends -- the PyTorch and JAX graph builders both call
:func:`build_edge_index_np`, so an edge that exists in one backend exists in
the other for the same positions and radii.

This module is pure NumPy/SciPy and depends on neither Torch nor JAX, so it
can run on detached host snapshots without pulling in either framework.
"""

import numpy as np
from scipy.spatial import cKDTree

from .constants import EPS_DIST


def build_edge_index_np(
    pos: np.ndarray,
    rad: np.ndarray,
    eps_dist: float = EPS_DIST,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bidirectional contact-adjacency edges from positions and radii.

    An edge ``(i, j)`` exists when ``dist(pos[i], pos[j]) <= rad[i] + rad[j] + eps_dist``
    and ``i != j``. A single :class:`scipy.spatial.cKDTree` first finds all
    pairs within ``2 * max(rad) + eps_dist`` (an upper bound on any pair's
    contact threshold) in O(N log N), then the per-pair radius-sum threshold is
    applied exactly. This is the one place contact adjacency is computed for
    both backends, so topology stays identical across PyTorch and JAX.

    Args:
        pos: Particle positions with shape ``[N, 3]``.
        rad: Particle radii with shape ``[N]``.
        eps_dist: Contact buffer distance added to each pair threshold.

    Returns:
        Pair ``(senders, receivers)`` of ``int64`` arrays with shape ``[E]``.
        Edges are directed and include both directions for every contact pair.

    See Also:
        :func:`waxmorph.torch.graph.build_edge_index`: PyTorch caller.
        :func:`waxmorph.jax.graph.build_edge_index`: JAX caller.

    Examples:
        >>> pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        >>> rad = np.array([0.6, 0.6, 0.6])
        >>> senders, receivers = build_edge_index_np(pos, rad, eps_dist=0.0)
        >>> print(senders.tolist(), receivers.tolist())
        [0, 1] [1, 0]
    """
    if len(pos) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty

    max_r = float(rad.max()) * 2 + eps_dist
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=max_r, output_type="ndarray")  # [P, 2]

    if len(pairs) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty

    ii, jj = pairs[:, 0], pairs[:, 1]
    dists = np.linalg.norm(pos[ii] - pos[jj], axis=-1)
    thresholds = rad[ii] + rad[jj] + eps_dist
    mask = dists <= thresholds
    ii, jj = ii[mask], jj[mask]

    # Make bidirectional
    senders = np.concatenate([ii, jj]).astype(np.int64)
    receivers = np.concatenate([jj, ii]).astype(np.int64)
    return senders, receivers
