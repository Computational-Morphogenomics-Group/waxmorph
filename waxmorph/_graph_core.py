"""Framework-agnostic graph topology from numpy arrays.

Extracts the cKDTree-based contact adjacency computation so that both
the PyTorch and (future) JAX graph builders can share the same logic.
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
    and ``i != j``.

    Parameters
    ----------
    pos : np.ndarray, shape ``[N, 3]``, dtype float32
        Particle positions.
    rad : np.ndarray, shape ``[N]``, dtype float32
        Particle radii.
    eps_dist : float
        Contact buffer distance.

    Returns
    -------
    senders : np.ndarray, shape ``[E]``, dtype int64
    receivers : np.ndarray, shape ``[E]``, dtype int64
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
