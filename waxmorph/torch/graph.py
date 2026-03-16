"""Build PyTorch-geometric-style graph data from Warp simulation state.

Converts Warp arrays into PyTorch tensors and constructs the adjacency
graph, node features, and edge features needed by the GNS.
"""

import numpy as np
import torch
import warp as wp

from .._graph_core import build_edge_index_np
from ..constants import EPS_DIST


def _wp_to_torch(arr: wp.array, particle_count: int) -> torch.Tensor:
    """Convert a Warp array to a PyTorch tensor, sliced to active particles.

    Handles both scalar (float32, uint32) and vector (vec3f) dtypes by
    going through numpy when ``wp.to_torch`` is unavailable or when the
    dtype is not directly supported (e.g. uint32).
    """
    try:
        t = wp.to_torch(arr)
    except Exception:
        t = torch.from_numpy(arr.numpy())
    # vec3f arrays come out as (max_particles, 3); scalars as (max_particles,)
    return t[:particle_count]


def build_edge_index(
    X: wp.array,
    R: wp.array,
    particle_count: int,
    eps_dist: float = EPS_DIST,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct COO edge_index ``[2, E]`` from contact adjacency.

    An edge ``(i, j)`` exists when ``dist(X[i], X[j]) <= R[i] + R[j] + eps_dist``
    and ``i != j``.  Returns directed edges (both ``i->j`` and ``j->i``).

    Uses ``scipy.spatial.cKDTree`` for O(N log N) neighbour queries instead
    of an O(N^2) pairwise distance matrix.

    Parameters
    ----------
    X : wp.array(dtype=wp.vec3f)
        Position array (preallocated to ``max_particles``).
    R : wp.array(dtype=wp.float32)
        Radius array.
    particle_count : int
        Number of active particles.
    eps_dist : float
        Contact buffer matching ``simulator.EPS_DIST``.
    device : torch.device, optional
        Target device.  Defaults to CUDA if available.

    Returns
    -------
    edge_index : torch.Tensor, shape ``[2, E]``, dtype ``torch.long``
    """
    pos = _wp_to_torch(X, particle_count).float().cpu().numpy()
    rad = _wp_to_torch(R, particle_count).float().cpu().numpy()

    senders, receivers = build_edge_index_np(pos, rad, eps_dist)

    if len(senders) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.from_numpy(np.stack([senders, receivers], axis=0))

    target = device if device is not None else str(X.device)
    edge_index = edge_index.to(target)
    return edge_index


def build_node_features(
    G: wp.array,
    particle_count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Assemble per-node feature tensor from Warp state arrays.

    Feature layout per node::

        [g_0, g_1, ..., g_{G-1}]

    Dimensions: ``G`` (number of genes).

    Returns
    -------
    node_features : torch.Tensor, shape ``[N, G]``
    """
    genes = _wp_to_torch(G, particle_count).float()
    if genes.ndim == 1:
        genes = genes.unsqueeze(-1)

    if device is not None:
        genes = genes.to(device)

    return genes


def build_edge_features(
    X: wp.array,
    P: wp.array,
    edge_index: torch.Tensor,
    particle_count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute per-edge feature tensor.

    Feature layout per edge ``(i -> j)``::

        [dx, dy, dz, angle(P_i, P_j)]

    Dimensions: ``3 + 1 = 4``.

    Returns
    -------
    edge_features : torch.Tensor, shape ``[E, 4]``
    """
    pos = _wp_to_torch(X, particle_count).float()
    pol = _wp_to_torch(P, particle_count).float()

    senders = edge_index[0]
    receivers = edge_index[1]

    rel_pos = pos[senders] - pos[receivers]

    p_s = pol[senders]
    p_r = pol[receivers]
    cos_angle = (p_s * p_r).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)

    feats = torch.cat([rel_pos, angle], dim=-1)

    if device is not None:
        feats = feats.to(device)

    return feats


def build_graph(
    X: wp.array,
    P: wp.array,
    R: wp.array,
    particle_count: int = 0,
    G: wp.array | None = None,
    eps_dist: float = EPS_DIST,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(node_features, edge_index, edge_features)`` in one call.

    Returns
    -------
    node_features : torch.Tensor ``[N, G]``
    edge_index    : torch.Tensor ``[2, E]``
    edge_features : torch.Tensor ``[E, 4]``
    """
    edge_index = build_edge_index(X, R, particle_count, eps_dist, device)
    node_features = build_node_features(G, particle_count, device)
    edge_features = build_edge_features(X, P, edge_index, particle_count, device)
    return node_features, edge_index, edge_features
