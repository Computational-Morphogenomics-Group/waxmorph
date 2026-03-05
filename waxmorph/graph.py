"""Build PyTorch-geometric-style graph data from Warp simulation state.

Converts Warp arrays into PyTorch tensors and constructs the adjacency
graph, node features, and edge features needed by the GNS.
"""

import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree

from .constants import EPS_DIST


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
    of an O(N²) pairwise distance matrix.

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
    pos = _wp_to_torch(X, particle_count).float().cpu().numpy()  # [N, 3]
    rad = _wp_to_torch(R, particle_count).float().cpu().numpy()  # [N]

    # Upper bound on contact distance for any pair
    max_r = float(rad.max()) * 2 + eps_dist
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=max_r, output_type="ndarray")  # [P, 2]

    if len(pairs) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        # Filter by per-pair threshold: dist <= R[i] + R[j] + eps_dist
        ii, jj = pairs[:, 0], pairs[:, 1]
        dists = np.linalg.norm(pos[ii] - pos[jj], axis=-1)
        thresholds = rad[ii] + rad[jj] + eps_dist
        mask = dists <= thresholds
        ii, jj = ii[mask], jj[mask]

        # Make bidirectional
        senders = np.concatenate([ii, jj])
        receivers = np.concatenate([jj, ii])
        edge_index = torch.from_numpy(
            np.stack([senders, receivers], axis=0).astype(np.int64)
        )

    # Infer target device from the Warp input array when not specified.
    target = device if device is not None else str(X.device)
    edge_index = edge_index.to(target)

    return edge_index


def build_node_features(
    X: wp.array,
    P: wp.array,
    R: wp.array,
    CT: wp.array | None = None,
    particle_count: int = 0,
    G: wp.array | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Assemble per-node feature tensor from Warp state arrays.

    Feature layout per node (when CT is provided)::

        [x, y, z, px, py, pz, radius, ct_0, ct_1, g_0, ..., g_{G-1}]

    Dimensions: ``3 + 3 + 1 + 2 + G = 9 + G``.

    When ``CT=None`` the one-hot columns are omitted: ``7 + G``.

    Returns
    -------
    node_features : torch.Tensor, shape ``[N, F_node]``
    """
    pos = _wp_to_torch(X, particle_count).float()  # [N, 3]
    pol = _wp_to_torch(P, particle_count).float()  # [N, 3]
    rad = _wp_to_torch(R, particle_count).float().unsqueeze(-1)  # [N, 1]

    parts = [pos, pol, rad]

    if CT is not None:
        ct = _wp_to_torch(CT, particle_count).long()  # [N]
        n = pos.size(0)
        ct_onehot = torch.zeros(n, 2, device=pos.device, dtype=pos.dtype)
        ct_onehot[torch.arange(n, device=pos.device), ct.clamp(max=1)] = 1.0
        parts.append(ct_onehot)

    if G is not None:
        genes = _wp_to_torch(G, particle_count).float()  # [N, num_genes]
        if genes.ndim == 1:
            genes = genes.unsqueeze(-1)
        parts.append(genes)

    feats = torch.cat(parts, dim=-1)

    if device is not None:
        feats = feats.to(device)

    return feats


def build_edge_features(
    X: wp.array,
    R: wp.array,
    CT: wp.array | None = None,
    edge_index: torch.Tensor | None = None,
    particle_count: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute per-edge feature tensor.

    Feature layout per edge ``(i -> j)`` when CT is provided::

        [dx, dy, dz, dist, r_i, r_j, ct_same]

    Dimensions: ``3 + 1 + 1 + 1 + 1 = 7``.

    When ``CT=None`` the ``ct_same`` column is omitted (dim = 6).

    Returns
    -------
    edge_features : torch.Tensor, shape ``[E, 6 or 7]``
    """
    pos = _wp_to_torch(X, particle_count).float()  # [N, 3]
    rad = _wp_to_torch(R, particle_count).float()  # [N]

    senders = edge_index[0]  # [E]
    receivers = edge_index[1]  # [E]

    rel_pos = pos[senders] - pos[receivers]  # [E, 3]
    dist = rel_pos.norm(dim=-1, keepdim=True)  # [E, 1]
    r_s = rad[senders].unsqueeze(-1)  # [E, 1]
    r_r = rad[receivers].unsqueeze(-1)  # [E, 1]

    parts = [rel_pos, dist, r_s, r_r]

    if CT is not None:
        ct = _wp_to_torch(CT, particle_count).long()  # [N]
        ct_same = (ct[senders] == ct[receivers]).float().unsqueeze(-1)  # [E, 1]
        parts.append(ct_same)

    feats = torch.cat(parts, dim=-1)

    if device is not None:
        feats = feats.to(device)

    return feats


def build_graph(
    X: wp.array,
    P: wp.array,
    R: wp.array,
    CT: wp.array | None = None,
    particle_count: int = 0,
    G: wp.array | None = None,
    eps_dist: float = EPS_DIST,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(node_features, edge_index, edge_features)`` in one call.

    Parameters
    ----------
    CT : wp.array, optional
        Cell type array.  When ``None``, cell-type features are omitted.

    Returns
    -------
    node_features : torch.Tensor ``[N, F_node]``
    edge_index    : torch.Tensor ``[2, E]``
    edge_features : torch.Tensor ``[E, 6 or 7]``
    """
    edge_index = build_edge_index(X, R, particle_count, eps_dist, device)
    node_features = build_node_features(X, P, R, CT, particle_count, G, device)
    edge_features = build_edge_features(X, R, CT, edge_index, particle_count, device)
    return node_features, edge_index, edge_features
