"""Build PyTorch-geometric-style graph data from simulation state.

Supports both Warp arrays and Torch tensors as inputs. When Torch
tensors are provided, node and edge feature construction remains on the
PyTorch autograd graph; only adjacency construction uses a detached
snapshot of positions and radii.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from .._graph_core import build_edge_index_np
from ..constants import EPS_DIST

ANGLE_EPS = 1e-6


def _resolve_device(
    device: torch.device | str | None,
    *arrays: torch.Tensor | wp.array | None,
) -> torch.device:
    """Pick the output device from an explicit request or the first input."""
    if device is not None:
        return torch.device(device)

    for arr in arrays:
        if isinstance(arr, torch.Tensor):
            return arr.device
        if arr is not None:
            return torch.device(str(arr.device))

    return torch.device("cpu")


def _slice_active(
    arr: torch.Tensor | wp.array,
    particle_count: int,
) -> torch.Tensor | wp.array:
    """Slice inputs down to active particles when a count is provided."""
    if particle_count <= 0:
        return arr
    return arr[:particle_count]


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
    return _slice_active(t, particle_count)


def _as_torch(
    arr: torch.Tensor | wp.array,
    particle_count: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert Warp arrays or slice Torch tensors without breaking autograd."""
    target = None if device is None else torch.device(device)

    if isinstance(arr, torch.Tensor):
        t = _slice_active(arr, particle_count)
        if target is not None and t.device != target:
            t = t.to(target)
        return t

    t = _wp_to_torch(arr, particle_count)
    if target is not None and t.device != target:
        t = t.to(target)
    return t


def _snapshot_numpy(arr: torch.Tensor | wp.array, particle_count: int) -> np.ndarray:
    """Materialize a detached CPU snapshot for non-differentiable topology."""
    return _as_torch(arr, particle_count).detach().cpu().numpy()


def _validate_finite_numpy(name: str, arr: np.ndarray) -> None:
    """Raise a clear error before passing invalid data into scipy/spatial ops."""
    finite_mask = np.isfinite(arr)
    if finite_mask.all():
        return

    bad_indices = np.argwhere(~finite_mask)
    first_bad = tuple(int(i) for i in bad_indices[0])
    bad_value = arr[first_bad]
    raise ValueError(
        f"Non-finite values detected in {name} before KDTree construction: "
        f"total_bad={(~finite_mask).sum()}, first_bad_index={first_bad}, "
        f"first_bad_value={bad_value!r}"
    )


def build_edge_index(
    X: torch.Tensor | wp.array,
    R: torch.Tensor | wp.array,
    particle_count: int,
    eps_dist: float = EPS_DIST,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct COO edge_index ``[2, E]`` from contact adjacency.

    An edge ``(i, j)`` exists when ``dist(X[i], X[j]) <= R[i] + R[j] + eps_dist``
    and ``i != j``. Returns directed edges (both ``i->j`` and ``j->i``).

    Uses ``scipy.spatial.cKDTree`` for O(N log N) neighbor queries instead
    of an O(N^2) pairwise distance matrix. For Torch inputs this function
    intentionally snapshots detached CPU copies of positions and radii, so
    edge construction is frozen for the current rollout step.
    """
    pos = _snapshot_numpy(X, particle_count).astype(np.float32, copy=False)
    rad = _snapshot_numpy(R, particle_count).astype(np.float32, copy=False)
    _validate_finite_numpy("positions", pos)
    _validate_finite_numpy("radii", rad)

    senders, receivers = build_edge_index_np(pos, rad, eps_dist)

    if len(senders) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.from_numpy(np.stack([senders, receivers], axis=0))

    return edge_index.to(_resolve_device(device, X, R))


def build_node_features(
    G: torch.Tensor | wp.array,
    particle_count: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Assemble per-node feature tensor from Warp state arrays or Torch tensors.

    Feature layout per node::

        [g_0, g_1, ..., g_{G-1}]

    Dimensions: ``G`` (number of genes).
    """
    genes = _as_torch(G, particle_count, device).float()
    if genes.ndim == 1:
        genes = genes.unsqueeze(-1)
    return genes


def build_edge_features(
    X: torch.Tensor | wp.array,
    P: torch.Tensor | wp.array,
    edge_index: torch.Tensor,
    particle_count: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute per-edge feature tensor.

    Feature layout per edge ``(i -> j)``::

        [dist, angle(P_i, P_j)]

    Dimensions: ``1 + 1 = 2``.
    """
    target = _resolve_device(device, X, P)
    pos = _as_torch(X, particle_count, target).float()
    pol = _as_torch(P, particle_count, target).float()
    if edge_index.device != pos.device:
        edge_index = edge_index.to(pos.device)

    senders = edge_index[0]
    receivers = edge_index[1]

    rel_pos = pos[senders] - pos[receivers]
    dist = rel_pos.norm(dim=-1, keepdim=True)

    p_s = pol[senders]
    p_r = pol[receivers]
    cos_angle = (p_s * p_r).sum(dim=-1, keepdim=True).clamp(-1.0 + ANGLE_EPS, 1.0 - ANGLE_EPS)
    angle = torch.acos(cos_angle)

    return torch.cat([dist, angle], dim=-1)


def build_graph(
    X: torch.Tensor | wp.array,
    P: torch.Tensor | wp.array,
    R: torch.Tensor | wp.array,
    particle_count: int = 0,
    G: torch.Tensor | wp.array | None = None,
    eps_dist: float = EPS_DIST,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(node_features, edge_index, edge_features)`` in one call.

    For Torch inputs, gradients flow through ``node_features`` and
    ``edge_features`` back to the live state tensors. ``edge_index`` is
    intentionally built from a detached snapshot of ``X`` and ``R`` and
    should be treated as frozen for that rollout step.
    """
    edge_index = build_edge_index(X, R, particle_count, eps_dist, device)
    node_features = build_node_features(G, particle_count, device)
    edge_features = build_edge_features(X, P, edge_index, particle_count, device)
    return node_features, edge_index, edge_features
