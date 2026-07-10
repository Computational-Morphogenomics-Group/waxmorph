"""Build PyTorch-geometric-style graph data from simulation state.

Supports both Warp arrays and Torch tensors as inputs. When Torch
tensors are provided, node and edge feature construction remains on the
:mod:`torch.autograd` graph; only adjacency construction uses a detached
snapshot of positions and radii.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from .._graph_core import (
    _active_particle_count,
    _validate_feature_shapes,
    build_edge_index_np,
)
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
            inferred = torch.device(str(arr.device))
            if inferred.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return inferred

    return torch.device("cpu")


def _slice_active(
    arr: torch.Tensor | wp.array,
    particle_count: int,
) -> torch.Tensor | wp.array:
    """Slice inputs down to active particles when a count is provided."""
    active = _active_particle_count(particle_count, array=arr)
    if particle_count <= 0:
        return arr
    return arr[:active]


def _wp_to_torch(arr: wp.array, particle_count: int) -> torch.Tensor:
    """Convert a Warp array to a :class:`torch.Tensor`, sliced to active particles.

    Handles both scalar (float32, uint32) and vector (vec3f) dtypes by
    going through :mod:`numpy` when :func:`warp.to_torch` is unavailable or when the
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

    A detached CPU snapshot freezes topology for the current rollout step.
    KD-tree queries avoid constructing a dense pairwise-distance matrix; work
    also scales with the candidate and output pair counts.

    Args:
        X: Position array with shape ``[N, 3]``.
        R: Radius array with shape ``[N]``.
        particle_count: Number of active particles. Non-positive values use
            the full arrays.
        eps_dist: Contact buffer distance.
        device: Output device for the edge tensor. Defaults to the device of
            the first non-``None`` input array.

    Returns:
        Directed COO edge tensor with shape ``[2, E]``, where ``E`` is the
        number of real contact edges (no padding; the JAX twin returns a
        padded ``[2, max_edges]`` tensor plus an explicit ``num_edges`` count).

    See Also:
        :func:`waxmorph.jax.graph.build_edge_index`: JAX twin that pads to a
        static ``max_edges`` capacity and returns ``(edge_index, num_edges)``
        for compile-time array sizes.

    Examples:
        >>> X = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        >>> R = torch.tensor([0.6, 0.6, 0.6])
        >>> print(build_edge_index(X, R, 3, eps_dist=0.0).tolist())
        [[0, 1], [1, 0]]
    """
    active = _active_particle_count(particle_count, positions=X, radii=R)
    pos = _snapshot_numpy(X, active).astype(np.float32, copy=False)
    rad = _snapshot_numpy(R, active).astype(np.float32, copy=False)

    senders, receivers = build_edge_index_np(pos, rad, eps_dist)

    if len(senders) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.from_numpy(np.stack([senders, receivers], axis=0))

    return edge_index.to(_resolve_device(device, X, R))


def build_node_features(
    c: torch.Tensor | wp.array,
    particle_count: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Assemble per-node signaling-molecule features from Warp or Torch arrays.

    Feature layout per node::

        [c_0, c_1, ..., c_{C-1}]

    Args:
        c: Signaling-molecule concentration array. One-dimensional arrays are
            promoted to shape ``[N, 1]``.
        particle_count: Number of active particles. Non-positive values use
            the full array.
        device: Output device for the feature tensor. Defaults to the device
            of the input array.

    Returns:
        Float feature tensor with shape ``[N, C]``, where ``C`` is the number
        of signaling molecules.

    See Also:
        :func:`waxmorph.jax.graph.build_node_features`: JAX twin.

    Examples:
        >>> c = torch.tensor([0.2, 0.4, 0.8])
        >>> print(build_node_features(c, 3).tolist())
        [[0.20000000298023224], [0.4000000059604645], [0.800000011920929]]
    """
    active = _active_particle_count(particle_count, concentrations=c)
    c = _as_torch(c, active, device).float()
    _validate_feature_shapes(active, concentrations=c)
    if c.ndim == 1:
        c = c.unsqueeze(-1)
    return c


def build_edge_features(
    X: torch.Tensor | wp.array,
    P: torch.Tensor | wp.array,
    edge_index: torch.Tensor,
    particle_count: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Compute per-edge feature tensor.

    Feature layout per edge ``(i -> j)``::

        [dist, angle(P_i, P_j)]

    The two features encode the mechanical relationship of a neighbor pair:

    .. math::

        d_{ij} = \lVert x_i - x_j \rVert_2, \qquad
        \theta_{ij} = \arccos\!\left( p_i^\top p_j \right).

    ``P`` must contain unit vectors because the raw dot product is not
    normalized. Clamping to ``[-1 + ANGLE_EPS, 1 - ANGLE_EPS]`` keeps the
    :func:`torch.acos` gradient finite at parallel and antiparallel inputs.

    Args:
        X: Position array with shape ``[N, 3]``.
        P: Polarity array with shape ``[N, 3]``.
        edge_index: Directed COO edge tensor with shape ``[2, E]``.
        particle_count: Number of active particles. Non-positive values use
            the full arrays.
        device: Output device for the feature tensor. Defaults to the device
            of the first non-``None`` input array.

    Returns:
        Float edge feature tensor with shape ``[E, 2]`` whose columns are
        ``[d_ij, theta_ij]``.

    See Also:
        :func:`waxmorph.jax.graph.build_edge_features`: JAX twin.

    Examples:
        >>> X = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> P = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> edge_index = torch.tensor([[0, 1], [1, 0]])
        >>> print(build_edge_features(X, P, edge_index, 2).round(decimals=4).tolist())
        [[1.0, 0.00139999995008111], [1.0, 0.00139999995008111]]
    """
    active = _active_particle_count(particle_count, positions=X, polarities=P)
    target = _resolve_device(device, X, P)
    pos = _as_torch(X, active, target).float()
    pol = _as_torch(P, active, target).float()
    _validate_feature_shapes(active, positions=pos, polarities=pol, edge_index=edge_index)
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("edge_index dtype must be torch.int32 or torch.int64.")
    if edge_index.device != pos.device:
        edge_index = edge_index.to(pos.device)
    if edge_index.numel() and (edge_index.min().item() < 0 or edge_index.max().item() >= active):
        raise ValueError(f"edge_index values must lie in [0, {active}).")

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
    c: torch.Tensor | wp.array | None = None,
    eps_dist: float = EPS_DIST,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(node_features, edge_index, edge_features)`` in one call.

    For Torch inputs, gradients flow through ``node_features`` and
    ``edge_features`` back to the live state tensors. ``edge_index`` is
    intentionally built from a detached snapshot of ``X`` and ``R`` and
    should be treated as frozen for that rollout step.

    Args:
        X: Position array with shape ``[N, 3]``.
        P: Polarity array with shape ``[N, 3]``.
        R: Radius array with shape ``[N]``.
        particle_count: Number of active particles. Non-positive values use
            the full arrays.
        c: Required signaling-molecule concentration array.
        eps_dist: Contact buffer distance.
        device: Output device for all returned tensors. Defaults to the device
            of the first non-``None`` input array.

    Returns:
        Tuple ``(node_features, edge_index, edge_features)`` with shapes
        ``[N, C]``, ``[2, E]``, and ``[E, 2]``. This is a 3-tuple; the JAX twin
        returns a 4-tuple with a trailing ``num_edges`` count because its
        ``edge_index`` is padded to a static ``max_edges`` capacity for
        compile-time array sizes.

    See Also:
        :func:`waxmorph.jax.graph.build_graph`: JAX twin returning the extra
        ``num_edges`` count for static-shape compilation.

    Examples:
        >>> X = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        >>> P, R, c = torch.ones(3, 3), torch.tensor([0.6, 0.6, 0.6]), torch.ones(3)
        >>> print([tuple(t.shape) for t in build_graph(X, P, R, 3, c, eps_dist=0.0)])
        [(3, 1), (2, 2), (2, 2)]
    """
    if c is None:
        raise TypeError("build_graph() requires `c`.")
    active = _active_particle_count(
        particle_count,
        positions=X,
        polarities=P,
        radii=R,
        concentrations=c,
    )
    edge_index = build_edge_index(X, R, active, eps_dist, device)
    node_features = build_node_features(c, active, device)
    edge_features = build_edge_features(X, P, edge_index, active, device)
    return node_features, edge_index, edge_features
