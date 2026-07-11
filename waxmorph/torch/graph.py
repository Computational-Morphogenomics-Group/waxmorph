"""PyTorch graph features with detached contact topology.

Torch node and edge features retain autograd; contacts use a detached host snapshot.
Warp arrays are also accepted.
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
    active = _active_particle_count(particle_count, array=arr)
    if particle_count <= 0:
        return arr
    return arr[:active]


def _wp_to_torch(arr: wp.array, particle_count: int) -> torch.Tensor:
    """Use NumPy if :func:`warp.to_torch` raises."""
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
    return _as_torch(arr, particle_count).detach().cpu().numpy()


def build_edge_index(
    X: torch.Tensor | wp.array,
    R: torch.Tensor | wp.array,
    particle_count: int,
    eps_dist: float = EPS_DIST,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Return unpadded directed COO contacts with shape ``[2, E]``.

    A detached float32 CPU snapshot includes both orientations when :math:`i\ne j` and
    :math:`\lVert x_i-x_j\rVert_2\le R_i+R_j+\varepsilon`, where
    :math:`\varepsilon=\mathtt{eps\_dist}`. This excludes topology from autodiff. Nonpositive
    ``particle_count`` uses all rows; ``device`` otherwise defaults from the inputs. JAX can
    pad to a fixed edge capacity and returns an additional edge count.

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
    edge_index = torch.from_numpy(np.stack([senders, receivers], axis=0))
    return edge_index.to(_resolve_device(device, X, R))


def build_node_features(
    c: torch.Tensor | wp.array,
    particle_count: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return float concentrations as ``[N, C]``, promoting ``[N]`` to ``[N, 1]``.

    Nonpositive ``particle_count`` uses all rows; ``device`` defaults from the input.

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
    r"""Return geometric features for each directed edge.

    .. math::

        [d_{ij},\theta_{ij}]
        = [\lVert x_i-x_j\rVert_2,\arccos(p_i^\top p_j)].

    ``P`` must contain unit vectors because its dot products are not normalized. The dot
    product is clamped to ``[-1 + ANGLE_EPS, 1 - ANGLE_EPS]`` before ``acos``. Torch uses
    the exact Euclidean norm; JAX uses ``sqrt(||x_i-x_j||^2 + EPS_NORM^2)`` and masks padded
    edges. Torch inputs retain gradients through positions and polarities. The output shape
    is ``[E, 2]``.

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
    """Return ``(node_features, edge_index, edge_features)``.

    Shapes are ``[N, C]``, ``[2, E]``, and ``[E, 2]``. Torch features remain
    differentiable, but contacts use a detached position/radius snapshot. ``c`` is required.
    JAX instead returns a 4-tuple with an edge count and optional padding.

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
