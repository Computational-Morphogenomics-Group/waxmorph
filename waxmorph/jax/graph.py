"""JAX graph construction with detached topology and differentiable features.

Positions and radii are copied to the host for adjacency. The gradient contract covers JAX
node and edge features built from live state arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from .._graph_core import (
    _active_particle_count,
    _validate_feature_shapes,
    build_edge_index_np,
)
from ..constants import EPS_DIST, EPS_NORM

ANGLE_EPS = 1e-6


def _slice_active(arr, particle_count: int):
    active = _active_particle_count(particle_count, array=arr)
    if particle_count <= 0:
        return arr
    return arr[:active]


def _wp_to_jax(arr: wp.array, particle_count: int) -> jax.Array:
    try:
        t = wp.to_jax(arr)
    except Exception:
        t = jnp.asarray(arr.numpy())
    return _slice_active(t, particle_count)


def _as_jax(arr, particle_count: int) -> jax.Array:
    if arr is None:
        raise TypeError("Expected a Warp or JAX array, got None.")

    if isinstance(arr, jax.Array):
        return _slice_active(arr, particle_count)

    if isinstance(arr, wp.array):
        return _wp_to_jax(arr, particle_count)

    return _slice_active(jnp.asarray(arr), particle_count)


def _snapshot_numpy(arr, particle_count: int) -> np.ndarray:
    return np.asarray(jax.device_get(_as_jax(arr, particle_count)))


def build_edge_index(
    X,
    R,
    particle_count: int,
    eps_dist: float = EPS_DIST,
    max_edges: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build directed COO adjacency from a detached host snapshot.

    A detached float32 host snapshot includes both directions when ``i != j`` and
    ``dist(X[i], X[j]) <= R[i] + R[j] + eps_dist``. Nonpositive ``particle_count`` uses all
    rows. ``max_edges=None`` returns ``[2, E]`` with ``num_edges == E``. A fixed capacity
    returns ``[2, max_edges]`` padded by ``(0, 0)``; real edges occupy columns below
    ``num_edges``. Capacity overflow raises ``ValueError``.

    Examples:
        >>> X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        >>> R = jnp.array([0.6, 0.6, 0.6])
        >>> edge_index, num_edges = build_edge_index(X, R, 3, eps_dist=0.0)
        >>> print(edge_index.tolist(), int(num_edges))
        [[0, 1], [1, 0]] 2
    """
    active = _active_particle_count(particle_count, positions=X, radii=R)
    pos = _snapshot_numpy(X, active).astype(np.float32, copy=False)
    rad = _snapshot_numpy(R, active).astype(np.float32, copy=False)

    senders, receivers = build_edge_index_np(pos, rad, eps_dist)
    ei = np.stack([senders, receivers], axis=0)
    num_edges = ei.shape[1]

    if max_edges is not None:
        if num_edges > max_edges:
            raise ValueError(
                f"Graph has {num_edges} edges but max_edges={max_edges}. "
                f"Increase max_edges to at least {num_edges}."
            )
        padded = np.zeros((2, max_edges), dtype=np.int64)
        padded[:, :num_edges] = ei
        ei = padded

    return jnp.array(ei, dtype=jnp.int32), jnp.int32(num_edges)


def build_node_features(
    c,
    particle_count: int,
) -> jax.Array:
    """Return concentrations as float32 ``[N, C]`` node features.

    Rank-one input becomes ``[N, 1]``; nonpositive ``particle_count`` uses every row.

    Examples:
        >>> c = jnp.array([0.2, 0.4, 0.8])
        >>> print(build_node_features(c, 3).tolist())
        [[0.20000000298023224], [0.4000000059604645], [0.800000011920929]]
    """
    active = _active_particle_count(particle_count, concentrations=c)
    c = _as_jax(c, active).astype(jnp.float32)
    _validate_feature_shapes(active, concentrations=c)
    if c.ndim == 1:
        c = c[..., None]
    return c


def build_edge_features(
    X,
    P,
    edge_index: jax.Array,
    particle_count: int,
) -> jax.Array:
    r"""Return float32 ``[distance, polarity angle]`` for each directed edge.

    .. math::

        d_{ij} &= \sqrt{\lVert x_i-x_j\rVert_2^2 + \mathrm{EPS\_NORM}^2}, \\
        \theta_{ij} &= \arccos\!\left(\operatorname{clip}
        (p_i^\top p_j,-1+\mathrm{ANGLE\_EPS},1-\mathrm{ANGLE\_EPS})\right).

    JAX regularizes every real-edge distance by ``EPS_NORM``; PyTorch uses the exact norm.
    ``P`` must contain unit vectors because the feature uses supplied dot products directly.
    Clamping keeps ``arccos`` gradients finite. Self-edge padding is masked to zero in both
    columns.

    Examples:
        >>> X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> P = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> edge_index = jnp.array([[0, 1], [1, 0]])
        >>> print(jnp.round(build_edge_features(X, P, edge_index, 2), 4).tolist())
        [[1.0, 0.00139999995008111], [1.0, 0.00139999995008111]]
    """
    active = _active_particle_count(particle_count, positions=X, polarities=P)
    pos = _as_jax(X, active).astype(jnp.float32)
    pol = _as_jax(P, active).astype(jnp.float32)
    edge_index = jnp.asarray(edge_index)
    _validate_feature_shapes(active, positions=pos, polarities=pol, edge_index=edge_index)
    if not jnp.issubdtype(edge_index.dtype, jnp.integer):
        raise TypeError("edge_index must have an integer dtype.")
    # Concrete indices are range-checked; tracing skips host validation.
    try:
        concrete_edges = np.asarray(jax.device_get(edge_index))
    except jax.errors.TracerArrayConversionError:
        concrete_edges = None
    if (
        concrete_edges is not None
        and concrete_edges.size
        and (concrete_edges.min() < 0 or concrete_edges.max() >= active)
    ):
        raise ValueError(f"edge_index values must lie in [0, {active}).")

    senders = edge_index[0]
    receivers = edge_index[1]
    self_edge = senders == receivers

    rel_pos = pos[senders] - pos[receivers]
    rel_pos = jnp.where(
        self_edge[:, None],
        jnp.array([EPS_DIST, 0.0, 0.0], dtype=pos.dtype),
        rel_pos,
    )
    dist = jnp.sqrt(jnp.sum(rel_pos * rel_pos, axis=-1, keepdims=True) + EPS_NORM**2)

    p_s = pol[senders]
    p_r = pol[receivers]
    cos_angle = jnp.clip(
        jnp.sum(p_s * p_r, axis=-1, keepdims=True),
        -1.0 + ANGLE_EPS,
        1.0 - ANGLE_EPS,
    )
    angle = jnp.arccos(cos_angle)

    edge_features = jnp.concatenate([dist, angle], axis=-1)
    return jnp.where(self_edge[:, None], 0.0, edge_features)


def build_graph(
    X,
    P,
    R,
    particle_count: int = 0,
    c=None,
    eps_dist: float = EPS_DIST,
    max_edges: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build JAX node features, directed edges, edge features, and real-edge count.

    ``c`` is required despite its compatibility default. Shapes are ``[N, C]``, ``[2, Q]``,
    ``[Q, 2]``, and scalar, where ``Q`` is the real edge count or ``max_edges`` when padded.
    JAX's fourth value and optional padding differ from PyTorch's unpadded three-tuple.

    Examples:
        >>> X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        >>> P, R, c = jnp.ones((3, 3)), jnp.array([0.6, 0.6, 0.6]), jnp.ones(3)
        >>> out = build_graph(X, P, R, 3, c, eps_dist=0.0)
        >>> print([tuple(a.shape) for a in out[:3]], int(out[3]))
        [(3, 1), (2, 2), (2, 2)] 2
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
    edge_index, num_edges = build_edge_index(X, R, active, eps_dist, max_edges)
    node_features = build_node_features(c, active)
    if active == 0 and edge_index.shape[1]:
        # Empty graphs construct padding features directly because they have zero nodes.
        pos = _as_jax(X, active).astype(jnp.float32)
        pol = _as_jax(P, active).astype(jnp.float32)
        _validate_feature_shapes(active, positions=pos, polarities=pol)
        edge_features = jnp.zeros((edge_index.shape[1], 2), dtype=jnp.float32)
    else:
        edge_features = build_edge_features(X, P, edge_index, active)
    return node_features, edge_index, edge_features, num_edges
