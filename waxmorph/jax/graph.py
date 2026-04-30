"""Build JAX-style graph data from Warp simulation state or live JAX arrays.

Topology construction is intentionally non-differentiable: positions/radii are
snapshotted to the host to build adjacency.  Feature construction stays in JAX,
so gradients can flow through node and edge features to live state arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from .._graph_core import build_edge_index_np
from ..constants import EPS_DIST

ANGLE_EPS = 1e-6


def _slice_active(arr, particle_count: int):
    """Slice inputs down to active particles when a count is provided."""
    if particle_count <= 0:
        return arr
    return arr[:particle_count]


def _wp_to_jax(arr: wp.array, particle_count: int) -> jax.Array:
    """Convert a Warp array to a :class:`jax.Array`, sliced to active particles.

    Uses Warp's JAX conversion for zero-copy DLPack transfer when available,
    with a :mod:`numpy` round-trip fallback.
    """
    try:
        t = wp.to_jax(arr)
    except Exception:
        t = jnp.asarray(arr.numpy())
    return _slice_active(t, particle_count)


def _as_jax(arr, particle_count: int) -> jax.Array:
    """Convert supported arrays to JAX without detaching live :class:`jax.Array` inputs."""
    if arr is None:
        raise TypeError("Expected a Warp or JAX array, got None.")

    if isinstance(arr, jax.Array):
        return _slice_active(arr, particle_count)

    if isinstance(arr, wp.array):
        return _wp_to_jax(arr, particle_count)

    return _slice_active(jnp.asarray(arr), particle_count)


def _snapshot_numpy(arr, particle_count: int) -> np.ndarray:
    """Materialize a detached NumPy host snapshot for non-differentiable topology."""
    return np.asarray(jax.device_get(_as_jax(arr, particle_count)))


def build_edge_index(
    X,
    R,
    particle_count: int,
    eps_dist: float = EPS_DIST,
    max_edges: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct COO edge_index ``[2, E]`` from contact adjacency.

    An edge ``(i, j)`` exists when ``dist(X[i], X[j]) <= R[i] + R[j] + eps_dist``
    and ``i != j``.  Returns directed edges (both ``i->j`` and ``j->i``).

    Args:
        X: Position array with shape ``[N, 3]``.
        R: Radius array with shape ``[N]``.
        particle_count: Number of active particles. Non-positive values use
            the full arrays.
        eps_dist: Contact buffer matching ``simulator.EPS_DIST``.
        max_edges: Optional output capacity. If provided, padding entries use
            sender ``0`` and receiver ``0`` so shapes stay static across JIT
            calls.

    Returns:
        Pair ``(edge_index, num_edges)`` where ``edge_index`` has shape
        ``[2, E]`` or ``[2, max_edges]`` and ``num_edges`` is a
        :class:`jax.Array` scalar count of real, non-padding directed edges.

    Raises:
        ValueError: If the observed edge count exceeds ``max_edges``.
    """
    pos = _snapshot_numpy(X, particle_count).astype(np.float32, copy=False)
    rad = _snapshot_numpy(R, particle_count).astype(np.float32, copy=False)

    senders, receivers = build_edge_index_np(pos, rad, eps_dist)

    if len(senders) == 0:
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
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
    G,
    particle_count: int,
) -> jax.Array:
    """Assemble per-node gene features from Warp or JAX arrays.

    Feature layout per node::

        [g_0, g_1, ..., g_{G-1}]

    Args:
        G: Gene concentration array. One-dimensional arrays are promoted to
            shape ``[N, 1]``.
        particle_count: Number of active particles. Non-positive values use
            the full array.

    Returns:
        Float :class:`jax.Array` with shape ``[N, G]``.
    """
    genes = _as_jax(G, particle_count).astype(jnp.float32)
    if genes.ndim == 1:
        genes = genes[..., None]
    return genes


def build_edge_features(
    X,
    P,
    edge_index: jax.Array,
    particle_count: int,
) -> jax.Array:
    """Compute per-edge feature tensor.

    Feature layout per edge ``(i -> j)``::

        [dist, angle(P_i, P_j)]

    Args:
        X: Position array with shape ``[N, 3]``.
        P: Polarity array with shape ``[N, 3]``.
        edge_index: Directed COO edge array with shape ``[2, E]``.
        particle_count: Number of active particles. Non-positive values use
            the full arrays.

    Returns:
        Float :class:`jax.Array` edge feature array with shape ``[E, 2]``.
    """
    pos = _as_jax(X, particle_count).astype(jnp.float32)
    pol = _as_jax(P, particle_count).astype(jnp.float32)

    senders = edge_index[0]
    receivers = edge_index[1]
    self_edge = senders == receivers

    rel_pos = pos[senders] - pos[receivers]
    rel_pos = jnp.where(
        self_edge[:, None],
        jnp.array([EPS_DIST, 0.0, 0.0], dtype=pos.dtype),
        rel_pos,
    )
    dist = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True)

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
    G=None,
    eps_dist: float = EPS_DIST,
    max_edges: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build ``(node_features, edge_index, edge_features, num_edges)`` in one call.

    Args:
        X: Position array with shape ``[N, 3]``.
        P: Polarity array with shape ``[N, 3]``.
        R: Radius array with shape ``[N]``.
        particle_count: Number of active particles. Non-positive values use
            the full arrays.
        G: Gene concentration array.
        eps_dist: Contact buffer distance.
        max_edges: Optional capacity passed to :func:`build_edge_index`.

    Returns:
        Tuple ``(node_features, edge_index, edge_features, num_edges)``.
    """
    edge_index, num_edges = build_edge_index(X, R, particle_count, eps_dist, max_edges)
    node_features = build_node_features(G, particle_count)
    edge_features = build_edge_features(X, P, edge_index, particle_count)
    return node_features, edge_index, edge_features, num_edges
