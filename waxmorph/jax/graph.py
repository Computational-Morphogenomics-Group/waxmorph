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
    """Convert a Warp array to a JAX array, sliced to active particles.

    Uses ``wp.to_jax()`` for zero-copy dlpack transfer when available,
    with a numpy round-trip fallback.
    """
    try:
        t = wp.to_jax(arr)
    except Exception:
        t = jnp.asarray(arr.numpy())
    return _slice_active(t, particle_count)


def _as_jax(arr, particle_count: int) -> jax.Array:
    """Convert supported arrays to JAX without detaching live JAX inputs."""
    if arr is None:
        raise TypeError("Expected a Warp or JAX array, got None.")

    if isinstance(arr, jax.Array):
        return _slice_active(arr, particle_count)

    if isinstance(arr, wp.array):
        return _wp_to_jax(arr, particle_count)

    return _slice_active(jnp.asarray(arr), particle_count)


def _snapshot_numpy(arr, particle_count: int) -> np.ndarray:
    """Materialize a detached host snapshot for non-differentiable topology."""
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

    Parameters
    ----------
    X : wp.array(dtype=wp.vec3f) or jax.Array
        Position array.
    R : wp.array(dtype=wp.float32) or jax.Array
        Radius array.
    particle_count : int
        Number of active particles.
    eps_dist : float
        Contact buffer matching ``simulator.EPS_DIST``.
    max_edges : int, optional
        If given, pad the output to exactly ``max_edges`` columns.
        Padding entries point sender=0, receiver=0.  This keeps array shapes
        static across calls, which is **critical** for ``jax.jit`` — without
        it, every new edge count triggers a full re-trace (~3-5 s each).

        When padding is used, pass the returned ``num_edges`` to the GNS so
        it can mask out padding before aggregation (see
        :class:`~waxmorph.jax.gnn.GraphNetworkBlock`).

    Returns
    -------
    edge_index : jnp.ndarray, shape ``[2, E]`` or ``[2, max_edges]``, dtype ``int32``
    num_edges : jnp.int32
        Number of real (non-padding) edges.  Returned as a JAX scalar so
        it can be passed into ``jax.jit``-compiled functions as a traced
        (dynamic) value — a plain Python ``int`` would be treated as a
        static constant, causing recompilation every time the edge count
        changes.
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
    """Assemble per-node feature tensor from Warp state arrays.

    Feature layout per node::

        [g_0, g_1, ..., g_{G-1}]

    Returns
    -------
    node_features : jnp.ndarray, shape ``[N, G]``
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

    Returns
    -------
    edge_features : jnp.ndarray, shape ``[E, 2]``
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

    Parameters
    ----------
    max_edges : int, optional
        Passed to :func:`build_edge_index` to pad edge arrays to a fixed
        size.  **Required** when calling the GNS inside ``jax.jit`` to
        prevent re-tracing on every graph with a different edge count.

    Returns
    -------
    node_features : jnp.ndarray ``[N, G]``
    edge_index    : jnp.ndarray ``[2, E]`` or ``[2, max_edges]``
    edge_features : jnp.ndarray ``[E, 2]`` or ``[max_edges, 2]``
    num_edges : jnp.int32
        Number of real (non-padding) edges (JAX scalar).
    """
    edge_index, num_edges = build_edge_index(X, R, particle_count, eps_dist, max_edges)
    node_features = build_node_features(G, particle_count)
    edge_features = build_edge_features(X, P, edge_index, particle_count)
    return node_features, edge_index, edge_features, num_edges
