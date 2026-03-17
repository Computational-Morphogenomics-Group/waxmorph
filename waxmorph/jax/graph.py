"""Build JAX-style graph data from Warp simulation state.

Converts Warp arrays into JAX arrays and constructs the adjacency
graph, node features, and edge features needed by the GNS.
"""

import jax.numpy as jnp
import numpy as np
import warp as wp

from .._graph_core import build_edge_index_np
from ..constants import EPS_DIST


def _wp_to_jax(arr: wp.array, particle_count: int) -> jnp.ndarray:
    """Convert a Warp array to a JAX array, sliced to active particles.

    Uses ``wp.to_jax()`` for zero-copy dlpack transfer when available,
    with a numpy round-trip fallback.
    """
    try:
        t = wp.to_jax(arr)
    except Exception:
        t = jnp.asarray(arr.numpy())
    return t[:particle_count]


def build_edge_index(
    X: wp.array,
    R: wp.array,
    particle_count: int,
    eps_dist: float = EPS_DIST,
    max_edges: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct COO edge_index ``[2, E]`` from contact adjacency.

    An edge ``(i, j)`` exists when ``dist(X[i], X[j]) <= R[i] + R[j] + eps_dist``
    and ``i != j``.  Returns directed edges (both ``i->j`` and ``j->i``).

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
    pos = np.asarray(_wp_to_jax(X, particle_count), dtype=np.float32)
    rad = np.asarray(_wp_to_jax(R, particle_count), dtype=np.float32)

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
    G: wp.array,
    particle_count: int,
) -> jnp.ndarray:
    """Assemble per-node feature tensor from Warp state arrays.

    Feature layout per node::

        [g_0, g_1, ..., g_{G-1}]

    Returns
    -------
    node_features : jnp.ndarray, shape ``[N, G]``
    """
    genes = _wp_to_jax(G, particle_count).astype(jnp.float32)
    if genes.ndim == 1:
        genes = genes[..., None]
    return genes


def build_edge_features(
    X: wp.array,
    P: wp.array,
    edge_index: jnp.ndarray,
    particle_count: int,
) -> jnp.ndarray:
    """Compute per-edge feature tensor.

    Feature layout per edge ``(i -> j)``::

        [dist, angle(P_i, P_j)]

    Returns
    -------
    edge_features : jnp.ndarray, shape ``[E, 2]``
    """
    pos = _wp_to_jax(X, particle_count).astype(jnp.float32)
    pol = _wp_to_jax(P, particle_count).astype(jnp.float32)

    senders = edge_index[0]
    receivers = edge_index[1]

    rel_pos = pos[senders] - pos[receivers]
    dist = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True)

    p_s = pol[senders]
    p_r = pol[receivers]
    cos_angle = jnp.clip(jnp.sum(p_s * p_r, axis=-1, keepdims=True), -1.0, 1.0)
    angle = jnp.arccos(cos_angle)

    return jnp.concatenate([dist, angle], axis=-1)


def build_graph(
    X: wp.array,
    P: wp.array,
    R: wp.array,
    particle_count: int = 0,
    G: wp.array | None = None,
    eps_dist: float = EPS_DIST,
    max_edges: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
