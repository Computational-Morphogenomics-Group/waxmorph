"""Tests for JAX graph construction from Warp arrays."""

import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from waxmorph.jax.graph import (
    build_edge_features,
    build_edge_index,
    build_graph,
    build_node_features,
)

wp.init()

MAX_PARTICLES = 20

DEVICE = "cpu"

try:
    if wp.is_device_available("cuda"):
        DEVICE = "cuda"
except RuntimeError:
    DEVICE = "cpu"


def _make_state(positions, radii, polarities=None, num_genes=2):
    """Helper to create Warp arrays from numpy data."""
    n = len(positions)
    pos = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)
    pos[:n] = positions
    X = wp.from_numpy(pos, dtype=wp.vec3f, device=DEVICE)

    rad = np.zeros(MAX_PARTICLES, dtype=np.float32)
    rad[:n] = radii
    R = wp.from_numpy(rad, dtype=wp.float32, device=DEVICE)

    pol = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)
    if polarities is not None:
        pol[:n] = polarities
    else:
        pol[:n, 2] = 1.0  # default: z-axis
    P = wp.from_numpy(pol, dtype=wp.vec3f, device=DEVICE)

    genes = np.zeros((MAX_PARTICLES, num_genes), dtype=np.float32)
    genes[:n] = np.random.default_rng(42).random((n, num_genes)).astype(np.float32)
    G = wp.from_numpy(genes, dtype=wp.float32, device=DEVICE)

    return X, P, R, G, n


class TestBuildEdgeIndex:
    def test_two_touching_one_far(self):
        """Two particles within contact range, third far away."""
        positions = [[0, 0, 0], [0.9, 0, 0], [10, 0, 0]]
        radii = [0.5, 0.5, 0.5]
        X, _P, R, _G, n = _make_state(positions, radii)

        edge_index, num_edges = build_edge_index(X, R, particle_count=n)

        assert edge_index.shape[0] == 2
        assert num_edges == 2
        edges = set(map(tuple, np.asarray(edge_index.T).tolist()))
        assert (0, 1) in edges
        assert (1, 0) in edges
        assert len(edges) == 2

    def test_no_self_loops(self):
        positions = [[0, 0, 0], [0.5, 0, 0]]
        radii = [0.5, 0.5]
        X, _P, R, _G, n = _make_state(positions, radii)

        edge_index, _num_edges = build_edge_index(X, R, particle_count=n)

        for s, r in zip(
            np.asarray(edge_index[0]).tolist(),
            np.asarray(edge_index[1]).tolist(),
            strict=False,
        ):
            assert s != r

    def test_all_connected(self):
        positions = [[0, 0, 0], [0.5, 0, 0], [0.25, 0.4, 0]]
        radii = [0.5, 0.5, 0.5]
        X, _P, R, _G, n = _make_state(positions, radii)

        edge_index, num_edges = build_edge_index(X, R, particle_count=n)

        assert edge_index.shape[1] == 6
        assert num_edges == 6

    def test_no_edges_far_apart(self):
        positions = [[0, 0, 0], [100, 0, 0]]
        radii = [0.5, 0.5]
        X, _P, R, _G, n = _make_state(positions, radii)

        edge_index, num_edges = build_edge_index(X, R, particle_count=n)

        assert edge_index.shape[1] == 0
        assert num_edges == 0


class TestBuildNodeFeatures:
    def test_shape_genes_only(self):
        positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        radii = [0.5, 0.5, 0.5]
        _X, _P, _R, G, n = _make_state(positions, radii, num_genes=4)

        feats = build_node_features(G, particle_count=n)

        assert feats.shape == (3, 4)

    def test_single_gene(self):
        positions = [[0, 0, 0], [1, 0, 0]]
        radii = [0.5, 0.5]
        _X, _P, _R, G, n = _make_state(positions, radii, num_genes=1)

        feats = build_node_features(G, particle_count=n)

        assert feats.shape == (2, 1)


class TestBuildEdgeFeatures:
    def test_shape(self):
        positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        radii = [0.8, 0.8, 0.8]
        X, P, R, _G, n = _make_state(positions, radii)

        edge_index, _ne = build_edge_index(X, R, particle_count=n)
        edge_feats = build_edge_features(X, P, edge_index, particle_count=n)

        assert edge_feats.shape[0] == edge_index.shape[1]
        assert edge_feats.shape[1] == 2  # dist, angle

    def test_parallel_polarities_zero_angle(self):
        """Same polarity direction -> angle = 0."""
        positions = [[0, 0, 0], [0.5, 0, 0]]
        radii = [0.5, 0.5]
        polarities = [[0, 0, 1], [0, 0, 1]]
        X, P, R, _G, n = _make_state(positions, radii, polarities=polarities)

        edge_index, _ne = build_edge_index(X, R, particle_count=n)
        edge_feats = build_edge_features(X, P, edge_index, particle_count=n)

        # angle column (index 1) should be ~0
        assert float(jnp.abs(edge_feats[:, 1]).max()) == pytest.approx(0.0, abs=1e-5)

    def test_antiparallel_polarities_pi_angle(self):
        """Opposite polarity direction -> angle = pi."""
        positions = [[0, 0, 0], [0.5, 0, 0]]
        radii = [0.5, 0.5]
        polarities = [[0, 0, 1], [0, 0, -1]]
        X, P, R, _G, n = _make_state(positions, radii, polarities=polarities)

        edge_index, _ne = build_edge_index(X, R, particle_count=n)
        edge_feats = build_edge_features(X, P, edge_index, particle_count=n)

        assert float(edge_feats[:, 1].max()) == pytest.approx(np.pi, abs=1e-5)

    def test_distance_symmetric(self):
        """Distance should be the same for i->j and j->i."""
        positions = [[0, 0, 0], [1, 2, 3]]
        radii = [3.0, 3.0]
        X, P, R, _G, n = _make_state(positions, radii)

        edge_index, _ne = build_edge_index(X, R, particle_count=n)
        edge_feats = build_edge_features(X, P, edge_index, particle_count=n)

        # Two edges: 0->1 and 1->0, distance should match
        assert float(edge_feats[0, 0]) == pytest.approx(float(edge_feats[1, 0]), abs=1e-5)
        # Check actual distance value
        expected_dist = (1**2 + 2**2 + 3**2) ** 0.5
        assert float(edge_feats[0, 0]) == pytest.approx(expected_dist, abs=1e-4)


class TestSparseMatchesDense:
    def test_same_edges(self):
        positions = [[0, 0, 0], [0.9, 0, 0], [10, 0, 0], [0.25, 0.4, 0]]
        radii = [0.5, 0.5, 0.5, 0.5]
        X, _P, R, _G, n = _make_state(positions, radii)

        edge_index, _ne = build_edge_index(X, R, particle_count=n)

        edges = set(map(tuple, np.asarray(edge_index.T).tolist()))
        assert (0, 1) in edges
        assert (1, 0) in edges
        for e in edges:
            assert 2 not in e


class TestBuildGraph:
    def test_convenience_wrapper(self):
        positions = [[0, 0, 0], [0.5, 0, 0], [0.25, 0.4, 0]]
        radii = [0.5, 0.5, 0.5]
        X, P, R, G, n = _make_state(positions, radii, num_genes=5)

        node_feats, edge_index, edge_feats, num_edges = build_graph(
            X,
            P,
            R,
            particle_count=n,
            G=G,
        )

        assert node_feats.shape == (3, 5)  # genes only
        assert edge_index.shape[0] == 2
        assert edge_feats.shape[1] == 2  # dist, angle
        assert edge_feats.shape[0] == edge_index.shape[1]
        assert num_edges == edge_index.shape[1]


class TestEdgePadding:
    def test_padded_shape(self):
        """max_edges pads edge_index and edge_features to a fixed size."""
        positions = [[0, 0, 0], [0.5, 0, 0]]
        radii = [0.5, 0.5]
        X, P, R, G, n = _make_state(positions, radii)

        node_feats, edge_index, edge_feats, num_edges = build_graph(
            X,
            P,
            R,
            particle_count=n,
            G=G,
            max_edges=100,
        )

        assert edge_index.shape == (2, 100)
        assert edge_feats.shape == (100, 2)
        assert num_edges == 2  # only 2 real edges
        # Real edges are 0->1 and 1->0; rest are padding (0, 0)
        assert node_feats.shape == (2, 2)

    def test_padded_preserves_real_edges(self):
        """Real edges in padded output match unpadded output."""
        positions = [[0, 0, 0], [0.9, 0, 0], [10, 0, 0]]
        radii = [0.5, 0.5, 0.5]
        X, P, R, G, n = _make_state(positions, radii)

        _, ei_unpadded, ef_unpadded, ne_unpadded = build_graph(
            X,
            P,
            R,
            particle_count=n,
            G=G,
        )
        _, ei_padded, ef_padded, ne_padded = build_graph(
            X,
            P,
            R,
            particle_count=n,
            G=G,
            max_edges=50,
        )

        assert ne_unpadded == ne_padded
        actual_e = ne_unpadded
        np.testing.assert_array_equal(
            np.asarray(ei_padded[:, :actual_e]),
            np.asarray(ei_unpadded),
        )
        np.testing.assert_allclose(
            np.asarray(ef_padded[:actual_e]),
            np.asarray(ef_unpadded),
            atol=1e-6,
        )

    def test_padded_raises_on_overflow(self):
        """max_edges too small raises ValueError."""
        positions = [[0, 0, 0], [0.5, 0, 0], [0.25, 0.4, 0]]
        radii = [0.5, 0.5, 0.5]
        X, _P, R, _G, n = _make_state(positions, radii)

        with pytest.raises(ValueError, match="max_edges"):
            build_edge_index(X, R, particle_count=n, max_edges=1)
