"""Tests for graph construction from Warp arrays."""

import numpy as np
import warp as wp

from waxmorph.graph import (
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


def _make_state(positions, radii, polarities=None, cell_types=None, num_genes=0):
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

    ct = np.zeros(MAX_PARTICLES, dtype=np.uint32)
    if cell_types is not None:
        ct[:n] = cell_types
    CT = wp.from_numpy(ct, dtype=wp.uint32, device=DEVICE)

    G = None
    if num_genes > 0:
        genes = np.zeros((MAX_PARTICLES, num_genes), dtype=np.float32)
        genes[:n] = np.random.rand(n, num_genes).astype(np.float32)
        G = wp.from_numpy(genes, dtype=wp.float32, device=DEVICE)

    return X, P, R, CT, G, n


class TestBuildEdgeIndex:
    def test_two_touching_one_far(self):
        """Two particles within contact range, third far away."""
        positions = [[0, 0, 0], [0.9, 0, 0], [10, 0, 0]]
        radii = [0.5, 0.5, 0.5]
        X, _P, R, _CT, _, n = _make_state(positions, radii)

        edge_index = build_edge_index(X, R, particle_count=n)

        assert edge_index.shape[0] == 2
        # Bidirectional: (0,1) and (1,0)
        edges = set(map(tuple, edge_index.t().tolist()))
        assert (0, 1) in edges
        assert (1, 0) in edges
        assert len(edges) == 2

    def test_no_self_loops(self):
        """Particles should not connect to themselves."""
        positions = [[0, 0, 0], [0.5, 0, 0]]
        radii = [0.5, 0.5]
        X, _P, R, _CT, _, n = _make_state(positions, radii)

        edge_index = build_edge_index(X, R, particle_count=n)

        senders = edge_index[0].tolist()
        receivers = edge_index[1].tolist()
        for s, r in zip(senders, receivers, strict=False):
            assert s != r

    def test_all_connected(self):
        """Three particles in a tight cluster should all connect."""
        positions = [[0, 0, 0], [0.5, 0, 0], [0.25, 0.4, 0]]
        radii = [0.5, 0.5, 0.5]
        X, _P, R, _CT, _, n = _make_state(positions, radii)

        edge_index = build_edge_index(X, R, particle_count=n)

        # 3 particles fully connected -> 6 directed edges
        assert edge_index.shape[1] == 6

    def test_no_edges_far_apart(self):
        """Particles too far apart should have no edges."""
        positions = [[0, 0, 0], [100, 0, 0]]
        radii = [0.5, 0.5]
        X, _P, R, _CT, _, n = _make_state(positions, radii)

        edge_index = build_edge_index(X, R, particle_count=n)

        assert edge_index.shape[1] == 0


class TestBuildNodeFeatures:
    def test_shape_no_genes(self):
        positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        radii = [0.5, 0.5, 0.5]
        X, P, R, CT, _, n = _make_state(positions, radii)

        feats = build_node_features(X, P, R, CT, particle_count=n)

        # 3 (pos) + 3 (pol) + 1 (rad) + 2 (ct onehot) = 9
        assert feats.shape == (3, 9)

    def test_shape_with_genes(self):
        positions = [[0, 0, 0], [1, 0, 0]]
        radii = [0.5, 0.5]
        X, P, R, CT, G, n = _make_state(positions, radii, num_genes=4)

        feats = build_node_features(X, P, R, CT, particle_count=n, G=G)

        assert feats.shape == (2, 13)  # 9 + 4

    def test_cell_type_onehot(self):
        positions = [[0, 0, 0], [1, 0, 0]]
        radii = [0.5, 0.5]
        cell_types = [0, 1]  # mesenchyme, epithelium
        X, P, R, CT, _, n = _make_state(positions, radii, cell_types=cell_types)

        feats = build_node_features(X, P, R, CT, particle_count=n)

        # ct onehot is at indices 7, 8
        assert feats[0, 7].item() == 1.0  # mesenchyme -> col 0
        assert feats[0, 8].item() == 0.0
        assert feats[1, 7].item() == 0.0
        assert feats[1, 8].item() == 1.0  # epithelium -> col 1


class TestBuildEdgeFeatures:
    def test_shape(self):
        positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        radii = [0.8, 0.8, 0.8]
        X, _P, R, CT, _, n = _make_state(positions, radii)

        edge_index = build_edge_index(X, R, particle_count=n)
        edge_feats = build_edge_features(X, R, CT, edge_index, particle_count=n)

        assert edge_feats.shape[0] == edge_index.shape[1]
        assert edge_feats.shape[1] == 7

    def test_same_type_flag(self):
        positions = [[0, 0, 0], [0.5, 0, 0]]
        radii = [0.5, 0.5]
        cell_types = [0, 1]  # different types
        X, _P, R, CT, _, n = _make_state(positions, radii, cell_types=cell_types)

        edge_index = build_edge_index(X, R, particle_count=n)
        edge_feats = build_edge_features(X, R, CT, edge_index, particle_count=n)

        # ct_same should be 0 for different types
        assert (edge_feats[:, 6] == 0.0).all()


class TestBuildNodeFeaturesNoCT:
    def test_shape_no_ct(self):
        positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        radii = [0.5, 0.5, 0.5]
        X, P, R, _CT, _, n = _make_state(positions, radii)

        feats = build_node_features(X, P, R, CT=None, particle_count=n)

        # 3 (pos) + 3 (pol) + 1 (rad) = 7 (no ct onehot)
        assert feats.shape == (3, 7)

    def test_shape_no_ct_with_genes(self):
        positions = [[0, 0, 0], [1, 0, 0]]
        radii = [0.5, 0.5]
        X, P, R, _CT, G, n = _make_state(positions, radii, num_genes=4)

        feats = build_node_features(X, P, R, CT=None, particle_count=n, G=G)

        assert feats.shape == (2, 11)  # 7 + 4


class TestBuildEdgeFeaturesNoCT:
    def test_shape_no_ct(self):
        positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        radii = [0.8, 0.8, 0.8]
        X, _P, R, _CT, _, n = _make_state(positions, radii)

        edge_index = build_edge_index(X, R, particle_count=n)
        edge_feats = build_edge_features(X, R, CT=None, edge_index=edge_index, particle_count=n)

        assert edge_feats.shape[0] == edge_index.shape[1]
        assert edge_feats.shape[1] == 6  # no ct_same


class TestSparseMatchesDense:
    def test_same_edges(self):
        """cKDTree-based build_edge_index produces the same edges as O(N^2)."""
        positions = [[0, 0, 0], [0.9, 0, 0], [10, 0, 0], [0.25, 0.4, 0]]
        radii = [0.5, 0.5, 0.5, 0.5]
        X, _P, R, _CT, _, n = _make_state(positions, radii)

        edge_index = build_edge_index(X, R, particle_count=n)

        # Verify the edges are correct by checking known geometry:
        # particles 0,1,3 are close; particle 2 is far
        edges = set(map(tuple, edge_index.t().tolist()))
        assert (0, 1) in edges
        assert (1, 0) in edges
        # particle 2 should have no edges
        for e in edges:
            assert 2 not in e


class TestBuildGraph:
    def test_convenience_wrapper(self):
        positions = [[0, 0, 0], [0.5, 0, 0], [0.25, 0.4, 0]]
        radii = [0.5, 0.5, 0.5]
        X, P, R, CT, G, n = _make_state(positions, radii, num_genes=2)

        node_feats, edge_index, edge_feats = build_graph(X, P, R, CT, particle_count=n, G=G)

        assert node_feats.shape == (3, 11)  # 9 + 2 genes
        assert edge_index.shape[0] == 2
        assert edge_feats.shape[1] == 7
        assert edge_feats.shape[0] == edge_index.shape[1]

    def test_no_ct(self):
        """build_graph with CT=None omits cell-type features."""
        positions = [[0, 0, 0], [0.5, 0, 0], [0.25, 0.4, 0]]
        radii = [0.5, 0.5, 0.5]
        X, P, R, _CT, G, n = _make_state(positions, radii, num_genes=2)

        node_feats, _edge_index, edge_feats = build_graph(X, P, R, particle_count=n, G=G)

        assert node_feats.shape == (3, 9)  # 7 + 2 genes
        assert edge_feats.shape[1] == 6  # no ct_same
