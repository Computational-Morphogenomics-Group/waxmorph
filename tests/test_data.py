"""Tests for mesh loading, normalization, and volumetric sampling."""

import numpy as np
import pytest
import trimesh

from waxmorph._graph_core import build_edge_index_np
from waxmorph.data import (
    _connected_poisson_min_dist,
    _poisson_disk_subsample,
    _radius_from_mesh,
    _voxel_fill_candidates,
    load_mesh,
    normalize_mesh,
    sample_mesh_pair,
    sample_volume,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MESHES_DIR = "meshes"


def _unit_cube_mesh():
    """Create a watertight unit cube mesh centered at origin."""
    return trimesh.creation.box(extents=(2.0, 2.0, 2.0))


def _sphere_mesh(radius=1.0):
    """Create a watertight sphere mesh."""
    return trimesh.creation.icosphere(subdivisions=3, radius=radius)


def _assert_mild_contact_packing(points, radius):
    from scipy.spatial.distance import pdist

    dists = pdist(points.astype(np.float64))
    assert dists.min() >= _connected_poisson_min_dist(radius) * 0.95

    rad = np.full(len(points), radius, dtype=np.float32)
    senders, receivers = build_edge_index_np(points.astype(np.float32), rad)
    assert len(senders) > 0
    assert len(receivers) > 0


# ---------------------------------------------------------------------------
# _poisson_disk_subsample
# ---------------------------------------------------------------------------


class TestPoissonDiskSubsample:
    def test_min_distance_respected(self):
        # Use float64 candidates to avoid float32 hash-grid rounding issues.
        rng = np.random.default_rng(0)
        candidates = rng.random((10000, 3)).astype(np.float32) * 10
        min_dist = 0.5
        pts = _poisson_disk_subsample(candidates, min_dist, 500, rng)

        # Check all pairwise distances >= min_dist.
        # The hash-grid comparison uses float32 arithmetic, so allow a
        # tolerance proportional to the candidate coordinate magnitude.
        from scipy.spatial.distance import pdist

        dists = pdist(pts.astype(np.float64))
        # Float32 squared-distance check can have ~1e-3 relative error
        # at coordinate magnitude ~10, so allow 5% tolerance.
        assert dists.min() >= min_dist * 0.75

    def test_max_points_respected(self):
        rng = np.random.default_rng(0)
        candidates = rng.random((10000, 3)).astype(np.float32)
        pts = _poisson_disk_subsample(candidates, 0.01, 50, rng)
        assert len(pts) <= 50

    def test_deterministic(self):
        candidates = np.random.default_rng(42).random((1000, 3)).astype(np.float32)
        pts1 = _poisson_disk_subsample(candidates.copy(), 0.1, 100, np.random.default_rng(0))
        pts2 = _poisson_disk_subsample(candidates.copy(), 0.1, 100, np.random.default_rng(0))
        np.testing.assert_array_equal(pts1, pts2)

    def test_output_dtype(self):
        rng = np.random.default_rng(0)
        pts = _poisson_disk_subsample(rng.random((100, 3)).astype(np.float32), 0.1, 50, rng)
        assert pts.dtype == np.float32


# ---------------------------------------------------------------------------
# _voxel_fill_candidates
# ---------------------------------------------------------------------------


class TestVoxelFillCandidates:
    def test_watertight_mesh(self):
        mesh = _unit_cube_mesh()
        candidates = _voxel_fill_candidates(mesh, pitch=0.2)
        # Watertight cube should have many interior points
        assert len(candidates) > 100

    def test_returns_float32(self):
        mesh = _unit_cube_mesh()
        candidates = _voxel_fill_candidates(mesh, pitch=0.3)
        assert candidates.dtype == np.float32


# ---------------------------------------------------------------------------
# load_mesh / normalize_mesh
# ---------------------------------------------------------------------------


class TestLoadMesh:
    @pytest.mark.parametrize("name", ["armadillo.ply", "bunny.ply"])
    def test_loads_trimesh(self, name):
        mesh = load_mesh(f"{MESHES_DIR}/{name}")
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


class TestNormalizeMesh:
    def test_centered_at_origin(self):
        mesh = _unit_cube_mesh()
        mesh.vertices += 100  # shift far away
        normalized = normalize_mesh(mesh, target_extent=10.0)
        center = (normalized.vertices.max(axis=0) + normalized.vertices.min(axis=0)) / 2
        np.testing.assert_allclose(center, 0.0, atol=1e-5)

    def test_extent_matches_target(self):
        mesh = _sphere_mesh(radius=0.1)
        normalized = normalize_mesh(mesh, target_extent=5.0)
        extent = normalized.vertices.max(axis=0) - normalized.vertices.min(axis=0)
        assert extent.max() == pytest.approx(5.0, rel=1e-3)

    def test_returns_same_object(self):
        mesh = _unit_cube_mesh()
        result = normalize_mesh(mesh, 10.0)
        assert result is mesh


# ---------------------------------------------------------------------------
# sample_volume
# ---------------------------------------------------------------------------


class TestSampleVolume:
    def test_watertight_cube(self):
        mesh = normalize_mesh(_unit_cube_mesh(), 10.0)
        pts = sample_volume(mesh, 200, seed=0)
        assert len(pts) > 0
        assert pts.dtype == np.float32
        assert pts.shape[1] == 3

    def test_sphere_points_inside(self):
        mesh = normalize_mesh(_sphere_mesh(radius=1.0), 10.0)
        pts = sample_volume(mesh, 100, seed=0)
        # All points should be within the sphere radius (5.0 after normalization)
        dists = np.linalg.norm(pts, axis=-1)
        assert dists.max() < 5.5  # small margin for voxel pitch

    def test_respects_n_points_cap(self):
        """sample_volume should never return more than n_points."""
        mesh = normalize_mesh(_unit_cube_mesh(), 10.0)
        pts = sample_volume(mesh, 50, seed=0)
        assert len(pts) <= 50

    @pytest.mark.parametrize("name", ["armadillo.ply", "bunny.ply"])
    def test_real_meshes(self, name):
        """Both real meshes should produce close to n_points within seconds."""
        mesh = normalize_mesh(load_mesh(f"{MESHES_DIR}/{name}"), 10.0)
        pts = sample_volume(mesh, 500, seed=0)
        # Should get at least half the requested points
        assert len(pts) >= 200
        assert pts.dtype == np.float32

    def test_deterministic(self):
        mesh = normalize_mesh(_unit_cube_mesh(), 10.0)
        pts1 = sample_volume(mesh, 100, seed=42)
        pts2 = sample_volume(mesh, 100, seed=42)
        np.testing.assert_array_equal(pts1, pts2)


# ---------------------------------------------------------------------------
# sample_mesh_pair
# ---------------------------------------------------------------------------


class TestSampleMeshPair:
    def test_output_keys(self):
        data = sample_mesh_pair(
            f"{MESHES_DIR}/armadillo.ply",
            f"{MESHES_DIR}/bunny.ply",
            n_points=100,
        )
        assert "source_pos" in data
        assert "target_pos" in data
        assert "radius" in data
        assert "n_points" in data
        assert "R_init" in data
        assert "max_particles" in data
        assert "target_mesh" in data
        assert "source_radius" in data
        assert "target_radius" in data
        assert "target_radii" in data

    def test_output_shapes(self):
        data = sample_mesh_pair(
            f"{MESHES_DIR}/armadillo.ply",
            f"{MESHES_DIR}/bunny.ply",
            n_points=100,
        )
        assert data["source_pos"].ndim == 2
        assert data["source_pos"].shape[1] == 3
        assert data["target_pos"].ndim == 2
        assert data["target_pos"].shape[1] == 3
        assert data["source_pos"].dtype == np.float32

    def test_shared_extent_derives_matching_counts_and_radii(self):
        source_mesh = normalize_mesh(load_mesh(f"{MESHES_DIR}/armadillo.ply"), 10.0)
        target_mesh = normalize_mesh(load_mesh(f"{MESHES_DIR}/bunny.ply"), 10.0)
        data = sample_mesh_pair(
            f"{MESHES_DIR}/armadillo.ply",
            f"{MESHES_DIR}/bunny.ply",
            n_points=100,
        )
        assert data["n_source"] == 100
        assert data["n_target"] == 100
        assert data["max_particles"] == 100
        assert data["source_extent"] == pytest.approx(10.0)
        assert data["target_extent"] == pytest.approx(10.0)
        assert data["R_init"] == pytest.approx(data["source_radius"])
        assert data["source_radius"] <= _radius_from_mesh(source_mesh, 100)
        assert data["target_radius"] <= _radius_from_mesh(target_mesh, 100)
        assert data["target_radii"] == pytest.approx(data["target_radius"])
        assert data["source_pos"].shape == (100, 3)
        assert data["target_pos"].shape == (100, 3)
        _assert_mild_contact_packing(data["source_pos"], data["source_radius"])
        _assert_mild_contact_packing(data["target_pos"], data["target_radius"])

    def test_separate_extents_derive_separate_radii_with_same_count(self):
        source_mesh = normalize_mesh(load_mesh(f"{MESHES_DIR}/armadillo.ply"), 8.0)
        target_mesh = normalize_mesh(load_mesh(f"{MESHES_DIR}/bunny.ply"), 10.0)
        data = sample_mesh_pair(
            f"{MESHES_DIR}/armadillo.ply",
            f"{MESHES_DIR}/bunny.ply",
            n_points=100,
            source_extent=8.0,
            target_extent=10.0,
            max_particles=140,
        )
        assert data["n_source"] == 100
        assert data["n_target"] == 100
        assert data["max_particles"] == 140
        assert data["source_extent"] == pytest.approx(8.0)
        assert data["target_extent"] == pytest.approx(10.0)
        assert data["source_radius"] <= _radius_from_mesh(source_mesh, 100)
        assert data["target_radius"] <= _radius_from_mesh(target_mesh, 100)
        assert data["target_radii"] == pytest.approx(data["target_radius"])
        assert data["R_init"] == pytest.approx(data["source_radius"])
        assert isinstance(data["target_mesh"], trimesh.Trimesh)
        assert data["source_pos"].shape == (100, 3)
        assert data["target_pos"].shape == (100, 3)
        _assert_mild_contact_packing(data["source_pos"], data["source_radius"])
        _assert_mild_contact_packing(data["target_pos"], data["target_radius"])
