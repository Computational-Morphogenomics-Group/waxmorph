"""Tests for sample_mesh_sequence (multi-target trajectory loader)."""

import numpy as np
import pytest

import waxmorph.data as data_module
from waxmorph.data import sample_mesh_sequence

MESHES_DIR = "meshes"


def test_sample_mesh_sequence_basic_shape():
    """Loader returns source_pos and a sorted list of frame-tagged targets."""
    data = sample_mesh_sequence(
        f"{MESHES_DIR}/sphere.ply",
        target_specs=[
            (199, f"{MESHES_DIR}/armadillo.ply"),
            (50, f"{MESHES_DIR}/bunny.ply"),
        ],
        n_points=200,
        target_extent=8.0,
        seed=0,
    )

    assert data["source_pos"].shape[1] == 3
    assert data["source_pos"].dtype == np.float32
    assert data["source_extent"] == 8.0
    assert data["target_extent"] == 8.0

    assert len(data["targets"]) == 2
    frames = [t["frame"] for t in data["targets"]]
    assert frames == sorted(frames), "targets must be sorted by frame"
    assert frames == [50, 199]

    for t in data["targets"]:
        assert t["pos"].shape[1] == 3
        assert t["pos"].dtype == np.float32
        # All meshes normalized to same extent -> points within half-extent bbox.
        assert np.abs(t["pos"]).max() <= 8.0


def test_sample_mesh_sequence_uses_connected_radius_spacing(monkeypatch):
    """Every source/target sample should use extent-derived connected spacing."""
    min_dist_calls = []

    def fake_sample_volume(mesh, n_points, pitch=None, min_dist=None, seed=0):
        del mesh, pitch, seed
        min_dist_calls.append(min_dist)
        return np.zeros((n_points, 3), dtype=np.float32)

    monkeypatch.setattr(data_module, "sample_volume", fake_sample_volume)

    data = sample_mesh_sequence(
        f"{MESHES_DIR}/sphere.ply",
        target_specs=[
            (10, f"{MESHES_DIR}/armadillo.ply"),
            (20, f"{MESHES_DIR}/bunny.ply"),
        ],
        n_points=32,
        target_extent=8.0,
        seed=0,
    )

    expected_min_dist_calls = [
        data_module._connected_poisson_min_dist(data["source_radius"]),
        *[data_module._connected_poisson_min_dist(r) for r in data["target_radii"]],
    ]
    assert min_dist_calls == pytest.approx(expected_min_dist_calls)
    assert data["radius"] == pytest.approx(data["source_radius"])
    assert data["R_init"] == pytest.approx(data["source_radius"])


def test_sample_mesh_sequence_sorts_per_target_extents(monkeypatch):
    def fake_sample_volume(mesh, n_points, pitch=None, min_dist=None, seed=0):
        del mesh, pitch, min_dist, seed
        return np.zeros((n_points, 3), dtype=np.float32)

    monkeypatch.setattr(data_module, "sample_volume", fake_sample_volume)

    data = sample_mesh_sequence(
        f"{MESHES_DIR}/sphere.ply",
        target_specs=[
            (20, f"{MESHES_DIR}/armadillo.ply"),
            (10, f"{MESHES_DIR}/bunny.ply"),
        ],
        n_points=32,
        target_extent=[9.0, 7.0],
        source_extent=6.0,
        seed=0,
    )

    assert data["source_extent"] == pytest.approx(6.0)
    assert data["target_extent"] == pytest.approx([7.0, 9.0])
    assert data["target_extents"] == pytest.approx([7.0, 9.0])
    assert [target["frame"] for target in data["targets"]] == [10, 20]
    assert [target["extent"] for target in data["targets"]] == pytest.approx([7.0, 9.0])
    assert data["target_radii"] == pytest.approx([target["radius"] for target in data["targets"]])
    assert data["target_radii"][0] != pytest.approx(data["target_radii"][1])


def test_sample_mesh_sequence_requires_source_extent_with_extent_sequence():
    with pytest.raises(ValueError, match="source_extent is required"):
        sample_mesh_sequence(
            f"{MESHES_DIR}/sphere.ply",
            target_specs=[
                (10, f"{MESHES_DIR}/armadillo.ply"),
                (20, f"{MESHES_DIR}/bunny.ply"),
            ],
            n_points=32,
            target_extent=[9.0, 7.0],
            seed=0,
        )


def test_sample_mesh_sequence_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        sample_mesh_sequence(
            f"{MESHES_DIR}/sphere.ply",
            target_specs=[],
            n_points=100,
        )


def test_sample_mesh_sequence_rejects_duplicate_frames():
    with pytest.raises(ValueError, match="Duplicate target frame"):
        sample_mesh_sequence(
            f"{MESHES_DIR}/sphere.ply",
            target_specs=[
                (10, f"{MESHES_DIR}/armadillo.ply"),
                (10, f"{MESHES_DIR}/bunny.ply"),
            ],
            n_points=100,
        )


def test_sample_mesh_sequence_rejects_negative_frame():
    with pytest.raises(ValueError, match="non-negative"):
        sample_mesh_sequence(
            f"{MESHES_DIR}/sphere.ply",
            target_specs=[(-1, f"{MESHES_DIR}/armadillo.ply")],
            n_points=100,
        )
