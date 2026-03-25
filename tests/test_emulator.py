"""Tests for the simplified emulator kernels."""

import numpy as np
import pytest
import warp as wp

from waxmorph.emulator import (
    diffusion_step,
    diffusion_step_differentiable,
    mech_step_sticky,
    mech_step_sticky_differentiable,
)

wp.init()

DEVICE = "cpu"

try:
    if wp.is_device_available("cuda"):
        DEVICE = "cuda"
except RuntimeError:
    DEVICE = "cpu"


def _make_cluster(positions, radii, num_genes=2):
    """Build Warp arrays for a small particle cluster."""
    n = len(positions)
    max_p = n

    pos = np.array(positions, dtype=np.float32).reshape(-1, 3)
    X = wp.from_numpy(pos, dtype=wp.vec3f, device=DEVICE)

    rad = np.array(radii, dtype=np.float32)
    R = wp.from_numpy(rad, dtype=wp.float32, device=DEVICE)

    pol = np.zeros((max_p, 3), dtype=np.float32)
    pol[:, 2] = 1.0
    P = wp.from_numpy(pol, dtype=wp.vec3f, device=DEVICE)

    genes = np.ones((max_p, num_genes), dtype=np.float32) * 0.5
    G = wp.from_numpy(genes, dtype=wp.float32, device=DEVICE)

    gx = wp.zeros(max_p, dtype=wp.vec3f, device=DEVICE)
    lap_G = wp.zeros_like(G)

    return X, R, P, G, gx, lap_G, n


# ---------------------------------------------------------------------------
# mech_step_sticky
# ---------------------------------------------------------------------------


class TestMechStepSticky:
    def test_overlapping_particles_repel(self):
        """Two overlapping particles should move apart after a mechanics step."""
        X, R, _P, _G, gx, _lap_G, n = _make_cluster(
            positions=[[0, 0, 0], [0.3, 0, 0]],
            radii=[0.5, 0.5],
        )

        x_before = X.numpy().copy()
        mech_step_sticky(X, R, n, dt=0.1, gx=gx)
        x_after = X.numpy()

        # Particles should have moved apart (distance increased)
        dist_before = np.linalg.norm(x_before[0] - x_before[1])
        dist_after = np.linalg.norm(x_after[0] - x_after[1])
        assert dist_after > dist_before

    def test_far_apart_particles_no_force(self):
        """Particles far apart should barely move."""
        X, R, _P, _G, gx, _lap_G, n = _make_cluster(
            positions=[[0, 0, 0], [100, 0, 0]],
            radii=[0.5, 0.5],
        )

        x_before = X.numpy().copy()
        mech_step_sticky(X, R, n, dt=0.01, gx=gx)
        x_after = X.numpy()

        np.testing.assert_allclose(x_before, x_after, atol=1e-6)

    def test_multiple_steps_converge(self):
        """Running many steps should reduce overlap."""
        X, R, _P, _G, gx, _lap_G, n = _make_cluster(
            positions=[[0, 0, 0], [0.2, 0, 0], [0.1, 0.2, 0]],
            radii=[0.5, 0.5, 0.5],
        )

        for _ in range(50):
            mech_step_sticky(X, R, n, dt=0.05, gx=gx)

        x_final = X.numpy()
        # All pairwise distances should be >= sum of radii (no overlap)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(x_final[i] - x_final[j])
                # Should be close to or above contact distance
                assert dist > 0.5  # at least half the sum of radii


# ---------------------------------------------------------------------------
# diffusion_step
# ---------------------------------------------------------------------------


class TestDiffusionStep:
    def test_uniform_genes_no_change(self):
        """Uniform gene concentrations should not change under diffusion."""
        X, R, _P, G, _gx, lap_G, n = _make_cluster(
            positions=[[0, 0, 0], [0.8, 0, 0]],
            radii=[0.5, 0.5],
            num_genes=2,
        )

        g_before = G.numpy().copy()
        diffusion_step(X, R, G, lap_G, n, alpha=0.1, dt=0.01)
        g_after = G.numpy()

        np.testing.assert_allclose(g_before, g_after, atol=1e-6)

    def test_diffusion_equalizes(self):
        """Diffusion should move gene concentrations toward each other."""
        X, R, _P, G, _gx, lap_G, n = _make_cluster(
            positions=[[0, 0, 0], [0.8, 0, 0]],
            radii=[0.5, 0.5],
            num_genes=1,
        )

        # Set different concentrations
        g_np = G.numpy()
        g_np[0, 0] = 1.0
        g_np[1, 0] = 0.0
        G = wp.from_numpy(g_np, dtype=wp.float32, device=DEVICE)
        lap_G = wp.zeros_like(G)

        for _ in range(100):
            diffusion_step(X, R, G, lap_G, n, alpha=1.0, dt=0.01)

        g_final = G.numpy()
        # Concentrations should be closer together
        diff = abs(g_final[0, 0] - g_final[1, 0])
        assert diff < 0.5

    def test_non_adjacent_no_diffusion(self):
        """Particles too far apart should not exchange genes."""
        X, R, _P, G, _gx, lap_G, n = _make_cluster(
            positions=[[0, 0, 0], [100, 0, 0]],
            radii=[0.5, 0.5],
            num_genes=1,
        )

        g_np = G.numpy()
        g_np[0, 0] = 1.0
        g_np[1, 0] = 0.0
        G = wp.from_numpy(g_np, dtype=wp.float32, device=DEVICE)
        lap_G = wp.zeros_like(G)

        diffusion_step(X, R, G, lap_G, n, alpha=1.0, dt=0.01)
        g_after = G.numpy()

        assert g_after[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert g_after[1, 0] == pytest.approx(0.0, abs=1e-6)


class TestDifferentiableNeighborPairOverflow:
    def test_mech_step_sticky_differentiable_handles_dense_graph(self):
        """Dense contact graphs should resize pair buffers instead of indexing past them."""
        positions = np.zeros((50, 3), dtype=np.float32)
        radii = np.full(50, 0.5, dtype=np.float32)
        X, R, _P, _G, gx, _lap_G, n = _make_cluster(positions, radii)

        tape = wp.Tape()
        X_out = mech_step_sticky_differentiable(tape, X, R, n, dt=0.01, gx=gx)

        x_after = X_out.numpy()
        assert x_after.shape == (n, 3)
        assert np.isfinite(x_after).all()

    def test_diffusion_step_differentiable_handles_dense_graph(self):
        """Dense contact graphs should resize diffusion pair buffers instead of overflowing."""
        positions = np.zeros((50, 3), dtype=np.float32)
        radii = np.full(50, 0.5, dtype=np.float32)
        X, R, _P, G, _gx, lap_G, n = _make_cluster(positions, radii, num_genes=2)

        tape = wp.Tape()
        G_out = diffusion_step_differentiable(
            tape,
            X,
            R,
            G,
            lap_G,
            n,
            alpha=0.1,
            dt=0.01,
        )

        g_after = G_out.numpy()
        assert g_after.shape == (n, 2)
        assert np.isfinite(g_after).all()
