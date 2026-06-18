"""Tests for the simplified emulator kernels."""

import numpy as np
import warp as wp

from waxmorph.emulator import (
    diffusion_step_differentiable,
    mech_step_sticky_differentiable,
)

wp.init()

DEVICE = "cpu"

try:
    if wp.is_device_available("cuda"):
        DEVICE = "cuda"
except RuntimeError:
    DEVICE = "cpu"


def _make_cluster(positions, radii, num_molecules=2):
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

    c_init = np.ones((max_p, num_molecules), dtype=np.float32) * 0.5
    c = wp.from_numpy(c_init, dtype=wp.float32, device=DEVICE)

    f_net = wp.zeros(max_p, dtype=wp.vec3f, device=DEVICE)
    lap_c = wp.zeros_like(c)

    return X, R, P, c, f_net, lap_c, n


class TestDifferentiableNeighborPairOverflow:
    def test_mech_step_sticky_differentiable_handles_dense_graph(self):
        """Dense contact graphs should resize pair buffers instead of indexing past them."""
        positions = np.zeros((50, 3), dtype=np.float32)
        radii = np.full(50, 0.5, dtype=np.float32)
        X, R, _P, _c, f_net, _lap_c, n = _make_cluster(positions, radii)

        tape = wp.Tape()
        X_out = mech_step_sticky_differentiable(tape, X, R, n, dt=0.01, f_net=f_net)

        x_after = X_out.numpy()
        assert x_after.shape == (n, 3)
        assert np.isfinite(x_after).all()

    def test_diffusion_step_differentiable_handles_dense_graph(self):
        """Dense contact graphs should resize diffusion pair buffers instead of overflowing."""
        positions = np.zeros((50, 3), dtype=np.float32)
        radii = np.full(50, 0.5, dtype=np.float32)
        X, R, _P, c, _f_net, lap_c, n = _make_cluster(positions, radii, num_molecules=2)

        tape = wp.Tape()
        c_out = diffusion_step_differentiable(
            tape,
            X,
            R,
            c,
            lap_c,
            n,
            D_emu=0.1,
            dt=0.01,
        )

        g_after = c_out.numpy()
        assert g_after.shape == (n, 2)
        assert np.isfinite(g_after).all()
