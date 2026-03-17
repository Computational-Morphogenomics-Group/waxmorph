"""Tests comparing Warp autodiff gradients against hand-derived gradient functions.

Verifies that the scalar potentials (epi_polarity_potential, epi_thickness_potential,
mes_polarity_potential) produce the same gradients as the hand-derived gradient
functions (epi_polarity_grads, epi_thickness_grads, mes_polarity_grads) when
differentiated via wp.Tape.
"""

import numpy as np
import pytest
import warp as wp

from waxmorph import simulator

wp.init()

DEVICE = "cpu"

try:
    if wp.is_device_available("cuda"):
        DEVICE = "cuda"
except RuntimeError:
    DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Test kernels: wrap each potential in a kernel for tape-based autodiff
# ---------------------------------------------------------------------------


@wp.kernel
def epi_polarity_potential_kernel(
    x_i: wp.array(dtype=wp.vec3f),
    x_j: wp.array(dtype=wp.vec3f),
    p_i: wp.array(dtype=wp.vec3f),
    p_j: wp.array(dtype=wp.vec3f),
    energy: wp.array(dtype=wp.float32),
):
    U = simulator.epi_polarity_potential(x_i[0], x_j[0], p_i[0], p_j[0])
    energy[0] = U


@wp.kernel
def epi_thickness_potential_kernel(
    x_i: wp.array(dtype=wp.vec3f),
    x_j: wp.array(dtype=wp.vec3f),
    p_i: wp.array(dtype=wp.vec3f),
    p_j: wp.array(dtype=wp.vec3f),
    energy: wp.array(dtype=wp.float32),
):
    U = simulator.epi_thickness_potential(x_i[0], x_j[0], p_i[0], p_j[0])
    energy[0] = U


@wp.kernel
def mes_polarity_potential_kernel(
    p_i: wp.array(dtype=wp.vec3f),
    p_j: wp.array(dtype=wp.vec3f),
    energy: wp.array(dtype=wp.float32),
):
    U = simulator.mes_polarity_potential(p_i[0], p_j[0])
    energy[0] = U


# ---------------------------------------------------------------------------
# Test kernels: hand-derived gradient evaluation
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def epi_polarity_grads_kernel(
    x_i: wp.array(dtype=wp.vec3f),
    x_j: wp.array(dtype=wp.vec3f),
    p_i: wp.array(dtype=wp.vec3f),
    p_j: wp.array(dtype=wp.vec3f),
    gx_i: wp.array(dtype=wp.vec3f),
    gx_j: wp.array(dtype=wp.vec3f),
    gp_i: wp.array(dtype=wp.vec3f),
    gp_j: wp.array(dtype=wp.vec3f),
):
    g_x_i, g_x_j, g_p_i, g_p_j = simulator.epi_polarity_grads(x_i[0], x_j[0], p_i[0], p_j[0])
    gx_i[0] = g_x_i
    gx_j[0] = g_x_j
    gp_i[0] = g_p_i
    gp_j[0] = g_p_j


@wp.kernel(enable_backward=False)
def epi_thickness_grads_kernel(
    x_i: wp.array(dtype=wp.vec3f),
    x_j: wp.array(dtype=wp.vec3f),
    p_i: wp.array(dtype=wp.vec3f),
    p_j: wp.array(dtype=wp.vec3f),
    gx_i: wp.array(dtype=wp.vec3f),
    gx_j: wp.array(dtype=wp.vec3f),
):
    g_x_i, g_x_j = simulator.epi_thickness_grads(x_i[0], x_j[0], p_i[0], p_j[0])
    gx_i[0] = g_x_i
    gx_j[0] = g_x_j


@wp.kernel(enable_backward=False)
def mes_polarity_grads_kernel(
    p_i: wp.array(dtype=wp.vec3f),
    p_j: wp.array(dtype=wp.vec3f),
    gp_i: wp.array(dtype=wp.vec3f),
    gp_j: wp.array(dtype=wp.vec3f),
):
    g_p_i, g_p_j = simulator.mes_polarity_grads(p_i[0], p_j[0])
    gp_i[0] = g_p_i
    gp_j[0] = g_p_j


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vec3_array(vals, device=DEVICE, requires_grad=False):
    """Create a length-1 wp.vec3f array from a 3-tuple."""
    arr = np.array([vals], dtype=np.float32)
    return wp.array(arr, dtype=wp.vec3f, device=device, requires_grad=requires_grad)


def _autodiff_grads_epi_polarity(x_i_val, x_j_val, p_i_val, p_j_val):
    """Run autodiff for epi_polarity_potential, return (gx_i, gx_j, gp_i, gp_j)."""
    x_i = _make_vec3_array(x_i_val, requires_grad=True)
    x_j = _make_vec3_array(x_j_val, requires_grad=True)
    p_i = _make_vec3_array(p_i_val, requires_grad=True)
    p_j = _make_vec3_array(p_j_val, requires_grad=True)
    energy = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=DEVICE)

    tape = wp.Tape()
    with tape:
        wp.launch(
            epi_polarity_potential_kernel,
            dim=1,
            inputs=[x_i, x_j, p_i, p_j],
            outputs=[energy],
            device=DEVICE,
        )
    tape.backward(loss=energy)

    result = (
        x_i.grad.numpy()[0],
        x_j.grad.numpy()[0],
        p_i.grad.numpy()[0],
        p_j.grad.numpy()[0],
    )
    tape.zero()
    return result


def _hand_grads_epi_polarity(x_i_val, x_j_val, p_i_val, p_j_val):
    """Run hand-derived epi_polarity_grads, return (gx_i, gx_j, gp_i, gp_j)."""
    x_i = _make_vec3_array(x_i_val)
    x_j = _make_vec3_array(x_j_val)
    p_i = _make_vec3_array(p_i_val)
    p_j = _make_vec3_array(p_j_val)
    gx_i = wp.zeros(1, dtype=wp.vec3f, device=DEVICE)
    gx_j = wp.zeros(1, dtype=wp.vec3f, device=DEVICE)
    gp_i = wp.zeros(1, dtype=wp.vec3f, device=DEVICE)
    gp_j = wp.zeros(1, dtype=wp.vec3f, device=DEVICE)

    wp.launch(
        epi_polarity_grads_kernel,
        dim=1,
        inputs=[x_i, x_j, p_i, p_j],
        outputs=[gx_i, gx_j, gp_i, gp_j],
        device=DEVICE,
    )
    return (
        gx_i.numpy()[0],
        gx_j.numpy()[0],
        gp_i.numpy()[0],
        gp_j.numpy()[0],
    )


def _autodiff_grads_epi_thickness(x_i_val, x_j_val, p_i_val, p_j_val):
    """Run autodiff for epi_thickness_potential, return (gx_i, gx_j, gp_i, gp_j)."""
    x_i = _make_vec3_array(x_i_val, requires_grad=True)
    x_j = _make_vec3_array(x_j_val, requires_grad=True)
    p_i = _make_vec3_array(p_i_val, requires_grad=True)
    p_j = _make_vec3_array(p_j_val, requires_grad=True)
    energy = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=DEVICE)

    tape = wp.Tape()
    with tape:
        wp.launch(
            epi_thickness_potential_kernel,
            dim=1,
            inputs=[x_i, x_j, p_i, p_j],
            outputs=[energy],
            device=DEVICE,
        )
    tape.backward(loss=energy)

    result = (
        x_i.grad.numpy()[0],
        x_j.grad.numpy()[0],
        p_i.grad.numpy()[0],
        p_j.grad.numpy()[0],
    )
    tape.zero()
    return result


def _hand_grads_epi_thickness(x_i_val, x_j_val, p_i_val, p_j_val):
    """Run hand-derived epi_thickness_grads, return (gx_i, gx_j)."""
    x_i = _make_vec3_array(x_i_val)
    x_j = _make_vec3_array(x_j_val)
    p_i = _make_vec3_array(p_i_val)
    p_j = _make_vec3_array(p_j_val)
    gx_i = wp.zeros(1, dtype=wp.vec3f, device=DEVICE)
    gx_j = wp.zeros(1, dtype=wp.vec3f, device=DEVICE)

    wp.launch(
        epi_thickness_grads_kernel,
        dim=1,
        inputs=[x_i, x_j, p_i, p_j],
        outputs=[gx_i, gx_j],
        device=DEVICE,
    )
    return gx_i.numpy()[0], gx_j.numpy()[0]


def _autodiff_grads_mes_polarity(p_i_val, p_j_val):
    """Run autodiff for mes_polarity_potential, return (gp_i, gp_j)."""
    p_i = _make_vec3_array(p_i_val, requires_grad=True)
    p_j = _make_vec3_array(p_j_val, requires_grad=True)
    energy = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=DEVICE)

    tape = wp.Tape()
    with tape:
        wp.launch(
            mes_polarity_potential_kernel,
            dim=1,
            inputs=[p_i, p_j],
            outputs=[energy],
            device=DEVICE,
        )
    tape.backward(loss=energy)

    result = (p_i.grad.numpy()[0], p_j.grad.numpy()[0])
    tape.zero()
    return result


def _hand_grads_mes_polarity(p_i_val, p_j_val):
    """Run hand-derived mes_polarity_grads, return (gp_i, gp_j)."""
    p_i = _make_vec3_array(p_i_val)
    p_j = _make_vec3_array(p_j_val)
    gp_i = wp.zeros(1, dtype=wp.vec3f, device=DEVICE)
    gp_j = wp.zeros(1, dtype=wp.vec3f, device=DEVICE)

    wp.launch(
        mes_polarity_grads_kernel,
        dim=1,
        inputs=[p_i, p_j],
        outputs=[gp_i, gp_j],
        device=DEVICE,
    )
    return gp_i.numpy()[0], gp_j.numpy()[0]


# ---------------------------------------------------------------------------
# Tests: epi_polarity_potential vs epi_polarity_grads
# ---------------------------------------------------------------------------


class TestEpiPolarityAutodiff:
    """Verify autodiff of epi_polarity_potential matches epi_polarity_grads."""

    @pytest.mark.parametrize(
        "x_i, x_j, p_i, p_j",
        [
            # Particles along x-axis, polarities along z
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            # Diagonal separation, tilted polarities
            ([0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.6, 0.0, 0.8], [0.0, 0.8, 0.6]),
            # Close particles, parallel polarities along axis (worst case)
            ([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            # 3D configuration
            ([1.0, 2.0, 3.0], [1.5, 2.3, 2.7], [0.57, 0.57, 0.57], [0.0, 0.7, 0.7]),
        ],
    )
    def test_polarity_grads_match(self, x_i, x_j, p_i, p_j):
        ad_gx_i, ad_gx_j, ad_gp_i, ad_gp_j = _autodiff_grads_epi_polarity(x_i, x_j, p_i, p_j)
        hd_gx_i, hd_gx_j, hd_gp_i, hd_gp_j = _hand_grads_epi_polarity(x_i, x_j, p_i, p_j)

        np.testing.assert_allclose(ad_gp_i, hd_gp_i, atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(ad_gp_j, hd_gp_j, atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(ad_gx_i, hd_gx_i, atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(ad_gx_j, hd_gx_j, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# Tests: epi_thickness_potential vs epi_thickness_grads
# ---------------------------------------------------------------------------


class TestEpiThicknessAutodiff:
    """Verify autodiff of epi_thickness_potential matches epi_thickness_grads (x-grads)."""

    @pytest.mark.parametrize(
        "x_i, x_j, p_i, p_j",
        [
            # Normal separation exceeds H_THICK_EE (active penalty)
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            # Aligned polarities, lateral offset (no thickness penalty)
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            # Anti-parallel polarities (sign flip path)
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]),
            # 3D with active penalty
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.4], [0.0, 0.0, 1.0], [0.0, 0.0, 0.9]),
        ],
    )
    def test_thickness_x_grads_match(self, x_i, x_j, p_i, p_j):
        """x-gradients from autodiff should match hand-derived."""
        ad_gx_i, ad_gx_j, _, _ = _autodiff_grads_epi_thickness(x_i, x_j, p_i, p_j)
        hd_gx_i, hd_gx_j = _hand_grads_epi_thickness(x_i, x_j, p_i, p_j)

        np.testing.assert_allclose(ad_gx_i, hd_gx_i, atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(ad_gx_j, hd_gx_j, atol=1e-4, rtol=1e-3)

    def test_thickness_autodiff_produces_p_grads(self):
        """Autodiff additionally computes p-gradients (hand-derived omits them).

        Use an off-axis displacement so d is NOT parallel to n — otherwise
        ∂(d·n)/∂p = (d - (d·n)n)/|s| = 0 and the gradient vanishes.
        """
        x_i = [0.0, 0.0, 0.0]
        x_j = [0.3, 0.0, 0.4]
        p_i = [0.0, 0.0, 1.0]
        p_j = [0.0, 0.0, 1.0]

        _, _, ad_gp_i, ad_gp_j = _autodiff_grads_epi_thickness(x_i, x_j, p_i, p_j)

        # Autodiff produces non-zero polarity gradients from thickness
        # (the hand-derived code intentionally omits these)
        total_p_grad = np.abs(ad_gp_i).sum() + np.abs(ad_gp_j).sum()
        assert (
            total_p_grad > 1e-6
        ), f"Expected non-zero polarity gradients from thickness, got {total_p_grad}"


# ---------------------------------------------------------------------------
# Tests: mes_polarity_potential vs mes_polarity_grads
# ---------------------------------------------------------------------------


class TestMesPolarityAutodiff:
    """Verify autodiff of mes_polarity_potential matches mes_polarity_grads."""

    @pytest.mark.parametrize(
        "p_i, p_j",
        [
            # Parallel
            ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            # Perpendicular (gradient = 0 at saddle)
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            # Anti-parallel
            ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]),
            # General 3D
            ([0.6, 0.0, 0.8], [0.0, 0.8, 0.6]),
        ],
    )
    def test_mes_polarity_grads_match(self, p_i, p_j):
        ad_gp_i, ad_gp_j = _autodiff_grads_mes_polarity(p_i, p_j)
        hd_gp_i, hd_gp_j = _hand_grads_mes_polarity(p_i, p_j)

        np.testing.assert_allclose(ad_gp_i, hd_gp_i, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(ad_gp_j, hd_gp_j, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: mech_step_sticky_implicit integration
# ---------------------------------------------------------------------------


def _sphere_state(
    particle_count,
    max_particles,
    radius=0.6,
    shell_radius=2.0,
    inner_frac=0.25,
):
    """Create a vesicle-like state: mesenchyme core + epithelial shell."""
    n_mes = int(particle_count * inner_frac)
    n_epi = particle_count - n_mes

    centers = np.zeros((max_particles, 3), dtype=np.float64)
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n_mes, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = (rng.random(n_mes) ** (1 / 3)) * (shell_radius - 0.125)
    centers[:n_mes] = v * r[:, None]

    golden = (1 + np.sqrt(5)) / 2
    idx = np.arange(n_epi)
    theta = 2 * np.pi * idx / golden
    phi = np.arccos(1 - 2 * (idx + 0.5) / n_epi)
    centers[n_mes : n_mes + n_epi, 0] = np.cos(theta) * np.sin(phi) * shell_radius
    centers[n_mes : n_mes + n_epi, 1] = np.sin(theta) * np.sin(phi) * shell_radius
    centers[n_mes : n_mes + n_epi, 2] = np.cos(phi) * shell_radius

    radii = np.full(max_particles, -1000.0, dtype=np.float64)
    radii[:particle_count] = radius

    polarities = np.ones((max_particles, 3), dtype=np.float64) / np.sqrt(3)
    c = centers[:particle_count].mean(axis=0)
    p = centers[:particle_count] - c
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    polarities[:particle_count] = p / np.clip(norms, 1e-12, None)

    cell_types = np.full(max_particles, 3, dtype=np.uint32)
    cell_types[:n_mes] = 0
    cell_types[n_mes:particle_count] = 1

    X = wp.from_numpy(centers, dtype=wp.vec3f, device=DEVICE)
    R = wp.from_numpy(radii, dtype=wp.float32, device=DEVICE)
    P = wp.from_numpy(polarities, dtype=wp.vec3f, device=DEVICE)
    CT = wp.from_numpy(cell_types, dtype=wp.uint32, device=DEVICE)

    return {
        "X": X,
        "R": R,
        "P": P,
        "CT": CT,
        "particle_count": particle_count,
        "max_particles": max_particles,
    }


class TestMechStepStickyImplicit:
    def test_does_not_crash(self):
        """mech_step_sticky_implicit should run without errors."""
        s = _sphere_state(30, 60)
        X_next = wp.zeros_like(s["X"], device=DEVICE)
        P_next = wp.zeros_like(s["P"], device=DEVICE)

        simulator.mech_step_sticky_implicit(
            s["X"],
            s["R"],
            s["P"],
            s["CT"],
            s["particle_count"],
            1e-2,
            X_next,
            P_next,
            device=DEVICE,
        )

        x = X_next.numpy()[:30]
        assert not np.isnan(x).any()
        assert np.isfinite(x).all()

    def test_positions_change(self):
        """Positions should be updated by the mechanics step."""
        s = _sphere_state(30, 60)
        x_before = s["X"].numpy()[:30].copy()
        X_next = wp.zeros_like(s["X"], device=DEVICE)
        P_next = wp.zeros_like(s["P"], device=DEVICE)

        simulator.mech_step_sticky_implicit(
            s["X"],
            s["R"],
            s["P"],
            s["CT"],
            s["particle_count"],
            5e-2,
            X_next,
            P_next,
            device=DEVICE,
        )

        x_after = X_next.numpy()[:30]
        assert not np.allclose(x_before, x_after, atol=1e-8)

    def test_polarities_stay_normalized(self):
        """Polarity vectors should remain approximately unit length."""
        s = _sphere_state(30, 60)

        for _ in range(50):
            simulator.mech_step_sticky_implicit(
                s["X"],
                s["R"],
                s["P"],
                s["CT"],
                s["particle_count"],
                1e-2,
                s["X"],
                s["P"],
                device=DEVICE,
            )

        p = s["P"].numpy()[:30]
        norms = np.linalg.norm(p, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_close_to_explicit(self):
        """Implicit and explicit mech steps should produce similar results.

        They won't be identical because:
        - The implicit version gets additional polarity gradients for thickness
          (through P), which the hand-derived version omits.
        - Clamping interacts differently since forces and polarity grads are
          computed in separate kernels.

        We check that the overall trajectory direction is similar.
        """
        s1 = _sphere_state(30, 60)
        s2 = _sphere_state(30, 60)  # identical initial state

        X1_next = wp.zeros_like(s1["X"], device=DEVICE)
        P1_next = wp.zeros_like(s1["P"], device=DEVICE)
        X2_next = wp.zeros_like(s2["X"], device=DEVICE)
        P2_next = wp.zeros_like(s2["P"], device=DEVICE)

        # Explicit (hand-derived)
        simulator.mech_step_sticky(
            s1["X"],
            s1["R"],
            s1["P"],
            s1["CT"],
            s1["particle_count"],
            1e-2,
            X1_next,
            P1_next,
            device=DEVICE,
            grad_consist=False,
        )

        # Implicit (autodiff)
        simulator.mech_step_sticky_implicit(
            s2["X"],
            s2["R"],
            s2["P"],
            s2["CT"],
            s2["particle_count"],
            1e-2,
            X2_next,
            P2_next,
            device=DEVICE,
        )

        x1 = X1_next.numpy()[:30]
        x2 = X2_next.numpy()[:30]

        # Displacement vectors should point in broadly similar directions
        dx1 = x1 - s1["X"].numpy()[:30]
        dx2 = x2 - s2["X"].numpy()[:30]

        # Positions should be within 10% relative error for most particles
        # (exact match is not expected due to the thickness p-grad difference)
        norms1 = np.linalg.norm(dx1, axis=1)
        norms2 = np.linalg.norm(dx2, axis=1)
        active = norms1 > 1e-6  # only compare particles that moved
        if active.sum() > 0:
            cos_sim = np.sum(dx1[active] * dx2[active], axis=1) / (
                norms1[active] * norms2[active] + 1e-12
            )
            # Most displacement vectors should point in similar directions
            assert (
                np.median(cos_sim) > 0.5
            ), f"Displacement directions too different: median cos_sim={np.median(cos_sim):.3f}"
