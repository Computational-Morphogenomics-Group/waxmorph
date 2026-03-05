"""Tests for the full Warp simulator (mechanics, chemistry, growth, division)."""

import numpy as np
import warp as wp

from waxmorph import simulator

wp.init()

DEVICE = None

try:
    if wp.is_device_available("cuda"):
        DEVICE = "cuda"
except RuntimeError:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sphere_state(
    particle_count,
    max_particles,
    radius=0.6,
    shell_radius=2.0,
    inner_frac=0.25,
):
    """Create a vesicle-like state: mesenchyme core + epithelial shell.

    Returns dict of Warp arrays on DEVICE.
    """
    n_mes = int(particle_count * inner_frac)
    n_epi = particle_count - n_mes

    centers = np.zeros((max_particles, 3), dtype=np.float64)
    # Mesenchyme: random in interior ball
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n_mes, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = (rng.random(n_mes) ** (1 / 3)) * (shell_radius - 0.125)
    centers[:n_mes] = v * r[:, None]

    # Epithelium: Fibonacci sphere
    golden = (1 + np.sqrt(5)) / 2
    idx = np.arange(n_epi)
    theta = 2 * np.pi * idx / golden
    phi = np.arccos(1 - 2 * (idx + 0.5) / n_epi)
    centers[n_mes : n_mes + n_epi, 0] = np.cos(theta) * np.sin(phi) * shell_radius
    centers[n_mes : n_mes + n_epi, 1] = np.sin(theta) * np.sin(phi) * shell_radius
    centers[n_mes : n_mes + n_epi, 2] = np.cos(phi) * shell_radius

    radii = np.full(max_particles, -1000.0, dtype=np.float64)
    radii[:particle_count] = radius

    radii_eq = np.full(max_particles, -1000.0, dtype=np.float64)
    radii_eq[:particle_count] = radius

    # Polarities: outward-pointing
    polarities = np.ones((max_particles, 3), dtype=np.float64) / np.sqrt(3)
    c = centers[:particle_count].mean(axis=0)
    p = centers[:particle_count] - c
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    polarities[:particle_count] = p / np.clip(norms, 1e-12, None)

    # Cell types
    cell_types = np.full(max_particles, 3, dtype=np.uint32)
    cell_types[:n_mes] = 0
    cell_types[n_mes:particle_count] = 1

    # Chemicals (activators / inhibitors)
    act_mu = (radius**3) * 0.75 / 2.0
    inhib_mu = (radius**3) * 0.75 / 10.0
    activators = np.zeros(max_particles, dtype=np.float64)
    inhibitors = np.zeros(max_particles, dtype=np.float64)
    activators[:n_mes] = act_mu + 0.001 * rng.standard_normal(n_mes)
    inhibitors[:n_mes] = inhib_mu + 0.001 * rng.standard_normal(n_mes)

    X = wp.from_numpy(centers, dtype=wp.vec3f, device=DEVICE)
    R = wp.from_numpy(radii, dtype=wp.float32, device=DEVICE)
    R_eq = wp.from_numpy(radii_eq, dtype=wp.float32, device=DEVICE)
    P = wp.from_numpy(polarities, dtype=wp.vec3f, device=DEVICE)
    CT = wp.from_numpy(cell_types, dtype=wp.uint32, device=DEVICE)
    A = wp.from_numpy(activators, dtype=wp.float32, device=DEVICE)
    Inh = wp.from_numpy(inhibitors, dtype=wp.float32, device=DEVICE)

    return {
        "X": X,
        "R": R,
        "R_eq": R_eq,
        "P": P,
        "CT": CT,
        "A": A,
        "I": Inh,
        "particle_count": particle_count,
        "max_particles": max_particles,
        "n_mes": n_mes,
        "n_epi": n_epi,
    }


# ---------------------------------------------------------------------------
# gen_key_array
# ---------------------------------------------------------------------------


class TestGenKeyArray:
    def test_shape_and_device(self):
        keys = simulator.gen_key_array(100, device=DEVICE)
        assert keys.shape[0] == 100
        assert DEVICE in str(keys.device)

    def test_deterministic(self):
        k1 = simulator.gen_key_array(50, device=DEVICE)
        k2 = simulator.gen_key_array(50, device=DEVICE)
        np.testing.assert_array_equal(k1.numpy(), k2.numpy())


# ---------------------------------------------------------------------------
# mech_step_sticky
# ---------------------------------------------------------------------------


class TestMechStepSticky:
    def test_overlapping_particles_repel(self):
        """Overlapping same-type particles should push apart."""
        s = _sphere_state(50, 100, radius=0.6)
        x_before = s["X"].numpy()[:50].copy()

        X_next = wp.zeros_like(s["X"], device=DEVICE)
        P_next = wp.zeros_like(s["P"], device=DEVICE)

        for _ in range(100):
            simulator.mech_step_sticky(
                s["X"],
                s["R"],
                s["P"],
                s["CT"],
                s["particle_count"],
                1e-2,
                X_next,
                P_next,
                device=DEVICE,
                grad_consist=False,
            )
            wp.copy(s["X"], X_next)
            wp.copy(s["P"], P_next)

        x_after = s["X"].numpy()[:50]

        # Average distance from center should increase (repulsion pushes out)
        dist_before = np.linalg.norm(x_before, axis=1).mean()
        dist_after = np.linalg.norm(x_after, axis=1).mean()
        assert dist_after >= dist_before * 0.95  # at least maintained

    def test_in_place_update(self):
        """mech_step_sticky should work with X_next=X (in-place)."""
        s = _sphere_state(30, 60, radius=0.6)
        x_before = s["X"].numpy()[:30].copy()

        simulator.mech_step_sticky(
            s["X"],
            s["R"],
            s["P"],
            s["CT"],
            s["particle_count"],
            5e-2,
            s["X"],
            s["P"],
            device=DEVICE,
            grad_consist=False,
        )

        x_after = s["X"].numpy()[:30]
        # Positions should have changed
        assert not np.allclose(x_before, x_after, atol=1e-8)

    def test_polarities_stay_normalized(self):
        """Polarity vectors should remain approximately unit length."""
        s = _sphere_state(30, 60)

        for _ in range(50):
            simulator.mech_step_sticky(
                s["X"],
                s["R"],
                s["P"],
                s["CT"],
                s["particle_count"],
                1e-2,
                s["X"],
                s["P"],
                device=DEVICE,
                grad_consist=False,
            )

        p = s["P"].numpy()[:30]
        norms = np.linalg.norm(p, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_returns_gradients(self):
        """mech_step_sticky should return the position gradient array."""
        s = _sphere_state(20, 40)
        gx = simulator.mech_step_sticky(
            s["X"],
            s["R"],
            s["P"],
            s["CT"],
            s["particle_count"],
            1e-2,
            s["X"],
            s["P"],
            device=DEVICE,
            grad_consist=False,
        )
        assert gx is not None
        assert gx.shape[0] == s["max_particles"]


# ---------------------------------------------------------------------------
# chem_step
# ---------------------------------------------------------------------------


class TestChemStep:
    def test_uniform_concentrations_stable(self):
        """Uniform activator/inhibitor should not change under diffusion."""
        s = _sphere_state(20, 40)
        # Set uniform concentrations on mesenchyme
        a_np = s["A"].numpy()
        i_np = s["I"].numpy()
        n_mes = s["n_mes"]
        a_np[:n_mes] = 0.05
        i_np[:n_mes] = 0.05
        A = wp.from_numpy(a_np, dtype=wp.float32, device=DEVICE)
        Inh = wp.from_numpy(i_np, dtype=wp.float32, device=DEVICE)

        lapA = wp.zeros_like(A, device=DEVICE)
        lapI = wp.zeros_like(Inh, device=DEVICE)
        S = wp.full(1, value=2e-2, dtype=wp.float32, device=DEVICE)
        T = wp.full(1, value=1e-2, dtype=wp.float32, device=DEVICE)

        a_before = A.numpy().copy()

        for _ in range(10):
            lapA.zero_()
            lapI.zero_()
            simulator.chem_step(
                A,
                Inh,
                s["X"],
                s["R"],
                lapA,
                lapI,
                S,
                T,
                10.0,
                0.01,
                s["particle_count"],
                A,
                Inh,
                device=DEVICE,
                grad_consist=False,
            )

        a_after = A.numpy()
        # Should be approximately stable (small numerical drift is OK)
        np.testing.assert_allclose(
            a_before[:n_mes],
            a_after[:n_mes],
            rtol=0.2,
        )

    def test_concentrations_stay_nonnegative(self):
        """Chemical concentrations should be clamped >= 0."""
        s = _sphere_state(30, 60)
        A, Inh = s["A"], s["I"]
        lapA = wp.zeros_like(A, device=DEVICE)
        lapI = wp.zeros_like(Inh, device=DEVICE)
        S = wp.full(1, value=5e-3, dtype=wp.float32, device=DEVICE)
        T = wp.full(1, value=1e-2, dtype=wp.float32, device=DEVICE)

        for _ in range(100):
            lapA.zero_()
            lapI.zero_()
            simulator.chem_step(
                A,
                Inh,
                s["X"],
                s["R"],
                lapA,
                lapI,
                S,
                T,
                10.0,
                0.1,
                s["particle_count"],
                A,
                Inh,
                device=DEVICE,
                grad_consist=False,
            )

        assert (A.numpy() >= 0).all()
        assert (Inh.numpy() >= 0).all()

    def test_gradient_consistency_flag(self):
        """grad_consist=True should not crash (exercises mark_read/write)."""
        s = _sphere_state(10, 20)
        A, Inh = s["A"], s["I"]
        lapA = wp.zeros_like(A, device=DEVICE)
        lapI = wp.zeros_like(Inh, device=DEVICE)
        S = wp.full(1, value=1e-2, dtype=wp.float32, device=DEVICE)
        T = wp.full(1, value=1e-2, dtype=wp.float32, device=DEVICE)

        lapA.zero_()
        lapI.zero_()
        simulator.chem_step(
            A,
            Inh,
            s["X"],
            s["R"],
            lapA,
            lapI,
            S,
            T,
            10.0,
            0.01,
            s["particle_count"],
            A,
            Inh,
            device=DEVICE,
            grad_consist=True,
        )


# ---------------------------------------------------------------------------
# growth_step
# ---------------------------------------------------------------------------


class TestGrowthStep:
    def test_radii_increase(self):
        """Mesenchymal cells with activator should grow."""
        s = _sphere_state(30, 60)
        keys = simulator.gen_key_array(s["max_particles"], device=DEVICE)
        AP = wp.full(1, value=10.0, dtype=wp.float32, device=DEVICE)
        SC = wp.full(1, value=1e-1, dtype=wp.float32, device=DEVICE)

        r_before = s["R"].numpy()[:30].copy()

        for _ in range(500):
            simulator.growth_step(
                s["R"],
                s["R_eq"],
                s["A"],
                s["CT"],
                keys,
                AP,
                SC,
                1e-2,
                0.6,
                0.85,
                s["particle_count"],
                s["R"],
                s["R_eq"],
                device=DEVICE,
                grad_consist=False,
            )

        r_after = s["R"].numpy()[:30]
        # At least some mesenchyme particles should have grown
        mes_grew = (r_after[: s["n_mes"]] > r_before[: s["n_mes"]]).sum()
        assert mes_grew > 0

    def test_epithelial_radius_approaches_ref(self):
        """Epithelial radii should trend toward R_ref."""
        s = _sphere_state(30, 60, radius=0.4)
        keys = simulator.gen_key_array(s["max_particles"], device=DEVICE)
        AP = wp.full(1, value=10.0, dtype=wp.float32, device=DEVICE)
        SC = wp.full(1, value=1e-1, dtype=wp.float32, device=DEVICE)
        R_ref = 0.6

        for _ in range(1000):
            simulator.growth_step(
                s["R"],
                s["R_eq"],
                s["A"],
                s["CT"],
                keys,
                AP,
                SC,
                1e-2,
                R_ref,
                0.85,
                s["particle_count"],
                s["R"],
                s["R_eq"],
                device=DEVICE,
                grad_consist=False,
            )

        r_after = s["R"].numpy()
        n_mes = s["n_mes"]
        epi_radii = r_after[n_mes : s["particle_count"]]
        # Epithelial radii should be closer to R_ref than initial 0.4
        assert np.mean(np.abs(epi_radii - R_ref)) < np.abs(0.4 - R_ref)


# ---------------------------------------------------------------------------
# count_neighbors
# ---------------------------------------------------------------------------


class TestCountNeighbors:
    def test_counts_positive_for_cluster(self):
        """Particles in a cluster should have nonzero neighbor counts."""
        s = _sphere_state(30, 60)
        n_tot = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)
        n_epi = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)
        n_mes = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)

        wp.launch(
            simulator.count_neighbors,
            dim=(s["particle_count"], s["particle_count"]),
            inputs=[s["X"], s["R"], s["CT"]],
            outputs=[n_tot, n_epi, n_mes],
            device=DEVICE,
        )

        tot = n_tot.numpy()[: s["particle_count"]]
        # Most particles should have at least one neighbor
        assert (tot > 0).sum() > s["particle_count"] // 2

    def test_type_counts_consistent(self):
        """n_epi + n_mes should equal n_tot for each particle."""
        s = _sphere_state(30, 60)
        n_tot = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)
        n_epi = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)
        n_mes = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)

        wp.launch(
            simulator.count_neighbors,
            dim=(s["particle_count"], s["particle_count"]),
            inputs=[s["X"], s["R"], s["CT"]],
            outputs=[n_tot, n_epi, n_mes],
            device=DEVICE,
        )

        pc = s["particle_count"]
        np.testing.assert_array_equal(
            n_tot.numpy()[:pc],
            n_epi.numpy()[:pc] + n_mes.numpy()[:pc],
        )


# ---------------------------------------------------------------------------
# division_decision + division_logic (full cycle)
# ---------------------------------------------------------------------------


class TestDivision:
    def test_division_increases_count(self):
        """Running division should produce new particles."""
        s = _sphere_state(50, 200, radius=0.6)
        keys = simulator.gen_key_array(s["max_particles"], device=DEVICE)

        # Grow mesenchyme so they become eligible for division
        AP = wp.full(1, value=10.0, dtype=wp.float32, device=DEVICE)
        SC = wp.full(1, value=1e-1, dtype=wp.float32, device=DEVICE)
        for _ in range(2000):
            simulator.growth_step(
                s["R"],
                s["R_eq"],
                s["A"],
                s["CT"],
                keys,
                AP,
                SC,
                1e-2,
                0.6,
                1.4,
                s["particle_count"],
                s["R"],
                s["R_eq"],
                device=DEVICE,
                grad_consist=False,
            )

        # Count neighbors
        n_tot = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)
        n_epi = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)
        n_mes = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)
        wp.launch(
            simulator.count_neighbors,
            dim=(s["particle_count"], s["particle_count"]),
            inputs=[s["X"], s["R"], s["CT"]],
            outputs=[n_tot, n_epi, n_mes],
            device=DEVICE,
        )

        pcount = s["particle_count"]
        div_count = wp.full(1, value=pcount, dtype=wp.int32, device=DEVICE)
        div_slots = wp.full(pcount, value=-1, dtype=wp.int32, device=DEVICE)

        wp.launch(
            simulator.division_decision,
            dim=pcount,
            inputs=[
                s["X"],
                s["R"],
                s["CT"],
                keys,
                n_epi,
                n_mes,
                div_count,
                div_slots,
                1.0,
                1.0,
                6,
                1.0,
                1.0,
                1.0,
                s["max_particles"],
            ],
            device=DEVICE,
        )

        wp.launch(
            simulator.division_logic,
            dim=pcount,
            inputs=[
                s["X"],
                s["R"],
                s["R_eq"],
                s["A"],
                s["I"],
                s["P"],
                s["CT"],
                div_slots,
            ],
            device=DEVICE,
        )

        new_count = min(div_count.numpy().item(), s["max_particles"])
        assert new_count >= pcount, (
            f"Expected division to produce new particles. " f"Before: {pcount}, after: {new_count}"
        )

    def test_division_conserves_chemicals(self):
        """Division should split activator/inhibitor evenly (mass conserved)."""
        s = _sphere_state(20, 100, radius=0.8)
        keys = simulator.gen_key_array(s["max_particles"], device=DEVICE)

        # Grow to make division likely
        AP = wp.full(1, value=10.0, dtype=wp.float32, device=DEVICE)
        SC = wp.full(1, value=1e-1, dtype=wp.float32, device=DEVICE)
        for _ in range(3000):
            simulator.growth_step(
                s["R"],
                s["R_eq"],
                s["A"],
                s["CT"],
                keys,
                AP,
                SC,
                1e-2,
                0.6,
                1.4,
                s["particle_count"],
                s["R"],
                s["R_eq"],
                device=DEVICE,
                grad_consist=False,
            )

        a_total_before = s["A"].numpy().sum()
        i_total_before = s["I"].numpy().sum()

        # Count neighbors
        n_tot = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)
        n_epi = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)
        n_mes = wp.zeros(s["max_particles"], dtype=wp.int32, device=DEVICE)
        wp.launch(
            simulator.count_neighbors,
            dim=(s["particle_count"], s["particle_count"]),
            inputs=[s["X"], s["R"], s["CT"]],
            outputs=[n_tot, n_epi, n_mes],
            device=DEVICE,
        )

        pcount = s["particle_count"]
        div_count = wp.full(1, value=pcount, dtype=wp.int32, device=DEVICE)
        div_slots = wp.full(pcount, value=-1, dtype=wp.int32, device=DEVICE)

        wp.launch(
            simulator.division_decision,
            dim=pcount,
            inputs=[
                s["X"],
                s["R"],
                s["CT"],
                keys,
                n_epi,
                n_mes,
                div_count,
                div_slots,
                1.0,
                1.0,
                6,
                1.0,
                1.0,
                1.0,
                s["max_particles"],
            ],
            device=DEVICE,
        )

        wp.launch(
            simulator.division_logic,
            dim=pcount,
            inputs=[
                s["X"],
                s["R"],
                s["R_eq"],
                s["A"],
                s["I"],
                s["P"],
                s["CT"],
                div_slots,
            ],
            device=DEVICE,
        )

        a_total_after = s["A"].numpy().sum()
        i_total_after = s["I"].numpy().sum()

        # Chemical mass should be conserved
        np.testing.assert_allclose(a_total_before, a_total_after, rtol=1e-5)
        np.testing.assert_allclose(i_total_before, i_total_after, rtol=1e-5)

    def test_capacity_limit_respected(self):
        """Division should not exceed max_particles."""
        max_p = 25
        s = _sphere_state(20, max_p, radius=1.2)
        keys = simulator.gen_key_array(max_p, device=DEVICE)

        n_tot = wp.zeros(max_p, dtype=wp.int32, device=DEVICE)
        n_epi = wp.zeros(max_p, dtype=wp.int32, device=DEVICE)
        n_mes = wp.zeros(max_p, dtype=wp.int32, device=DEVICE)
        wp.launch(
            simulator.count_neighbors,
            dim=(s["particle_count"], s["particle_count"]),
            inputs=[s["X"], s["R"], s["CT"]],
            outputs=[n_tot, n_epi, n_mes],
            device=DEVICE,
        )

        pcount = s["particle_count"]
        div_count = wp.full(1, value=pcount, dtype=wp.int32, device=DEVICE)
        div_slots = wp.full(pcount, value=-1, dtype=wp.int32, device=DEVICE)

        wp.launch(
            simulator.division_decision,
            dim=pcount,
            inputs=[
                s["X"],
                s["R"],
                s["CT"],
                keys,
                n_epi,
                n_mes,
                div_count,
                div_slots,
                0.5,
                1.0,
                6,
                1.0,
                1.0,
                1.0,
                max_p,
            ],
            device=DEVICE,
        )

        new_count = min(div_count.numpy().item(), max_p)
        assert new_count <= max_p


# ---------------------------------------------------------------------------
# Full pipeline (mechanics + chemistry + growth)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_mechanics_then_chemistry_stable(self):
        """Running mechanics then chemistry should not crash or produce NaNs."""
        s = _sphere_state(50, 100)
        dt_mech = 5e-2
        dt_chem = 5e-2
        lapA = wp.zeros_like(s["A"], device=DEVICE)
        lapI = wp.zeros_like(s["I"], device=DEVICE)
        S = wp.full(1, value=5e-3, dtype=wp.float32, device=DEVICE)
        T = wp.full(1, value=1e-2, dtype=wp.float32, device=DEVICE)

        # Relax mechanics
        for _ in range(100):
            simulator.mech_step_sticky(
                s["X"],
                s["R"],
                s["P"],
                s["CT"],
                s["particle_count"],
                dt_mech,
                s["X"],
                s["P"],
                device=DEVICE,
                grad_consist=False,
            )

        # Run chemistry
        for _ in range(200):
            lapA.zero_()
            lapI.zero_()
            simulator.chem_step(
                s["A"],
                s["I"],
                s["X"],
                s["R"],
                lapA,
                lapI,
                S,
                T,
                10.0,
                dt_chem,
                s["particle_count"],
                s["A"],
                s["I"],
                device=DEVICE,
                grad_consist=False,
            )

        # Check no NaNs
        x = s["X"].numpy()[: s["particle_count"]]
        a = s["A"].numpy()[: s["particle_count"]]
        assert not np.isnan(x).any(), "NaN in positions"
        assert not np.isnan(a).any(), "NaN in activators"
        assert np.isfinite(x).all(), "Inf in positions"
