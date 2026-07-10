"""Tests for the full Warp simulator (mechanics, chemistry, growth, division)."""

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

WARP_DEVICES = ["cpu"] + (["cuda"] if DEVICE == "cuda" else [])

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


@wp.kernel(enable_backward=False)
def _reserve_division_slots(
    div_count: wp.array(dtype=wp.int32),
    max_particles: wp.int32,
    div_slots: wp.array(dtype=wp.int32),
) -> None:
    div_slots[wp.tid()] = simulator._reserve_division_slot(div_count, max_particles)


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
        f_net = simulator.mech_step_sticky(
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
        assert f_net is not None
        assert f_net.shape[0] == s["max_particles"]


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
        chi = wp.full(1, value=2e-2, dtype=wp.float32, device=DEVICE)
        gamma = wp.full(1, value=1e-2, dtype=wp.float32, device=DEVICE)

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
                chi,
                gamma,
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
        chi = wp.full(1, value=5e-3, dtype=wp.float32, device=DEVICE)
        gamma = wp.full(1, value=1e-2, dtype=wp.float32, device=DEVICE)

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
                chi,
                gamma,
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
        chi = wp.full(1, value=1e-2, dtype=wp.float32, device=DEVICE)
        gamma = wp.full(1, value=1e-2, dtype=wp.float32, device=DEVICE)

        lapA.zero_()
        lapI.zero_()
        simulator.chem_step(
            A,
            Inh,
            s["X"],
            s["R"],
            lapA,
            lapI,
            chi,
            gamma,
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


def _growth_boundary_state(device):
    return {
        "R": wp.array([1.2, 0.5, 1.0, 1.2, 0.5, 1.0], dtype=wp.float32, device=device),
        "R_eq": wp.array([0.8, 1.0, 1.0, 0.7, 0.8, 0.9], dtype=wp.float32, device=device),
        "A": wp.ones(6, dtype=wp.float32, device=device),
        "CT": wp.array([0, 0, 0, 1, 1, 1], dtype=wp.uint32, device=device),
        "keys": simulator.gen_key_array(6, device=device),
        "alpha_grow": wp.ones(1, dtype=wp.float32, device=device),
        "ell_sw": wp.full(1, value=0.1, dtype=wp.float32, device=device),
        "R_next": wp.full(6, value=-99.0, dtype=wp.float32, device=device),
        "R_eq_next": wp.full(6, value=-99.0, dtype=wp.float32, device=device),
    }


def _run_boundary_growth(s, dt, R_ref=1.0, R_max=1.0, device="cpu"):
    simulator.growth_step(
        s["R"],
        s["R_eq"],
        s["A"],
        s["CT"],
        s["keys"],
        s["alpha_grow"],
        s["ell_sw"],
        dt,
        R_ref,
        R_max,
        6,
        s["R_next"],
        s["R_eq_next"],
        device=device,
        grad_consist=False,
    )


class TestGrowthStep:
    @pytest.mark.parametrize("device", WARP_DEVICES)
    @pytest.mark.parametrize(
        "dt,expected_r,expected_r_eq",
        [
            (
                0.0,
                [1.2, 0.5, 1.0, 1.2, 0.5, 1.0],
                [0.8, 1.0, 1.0, 0.7, 0.8, 0.9],
            ),
            (
                10.0,
                [1.2, 1.0, 1.0, 1.2, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.7, 0.8, 0.9],
            ),
        ],
        ids=["zero-step", "large-step"],
    )
    def test_growth_is_bounded_and_writes_separate_outputs(
        self, device, dt, expected_r, expected_r_eq
    ):
        s = _growth_boundary_state(device)
        r_before = s["R"].numpy().copy()
        r_eq_before = s["R_eq"].numpy().copy()
        keys_before = s["keys"].numpy().copy()

        _run_boundary_growth(s, dt, device=device)

        np.testing.assert_allclose(s["R_next"].numpy(), expected_r, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(s["R_eq_next"].numpy(), expected_r_eq, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(s["R"].numpy(), r_before)
        np.testing.assert_array_equal(s["R_eq"].numpy(), r_eq_before)
        keys_after = s["keys"].numpy()
        assert np.all(keys_after[:3] != keys_before[:3])
        np.testing.assert_array_equal(keys_after[3:], keys_before[3:])

    @pytest.mark.parametrize("device", WARP_DEVICES)
    def test_mesenchymal_rate_distribution(self, device):
        n = 4096
        radius = 0.5
        dt = 0.01
        R = wp.full(n, value=radius, dtype=wp.float32, device=device)
        R_eq = wp.full(n, value=radius, dtype=wp.float32, device=device)
        A = wp.ones(n, dtype=wp.float32, device=device)
        CT = wp.zeros(n, dtype=wp.uint32, device=device)
        keys = simulator.gen_key_array(n, device=device)
        alpha_grow = wp.ones(1, dtype=wp.float32, device=device)
        ell_sw = wp.full(1, value=0.1, dtype=wp.float32, device=device)
        R_next = wp.empty_like(R)
        R_eq_next = wp.empty_like(R_eq)

        simulator.growth_step(
            R,
            R_eq,
            A,
            CT,
            keys,
            alpha_grow,
            ell_sw,
            dt,
            1.0,
            2.0,
            n,
            R_next,
            R_eq_next,
            device=device,
            grad_consist=False,
        )

        volume = simulator.FOUR_THIRDS_PI * radius**3
        concentration = 1.0 / (volume + simulator.EPS_DEN)
        fraction = concentration / (0.1 + concentration + simulator.EPS_DEN)
        rates = (R_eq_next.numpy() - radius) / (fraction * dt)
        standard_error = (0.2 / np.sqrt(12.0)) / np.sqrt(n)
        assert rates.min() >= 0.8 - 1e-5
        assert rates.max() <= 1.0 + 1e-5
        assert rates.mean() == pytest.approx(0.9, abs=3 * standard_error)

    @pytest.mark.parametrize(
        "name,value",
        [
            ("dt", -1.0),
            ("dt", np.nan),
            ("dt", np.inf),
            ("R_max", -1.0),
            ("R_max", np.nan),
            ("R_max", np.inf),
            ("R_ref", 0.0),
            ("R_ref", -1.0),
            ("R_ref", np.nan),
            ("R_ref", np.inf),
        ],
    )
    def test_growth_rejects_invalid_scalars_before_launch(self, name, value):
        s = _growth_boundary_state("cpu")
        scalars = {"dt": 0.1, "R_ref": 1.0, "R_max": 1.0}
        scalars[name] = value
        keys_before = s["keys"].numpy().copy()

        with pytest.raises(ValueError, match=name):
            _run_boundary_growth(s, **scalars)

        np.testing.assert_array_equal(s["R_next"].numpy(), np.full(6, -99.0))
        np.testing.assert_array_equal(s["R_eq_next"].numpy(), np.full(6, -99.0))
        np.testing.assert_array_equal(s["keys"].numpy(), keys_before)

    def test_radii_increase(self):
        """Mesenchymal cells with activator should grow."""
        s = _sphere_state(30, 60)
        keys = simulator.gen_key_array(s["max_particles"], device=DEVICE)
        alpha_grow = wp.full(1, value=10.0, dtype=wp.float32, device=DEVICE)
        ell_sw = wp.full(1, value=1e-1, dtype=wp.float32, device=DEVICE)

        r_before = s["R"].numpy()[:30].copy()

        for _ in range(500):
            simulator.growth_step(
                s["R"],
                s["R_eq"],
                s["A"],
                s["CT"],
                keys,
                alpha_grow,
                ell_sw,
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
        alpha_grow = wp.full(1, value=10.0, dtype=wp.float32, device=DEVICE)
        ell_sw = wp.full(1, value=1e-1, dtype=wp.float32, device=DEVICE)
        R_ref = 0.6

        for _ in range(1000):
            simulator.growth_step(
                s["R"],
                s["R_eq"],
                s["A"],
                s["CT"],
                keys,
                alpha_grow,
                ell_sw,
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

        simulator.count_neighbors_step(
            s["X"],
            s["R"],
            s["CT"],
            s["particle_count"],
            n_tot,
            n_epi,
            n_mes,
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

        simulator.count_neighbors_step(
            s["X"],
            s["R"],
            s["CT"],
            s["particle_count"],
            n_tot,
            n_epi,
            n_mes,
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


def _reservation_result(device, initial, capacity, threads):
    div_count = wp.full(1, value=initial, dtype=wp.int32, device=device)
    div_slots = wp.full(threads, value=-1, dtype=wp.int32, device=device)
    wp.launch(
        _reserve_division_slots,
        dim=threads,
        inputs=[div_count, capacity, div_slots],
        device=device,
    )
    wp.synchronize_device(device)
    return int(div_count.numpy()[0]), div_slots.numpy()


class TestDivision:
    @pytest.mark.parametrize("device", WARP_DEVICES)
    @pytest.mark.parametrize(
        "initial,capacity,expected",
        [(17, 17, []), (9, 17, list(range(9, 17)))],
        ids=["full", "multiple-free"],
    )
    def test_reservation_is_bounded_and_unique(self, device, initial, capacity, expected):
        count, slots = _reservation_result(device, initial, capacity, threads=256)
        accepted = slots[slots >= 0]
        assert count == capacity
        assert sorted(accepted.tolist()) == expected
        assert len(accepted) == len(np.unique(accepted))
        assert np.all(slots[slots < 0] == -1)

    @pytest.mark.parametrize("device", WARP_DEVICES)
    def test_one_free_slot_under_contention(self, device):
        repetitions = 20 if device.startswith("cuda") else 1
        for _ in range(repetitions):
            count, slots = _reservation_result(device, 16, 17, threads=4096)
            accepted = slots[slots >= 0]
            assert count == 17
            np.testing.assert_array_equal(accepted, [16])
            assert np.count_nonzero(slots == -1) == 4095

    def test_division_increases_count(self):
        """Running division should produce new particles."""
        s = _sphere_state(50, 200, radius=0.6)
        keys = simulator.gen_key_array(s["max_particles"], device=DEVICE)

        # Grow mesenchyme so they become eligible for division
        alpha_grow = wp.full(1, value=10.0, dtype=wp.float32, device=DEVICE)
        ell_sw = wp.full(1, value=1e-1, dtype=wp.float32, device=DEVICE)
        for _ in range(2000):
            simulator.growth_step(
                s["R"],
                s["R_eq"],
                s["A"],
                s["CT"],
                keys,
                alpha_grow,
                ell_sw,
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
        simulator.count_neighbors_step(
            s["X"],
            s["R"],
            s["CT"],
            s["particle_count"],
            n_tot,
            n_epi,
            n_mes,
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

        accepted = np.count_nonzero(div_slots.numpy() >= 0)
        new_count = div_count.numpy().item()
        assert accepted > 0
        assert new_count == pcount + accepted
        assert new_count <= s["max_particles"]

    def test_division_conserves_chemicals(self):
        """Division should split activator/inhibitor evenly (mass conserved)."""
        s = _sphere_state(20, 100, radius=0.8)
        keys = simulator.gen_key_array(s["max_particles"], device=DEVICE)

        # Grow to make division likely
        alpha_grow = wp.full(1, value=10.0, dtype=wp.float32, device=DEVICE)
        ell_sw = wp.full(1, value=1e-1, dtype=wp.float32, device=DEVICE)
        for _ in range(3000):
            simulator.growth_step(
                s["R"],
                s["R_eq"],
                s["A"],
                s["CT"],
                keys,
                alpha_grow,
                ell_sw,
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
        simulator.count_neighbors_step(
            s["X"],
            s["R"],
            s["CT"],
            s["particle_count"],
            n_tot,
            n_epi,
            n_mes,
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

        assert np.count_nonzero(div_slots.numpy() >= 0) > 0

        a_total_after = s["A"].numpy().sum()
        i_total_after = s["I"].numpy().sum()

        # Chemical mass should be conserved
        np.testing.assert_allclose(a_total_before, a_total_after, rtol=1e-5)
        np.testing.assert_allclose(i_total_before, i_total_after, rtol=1e-5)

    def test_capacity_limit_respected(self):
        """Division should not exceed max_particles."""
        max_p = 25
        s = _sphere_state(20, max_p, radius=1.2)
        s["CT"].fill_(0)
        keys = simulator.gen_key_array(max_p, device=DEVICE)
        keys_before = keys.numpy().copy()
        n_epi = wp.zeros(max_p, dtype=wp.int32, device=DEVICE)
        n_mes = wp.zeros(max_p, dtype=wp.int32, device=DEVICE)

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

        slots = div_slots.numpy()
        accepted_slots = slots[slots >= 0]
        new_count = div_count.numpy().item()
        assert new_count == pcount + len(accepted_slots)
        assert new_count <= max_p
        assert len(accepted_slots) == len(np.unique(accepted_slots))
        assert np.all((accepted_slots >= pcount) & (accepted_slots < max_p))
        assert np.all(keys.numpy()[:pcount] != keys_before[:pcount])


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
        chi = wp.full(1, value=5e-3, dtype=wp.float32, device=DEVICE)
        gamma = wp.full(1, value=1e-2, dtype=wp.float32, device=DEVICE)

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
                chi,
                gamma,
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
