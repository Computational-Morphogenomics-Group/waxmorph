"""Tests for the spike-aware learning rate scheduler."""

import warnings

import pytest
import torch

from waxmorph.scheduler import SpikeAwareScheduler


def _make_optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


class TestSpikeDetection:
    def test_no_spike_on_smooth_descent(self):
        opt = _make_optimizer()
        sched = SpikeAwareScheduler(opt, spike_factor=0.5, window_size=5)
        for loss in [1.0, 0.95, 0.90, 0.85, 0.80]:
            info = sched.step(loss)
            assert not info["spike"]

    def test_spike_detected_on_large_increase(self):
        opt = _make_optimizer()
        sched = SpikeAwareScheduler(opt, spike_factor=0.5, window_size=5)
        # Fill the window with low values
        for _ in range(5):
            sched.step(1.0)
        # Now a spike: 1.0 * (1 + 0.5) = 1.5, loss=2.0 > 1.5
        info = sched.step(2.0)
        assert info["spike"]

    def test_spike_uses_window_mean(self):
        opt = _make_optimizer()
        sched = SpikeAwareScheduler(opt, spike_factor=0.5, window_size=3)
        sched.step(1.0)
        sched.step(2.0)
        sched.step(3.0)
        # Window mean = 2.0, threshold = 2.0 * 1.5 = 3.0
        info_no = sched.step(3.0)
        assert not info_no["spike"]


class TestRecovery:
    def test_productive_spike_returns_to_normal(self):
        opt = _make_optimizer()
        sched = SpikeAwareScheduler(
            opt,
            spike_factor=0.5,
            window_size=5,
            min_improvement=0.01,
        )
        for _ in range(5):
            sched.step(1.0)
        # Spike
        info = sched.step(2.0)
        assert info["phase"] == "RECOVERING"
        # Recover to new min (within tolerance: < 1.0 * 1.01 = 1.01)
        info = sched.step(0.95)
        assert info["phase"] == "NORMAL"
        assert info["new_best"]

    def test_unproductive_spike_reduces_lr(self):
        opt = _make_optimizer(lr=1e-2)
        sched = SpikeAwareScheduler(
            opt,
            spike_factor=0.5,
            window_size=3,
            decay_factor=0.5,
            min_improvement=0.0,
        )
        # Fill window with low values
        for _ in range(3):
            sched.step(1.0)
        # First spike -> RECOVERING
        sched.step(3.0)
        # No recovery (1.1 > best=1.0), keep window mean low for next spike
        sched.step(1.1)
        sched.step(1.1)
        # Second spike without new min -> LR reduction
        # Window = [3.0, 1.1, 1.1], mean ≈ 1.73, threshold ≈ 2.6
        info = sched.step(3.0)
        assert info["spike"]
        assert info["num_reductions"] == 1
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-3)


class TestLRFloor:
    def test_lr_does_not_go_below_floor(self):
        opt = _make_optimizer(lr=1e-3)
        sched = SpikeAwareScheduler(
            opt,
            spike_factor=0.3,
            window_size=3,
            decay_factor=0.1,
            lr_floor=1e-4,
        )
        # Force many reductions
        for _ in range(20):
            for _ in range(3):
                sched.step(1.0)
            sched.step(5.0)
        assert opt.param_groups[0]["lr"] >= 1e-4

    def test_floor_patience_warning(self):
        opt = _make_optimizer(lr=1e-6)
        sched = SpikeAwareScheduler(
            opt,
            lr_floor=1e-6,
            floor_patience=5,
            min_improvement=0.0,
        )
        # Set a best loss, then feed non-improving (higher) losses
        sched.step(1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(10):
                sched.step(1.1)
            assert any("no improvement" in str(x.message).lower() for x in w)

    def test_floor_warning_emitted_once(self):
        opt = _make_optimizer(lr=1e-6)
        sched = SpikeAwareScheduler(
            opt,
            lr_floor=1e-6,
            floor_patience=3,
            min_improvement=0.0,
        )
        sched.step(1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(20):
                sched.step(1.1)
            spike_warnings = [x for x in w if "no improvement" in str(x.message).lower()]
            assert len(spike_warnings) == 1


class TestMinImprovement:
    def test_within_tolerance_counts_as_new_best(self):
        opt = _make_optimizer()
        sched = SpikeAwareScheduler(
            opt,
            spike_factor=0.5,
            window_size=3,
            min_improvement=0.05,
        )
        sched.step(1.0)
        # 1.02 < 1.0 * 1.05 = 1.05, so within tolerance
        info = sched.step(1.02)
        assert info["new_best"]

    def test_outside_tolerance_not_new_best(self):
        opt = _make_optimizer()
        sched = SpikeAwareScheduler(
            opt,
            spike_factor=0.5,
            window_size=3,
            min_improvement=0.01,
        )
        sched.step(1.0)
        # 1.05 > 1.0 * 1.01 = 1.01
        info = sched.step(1.05)
        assert not info["new_best"]


class TestStateDict:
    def test_save_and_load(self):
        opt = _make_optimizer()
        sched = SpikeAwareScheduler(opt, window_size=5)
        for v in [1.0, 0.9, 0.8, 0.7, 0.6]:
            sched.step(v)
        state = sched.state_dict()

        opt2 = _make_optimizer()
        sched2 = SpikeAwareScheduler(opt2, window_size=5)
        sched2.load_state_dict(state)
        assert sched2._best_loss == sched._best_loss
        assert list(sched2._window) == list(sched._window)
        assert sched2._phase == sched._phase
        assert sched2._num_reductions == sched._num_reductions


class TestDiagnostics:
    def test_step_returns_expected_keys(self):
        opt = _make_optimizer()
        sched = SpikeAwareScheduler(opt)
        info = sched.step(1.0)
        assert set(info.keys()) == {"lr", "phase", "spike", "new_best", "num_reductions"}

    def test_get_lr(self):
        opt = _make_optimizer(lr=0.01)
        sched = SpikeAwareScheduler(opt)
        assert sched.get_lr() == pytest.approx(0.01)
