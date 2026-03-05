"""Spike-aware learning rate scheduler.

Motivated by the "edge of stability" regime described in *Understanding
Optimization in Deep Learning with Central Flows* (Cohen et al., 2025).
Training at the edge of stability produces loss spikes; productive spikes
are followed by new minima.  When spikes stop being productive the learning
rate is too high and should be reduced.

State machine
-------------
::

    NORMAL ──(spike)──► RECOVERING
      ▲                     │
      ├──(new min)──────────┘
      │                     │
      └──(spike, no min)──► reduce LR, stay RECOVERING
"""

from __future__ import annotations

import warnings
from collections import deque
from enum import Enum, auto

import torch


class _Phase(Enum):
    NORMAL = auto()
    RECOVERING = auto()


class SpikeAwareScheduler:
    """Reduce LR when loss spikes fail to produce new minima.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    spike_factor : float
        A spike is detected when the current loss exceeds
        ``window_mean * (1 + spike_factor)``.
    window_size : int
        Number of recent loss values used to compute the running mean
        for spike detection.
    min_improvement : float
        After a spike, a "new minimum" is accepted when
        ``loss < best_loss * (1 + min_improvement)``.  Set to 0 for
        strict improvement.
    decay_factor : float
        Multiplicative factor applied to LR on each reduction.
    lr_floor : float
        Minimum learning rate (will not decay below this).
    floor_patience : int
        After the LR has reached the floor, if no new minimum is seen
        for this many steps a warning is emitted.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        spike_factor: float = 0.5,
        window_size: int = 20,
        min_improvement: float = 0.01,
        decay_factor: float = 0.5,
        lr_floor: float = 1e-6,
        floor_patience: int = 100,
    ) -> None:
        self.optimizer = optimizer
        self.spike_factor = spike_factor
        self.window_size = window_size
        self.min_improvement = min_improvement
        self.decay_factor = decay_factor
        self.lr_floor = lr_floor
        self.floor_patience = floor_patience

        self._window: deque[float] = deque(maxlen=window_size)
        self._phase = _Phase.NORMAL
        self._best_loss = float("inf")
        self._steps_since_best_at_floor = 0
        self._floor_warning_emitted = False
        self._num_reductions = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss: float) -> dict[str, object]:
        """Feed the current loss value and potentially adjust LR.

        Returns a dict with diagnostics:
        ``{"lr", "phase", "spike", "new_best", "num_reductions"}``.
        """
        spike = False
        new_best = False

        # Spike detection: loss > window_mean * (1 + spike_factor)
        if len(self._window) >= 1:
            window_mean = sum(self._window) / len(self._window)
            spike = loss > window_mean * (1 + self.spike_factor)

        # Track new minima (within tolerance)
        if loss < self._best_loss * (1 + self.min_improvement):
            if loss < self._best_loss:
                self._best_loss = loss
            new_best = True
            self._steps_since_best_at_floor = 0
            self._floor_warning_emitted = False

        # State transitions
        if self._phase == _Phase.NORMAL:
            if spike:
                self._phase = _Phase.RECOVERING
        elif self._phase == _Phase.RECOVERING:
            if new_best:
                # Productive spike — recovered to new minimum
                self._phase = _Phase.NORMAL
            elif spike:
                # Another spike without recovery — reduce LR
                self._reduce_lr()
                print(f"LR reduced to {self.get_lr()}")
                # Stay in RECOVERING; need a new min to exit

        # Floor patience: warn if stuck at floor with no improvement
        if self._at_floor():
            if not new_best:
                self._steps_since_best_at_floor += 1
            if (
                self._steps_since_best_at_floor >= self.floor_patience
                and not self._floor_warning_emitted
            ):
                warnings.warn(
                    f"SpikeAwareScheduler: LR at floor ({self.lr_floor:.2e}) "
                    f"with no improvement for {self.floor_patience} steps. "
                    "Consider stopping training or adjusting hyperparameters.",
                    stacklevel=2,
                )
                self._floor_warning_emitted = True

        self._window.append(loss)

        return {
            "lr": self.get_lr(),
            "phase": self._phase.name,
            "spike": spike,
            "new_best": new_best,
            "num_reductions": self._num_reductions,
        }

    def get_lr(self) -> float:
        """Return current learning rate (from first param group)."""
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        """Serialize scheduler state for checkpointing."""
        return {
            "window": list(self._window),
            "phase": self._phase.name,
            "best_loss": self._best_loss,
            "steps_since_best_at_floor": self._steps_since_best_at_floor,
            "floor_warning_emitted": self._floor_warning_emitted,
            "num_reductions": self._num_reductions,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from checkpoint."""
        self._window = deque(state["window"], maxlen=self.window_size)
        self._phase = _Phase[state["phase"]]
        self._best_loss = state["best_loss"]
        self._steps_since_best_at_floor = state["steps_since_best_at_floor"]
        self._floor_warning_emitted = state["floor_warning_emitted"]
        self._num_reductions = state["num_reductions"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _at_floor(self) -> bool:
        return self.get_lr() <= self.lr_floor

    def _reduce_lr(self) -> None:
        for group in self.optimizer.param_groups:
            new_lr = max(group["lr"] * self.decay_factor, self.lr_floor)
            group["lr"] = new_lr
        self._num_reductions += 1
