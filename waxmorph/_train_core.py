from __future__ import annotations

import math
import operator
from numbers import Integral, Real
from typing import Any

import numpy as np

_INTEGER_MINIMUMS = {
    "n_epochs": 1,
    "t_rollout": 1,
    "mech_steps": 0,
    "diff_steps": 0,
    "log_every": 1,
}
_NONNEGATIVE_SCALARS = ("dt_mech", "dt_diff", "dt_gns", "D_emu", "lambda_reg")


def _validate_train_config(config: Any) -> None:
    minimums = _INTEGER_MINIMUMS | (
        {"max_edges_factor": 1} if hasattr(config, "max_edges_factor") else {}
    )
    for name, minimum in minimums.items():
        value = getattr(config, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"config.{name} must be an integer, got {value!r}.")
        if value < minimum:
            raise ValueError(f"config.{name} must be >= {minimum}, got {value}.")

    for name in _NONNEGATIVE_SCALARS:
        _validate_real_scalar(name, getattr(config, name), minimum=0.0)

    if config.grad_clip_norm is not None:
        _validate_real_scalar("grad_clip_norm", config.grad_clip_norm, minimum=0.0, strict=True)


def _validate_real_scalar(name: str, value: Any, *, minimum: float, strict: bool = False) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"config.{name} must be a real number, got {value!r}.")
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        comparison = ">" if strict else ">="
        raise ValueError(f"config.{name} must be finite and {comparison} {minimum:g}, got {value}.")


def _float32_array(
    name: str,
    value: Any,
    *,
    ndim: int,
    width: int | None = None,
    minimum: float | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or (width is not None and array.shape[-1] != width):
        shape = f"[N,{width}]" if ndim == 2 and width is not None else f"{ndim}-D"
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numbers, got dtype {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    if minimum is not None and np.any(array < minimum):
        raise ValueError(f"{name} must be >= {minimum:g}.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} cannot be represented as finite float32 values.")
    return array


def _prepare_training_inputs(
    config: Any,
    source_pos: Any,
    polarities: Any,
    c: Any,
    radii: Any,
    targets: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, np.ndarray]],
]:
    _validate_train_config(config)
    source_pos = _float32_array("source_pos", source_pos, ndim=2, width=3)
    if not source_pos.shape[0]:
        raise ValueError("source_pos must contain at least one particle.")

    particle_count = source_pos.shape[0]
    polarities = _float32_array("polarities", polarities, ndim=2, width=3)
    c = _float32_array("c", c, ndim=2)
    radii = _float32_array("radii", radii, ndim=1, minimum=0.0)
    for name, array in (("polarities", polarities), ("c", c), ("radii", radii)):
        if array.shape[0] != particle_count:
            raise ValueError(
                f"{name} must contain {particle_count} particles, got {array.shape[0]}."
            )
    if targets is None:
        raise ValueError("train() requires `targets`.")
    try:
        targets = list(targets)
    except TypeError as exc:
        raise TypeError("targets must be an iterable of (frame, positions) pairs.") from exc
    if not targets:
        raise ValueError("targets must be a nonempty collection.")

    prepared_targets = []
    seen_frames = set()
    for target_index, pair in enumerate(targets):
        try:
            frame, positions = pair
        except (TypeError, ValueError) as exc:
            raise ValueError(f"targets[{target_index}] must be a (frame, positions) pair.") from exc
        try:
            frame = operator.index(frame)
        except TypeError as exc:
            raise TypeError(f"targets[{target_index}] frame must implement __index__.") from exc
        if not 0 <= frame < config.t_rollout:
            raise ValueError(f"Target frame {frame} outside [0, {config.t_rollout}).")
        if frame in seen_frames:
            raise ValueError(f"Duplicate target frame {frame}.")

        positions = _float32_array(f"targets[{target_index}] positions", positions, ndim=2, width=3)
        if not positions.shape[0]:
            raise ValueError(f"targets[{target_index}] positions must be nonempty.")
        prepared_targets.append((frame, positions))
        seen_frames.add(frame)

    return source_pos, polarities, c, radii, prepared_targets
