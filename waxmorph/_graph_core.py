"""Detached NumPy contact topology shared by Torch and JAX."""

from numbers import Integral

import numpy as np
from scipy.spatial import cKDTree

from .constants import EPS_DIST


def _active_particle_count(particle_count: int, **arrays) -> int:
    if isinstance(particle_count, bool) or not isinstance(particle_count, Integral):
        raise TypeError("particle_count must be a non-boolean integer.")

    lengths = {}
    for name, array in arrays.items():
        shape = getattr(array, "shape", None)
        if shape is None:
            shape = np.shape(array)
        if not shape:
            raise ValueError(f"{name} must have a particle axis.")
        lengths[name] = int(shape[0])

    active = int(particle_count)
    if active > 0:
        short = {name: length for name, length in lengths.items() if length < active}
        if short:
            details = ", ".join(f"{name}={length}" for name, length in short.items())
            raise ValueError(f"particle_count={active} exceeds input length: {details}.")
        return active

    if len(set(lengths.values())) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Full-array input lengths must match: {details}.")
    return next(iter(lengths.values()), 0)


def _validate_graph_geometry(pos: np.ndarray, rad: np.ndarray, eps_dist: float) -> float:
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape [N, 3], got {pos.shape}.")
    if rad.ndim != 1:
        raise ValueError(f"radii must have shape [N], got {rad.shape}.")
    if len(pos) != len(rad):
        raise ValueError(f"Position/radius lengths differ: {len(pos)} != {len(rad)}.")
    for name, array in (("positions", pos), ("radii", rad)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values.")
    if (rad < 0).any():
        raise ValueError("radii must be nonnegative.")

    value = np.asarray(eps_dist)
    if value.ndim != 0:
        raise ValueError("eps_dist must be a finite nonnegative scalar.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("eps_dist must be a finite nonnegative scalar.") from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError("eps_dist must be a finite nonnegative scalar.")
    return value


def _validate_feature_shapes(
    active: int,
    *,
    positions=None,
    polarities=None,
    concentrations=None,
    edge_index=None,
) -> None:
    for name, array in (("positions", positions), ("polarities", polarities)):
        if array is not None and tuple(array.shape) != (active, 3):
            raise ValueError(f"{name} must have shape [{active}, 3], got {array.shape}.")
    if concentrations is not None and (
        concentrations.ndim not in (1, 2) or concentrations.shape[0] != active
    ):
        raise ValueError(
            f"concentrations must have shape [{active}] or [{active}, C], "
            f"got {concentrations.shape}."
        )
    if edge_index is not None and (edge_index.ndim != 2 or edge_index.shape[0] != 2):
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}.")


def build_edge_index_np(
    pos: np.ndarray,
    rad: np.ndarray,
    eps_dist: float = EPS_DIST,
) -> tuple[np.ndarray, np.ndarray]:
    """Build detached, bidirectional contacts as ``int64`` sender and receiver arrays ``[E]``.

    Inputs have shapes ``pos[N,3]`` and ``rad[N]``. Distinct particles contact when
    ``||pos[i] - pos[j]|| <= rad[i] + rad[j] + eps_dist``. A host
    :class:`scipy.spatial.cKDTree` supplies candidates, so topology is outside backend
    autodiff; candidate and output work depends on pair counts and can be quadratic for
    dense graphs. Inputs must be finite, with nonnegative radii and ``eps_dist``.

    Examples:
        >>> pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        >>> rad = np.array([0.6, 0.6, 0.6])
        >>> senders, receivers = build_edge_index_np(pos, rad, eps_dist=0.0)
        >>> print(senders.tolist(), receivers.tolist())
        [0, 1] [1, 0]
    """
    pos, rad = np.asarray(pos), np.asarray(rad)
    eps_dist = _validate_graph_geometry(pos, rad, eps_dist)
    if len(pos) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty

    max_r = float(rad.max()) * 2 + eps_dist
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=max_r, output_type="ndarray")
    ii, jj = pairs[:, 0], pairs[:, 1]
    dists = np.linalg.norm(pos[ii] - pos[jj], axis=-1)
    thresholds = rad[ii] + rad[jj] + eps_dist
    mask = dists <= thresholds
    ii, jj = ii[mask], jj[mask]

    senders = np.concatenate([ii, jj]).astype(np.int64)
    receivers = np.concatenate([jj, ii]).astype(np.int64)
    return senders, receivers
