r"""Normalize meshes and sample their volumes for rollout point clouds.

For mesh volume ``V`` and requested count ``N``, ``r = (3V / 4 pi N)^(1/3)`` and
the Poisson exclusion distance is ``d_min = 1.2 r``. This permits equal-radius overlap
but guarantees neither contact nor graph connectivity. ``N=2000`` is a public default,
not an invariant. Exact-count wrappers may reduce ``r`` and ``d_min`` together; returned
radius fields report that effective value. Source and target are sampled independently, not
as correspondences.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from tqdm import tqdm

# N equal sphere volumes sum to V when this is 1.0.
_PACKING_FRACTION = 1.0
# d_min=1.2r permits overlap but guarantees neither contact nor connectivity.
_CONTACT_SPACING_RATIO = 1.2
_UNIT_SPHERE_VOLUME = 4.0 * np.pi / 3.0


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load one mesh, reorient its normals, and fill small holes in place.

    Repair does not guarantee watertightness. :func:`normalize_mesh` handles scale separately.
    """
    mesh = trimesh.load(path, force="mesh")
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    return mesh


def _validate_n_points(n_points: int) -> int:
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}.")
    return n_points


def _validate_extent(name: str, extent: float) -> float:
    extent = float(extent)
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError(f"{name} must be a positive finite value, got {extent}.")
    return extent


def _mesh_volume(mesh: trimesh.Trimesh) -> float:
    """Use ``abs(mesh.volume)``, then the bounding-box volume if the first is unusable.

    Raises ``ValueError`` when neither estimate is positive and finite.
    """
    volume = abs(float(mesh.volume))
    if np.isfinite(volume) and volume > 0:
        return volume

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    bbox_volume = float(np.prod(bounds[1] - bounds[0]))
    if np.isfinite(bbox_volume) and bbox_volume > 0:
        return bbox_volume
    raise ValueError("mesh must have positive volume or bounding-box volume.")


def _radius_from_volume(volume: float, n_points: int) -> float:
    r"""Size equal spheres by equating their total volume to the packed mesh volume.

    .. math::

        N \cdot \tfrac{4}{3}\pi r^3 = \phi V
        \quad\Longrightarrow\quad
        r = \left(\frac{3\,\phi V}{4\pi N}\right)^{1/3},

    Here ``phi = _PACKING_FRACTION = 1``. ``V`` must be positive and finite.
    """
    volume = _validate_extent("volume", volume)
    n_points = _validate_n_points(n_points)
    particle_volume = (_PACKING_FRACTION * volume) / float(n_points)
    return float(np.cbrt(particle_volume / _UNIT_SPHERE_VOLUME))


def _radius_from_mesh(mesh: trimesh.Trimesh, n_points: int) -> float:
    return _radius_from_volume(_mesh_volume(mesh), n_points)


def _connected_poisson_min_dist(radius: float) -> float:
    """Compute ``d_min=1.2r``; this permits overlap without guaranteeing contact."""
    radius = _validate_extent("radius", radius)
    return _CONTACT_SPACING_RATIO * radius


def _pitch_from_min_dist(min_dist: float) -> float:
    """Use ``0.4 d_min`` to provide multiple voxel candidates across the exclusion distance."""
    min_dist = _validate_extent("min_dist", min_dist)
    return 0.4 * min_dist


def normalize_mesh(mesh: trimesh.Trimesh, target_extent: float = 10.0) -> trimesh.Trimesh:
    """Center and isotropically scale a mesh in place, preserving aspect ratio.

    Returns the same object with its bounding-box center at the origin and longest side equal
    to ``target_extent``. Both target and original maximum extents must be positive and finite.

    Examples:
        >>> mesh = trimesh.creation.box(extents=(2, 4, 6))
        >>> _ = normalize_mesh(mesh, target_extent=3.0)
        >>> print(mesh.bounds.round(1).tolist(), mesh.extents.round(1).tolist())
        [[-0.5, -1.0, -1.5], [0.5, 1.0, 1.5]] [1.0, 2.0, 3.0]
    """
    target_extent = _validate_extent("target_extent", target_extent)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    raw_extent = verts.max(axis=0) - verts.min(axis=0)
    max_extent = float(raw_extent.max())
    if not np.isfinite(max_extent) or max_extent <= 0:
        raise ValueError(f"mesh must have positive extent, got {max_extent}.")

    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    mesh.vertices -= center
    mesh.vertices *= target_extent / max_extent
    return mesh


def _poisson_disk_subsample(
    candidates: np.ndarray,
    min_dist: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Accept RNG-permuted candidates using float32 squared-distance tests.

    A spatial hash limits tests to 27 neighboring cells. Returns float32 ``[K,3]`` with
    ``K <= n_points``; candidates are rejected when their squared distance is below
    ``min_dist**2``. An empty input returns ``[0,3]``.
    """
    n_points = _validate_n_points(n_points)
    min_dist = _validate_extent("min_dist", min_dist)
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError(f"candidates must have shape [M, 3], got {candidates.shape}.")

    candidates = candidates[rng.permutation(len(candidates))]
    cell_size = min_dist
    inv_cell = 1.0 / cell_size
    min_dist2 = min_dist * min_dist

    occupied: dict[tuple[int, int, int], list[np.ndarray]] = {}
    accepted: list[np.ndarray] = []

    for pt in candidates:
        if len(accepted) >= n_points:
            break

        cell = tuple(np.floor(pt * inv_cell).astype(np.int64).tolist())
        too_close = False
        for dx in range(-1, 2):
            if too_close:
                break
            for dy in range(-1, 2):
                if too_close:
                    break
                for dz in range(-1, 2):
                    for other in occupied.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                        if float(np.sum((pt - other) ** 2)) < min_dist2:
                            too_close = True
                            break
                    if too_close:
                        break

        if not too_close:
            accepted.append(pt)
            occupied.setdefault(cell, []).append(pt)

    if not accepted:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(accepted, dtype=np.float32).reshape(-1, 3)


def _voxel_fill_candidates(
    mesh: trimesh.Trimesh, pitch: float, dilate_iters: int = 5
) -> np.ndarray:
    """Return float32 centers from a filled voxel volume as ``[M,3]``.

    Direct filling must exceed the surface voxel count by 20%; otherwise dilation, filling,
    and equal erosion provide a fallback. Returns ``[0,3]`` when neither result passes.
    """
    pitch = _validate_extent("pitch", pitch)
    vox = mesh.voxelized(pitch)
    n_surface = int(vox.points.shape[0])

    filled = vox.fill()
    if filled.points.shape[0] > n_surface * 1.2:
        return filled.points.astype(np.float32)

    matrix = np.pad(vox.matrix, dilate_iters)
    dilated = ndimage.binary_dilation(matrix, iterations=dilate_iters)
    filled_arr = ndimage.binary_fill_holes(dilated)
    interior = ndimage.binary_erosion(filled_arr, iterations=dilate_iters)

    if int(interior.sum()) > n_surface * 1.2:
        ijk = (np.argwhere(interior) - dilate_iters).astype(np.float32)
        origin = np.array(vox.transform[:3, 3], dtype=np.float32)
        scale = np.asarray(vox.pitch, dtype=np.float32).ravel()
        return (origin + ijk * scale).astype(np.float32)

    return np.empty((0, 3), dtype=np.float32)


def sample_volume(
    mesh: trimesh.Trimesh,
    n_points: int,
    pitch: float | None = None,
    min_dist: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Sample at most ``n_points`` centers from a normalized mesh interior.

    Defaults use ``r=(3V/4 pi N)^(1/3)``, ``min_dist=1.2r``, and ``pitch=0.4*min_dist``.
    ``seed`` fixes candidate order. Returns float32 ``[M,3]``, ``M <= n_points``; an empty
    fill returns ``[0,3]``. This function does not relax spacing to reach the requested count.

    Examples:
        >>> mesh = trimesh.creation.box(extents=(1, 1, 1))
        >>> pts = sample_volume(mesh, n_points=3, min_dist=0.4, pitch=0.2, seed=0)
        >>> print(pts.shape)
        (3, 3)
        >>> print(pts.astype(float).round(1).tolist())
        [[0.0, 0.2, 0.2], [0.4, 0.4, 0.0], [-0.2, 0.0, -0.2]]
    """
    n_points = _validate_n_points(n_points)
    if min_dist is None:
        radius = _radius_from_mesh(mesh, n_points)
        min_dist = _connected_poisson_min_dist(radius)
    else:
        min_dist = _validate_extent("min_dist", min_dist)

    pitch = _pitch_from_min_dist(min_dist) if pitch is None else _validate_extent("pitch", pitch)

    candidates = _voxel_fill_candidates(mesh, pitch)
    if len(candidates) == 0:
        return np.empty((0, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)
    return _poisson_disk_subsample(candidates, min_dist, n_points, rng)


def _sample_volume_exact(
    mesh: trimesh.Trimesh,
    n_points: int,
    radius: float,
    seed: int,
    label: str,
) -> tuple[np.ndarray, float]:
    """Relax radius and spacing until exactly ``n_points`` fit.

    Factors are tried largest first. Returns float32 ``points[n_points,3]`` and the first
    successful effective radius. Raises ``RuntimeError`` with the best count if all factors fail.
    """
    n_points = _validate_n_points(n_points)
    radius = _validate_extent("radius", radius)

    best = np.empty((0, 3), dtype=np.float32)
    # Largest viable factor preserves the widest exclusion spacing.
    for factor in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        effective_radius = radius * factor
        min_dist = _connected_poisson_min_dist(effective_radius)
        pts = sample_volume(
            mesh,
            n_points,
            pitch=_pitch_from_min_dist(min_dist),
            min_dist=min_dist,
            seed=seed,
        )
        if len(pts) > len(best):
            best = pts
        if len(pts) == n_points:
            return pts.astype(np.float32, copy=False), effective_radius

    raise RuntimeError(
        f"Could only sample {len(best)} of {n_points} points for {label}. "
        "Try reducing n_points or using a larger mesh extent."
    )


def _resolve_scalar_or_sequence_extents(
    target_extent: float | Sequence[float] | None,
    n_targets: int,
) -> tuple[list[float], bool]:
    """Resolve one extent per target and report whether the caller supplied a sequence."""
    if target_extent is None:
        return [10.0] * n_targets, False
    if isinstance(target_extent, int | float | np.integer | np.floating):
        extent = _validate_extent("target_extent", target_extent)
        return [extent] * n_targets, False
    if isinstance(target_extent, Sequence) and not isinstance(target_extent, str | bytes):
        extents = [_validate_extent("target_extent", extent) for extent in target_extent]
        if len(extents) != n_targets:
            raise ValueError(
                f"target_extent sequence length {len(extents)} does not match "
                f"number of target_specs ({n_targets})."
            )
        return extents, True
    raise TypeError(
        "target_extent must be a positive float, a sequence of positive floats, or None; "
        f"got {type(target_extent).__name__}."
    )


def sample_mesh_pair(
    source_path: str | Path,
    target_path: str | Path,
    n_points: int = 2000,
    target_extent: float | None = 10.0,
    *,
    source_extent: float | None = None,
    max_particles: int | None = None,
    seed: int = 0,
) -> dict:
    r"""Sample independent, equal-count source and target clouds.

    Each repaired mesh is normalized and starts from its own volume-derived radius:

    .. math::

        r = \left(\frac{3V}{4\pi N}\right)^{1/3},
        \qquad d_{\min} = 1.2\,r,

    Exact-count sampling may lower ``r`` and ``d_min`` together; returned radius fields hold
    the effective values. Source and target use ``seed`` and ``seed+1``. Their reproducible
    samples are not paired correspondences. ``target_extent=None`` means 10;
    ``source_extent`` defaults to the target extent. Recorded ``max_particles`` is never below
    ``n_points``.

    Returns:
        - ``"source_pos"``: source agent centers, float32, shape ``[N, 3]``.
        - ``"target_pos"``: target agent centers, float32, shape ``[N, 3]``.
        - ``"radius"``: effective source radius (alias of ``"source_radius"``).
        - ``"R_init"``: initial uniform radius for the rollout (equals the source radius).
        - ``"n_points"``: requested count ``N``.
        - ``"n_source"``: number of source points (equals ``N``).
        - ``"n_target"``: number of target points (equals ``N``).
        - ``"max_particles"``: recorded capacity, ``max(max_particles, n_points)``.
        - ``"source_extent"``: source normalization extent (float).
        - ``"target_extent"``: target normalization extent (float).
        - ``"source_radius"``: effective source particle radius (float).
        - ``"target_radius"``: effective target particle radius (float).
        - ``"target_radii"``: alias of ``"target_radius"`` for sequence-API symmetry (float).
        - ``"target_mesh"``: the normalized target :class:`trimesh.Trimesh`.

    Exact-count failure raises ``RuntimeError``; invalid counts, extents, or volume raise
    ``ValueError``.
    """
    n_points = _validate_n_points(n_points)

    target_extent_f = _validate_extent(
        "target_extent", 10.0 if target_extent is None else target_extent
    )
    source_extent_f = (
        target_extent_f
        if source_extent is None
        else _validate_extent("source_extent", source_extent)
    )

    source_mesh = normalize_mesh(load_mesh(source_path), source_extent_f)
    target_mesh = normalize_mesh(load_mesh(target_path), target_extent_f)

    source_pts, source_radius = _sample_volume_exact(
        source_mesh, n_points, _radius_from_mesh(source_mesh, n_points), seed, "source mesh"
    )
    target_pts, target_radius = _sample_volume_exact(
        target_mesh,
        n_points,
        _radius_from_mesh(target_mesh, n_points),
        seed + 1,
        "target mesh",
    )
    max_particles_f = n_points if max_particles is None else max(int(max_particles), n_points)

    return {
        "source_pos": source_pts,
        "target_pos": target_pts,
        "radius": source_radius,
        "R_init": source_radius,
        "n_points": n_points,
        "n_source": n_points,
        "n_target": n_points,
        "max_particles": max_particles_f,
        "source_extent": source_extent_f,
        "target_extent": target_extent_f,
        "source_radius": source_radius,
        "target_radius": target_radius,
        "target_radii": target_radius,
        "target_mesh": target_mesh,
    }


def sample_mesh_sequence(
    source_path: str | Path,
    target_specs: list[tuple[int, str | Path]],
    n_points: int = 2000,
    *,
    target_extent: float | Sequence[float] | None = 10.0,
    source_extent: float | None = None,
    seed: int = 0,
) -> dict:
    r"""Sample a source cloud and frame-tagged target clouds at exact equal counts.

    Each repaired, normalized mesh starts from its own volume-derived radius and spacing:

    .. math::

        r = \left(\frac{3V}{4\pi N}\right)^{1/3},
        \qquad d_{\min} = 1.2\,r.

    Exact-count sampling may lower each ``r`` and ``d_min`` together; returned radius fields
    hold the effective values. Frames are unique, nonnegative, and sorted in the result;
    trainers interpret them as zero-based post-update rollout steps. Extent sequences match
    input ``target_specs`` before sorting. A scalar or ``None`` extent applies to every target,
    with ``None`` meaning 10;
    per-target extents require ``source_extent``. The source uses ``seed`` and sorted target
    ``k`` uses ``seed+1+k``, producing independent reproducible clouds.

    Returns:
        - ``"source_pos"``: source agent centers, float32, shape ``[N, 3]``.
        - ``"targets"``: list of per-target dicts sorted by frame, each with
          ``"frame"`` (int rollout step), ``"pos"`` (float32 ``[N, 3]`` centers),
          ``"mesh"`` (normalized :class:`trimesh.Trimesh`), ``"extent"`` (float),
          and ``"radius"`` (that target's effective particle radius, float).
        - ``"radius"``: effective source radius (alias of ``"source_radius"``).
        - ``"R_init"``: initial uniform radius for the rollout (equals the source radius).
        - ``"n_points"``: requested count ``N``.
        - ``"source_extent"``: source normalization extent (float).
        - ``"target_extent"``: per-target extent list if a sequence was given, else the single
          shared extent (float); ordered by sorted frame.
        - ``"target_extents"``: list of per-target extents in sorted-frame order.
        - ``"source_radius"``: effective source particle radius (float).
        - ``"target_radii"``: effective target radii in sorted-frame order.

    Invalid specs, frames, extents, or counts raise ``ValueError``. Exact-count sampling
    failure raises ``RuntimeError``.
    """
    n_points = _validate_n_points(n_points)
    if len(target_specs) < 1:
        raise ValueError("sample_mesh_sequence requires at least one (frame, path) target.")

    per_target_extent, target_extent_is_sequence = _resolve_scalar_or_sequence_extents(
        target_extent, len(target_specs)
    )
    if target_extent_is_sequence and source_extent is None:
        raise ValueError("source_extent is required when target_extent is a per-target sequence.")

    seen_frames: set[int] = set()
    triples: list[tuple[int, Path, float]] = []
    for (frame, path), extent in zip(target_specs, per_target_extent, strict=True):
        frame_int = int(frame)
        if frame_int < 0:
            raise ValueError(f"Target frame must be non-negative, got {frame_int}.")
        if frame_int in seen_frames:
            raise ValueError(f"Duplicate target frame {frame_int} in target_specs.")
        seen_frames.add(frame_int)
        triples.append((frame_int, Path(path), extent))
    triples.sort(key=lambda item: item[0])

    source_extent_f = (
        per_target_extent[0]
        if source_extent is None
        else _validate_extent("source_extent", source_extent)
    )
    source_mesh = normalize_mesh(load_mesh(source_path), source_extent_f)
    source_pts, source_radius = _sample_volume_exact(
        source_mesh, n_points, _radius_from_mesh(source_mesh, n_points), seed, "source mesh"
    )

    targets: list[dict] = []
    target_extents_sorted: list[float] = []
    target_radii: list[float] = []
    for k, (frame, path, extent) in tqdm(enumerate(triples), total=len(triples)):
        target_mesh = normalize_mesh(load_mesh(path), extent)
        target_pts, target_radius = _sample_volume_exact(
            target_mesh,
            n_points,
            _radius_from_mesh(target_mesh, n_points),
            seed + 1 + k,
            f"target frame {frame}",
        )
        targets.append(
            {
                "frame": frame,
                "pos": target_pts,
                "mesh": target_mesh,
                "extent": extent,
                "radius": target_radius,
            }
        )
        target_extents_sorted.append(extent)
        target_radii.append(target_radius)

    target_extent_repr: float | list[float]
    if target_extent_is_sequence:
        target_extent_repr = list(target_extents_sorted)
    else:
        target_extent_repr = target_extents_sorted[0]

    return {
        "source_pos": source_pts,
        "targets": targets,
        "radius": source_radius,
        "R_init": source_radius,
        "n_points": n_points,
        "source_extent": source_extent_f,
        "target_extent": target_extent_repr,
        "target_extents": target_extents_sorted,
        "source_radius": source_radius,
        "target_radii": target_radii,
    }
