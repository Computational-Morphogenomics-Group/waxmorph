"""Mesh loading, normalization, and volumetric point sampling utilities."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from tqdm import tqdm

_PACKING_FRACTION = 1.0
_CONTACT_SPACING_RATIO = 1.2
_UNIT_SPHERE_VOLUME = 4.0 * np.pi / 3.0


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file and apply basic repairs.

    Args:
        path: Filesystem path accepted by :func:`trimesh.load`.

    Returns:
        Repaired :class:`trimesh.Trimesh` loaded with ``force="mesh"``.
    """
    mesh = trimesh.load(path, force="mesh")
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    return mesh


def _validate_n_points(n_points: int) -> int:
    """Return ``n_points`` as a positive integer."""
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}.")
    return n_points


def _validate_extent(name: str, extent: float) -> float:
    """Return ``extent`` as a positive finite float."""
    extent = float(extent)
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError(f"{name} must be a positive finite value, got {extent}.")
    return extent


def _mesh_volume(mesh: trimesh.Trimesh) -> float:
    """Return positive mesh volume, falling back to bounding-box volume."""
    volume = abs(float(mesh.volume))
    if np.isfinite(volume) and volume > 0:
        return volume

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    bbox_volume = float(np.prod(bounds[1] - bounds[0]))
    if np.isfinite(bbox_volume) and bbox_volume > 0:
        return bbox_volume
    raise ValueError("mesh must have positive volume or bounding-box volume.")


def _radius_from_volume(volume: float, n_points: int) -> float:
    """Choose a conservative non-overlap packing radius from occupied volume."""
    volume = _validate_extent("volume", volume)
    n_points = _validate_n_points(n_points)
    particle_volume = (_PACKING_FRACTION * volume) / float(n_points)
    return float(np.cbrt(particle_volume / _UNIT_SPHERE_VOLUME))


def _radius_from_mesh(mesh: trimesh.Trimesh, n_points: int) -> float:
    """Estimate a particle radius from mesh volume and particle count."""
    return _radius_from_volume(_mesh_volume(mesh), n_points)


def _connected_poisson_min_dist(radius: float) -> float:
    """Choose mildly overlapping center spacing for contact-graph edges."""
    radius = _validate_extent("radius", radius)
    return _CONTACT_SPACING_RATIO * radius


def _pitch_from_min_dist(min_dist: float) -> float:
    """Choose voxelization pitch used to generate candidate sample points."""
    min_dist = _validate_extent("min_dist", min_dist)
    return 0.4 * min_dist


def normalize_mesh(mesh: trimesh.Trimesh, target_extent: float = 10.0) -> trimesh.Trimesh:
    """Center a mesh at the origin and scale its largest extent.

    The input mesh is mutated in place.

    Args:
        mesh: Mesh whose vertex coordinates are normalized.
        target_extent: Desired length of the largest bounding-box side.

    Returns:
        The same mesh object after centering and scaling.

    Raises:
        ValueError: If ``target_extent`` or the mesh extent is not positive.
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
    """Greedily choose up to *n_points* candidates with Poisson-like spacing."""
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
    """Find volume candidate points via voxelization plus flood fill."""
    pitch = _validate_extent("pitch", pitch)
    vox = mesh.voxelized(pitch)
    n_surface = int(vox.points.shape[0])

    filled = vox.fill()
    if filled.points.shape[0] > n_surface * 1.2:
        return filled.points.astype(np.float32)

    matrix = vox.matrix.copy()
    dilated = ndimage.binary_dilation(matrix, iterations=dilate_iters)
    filled_arr = ndimage.binary_fill_holes(dilated)
    interior = ndimage.binary_erosion(filled_arr, iterations=dilate_iters)

    if int(interior.sum()) > n_surface * 1.2:
        ijk = np.argwhere(interior).astype(np.float32)
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
    """Sample volumetric mesh points with Poisson-like spacing.

    If ``min_dist`` is omitted, spacing is derived from the mesh's current
    normalized volume and ``n_points``. This function is allowed to return fewer
    points when a strict spacing cannot fit; pair/sequence helpers wrap it with
    exact-count validation.

    Args:
        mesh: Mesh whose filled interior is sampled.
        n_points: Maximum number of points to return.
        pitch: Optional voxel pitch. When omitted, it is derived from
            ``min_dist``.
        min_dist: Optional minimum accepted spacing between sampled points.
        seed: Seed for candidate-order randomization.

    Returns:
        Float32 point array with shape ``[M, 3]`` where ``M <= n_points``.

    Raises:
        ValueError: If counts, extents, spacing, or candidate shapes are invalid.
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
    """Sample exactly *n_points*, relaxing spacing only if the ideal radius cannot fit."""
    n_points = _validate_n_points(n_points)
    radius = _validate_extent("radius", radius)

    best = np.empty((0, 3), dtype=np.float32)
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
    """Normalize scalar or per-target extents for sequence sampling."""
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
    """Load a source and target mesh and sample matching volume point clouds.

    Radii are derived from extents and ``n_points``.

    Args:
        source_path: Path to the source mesh.
        target_path: Path to the target mesh.
        n_points: Exact number of source and target points to sample.
        target_extent: Target mesh normalization extent. ``None`` uses ``10.0``.
        source_extent: Optional source mesh normalization extent. Defaults to
            the target extent.
        max_particles: Optional particle capacity recorded in the returned
            metadata. Values below ``n_points`` are clamped up to ``n_points``.
        seed: Seed used for deterministic volume sampling.

    Returns:
        Dictionary containing source and target point clouds, radii, extents,
        counts, capacity metadata, and the normalized target mesh.

    Raises:
        RuntimeError: If exact-count sampling cannot place all requested points.
        ValueError: If counts, extents, or mesh volume are invalid.
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
    """Load one source mesh and frame-tagged target volume samples.

    Radii are derived from extents and ``n_points``.

    Args:
        source_path: Path to the source mesh.
        target_specs: ``(frame, path)`` pairs for target meshes. Frames must be
            non-negative and unique.
        n_points: Exact number of points sampled from every mesh.
        target_extent: Scalar target extent, per-target extent sequence, or
            ``None`` for the default extent.
        source_extent: Optional source extent. Required when ``target_extent``
            is a per-target sequence.
        seed: Base seed used for deterministic per-mesh sampling.

    Returns:
        Dictionary containing source points, sorted target records, radii, and
        extent metadata.

    Raises:
        ValueError: If target specs, frames, extents, or counts are invalid.
        RuntimeError: If exact-count sampling cannot place all requested points.
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
