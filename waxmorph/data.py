"""Mesh loading, normalization, and volumetric point sampling utilities."""

from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh and apply basic repairs (normals, hole filling).

    Parameters
    ----------
    path : str or Path
        Path to mesh file (PLY, OBJ, STL, etc.).

    Returns
    -------
    mesh : trimesh.Trimesh
    """
    mesh = trimesh.load(path, force="mesh")
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, target_extent: float = 10.0) -> trimesh.Trimesh:
    """Center mesh at origin and scale so the largest bbox dimension equals *target_extent*.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to normalize (modified in-place and returned).
    target_extent : float
        Desired maximum bounding-box side length.

    Returns
    -------
    mesh : trimesh.Trimesh
    """
    verts = mesh.vertices.copy()
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    mesh.vertices -= center
    scale = target_extent / (verts.max(axis=0) - verts.min(axis=0)).max()
    mesh.vertices *= scale
    return mesh


def _poisson_disk_subsample(
    candidates: np.ndarray,
    min_dist: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Greedy Poisson disk subsampling from a set of candidate points.

    Shuffles candidates, then greedily accepts points that are at least
    *min_dist* away from all previously accepted points.  Uses a spatial
    hash grid for O(1) neighbor lookups.

    Parameters
    ----------
    candidates : np.ndarray ``[M, 3]``
        Dense set of candidate interior points.
    min_dist : float
        Minimum distance between any two accepted points.
    n_points : int
        Maximum number of points to accept.
    rng : np.random.Generator
        Random generator for shuffling.

    Returns
    -------
    points : np.ndarray ``[K, 3]`` where ``K <= n_points``
    """
    order = rng.permutation(len(candidates))
    candidates = candidates[order]

    cell_size = min_dist / np.sqrt(3.0)  # 3D diagonal of cell < min_dist
    inv_cell = 1.0 / cell_size

    occupied: dict[tuple[int, int, int], list[np.ndarray]] = {}
    accepted = []

    for pt in candidates:
        if len(accepted) >= n_points:
            break

        cx, cy, cz = (
            int(np.floor(pt[0] * inv_cell)),
            int(np.floor(pt[1] * inv_cell)),
            int(np.floor(pt[2] * inv_cell)),
        )

        # Check 3x3x3 neighborhood
        too_close = False
        for dx in range(-1, 2):
            if too_close:
                break
            for dy in range(-1, 2):
                if too_close:
                    break
                for dz in range(-1, 2):
                    key = (cx + dx, cy + dy, cz + dz)
                    if key in occupied:
                        for other in occupied[key]:
                            if np.sum((pt - other) ** 2) < min_dist * min_dist:
                                too_close = True
                                break

        if not too_close:
            accepted.append(pt)
            key = (cx, cy, cz)
            if key not in occupied:
                occupied[key] = []
            occupied[key].append(pt)

    return np.array(accepted, dtype=np.float32)


def _make_grid(bbox_min: np.ndarray, extent: np.ndarray, pitch: float) -> np.ndarray:
    """Generate a regular 3D grid of points within a bounding box.

    Returns
    -------
    grid : np.ndarray ``[M, 3]``
    """
    axes = [np.arange(bbox_min[d], bbox_min[d] + extent[d], pitch) for d in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)


def _voxel_fill_candidates(
    mesh: trimesh.Trimesh, pitch: float, dilate_iters: int = 5
) -> np.ndarray:
    """Find interior candidate points via voxelization + flood fill.

    For watertight meshes the standard ``fill()`` works directly.  For
    non-watertight meshes the surface voxels may have gaps that let the
    flood fill leak.  In that case a morphological **dilate → fill →
    erode** pipeline closes the gaps before filling.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    pitch : float
        Voxel edge length.
    dilate_iters : int
        Number of binary dilation iterations used to close gaps in the
        surface shell for non-watertight meshes.

    Returns
    -------
    candidates : np.ndarray ``[M, 3]``, dtype ``float32``
    """
    vox = mesh.voxelized(pitch)
    n_surface = vox.points.shape[0]

    # --- Fast path: plain flood fill (works for watertight meshes) ---
    filled = vox.fill()
    if filled.points.shape[0] > n_surface * 1.2:
        return filled.points.astype(np.float32)

    # --- Slow path: morphological close to seal holes, then fill ---
    matrix = vox.matrix.copy()
    dilated = ndimage.binary_dilation(matrix, iterations=dilate_iters)
    filled_arr = ndimage.binary_fill_holes(dilated)
    interior = ndimage.binary_erosion(filled_arr, iterations=dilate_iters)

    if interior.sum() > n_surface * 1.2:
        # Convert boolean voxel grid back to world-space points.
        ijk = np.argwhere(interior).astype(np.float32)
        origin = np.array(vox.transform[:3, 3], dtype=np.float32)
        scale = np.asarray(vox.pitch, dtype=np.float32).ravel()
        return (origin + ijk * scale).astype(np.float32)

    # Everything failed — return empty.
    return np.empty((0, 3), dtype=np.float32)


def sample_volume(
    mesh: trimesh.Trimesh,
    n_points: int,
    pitch: float | None = None,
    min_dist: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Sample approximately *n_points* from the interior volume of a mesh.

    Uses voxelization + flood fill to find interior points.  For
    non-watertight meshes a morphological dilate → fill → erode pipeline
    is applied automatically to close surface gaps before filling.  Then
    applies Poisson disk subsampling for spatially uniform coverage.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (should be normalized first).
    n_points : int
        Desired number of output points.
    pitch : float, optional
        Voxel edge length for interior detection.  If ``None``, automatically
        chosen so that the filled voxel grid is dense enough to yield at
        least ``4 * n_points`` candidates.
    min_dist : float, optional
        Minimum distance between accepted points.  If ``None``, estimated
        from the mesh volume and *n_points* so that the points pack
        roughly uniformly.
    seed : int
        Random seed.

    Returns
    -------
    points : np.ndarray, shape ``[<=n_points, 3]``, dtype ``float32``
    """
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    bbox_vol = extent.prod()

    auto_pitch = float((bbox_vol / (10.0 * n_points)) ** (1.0 / 3.0))
    if pitch is None:
        pitch = auto_pitch

    if min_dist is not None and pitch > float(min_dist) * 0.6:
        pitch = float(min_dist) * 0.4

    candidates = _voxel_fill_candidates(mesh, pitch)

    if min_dist is None:
        interior_vol = len(candidates) * (pitch**3) if len(candidates) > 0 else bbox_vol
        min_dist = float((interior_vol / n_points) ** (1.0 / 3.0)) * 0.75
        # Ensure pitch is fine enough relative to min_dist.
        if pitch > min_dist * 0.6:
            pitch = min_dist * 0.4
            candidates = _voxel_fill_candidates(mesh, pitch)

    if len(candidates) == 0:
        return np.empty((0, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)
    pts = _poisson_disk_subsample(candidates, min_dist, n_points, rng)

    return pts


def _connected_poisson_min_dist(radius: float) -> float:
    """Choose a dense spacing that keeps neighboring spheres visually connected."""
    return 1.6 * float(radius)


def sample_mesh_pair(
    source_path: str | Path,
    target_path: str | Path,
    n_points: int = 2000,
    target_extent: float | None = None,
    *,
    n_source: int | None = None,
    n_target: int | None = None,
    source_extent: float | None = None,
    max_particles: int | None = None,
    radius: float = 0.2,
    source_radius: float | None = None,
    target_radius: float | None = None,
    seed: int = 0,
) -> dict:
    """Load two meshes, normalize, and sample matching point clouds.

    Returns a dict ready for use with Warp arrays and the GNS pipeline.

    The function has two modes:
    - Legacy mode: when ``source_extent`` is omitted, preserve the latest
      symmetric shared-extent behavior.
    - Growing mode: when both ``source_extent`` and ``target_extent`` are
      provided, sample each mesh with dense radius-aware Poisson disk spacing
      and allow asymmetric counts / radii.

    Parameters
    ----------
    source_path, target_path : str or Path
        Paths to the source and target mesh files.
    n_points : int
        Legacy symmetric number of points to sample from each mesh volume.
    target_extent : float, optional
        Target normalization scale. If provided without ``source_extent``, the
        legacy shared-extent path is used.
    n_source, n_target : int, optional
        Asymmetric source and target point counts. If omitted, both fall back
        to ``n_points``.
    source_extent : float, optional
        Independent normalization extent for the source mesh. Must be provided
        together with ``target_extent`` to enable the growing-mode path.
    max_particles : int, optional
        Preallocation capacity for growing rollouts. Defaults to ``n_target``.
    radius : float
        Legacy uniform sphere radius for all particles. Also used as the
        fallback source/target radius in growing mode.
    source_radius, target_radius : float, optional
        Radius metadata used to choose a dense Poisson spacing in growing mode.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        ``source_pos`` : ``[n_source, 3]`` float32
        ``target_pos`` : ``[n_target, 3]`` float32
        ``radius`` : float
        ``R_init`` : float
        ``n_points`` : int
        ``n_source`` : int
        ``n_target`` : int
        ``max_particles`` : int
        ``target_mesh`` : normalized trimesh target mesh
        ``source_radius`` : float
        ``target_radius`` : float
        ``target_radii`` : float
    """
    use_growing_sampling = source_extent is not None and target_extent is not None

    if not use_growing_sampling:
        if source_extent is not None:
            raise ValueError("Growing-mode sampling requires both source_extent and target_extent.")
        if any(
            value is not None
            for value in (n_source, n_target, max_particles, source_radius, target_radius)
        ):
            raise ValueError(
                "Asymmetric counts, radius-aware sampling, and max_particles require "
                "both source_extent and target_extent."
            )

        legacy_extent = 10.0 if target_extent is None else float(target_extent)
        legacy_radius = float(radius)
        source_mesh = normalize_mesh(load_mesh(source_path), legacy_extent)
        target_mesh = normalize_mesh(load_mesh(target_path), legacy_extent)

        source_pts = sample_volume(source_mesh, n_points, seed=seed)
        target_pts = sample_volume(target_mesh, n_points, seed=seed + 1)

        return {
            "source_pos": source_pts,
            "target_pos": target_pts,
            "radius": legacy_radius,
            "R_init": legacy_radius,
            "n_points": n_points,
            "n_source": n_points,
            "n_target": n_points,
            "max_particles": n_points,
            "source_extent": legacy_extent,
            "target_extent": legacy_extent,
            "source_radius": legacy_radius,
            "target_radius": legacy_radius,
            "target_radii": legacy_radius,
            "target_mesh": target_mesh,
        }

    n_source = n_points if n_source is None else int(n_source)
    n_target = n_points if n_target is None else int(n_target)
    if n_source > n_target:
        raise ValueError("sample_mesh_pair() requires n_source <= n_target for growing rollouts.")

    source_extent = float(source_extent)
    target_extent = float(target_extent)
    source_radius = float(radius) if source_radius is None else float(source_radius)
    target_radius = float(radius) if target_radius is None else float(target_radius)

    source_mesh = normalize_mesh(load_mesh(source_path), source_extent)
    target_mesh = normalize_mesh(load_mesh(target_path), target_extent)

    source_pts = sample_volume(
        source_mesh,
        n_source,
        min_dist=_connected_poisson_min_dist(source_radius),
        seed=seed,
    )
    target_pts = sample_volume(
        target_mesh,
        n_target,
        min_dist=_connected_poisson_min_dist(target_radius),
        seed=seed + 1,
    )

    max_particles = n_target if max_particles is None else int(max_particles)
    max_particles = max(max_particles, n_target)

    return {
        "source_pos": source_pts,
        "target_pos": target_pts,
        "radius": source_radius,
        "R_init": source_radius,
        "n_points": n_points,
        "n_source": n_source,
        "n_target": n_target,
        "max_particles": max_particles,
        "source_extent": source_extent,
        "target_extent": target_extent,
        "source_radius": source_radius,
        "target_radius": target_radius,
        "target_radii": target_radius,
        "target_mesh": target_mesh,
    }
