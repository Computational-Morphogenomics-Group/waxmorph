"""Mesh loading, normalization, and volumetric point sampling utilities."""

from pathlib import Path

import numpy as np
import trimesh


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


def normalize_mesh(
    mesh: trimesh.Trimesh, target_extent: float = 10.0
) -> trimesh.Trimesh:
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
    verts = mesh.vertices
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


def sample_volume(
    mesh: trimesh.Trimesh,
    n_points: int,
    pitch: float | None = None,
    min_dist: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Sample approximately *n_points* from the interior volume of a mesh.

    Uses ``mesh.contains()`` (ray-based containment test) to identify
    interior points, which works reliably for both watertight and
    non-watertight meshes.  Then applies Poisson disk subsampling for
    spatially uniform coverage.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (should be normalized first).
    n_points : int
        Desired number of output points.
    pitch : float, optional
        Grid spacing for candidate generation.  If ``None``, automatically
        chosen so that the grid is dense enough to yield many more
        candidates than *n_points*.
    min_dist : float, optional
        Minimum distance between accepted points.  If ``None``, estimated
        from the interior volume and *n_points*.
    seed : int
        Random seed.

    Returns
    -------
    points : np.ndarray, shape ``[<=n_points, 3]``, dtype ``float32``
    """
    bbox_min = mesh.vertices.min(axis=0)
    extent = mesh.vertices.max(axis=0) - bbox_min
    bbox_vol = extent.prod()

    if pitch is None:
        pitch = float((bbox_vol / (10.0 * n_points)) ** (1.0 / 3.0))

    grid = _make_grid(bbox_min, extent, pitch)
    inside = mesh.contains(grid)
    candidates = grid[inside]

    if min_dist is None:
        interior_vol = len(candidates) * (pitch**3)
        min_dist = float((interior_vol / n_points) ** (1.0 / 3.0)) * 0.75
        # Ensure pitch is fine enough relative to min_dist.
        if pitch > min_dist * 0.6:
            pitch = min_dist * 0.4
            grid = _make_grid(bbox_min, extent, pitch)
            inside = mesh.contains(grid)
            candidates = grid[inside]

    rng = np.random.default_rng(seed)
    pts = _poisson_disk_subsample(candidates, min_dist, n_points, rng)

    return pts


def sample_mesh_pair(
    source_path: str | Path,
    target_path: str | Path,
    n_points: int = 2000,
    target_extent: float = 10.0,
    radius: float = 0.2,
    seed: int = 0,
) -> dict:
    """Load two meshes, normalize, and sample matching point clouds.

    Returns a dict ready for use with Warp arrays and the GNS pipeline.

    Parameters
    ----------
    source_path, target_path : str or Path
        Paths to the source and target mesh files.
    n_points : int
        Number of points to sample from each mesh volume.
    target_extent : float
        Bounding-box normalization scale.
    radius : float
        Uniform sphere radius for all particles.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        ``source_pos``, ``target_pos`` : ``[n_points, 3]`` float32
        ``radius`` : float
        ``n_points`` : int
    """
    source_mesh = normalize_mesh(load_mesh(source_path), target_extent)
    target_mesh = normalize_mesh(load_mesh(target_path), target_extent)

    source_pts = sample_volume(source_mesh, n_points, seed=seed)
    target_pts = sample_volume(target_mesh, n_points, seed=seed + 1)

    return {
        "source_pos": source_pts,
        "target_pos": target_pts,
        "radius": radius,
        "n_points": n_points,
    }
