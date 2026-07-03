r"""Turn a target-shape mesh into the initial agent positions a waxMorph rollout starts from.

The inverse-design emulator needs an initial cloud of ``N`` agent centers that fills the
volume bounded by a mesh, spaced like cells in tissue (touching but not interpenetrating).
This module loads/repairs/reorients/normalizes a mesh with trimesh, picks a uniform particle
radius so that ``N`` equal spheres tile the mesh volume, then dart-throws Poisson-disk samples
toward that spacing. The same routine samples every dataset volume -- both synthetic meshes
and segmented biological shapes -- with ``N = 2000`` agents.

Core sizing principle (see :func:`sample_mesh_pair`): for requested particle count ``N`` and
mesh volume ``V``, the uniform radius ``r`` is fixed by equating ``N`` cell volumes to ``V``,
giving ``r = (3V / 4 pi N)^(1/3)``; centers are kept at least the exclusion distance
``d_min = 1.2 r`` apart so neighboring agents overlap slightly and form a connected contact
graph. These are the sizing rules used for the inverse-design emulator across all datasets.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from tqdm import tqdm

# Fraction of mesh volume V assigned to particles when sizing the radius. At 1.0 the N cell
# volumes sum to exactly V, so r = (3V / 4 pi N)^(1/3).
_PACKING_FRACTION = 1.0
# Exclusion-distance-to-radius ratio: d_min = 1.2 r. Centers held >= 1.2 r apart overlap by
# ~0.8 r, giving touching-but-not-interpenetrating agents and a connected contact graph.
_CONTACT_SPACING_RATIO = 1.2
# Volume of a unit-radius sphere, 4 pi / 3; divides particle_volume to invert v = (4/3) pi r^3.
_UNIT_SPHERE_VOLUME = 4.0 * np.pi / 3.0


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a dataset mesh and repair it enough to define a closed, correctly oriented volume.

    Volumetric sampling needs a watertight, outward-facing surface so the mesh has a well-defined
    interior and a positive signed volume. ``force="mesh"`` collapses any scene/multi-part file
    into a single :class:`trimesh.Trimesh`; ``fix_normals`` reorients faces so normals point
    outward (correct sign of the volume) and ``fill_holes`` patches small gaps left by the scan
    or by reorientation. This is the load/repair/reorient step of initial-state sampling;
    downstream normalization is applied separately by :func:`normalize_mesh`.

    Args:
        path: Filesystem path to the mesh, in any format accepted by :func:`trimesh.load`.

    Returns:
        The loaded mesh with normals reoriented outward and small holes filled.
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
    """Return a usable positive volume ``V`` for radius sizing, robust to non-watertight meshes.

    The signed mesh volume is meaningless (zero or wrong sign) when repair leaves the surface
    non-watertight, so the absolute value is used and, if that is still non-positive or
    non-finite, the axis-aligned bounding-box volume is substituted as a coarse fallback.

    Args:
        mesh: Mesh whose enclosed volume drives the particle radius.

    Returns:
        A strictly positive, finite volume estimate in normalized mesh units.

    Raises:
        ValueError: If neither the mesh volume nor the bounding-box volume is positive and finite.
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
    r"""Size the uniform particle radius so ``n_points`` equal spheres tile the volume.

    Equate the total particle volume to the (packed) mesh volume and invert the sphere-volume
    formula:

    .. math::

        N \cdot \tfrac{4}{3}\pi r^3 = \phi V
        \quad\Longrightarrow\quad
        r = \left(\frac{3\,\phi V}{4\pi N}\right)^{1/3},

    where ``V`` is the mesh volume, ``N`` is ``n_points``, and ``phi`` is ``_PACKING_FRACTION``
    (1.0, so this reduces to ``r = (3V / 4 pi N)^(1/3)``).

    Args:
        volume: Enclosed mesh volume ``V`` in normalized units; must be positive and finite.
        n_points: Requested particle count ``N``.

    Returns:
        The uniform particle radius ``r`` in normalized mesh units.
    """
    volume = _validate_extent("volume", volume)
    n_points = _validate_n_points(n_points)
    particle_volume = (_PACKING_FRACTION * volume) / float(n_points)
    return float(np.cbrt(particle_volume / _UNIT_SPHERE_VOLUME))


def _radius_from_mesh(mesh: trimesh.Trimesh, n_points: int) -> float:
    """Compute the uniform particle radius ``r = (3V / 4 pi N)^(1/3)`` for a mesh.

    Reads the enclosed volume ``V`` via :func:`_mesh_volume` (with bounding-box fallback) and
    passes it to :func:`_radius_from_volume`.

    Args:
        mesh: Normalized mesh supplying the volume ``V``.
        n_points: Requested particle count ``N``.

    Returns:
        The uniform particle radius ``r`` in normalized mesh units.
    """
    return _radius_from_volume(_mesh_volume(mesh), n_points)


def _connected_poisson_min_dist(radius: float) -> float:
    """Convert a particle radius to the Poisson-disk exclusion distance ``d_min = 1.2 r``.

    The 1.2 factor (``_CONTACT_SPACING_RATIO``) is below the ``2 r`` no-overlap distance, so
    neighboring centers sit slightly inside one another's spheres. That deliberate mild overlap
    is what makes adjacent agents register as contacts and yields a connected interaction graph.

    Args:
        radius: Uniform particle radius ``r``; must be positive and finite.

    Returns:
        The minimum accepted center-to-center spacing ``d_min``.
    """
    radius = _validate_extent("radius", radius)
    return _CONTACT_SPACING_RATIO * radius


def _pitch_from_min_dist(min_dist: float) -> float:
    """Set the voxelization pitch to ``0.4 d_min`` for the candidate grid.

    Candidate sample points are voxel centers, so the pitch must be well below ``d_min`` for
    dart-throwing to have enough distinct candidates to reach the requested count after
    exclusion. The 0.4 factor gives several candidate voxels per exclusion sphere while keeping
    the voxel grid affordable.

    Args:
        min_dist: Target average spacing / exclusion distance ``d_min``; positive and finite.

    Returns:
        The voxel edge length used to discretize the mesh interior.
    """
    min_dist = _validate_extent("min_dist", min_dist)
    return 0.4 * min_dist


def normalize_mesh(mesh: trimesh.Trimesh, target_extent: float = 10.0) -> trimesh.Trimesh:
    """Put every dataset mesh on a common scale so one radius/spacing rule applies to all.

    Different source meshes arrive in arbitrary units and offsets; sampling and the derived
    radius ``r = (3V / 4 pi N)^(1/3)`` are only comparable across volumes once each mesh is
    centered and rescaled the same way. The bounding-box center is moved to the origin (removing
    translation) and all vertices are scaled isotropically so the longest bounding-box side
    equals ``target_extent`` (aspect ratio preserved). This is the reorient/normalize step of
    initial-state sampling. The input :class:`trimesh.Trimesh` is mutated in place.

    Args:
        mesh: Mesh whose vertex coordinates are recentered and rescaled in place.
        target_extent: Desired length of the longest bounding-box side after scaling, in the
            normalized units shared by every dataset mesh.

    Returns:
        The same mesh object, now centered at the origin with its longest side at
        ``target_extent``.

    Raises:
        ValueError: If ``target_extent`` is not positive, or the mesh has non-positive extent.

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
    """Dart-throwing Poisson-disk subsampling: accept candidates kept >= ``min_dist`` apart.

    Implements the acceptance loop of initial-state sampling: candidates are visited in
    random order (the permutation gives the "random order" thinning toward average
    spacing ``d_min``) and a candidate ``x_i`` is accepted iff no already-accepted center ``x_j``
    lies within ``min_dist``, i.e. ``min_j ||x_i - x_j||_2 > min_dist``. A uniform spatial-hash
    grid with cell size ``min_dist`` restricts each distance test to the 27 neighboring cells, so
    cost stays roughly linear in the number of candidates. Iteration stops once ``n_points`` are
    accepted, so the result holds at most ``n_points`` points and may hold fewer if the
    candidate pool runs out before the quota is met.

    Args:
        candidates: Candidate point positions, shape ``[M, 3]`` (interior voxel centers).
        min_dist: Exclusion distance ``d_min``; accepted centers stay strictly farther apart.
        n_points: Maximum number of points to accept.
        rng: Generator supplying the random visitation order (controls reproducibility).

    Returns:
        Accepted point positions, float32, shape ``[K, 3]`` with ``K <= n_points``; empty
        ``[0, 3]`` if no candidates are supplied or none are accepted.

    Raises:
        ValueError: If ``n_points`` or ``min_dist`` is invalid, or ``candidates`` is not ``[M, 3]``.
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
    """Generate interior candidate points by voxelizing the mesh and flood-filling its volume.

    Poisson-disk sampling needs candidates spread through the mesh *interior*, not just on its
    surface. The mesh is voxelized at ``pitch``, then the hollow shell is solidified: first via
    trimesh's own ``fill``, and if that fails (a leaky surface lets the flood fill escape), via a
    SciPy morphological close-then-erode that seals small gaps before filling. Candidates are the
    centers of the filled voxels, returned in world coordinates. This is the voxelization plus
    binary-dilation, hole-filling, and erosion path implemented with SciPy.

    Args:
        mesh: Watertight-ish, normalized mesh to fill.
        pitch: Voxel edge length (typically ``0.4 d_min``).
        dilate_iters: Number of binary-dilation iterations used to bridge surface gaps before
            hole filling, undone by the same number of erosion iterations so the interior keeps
            its original size; 5 closes typical scan-mesh leaks without inflating the volume.

    Returns:
        Interior candidate positions, float32, shape ``[M, 3]``; empty ``[0, 3]`` if neither
        fill path produces a solid interior.

    Raises:
        ValueError: If ``pitch`` is not positive and finite.
    """
    pitch = _validate_extent("pitch", pitch)
    vox = mesh.voxelized(pitch)
    n_surface = int(vox.points.shape[0])

    # Accept the cheap trimesh flood fill only if it actually solidified the shell: an interior
    # fill yields far more voxels than the surface alone, so require a >20% increase (x1.2).
    filled = vox.fill()
    if filled.points.shape[0] > n_surface * 1.2:
        return filled.points.astype(np.float32)

    # Fallback for leaky surfaces: close gaps with binary dilation, fill the now-sealed interior,
    # then erode by the same dilate_iters to restore the true boundary (morphological closing
    # of the hole-filled volume).
    matrix = np.pad(vox.matrix, dilate_iters)
    dilated = ndimage.binary_dilation(matrix, iterations=dilate_iters)
    filled_arr = ndimage.binary_fill_holes(dilated)
    interior = ndimage.binary_erosion(filled_arr, iterations=dilate_iters)

    # Same >20% (x1.2) interior-vs-surface guard before trusting the morphological fill.
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
    """Sample up to ``n_points`` interior agent centers from a mesh with Poisson-disk spacing.

    Ties together the three steps of initial-state sampling: size the spacing, build
    interior candidates, then dart-throw. When ``min_dist`` is omitted it is set from the
    mesh's normalized volume ``V`` and ``n_points`` via ``r = (3V / 4 pi N)^(1/3)`` and
    ``d_min = 1.2 r``; the voxel ``pitch`` defaults to ``0.4 d_min``. Candidates come from
    :func:`_voxel_fill_candidates` and are thinned by :func:`_poisson_disk_subsample` toward
    average spacing ``d_min``.

    This is the loose entry point: it returns fewer than ``n_points`` when the geometry cannot
    hold that many at the requested spacing. The exact-count wrapper :func:`_sample_volume_exact`
    (used by :func:`sample_mesh_pair` / :func:`sample_mesh_sequence`) retries with a relaxation
    ladder to hit the count.

    Args:
        mesh: Mesh whose filled interior is sampled (already normalized to its target extent).
        n_points: Maximum number of points ``N`` to return.
        pitch: Voxel edge length for candidate generation; when omitted, set to ``0.4 d_min``.
        min_dist: Exclusion distance ``d_min``; when omitted, set to ``1.2 r`` from volume sizing.
        seed: Seed for the random candidate-visitation order, for reproducible sampling.

    Returns:
        Interior point positions, float32, shape ``[M, 3]`` with ``M <= n_points``; empty
        ``[0, 3]`` if no interior candidates are found.

    Raises:
        ValueError: If counts, extents, spacing, or candidate shapes are invalid.

    See Also:
        :func:`_sample_volume_exact`: exact-count wrapper with the spacing-relaxation ladder.

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
    """Force exactly ``n_points`` samples, shrinking the spacing only as much as needed.

    A rollout needs a fixed agent count ``N``, but the ideal spacing ``d_min = 1.2 r`` from the
    volume-derived radius may not pack ``N`` points into a thin or concave mesh. This walks a
    relaxation ladder of shrink factors applied to the radius (and hence to ``d_min`` and the
    voxel pitch), trying the ideal spacing first and accepting the first factor that yields the
    full count. The largest spacing that succeeds is preferred, keeping agents as well-separated
    as the geometry allows. If even the smallest factor falls short it raises, reporting the best
    count reached.

    Args:
        mesh: Normalized mesh to sample.
        n_points: Exact number of points ``N`` required.
        radius: Ideal volume-derived particle radius ``r`` (the ladder's factor-1.0 spacing).
        seed: Seed for deterministic candidate ordering.
        label: Human-readable mesh label used in the failure message.

    Returns:
        Tuple ``(points, effective_radius)`` where ``points`` is float32 with shape
        ``[n_points, 3]`` and ``effective_radius`` is the (possibly shrunk) radius whose spacing
        achieved the count.

    Raises:
        RuntimeError: If no ladder factor places all ``n_points`` points.
        ValueError: If ``n_points`` or ``radius`` is invalid.
    """
    n_points = _validate_n_points(n_points)
    radius = _validate_extent("radius", radius)

    best = np.empty((0, 3), dtype=np.float32)
    # Spacing-relaxation ladder: multipliers on the ideal radius, ordered largest-first so the
    # loosest spacing that still packs all N points wins. Fine steps near 1.0 (0.95..0.8) trade
    # a little spacing for the count; coarse steps (0.7..0.2) are last-resort fallbacks for
    # awkward geometry. d_min and pitch shrink with the radius on each rung.
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
    """Broadcast a scalar/None extent or validate a per-target extent list for sequence sampling.

    :func:`sample_mesh_sequence` lets the caller normalize every target frame to one shared
    extent or to a distinct extent per frame. This resolves either form into one extent per
    target and reports which form was given (the boolean), so the caller can require an explicit
    source extent in the per-target case.

    Args:
        target_extent: A single positive extent applied to all targets, a sequence of positive
            extents (one per target, in ``target_specs`` order), or ``None`` for the default 10.0.
        n_targets: Number of target meshes the extents must cover.

    Returns:
        Tuple ``(extents, is_sequence)``: a list of length ``n_targets`` and a flag that is True
        only when a per-target sequence was supplied.

    Raises:
        ValueError: If a supplied sequence length does not match ``n_targets``.
        TypeError: If ``target_extent`` is not a positive float, such a sequence, or ``None``.
    """
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
    r"""Build the start/end agent clouds for a single-target deformation from two meshes.

    The inverse-design emulator deforms a source cloud into a target shape; this produces both as
    matched ``N``-agent point clouds sampled from the source and target meshes. Each mesh is
    loaded/repaired/reoriented (:func:`load_mesh`), normalized to its extent
    (:func:`normalize_mesh`), and volumetrically sampled. Each mesh gets its own uniform radius
    from its own volume ``V``:

    .. math::

        r = \left(\frac{3V}{4\pi N}\right)^{1/3},
        \qquad d_{\min} = 1.2\,r,

    with ``N`` = ``n_points`` and exclusion distance ``d_min`` enforcing Poisson-disk spacing
    (source and target radii differ when the meshes enclose different volumes). The source and
    target use seeds ``seed`` and ``seed + 1`` so their clouds are independent yet reproducible.

    Args:
        source_path: Path to the source (starting-shape) mesh.
        target_path: Path to the target (goal-shape) mesh.
        n_points: Exact number of points ``N`` sampled from each mesh.
        target_extent: Longest-side extent the target mesh is normalized to; ``None`` uses 10.0.
        source_extent: Longest-side extent for the source mesh; defaults to the target extent.
        max_particles: Particle-capacity value recorded in the metadata (e.g. a preallocation
            bound). Values below ``n_points`` are raised to ``n_points``; ``None`` uses
            ``n_points``.
        seed: Base seed; source uses ``seed`` and target uses ``seed + 1`` for sampling.

    Returns:
        Dictionary with the sampled clouds and their sizing metadata:

        - ``"source_pos"``: source agent centers, float32, shape ``[N, 3]``.
        - ``"target_pos"``: target agent centers, float32, shape ``[N, 3]``.
        - ``"radius"``: source particle radius ``r`` (alias of ``"source_radius"``).
        - ``"R_init"``: initial uniform radius for the rollout (equals the source radius).
        - ``"n_points"``: requested count ``N``.
        - ``"n_source"``: number of source points (equals ``N``).
        - ``"n_target"``: number of target points (equals ``N``).
        - ``"max_particles"``: recorded capacity, ``max(max_particles, n_points)``.
        - ``"source_extent"``: source normalization extent (float).
        - ``"target_extent"``: target normalization extent (float).
        - ``"source_radius"``: source particle radius ``r`` (float).
        - ``"target_radius"``: target particle radius ``r`` (float).
        - ``"target_radii"``: alias of ``"target_radius"`` for sequence-API symmetry (float).
        - ``"target_mesh"``: the normalized target :class:`trimesh.Trimesh`.

    Raises:
        RuntimeError: If exact-count sampling cannot place all ``n_points`` for either mesh.
        ValueError: If counts, extents, or mesh volume are invalid.

    See Also:
        :func:`sample_mesh_sequence`: same sampling for several frame-tagged intermediate targets.
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
    r"""Build the start cloud plus frame-tagged target clouds for a multi-keyframe deformation.

    A deformation can be steered through intermediate shapes by supplying several targets, each
    attached to the rollout step (frame) at which the cloud should match it. This samples one
    source cloud and one ``N``-agent cloud per target mesh, just like :func:`sample_mesh_pair`
    but over a sequence: every mesh is loaded/repaired/reoriented, normalized to its extent, and
    sampled with its own volume-derived radius

    .. math::

        r = \left(\frac{3V}{4\pi N}\right)^{1/3},
        \qquad d_{\min} = 1.2\,r.

    Targets are sorted by frame; frames must be non-negative and unique. Sampling seeds are
    ``seed`` for the source and ``seed + 1 + k`` for the ``k``-th (sorted) target, for
    independent yet reproducible clouds. Frame indices are interpreted by the trainer as
    zero-based rollout steps after updates.

    Args:
        source_path: Path to the source (starting-shape) mesh.
        target_specs: ``(frame, path)`` pairs, one per intermediate target; frames must be
            non-negative and unique (any input order; the result is sorted by frame).
        n_points: Exact number of points ``N`` sampled from every mesh.
        target_extent: A single extent shared by all targets, a sequence of extents (one per
            target, matched to ``target_specs`` order before sorting), or ``None`` for 10.0.
        source_extent: Source normalization extent; defaults to the first target's extent.
            Required when ``target_extent`` is a per-target sequence (no single default exists).
        seed: Base seed for deterministic per-mesh sampling, offset per target as above.

    Returns:
        Dictionary with the source cloud and the sorted target records:

        - ``"source_pos"``: source agent centers, float32, shape ``[N, 3]``.
        - ``"targets"``: list of per-target dicts sorted by frame, each with
          ``"frame"`` (int rollout step), ``"pos"`` (float32 ``[N, 3]`` centers),
          ``"mesh"`` (normalized :class:`trimesh.Trimesh`), ``"extent"`` (float),
          and ``"radius"`` (that target's particle radius ``r``, float).
        - ``"radius"``: source particle radius ``r`` (alias of ``"source_radius"``).
        - ``"R_init"``: initial uniform radius for the rollout (equals the source radius).
        - ``"n_points"``: requested count ``N``.
        - ``"source_extent"``: source normalization extent (float).
        - ``"target_extent"``: per-target extent list if a sequence was given, else the single
          shared extent (float); ordered by sorted frame.
        - ``"target_extents"``: list of per-target extents in sorted-frame order.
        - ``"source_radius"``: source particle radius ``r`` (float).
        - ``"target_radii"``: list of per-target particle radii in sorted-frame order.

    Raises:
        ValueError: If target specs, frames, extents, or counts are invalid (empty specs,
            negative or duplicate frame, mismatched extent-sequence length, or missing
            ``source_extent`` for a per-target sequence).
        RuntimeError: If exact-count sampling cannot place all ``n_points`` for any mesh.

    See Also:
        :func:`sample_mesh_pair`: the single-target counterpart.
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
