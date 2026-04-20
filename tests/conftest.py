"""Shared fixtures for WaxMorph tests."""

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

try:
    import warp as wp

    _WARP_CACHE_DIR = Path("/tmp/waxmorph-warp-cache")
    _WARP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(_WARP_CACHE_DIR)
except Exception:  # pragma: no cover - Warp is an optional runtime dependency.
    wp = None


@pytest.fixture()
def particle_state():
    """Create a minimal particle state as numpy arrays.

    Returns a dict with positions, polarities, radii, cell_types, and genes
    for 50 particles preallocated to max_particles=100.
    """
    particle_count = 50
    max_particles = 100
    num_genes = 2

    rng = np.random.default_rng(42)

    positions = np.zeros((max_particles, 3), dtype=np.float32)
    positions[:particle_count] = rng.standard_normal((particle_count, 3)).astype(np.float32)

    radii = np.zeros(max_particles, dtype=np.float32)
    radii[:particle_count] = 0.5

    polarities = np.zeros((max_particles, 3), dtype=np.float32)
    norms = np.linalg.norm(positions[:particle_count], axis=-1, keepdims=True) + 1e-9
    polarities[:particle_count] = positions[:particle_count] / norms

    cell_types = np.zeros(max_particles, dtype=np.uint32)
    cell_types[:particle_count] = rng.integers(0, 2, size=particle_count).astype(np.uint32)

    genes = np.zeros((max_particles, num_genes), dtype=np.float32)
    genes[:particle_count] = rng.random((particle_count, num_genes)).astype(np.float32)

    return {
        "positions": positions,
        "radii": radii,
        "polarities": polarities,
        "cell_types": cell_types,
        "genes": genes,
        "particle_count": particle_count,
        "max_particles": max_particles,
        "num_genes": num_genes,
    }
