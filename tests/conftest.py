"""Shared fixtures for WaxMorph tests."""

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    import warp as wp

    _WARP_CACHE_DIR = Path("/tmp/waxmorph-warp-cache")
    _WARP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(_WARP_CACHE_DIR)
except Exception:  # pragma: no cover - Warp is an optional runtime dependency.
    pass
