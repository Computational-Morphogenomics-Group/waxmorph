"""Tests for waxmorph package-level exports."""

import os
import re
import subprocess
import sys
from pathlib import Path

import waxmorph

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _toml_string_array(text: str, key: str) -> set[str]:
    match = re.search(rf"(?ms)^{re.escape(key)}\s*=\s*\[(.*?)^\]", text)
    assert match is not None, f"missing TOML array {key!r}"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


class TestPackage:
    def test_version_is_string(self):
        assert isinstance(waxmorph.__version__, str)

    def test_all_exports_importable(self):
        for name in waxmorph.__all__:
            assert hasattr(waxmorph, name), f"{name} listed in __all__ but not importable"

    def test_key_classes_exported(self):
        assert hasattr(waxmorph, "GNS")
        assert hasattr(waxmorph, "GraphNetworkBlock")
        assert hasattr(waxmorph, "MLP")

    def test_key_functions_exported(self):
        assert callable(waxmorph.build_graph)
        assert callable(waxmorph.build_edge_index)
        assert callable(waxmorph.build_node_features)
        assert callable(waxmorph.build_edge_features)
        assert callable(waxmorph.chamfer_distance)
        assert callable(waxmorph.squared_loss)

    def test_import_does_not_clear_warp_caches(self):
        code = """
import warp as wp

def forbidden(*args, **kwargs):
    raise AssertionError("waxmorph import cleared a Warp cache")

wp.clear_kernel_cache = forbidden
wp.clear_lto_cache = forbidden
import waxmorph
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    def test_import_with_read_only_home(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        home.chmod(0o555)
        env = os.environ.copy()
        env.pop("WARP_CACHE_PATH", None)
        env.pop("XDG_CACHE_HOME", None)
        env["HOME"] = str(home)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_REPO_ROOT), value] if (value := env.get("PYTHONPATH")) else [str(_REPO_ROOT)]
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", "import waxmorph"],
                cwd=_REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            home.chmod(0o755)
        assert completed.returncode == 0, completed.stderr

    def test_eager_import_dependencies_are_core_requirements(self):
        text = (_REPO_ROOT / "pyproject.toml").read_text()
        assert _toml_string_array(text, "dependencies") == {
            "numpy>=1.24",
            "scipy>=1.11",
            "torch>=2",
            "tqdm>=4.66",
            "warp-lang>=1.10",
        }

    def test_simulation_extra_includes_usd_without_core_duplicates(self):
        text = (_REPO_ROOT / "pyproject.toml").read_text()
        simulation = _toml_string_array(text, "optional-dependencies.simulation")
        learning = _toml_string_array(text, "optional-dependencies.learning")
        assert "usd-core" in simulation
        assert simulation.isdisjoint({"scipy>=1.11", "tqdm>=4.66", "warp-lang>=1.10"})
        assert "torch>=2" not in learning
