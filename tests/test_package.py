"""Tests for waxmorph package-level exports."""

import waxmorph


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
