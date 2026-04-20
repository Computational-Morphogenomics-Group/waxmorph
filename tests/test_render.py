"""Tests for waxmorph.render module."""

import numpy as np

# Must set headless before pyglet is imported by warp.render
import pyglet
import pytest

pyglet.options["headless"] = True

import warp as wp

wp.init()

from waxmorph import render

DEVICE = "cpu"

try:
    if wp.is_device_available("cuda"):
        DEVICE = "cuda"
except RuntimeError:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_state():
    """10 particles with simple geometry for render tests."""
    rng = np.random.default_rng(99)
    n = 10
    centers = rng.standard_normal((n, 3)).astype(np.float32)
    radii = np.full(n, 0.3, dtype=np.float32)
    morphogens = rng.random(n).astype(np.float32)
    polarities = rng.standard_normal((n, 3)).astype(np.float32)
    polarities /= np.linalg.norm(polarities, axis=1, keepdims=True) + 1e-9
    cell_types = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=np.uint32)
    return dict(
        centers=centers,
        radii=radii,
        morphogens=morphogens,
        polarities=polarities,
        cell_types=cell_types,
        n=n,
    )


# ---------------------------------------------------------------------------
# MPLInterface
# ---------------------------------------------------------------------------


class TestMPLInterface:
    def test_draw_3d_view_returns_figure(self, small_state):
        """MPLInterface.draw_3d_view should return a matplotlib Figure."""
        import matplotlib

        matplotlib.use("Agg")

        fig = render.MPLInterface.draw_3d_view(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"],
            particle_count=small_state["n"],
            blim=-5,
            tlim=5,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# PyVistaInterface helpers
# ---------------------------------------------------------------------------


class TestPyVistaHelpers:
    def test_husl_palette_shape(self):
        pal = render.PyVistaInterface._husl_palette(5)
        assert pal.shape == (5, 3)
        assert pal.min() >= 0.0
        assert pal.max() <= 1.0

    def test_husl_palette_single(self):
        pal = render.PyVistaInterface._husl_palette(1)
        assert pal.shape == (1, 3)

    def test_rgb_from_categories(self):
        cats = np.array([0, 1, 0, 2, 1], dtype=np.uint32)
        rgb = render.PyVistaInterface._rgb_from_categories(cats)
        assert rgb.shape == (5, 3)
        assert rgb.dtype == np.uint8
        # Same category should get same color
        np.testing.assert_array_equal(rgb[0], rgb[2])
        np.testing.assert_array_equal(rgb[1], rgb[4])

    def test_rgb_from_morph(self):
        m = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        rgb = render.PyVistaInterface._rgb_from_morph(m)
        assert rgb.shape == (3, 3)
        assert rgb.dtype == np.uint8

    def test_rgb_from_morph_clamps(self):
        """Values outside [0,1] should be clamped."""
        m = np.array([-0.5, 1.5], dtype=np.float32)
        rgb = render.PyVistaInterface._rgb_from_morph(m)
        assert rgb.shape == (2, 3)
        # Clamped 0.0 and 1.0 should match direct calls
        expected_0 = render.PyVistaInterface._rgb_from_morph(np.array([0.0]))
        expected_1 = render.PyVistaInterface._rgb_from_morph(np.array([1.0]))
        np.testing.assert_array_equal(rgb[0], expected_0[0])
        np.testing.assert_array_equal(rgb[1], expected_1[0])

    def test_points_polydata_basic(self, small_state):
        pd = render.PyVistaInterface._points_polydata(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"],
            n=small_state["n"],
        )
        assert pd.n_points == small_state["n"]
        assert "radius" in pd.array_names
        assert "rgb" in pd.array_names

    def test_points_polydata_with_polarities(self, small_state):
        pd = render.PyVistaInterface._points_polydata(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"],
            polarities=small_state["polarities"],
            n=small_state["n"],
        )
        assert "polarity" in pd.array_names
        # Polarity vectors should be unit length
        pol = pd["polarity"]
        norms = np.linalg.norm(pol, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_points_polydata_with_cell_types(self, small_state):
        pd = render.PyVistaInterface._points_polydata(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"],
            n=small_state["n"],
            cell_types=small_state["cell_types"],
        )
        assert "cell_type" in pd.array_names

    def test_glyph_spheres(self, small_state):
        pd = render.PyVistaInterface._points_polydata(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"],
            n=small_state["n"],
        )
        glyphs = render.PyVistaInterface._glyph_spheres(pd)
        assert glyphs.n_points > 0
        assert glyphs.n_cells > 0

    def test_glyph_polarity_arrows(self, small_state):
        pd = render.PyVistaInterface._points_polydata(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"],
            polarities=small_state["polarities"],
            n=small_state["n"],
        )
        arrows = render.PyVistaInterface._glyph_polarity_arrows(pd, length=2.0)
        assert arrows.n_points > 0

    def test_glyph_polarity_arrows_missing_raises(self, small_state):
        pd = render.PyVistaInterface._points_polydata(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"],
            n=small_state["n"],
        )
        with pytest.raises(ValueError, match="polarity"):
            render.PyVistaInterface._glyph_polarity_arrows(pd)


# ---------------------------------------------------------------------------
# PyVistaInterface.draw_3d_view
# ---------------------------------------------------------------------------


class TestPyVistaDrawView:
    def test_draw_3d_view_returns_plotter(self, small_state):
        import pyvista as pv

        plotter = render.PyVistaInterface.draw_3d_view(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"],
            small_state["polarities"],
            particle_count=small_state["n"],
            blim=-5,
            tlim=5,
            show_polarities=False,
        )
        assert isinstance(plotter, pv.Plotter)
        plotter.close()

    def test_draw_3d_view_with_cell_types(self, small_state):
        plotter = render.PyVistaInterface.draw_3d_view(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"],
            small_state["polarities"],
            particle_count=small_state["n"],
            blim=-5,
            tlim=5,
            show_polarities=True,
            polarity_length=1.0,
            cell_types=small_state["cell_types"],
        )
        plotter.close()


# ---------------------------------------------------------------------------
# WarpMovieRenderer — _pack_buffers_kernel
# ---------------------------------------------------------------------------


class TestPackBuffers:
    def test_pack_buffers_basic(self):
        """Verify _pack_buffers_kernel copies positions/radii and produces valid RGB."""
        n = 5
        max_p = 8
        device = DEVICE

        centers = wp.from_numpy(
            np.random.randn(max_p, 3).astype(np.float32), dtype=wp.vec3f, device=device
        )
        radii = wp.from_numpy(
            np.full(max_p, 0.5, dtype=np.float32), dtype=wp.float32, device=device
        )
        A = wp.from_numpy(
            np.linspace(0, 1, max_p, dtype=np.float32), dtype=wp.float32, device=device
        )
        I_arr = wp.from_numpy(
            np.linspace(1, 0, max_p, dtype=np.float32), dtype=wp.float32, device=device
        )

        points_out = wp.empty(max_p, dtype=wp.vec3, device=device)
        radii_out = wp.empty(max_p, dtype=wp.float32, device=device)
        colors_out = wp.empty(max_p, dtype=wp.vec3, device=device)

        wp.launch(
            WarpMovieRenderer._pack_buffers_kernel,
            dim=max_p,
            inputs=[
                centers,
                radii,
                A,
                I_arr,
                points_out,
                radii_out,
                colors_out,
                n,
                0,
                1.0,
                0.8,
                0,
                0.0,
                0.0,
                0.0,
            ],
            device=device,
        )

        pts_np = points_out.numpy()
        rad_np = radii_out.numpy()
        col_np = colors_out.numpy()

        # Active particles should have non-zero radii
        assert np.all(rad_np[:n] > 0)
        # Inactive particles should be zeroed
        assert np.all(rad_np[n:] == 0)
        # Colors should be in [0, 1]
        assert col_np[:n].min() >= 0.0
        assert col_np[:n].max() <= 1.0

    def test_pack_buffers_morph_modes(self):
        """Test morph_mode 0 (A), 1 (I), 2 (ratio) produce different colors."""
        n = 3
        device = DEVICE

        centers = wp.from_numpy(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3f, device=device)
        radii = wp.from_numpy(np.full(n, 0.5, dtype=np.float32), dtype=wp.float32, device=device)
        A = wp.from_numpy(
            np.array([0.8, 0.2, 0.5], dtype=np.float32), dtype=wp.float32, device=device
        )
        I_arr = wp.from_numpy(
            np.array([0.2, 0.8, 0.5], dtype=np.float32), dtype=wp.float32, device=device
        )

        results = {}
        for mode_idx, mode_name in enumerate(["A", "I", "ratio"]):
            colors_out = wp.empty(n, dtype=wp.vec3, device=device)
            points_out = wp.empty(n, dtype=wp.vec3, device=device)
            radii_out = wp.empty(n, dtype=wp.float32, device=device)

            wp.launch(
                WarpMovieRenderer._pack_buffers_kernel,
                dim=n,
                inputs=[
                    centers,
                    radii,
                    A,
                    I_arr,
                    points_out,
                    radii_out,
                    colors_out,
                    n,
                    mode_idx,
                    1.0,
                    0.5,
                    0,
                    0.0,
                    0.0,
                    0.0,
                ],
                device=device,
            )
            results[mode_name] = colors_out.numpy().copy()

        # Mode A and I should produce different colors for asymmetric inputs
        assert not np.allclose(results["A"], results["I"])


WarpMovieRenderer = render.WarpMovieRenderer
