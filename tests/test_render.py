"""Tests for waxmorph.render module."""

import base64
import json

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

    def test_cell_types_ignore_short_morphogens(self, small_state):
        n = 7
        cell_types = small_state["cell_types"].copy()
        cell_types[n:] = 9
        expected_types = cell_types[:n].astype(np.int32)
        for morphogens in (small_state["morphogens"][:2], None, object()):
            pd = render.PyVistaInterface._points_polydata(
                small_state["centers"],
                small_state["radii"],
                morphogens,
                n=n,
                cell_types=cell_types,
            )

            assert pd.n_points == n
            assert pd["cell_type"].dtype == np.int32
            assert pd["rgb"].dtype == np.uint8
            np.testing.assert_array_equal(pd.points, small_state["centers"][:n])
            np.testing.assert_array_equal(pd["radius"], small_state["radii"][:n])
            np.testing.assert_array_equal(pd["cell_type"], expected_types)
            np.testing.assert_array_equal(
                pd["rgb"], render.PyVistaInterface._rgb_from_categories(expected_types)
            )

    def test_morphogens_limit_points_without_cell_types(self, small_state):
        pd = render.PyVistaInterface._points_polydata(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"][:3],
            n=small_state["n"],
        )

        assert pd.n_points == 3

    def test_cell_types_limit_points(self, small_state):
        cell_types = small_state["cell_types"][:4]
        pd = render.PyVistaInterface._points_polydata(
            small_state["centers"],
            small_state["radii"],
            small_state["morphogens"],
            n=small_state["n"],
            cell_types=cell_types,
        )

        assert pd.n_points == 4
        np.testing.assert_array_equal(pd["cell_type"], cell_types.astype(np.int32))

    def test_points_polydata_rejects_short_polarities(self, small_state):
        with pytest.raises(ValueError, match=r"polarities.*7"):
            render.PyVistaInterface._points_polydata(
                small_state["centers"],
                small_state["radii"],
                small_state["morphogens"],
                polarities=small_state["polarities"][:3],
                n=7,
            )

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
# K3DInterface
# ---------------------------------------------------------------------------


def _packed(rgb_u8):
    return rgb_u8.astype(np.uint32) @ np.array([1 << 16, 1 << 8, 1], dtype=np.uint32)


def _panel_data(code):
    return json.loads(code.rsplit("waxmorphControls(K3DInstance, ", 1)[1].split(");\n", 1)[0])


def _decode(text, dtype, shape):
    return np.frombuffer(base64.b64decode(text), dtype).reshape(shape)


def _k3d_view(state, **kwargs):
    return render.K3DInterface.draw_3d_view(
        state["centers"],
        state["radii"],
        state["morphogens"],
        state["polarities"],
        particle_count=kwargs.pop("particle_count", state["n"]),
        blim=-5,
        tlim=5,
        **kwargs,
    )


class TestK3DInterface:
    def test_pack_rgb(self):
        rgb = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2.0, -1.0, 0.5]]
        np.testing.assert_array_equal(
            render._pack_rgb(rgb), [0xFF0000, 0x00FF00, 0x0000FF, 0xFF0080]
        )

    def test_points_match_pyvista(self, small_state):
        plot = _k3d_view(small_state, particle_count=7, show_polarities=False)
        (points,) = plot.objects
        pd = render.PyVistaInterface._points_polydata(
            small_state["centers"], small_state["radii"], small_state["morphogens"], n=7
        )

        np.testing.assert_allclose(points.positions, pd.points, rtol=1e-6)
        np.testing.assert_allclose(points.point_sizes, 2 * pd["radius"], rtol=1e-6)
        np.testing.assert_array_equal(points.colors, _packed(pd["rgb"]))
        assert list(plot.grid) == [-5, -5, -5, 5, 5, 5]
        assert not plot.grid_auto_fit

    def test_cell_types_set_colors(self, small_state):
        plot = _k3d_view(small_state, show_polarities=False, cell_types=small_state["cell_types"])
        (points,) = plot.objects
        expected = render.PyVistaInterface._rgb_from_categories(small_state["cell_types"])
        np.testing.assert_array_equal(points.colors, _packed(expected))

    def test_polarity_arrows_are_centered(self, small_state):
        plot = _k3d_view(small_state, polarity_length=2.0, polarity_color="red")
        _, arrows = plot.objects

        np.testing.assert_allclose(
            arrows.origins + 0.5 * arrows.vectors, small_state["centers"], atol=1e-6
        )
        np.testing.assert_allclose(np.linalg.norm(arrows.vectors, axis=1), 2.0, rtol=1e-5)
        assert arrows.origin_color == arrows.head_color == 0xFF0000

    def test_panel_needs_genes(self, small_state):
        assert _k3d_view(small_state).additional_js_code == ""
        code = _k3d_view(small_state, c=np.ones((10, 2))).additional_js_code
        assert "function waxmorphControls" in code

    def test_panel_scales_each_gene_to_its_range(self, small_state):
        c = np.stack([np.arange(10.0), np.full(10, 3.0)], axis=1)
        data = _panel_data(_k3d_view(small_state, c=c).additional_js_code)
        levels = _decode(data["genes"]["levels"], np.uint8, (2, 1, 10))

        np.testing.assert_array_equal(levels[0, 0], np.round(np.arange(10) / 9 * 255))
        np.testing.assert_array_equal(levels[1, 0], 0)
        assert data["genes"]["ranges"] == [[0.0, 9.0], [3.0, 3.0]]
        assert data["keys"] is None

    def test_snapshot_is_self_contained(self, small_state):
        page = _k3d_view(small_state).get_snapshot()
        assert "<script src" not in page
        assert "unpkg.com" not in page

    def test_embed_escapes_page_into_iframe(self):
        out = render.K3DInterface.embed('<p class="k">a & b</p>', height=300)
        assert out.data == (
            '<div><iframe srcdoc="&lt;p class=&quot;k&quot;&gt;a &amp; b&lt;/p&gt;" '
            'style="width:100%;height:300px;border:0"></iframe></div>'
        )


# ---------------------------------------------------------------------------
# WarpMovieRenderer — K3D backend
# ---------------------------------------------------------------------------


class TestK3DMovie:
    def test_frames_pad_growing_counts(self, tmp_path):
        """Unborn slots wait at their next position and color with zero size."""
        path = tmp_path / "movie.html"
        centers = np.arange(12, dtype=np.float32).reshape(4, 3)
        with WarpMovieRenderer(str(path), 4, backend="k3d", fps=5, device=DEVICE) as movie:
            for t, shift, radius, rgb, count in (
                (0.0, 0, 0.5, [1, 0, 0], 2),
                (1.0, 1, 0.25, [0, 0, 1], 4),
            ):
                movie.write_frame_from_numpy(
                    t=t,
                    centers=centers + shift,
                    radii=np.full(4, radius),
                    colors=np.tile(rgb, (4, 1)),
                    particle_count=count,
                )
        plot = movie._backend.plot
        (points,) = plot.objects

        assert sorted(points.positions) == ["0.0", "1.0"]
        np.testing.assert_array_equal(
            points.positions["0.0"], np.concatenate([centers[:2], centers[2:] + 1])
        )
        np.testing.assert_array_equal(points.point_sizes["0.0"], [1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(
            points.colors["0.0"], [0xFF0000, 0xFF0000, 0x0000FF, 0x0000FF]
        )
        np.testing.assert_array_equal(points.positions["1.0"], centers + 1)
        np.testing.assert_array_equal(points.point_sizes["1.0"], np.full(4, 0.5))
        assert points.name == "cells"
        assert plot.fps == 5
        assert "<script src" not in path.read_text()

    def test_state_frames_use_device_colors(self, tmp_path):
        """Hue 0 with saturation A gives RGB (1, 1 - A, 1 - A)."""
        A = wp.from_numpy(np.array([0.0, 0.5, 1.0], np.float32), dtype=wp.float32, device=DEVICE)
        centers = wp.from_numpy(np.zeros((3, 3), np.float32), dtype=wp.vec3f, device=DEVICE)
        radii = wp.from_numpy(np.full(3, 0.5, np.float32), dtype=wp.float32, device=DEVICE)
        path = tmp_path / "movie.html"
        with WarpMovieRenderer(str(path), 3, backend="k3d", device=DEVICE) as movie:
            movie.write_frame_from_state(
                t=0.0, centers_wp=centers, radii_wp=radii, A_wp=A, I_wp=A, particle_count=3, hue=0.0
            )
        (points,) = movie._backend.plot.objects

        np.testing.assert_array_equal(points.colors["0.0"], [0xFFFFFF, 0xFF8080, 0xFF0000])

    def test_online_snapshot_loads_k3d_from_cdn(self, tmp_path):
        path = tmp_path / "movie.html"
        with WarpMovieRenderer(
            str(path), 2, backend="k3d", device=DEVICE, k3d_snapshot_type="online"
        ) as movie:
            movie.write_frame_from_numpy(
                t=0.0,
                centers=np.zeros((2, 3)),
                radii=np.ones(2),
                colors=np.ones((2, 3)),
                particle_count=2,
            )
        assert "unpkg.com/k3d@" in path.read_text()

    def test_frames_carry_genes(self, tmp_path):
        """Absent slots repeat their next concentrations; the panel is stored once, in the page."""
        path = tmp_path / "movie.html"
        c = np.arange(8.0).reshape(4, 2)
        with WarpMovieRenderer(str(path), 4, backend="k3d", device=DEVICE) as movie:
            for t, count in ((0.0, 2), (1.0, 4)):
                movie.write_frame_from_numpy(
                    t=t,
                    centers=np.zeros((4, 3)),
                    radii=np.ones(4),
                    colors=np.ones((4, 3)),
                    particle_count=count,
                    c=c + t,
                )
        data = _panel_data(path.read_text())
        levels = _decode(data["genes"]["levels"], np.uint8, (2, 2, 4))

        assert movie._backend.plot.additional_js_code == ""
        assert data["keys"] == ["0.0", "1.0"]
        np.testing.assert_array_equal(levels[0], [[0, 73, 182, 255], [36, 109, 182, 255]])
        assert data["genes"]["ranges"] == [[0.0, 7.0], [1.0, 8.0]]

    def test_usd_ignores_genes(self, tmp_path):
        path = tmp_path / "movie.usd"
        with WarpMovieRenderer(str(path), 2, backend="usd", device=DEVICE) as movie:
            movie.write_frame_from_numpy(
                t=0.0,
                centers=np.zeros((2, 3)),
                radii=np.ones(2),
                colors=np.ones((2, 3)),
                particle_count=2,
                c=np.ones((2, 4)),
            )
        assert path.exists()

    def test_close_without_frames_writes_page(self, tmp_path):
        path = tmp_path / "nested" / "empty.html"
        WarpMovieRenderer(str(path), 1, backend="k3d", device=DEVICE).close()
        assert path.read_text().startswith("<!doctype html>")


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
