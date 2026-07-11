"""Renderer interfaces for displaying 3D spheroids with morphogens."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Literal

import imageio
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import warp.render as wpr
from matplotlib import colors

try:
    import pyvista as pv
except Exception:  # pragma: no cover - optional dependency for rendering backends.
    pv = None


def _require_pyvista() -> None:
    if pv is None:
        raise ImportError(
            "waxmorph.render requires optional dependency 'pyvista'. "
            "Install extras with: pip install -e .[simulation]"
        )


class RenderInterface(ABC):
    @staticmethod
    @abstractmethod
    def draw_sphere(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def cleanup(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def draw_3d_view(*args, **kwargs):
        pass


class MPLInterface(RenderInterface):
    """Static Matplotlib spheres colored by morphogen saturation."""

    @staticmethod
    def draw_sphere(
        ax,
        center,
        radius,
        alpha=0.35,
        facecolor="blue",
        edgecolor="red",
        theta_res=24,
        phi_res=12,
        antialiased=True,
    ):
        cx, cy, cz = center
        theta = np.linspace(0, 2 * np.pi, theta_res)
        phi = np.linspace(0, np.pi, phi_res)
        TH, PH = np.meshgrid(theta, phi)

        X = cx + radius * np.cos(TH) * np.sin(PH)
        Y = cy + radius * np.sin(TH) * np.sin(PH)
        Z = cz + radius * np.cos(PH)

        ax.plot_surface(
            X,
            Y,
            Z,
            rcount=phi_res,
            ccount=theta_res,
            color=facecolor,
            edgecolor=edgecolor,
            linewidth=0,
            antialiased=antialiased,
            alpha=alpha,
            shade=True,
        )

    @staticmethod
    def cleanup(ax, blim=-10, tlim=20):
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.fill = False
                axis.pane.set_edgecolor("none")
            except Exception:
                pass
        ax.set_facecolor("white")
        ax.tick_params(pad=4, labelsize=9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(blim, tlim)
        ax.set_ylim(blim, tlim)
        ax.set_zlim(blim, tlim)

    @staticmethod
    def draw_3d_view(centers, radii, morphogens, particle_count, blim=-10, tlim=20):
        """Render a static figure; ``particle_count`` must be nonnegative."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        for center, radius, morphogen in zip(
            centers[:particle_count],
            radii[:particle_count],
            morphogens[:particle_count],
            strict=False,
        ):
            MPLInterface.draw_sphere(
                ax,
                center=center,
                facecolor=colors.hsv_to_rgb((0.8, morphogen.item(), 1.0)),
                radius=radius,
                alpha=0.35,
            )

        MPLInterface.cleanup(ax, blim, tlim)

        return fig


class PyVistaInterface(RenderInterface):
    """Interactive PyVista glyphs with morphogen or categorical colors."""

    @staticmethod
    def _husl_palette(n_colors: int, *, s: float = 90.0, lightness: float = 65.0) -> np.ndarray:
        """Use HSLuv when installed, otherwise an evenly spaced HSV palette."""
        n_colors = int(max(1, n_colors))
        hues = np.linspace(0.0, 360.0, n_colors, endpoint=False)

        try:
            import hsluv  # type: ignore

            rgb = np.array(
                [hsluv.hsluv_to_rgb((float(h), float(s), float(lightness))) for h in hues],
                dtype=float,
            )
            return np.clip(rgb, 0.0, 1.0)
        except Exception:
            h01 = (hues / 360.0).astype(float)
            hsv = np.stack([h01, np.full_like(h01, 0.85), np.full_like(h01, 0.95)], axis=-1)
            return colors.hsv_to_rgb(hsv)

    @staticmethod
    def _rgb_from_categories(categories: np.ndarray) -> np.ndarray:
        """Map sorted-category ranks to uint8 colors; fixed category sets are deterministic."""
        cats = np.asarray(categories).reshape(-1)
        cats = cats.astype(np.int64, copy=False)

        uniq, inv = np.unique(cats, return_inverse=True)
        palette = PyVistaInterface._husl_palette(len(uniq))
        return (palette[inv] * 255).astype(np.uint8)

    @staticmethod
    def _rgb_from_morph(morphogens: np.ndarray) -> np.ndarray:
        m = np.asarray(morphogens).astype(float)
        m = np.clip(m, 0.0, 1.0)
        rgb_float = colors.hsv_to_rgb(
            np.stack([np.full(shape=m.shape, fill_value=0.08), m, np.ones_like(m)], axis=-1)
        )
        return (rgb_float * 255).astype(np.uint8)

    @staticmethod
    def _points_polydata(
        centers,
        radii,
        morphogens,
        polarities=None,
        n=None,
        *,
        cell_types: np.ndarray | None = None,
    ) -> pv.PolyData:
        """Build glyph data; categories override morphogens as the color source.

        The rendered count is bounded by the shortest selected center, radius,
        and color input. Polarities must have shape ``[N,3]`` with at least that
        many rows. Norms at least ``1e-12`` are normalized; smaller vectors use
        ``1e-12`` as the denominator.
        """
        c = np.asarray(centers, dtype=float)
        r = np.asarray(radii, dtype=float).reshape(-1)
        if cell_types is None:
            color_values = np.asarray(morphogens, dtype=float).reshape(-1)
        else:
            color_values = np.asarray(cell_types).reshape(-1)

        if n is None:
            n = len(c)
        n = min(n, len(c), len(r), len(color_values))

        pts = c[:n]
        rad = r[:n]

        _require_pyvista()

        if cell_types is not None:
            rgb = PyVistaInterface._rgb_from_categories(color_values[:n])
        else:
            rgb = PyVistaInterface._rgb_from_morph(color_values[:n])

        pd = pv.PolyData(pts)
        pd["radius"] = rad
        pd["rgb"] = rgb

        if cell_types is not None:
            pd["cell_type"] = color_values[:n].astype(np.int32, copy=False)

        if polarities is not None:
            p = np.asarray(polarities, dtype=float)
            if p.ndim != 2 or p.shape[1] != 3:
                raise ValueError(f"polarities must have shape (N,3); got {p.shape}")
            if len(p) < n:
                raise ValueError(f"polarities has {len(p)} rows but {n} points are rendered")
            p = p[:n]

            norms = np.linalg.norm(p, axis=1, keepdims=True)
            p = p / np.clip(norms, 1e-12, None)

            pd["polarity"] = p

        return pd

    @staticmethod
    def _glyph_spheres(points_pd: pv.PolyData, theta_res=24, phi_res=12) -> pv.PolyData:
        _require_pyvista()
        base = pv.Sphere(radius=1.0, theta_resolution=theta_res, phi_resolution=phi_res)
        return points_pd.glyph(geom=base, scale="radius", orient=False)

    @staticmethod
    def _glyph_polarity_arrows(
        points_pd: pv.PolyData,
        *,
        vector_name: str = "polarity",
        length: float = 1.0,
        shaft_radius: float = 0.03,
        tip_length: float = 0.25,
        tip_radius: float = 0.06,
    ) -> pv.PolyData:
        """Center an arrow defined on ``[0, 1]`` before orienting each glyph."""
        _require_pyvista()
        if vector_name not in points_pd.array_names:
            raise ValueError(f"points_pd missing '{vector_name}' array for polarity vectors.")

        arrow = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            tip_length=float(tip_length),
            tip_radius=float(tip_radius),
            shaft_radius=float(shaft_radius),
        )
        arrow.translate((-0.5, 0.0, 0.0), inplace=True)

        return points_pd.glyph(
            geom=arrow,
            orient=vector_name,
            scale=False,
            factor=float(length),
        )

    @staticmethod
    def cleanup(
        plotter: pv.Plotter,
        blim: float = -10,
        tlim: float = 20,
        show_bounds: bool = True,
        show_axes: bool = True,
    ):
        plotter.set_background("white")

        if show_bounds:
            plotter.show_bounds(
                grid=False,
                xtitle="X",
                ytitle="Y",
                ztitle="Z",
                bounds=(blim, tlim, blim, tlim, blim, tlim),
                axes_ranges=(blim, tlim, blim, tlim, blim, tlim),
            )

        if show_axes:
            plotter.add_axes()

        plotter.set_scale(1, 1, 1)
        plotter.camera_position = "iso"

    @staticmethod
    def draw_3d_view(
        centers,
        radii,
        morphogens,
        polarities,
        particle_count,
        blim=-10,
        tlim=20,
        theta_res=24,
        phi_res=12,
        alpha=0.5,
        show_polarities: bool = True,
        polarity_length: float = 1.0,
        polarity_color="black",
        polarity_opacity: float = 1.0,
        polarity_shaft_radius: float = 0.03,
        polarity_tip_length: float = 0.25,
        polarity_tip_radius: float = 0.06,
        cell_types: np.ndarray | None = None,
    ):
        """Render a PyVista view; ``particle_count`` must be nonnegative.

        Categories override morphogen colors; the shortest selected input limits
        the rendered count.
        """
        _require_pyvista()
        plotter = pv.Plotter(notebook=True)

        sphere_kwargs = dict(
            smooth_shading=True,
            opacity=float(alpha),
            ambient=0.55,
            diffuse=0.2,
            specular=0.2,
            specular_power=1.0,
            show_edges=False,
            rgb=True,
        )

        polarity_kwargs = dict(
            smooth_shading=True,
            color=polarity_color,
            opacity=float(polarity_opacity),
            ambient=0.25,
            diffuse=0.75,
            specular=0.1,
            specular_power=8.0,
            show_edges=False,
        )

        n = int(particle_count)
        pd = PyVistaInterface._points_polydata(
            centers,
            radii,
            morphogens,
            polarities=polarities,
            n=n,
            cell_types=cell_types,
        )

        glyphs = PyVistaInterface._glyph_spheres(pd, theta_res, phi_res)
        plotter.add_mesh(glyphs, scalars="rgb", **sphere_kwargs)

        if show_polarities and polarities is not None:
            pol_glyphs = PyVistaInterface._glyph_polarity_arrows(
                pd,
                length=polarity_length,
                shaft_radius=polarity_shaft_radius,
                tip_length=polarity_tip_length,
                tip_radius=polarity_tip_radius,
            )
            plotter.add_mesh(pol_glyphs, **polarity_kwargs)

        PyVistaInterface.cleanup(plotter, blim, tlim)
        return plotter


class _BaseBackend:
    """Backend contract for fixed-capacity Warp rendering.

    USD consumes the full buffer with zero-radius inactive slots; OpenGL
    consumes the active prefix.
    """

    needs_full_buffer: bool = False

    def render_points_frame(self, *, t: float, points, radius, colors, name: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _OpenGLVideoBackend(_BaseBackend):
    """Offscreen OpenGL video with optional mesh overlay.

    Framebuffer readback flips OpenGL's bottom-up origin before encoding.
    """

    def __init__(
        self,
        filename: str,
        *,
        width: int,
        height: int,
        fps: int,
        device: str,
        camera_pos,
        camera_front,
        camera_up,
        background_color,
        draw_grid: bool,
        draw_axis: bool,
        draw_sky: bool,
        render_wireframe: bool,
        codec: str,
        quality: int,
        pixelformat: str,
        reset_instancers_each_frame: bool = True,
    ):
        self.filename = filename
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.device = device
        self.reset_instancers_each_frame = bool(reset_instancers_each_frame)

        self.renderer = wpr.OpenGLRenderer(
            title="WarpMovie",
            screen_width=self.width,
            screen_height=self.height,
            headless=True,
            device=self.device,
            background_color=background_color,
            draw_grid=draw_grid,
            draw_axis=draw_axis,
            draw_sky=draw_sky,
            render_wireframe=render_wireframe,
            show_info=False,
            camera_pos=camera_pos,
            camera_front=camera_front,
            camera_up=camera_up,
        )

        self._pixels_u8 = wp.empty((self.height, self.width, 3), dtype=wp.uint8, device=self.device)

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        self.writer = imageio.get_writer(
            filename,
            fps=self.fps,
            codec=codec,
            quality=quality,
            pixelformat=pixelformat,
        )

    def render_points_frame(
        self,
        *,
        t: float,
        points,
        radius,
        colors,
        name: str,
        mesh_points=None,
        mesh_indices=None,
    ) -> None:
        self.renderer.clear()

        if self.reset_instancers_each_frame:
            # Avoid stale same-count radii and colors in Warp's named instancer.
            self.renderer._shape_instancers = {}

        self.renderer.begin_frame(float(t))
        self.renderer.render_points(
            name,
            points=points,
            radius=radius,
            colors=colors,
            as_spheres=True,
            visible=True,
        )

        if mesh_points is not None:

            self.renderer.render_mesh(
                "target",
                mesh_points,
                mesh_indices,
            )

        self.renderer.end_frame()

        ok = self.renderer.get_pixels(
            self._pixels_u8,
            split_up_tiles=False,
            mode="rgb",
            use_uint8=True,
        )
        if not ok:
            raise RuntimeError("OpenGLRenderer.get_pixels() failed.")

        frame = self._pixels_u8.numpy()
        frame = frame[::-1]  # OpenGL origin -> image origin
        self.writer.append_data(frame)

    def close(self) -> None:
        try:
            self.writer.close()
        finally:
            self.renderer.close()
            del self._pixels_u8
            del self.renderer


class _UsdStageBackend(_BaseBackend):
    """USD PointInstancer with fixed capacity and zero-radius inactive slots.

    OpenGL provides mesh overlays. The USD stage saves on close or after each
    frame when ``save_every_frame`` is enabled.
    """

    needs_full_buffer: bool = True

    def __init__(
        self,
        stage_path: str,
        *,
        fps: int,
        up_axis: str = "Y",
        scaling: float = 1.0,
        save_every_frame: bool = False,
    ):
        os.makedirs(os.path.dirname(stage_path) or ".", exist_ok=True)

        self.stage_path = stage_path
        self.save_every_frame = bool(save_every_frame)

        self.renderer = wpr.UsdRenderer(
            stage=stage_path,
            up_axis=up_axis,
            fps=int(fps),
            scaling=float(scaling),
        )

    def render_points_frame(
        self,
        *,
        t: float,
        points,
        radius,
        colors,
        name: str,
        mesh_points=None,
        mesh_indices=None,
    ) -> None:
        self.renderer.begin_frame(float(t))
        self.renderer.render_points(
            name,
            points=points,
            radius=radius,
            colors=colors,
            as_spheres=True,
            visible=True,
        )
        self.renderer.end_frame()

        if self.save_every_frame:
            self.renderer.save()

    def close(self) -> None:
        self.renderer.save()


class WarpMovieRenderer:
    """Context-managed OpenGL video or USD animation from fixed Warp buffers.

    NumPy frames provide RGB directly; Warp-state frames derive HSV colors on
    the selected device before this implementation copies them to NumPy. USD
    uses ``usd-core`` and ``usd_*`` options; OpenGL uses camera, scene,
    encoding, and ``opengl_*`` options.
    """

    def __init__(
        self,
        filename: str,
        max_particles: int,
        *,
        backend: Literal["opengl", "usd"] = "usd",
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        device: str = "cuda",
        # Camera, scene, and encoding options configure OpenGL.
        camera_pos=(0.0, 2.0, 10.0),
        camera_front=(0.0, 0.0, -1.0),
        camera_up=(0.0, 1.0, 0.0),
        background_color=(1.0, 1.0, 1.0),
        draw_grid: bool = False,
        draw_axis: bool = False,
        draw_sky: bool = False,
        render_wireframe: bool = False,
        codec: str = "libx264",
        quality: int = 8,
        pixelformat: str = "yuv420p",
        usd_up_axis: str = "Y",
        usd_scaling: float = 1.0,
        usd_save_every_frame: bool = False,
        opengl_reset_instancers_each_frame: bool = True,
        prim_name: str = "cells",
    ):
        wp.init()

        self.filename = filename
        self.max_particles = int(max_particles)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.device = device
        self.prim_name = prim_name

        self._points_f32 = wp.empty(self.max_particles, dtype=wp.vec3, device=self.device)
        self._radii_f32 = wp.empty(self.max_particles, dtype=wp.float32, device=self.device)
        self._colors_f32 = wp.empty(self.max_particles, dtype=wp.vec3, device=self.device)

        if backend == "opengl":
            self._backend: _BaseBackend = _OpenGLVideoBackend(
                filename,
                width=self.width,
                height=self.height,
                fps=self.fps,
                device=self.device,
                camera_pos=camera_pos,
                camera_front=camera_front,
                camera_up=camera_up,
                background_color=background_color,
                draw_grid=draw_grid,
                draw_axis=draw_axis,
                draw_sky=draw_sky,
                render_wireframe=render_wireframe,
                codec=codec,
                quality=quality,
                pixelformat=pixelformat,
                reset_instancers_each_frame=opengl_reset_instancers_each_frame,
            )
        elif backend == "usd":
            self._backend = _UsdStageBackend(
                filename,
                fps=self.fps,
                up_axis=usd_up_axis,
                scaling=usd_scaling,
                save_every_frame=usd_save_every_frame,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    @wp.kernel
    def _pack_buffers_kernel(
        centers_in: wp.array(dtype=wp.vec3f),
        radii_in: wp.array(dtype=wp.float32),
        A_in: wp.array(dtype=wp.float32),
        I_in: wp.array(dtype=wp.float32),
        points_out: wp.array(dtype=wp.vec3),
        radii_out: wp.array(dtype=wp.float32),
        colors_out: wp.array(dtype=wp.vec3),
        particle_count: wp.int32,
        morph_mode: wp.int32,  # 0=A, 1=I, 2=A/(A+I+1e-8)
        morph_scale: wp.float32,
        hue: wp.float32,
        use_base_color: wp.int32,
        base_r: wp.float32,
        base_g: wp.float32,
        base_b: wp.float32,
    ):
        i = wp.tid()

        if i >= particle_count:
            points_out[i] = wp.vec3(0.0, 0.0, 0.0)
            radii_out[i] = wp.float32(0.0)
            colors_out[i] = wp.vec3(0.0, 0.0, 0.0)
            return

        c = centers_in[i]
        points_out[i] = wp.vec3(wp.float32(c[0]), wp.float32(c[1]), wp.float32(c[2]))
        radii_out[i] = wp.float32(radii_in[i])

        if use_base_color != 0:
            colors_out[i] = wp.vec3(base_r, base_g, base_b)
            return

        a = wp.float32(A_in[i])
        b = wp.float32(I_in[i])
        eps = wp.float32(1e-8)

        m = a
        if morph_mode == 1:
            m = b
        elif morph_mode == 2:
            m = a / (a + b + eps)

        s = wp.max(morph_scale, wp.float32(1e-8))
        m = wp.clamp(m / s, wp.float32(0.0), wp.float32(1.0))

        # Six-sector HSV arithmetic runs directly in Warp kernels.
        h = hue
        sat = m
        v = wp.float32(1.0)

        h6 = h * wp.float32(6.0)
        hi = wp.int32(wp.floor(h6))
        f = h6 - wp.float32(hi)

        p = v * (wp.float32(1.0) - sat)
        q = v * (wp.float32(1.0) - sat * f)
        t = v * (wp.float32(1.0) - sat * (wp.float32(1.0) - f))

        r = v
        g = t
        bb = p
        if hi == 1:
            r = q
            g = v
            bb = p
        elif hi == 2:
            r = p
            g = v
            bb = t
        elif hi == 3:
            r = p
            g = q
            bb = v
        elif hi == 4:
            r = t
            g = p
            bb = v
        elif hi == 5:
            r = v
            g = p
            bb = q

        colors_out[i] = wp.vec3(r, g, bb)

    def _pack_gpu_buffers(
        self,
        centers_wp: wp.array,
        radii_wp: wp.array,
        A_wp: wp.array,
        I_wp: wp.array,
        particle_count: int,
        *,
        morph: str = "A",
        morph_scale: float = 1.0,
        hue: float = 0.80,
        base_color: tuple[float, float, float] | None = None,
    ) -> int:
        """Pack HSV colors on-device and zero inactive fixed-capacity slots.

        Centers use ``wp.vec3f``; radii, A, and I use ``wp.float32`` on
        ``self.device``, each with at least the clamped active count.
        Case-insensitive ``i|inhibitor|inhibitors`` and
        ``ratio|a_over_a_plus_i|a/(a+i)`` select ``I`` or
        ``A/(A+I+1e-8)``; unknown aliases use ``A``. The signal divisor has a
        ``1e-8`` floor before clipping to ``[0,1]``; ``hue`` must lie in
        ``[0,1]``. ``base_color`` overrides HSV after RGB clipping.
        """
        n = int(particle_count)
        n = max(0, min(n, self.max_particles))

        morph_mode = 0
        if morph.lower() in ("i", "inhibitor", "inhibitors"):
            morph_mode = 1
        elif morph.lower() in ("ratio", "a_over_a_plus_i", "a/(a+i)"):
            morph_mode = 2

        use_base_color = 0
        base_r = base_g = base_b = 0.0
        if base_color is not None:
            base_r, base_g, base_b = [float(np.clip(value, 0.0, 1.0)) for value in base_color]
            use_base_color = 1

        wp.launch(
            self._pack_buffers_kernel,
            dim=self.max_particles,
            inputs=[
                centers_wp,
                radii_wp,
                A_wp,
                I_wp,
                self._points_f32,
                self._radii_f32,
                self._colors_f32,
                n,
                morph_mode,
                float(morph_scale),
                float(hue),
                use_base_color,
                base_r,
                base_g,
                base_b,
            ],
            device=self.device,
        )
        return n

    def _render(self, t: float, n_active: int, mesh_points=None, mesh_indices=None) -> None:
        # USD receives full capacity with zero-radius inactive slots; OpenGL receives n_active.
        if self._backend.needs_full_buffer:
            pts = self._points_f32.numpy()
            rad = self._radii_f32.numpy()
            col = self._colors_f32.numpy()
        else:
            pts = self._points_f32.numpy()[:n_active]
            rad = self._radii_f32.numpy()[:n_active]
            col = self._colors_f32.numpy()[:n_active]

        self._backend.render_points_frame(
            t=float(t),
            points=pts,
            radius=rad,
            colors=col,
            name=self.prim_name,
            mesh_points=mesh_points,
            mesh_indices=mesh_indices,
        )

    def write_frame_from_numpy(
        self,
        *,
        t: float,
        centers: np.ndarray,
        radii: np.ndarray,
        colors: np.ndarray,
        particle_count: int,
        mesh_points=None,
        mesh_indices=None,
    ) -> None:
        """Write clipped RGB from NumPy state.

        ``particle_count`` is clamped to ``[0, max_particles]`` and arrays are
        sliced to that count, then zero-padded. Centers and colors must have
        shape ``[N,3]`` and radii ``[N]``, with ``N`` at least the clamped count.
        OpenGL frames include supplied mesh overlays; USD frames render packed particles.
        """
        n = int(particle_count)
        n = max(0, min(n, self.max_particles))

        c = np.asarray(centers, dtype=np.float32)[:n]
        r = np.asarray(radii, dtype=np.float32).ravel()[:n]
        col = np.clip(np.asarray(colors, dtype=np.float32)[:n], 0.0, 1.0)

        c_pad = np.zeros((self.max_particles, 3), dtype=np.float32)
        r_pad = np.zeros(self.max_particles, dtype=np.float32)
        col_pad = np.zeros((self.max_particles, 3), dtype=np.float32)

        c_pad[:n] = c
        r_pad[:n] = r
        col_pad[:n] = col

        wp.copy(self._points_f32, wp.from_numpy(c_pad, dtype=wp.vec3, device=self.device))
        wp.copy(self._radii_f32, wp.from_numpy(r_pad, dtype=wp.float32, device=self.device))
        wp.copy(self._colors_f32, wp.from_numpy(col_pad, dtype=wp.vec3, device=self.device))

        self._render(t=float(t), n_active=n, mesh_points=mesh_points, mesh_indices=mesh_indices)

    def write_frame_from_state(
        self,
        *,
        t: float,
        centers_wp: wp.array,
        radii_wp: wp.array,
        A_wp: wp.array,
        I_wp: wp.array,
        particle_count: int,
        morph: str = "A",
        morph_scale: float = 1.0,
        hue: float = 0.80,
        base_color: tuple[float, float, float] | None = None,
        mesh_points=None,
        mesh_indices=None,
    ) -> None:
        """Write Warp state with HSV colors derived on the selected device.

        The active count is clamped to renderer capacity, inactive slots are
        zeroed, and packed buffers are copied to NumPy. Centers use ``wp.vec3f``;
        radii, A, and I use ``wp.float32`` on ``self.device``, each with at least
        the clamped count. OpenGL frames include supplied mesh overlays; USD frames render
        packed particles.
        """
        n = self._pack_gpu_buffers(
            centers_wp,
            radii_wp,
            A_wp,
            I_wp,
            particle_count,
            morph=morph,
            morph_scale=morph_scale,
            hue=hue,
            base_color=base_color,
        )

        self._render(t=float(t), n_active=n, mesh_points=mesh_points, mesh_indices=mesh_indices)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
