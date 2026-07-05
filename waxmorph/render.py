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
    """Raise if the optional PyVista rendering dependency is unavailable."""
    if pv is None:
        raise ImportError(
            "waxmorph.render requires optional dependency 'pyvista'. "
            "Install extras with: pip install -e .[simulation]"
        )


class RenderInterface(ABC):
    """Abstract interface for interactive cell-state renderers."""

    @staticmethod
    @abstractmethod
    def draw_sphere(*args, **kwargs):
        """Draw a sphere mesh given parameters."""
        pass

    @staticmethod
    @abstractmethod
    def cleanup(*args, **kwargs):
        """Anything except the sphere drawing should go here."""
        pass

    @staticmethod
    @abstractmethod
    def draw_3d_view(*args, **kwargs):
        """Create plot and put the spheres in, iteratively calling `BaseClass.draw_sphere`. On
        exit should call `BaseClass.cleanup`.
        """
        pass


############################################################
############################################################
############################################################

# INTERACTIVE RENDERERS

############################################################
############################################################
############################################################


class MPLInterface(RenderInterface):
    """Matplotlib renderer for static, non-interactive cell-state snapshots.

    Draws each active particle as a translucent parametric sphere surface in a
    fixed-view 3D axes, tinting it by morphogen value (HSV saturation). Use this
    for lightweight figures and quick checks; prefer :class:`PyVistaInterface`
    for many particles or interactive inspection.

    See Also:
        PyVistaInterface: GPU-glyphed interactive renderer of the same state.
    """

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
        """Draw a translucent parametric sphere surface at ``center`` with ``radius``.

        ``theta_res``/``phi_res`` set longitude/latitude mesh resolution.
        """
        cx, cy, cz = center
        # parametric sphere: theta=longitude, phi=latitude, gridded then mapped to xyz
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
        """Apply axes, bounds, and background styling to a Matplotlib view.

        Args:
            ax: Matplotlib 3D axes object.
            blim: Lower bound for each spatial axis.
            tlim: Upper bound for each spatial axis.
        """
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
        """Render a static Matplotlib 3D view of active particles.

        Args:
            centers: Particle centers with shape ``[N, 3]``.
            radii: Particle radii with shape ``[N]``.
            morphogens: Scalar morphogen values used for color saturation.
            particle_count: Number of active particles to draw.
            blim: Lower bound for each spatial axis.
            tlim: Upper bound for each spatial axis.

        Returns:
            Matplotlib figure containing the rendered view.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # one surface per active particle; morphogen drives HSV saturation
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
    """Interactive GPU renderer for cell state, scalable to many particles.

    Built on PyVista/VTK and tuned for throughput: all N spheres are drawn as a
    single glyphed actor (one GPU draw call), with optional polarity arrows as a
    second glyphed actor. Particles are colored by morphogen value, or by a
    stable per-category palette when ``cell_types`` is supplied. Supports static
    and multi-frame inputs (the latter with an interactive frame slider).

    See Also:
        MPLInterface: lightweight static Matplotlib renderer of the same state.
    """

    @staticmethod
    def _husl_palette(n_colors: int, *, s: float = 90.0, lightness: float = 65.0) -> np.ndarray:
        """
        Returns (n_colors, 3) float RGB in [0,1].
        Prefers HSLuv/HUSL if installed; otherwise falls back to HSV palette.
        """
        n_colors = int(max(1, n_colors))
        hues = np.linspace(0.0, 360.0, n_colors, endpoint=False)

        # Try true HUSL/HSLuv (pip install hsluv)
        try:
            import hsluv  # type: ignore

            rgb = np.array(
                [hsluv.hsluv_to_rgb((float(h), float(s), float(lightness))) for h in hues],
                dtype=float,
            )
            rgb = np.clip(rgb, 0.0, 1.0)
            return rgb
        except Exception:
            # Fallback: HSV evenly spaced hues, fixed saturation/value
            h01 = (hues / 360.0).astype(float)
            hsv = np.stack([h01, np.full_like(h01, 0.85), np.full_like(h01, 0.95)], axis=-1)
            return colors.hsv_to_rgb(hsv)

    # ---------- internal helpers ----------
    @staticmethod
    def _rgb_from_categories(categories: np.ndarray) -> np.ndarray:
        """
        categories: (N,) unsigned ints (or any ints)
        Returns rgb: (N,3) uint8, with a stable color per unique category.
        """
        cats = np.asarray(categories).reshape(-1)
        # enforce unsigned-ish, but keep it robust if negatives sneak in
        cats = cats.astype(np.int64, copy=False)

        # palette indexed by sorted-unique rank → same category always same color
        uniq, inv = np.unique(cats, return_inverse=True)  # uniq sorted
        palette = PyVistaInterface._husl_palette(len(uniq))  # (K,3) float
        rgb = palette[inv]  # (N,3) float
        return (rgb * 255).astype(np.uint8)

    @staticmethod
    def _rgb_from_morph(morphogens: np.ndarray) -> np.ndarray:
        """Map scalar morphogen values to clipped uint8 RGB colors."""
        m = np.asarray(morphogens).astype(float)
        m = np.clip(m, 0.0, 1.0)
        rgb_float = colors.hsv_to_rgb(
            np.stack([np.full(shape=m.shape, fill_value=0.08), m, np.ones_like(m)], axis=-1)
        )  # (N,3) in [0,1]
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
        """
        Build a point-cloud PolyData with per-point arrays:
          - 'radius' (float)
          - 'rgb'    (uint8[3])    (from morphogens OR cell_types override)
          - 'polarity' (float[3]) optional
          - 'cell_type' (int) optional
        """
        c = np.asarray(centers, dtype=float)
        r = np.asarray(radii, dtype=float).reshape(-1)
        m = np.asarray(morphogens, dtype=float).reshape(-1)

        # clamp n to the shortest input so per-point arrays stay aligned
        if n is None:
            n = len(c)
        n = min(n, len(c), len(r), len(m))

        pts = c[:n]
        rad = r[:n]

        _require_pyvista()

        if cell_types is not None:
            ct = np.asarray(cell_types).reshape(-1)
            n = min(n, len(ct))
            pts = pts[:n]
            rad = rad[:n]
            rgb = PyVistaInterface._rgb_from_categories(ct[:n])
        else:
            rgb = PyVistaInterface._rgb_from_morph(m[:n])

        pd = pv.PolyData(pts)
        pd["radius"] = rad
        pd["rgb"] = rgb  # used with rgb=True

        # Optional: store the category itself (handy for picking/inspection)
        if cell_types is not None:
            pd["cell_type"] = np.asarray(cell_types).reshape(-1)[:n].astype(np.int32, copy=False)

        if polarities is not None:
            p = np.asarray(polarities, dtype=float)
            if p.ndim != 2 or p.shape[1] != 3:
                raise ValueError(f"polarities must have shape (N,3); got {p.shape}")
            p = p[:n]

            # unit-normalize so arrow glyphs encode direction only, not magnitude
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            p = p / np.clip(norms, 1e-12, None)

            pd["polarity"] = p

        return pd

    @staticmethod
    def _glyph_spheres(points_pd: pv.PolyData, theta_res=24, phi_res=12) -> pv.PolyData:
        """Create sphere glyph geometry from point-cloud radius data."""
        _require_pyvista()
        # instance one unit sphere at every point, scaled by per-point 'radius'
        base = pv.Sphere(radius=1.0, theta_resolution=theta_res, phi_resolution=phi_res)
        glyphs = points_pd.glyph(geom=base, scale="radius", orient=False)
        return glyphs

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
        """
        Glyph centered arrows oriented by points_pd[vector_name].

        The base arrow is built along +X from 0->1, then translated by -0.5 in X
        so its midpoint is at the origin. Glyphing then centers each arrow at the point.
        """
        _require_pyvista()
        if vector_name not in points_pd.array_names:
            raise ValueError(f"points_pd missing '{vector_name}' array for polarity vectors.")

        # Arrow points along +X by default (direction=(1,0,0)), from start to start+direction.
        arrow = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            tip_length=float(tip_length),
            tip_radius=float(tip_radius),
            shaft_radius=float(shaft_radius),
        )
        # Center it: make the arrow span roughly [-0.5, +0.5] in local X before scaling.
        arrow.translate((-0.5, 0.0, 0.0), inplace=True)

        # orient by 'polarity', constant scale via factor=length
        glyphs = points_pd.glyph(
            geom=arrow,
            orient=vector_name,
            scale=False,
            factor=float(length),
        )
        return glyphs

    # ---------- public API ----------
    @staticmethod
    def cleanup(
        plotter: pv.Plotter,
        blim: float = -10,
        tlim: float = 20,
        show_bounds: bool = True,
        show_axes: bool = True,
    ):
        """Apply PyVista scene bounds, axes, background, and camera defaults.

        Args:
            plotter: PyVista plotter to mutate.
            blim: Lower bound for each spatial axis.
            tlim: Upper bound for each spatial axis.
            show_bounds: Whether to draw axis bounds.
            show_axes: Whether to add a 3D axes widget.
        """
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

    # ---------- public API ----------
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
        # polarity rendering controls
        show_polarities: bool = True,
        polarity_length: float = 1.0,
        polarity_color="black",
        polarity_opacity: float = 1.0,
        polarity_shaft_radius: float = 0.03,
        polarity_tip_length: float = 0.25,
        polarity_tip_radius: float = 0.06,
        cell_types: np.ndarray | None = None,
    ):
        """Render an interactive PyVista view of active particles.

        Args:
            centers: Particle centers with shape ``[N, 3]`` or frame-compatible
                array-like input.
            radii: Particle radii with shape ``[N]``.
            morphogens: Scalar morphogen values used for coloring unless
                ``cell_types`` is provided.
            polarities: Optional polarity vectors with shape ``[N, 3]``.
            particle_count: Number of active particles to draw.
            blim: Lower bound for each spatial axis.
            tlim: Upper bound for each spatial axis.
            theta_res: Sphere glyph longitude resolution.
            phi_res: Sphere glyph latitude resolution.
            alpha: Sphere opacity.
            cell_types: Optional integer category labels used for stable
                categorical colors.

        Returns:
            Configured :class:`pyvista.Plotter`.
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

        # all spheres as one glyphed actor (single GPU draw); rgb=True reads 'rgb'
        glyphs = PyVistaInterface._glyph_spheres(pd, theta_res, phi_res)
        plotter.add_mesh(glyphs, scalars="rgb", **sphere_kwargs)

        # second actor: polarity arrows, only when vectors are supplied
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


############################################################
############################################################
############################################################

# ANIMATION RENDERERS

############################################################
############################################################
############################################################

# ------------------------
# Backend adapters
# ------------------------


class _BaseBackend:
    """Adapter contract for a single Warp rendering backend.

    Each backend (headless OpenGL video, USD stage) implements per-frame point
    rendering and resource teardown behind this common interface so that
    :class:`WarpMovieRenderer` stays backend-agnostic.

    Attributes:
        needs_full_buffer: When ``True``, the backend expects the full
            ``max_particles`` buffer every frame (inactive particles zeroed,
            radius 0 → invisible glyph) rather than a slice of only the active
            particles. USD point instancers require the fixed-capacity buffer;
            OpenGL can take just the active slice.
    """

    # If True, the renderer expects the full max_particles buffer every frame
    # (inactive particles zeroed out) rather than a slice of only active ones.
    needs_full_buffer: bool = False

    def render_points_frame(self, *, t: float, points, radius, colors, name: str) -> None:
        """Render one frame of point primitives."""
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources."""
        pass


class _OpenGLVideoBackend(_BaseBackend):
    """Headless OpenGL backend that encodes rendered frames to a video file.

    Drives a Warp ``OpenGLRenderer`` offscreen, reads back the framebuffer into
    a reusable GPU pixel buffer, flips it from OpenGL's bottom-up origin to the
    image top-down origin, and streams it to an :mod:`imageio` writer. Renders
    only the active-particle slice (``needs_full_buffer`` stays ``False``), and
    optionally overlays a static target mesh.
    """

    def __init__(
        self,
        filename: str,
        *,
        width: int,
        height: int,
        fps: int,
        device: str,
        # camera
        camera_pos,
        camera_front,
        camera_up,
        # visuals
        background_color,
        draw_grid: bool,
        draw_axis: bool,
        draw_sky: bool,
        render_wireframe: bool,
        # codec
        codec: str,
        quality: int,
        pixelformat: str,
        # instancing workaround
        reset_instancers_each_frame: bool = True,
    ):
        self.filename = filename
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.device = device
        self.reset_instancers_each_frame = bool(reset_instancers_each_frame)

        # Renderer
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

        # GPU pixel buffer for get_pixels()
        self._pixels_u8 = wp.empty((self.height, self.width, 3), dtype=wp.uint8, device=self.device)

        # Video writer
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
        """Render one OpenGL frame and append it to the video writer."""
        self.renderer.clear()

        # drop instancers so a fresh point count rebuilds geometry each frame
        if self.reset_instancers_each_frame:
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
        """Close the video writer and OpenGL renderer."""
        try:
            self.writer.close()
        finally:
            self.renderer.close()
            del self._pixels_u8
            del self.renderer


class _UsdStageBackend(_BaseBackend):
    """USD stage backend that writes particle frames to a ``.usd`` file.

    Drives a Warp ``UsdRenderer`` whose single PointInstancer is created at the
    full ``max_particles`` capacity, so ``needs_full_buffer`` is ``True`` and
    inactive particles are kept at radius 0 (scale 0 → invisible) rather than
    dropped. Mesh overlays are not supported here (OpenGL only). The stage is
    flushed on :meth:`close`, or after every frame when ``save_every_frame``.
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
        # UsdRenderer accepts either a filepath or a Usd.Stage object
        os.makedirs(os.path.dirname(stage_path) or ".", exist_ok=True)

        self.stage_path = stage_path
        self.save_every_frame = bool(save_every_frame)

        # require usd-core or usd-exchange installed (see Warp docs)
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
        """Render one USD frame of point primitives."""
        # Mesh overlays are currently only supported by the OpenGL backend.
        _ = (mesh_points, mesh_indices)
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
        """Save the USD stage."""
        self.renderer.save()


class WarpMovieRenderer:
    """Offscreen animator that writes Warp particle trajectories to a movie.

    Backend-agnostic front end over the rendering adapters: it owns fixed-size
    GPU buffers sized to ``max_particles``, packs per-frame state into them, and
    delegates to either :class:`_OpenGLVideoBackend` (headless video file) or
    :class:`_UsdStageBackend` (USD stage) chosen by ``backend``. Feed frames via
    :meth:`write_frame_from_numpy` (precomputed RGB) or
    :meth:`write_frame_from_state` (Warp arrays with morphogen-derived color).
    Usable as a context manager, which closes the backend on exit.

    Args:
        filename: Output video or USD file path.
        max_particles: Maximum particle capacity packed into renderer buffers.
        backend: ``"opengl"`` for a headless video file or ``"usd"`` for a
            USD stage.
        width: OpenGL video width in pixels.
        height: OpenGL video height in pixels.
        fps: Output frames per second.
        device: Warp device used for packed buffers.
        camera_pos: OpenGL camera position.
        camera_front: OpenGL camera forward vector.
        camera_up: OpenGL camera up vector.
        background_color: OpenGL RGB background color in ``[0, 1]``.
        draw_grid: Whether the OpenGL backend draws a grid.
        draw_axis: Whether the OpenGL backend draws axes.
        draw_sky: Whether the OpenGL backend draws sky.
        render_wireframe: Whether the OpenGL backend renders wireframes.
        codec: :mod:`imageio` video codec for the OpenGL backend.
        quality: ImageIO output quality for the OpenGL backend.
        pixelformat: ImageIO pixel format for the OpenGL backend.
        usd_up_axis: USD stage up axis.
        usd_scaling: USD stage scaling factor.
        usd_save_every_frame: Whether to save the USD stage after each frame.
        opengl_reset_instancers_each_frame: Whether to clear OpenGL instancers
            before each frame.
        prim_name: Primitive name used for rendered particles.

    Raises:
        ValueError: If ``backend`` is not ``"opengl"`` or ``"usd"``.
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
        # camera (OpenGL only; ignored by USD backend)
        camera_pos=(0.0, 2.0, 10.0),
        camera_front=(0.0, 0.0, -1.0),
        camera_up=(0.0, 1.0, 0.0),
        # visuals (OpenGL only; ignored by USD backend)
        background_color=(1.0, 1.0, 1.0),
        draw_grid: bool = False,
        draw_axis: bool = False,
        draw_sky: bool = False,
        render_wireframe: bool = False,
        # codec (OpenGL only; ignored by USD backend)
        codec: str = "libx264",
        quality: int = 8,
        pixelformat: str = "yuv420p",
        # USD options
        usd_up_axis: str = "Y",
        usd_scaling: float = 1.0,
        usd_save_every_frame: bool = False,
        # OpenGL instancing workaround
        opengl_reset_instancers_each_frame: bool = True,
        # prim name
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

        # --- common GPU packed buffers ---
        self._points_f32 = wp.empty(self.max_particles, dtype=wp.vec3, device=self.device)
        self._radii_f32 = wp.empty(self.max_particles, dtype=wp.float32, device=self.device)
        self._colors_f32 = wp.empty(self.max_particles, dtype=wp.vec3, device=self.device)

        # --- choose backend ---
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

    # ------------------------
    # GPU kernel: pack buffers
    # ------------------------
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
        morph_mode: wp.int32,  # 0:A, 1:I, 2:A/(A+I)
        morph_scale: wp.float32,  # divide then clamp to [0,1]
        hue: wp.float32,  # fixed hue in [0,1]
        use_base_color: wp.int32,
        base_r: wp.float32,
        base_g: wp.float32,
        base_b: wp.float32,
    ):
        # Pack per-particle simulation state into the fixed-size GPU buffers the
        # Warp renderer consumes each frame: position (vec3), radius (float32),
        # and an RGB colour (vec3). One thread handles one buffer slot.
        #
        # Colour comes from one of two sources:
        #   * a constant base_r/base_g/base_b RGB when use_base_color != 0, or
        #   * a morphogen-driven HSV ramp at the fixed hue `hue` with value=1,
        #     where saturation encodes a scalar morph signal selected by
        #     morph_mode: 0 -> activator A, 1 -> inhibitor I, 2 -> the ratio
        #     A/(A+I). The signal is divided by morph_scale and clamped to
        #     [0, 1], so saturation 0 reads near-white and 1 reads fully tinted.
        # The HSV->RGB conversion is hand-rolled here (no NumPy/colorsys inside
        # a kernel); it follows the standard 6-sector piecewise formula.
        i = wp.tid()

        # inactive slots: zero radius → glyph scale 0 → invisible
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

        # pick scalar by mode: activator, inhibitor, or normalized ratio
        m = a
        if morph_mode == 1:
            m = b
        elif morph_mode == 2:
            m = a / (a + b + eps)

        # guard against morph_scale=0
        s = wp.max(morph_scale, wp.float32(1e-8))
        m = wp.clamp(m / s, wp.float32(0.0), wp.float32(1.0))

        # HSV -> RGB (hue fixed, saturation=m, value=1) via the standard
        # 6-sector piecewise conversion.
        h = hue
        sat = m
        v = wp.float32(1.0)

        h6 = h * wp.float32(6.0)
        hi = wp.int32(wp.floor(h6))  # which of the 6 hue sectors, 0..5
        f = h6 - wp.float32(hi)  # fractional position within the sector

        # the three intermediate channel levels reused across sectors
        p = v * (wp.float32(1.0) - sat)
        q = v * (wp.float32(1.0) - sat * f)
        t = v * (wp.float32(1.0) - sat * (wp.float32(1.0) - f))

        # default branch is sector 0; the elif chain handles sectors 1..5
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
        """Pack Warp state into the renderer's fixed-size GPU buffers for a frame.

        Resolves the ``morph`` color mode and optional ``base_color`` override,
        then launches :meth:`_pack_buffers_kernel` over the full ``max_particles``
        capacity so that slots past the active count are zeroed (radius 0 renders
        as an invisible glyph). The packed results live in ``self._points_f32``,
        ``self._radii_f32``, and ``self._colors_f32``.

        Args:
            centers_wp: Warp position array with dtype ``wp.vec3f``.
            radii_wp: Warp radius array with dtype ``wp.float32``.
            A_wp: Warp activator morphogen array.
            I_wp: Warp inhibitor morphogen array.
            particle_count: Requested number of active particles; clamped to
                ``[0, max_particles]``.
            morph: Color source. ``"A"`` (default) uses the activator,
                ``"i"``/``"inhibitor"``/``"inhibitors"`` use the inhibitor, and
                ``"ratio"``/``"a_over_a_plus_i"``/``"a/(a+i)"`` use the
                normalized ratio ``A/(A+I)``. Matching is case-insensitive;
                unrecognized values fall back to the activator.
            morph_scale: Divisor applied to the selected scalar before clipping
                color saturation to ``[0, 1]``.
            hue: Fixed HSV hue in ``[0, 1]`` for morphogen coloring.
            base_color: Optional constant RGB in ``[0, 1]`` that overrides
                morphogen coloring for every active particle.

        Returns:
            The clamped active-particle count actually packed into the buffers.
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

        # launch over full capacity; kernel zeroes slots past n_active
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
        # render_points currently wants CPU-indexable arrays, so we copy here.
        # USD backend needs the full buffer so the PointInstancer is created at
        # max capacity; inactive particles have radius=0 (scale=0 → invisible).
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

    # ------------------------
    # public API
    # ------------------------
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
        """Write one frame from :mod:`numpy` arrays with explicit RGB colors.

        Use when colors are already computed (e.g. shape assembly with multiple
        signaling-molecule states) rather than derived from separate
        activator/inhibitor Warp arrays. Inputs are clamped to ``particle_count``
        and zero-padded to ``max_particles`` to match the fixed-size GPU buffers.

        Args:
            t: Frame time stamp (seconds in renderer time).
            centers: Particle positions with shape ``[N, 3]``.
            radii: Particle radii with shape ``[N]``.
            colors: Per-particle RGB colors with shape ``[N, 3]``, clipped to
                ``[0, 1]``.
            particle_count: Number of active particles; clamped to
                ``[0, max_particles]``.
            mesh_points: Optional target mesh vertices for the OpenGL backend.
            mesh_indices: Optional target mesh indices for the OpenGL backend.

        See Also:
            write_frame_from_state: Frames from Warp state with morphogen color.
        """
        n = int(particle_count)
        n = max(0, min(n, self.max_particles))

        c = np.asarray(centers, dtype=np.float32)[:n]
        r = np.asarray(radii, dtype=np.float32).ravel()[:n]
        col = np.clip(np.asarray(colors, dtype=np.float32)[:n], 0.0, 1.0)

        # Pad to max_particles to match fixed-size Warp buffers; tail stays zero
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
        """Write one frame straight from Warp state, coloring by morphogen.

        Use during live Warp simulation: state stays on the GPU and the color is
        derived on-device by :meth:`_pack_buffers_kernel` (HSV ramp at fixed
        ``hue``, saturation from the ``morph`` signal divided by ``morph_scale``
        and clamped to ``[0, 1]``), avoiding a host round-trip. A constant
        ``base_color`` overrides morphogen coloring when given.

        Args:
            t: Frame time stamp (seconds in renderer time).
            centers_wp: Warp position array with dtype ``wp.vec3f``.
            radii_wp: Warp radius array with dtype ``wp.float32``.
            A_wp: Warp activator morphogen array.
            I_wp: Warp inhibitor morphogen array.
            particle_count: Number of active particles; clamped to
                ``[0, max_particles]``.
            morph: Color source: ``"A"`` (activator), ``"I"`` (inhibitor), or a
                ratio alias for ``A/(A+I)``; see :meth:`_pack_gpu_buffers`.
            morph_scale: Divisor applied to the morphogen signal before clipping
                color saturation to ``[0, 1]``.
            hue: Fixed HSV hue in ``[0, 1]`` for morphogen coloring.
            base_color: Optional constant RGB in ``[0, 1]`` overriding morphogen
                coloring.
            mesh_points: Optional target mesh vertices for the OpenGL backend.
            mesh_indices: Optional target mesh indices for the OpenGL backend.

        See Also:
            write_frame_from_numpy: Frames from precomputed RGB color arrays.
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
        """Close the underlying renderer backend."""
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
