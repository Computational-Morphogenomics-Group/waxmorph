"""Includes renderer interfaces for displaying 3D spheroids with morphogens."""

import matplotlib.pyplot as plt
from matplotlib import colors
from abc import ABC, abstractmethod
from typing import Literal, Optional
import numpy as np
import pyvista as pv
import vtk, os
from tqdm import trange
import imageio
import warp as wp
import warp.render as wpr
from dataclasses import dataclass

class RenderInterface(ABC):
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

            #TODO: Update Docstrings
    
############################################################
############################################################
############################################################


    
class MPLInterface(RenderInterface):
    """Matplotlib interface for rendering frames. Fixed view / non-interactive."""
    @staticmethod
    def draw_sphere(
    ax, center, radius,
    alpha=0.35, facecolor="blue", edgecolor="red",
    theta_res=24, phi_res=12, antialiased=True
    ):
        """
        Draw a 3D sphere (surface) at 'center' with 'radius'.
        - facecolor = fill color (blue by default)
        - edgecolor = mesh edge color (red by default)
        - alpha     = transparency for the surface & edges
        - theta_res, phi_res control mesh resolution (longitude/latitude)
        """
        cx, cy, cz = center
        theta = np.linspace(0, 2*np.pi, theta_res)
        phi   = np.linspace(0, np.pi, phi_res)
        TH, PH = np.meshgrid(theta, phi)
    
        X = cx + radius * np.cos(TH) * np.sin(PH)
        Y = cy + radius * np.sin(TH) * np.sin(PH)
        Z = cz + radius * np.cos(PH)
    
        ax.plot_surface(
            X, Y, Z,
            rcount=phi_res, ccount=theta_res,
            color=facecolor,
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
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_xlim(blim, tlim); ax.set_ylim(blim, tlim); ax.set_zlim(blim, tlim)

    @staticmethod
    def draw_3d_view(centers, radii, morphogens, particle_count, blim=-10, tlim=20):           
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        for center, radius, morphogen in zip(centers[:particle_count], radii[:particle_count], morphogens[:particle_count]):
            MPLInterface.draw_sphere(ax, center=center, facecolor=colors.hsv_to_rgb((0.8, morphogen.item(), 1.)), radius=radius, alpha=0.35)

        MPLInterface.cleanup(ax, blim, tlim)

        return fig



class PyVistaInterface(RenderInterface):
    """
    High-performance renderer using PyVista/VTK.
    - Uses glyphing to draw N spheres as a single GPU-optimized actor.
    - Supports static and multi-frame inputs (with an interactive slider).
    """

    # ---------- internal helpers ----------
    @staticmethod
    def _rgb_from_morph(morphogens: np.ndarray) -> np.ndarray:
        m = np.asarray(morphogens).astype(float)
        m = np.clip(m, 0.0, 1.0)
        rgb_float = colors.hsv_to_rgb(np.stack([np.ones_like(m), m, np.ones_like(m)], axis=-1))  # (N,3) in [0,1]
        return (rgb_float * 255).astype(np.uint8)

    @staticmethod
    def _points_polydata(centers, radii, morphogens, n=None) -> pv.PolyData:
        """
        Build a point-cloud PolyData with per-point arrays:
          - 'radius' (float)
          - 'rgb' (uint8[3])
        """
        c = np.asarray(centers, dtype=float)
        r = np.asarray(radii, dtype=float).reshape(-1)
        m = np.asarray(morphogens, dtype=float).reshape(-1)

        if n is None:
            n = len(c)
        n = min(n, len(c), len(r), len(m))

        pts = c[:n]
        rad = r[:n]
        rgb = PyVistaInterface._rgb_from_morph(m[:n])

        pd = pv.PolyData(pts)
        pd["radius"] = rad
        pd["rgb"] = rgb  # will be replicated to glyph vertices; used with rgb=True
        return pd

    @staticmethod
    def _glyph_spheres(points_pd: pv.PolyData, theta_res=24, phi_res=12) -> pv.PolyData:
        """
        Create a glyph dataset instancing a unit sphere at each point, scaled by 'radius'.
        """
        base = pv.Sphere(radius=1.0, theta_resolution=theta_res, phi_resolution=phi_res)
        # Orient=False so we don't need normals/tangents per-point; scale by 'radius' directly
        glyphs = points_pd.glyph(geom=base, scale="radius", orient=False)
        return glyphs

    # ---------- public API ----------
    @staticmethod
    def draw_sphere(
        plotter: pv.Plotter, center, radius,
        alpha=0.35, facecolor="blue", edgecolor="red",
        theta_res=24, phi_res=12, antialiased=True
    ):
        """
        Add a single sphere actor (useful for quick tests or tiny N).
        For large N, prefer draw_3d_view() which uses glyphing.
        """
        # Normalize facecolor to uint8 RGB
        if isinstance(facecolor, str):
            rgb = (np.array(colors.to_rgb(facecolor)) * 255).astype(np.uint8)
        else:
            rgb = (np.array(facecolor) * 255).astype(np.uint8)

        sphere = pv.Sphere(radius=float(radius), center=np.asarray(center, float),
                           theta_resolution=theta_res, phi_resolution=phi_res)
        plotter.add_mesh(
            sphere,
            color=tuple(rgb.tolist()),
            smooth_shading=True,
            opacity=float(alpha),
            # lighting params
            ambient=0.55, diffuse=0.8, specular=0.2, specular_power=8.0,
            # edges cost perf; keep off
            show_edges=False,
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
                xtitle="X", ytitle="Y", ztitle="Z",
                bounds=(blim, tlim, blim, tlim, blim, tlim),
                axes_ranges=(blim, tlim, blim, tlim, blim, tlim),
            )

        if show_axes:
            plotter.add_axes()

        plotter.set_scale(1, 1, 1)
        plotter.camera_position = "iso"

    @staticmethod
    def draw_3d_view(
        centers, radii, morphogens, particle_count,
        blim=-10, tlim=20, theta_res=24, phi_res=12,
        alpha=0.5
    ):
        """
        If arrays for a single frame:
          - centers: (N, 3), radii: (N,), morphogens: (N,) in [0,1], particle_count: int
        -> Returns a pv.Plotter with a single glyph actor.

        If sequences (len T) of per-frame arrays:
          - centers[t]: (N_t,3), radii[t]: (N_t,), morphogens[t]: (N_t,), particle_count[t]: int
        -> Returns a pv.Plotter with a slider to scrub frames (fast in-place glyph updates).
        """
        plotter = pv.Plotter(notebook=True)
        is_animated = isinstance(centers, (list, tuple)) and len(centers) > 0 and np.asarray(centers[0]).ndim >= 2

        # Lighting tuned for translucent-ish surfaces
        # (note: PyVista handles multisample AA internally if available)
        lighting_kwargs = dict(
            smooth_shading=True,
            opacity=float(alpha),
            ambient=0.55, diffuse=0.2, specular=0.2, specular_power=1.0,
            show_edges=False,
            rgb=True,  # we will pass per-vertex RGB
        )

        if is_animated:
            T = len(centers)

            # Build initial frame
            n0 = int(np.asarray(particle_count[0]))
            pd0 = PyVistaInterface._points_polydata(centers[0], radii[0], morphogens[0], n=n0)
            glyphs0 = PyVistaInterface._glyph_spheres(pd0, theta_res, phi_res)

            actor = plotter.add_mesh(glyphs0, scalars="rgb", **lighting_kwargs)

            # Hold references so they don't get GC'd in callbacks
            state = {"actor": actor, "glyphs": glyphs0}

            def _update_frame(tfloat):
                t = int(round(tfloat))
                t = max(0, min(T - 1, t))
                n = int(np.asarray(particle_count[t]))
            
                pd = PyVistaInterface._points_polydata(centers[t], radii[t], morphogens[t], n=n)
                new_glyphs = PyVistaInterface._glyph_spheres(pd, theta_res, phi_res)  # -> PolyData
            
                mapper = state["actor"].mapper
            
                # Feed the new PolyData directly to the (PolyData) mapper
                try:
                    mapper.SetInputData(new_glyphs)
                except Exception:
                    # Fallback for odd VTK/PyVista combos (rare):
                    mapper.SetInputData(new_glyphs.cast_to_unstructured_grid())
            
                # Make sure RGB coloring stays attached after swapping inputs
                try:
                    mapper.ScalarVisibilityOn()
                    mapper.SetScalarModeToUsePointFieldData()
                    mapper.SelectColorArray("rgb")
                    mapper.SetColorModeToDirectScalars()
                except Exception:
                    pass
            
                mapper.Modified()
                state["glyphs"] = new_glyphs  # keep a ref so it doesn't get GC'd
                plotter.render()


            plotter.add_slider_widget(
                _update_frame,
                rng=[0, max(T - 1, 0)],
                value=0,
                title="Frame",
                pointa=(0.02, 0.06), pointb=(0.98, 0.06),
                style="modern",
            )

        else:
            n = int(particle_count)
            pd = PyVistaInterface._points_polydata(centers, radii, morphogens, n=n)
            glyphs = PyVistaInterface._glyph_spheres(pd, theta_res, phi_res)
            plotter.add_mesh(glyphs, scalars="rgb", **lighting_kwargs)

        PyVistaInterface.cleanup(plotter, blim, tlim)
        return plotter


    @staticmethod
    def write_movie(
        centers_seq,
        radii_seq,
        morphogens_seq,
        particle_count_seq,
        filename,
        blim=-10,
        tlim=20,
        theta_res=24,
        phi_res=12,
        alpha=0.5,
        fps=60,
        add_text=False,
        window_size=(1920, 1080),
    ):
        """
        Write a movie using PyVista's open_movie / write_frame API. Super slow, should be avoided if possible.

        Parameters
        ----------
        centers_seq, radii_seq, morphogens_seq, particle_count_seq :
            Sequences (len T) of per-frame arrays:
              - centers_seq[t] : (N_t, 3)
              - radii_seq[t]   : (N_t,)
              - morphogens_seq[t] : (N_t,) in [0,1]
              - particle_count_seq[t] : int
        filename : str
            Output movie filename (e.g. 'Output/pyvista_movie.mp4').
        """
        
        # Ensure everything has same length
        T = len(centers_seq)
        if not (
            len(radii_seq) == len(morphogens_seq) == len(particle_count_seq) == T
        ):
            raise ValueError("All input sequences must have the same length T.")

        # Make sure directory exists
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Off-screen plotter
        plotter = pv.Plotter(
            notebook=True,
            window_size=window_size,
        )

        # Open movie file
        plotter.open_movie(filename, framerate=fps)

        # Lighting / shading settings
        lighting_kwargs = dict(
            smooth_shading=True,
            opacity=float(alpha),
            ambient=0.55, diffuse=0.2, specular=0.2, specular_power=1.0,
            show_edges=False,
            rgb=True,  # use 'rgb' point data as direct colors
        )

        # ----- Initial frame (t = 0) -----
        n0 = int(np.asarray(particle_count_seq[0]))
        pd0 = PyVistaInterface._points_polydata(
            centers_seq[0], radii_seq[0], morphogens_seq[0], n=n0
        )
        glyphs0 = PyVistaInterface._glyph_spheres(pd0, theta_res, phi_res)
        actor = plotter.add_mesh(glyphs0, scalars="rgb", **lighting_kwargs)

        # Basic scene layout
        PyVistaInterface.cleanup(
            plotter,
            blim=blim,
            tlim=tlim,
            show_bounds=False,
            show_axes=False,
        )

        # Show once to initialize the render window (no GUI if off_screen)
        plotter.show(auto_close=False)

        # Optional text label; we update it each frame by reusing the same name
        if add_text:
            plotter.add_text("Iteration: 0", name="time-label")

        # Write the initial frame
        plotter.write_frame()

        # Keep state so glyphs don't get GC'd
        state = {"actor": actor, "glyphs": glyphs0}

        # ----- Subsequent frames -----
        for t in trange(1, T):
            n = int(np.asarray(particle_count_seq[t]))
            pd = PyVistaInterface._points_polydata(
                centers_seq[t], radii_seq[t], morphogens_seq[t], n=n
            )
            new_glyphs = PyVistaInterface._glyph_spheres(pd, theta_res, phi_res)

            mapper = state["actor"].mapper

            # Swap mapper input to new glyphs
            try:
                mapper.SetInputData(new_glyphs)
            except Exception:
                mapper.SetInputData(new_glyphs.cast_to_unstructured_grid())

            # Ensure RGB coloring is used
            try:
                mapper.ScalarVisibilityOn()
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray("rgb")
                mapper.SetColorModeToDirectScalars()
            except Exception:
                pass

            mapper.Modified()
            state["glyphs"] = new_glyphs

            if add_text:
                # Update the label with the same name
                plotter.add_text(f"Iteration: {t}", name="time-label")

            # Write this frame to the movie
            plotter.write_frame()

        # Close when done
        plotter.close()





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
    def render_points_frame(self, *, t: float, points, radius, colors, name: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _OpenGLVideoBackend(_BaseBackend):
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

    def render_points_frame(self, *, t: float, points, radius, colors, name: str, mesh_points=None, mesh_indices=None) -> None:
        self.renderer.clear()

        #TODO: Update instancer rather than clearing shape instancers
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

    def render_points_frame(self, *, t: float, points, radius, colors, name: str, mesh_points=None, mesh_indices=None) -> None:
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
    """
    Unified renderer:
      - backend="opengl": headless OpenGL -> video file
      - backend="usd":    USD stage -> .usd/.usda/.usdc file (no pixels/video)
    """

    def __init__(
        self,
        filename: str,
        max_particles: int,
        *,
        backend: Literal["opengl", "usd"] = "opengl",
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
        self._radii_f32  = wp.empty(self.max_particles, dtype=wp.float32, device=self.device)
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
        radii_in:   wp.array(dtype=wp.float32),
        A_in:       wp.array(dtype=wp.float32),
        I_in:       wp.array(dtype=wp.float32),
        points_out: wp.array(dtype=wp.vec3),
        radii_out:  wp.array(dtype=wp.float32),
        colors_out: wp.array(dtype=wp.vec3),
        particle_count: wp.int32,
        morph_mode: wp.int32,      # 0:A, 1:I, 2:A/(A+I)
        morph_scale: wp.float32,   # divide then clamp to [0,1]
        hue: wp.float32,           # fixed hue in [0,1]
    ):
        i = wp.tid()

        if i >= particle_count:
            points_out[i] = wp.vec3(0.0, 0.0, 0.0)
            radii_out[i] = wp.float32(0.0)
            colors_out[i] = wp.vec3(0.0, 0.0, 0.0)
            return

        c = centers_in[i]
        points_out[i] = wp.vec3(wp.float32(c[0]), wp.float32(c[1]), wp.float32(c[2]))
        radii_out[i]  = wp.float32(radii_in[i])

        a = wp.float32(A_in[i])
        b = wp.float32(I_in[i])
        eps = wp.float32(1e-8)

        m = a
        if morph_mode == 1:
            m = b
        elif morph_mode == 2:
            m = a / (a + b + eps)

        # guard against morph_scale=0
        s = wp.max(morph_scale, wp.float32(1e-8))
        m = wp.clamp(m / s, wp.float32(0.0), wp.float32(1.0))

        # HSV -> RGB (hue fixed, saturation=m, value=1)
        h = hue
        sat = m
        v = wp.float32(1.0)

        h6 = h * wp.float32(6.0)
        hi = wp.int32(wp.floor(h6))  # 0..5
        f  = h6 - wp.float32(hi)

        p = v * (wp.float32(1.0) - sat)
        q = v * (wp.float32(1.0) - sat * f)
        t = v * (wp.float32(1.0) - sat * (wp.float32(1.0) - f))

        r = v; g = t; bb = p
        if hi == 1:
            r = q; g = v; bb = p
        elif hi == 2:
            r = p; g = v; bb = t
        elif hi == 3:
            r = p; g = q; bb = v
        elif hi == 4:
            r = t; g = p; bb = v
        elif hi == 5:
            r = v; g = p; bb = q

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
    ) -> int:
        n = int(particle_count)
        n = max(0, min(n, self.max_particles))

        morph_mode = 0
        if morph.lower() in ("i", "inhibitor", "inhibitors"):
            morph_mode = 1
        elif morph.lower() in ("ratio", "a_over_a_plus_i", "a/(a+i)"):
            morph_mode = 2

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
            ],
            device=self.device,
        )
        return n

    def _render(self, t: float, n_active: int, mesh_points=None, mesh_indices=None) -> None:
        # render_points currently wants CPU-indexable arrays, so we copy here
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
            mesh_indices=mesh_indices
        )

    # ------------------------
    # public API
    # ------------------------
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
        mesh_points = None,
        mesh_indices = None,
    ) -> None:
        n = self._pack_gpu_buffers(
            centers_wp, radii_wp, A_wp, I_wp, particle_count,
            morph=morph, morph_scale=morph_scale, hue=hue,
        )

        self._render(t=float(t), n_active=n, mesh_points=mesh_points, mesh_indices=mesh_indices)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
