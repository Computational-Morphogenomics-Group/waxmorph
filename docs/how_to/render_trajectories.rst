Render trajectories
===================

Pick a renderer from :mod:`waxmorph.render` to match how you want to inspect a
trajectory — a quick static view, an interactive session, or an exported movie.

``MPLInterface``
   Render statically with Matplotlib for scripts and notebooks.

``PyVistaInterface``
   Inspect interactively with PyVista/VTK: spheroids, polarity arrows,
   signaling-molecule colors, and optional cell-type colors.

``WarpMovieRenderer``
   Export a trajectory through Warp, with ``backend="usd"`` for a USD stage and
   ``backend="opengl"`` for a headless video file.

Rendering a rollout movie
-------------------------

Feed simulator or emulator states directly to a renderer:

.. code-block:: python

   from pathlib import Path

   import numpy as np
   from waxmorph.render import WarpMovieRenderer

   trajectory = result.log["best_traj_pos"]  # [time, particles, 3]
   colors = np.full((n_cells, 3), [0.3, 0.6, 0.9], dtype=np.float32)

   Path("Output").mkdir(exist_ok=True)
   with WarpMovieRenderer(
       filename="Output/shape_assembly.usd",
       backend="usd",
       max_particles=n_cells,
       fps=30,
       device="cuda",
   ) as movie:
       for frame, centers in enumerate(trajectory):
           movie.write_frame_from_numpy(
               t=float(frame),
               centers=centers,
               radii=radii,
               colors=colors,
               particle_count=n_cells,
           )
