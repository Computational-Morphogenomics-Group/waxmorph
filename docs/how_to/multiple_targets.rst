Supervise multiple target frames
================================

Use multi-target training to supervise selected intermediate point clouds. A
final-frame run passes one ``(t_rollout - 1, target_pos)`` pair through the same
``targets`` argument, as in :ref:`mesh-to-emulator`.

Sample the intermediate shapes with
:func:`waxmorph.data.sample_mesh_sequence`, which accepts ``(frame, path)``
pairs, then pass a list of ``(frame, positions)`` targets to ``train``:

.. code-block:: python

   from waxmorph import TrainConfig, train, chamfer_distance

   config = TrainConfig(t_rollout=50, n_epochs=500)

   result = train(
       model,
       optimizer,
       chamfer_distance,
       source_pos=source_pos,
       polarities=polarities,
       c=c,
       radii=radii,
       targets=[
           (9, target_pos_early),
           (24, target_pos_mid),
           (49, target_pos_final),
       ],
       config=config,
       device="cuda",
   )

.. important::

   Frame ``k`` supervises the state after ``k + 1`` complete rollout steps in
   each model evaluation. Each step includes the learned update and configured
   mechanics and diffusion. Optimizer updates use the epoch axis; target indices
   use the within-trajectory frame axis.

Frames must be unique integers in ``[0, t_rollout)``; each target must be a
nonempty ``[M, 3]`` point cloud.

``lambda_reg`` weights
``sum_t ||dt_gns * GNS_X,t||_F^2``, measured after the learned position head and
before mechanics. The regularizer covers learned displacement, while the
physics update governs prescribed mechanical displacement. Histories
contain ``n_epochs`` post-update entries; one extra rollout evaluates the final
update, and the selected model, loss, and trajectory align.
