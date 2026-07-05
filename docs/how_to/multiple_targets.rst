Supervise multiple target frames
================================

Use multi-target training when you want the rollout to pass through prescribed
intermediate morphologies. Both single- and multi-target runs use the same
``targets`` argument — a final-only run passes one ``(t_rollout - 1, target_pos)``
pair, as in :ref:`mesh-to-emulator`.

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

   Frame indices are zero-based rollout steps *after* updates, so frame ``9``
   constrains the state after ten learned updates and frame ``49`` the state after
   fifty. With ``TrainConfig(t_rollout=100)``, ``(99, target_pos)`` therefore
   supervises the state after 100 learned updates.

``lambda_reg`` weights the squared-displacement regularization that penalizes
large inter-frame displacement, encouraging smoother deformations. The training
log carries the per-epoch losses and best rollout trajectories described in
:ref:`mesh-to-emulator`.
