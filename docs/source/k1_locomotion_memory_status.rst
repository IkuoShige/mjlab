.. _k1-locomotion-memory-status:

Booster K1 Locomotion-Memory Status
===================================

Snapshot date: April 15, 2026

This page records the current implementation status of the Booster K1
locomotion-memory project in ``mjlab`` against the roadmap defined in
``research.md``.

For restarting the work in another environment, see
:doc:`k1_locomotion_memory_handoff`. For the research roadmap and milestone
criteria, see :doc:`k1_locomotion_memory_roadmap`.

Current outcome
---------------

- A first usable K1 locomotion-memory training stack exists in ``mjlab``.
- The policy can learn forward locomotion with command tracking and partial
  turning behavior.
- The method is still short of the target quality for human-like straight
  walking at higher speed, reliable stepping in place during turns, stable
  cadence, and low-yaw tracking symmetry.
- The current best trained baseline is
  ``2026-04-14_18-37-50_k1_locomotion_memory_tracking_balance_v3_steady_forward_v1``.
- ``tracking_balance_v4`` has been implemented and smoke-tested, but the long
  training run has not been run yet.

Work history
------------

1. Added Booster K1 to the asset zoo and exposed K1 velocity and tracking task
   registrations.
2. Added ``csv_to_npz --robot k1`` and a local
   ``prepare-k1-locomotion-memory`` preprocessing pipeline.
3. Built a first-pass structured memory database from the retargeted K1 motion
   corpus under ``/workspace/k1_retarget/motions_k1/k1_fixed``.
4. Added experimental locomotion-memory tasks for Booster K1 on flat and rough
   terrain.
5. Added proposal-conditioned execution using retrieved snippet joint targets
   plus policy residual actions.
6. Added retrieval-conditioned observations, memory validity rewards, and
   retrieval metrics.
7. Fixed a critical K1 motion conversion bug by treating the retargeted root
   quaternion order as ``wxyz`` instead of ``xyzw``.
8. Synced the vendored K1 XML hip-pitch body offsets with ``booster_assets``
   and rebuilt the memory database after the asset correction.
9. Locked upper-body residual actions for locomotion-memory so arms and head no
   longer receive policy residuals during locomotion training.
10. Added upper-body collision proxies, enabled non-foot collision geoms, and
    re-enabled foot-foot collision so self-contact is represented more
    faithfully.
11. Added straight-walk foot placement shaping to penalize feet crowding the
    body midline and sweeping inward during straight walking.
12. Added retrieval-side turn intent bias, a minimum snippet hold time, and a
    turn-specific foot air-time bonus to improve in-place turning and reduce
    cadence-breaking snippet churn.
13. Added fixed-command FK evaluation via ``evaluate-locomotion-memory-fk`` to
    compare executed foot motion against the active memory reference.
14. Extended the memory database with root-frame foot-link trajectories and
    added ``reference_swing_foot_position`` so swing feet are penalized when
    they drift from the retrieved reference path.
15. Added ``research`` fixed-command FK evaluation presets covering forward
    sweeps, in-place yaw sweeps, and arc commands.
16. Compared ``tracking_balance_v1`` and ``tracking_balance_v2``. ``v2``
    improved cadence slightly, but regressed reward, forward velocity tracking,
    and in-place yaw FK execution.
17. Added a velocity-lag gate to forward scuff and swing-recovery shaping so
    auxiliary foot-quality costs do not suppress high-speed command tracking.
18. Included the newly cut steady high-speed K1 clips in the locomotion-memory
    selector and rebuilt the default memory database from 89 clips. Straight-ish
    ``vx >= 1.5`` snippet support increased from 7 snippets to 57 snippets.
19. Evaluated ``tracking_balance_v3_steady_forward_v1``. The new steady memory
    improved forward speed tracking, but exposed three next bottlenecks:
    straight-command yaw drift, return-swing scuff at high speed, and weak
    right-side low-yaw turning.
20. Added ``tracking_balance_v4`` changes: a straight-reference quality bias in
    retrieval, a straight-yaw tracking reward, and a low-yaw turn tracking
    reward with left/right diagnostic logs.

Roadmap status
--------------

Module A. Structured locomotion memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Status: first-pass implemented

- Snippetization matches the research spec: 0.8 s snippets, 24 frames,
  stride 6, 30 Hz.
- Contact re-annotation, phase labeling, gait labels, quality score, and
  transition flags are implemented in the memory builder.
- The memory database now stores ``local_foot_pos_seq`` for left/right foot-link
  trajectories, which is used by execution-side FK shaping.
- The current K1 memory database uses a focused motion subset: walk, jog, run,
  accel/decel, start-stop, and turn clips.
- The current default K1 memory database contains 1660 snippets from 89 clips.
  After the April 14 steady-forward update, ``vx >= 1.5`` support increased
  from 31 to 126 snippets, and straight-ish ``vx >= 1.5`` support increased
  from 7 to 57 snippets.
- The source set still omits lateral, backward, shuffle, and broad style
  coverage by design.
- ``stop`` support is incomplete in the current database. The first-pass memory
  file reports no dedicated stop snippets.

Module B. Feasibility-aware retriever
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Status: first-pass implemented, still under-tuned

- Retrieval is top-1, not mixture-based.
- The score uses command fit, phase consistency, contact compatibility,
  connection cost, quality score, and transition bias.
- Hysteresis-based switching is implemented.
- A minimum snippet hold time is now enforced to reduce rapid retrieval churn.
- A turn-intent bias is now applied for low-speed yaw commands so in-place turn
  snippets can win against low-speed walk snippets.
- Remaining issue: turn retrieval is still not yet strong enough to guarantee
  clear stepping in place across the full command range.

Module C. Proposal encoder
^^^^^^^^^^^^^^^^^^^^^^^^^^

Status: implemented

- Retrieved snippets are encoded into future proposal vectors and snippet
  metadata for actor and critic observations.
- The proposal remains a structured suggestion rather than a hard trajectory
  target.

Module D. Proposal-conditioned execution policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Status: implemented, still below target gait quality

- The execution stack is based on the velocity task surface, not tracking.
- The action path uses retrieved joint references plus residual policy output.
- Velocity tracking, upright behavior, and locomotion rewards are wired for K1.
- Upper-body residuals are locked out for locomotion-memory training.
- Swing-foot execution now has a direct FK-style cost against the retrieved
  reference foot-link trajectory.
- Remaining issue: gait quality is still not yet consistently human-like at
  higher speed, especially for straight foot lanes, knee usage, and turning.

Evaluation and baselines
^^^^^^^^^^^^^^^^^^^^^^^^

Status: partial

- Core locomotion-memory metrics exist: retrieval switch count, contact-valid
  rate, phase-valid rate, and transition-snippet rate.
- ``evaluate-locomotion-memory-fk`` provides a focused checkpoint evaluation
  pass for comparing executed K1 foot FK against the active memory reference
  under fixed velocity commands.
- The ``research`` preset now covers forward ``vx`` sweeps, symmetric in-place
  yaw sweeps, and mixed forward-yaw arc commands.
- Training, play, and W&B runs are working.
- The baselines from ``research.md`` are not fully implemented yet:
  command-only, hard reference tracking, and retrieval-only hard tracking are
  still missing as formal comparison tasks.
- The broader evaluation suite from ``research.md`` is only partially covered.
  There is no finalized benchmark pass yet for naturalness, executability, and
  transition quality.

What is working now
-------------------

- K1 assets, training, and play entrypoints are integrated.
- Memory generation and loading are robust against missing or mismatched files.
- A locomotion-memory policy can be trained end-to-end with W&B or TensorBoard.
- Forward walking is trainable and usable.
- Some turning behavior is present.

Main open problems
------------------

1. Straight walking still gets too close to the midline at higher speed.
2. In-place turning is still weak and sometimes fails to produce clear stepping.
3. Cadence occasionally becomes irregular, which suggests retrieval timing and
   phase continuity are still not strong enough.
4. Human-likeness is still below target in foot placement, knee usage, and
   overall gait expression.
5. ``tracking_balance_v2`` showed that aggressive foot-quality shaping can
   reduce task performance. Future variants should keep velocity and yaw
   tracking as the primary objective and gate auxiliary foot costs carefully.

Recommended next milestones
---------------------------

1. Quantify cadence stability and turn quality with explicit episode metrics
   instead of judging only by play-mode inspection.
2. Use ``evaluate-locomotion-memory-fk`` to separate memory-reference quality
   issues from execution-policy issues before further reward tuning.
3. Treat ``tracking_balance_v3_steady_forward_v1`` as the current trained
   baseline. It improved forward speed tracking after the steady-forward memory
   update, but regressed straight-command yaw drift and return-swing scuff.
4. Train and compare the current ``tracking_balance_v4`` configuration. Key
   success signals are lower straight-command ``yaw_abs_error_mean``, lower
   ``actual_return_swing_scuff_frac_z_lt_0.03`` at ``vx=1.5``, and improved
   right-side ``yaw_r_0.10``/``yaw_r_0.15`` tracking without sacrificing the
   forward ``vx_abs_error_mean`` improvement from v3.
5. Strengthen turn support with either better command sampling for near-zero
   forward speed yaw commands or a more explicit turn-state curriculum.
6. Tighten lower-body action and pose shaping around ``Hip_Yaw`` and knee usage
   if high-speed straight walking still sweeps inward after the current reward
   changes.
7. Implement the planned baselines so the locomotion-memory method can be
   compared against command-only and hard-tracking alternatives.
8. Expand evaluation to the four axes in ``research.md``: task success,
   naturalness, executability, and memory validity.
