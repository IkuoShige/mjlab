.. _k1-locomotion-memory-handoff:

Booster K1 Locomotion-Memory Handoff
====================================

Snapshot date: April 15, 2026

This page is the practical handoff note for restarting the Booster K1
locomotion-memory work in a fresh environment. The broader implementation
history is tracked in :doc:`k1_locomotion_memory_status`; the research roadmap
is tracked in :doc:`k1_locomotion_memory_roadmap`.

Project goal
------------

The current research goal is to train a Booster K1 locomotion policy that:

- follows velocity commands;
- uses a structured memory built from retargeted human-like K1 motions;
- walks naturally at low and high forward speeds;
- turns in place by stepping rather than by dragging the feet;
- keeps the memory proposal as guidance, not as a hard tracking target.

The method is intentionally not a plain velocity-only PPO task and not a pure
motion-tracking task. It is a velocity-commanded task with feasibility-aware
retrieval and proposal-conditioned execution.

Required local inputs
---------------------

The current workspace assumes these paths exist:

- ``/workspace/mjlab``: this repository.
- ``/workspace/k1_retarget/motions_k1/k1_fixed``: retargeted K1 motion CSVs.
- ``/workspace/booster_assets``: upstream Booster K1 XML/assets used for model
  checks.
- ``/workspace/cite``: local reference repositories and papers, if available.

The default generated memory assets live under:

- ``/workspace/mjlab/artifacts/locomotion_memory/booster_k1/converted``:
  converted per-clip ``.npz`` motions.
- ``/workspace/mjlab/artifacts/locomotion_memory/booster_k1/locomotion_memory.npz``:
  default structured memory file loaded by the K1 locomotion-memory task.
- ``/workspace/mjlab/artifacts/locomotion_memory/booster_k1/backups``:
  manual backups of earlier memory files.

If these artifacts are missing or stale, rebuild them from the retargeted CSVs
before training.

Curated generated artifacts are intentionally tracked in Git under
``artifacts/locomotion_memory/booster_k1``:

- ``locomotion_memory.npz``: current default structured memory.
- ``artifact_manifest.json``: summary of the tracked generated artifacts.
- ``evaluations/*.json``: small FK evaluation outputs for baseline comparison.
- ``policies/tracking_balance_v3_steady_forward_v1_model_29999.pt``: current
  trained baseline checkpoint for immediate play/evaluation.
- ``policies/tracking_balance_v3_steady_forward_v1.onnx``: exported baseline
  policy.

The full ``logs/``, ``wandb/``, converted motion directory, and intermediate
checkpoints are deliberately not tracked.

Current implementation surface
------------------------------

The main code paths are:

- ``src/mjlab/asset_zoo/robots/booster_k1``: vendored Booster K1 MJCF assets.
- ``src/mjlab/scripts/csv_to_npz.py``: K1 motion CSV to tracking-compatible
  ``.npz`` conversion.
- ``src/mjlab/scripts/prepare_k1_locomotion_memory.py``: one-shot K1 motion
  conversion plus memory build entrypoint.
- ``src/mjlab/scripts/build_locomotion_memory.py``: structured memory builder.
- ``src/mjlab/tasks/locomotion_memory``: locomotion-memory task, MDP terms,
  configs, and task registration.
- ``src/mjlab/scripts/evaluate_locomotion_memory_fk.py``: fixed-command FK
  evaluation for trained policies.
- ``tests/test_locomotion_memory_*.py`` and ``tests/test_evaluate_locomotion_memory_fk.py``:
  focused regression coverage for this feature.

Default task ids:

- ``Mjlab-LocomotionMemory-Flat-Booster-K1``: primary flat-ground training task.
- ``Mjlab-LocomotionMemory-Rough-Booster-K1``: rough-terrain variant.

Memory state
------------

The current default memory was rebuilt after adding steady high-speed forward
clips from ``/workspace/k1_retarget``.

Current memory summary:

- 89 source clips.
- 1660 snippets.
- 126 snippets with ``vx >= 1.5``.
- 57 straight-ish snippets with ``vx >= 1.5``.
- High-speed steady support is selected from ``walk_fast_steady_*.csv``,
  ``run_steady_*.csv``, and ``steady_*.csv``.

Important naming rule: ``steady_*`` clips are treated as steady support even
when their source name contains words such as ``accel`` or ``decel``. This is
because the new files were cut from steady windows.

Rebuild the memory
------------------

Use this when ``/workspace/k1_retarget`` changes, when K1 assets change, or
when the memory file is missing:

.. code-block:: bash

   uv run prepare-k1-locomotion-memory --overwrite-memory True

Useful variants:

.. code-block:: bash

   uv run prepare-k1-locomotion-memory --overwrite-converted True --overwrite-memory True
   uv run prepare-k1-locomotion-memory --max-clips 20 --overwrite-memory True

The command defaults to:

- source directory:
  ``/workspace/k1_retarget/motions_k1/k1_fixed``;
- converted directory:
  ``/workspace/mjlab/artifacts/locomotion_memory/booster_k1/converted``;
- memory file:
  ``/workspace/mjlab/artifacts/locomotion_memory/booster_k1/locomotion_memory.npz``;
- target frame rate: 30 Hz;
- default device: ``cuda:0``.

Train
-----

Use capitalized booleans for CLI flags. ``False`` is valid; ``false`` is not.

Current next training run:

.. code-block:: bash

   uv run train Mjlab-LocomotionMemory-Flat-Booster-K1 \
     --agent.logger wandb \
     --agent.upload-model False \
     --env.scene.num-envs 4096 \
     --agent.run-name k1_locomotion_memory_tracking_balance_v4

Smoke test before a long run:

.. code-block:: bash

   uv run train Mjlab-LocomotionMemory-Flat-Booster-K1 \
     --agent.max-iterations 1 \
     --agent.logger tensorboard \
     --agent.upload-model False \
     --env.scene.num-envs 32 \
     --agent.run-name k1_locomotion_memory_tracking_balance_v4_smoke

The v4 code has passed this one-iteration smoke test. The long v4 training run
has not been run as of this snapshot.

Play
----

Use Viser for qualitative inspection:

.. code-block:: bash

   uv run play Mjlab-LocomotionMemory-Flat-Booster-K1 \
     --checkpoint-file logs/rsl_rl/k1_locomotion_memory/<run-dir>/model_29999.pt \
     --viewer viser \
     --num-envs 1 \
     --no-terminations True

To replay the tracked v3 baseline without a local training log directory:

.. code-block:: bash

   uv run play Mjlab-LocomotionMemory-Flat-Booster-K1 \
     --checkpoint-file artifacts/locomotion_memory/booster_k1/policies/tracking_balance_v3_steady_forward_v1_model_29999.pt \
     --viewer viser \
     --num-envs 1 \
     --no-terminations True

Inspect at least these command cases:

- straight forward walking at ``vx = 0.5``, ``1.0``, and ``1.5``;
- low-yaw in-place turns around ``abs(wz) = 0.10`` to ``0.20``;
- larger in-place turns around ``abs(wz) = 0.4`` to ``0.6``;
- mixed arc walking with nonzero forward speed and yaw.

Evaluate
--------

Run the fixed-command FK evaluation after each serious training run:

.. code-block:: bash

   uv run evaluate-locomotion-memory-fk \
     --checkpoint-file logs/rsl_rl/k1_locomotion_memory/<run-dir>/model_29999.pt \
     --command-preset research \
     --num-envs 64 \
     --steps 500 \
     --warmup-steps 100 \
     --output-file logs/rsl_rl/k1_locomotion_memory/<run-dir>/fk_eval_research_model_29999.json

Key FK metrics to compare:

- ``forward.vx_abs_error_mean``: forward speed tracking.
- ``forward.yaw_abs_error_mean``: straight-command yaw drift.
- ``forward.scuff_frac_mean``: return-swing scuff risk.
- ``in_place_yaw.yaw_abs_error_mean``: in-place turn tracking.
- per-command ``yaw_r_0.10`` and ``yaw_r_0.15``: right low-yaw turn weakness.

Key training metrics to compare:

- ``Train/mean_reward`` and ``Train/mean_episode_length``.
- ``Episode_Termination/fell_over`` and ``Episode_Termination/time_out``.
- ``Metrics/twist/error_vel_xy`` and ``Metrics/twist/error_vel_yaw``.
- ``Episode_Metrics/forward_scuff_risk`` if present.
- ``Episode_Metrics/low_yaw_turn_right_error`` if present.
- ``Episode_Reward/straight_yaw_velocity_tracking`` if present.

Current trained baseline
------------------------

Current best trained baseline:

``2026-04-14_18-37-50_k1_locomotion_memory_tracking_balance_v3_steady_forward_v1``

Local W&B run:

``wandb/run-20260414_183758-0om9sx31``

FK output:

``logs/rsl_rl/k1_locomotion_memory/2026-04-14_18-37-50_k1_locomotion_memory_tracking_balance_v3_steady_forward_v1/fk_eval_research_model_29999.json``

Tracked artifact copies:

- ``artifacts/locomotion_memory/booster_k1/policies/tracking_balance_v3_steady_forward_v1_model_29999.pt``
- ``artifacts/locomotion_memory/booster_k1/policies/tracking_balance_v3_steady_forward_v1.onnx``
- ``artifacts/locomotion_memory/booster_k1/evaluations/tracking_balance_v3_steady_forward_v1_fk_eval_research_model_29999.json``

Important v3 outcome:

- Forward speed tracking improved.
- ``forward.vx_abs_error_mean`` reached about ``0.082``.
- ``forward_vx_1.5`` executed at about ``1.370 m/s``.
- Straight-command yaw drift worsened.
- Forward return-swing scuff risk worsened.
- Right low-yaw turning remained weak.

Pending untrained changes
-------------------------

The current code contains v4 changes that were implemented after the v3
evaluation and are ready for the next training run:

- retrieval now has a straight-reference quality bias;
- the command term exposes ``straight_intent_mask``;
- the command term exposes reference swing-lift and straight-reference quality
  diagnostics;
- a straight-yaw velocity tracking reward was added;
- a low-yaw turn velocity tracking reward was added with left/right diagnostic
  logs.

These changes are intended to improve straight yaw stability and low-yaw turn
tracking without giving back the v3 forward speed improvement.

Known failure modes
-------------------

- Upside-down or ceiling-walking K1 motion usually means a root quaternion or
  asset-frame mismatch. The K1 converter currently treats retargeted root
  quaternions as ``wxyz``.
- If new motion CSVs are added but training behavior does not change, rebuild
  both converted motions and the memory file.
- If Viser shows old K1 geometry, verify the vendored K1 XML under ``mjlab``
  matches the intended ``booster_assets`` edit.
- If ``--agent.upload-model false`` fails, use ``--agent.upload-model False``.
- If training metrics improve but play mode looks worse, run FK evaluation
  before tuning rewards. The issue may be memory-reference quality, execution
  quality, or command tracking, and each needs a different fix.

Restart checklist
-----------------

1. Verify that ``/workspace/k1_retarget/motions_k1/k1_fixed`` contains the
   expected K1 CSVs.
2. Rebuild ``locomotion_memory.npz`` if the motion corpus or K1 asset changed.
3. Run the focused tests for commands, rewards, task registration, and FK
   evaluation.
4. Run the one-iteration v4 training smoke test.
5. Launch the long v4 W&B training run.
6. Run ``evaluate-locomotion-memory-fk`` on the final checkpoint.
7. Play the final checkpoint in Viser and compare the same command cases used
   for the v3 baseline.
8. Update :doc:`k1_locomotion_memory_status` with the new run name, metrics,
   and qualitative result.
