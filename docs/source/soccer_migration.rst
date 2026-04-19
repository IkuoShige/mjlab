Humanoid Soccer Migration
=========================

Migration of `HumanoidSoccer <https://arxiv.org/abs/2602.05310>`_ (PAiD framework)
from Isaac Lab to mjlab (MuJoCo-Warp).

Status
------

.. list-table::
   :header-rows: 1
   :widths: 15 25 15 45

   * - Stage
     - Task ID
     - Status
     - Notes
   * - 2
     - ``Mjlab-Soccer-Kick-Flat-Unitree-G1``
     - Done
     - Kick training with blind-zone observation. 100k iter.
   * - 2.5
     - ``Mjlab-Soccer-Moving-Flat-Unitree-G1``
     - Done
     - Moving ball (initial velocity). 20k fine-tune from Kick.
   * - 3a
     - ``Mjlab-Soccer-Teacher-Flat-Unitree-G1``
     - Partial
     - Teacher with privileged actor obs. Stopped at 6.5k/100k.
   * - 3c
     - ``Mjlab-Soccer-Distill-Flat-Unitree-G1``
     - Blocked
     - Student-Teacher distillation. Fixed ``init_std=0.05`` (was 1.0). Needs re-run after Teacher completes.

Known Issues
------------

Double-kick behavior (fixed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: During play, the robot kicks the ball twice per episode.
The first kick is weak (foot contacts ball during approach),
the second kick is the intended motion-capture kick but misses because the
ball has already moved.

**Root cause**: The ``target_point_contact`` reward uses a one-shot
``KickContactTracker``. Any foot-ball contact above the force threshold
is treated as "the kick", consuming the one-shot reward. When the robot
approaches the ball, incidental foot-ball contact during the walking phase
triggers the tracker before the actual kick motion.

**Fix applied**: Two-layer filtering in ``KickContactTracker.detect()``:

1. Force threshold raised from 10 N to 30 N (filters light brushes).
2. Foot velocity gate: ``min_foot_speed=1.0`` m/s. Only registers a kick
   when at least one foot moves faster than 1 m/s. Approach contacts happen
   at low foot speed; the actual kick swing is fast.

Requires retraining from Stage 2.

Motion Data
-----------

Isaac Lab records motion data in **BFS (breadth-first)** body/joint ordering,
while MuJoCo uses **DFS (depth-first)** ordering. Motion ``.npz`` files must
be converted before use.

Conversion script::

    uv run python scripts/tools/convert_isaaclab_motion.py \
        --input motions/soccer-standard \
        --output motions/soccer-standard-mj

The converted files live in ``motions/soccer-standard-mj/`` and are referenced
by the default env config.

Distillation ``init_std`` fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RSL-RL's ``Distillation`` algorithm trains only the action **mean** via MSE
loss. The distribution std is not updated. With ``init_std=1.0`` (original),
deployed actions were effectively random, causing reward collapse
(mean reward -238k). Fixed to ``init_std=0.05`` in
``src/mjlab/tasks/soccer/config/g1/rl_cfg.py``.

Architecture
------------

.. code-block:: text

    src/mjlab/
    ├── asset_zoo/objects/soccer_ball/   # Ball MJCF + EntityCfg
    ├── tasks/soccer/
    │   ├── mdp/
    │   │   ├── commands.py              # SoccerMotionCommand (multi-motion loader)
    │   │   ├── kick_detection.py        # KickContactTracker
    │   │   ├── observations.py          # Ball pos, blind zone, destination
    │   │   ├── rewards.py               # Kick, speed, direction rewards
    │   │   └── terminations.py          # motion_finished
    │   └── config/g1/
    │       ├── env_cfgs.py              # All env config variants
    │       ├── rl_cfg.py                # PPO + Distillation configs
    │       └── __init__.py              # Task registration
    └── rl/
        └── config.py                    # RslRlDistillationRunnerCfg (added)

    scripts/tools/
    └── convert_isaaclab_motion.py       # BFS->DFS motion converter

Training Commands
-----------------

See README.md section "3. Humanoid Soccer (PAiD Framework)" for full
command examples.

Checkpoints (latest correct motion data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Stage
     - Run directory
     - Checkpoint
   * - 2 Kick
     - ``2026-04-05_03-10-28``
     - ``model_99999.pt``
   * - 2.5 Moving
     - ``2026-04-06_04-43-20``
     - ``model_119998.pt``
   * - 3a Teacher
     - ``2026-04-06_09-48-25``
     - ``model_6500.pt`` (incomplete)
