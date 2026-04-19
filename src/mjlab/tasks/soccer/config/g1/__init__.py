from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  g1_flat_soccer_distill_env_cfg,
  g1_flat_soccer_kick_env_cfg,
  g1_flat_soccer_moving_env_cfg,
  g1_flat_soccer_student_env_cfg,
  g1_flat_soccer_teacher_env_cfg,
)
from .rl_cfg import g1_soccer_distillation_runner_cfg, g1_soccer_ppo_runner_cfg

# Stage 2: Soccer kick (blind zone observation).
register_mjlab_task(
  task_id="Mjlab-Soccer-Kick-Flat-Unitree-G1",
  env_cfg=g1_flat_soccer_kick_env_cfg(),
  play_env_cfg=g1_flat_soccer_kick_env_cfg(play=True),
  rl_cfg=g1_soccer_ppo_runner_cfg(),
)

# Stage 2.5: Moving ball variant.
register_mjlab_task(
  task_id="Mjlab-Soccer-Moving-Flat-Unitree-G1",
  env_cfg=g1_flat_soccer_moving_env_cfg(),
  play_env_cfg=g1_flat_soccer_moving_env_cfg(play=True),
  rl_cfg=g1_soccer_ppo_runner_cfg(),
)

# Stage 3a: Teacher policy with privileged actor observations.
register_mjlab_task(
  task_id="Mjlab-Soccer-Teacher-Flat-Unitree-G1",
  env_cfg=g1_flat_soccer_teacher_env_cfg(),
  play_env_cfg=g1_flat_soccer_teacher_env_cfg(play=True),
  rl_cfg=g1_soccer_ppo_runner_cfg(),
)

# Stage 3b: Student policy with first-frame-only observations (PPO direct).
register_mjlab_task(
  task_id="Mjlab-Soccer-Student-Flat-Unitree-G1",
  env_cfg=g1_flat_soccer_student_env_cfg(),
  play_env_cfg=g1_flat_soccer_student_env_cfg(play=True),
  rl_cfg=g1_soccer_ppo_runner_cfg(),
)

# Stage 3c: Student-Teacher distillation.
# Student obs = actor group, Teacher obs = critic group (privileged).
register_mjlab_task(
  task_id="Mjlab-Soccer-Distill-Flat-Unitree-G1",
  env_cfg=g1_flat_soccer_distill_env_cfg(),
  play_env_cfg=g1_flat_soccer_distill_env_cfg(play=True),
  rl_cfg=g1_soccer_distillation_runner_cfg(),
)
