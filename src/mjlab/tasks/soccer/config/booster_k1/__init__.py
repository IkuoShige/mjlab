from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  k1_flat_soccer_distill_env_cfg,
  k1_flat_soccer_kick_env_cfg,
  k1_flat_soccer_moving_env_cfg,
  k1_flat_soccer_student_env_cfg,
  k1_flat_soccer_teacher_env_cfg,
  k1_flat_soccer_tracking_env_cfg,
)
from .rl_cfg import (
  k1_soccer_distillation_runner_cfg,
  k1_soccer_ppo_runner_cfg,
  k1_soccer_rnn_ppo_runner_cfg,
)

# Stage 1: multi-motion tracking in the soccer scene (ball present but kick
# rewards zeroed). Use as pre-training; resume Stage 2 from its checkpoint.
register_mjlab_task(
  task_id="Mjlab-Soccer-Tracking-Flat-Booster-K1",
  env_cfg=k1_flat_soccer_tracking_env_cfg(),
  play_env_cfg=k1_flat_soccer_tracking_env_cfg(play=True),
  rl_cfg=k1_soccer_rnn_ppo_runner_cfg(),
)

# Stage 2: MLP PPO (kept for back-compat / debugging).
register_mjlab_task(
  task_id="Mjlab-Soccer-Kick-Flat-Booster-K1",
  env_cfg=k1_flat_soccer_kick_env_cfg(),
  play_env_cfg=k1_flat_soccer_kick_env_cfg(play=True),
  rl_cfg=k1_soccer_ppo_runner_cfg(),
)

# Stage 2 (RNN): same env as Kick, LSTM PPO for resume from Stage 1 tracking.
register_mjlab_task(
  task_id="Mjlab-Soccer-Kick-RNN-Flat-Booster-K1",
  env_cfg=k1_flat_soccer_kick_env_cfg(),
  play_env_cfg=k1_flat_soccer_kick_env_cfg(play=True),
  rl_cfg=k1_soccer_rnn_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Soccer-Moving-Flat-Booster-K1",
  env_cfg=k1_flat_soccer_moving_env_cfg(),
  play_env_cfg=k1_flat_soccer_moving_env_cfg(play=True),
  rl_cfg=k1_soccer_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Soccer-Teacher-Flat-Booster-K1",
  env_cfg=k1_flat_soccer_teacher_env_cfg(),
  play_env_cfg=k1_flat_soccer_teacher_env_cfg(play=True),
  rl_cfg=k1_soccer_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Soccer-Student-Flat-Booster-K1",
  env_cfg=k1_flat_soccer_student_env_cfg(),
  play_env_cfg=k1_flat_soccer_student_env_cfg(play=True),
  rl_cfg=k1_soccer_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Soccer-Distill-Flat-Booster-K1",
  env_cfg=k1_flat_soccer_distill_env_cfg(),
  play_env_cfg=k1_flat_soccer_distill_env_cfg(play=True),
  rl_cfg=k1_soccer_distillation_runner_cfg(),
)
