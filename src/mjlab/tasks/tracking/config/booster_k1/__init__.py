from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  booster_k1_flat_multimotion_tracking_env_cfg,
  booster_k1_flat_tracking_env_cfg,
)
from .rl_cfg import booster_k1_tracking_ppo_runner_cfg

# Single-motion BeyondMimic tracking (one --motion-file or --registry-name).
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Booster-K1",
  env_cfg=booster_k1_flat_tracking_env_cfg(),
  play_env_cfg=booster_k1_flat_tracking_env_cfg(play=True),
  rl_cfg=booster_k1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Booster-K1-No-State-Estimation",
  env_cfg=booster_k1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=booster_k1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=booster_k1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Multi-motion tracking (HumanoidSoccer Stage 1) — trains on all retargeted
# K1 kick clips in motions/soccer-standard-mj-k1/. Use as Stage 1 pre-training
# before Mjlab-Soccer-Kick-Flat-Booster-K1.
register_mjlab_task(
  task_id="Mjlab-Tracking-MultiMotion-Flat-Booster-K1",
  env_cfg=booster_k1_flat_multimotion_tracking_env_cfg(),
  play_env_cfg=booster_k1_flat_multimotion_tracking_env_cfg(play=True),
  rl_cfg=booster_k1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
