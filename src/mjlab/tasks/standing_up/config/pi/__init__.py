"""Pi robot configuration for standing-up task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.standing_up.config.pi.env_cfgs import pi_standing_up_env_cfg
from mjlab.tasks.standing_up.config.pi.rl_cfg import (
  PI_STANDING_UP_RL_CFG,
  make_pi_standing_up_rl_cfg,
)
from mjlab.tasks.standing_up.rl import StandingUpRunner

# Register the Pi standing-up task.
register_mjlab_task(
  task_id="Mjlab-StandingUp-Pi",
  env_cfg=pi_standing_up_env_cfg(),
  play_env_cfg=pi_standing_up_env_cfg(play=True),
  rl_cfg=make_pi_standing_up_rl_cfg(),
  runner_cls=StandingUpRunner,
)

__all__ = [
  "pi_standing_up_env_cfg",
  "PI_STANDING_UP_RL_CFG",
  "make_pi_standing_up_rl_cfg",
]
