"""Booster K1 LVDRS-style kick task registration."""

from mjlab.tasks.kick.config.booster_k1.env_cfgs import k1_kick_env_cfg
from mjlab.tasks.kick.config.booster_k1.rl_cfg import (
  k1_kick_amp_ppo_runner_cfg,
  k1_kick_ppo_runner_cfg,
)
from mjlab.tasks.kick.rl.runner import KickAMPRunner
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Kick-Flat-Booster-K1",
  env_cfg=k1_kick_env_cfg(),
  play_env_cfg=k1_kick_env_cfg(play=True),
  rl_cfg=k1_kick_ppo_runner_cfg(),
)

# V2: PPO + AMP discriminator + mirror symmetry (LVDRS-style).
register_mjlab_task(
  task_id="Mjlab-Kick-AMP-Flat-Booster-K1",
  env_cfg=k1_kick_env_cfg(),
  play_env_cfg=k1_kick_env_cfg(play=True),
  rl_cfg=k1_kick_amp_ppo_runner_cfg(),
  runner_cls=KickAMPRunner,
)
