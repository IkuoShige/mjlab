"""Termination conditions for the kick task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.kick.mdp.commands import KickTargetCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf >= env.max_episode_length


def robot_fell_height(
  env: ManagerBasedRlEnv,
  min_height: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < min_height


def robot_fell_orientation(
  env: ManagerBasedRlEnv,
  max_tilt: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when trunk tilt exceeds ``max_tilt`` radians from upright."""
  asset: Entity = env.scene[asset_cfg.name]
  g = asset.data.projected_gravity_b
  tilt = torch.acos((-g[:, 2]).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
  return tilt > max_tilt


def ball_out_of_range(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  max_distance: float = 8.0,
) -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, KickTargetCommand)
  return term.robot_ball_distance > max_distance


def kick_completed(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  hold_steps: int = 10,
) -> torch.Tensor:
  """Terminate ``hold_steps`` after the first valid kick contact in the episode.

  Stops the policy from chasing the ball after striking it, which produced
  unnatural follow-through motion in viser playback. The hold window lets the
  ball travel briefly so kick-quality rewards (ball direction alignment, etc.)
  can still register before episode end.
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, KickTargetCommand)
  return term.steps_since_kick >= hold_steps
