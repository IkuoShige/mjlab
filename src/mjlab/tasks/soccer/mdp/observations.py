"""Soccer-specific observation functions.

Ported from HumanoidSoccer (Kong et al., 2026).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.soccer.mdp.commands import SoccerMotionCommand
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_command(env: ManagerBasedRlEnv, name: str) -> SoccerMotionCommand:
  return env.command_manager.get_term(name)  # type: ignore[return-value]


def get_target_point_world(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  command = _get_command(env, command_name)
  return command.target_point_pos + env.scene.env_origins


def get_target_point_base(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  command = _get_command(env, command_name)
  target_world = get_target_point_world(env, command_name)
  delta = target_world - command.robot_pelvis_pos_w
  return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def constant_target_point_pos(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Ball position in robot base frame."""
  return get_target_point_base(env, command_name)


def blind_zone_target_point_pos(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Ball position with blind-zone simulation.

  Ball invisible when robot-ball distance is outside [min, max].
  Returns last visible position when in blind zone.
  """
  command = _get_command(env, command_name)
  target_base = get_target_point_base(env, command_name)
  target_world = get_target_point_world(env, command_name)
  robot_pos = command.robot_pelvis_pos_w
  distance_xy = torch.norm(target_world[:, :2] - robot_pos[:, :2], dim=-1)

  in_visible = (distance_xy >= command.blind_distance_min) & (
    distance_xy <= command.blind_distance_max
  )

  if torch.any(in_visible):
    command.last_visible_target_point_base[in_visible] = target_base[in_visible]
    command.is_in_blind_zone[in_visible] = False

  command.is_in_blind_zone[~in_visible] = True

  return torch.where(
    command.is_in_blind_zone.unsqueeze(-1),
    command.last_visible_target_point_base,
    target_base,
  )


def target_destination_pos_local(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Goal/destination position in robot base frame."""
  command = _get_command(env, command_name)
  target_world = command.target_destination_pos + env.scene.env_origins
  delta = target_world - command.robot_pelvis_pos_w
  return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def target_point_pos_first_frame(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """First-frame cached ball position in robot base frame."""
  cache_name = f"_soccer_{command_name}_target_cache"
  target_local = get_target_point_base(env, command_name)

  cache = getattr(env, cache_name, None)
  if cache is None or cache.shape[0] != env.num_envs:
    cache = target_local.clone()
    setattr(env, cache_name, cache)

  first_step_mask = env.episode_length_buf == 0
  if torch.any(first_step_mask):
    cache = getattr(env, cache_name)
    cache[first_step_mask] = target_local[first_step_mask]
    setattr(env, cache_name, cache)

  return getattr(env, cache_name)


def target_destination_pos_local_first_frame(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """First-frame cached destination position in robot base frame."""
  cache_name = f"_soccer_{command_name}_dest_cache"
  target_local = target_destination_pos_local(env, command_name)

  cache = getattr(env, cache_name, None)
  if cache is None or cache.shape[0] != env.num_envs:
    cache = target_local.clone()
    setattr(env, cache_name, cache)

  first_step_mask = env.episode_length_buf == 0
  if torch.any(first_step_mask):
    cache = getattr(env, cache_name)
    cache[first_step_mask] = target_local[first_step_mask]
    setattr(env, cache_name, cache)

  return getattr(env, cache_name)
