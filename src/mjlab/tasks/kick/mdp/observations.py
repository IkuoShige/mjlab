"""Observation functions for the kick task.

Actor receives noisy / masked ball state via the virtual perception system.
Critic receives ground-truth ball state and privileged ball physics
parameters (mass, friction) for asymmetric actor-critic training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.kick.mdp.commands import KickTargetCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_command(env: ManagerBasedRlEnv, name: str) -> KickTargetCommand:
  term = env.command_manager.get_term(name)
  if not isinstance(term, KickTargetCommand):
    raise TypeError(
      f"Command {name!r} is {type(term).__name__}, expected KickTargetCommand"
    )
  return term


def ball_pos_b_perceived(
  env: ManagerBasedRlEnv, command_name: str = "kick_target"
) -> torch.Tensor:
  return _get_command(env, command_name).ball_pos_b_perceived


def ball_mask(
  env: ManagerBasedRlEnv, command_name: str = "kick_target"
) -> torch.Tensor:
  return _get_command(env, command_name).ball_mask.unsqueeze(-1)


def ball_pos_b(
  env: ManagerBasedRlEnv, command_name: str = "kick_target"
) -> torch.Tensor:
  return _get_command(env, command_name).ball_pos_b


def ball_vel_b(
  env: ManagerBasedRlEnv, command_name: str = "kick_target"
) -> torch.Tensor:
  return _get_command(env, command_name).ball_vel_b


def kick_target_dir_b(
  env: ManagerBasedRlEnv, command_name: str = "kick_target"
) -> torch.Tensor:
  return _get_command(env, command_name).kick_target_dir_b


def root_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def ball_physics_normalized(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  mass_ref: float = 0.43,
  friction_ref: float = 1.0,
) -> torch.Tensor:
  """Privileged ball physics state (mass, friction) normalized around defaults.

  Returns ``(num_envs, 2)`` shape: ``[mass / mass_ref - 1, friction / friction_ref - 1]``.
  Useful as critic observation when ball physics are randomized.
  """
  cmd = _get_command(env, command_name)
  ball = cmd.ball
  out = torch.zeros(env.num_envs, 2, device=env.device)
  mass = getattr(ball.data, "root_mass", None)
  friction = getattr(ball.data, "geom_friction", None)
  if mass is not None:
    out[:, 0] = mass.squeeze(-1) / mass_ref - 1.0
  if friction is not None:
    out[:, 1] = friction[..., 0].squeeze(-1) / friction_ref - 1.0
  return out
