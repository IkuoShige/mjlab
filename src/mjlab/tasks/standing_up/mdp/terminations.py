"""Termination conditions for standing-up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate if episode length exceeds maximum.

  Args:
    env: The environment.

  Returns:
    Boolean tensor of shape (num_envs,).
  """
  return env.episode_length_buf >= env.max_episode_length


def dof_vel_out(
  env: ManagerBasedRlEnv,
  limit: float = 300.0,
  unactuated_steps: int = 30,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate if DOF velocity exceeds limit.

  Only applies after the unactuated phase.

  Args:
    env: The environment.
    limit: Maximum DOF velocity. Default 300 rad/s.
    unactuated_steps: Number of steps in unactuated phase. Default 30.
    asset_cfg: Asset configuration.

  Returns:
    Boolean tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  max_vel = torch.max(torch.abs(asset.data.joint_vel), dim=-1)[0]
  past_unactuated = env.episode_length_buf > unactuated_steps
  return (max_vel > limit) & past_unactuated


def base_vel_out(
  env: ManagerBasedRlEnv,
  limit: float = 20.0,
  unactuated_steps: int = 30,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate if base linear velocity exceeds limit.

  Only applies after the unactuated phase.

  Args:
    env: The environment.
    limit: Maximum base linear velocity. Default 20 m/s.
    unactuated_steps: Number of steps in unactuated phase. Default 30.
    asset_cfg: Asset configuration.

  Returns:
    Boolean tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  base_vel = torch.norm(asset.data.root_link_lin_vel_w[:, :3], dim=-1)
  past_unactuated = env.episode_length_buf > unactuated_steps
  return (base_vel > limit) & past_unactuated


def nan_in_physics(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate if NaN detected in physics state.

  Checks for NaN in joint positions, velocities, and base pose.
  This helps catch physics instabilities early.

  Args:
    env: The environment.
    asset_cfg: Asset configuration.

  Returns:
    Boolean tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Check joint positions and velocities.
  joint_pos_nan = torch.isnan(asset.data.joint_pos).any(dim=-1)
  joint_vel_nan = torch.isnan(asset.data.joint_vel).any(dim=-1)

  # Check base pose.
  root_pos_nan = torch.isnan(asset.data.root_link_pos_w).any(dim=-1)
  root_quat_nan = torch.isnan(asset.data.root_link_quat_w).any(dim=-1)

  return joint_pos_nan | joint_vel_nan | root_pos_nan | root_quat_nan
