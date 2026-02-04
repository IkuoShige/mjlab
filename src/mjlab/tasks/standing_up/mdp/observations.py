"""Observation functions for standing-up task.

Observation structure (43 dim per frame):
- base_ang_vel (3) × scale 0.25
- projected_gravity (3)
- joint_pos (12) × scale 1.0
- joint_vel (12) × scale 0.05
- actions (12)
- action_rescale (1)

Total with 6-frame history: 43 × 6 = 258 features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_ang_vel_scaled(
  env: ManagerBasedRlEnv,
  scale: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Base angular velocity in body frame, scaled.

  Args:
    env: The environment.
    scale: Scaling factor for angular velocity. Default 0.25.
    asset_cfg: Asset configuration.

  Returns:
    Tensor of shape (num_envs, 3).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b * scale


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gravity vector projected into body frame.

  Args:
    env: The environment.
    asset_cfg: Asset configuration.

  Returns:
    Tensor of shape (num_envs, 3).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


def joint_pos_scaled(
  env: ManagerBasedRlEnv,
  scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint positions, scaled.

  Args:
    env: The environment.
    scale: Scaling factor for joint positions. Default 1.0.
    asset_cfg: Asset configuration.

  Returns:
    Tensor of shape (num_envs, num_joints).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids] * scale


def joint_vel_scaled(
  env: ManagerBasedRlEnv,
  scale: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint velocities, scaled.

  Args:
    env: The environment.
    scale: Scaling factor for joint velocities. Default 0.05.
    asset_cfg: Asset configuration.

  Returns:
    Tensor of shape (num_envs, num_joints).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids] * scale


def last_action(
  env: ManagerBasedRlEnv,
  action_name: str | None = None,
) -> torch.Tensor:
  """Previous actions from the action manager.

  Args:
    env: The environment.
    action_name: Optional name of specific action term. If None, returns all actions.

  Returns:
    Tensor of shape (num_envs, num_actions).
  """
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


def action_rescale(
  env: ManagerBasedRlEnv,
  noise_scale: float = 0.05,
) -> torch.Tensor:
  """Current action rescale value with optional noise.

  This observation provides the current action scale used in the curriculum.
  The action scale decreases as the robot learns to stand up better.

  Args:
    env: The environment.
    noise_scale: Scale of uniform noise to add. Default 0.05.

  Returns:
    Tensor of shape (num_envs, 1).
  """
  # Get action_rescale from env if it exists, otherwise return 1.0.
  if hasattr(env, "action_rescale"):
    base_value: torch.Tensor = env.action_rescale  # type: ignore[attr-defined]
  else:
    base_value = torch.ones(env.num_envs, 1, device=env.device)

  # Add noise: value + (rand - 0.5) * noise_scale.
  if noise_scale > 0:
    noise = (torch.rand_like(base_value) - 0.5) * noise_scale
    return base_value + noise
  return base_value
