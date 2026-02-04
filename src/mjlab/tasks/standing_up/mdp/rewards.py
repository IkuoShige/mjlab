"""Reward functions for standing-up task.

Ported from HoST pi_host_ground.py reward functions.

Reward structure:
- Task rewards: Product aggregation (Gaussian product).
- Constraint/style rewards: Additive aggregation.
- Groups: ['task', 'regu', 'style', 'target'] with weights [2.5, 0.1, 1, 1].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _tolerance(
  x: torch.Tensor,
  bounds: tuple[float, float],
  margin: float,
  sigmoid_slope: float = 0.1,
) -> torch.Tensor:
  """Compute tolerance reward using sigmoid bounds.

  Returns 1 if x is within bounds, smoothly decays outside.

  Args:
    x: Input tensor.
    bounds: (lower, upper) bounds.
    margin: Margin for sigmoid decay.
    sigmoid_slope: Slope of sigmoid. Default 0.1.

  Returns:
    Reward tensor in [0, 1].
  """
  lower, upper = bounds
  in_bounds = (x >= lower) & (x <= upper)

  # Below lower bound.
  below = torch.sigmoid((x - lower + margin) / sigmoid_slope)
  # Above upper bound.
  above = torch.sigmoid((upper - x + margin) / sigmoid_slope)

  result = torch.where(
    in_bounds,
    torch.ones_like(x),
    torch.where(x < lower, below, above),
  )
  return result.squeeze(-1) if result.dim() > 1 and result.shape[-1] == 1 else result


##
# Progress rewards (dense signal for early learning).
##


def upright_progress(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense reward for making progress toward upright orientation.

  Rewards any increase in the negative z-component of projected gravity,
  which indicates progress toward standing upright. This provides learning
  signal even when lying down, unlike the sparse orientation reward.

  Args:
    env: The environment.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  # projected_gravity_b[:, 2] is negative when upright (-1 = fully upright)
  # We want to reward making this more negative
  upright_score = -asset.data.projected_gravity_b[:, 2]
  # Clamp to [0, 1] range (0 = lying down, 1 = upright)
  return upright_score.clamp(0.0, 1.0)


def base_height_progress(
  env: ManagerBasedRlEnv,
  target_height: float = 0.34,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense reward for base height progress.

  Rewards any increase in base height toward the target, providing
  continuous gradient for the standing-up motion.

  Args:
    env: The environment.
    target_height: Target standing height. Default 0.34m.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  base_height = asset.data.root_link_pos_w[:, 2]
  # Normalize: 0 at ground level, 1 at target height
  progress = (base_height / target_height).clamp(0.0, 1.0)
  return progress


##
# Task rewards (Gaussian product).
##


def orientation(
  env: ManagerBasedRlEnv,
  threshold: float = 0.99,
  phase1_height: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Orientation reward: robot should be upright.

  Uses tolerance on projected gravity z-component.

  Args:
    env: The environment.
    threshold: Gravity z threshold for being "upright". Default 0.99.
    phase1_height: Minimum base height to start applying this reward.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  proj_gravity_z = -asset.data.projected_gravity_b[:, 2]
  reward = _tolerance(proj_gravity_z, (threshold, float("inf")), 1.0, 0.05)
  return reward


def head_height(
  env: ManagerBasedRlEnv,
  target_height: float = 0.37,
  margin: float = 0.37,
  head_body_name: str = "keyframe_head_link",
  feet_body_names: tuple[str, ...] = ("l_ankle_roll_link", "r_ankle_roll_link"),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Head height reward relative to feet.

  Args:
    env: The environment.
    target_height: Target head height above feet. Default 0.37m.
    margin: Tolerance margin. Default 0.37m.
    head_body_name: Name of head body.
    feet_body_names: Names of feet bodies.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Get head body index.
  head_ids, _ = asset.find_bodies(head_body_name)
  if not head_ids:
    return torch.ones(env.num_envs, device=env.device)
  head_id = head_ids[0]

  # Get feet body indices.
  feet_ids = []
  for name in feet_body_names:
    ids, _ = asset.find_bodies(name)
    if ids:
      feet_ids.extend(ids)

  if not feet_ids:
    return torch.ones(env.num_envs, device=env.device)

  # Compute relative head height.
  body_ids = asset.indexing.body_ids
  head_height_w = env.sim.data.xpos[:, body_ids[head_id], 2]
  feet_heights = torch.stack(
    [env.sim.data.xpos[:, body_ids[fid], 2] for fid in feet_ids], dim=-1
  )
  feet_height_mean = feet_heights.mean(dim=-1)
  relative_height = head_height_w - feet_height_mean

  reward = _tolerance(relative_height, (target_height, float("inf")), margin, 0.1)
  return reward


##
# Regularization rewards (additive).
##


def dof_acc_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize DOF accelerations using L2 squared kernel.

  Args:
    env: The environment.
    asset_cfg: Asset configuration.

  Returns:
    Penalty tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(
  env: ManagerBasedRlEnv,
  unactuated_steps: int = 30,
) -> torch.Tensor:
  """Penalize rate of change of actions using L2 squared kernel.

  Uses masked actions (zero during unactuated phase) to match Isaac Gym behavior
  where self.actions is masked before being used in reward computation.

  Args:
    env: The environment.
    unactuated_steps: Number of unactuated steps. Actions are treated as zero
      during this phase.

  Returns:
    Penalty tensor of shape (num_envs,).
  """
  # Get raw actions.
  action = env.action_manager.action
  prev_action = env.action_manager.prev_action

  # Create masks for current and previous timestep.
  # Current mask: episode_length_buf > unactuated_steps
  # Previous mask: episode_length_buf - 1 > unactuated_steps (i.e., > unactuated_steps + 1)
  current_mask = (env.episode_length_buf > unactuated_steps).float().unsqueeze(-1)
  prev_mask = (env.episode_length_buf > unactuated_steps + 1).float().unsqueeze(-1)

  # Mask actions (zero during unactuated phase, like Isaac Gym).
  masked_action = action * current_mask
  masked_prev_action = prev_action * prev_mask

  return torch.sum(torch.square(masked_action - masked_prev_action), dim=1)


def smoothness(
  env: ManagerBasedRlEnv,
  unactuated_steps: int = 30,
) -> torch.Tensor:
  """Penalize action acceleration (second-order smoothness).

  Uses masked actions (zero during unactuated phase) to match Isaac Gym behavior
  where self.actions is masked before being used in reward computation.

  Args:
    env: The environment.
    unactuated_steps: Number of unactuated steps. Actions are treated as zero
      during this phase.

  Returns:
    Penalty tensor of shape (num_envs,).
  """
  # Get raw actions.
  action = env.action_manager.action
  prev_action = env.action_manager.prev_action
  prev_prev_action = env.action_manager.prev_prev_action

  # Create masks for current, previous, and prev-previous timesteps.
  current_mask = (env.episode_length_buf > unactuated_steps).float().unsqueeze(-1)
  prev_mask = (env.episode_length_buf > unactuated_steps + 1).float().unsqueeze(-1)
  prev_prev_mask = (env.episode_length_buf > unactuated_steps + 2).float().unsqueeze(-1)

  # Mask actions (zero during unactuated phase, like Isaac Gym).
  masked_action = action * current_mask
  masked_prev_action = prev_action * prev_mask
  masked_prev_prev_action = prev_prev_action * prev_prev_mask

  action_acc = masked_action - 2 * masked_prev_action + masked_prev_prev_action
  return torch.sum(torch.square(action_acc), dim=1)


def torques_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint torques using L2 squared kernel.

  Args:
    env: The environment.
    asset_cfg: Asset configuration.

  Returns:
    Penalty tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.actuator_force), dim=1)


def joint_power(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint power consumption.

  Args:
    env: The environment.
    asset_cfg: Asset configuration.

  Returns:
    Penalty tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  tau = asset.data.actuator_force
  return torch.sum(torch.abs(vel) * torch.abs(tau), dim=1)


def dof_vel_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize DOF velocities using L2 squared kernel.

  Args:
    env: The environment.
    asset_cfg: Asset configuration.

  Returns:
    Penalty tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_tracking_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint position tracking error.

  Args:
    env: The environment.
    asset_cfg: Asset configuration.

  Returns:
    Penalty tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  target = asset.data.joint_pos_target[:, asset_cfg.joint_ids]
  current = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(target - current), dim=1)


def dof_pos_limits(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize DOF positions near limits.

  Args:
    env: The environment.
    asset_cfg: Asset configuration.

  Returns:
    Penalty tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  soft_limits = asset.data.soft_joint_pos_limits
  assert soft_limits is not None
  pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

  out_of_limits = -(pos - soft_limits[:, asset_cfg.joint_ids, 0]).clip(max=0.0)
  out_of_limits += (pos - soft_limits[:, asset_cfg.joint_ids, 1]).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


##
# Style rewards (additive).
##


def hip_deviation(
  env: ManagerBasedRlEnv,
  joint_type: str = "yaw",
  max_threshold: float = 1.4,
  min_threshold: float = 0.9,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize hip joint deviations.

  Args:
    env: The environment.
    joint_type: Type of hip joint ("yaw" or "roll").
    max_threshold: Maximum absolute value threshold.
    min_threshold: Minimum absolute value threshold.
    asset_cfg: Asset configuration.

  Returns:
    Penalty tensor of shape (num_envs,) as float (1 if violated, 0 otherwise).
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Find joint indices by pattern.
  if joint_type == "yaw":
    pattern = ".*thigh_joint"  # Hip yaw in Pi is thigh_joint.
  else:
    pattern = ".*hip_roll_joint"

  joint_ids, _ = asset.find_joints(pattern)
  if not joint_ids:
    return torch.zeros(env.num_envs, device=env.device)

  pos = asset.data.joint_pos[:, joint_ids]
  max_abs = torch.max(torch.abs(pos), dim=-1)[0]
  min_abs = torch.min(torch.abs(pos), dim=-1)[0]

  violated = (max_abs > max_threshold) | (min_abs > min_threshold)
  return violated.float()


def foot_displacement(
  env: ManagerBasedRlEnv,
  side: str = "left",
  sigma: float = -2.0,
  clamp_min: float = 0.3,
  height_threshold: float = 0.15,
  phase3_height: float = 0.34,
  foot_body_name: str = "l_ankle_pitch_link",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping foot under body COM.

  Args:
    env: The environment.
    side: "left" or "right".
    sigma: Exponential decay sigma. Default -2.0.
    clamp_min: Minimum MSE before reward starts. Default 0.3.
    height_threshold: Maximum foot height to apply reward. Default 0.15m.
    phase3_height: Minimum base height to apply reward. Default 0.34m.
    foot_body_name: Name of foot body.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Get foot body.
  if side == "right":
    foot_body_name = foot_body_name.replace("l_", "r_")

  foot_ids, _ = asset.find_bodies(foot_body_name)
  if not foot_ids:
    return torch.zeros(env.num_envs, device=env.device)
  foot_id = foot_ids[0]

  body_ids = asset.indexing.body_ids
  foot_pos = env.sim.data.xpos[:, body_ids[foot_id], :]
  base_xy = asset.data.root_link_pos_w[:, :2]
  foot_xy = foot_pos[:, :2]

  mse = torch.sum(torch.square(base_xy - foot_xy), dim=-1)
  mse_clamped = mse.clamp(clamp_min, float("inf"))
  reward = torch.exp(mse_clamped * sigma)

  # Only apply when foot is on ground and robot is standing.
  foot_on_ground = foot_pos[:, 2] < height_threshold
  standup = asset.data.root_link_pos_w[:, 2] > phase3_height

  return reward * foot_on_ground * standup


def ground_parallel(
  env: ManagerBasedRlEnv,
  decay_rate: float = 5.0,
  phase3_height: float = 0.34,
  left_foot_body: str = "l_ankle_pitch_link",
  right_foot_body: str = "r_ankle_pitch_link",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward feet being parallel to ground.

  Uses ankle orientation quaternions to compute roll/pitch deviation.

  Args:
    env: The environment.
    decay_rate: Exponential decay rate. Default 5.0.
    phase3_height: Height threshold for post-task. Default 0.34m.
    left_foot_body: Name of left foot body.
    right_foot_body: Name of right foot body.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]

  left_ids, _ = asset.find_bodies(left_foot_body)
  right_ids, _ = asset.find_bodies(right_foot_body)

  if not left_ids or not right_ids:
    return torch.ones(env.num_envs, device=env.device)

  body_ids = asset.indexing.body_ids
  left_quat = env.sim.data.xquat[:, body_ids[left_ids[0]], :]
  right_quat = env.sim.data.xquat[:, body_ids[right_ids[0]], :]

  # Convert to euler and get roll/pitch magnitude.
  def flatness(q: torch.Tensor) -> torch.Tensor:
    roll, pitch, _yaw = euler_xyz_from_quat(q)  # Returns (roll, pitch, yaw) tensors
    rp = torch.abs(roll) + torch.abs(pitch)
    return rp

  left_rp = flatness(left_quat)
  right_rp = flatness(right_quat)
  rp_mean = 0.5 * (left_rp + right_rp)
  reward = torch.exp(-decay_rate * rp_mean)

  # Disable reward when fully standing (post-task).
  standup = asset.data.root_link_pos_w[:, 2] > phase3_height
  reward = reward * (~standup) + standup.float()

  return reward


def feet_distance(
  env: ManagerBasedRlEnv,
  threshold: float = 0.45,
  left_foot_body: str = "l_ankle_pitch_link",
  right_foot_body: str = "r_ankle_pitch_link",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize feet being too close together.

  Args:
    env: The environment.
    threshold: Minimum distance threshold. Default 0.45m.
    left_foot_body: Name of left foot body.
    right_foot_body: Name of right foot body.
    asset_cfg: Asset configuration.

  Returns:
    Penalty tensor of shape (num_envs,) as float (1 if violated, 0 otherwise).
  """
  asset: Entity = env.scene[asset_cfg.name]

  left_ids, _ = asset.find_bodies(left_foot_body)
  right_ids, _ = asset.find_bodies(right_foot_body)

  if not left_ids or not right_ids:
    return torch.zeros(env.num_envs, device=env.device)

  body_ids = asset.indexing.body_ids
  left_pos = env.sim.data.xpos[:, body_ids[left_ids[0]], :]
  right_pos = env.sim.data.xpos[:, body_ids[right_ids[0]], :]

  distance = torch.norm(left_pos - right_pos, dim=-1)
  return (distance > threshold).float()


def feet_contact_balance(
  env: ManagerBasedRlEnv,
  decay_rate: float = 5.0,
  phase2_height: float = 0.25,
  contact_sensor_name: str = "feet_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward balanced contact forces between feet.

  Args:
    env: The environment.
    decay_rate: Exponential decay rate. Default 5.0.
    phase2_height: Minimum height to apply reward. Default 0.25m.
    contact_sensor_name: Name of contact sensor (if available).
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Try to get contact forces from sensor.
  try:
    sensor = env.scene[contact_sensor_name]
    left_fz = sensor.data[:, 0, 2] if sensor.data.dim() > 2 else sensor.data[:, 0]
    right_fz = sensor.data[:, 1, 2] if sensor.data.dim() > 2 else sensor.data[:, 1]
  except (KeyError, AttributeError):
    # Fallback: return neutral reward.
    return torch.ones(env.num_envs, device=env.device)

  left_fz = torch.abs(left_fz)
  right_fz = torch.abs(right_fz)
  diff = torch.abs(left_fz - right_fz)
  total = left_fz + right_fz + 1e-4
  imbalance = diff / total

  standup_mid = asset.data.root_link_pos_w[:, 2] > phase2_height
  reward = torch.exp(-decay_rate * imbalance) * standup_mid

  return reward


def ankle_pitch_neutral(
  env: ManagerBasedRlEnv,
  target: float = -0.1,
  decay_rate: float = 10.0,
  phase2_height: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward ankle pitch joints being near neutral position.

  Args:
    env: The environment.
    target: Target ankle pitch angle. Default -0.1 rad.
    decay_rate: Exponential decay rate. Default 10.0.
    phase2_height: Minimum height to apply reward. Default 0.25m.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]

  joint_ids, _ = asset.find_joints(".*ankle_pitch_joint")
  if not joint_ids:
    return torch.ones(env.num_envs, device=env.device)

  ankle_angles = asset.data.joint_pos[:, joint_ids]
  mse = torch.mean((ankle_angles - target) ** 2, dim=1)

  standup = asset.data.root_link_pos_w[:, 2] > phase2_height
  reward = torch.exp(-decay_rate * mse) * standup

  return reward


def style_ang_vel_xy(
  env: ManagerBasedRlEnv,
  decay_rate: float = 2.0,
  phase1_height: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize XY angular velocity.

  Args:
    env: The environment.
    decay_rate: Exponential decay rate. Default 2.0.
    phase1_height: Minimum height to apply reward. Default 0.25m.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel_xy = asset.data.root_link_ang_vel_b[:, :2]
  base_height = asset.data.root_link_pos_w[:, 2] > phase1_height
  return (
    torch.exp(-decay_rate * torch.sum(torch.square(ang_vel_xy), dim=1)) * base_height
  )


def soft_symmetry_action(
  env: ManagerBasedRlEnv,
  left_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
  right_joint_indices: tuple[int, ...] = (6, 7, 8, 9, 10, 11),
  negative_indices: tuple[int, ...] = (1, 2, 5),
  phase3_height: float = 0.34,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize asymmetric actions between left and right legs.

  Args:
    env: The environment.
    left_joint_indices: Indices of left leg joints in action.
    right_joint_indices: Indices of right leg joints in action.
    negative_indices: Indices that should be negated for symmetry comparison.
    phase3_height: Minimum height to apply penalty. Default 0.34m.
    asset_cfg: Asset configuration.

  Returns:
    Penalty tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  actions = env.action_manager.action

  left_actions = actions[:, list(left_joint_indices)].clone()
  right_actions = actions[:, list(right_joint_indices)]

  # Negate specified indices for symmetry comparison.
  neg_idx = torch.tensor(negative_indices, device=env.device, dtype=torch.long)
  left_actions[:, neg_idx] *= -1

  symmetry_error = torch.norm(left_actions - right_actions, dim=-1)

  standup = asset.data.root_link_pos_w[:, 2] > phase3_height
  symmetry_error[~standup] = 0

  # Scale by gravity projection.
  gravity_scale = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.9) / 0.9
  symmetry_error = symmetry_error * gravity_scale

  return symmetry_error


def soft_symmetry_body(
  env: ManagerBasedRlEnv,
  left_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
  right_joint_indices: tuple[int, ...] = (6, 7, 8, 9, 10, 11),
  negative_indices: tuple[int, ...] = (1, 2, 5),
  margin: float = 0.08,
  decay_rate: float = 40.0,
  phase3_height: float = 0.34,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward symmetric body posture between left and right legs.

  Args:
    env: The environment.
    left_joint_indices: Indices of left leg joints.
    right_joint_indices: Indices of right leg joints.
    negative_indices: Indices that should be negated for symmetry comparison.
    margin: Error margin before penalty starts. Default 0.08 rad.
    decay_rate: Exponential decay rate. Default 40.0.
    phase3_height: Minimum height to apply reward. Default 0.34m.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]

  left_pos = asset.data.joint_pos[:, list(left_joint_indices)].clone()
  right_pos = asset.data.joint_pos[:, list(right_joint_indices)]

  # Negate specified indices for symmetry comparison.
  neg_idx = torch.tensor(negative_indices, device=env.device, dtype=torch.long)
  left_pos[:, neg_idx] *= -1

  error = (torch.abs(left_pos - right_pos) - margin).clamp(min=0)
  symmetry_error = torch.norm(error, dim=-1)

  standup = asset.data.root_link_pos_w[:, 2] > phase3_height
  reward = torch.exp(-decay_rate * symmetry_error)
  reward[~standup] = 0

  # Scale by gravity projection.
  gravity_scale = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.9) / 0.9
  reward = reward * gravity_scale

  return reward


##
# Post-task (target) rewards.
##


def target_orientation(
  env: ManagerBasedRlEnv,
  decay_rate: float = 5.0,
  phase3_height: float = 0.34,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation after standing up.

  Args:
    env: The environment.
    decay_rate: Exponential decay rate. Default 5.0.
    phase3_height: Minimum height to apply reward. Default 0.34m.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  proj_gravity_xy = asset.data.projected_gravity_b[:, :2]
  standup = asset.data.root_link_pos_w[:, 2] > phase3_height
  return (
    torch.exp(-decay_rate * torch.sum(torch.square(proj_gravity_xy), dim=1)) * standup
  )


def target_base_height(
  env: ManagerBasedRlEnv,
  target_height: float = 0.34,
  decay_rate: float = 20.0,
  phase3_height: float = 0.34,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward base height near target after standing up.

  Args:
    env: The environment.
    target_height: Target base height. Default 0.34m.
    decay_rate: Exponential decay rate. Default 20.0.
    phase3_height: Minimum height to apply reward. Default 0.34m.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  base_height = asset.data.root_link_pos_w[:, 2]
  standup = base_height > phase3_height
  return torch.exp(-decay_rate * torch.abs(base_height - target_height)) * standup


def lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  decay_rate: float = 5.0,
  phase3_height: float = 0.34,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize XY linear velocity after standing up.

  Args:
    env: The environment.
    decay_rate: Exponential decay rate. Default 5.0.
    phase3_height: Minimum height to apply reward. Default 0.34m.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]
  lin_vel_xy = asset.data.root_link_lin_vel_b[:, :2]
  base_height = asset.data.root_link_pos_w[:, 2] > phase3_height
  return (
    torch.exp(-decay_rate * torch.sum(torch.square(lin_vel_xy), dim=1)) * base_height
  )


def feet_height_var(
  env: ManagerBasedRlEnv,
  scale: float = 10.0,
  decay_rate: float = 2.0,
  clamp_min: float = 0.2,
  phase3_height: float = 0.34,
  left_foot_body: str = "l_ankle_pitch_link",
  right_foot_body: str = "r_ankle_pitch_link",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize feet height variance after standing up.

  Args:
    env: The environment.
    scale: Scale factor for height. Default 10.0.
    decay_rate: Exponential decay rate. Default 2.0.
    clamp_min: Minimum variance before penalty. Default 0.2.
    phase3_height: Minimum height to apply reward. Default 0.34m.
    left_foot_body: Name of left foot body.
    right_foot_body: Name of right foot body.
    asset_cfg: Asset configuration.

  Returns:
    Reward tensor of shape (num_envs,).
  """
  asset: Entity = env.scene[asset_cfg.name]

  left_ids, _ = asset.find_bodies(left_foot_body)
  right_ids, _ = asset.find_bodies(right_foot_body)

  if not left_ids or not right_ids:
    return torch.ones(env.num_envs, device=env.device)

  body_ids = asset.indexing.body_ids
  left_height = env.sim.data.xpos[:, body_ids[left_ids[0]], 2] * scale
  right_height = env.sim.data.xpos[:, body_ids[right_ids[0]], 2] * scale

  feet_diff = torch.abs(left_height - right_height).clamp(clamp_min, float("inf"))
  standup = asset.data.root_link_pos_w[:, 2] > phase3_height
  reward = torch.exp(-decay_rate * feet_diff) * standup

  return reward
