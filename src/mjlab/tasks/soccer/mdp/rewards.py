"""Soccer-specific reward functions.

Ported from HumanoidSoccer (Kong et al., 2026).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.soccer.mdp.commands import SoccerMotionCommand
from mjlab.tasks.soccer.mdp.kick_detection import KickContactTracker
from mjlab.tasks.soccer.mdp.observations import get_target_point_world
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_command(env: ManagerBasedRlEnv, name: str) -> SoccerMotionCommand:
  return env.command_manager.get_term(name)  # type: ignore[return-value]


def _get_tracker(command: SoccerMotionCommand) -> KickContactTracker:
  return command.kick_contact_tracker


def _get_body_indexes(
  command: SoccerMotionCommand, body_names: list[str] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


# ------------------------------------------------------------------
# Motion tracking rewards (soccer-specific variants)
# ------------------------------------------------------------------


def motion_relative_foot_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  foot_body_names: list[str] | None = None,
) -> torch.Tensor:
  if foot_body_names is None:
    foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
  command = _get_command(env, command_name)
  body_indexes = _get_body_indexes(command, foot_body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


# ------------------------------------------------------------------
# Foot distance
# ------------------------------------------------------------------


def foot_distance(
  env: ManagerBasedRlEnv,
  threshold: float,
  std: float,
  foot_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Encourage minimum separation between feet."""
  robot = env.scene[foot_cfg.name]
  indices = torch.as_tensor(
    robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
    dtype=torch.long,
    device=env.device,
  )
  left = robot.data.body_link_pos_w[:, indices[0]]
  right = robot.data.body_link_pos_w[:, indices[1]]
  distance = torch.norm(left - right, dim=1)
  return torch.where(
    distance >= threshold,
    torch.tensor(1.0, device=distance.device),
    1.0 * torch.exp(-((distance / threshold - 1) ** 2) / (std**2)),
  )


# ------------------------------------------------------------------
# Target point proximity (with freeze at contact)
# ------------------------------------------------------------------


def target_point_proximity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "motion",
) -> torch.Tensor:
  """Reward proximity to ball; freeze reward value at first kick contact."""
  command = _get_command(env, command_name)
  tracker = _get_tracker(command)

  base_xy = command.robot_anchor_pos_w[..., :2]
  target = get_target_point_world(env, command_name).to(
    device=base_xy.device, dtype=base_xy.dtype
  )
  diff_xy = base_xy - target[..., :2]
  error = torch.sum(diff_xy * diff_xy, dim=-1)
  proximity_reward = torch.exp(-error / std**2)

  contact_awarded = tracker.contact_awarded
  frozen_reward = tracker.frozen_proximity_reward

  new_kick = contact_awarded & (frozen_reward == 0.0)
  if torch.any(new_kick):
    ids = torch.nonzero(new_kick, as_tuple=False).squeeze(-1)
    tracker.freeze_proximity(ids, proximity_reward[ids])
    frozen_reward = tracker.frozen_proximity_reward

  return torch.where(contact_awarded, frozen_reward, proximity_reward)


# ------------------------------------------------------------------
# Target point contact (one-shot)
# ------------------------------------------------------------------


def target_point_contact(
  env: ManagerBasedRlEnv,
  horizontal_force_threshold: float = 0.0,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_contact",
  foot_cfg: SceneEntityCfg | None = None,
  min_foot_speed: float = 0.0,
) -> torch.Tensor:
  """One-shot reward for contacting the ball at first valid touch."""
  command = _get_command(env, command_name)
  tracker = _get_tracker(command)
  event = tracker.detect(
    command, ball_sensor_name, horizontal_force_threshold, min_foot_speed
  )

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if not torch.any(event.new_contact):
    return reward

  reward_scale = torch.zeros_like(reward)
  correct_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  if foot_cfg is not None:
    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() > 0:
      valid = foot_info.expected >= 0
      correct = (foot_info.sides == foot_info.expected) & valid
      reward_scale[foot_info.env_ids] = correct.to(reward_scale.dtype)
      correct_mask[foot_info.env_ids] = correct

  tracker.record_expected_success(event.new_contact, correct_mask)
  return event.new_contact.to(reward.dtype) * reward_scale


# ------------------------------------------------------------------
# Sideways kick
# ------------------------------------------------------------------


def sideways_kick(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_contact",
  horizontal_force_threshold: float = 0.0,
  foot_cfg: SceneEntityCfg | None = None,
  min_foot_speed: float = 0.0,
) -> torch.Tensor:
  """Single-shot reward encouraging foot swing along the expected lateral
  axis."""
  if foot_cfg is None:
    raise ValueError("sideways_kick requires foot_cfg")

  command = _get_command(env, command_name)
  tracker = _get_tracker(command)
  event = tracker.detect(
    command, ball_sensor_name, horizontal_force_threshold, min_foot_speed
  )

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if not torch.any(event.new_contact):
    return reward

  foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
  if foot_info.env_ids.numel() == 0:
    return reward

  robot = command.robot
  foot_vel_w = robot.data.body_link_lin_vel_w[foot_info.env_ids, foot_info.body_indices]
  foot_quat_w = robot.data.body_link_quat_w[foot_info.env_ids, foot_info.body_indices]

  vel_local = quat_apply(quat_inv(foot_quat_w), foot_vel_w)
  vel_norm = torch.norm(vel_local, dim=-1)

  expected_leg = foot_info.expected.to(device=env.device, dtype=torch.int8)
  desired_sign = torch.zeros_like(vel_norm)
  desired_sign = torch.where(
    expected_leg == 0, torch.full_like(desired_sign, -1.0), desired_sign
  )
  desired_sign = torch.where(
    expected_leg == 1, torch.full_like(desired_sign, 1.0), desired_sign
  )

  directional = vel_local[:, 1] * desired_sign
  axis_component = torch.clamp(directional, min=0.0)
  alignment = torch.where(
    vel_norm > 1e-6,
    axis_component / vel_norm,
    torch.zeros_like(vel_norm),
  )
  reward[foot_info.env_ids] = alignment.to(reward.dtype)

  valid = expected_leg >= 0
  correct_foot = (foot_info.sides == foot_info.expected) & valid
  wrong_mask = ~correct_foot
  if torch.any(wrong_mask):
    reward[foot_info.env_ids[wrong_mask]] = 0.0

  return reward


# ------------------------------------------------------------------
# Ball velocity direction alignment
# ------------------------------------------------------------------


def ball_velocity_direction_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  velocity_threshold: float = 0.1,
  horizontal_force_threshold: float = 0.0,
  ball_sensor_name: str = "ball_contact",
  foot_cfg: SceneEntityCfg | None = None,
  min_foot_speed: float = 0.0,
) -> torch.Tensor:
  """Reward alignment between ball velocity and target-to-destination
  direction. Active in a short window after contact with expected foot."""
  command = _get_command(env, command_name)
  tracker = _get_tracker(command)
  soccer_ball = env.scene["soccer_ball"]
  vel = soccer_ball.data.root_link_lin_vel_w
  vel_xy = vel[:, :2]
  vel_xy_norm = torch.norm(vel_xy, dim=-1, keepdim=True)

  direction = command.target_destination_pos - command.initial_target_point_pos
  direction_xy = direction[:, :2]
  dir_norm = torch.norm(direction_xy, dim=-1, keepdim=True)

  timer = tracker.dir_align_timer

  # Trigger window on expected-foot contact.
  event = tracker.detect(
    command, ball_sensor_name, horizontal_force_threshold, min_foot_speed
  )
  if torch.any(event.new_contact) and foot_cfg is not None:
    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() > 0:
      valid = foot_info.expected >= 0
      correct = (foot_info.sides == foot_info.expected) & valid
      correct_ids = foot_info.env_ids[correct]
      if correct_ids.numel() > 0:
        timer[correct_ids] = 5

  speed_valid = (vel_xy_norm.squeeze(-1) > 1e-6) & (dir_norm.squeeze(-1) > 1e-6)
  active_mask = (timer > 0) & speed_valid

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active_mask):
    dir_unit = direction_xy[active_mask] / dir_norm[active_mask]
    vel_unit = vel_xy[active_mask] / vel_xy_norm[active_mask]
    cos_theta = torch.sum(vel_unit * dir_unit, dim=-1).clamp(-1.0, 1.0)
    error = torch.acos(cos_theta) ** 2
    reward[active_mask] = torch.exp(-error / (std**2))

  timer[:] = torch.where(timer > 0, timer - 1, timer)
  return reward


# ------------------------------------------------------------------
# Ball speed reward
# ------------------------------------------------------------------


def ball_speed_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  velocity_threshold: float = 0.1,
  horizontal_force_threshold: float = 0.0,
  ball_sensor_name: str = "ball_contact",
  foot_cfg: SceneEntityCfg | None = None,
  min_foot_speed: float = 0.0,
) -> torch.Tensor:
  """Reward ball speed in a short window after expected-foot contact."""
  command = _get_command(env, command_name)
  tracker = _get_tracker(command)
  soccer_ball = env.scene["soccer_ball"]
  vel = soccer_ball.data.root_link_lin_vel_w
  speed_xy = torch.norm(vel[:, :2], dim=-1)

  timer = tracker.speed_timer

  event = tracker.detect(
    command, ball_sensor_name, horizontal_force_threshold, min_foot_speed
  )
  if torch.any(event.new_contact) and foot_cfg is not None:
    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() > 0:
      valid = foot_info.expected >= 0
      correct = (foot_info.sides == foot_info.expected) & valid
      correct_ids = foot_info.env_ids[correct]
      if correct_ids.numel() > 0:
        timer[correct_ids] = 5

  speed_valid = speed_xy > 1e-6
  active_mask = (timer > 0) & speed_valid

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active_mask):
    reward[active_mask] = 1.0 - torch.exp(-(speed_xy[active_mask] ** 2) / (std**2))

  timer[:] = torch.where(timer > 0, timer - 1, timer)
  return reward


# ------------------------------------------------------------------
# Ball z-speed penalty
# ------------------------------------------------------------------


def ball_z_speed_penalty_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  velocity_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize excessive vertical ball speed after activation."""
  soccer_ball = env.scene["soccer_ball"]
  vel = soccer_ball.data.root_link_lin_vel_w
  z_speed = vel[:, 2]
  speed = torch.norm(vel, dim=-1)

  command = _get_command(env, command_name)
  tracker = _get_tracker(command)

  valid_mask = speed > velocity_threshold
  timer = tracker.z_speed_timer
  prev_valid = tracker.z_speed_prev

  rising = valid_mask & (~prev_valid)
  timer[rising] = 5
  active_mask = timer > 0

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active_mask):
    scale = std if std > 0 else 1.0
    reward[active_mask] = torch.tanh(torch.abs(z_speed[active_mask]) / (scale + 1e-8))

  timer[:] = torch.where(timer > 0, timer - 1, timer)
  tracker.z_speed_prev[:] = valid_mask
  return reward


# ------------------------------------------------------------------
# Pelvis orientation
# ------------------------------------------------------------------


def pelvis_orientation(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Penalize pelvis pitch/roll tilt."""
  command = _get_command(env, command_name)
  robot = command.robot
  gravity_vec_w = robot.data.gravity_vec_w
  pelvis_proj = quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
  return torch.sum(torch.square(pelvis_proj[:, :2]), dim=1)


# ------------------------------------------------------------------
# Waist action rate
# ------------------------------------------------------------------


def waist_action_rate_l2_clip(
  env: ManagerBasedRlEnv,
  waist_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize rate of change of waist joint actions."""
  robot = env.scene[waist_cfg.name]
  idx = torch.as_tensor(
    robot.find_joints(waist_cfg.joint_names, preserve_order=True)[0],
    device=env.device,
  )
  return torch.sum(
    torch.square(
      env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]
    ),
    dim=1,
  ).clamp(max=100.0)
