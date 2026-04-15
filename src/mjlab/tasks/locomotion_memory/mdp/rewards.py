from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import LocomotionMemoryCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_command(env: ManagerBasedRlEnv, command_name: str) -> LocomotionMemoryCommand:
  return cast(LocomotionMemoryCommand, env.command_manager.get_term(command_name))


def _turn_command_mask(
  command: torch.Tensor,
  *,
  min_abs_yaw_command: float,
  max_forward_speed: float,
  max_lateral_command: float,
) -> torch.Tensor:
  return (
    (torch.abs(command[:, 2]) >= min_abs_yaw_command)
    & (torch.abs(command[:, 0]) <= max_forward_speed)
    & (torch.abs(command[:, 1]) <= max_lateral_command)
  )


def _low_yaw_turn_command_mask(
  command: torch.Tensor,
  *,
  min_abs_yaw_command: float,
  max_abs_yaw_command: float,
  max_forward_speed: float,
  max_lateral_command: float,
) -> torch.Tensor:
  abs_yaw = torch.abs(command[:, 2])
  return (
    (abs_yaw >= min_abs_yaw_command)
    & (abs_yaw <= max_abs_yaw_command)
    & (torch.abs(command[:, 0]) <= max_forward_speed)
    & (torch.abs(command[:, 1]) <= max_lateral_command)
  )


def _straight_walk_mask(
  command: torch.Tensor,
  *,
  min_forward_speed: float,
  max_lateral_command: float,
  max_abs_yaw_command: float,
) -> torch.Tensor:
  return (
    (command[:, 0] >= min_forward_speed)
    & (torch.abs(command[:, 1]) <= max_lateral_command)
    & (torch.abs(command[:, 2]) <= max_abs_yaw_command)
  )


def _mean_over_mask(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  masked = value * mask.float()
  denom = mask.float().sum()
  return torch.sum(masked) / torch.clamp(denom, min=1.0)


def _forward_velocity_ready_mask(
  command: torch.Tensor,
  actual_lin_vel_b: torch.Tensor | None,
  *,
  max_forward_velocity_lag: float,
) -> torch.Tensor:
  if max_forward_velocity_lag <= 0.0:
    return torch.ones(command.shape[0], dtype=torch.bool, device=command.device)
  assert actual_lin_vel_b is not None
  return actual_lin_vel_b[:, 0] >= command[:, 0] - max_forward_velocity_lag


def _body_positions_in_root_frame(
  asset: Entity, body_ids: list[int] | slice
) -> torch.Tensor:
  body_pos_w = asset.data.body_link_pos_w[:, body_ids, :]
  root_pos_w = asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, body_pos_w.shape[1], -1
  )
  return quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)


def memory_contact_validity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _get_command(env, command_name)
  return command.contact_valid.float()


def memory_phase_validity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _get_command(env, command_name)
  return command.phase_valid.float()


def memory_retrieval_switch_cost(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = _get_command(env, command_name)
  return command.retrieval_switched.float()


def memory_phase_jump_cost(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _get_command(env, command_name)
  return command.phase_jump * command.retrieval_switched.float()


def memory_cadence_jump_cost(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _get_command(env, command_name)
  return command.cadence_jump * command.retrieval_switched.float()


def memory_transition_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  twist_command_name: str,
  speed_threshold: float,
  speed_margin: float,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  twist_command = env.command_manager.get_command(twist_command_name)
  assert twist_command is not None
  speed = torch.linalg.norm(twist_command[:, :2], dim=-1)
  near_boundary = torch.abs(speed - speed_threshold) <= speed_margin
  return near_boundary.float() * command.transition_active.float()


def straight_linear_velocity_tracking_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
  min_forward_speed: float = 0.4,
  max_lateral_command: float = 0.05,
  max_abs_yaw_command: float = 0.12,
) -> torch.Tensor:
  """Reward forward velocity tracking during straight walking only."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  straight_walk = _straight_walk_mask(
    command,
    min_forward_speed=min_forward_speed,
    max_lateral_command=max_lateral_command,
    max_abs_yaw_command=max_abs_yaw_command,
  )

  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.linalg.norm(command[:, :2] - actual[:, :2], dim=1)
  reward = torch.exp(-torch.square(xy_error) / std**2) * straight_walk.float()
  env.extras["log"]["Metrics/straight_linear_tracking_error_mean"] = _mean_over_mask(
    xy_error, straight_walk
  )
  return reward


def straight_yaw_velocity_tracking_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
  min_forward_speed: float = 0.4,
  max_lateral_command: float = 0.05,
  max_abs_yaw_command: float = 0.12,
) -> torch.Tensor:
  """Reward low yaw-rate drift while following straight forward commands."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  straight_walk = _straight_walk_mask(
    command,
    min_forward_speed=min_forward_speed,
    max_lateral_command=max_lateral_command,
    max_abs_yaw_command=max_abs_yaw_command,
  )

  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_ang_vel_b
  yaw_error = torch.abs(command[:, 2] - actual[:, 2])
  reward = torch.exp(-torch.square(yaw_error) / std**2) * straight_walk.float()
  env.extras["log"]["Metrics/straight_yaw_tracking_error_mean"] = _mean_over_mask(
    yaw_error, straight_walk
  )
  return reward


def turn_yaw_velocity_tracking_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
  min_abs_yaw_command: float = 0.12,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Reward yaw-rate tracking during low-translation turning only."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  turn_gate = _turn_command_mask(
    command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  )

  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_ang_vel_b
  yaw_error = torch.abs(command[:, 2] - actual[:, 2])
  reward = torch.exp(-torch.square(yaw_error) / std**2) * turn_gate.float()
  env.extras["log"]["Metrics/turn_yaw_tracking_error_mean"] = _mean_over_mask(
    yaw_error, turn_gate
  )
  return reward


def low_yaw_turn_velocity_tracking_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
  min_abs_yaw_command: float = 0.10,
  max_abs_yaw_command: float = 0.25,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Reward tight yaw-rate tracking for the weak low-yaw turn regime."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  low_yaw_gate = _low_yaw_turn_command_mask(
    command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_abs_yaw_command=max_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  )

  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_ang_vel_b
  yaw_error = torch.abs(command[:, 2] - actual[:, 2])
  reward = torch.exp(-torch.square(yaw_error) / std**2) * low_yaw_gate.float()

  left_mask = low_yaw_gate & (command[:, 2] > 0.0)
  right_mask = low_yaw_gate & (command[:, 2] < 0.0)
  env.extras["log"]["Metrics/low_yaw_turn_tracking_error_mean"] = _mean_over_mask(
    yaw_error, low_yaw_gate
  )
  env.extras["log"]["Metrics/low_yaw_turn_left_error_mean"] = _mean_over_mask(
    yaw_error, left_mask
  )
  env.extras["log"]["Metrics/low_yaw_turn_right_error_mean"] = _mean_over_mask(
    yaw_error, right_mask
  )
  return reward


def foot_midline_crossing_cost(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  min_lateral_offset: float = 0.04,
  min_separation: float = 0.12,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize feet that cross or crowd the body midline while locomoting."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  active = (total_command > command_threshold).float()

  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  root_pos_w = asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, foot_pos_w.shape[1], -1
  )
  foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)

  left_y = foot_pos_b[:, 0, 1]
  right_y = foot_pos_b[:, 1, 1]
  left_violation = torch.relu(min_lateral_offset - left_y)
  right_violation = torch.relu(right_y + min_lateral_offset)
  separation_violation = torch.relu(min_separation - (left_y - right_y))
  violation = left_violation + right_violation + separation_violation
  env.extras["log"]["Metrics/foot_midline_violation_mean"] = torch.mean(violation)
  return violation * active


def straight_foot_lane_cost(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  target_lateral_offset: float = 0.095,
  tolerance: float = 0.02,
  min_forward_speed: float = 0.3,
  max_lateral_command: float = 0.05,
  max_yaw_command: float = 0.1,
) -> torch.Tensor:
  """Keep each foot on a stable left/right lane during straight walking.

  When walking straight, human-like foot trajectories stay close to two parallel
  lanes instead of sweeping inward across the body midline. This term is gated to
  forward straight-walk commands so it does not fight turning behavior.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  straight_walk = (
    (command[:, 0] > min_forward_speed)
    & (torch.abs(command[:, 1]) <= max_lateral_command)
    & (torch.abs(command[:, 2]) <= max_yaw_command)
  ).float()

  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  root_pos_w = asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, foot_pos_w.shape[1], -1
  )
  foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)

  lane_targets = foot_pos_b.new_tensor([target_lateral_offset, -target_lateral_offset])
  lane_error = torch.abs(foot_pos_b[:, :, 1] - lane_targets[None, :])
  lane_violation = torch.relu(lane_error - tolerance).sum(dim=1)
  env.extras["log"]["Metrics/foot_lane_violation_mean"] = torch.mean(lane_violation)
  return lane_violation * straight_walk


def turn_feet_air_time_bonus(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.45,
  min_abs_yaw_command: float = 0.2,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Reward clear stepping during low-speed turning in place."""
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None

  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  turn_gate = _turn_command_mask(
    command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  ).float()

  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/turn_air_time_mean"] = mean_air_time
  return reward * turn_gate


def turn_reference_swing_height_bonus(
  env: ManagerBasedRlEnv,
  reference_command_name: str,
  twist_command_name: str,
  asset_cfg: SceneEntityCfg,
  threshold_min: float = 0.055,
  threshold_max: float = 0.20,
  min_abs_yaw_command: float = 0.2,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Reward swing-foot clearance when the active reference is a turn snippet."""
  reference_command = _get_command(env, reference_command_name)
  twist_command = env.command_manager.get_command(twist_command_name)
  assert twist_command is not None

  turn_gate = _turn_command_mask(
    twist_command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  )
  active_gate = turn_gate & reference_command.turn_snippet_active

  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  foot_heights = foot_pos_w[:, :, 2]
  support_height = torch.min(foot_heights, dim=1, keepdim=True).values
  clearance = foot_heights - support_height

  swing_mask = (~reference_command.reference_contact).float() * active_gate[
    :, None
  ].float()
  in_range = ((clearance >= threshold_min) & (clearance <= threshold_max)).float()
  reward = torch.sum(in_range * swing_mask, dim=1)

  swing_clearance = clearance * swing_mask
  swing_count = torch.sum(swing_mask)
  mean_swing_clearance = torch.sum(swing_clearance) / torch.clamp(swing_count, min=1.0)
  env.extras["log"]["Metrics/turn_reference_swing_clearance_mean"] = (
    mean_swing_clearance
  )
  return reward


def turn_swing_clearance_cost(
  env: ManagerBasedRlEnv,
  reference_command_name: str,
  twist_command_name: str,
  asset_cfg: SceneEntityCfg,
  min_clearance: float = 0.055,
  min_abs_yaw_command: float = 0.2,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Penalize turn-swing feet that stay close to the support foot height."""
  reference_command = _get_command(env, reference_command_name)
  twist_command = env.command_manager.get_command(twist_command_name)
  assert twist_command is not None

  turn_gate = _turn_command_mask(
    twist_command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  )
  active_gate = turn_gate & reference_command.turn_snippet_active

  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  foot_heights = foot_pos_w[:, :, 2]
  support_height = torch.min(foot_heights, dim=1, keepdim=True).values
  clearance = foot_heights - support_height

  swing_mask = (~reference_command.reference_contact.bool()).float() * active_gate[
    :, None
  ].float()
  deficit = torch.relu(min_clearance - clearance)
  cost = torch.sum(deficit * swing_mask, dim=1)

  swing_count = torch.sum(swing_mask)
  mean_deficit = torch.sum(deficit * swing_mask) / torch.clamp(swing_count, min=1.0)
  env.extras["log"]["Metrics/turn_swing_clearance_deficit_mean"] = mean_deficit
  return cost


def turn_scuff_cost(
  env: ManagerBasedRlEnv,
  reference_command_name: str,
  twist_command_name: str,
  height_sensor_name: str,
  asset_cfg: SceneEntityCfg,
  min_clearance: float = 0.04,
  planar_speed_threshold: float = 0.08,
  min_abs_yaw_command: float = 0.12,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Penalize low-clearance swing-foot sweeps during in-place turns."""
  reference_command = _get_command(env, reference_command_name)
  twist_command = env.command_manager.get_command(twist_command_name)
  assert twist_command is not None

  turn_gate = _turn_command_mask(
    twist_command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  )
  active_gate = turn_gate & reference_command.turn_snippet_active

  height_sensor = env.scene[height_sensor_name]
  foot_heights = height_sensor.data.heights

  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, foot_vel_w.shape[1], -1
  )
  foot_vel_b = quat_apply_inverse(root_quat_w, foot_vel_w)
  planar_sweep = torch.relu(
    torch.linalg.norm(foot_vel_b[:, :, :2], dim=2) - planar_speed_threshold
  )

  swing_mask = (
    (~reference_command.reference_contact.bool()).float()
    * active_gate[:, None].float()
    * (reference_command.snippet_id[:, None] >= 0).float()
  )
  low_clearance = torch.relu(min_clearance - foot_heights)
  cost = torch.sum(low_clearance * planar_sweep * swing_mask, dim=1)

  scuff_risk = ((foot_heights < min_clearance) & (planar_sweep > 0.0)).float()
  swing_count = torch.sum(swing_mask)
  env.extras["log"]["Metrics/turn_scuff_risk_mean"] = torch.sum(
    scuff_risk * swing_mask
  ) / torch.clamp(swing_count, min=1.0)
  env.extras["log"]["Metrics/turn_planar_swing_speed_mean"] = torch.sum(
    torch.linalg.norm(foot_vel_b[:, :, :2], dim=2) * swing_mask
  ) / torch.clamp(swing_count, min=1.0)
  return cost


def forward_reference_swing_height_bonus(
  env: ManagerBasedRlEnv,
  reference_command_name: str,
  twist_command_name: str,
  height_sensor_name: str,
  threshold_min: float = 0.04,
  threshold_max: float = 0.16,
  min_forward_speed: float = 0.4,
  max_lateral_command: float = 0.05,
  max_abs_yaw_command: float = 0.12,
) -> torch.Tensor:
  """Reward swing-foot clearance during straight walking.

  This targets the remaining forward-gait issue where the foot occasionally
  skims the ground even though the retrieved snippet expects a swing phase.
  """
  reference_command = _get_command(env, reference_command_name)
  twist_command = env.command_manager.get_command(twist_command_name)
  assert twist_command is not None

  straight_walk = _straight_walk_mask(
    twist_command,
    min_forward_speed=min_forward_speed,
    max_lateral_command=max_lateral_command,
    max_abs_yaw_command=max_abs_yaw_command,
  )

  height_sensor = env.scene[height_sensor_name]
  foot_heights = height_sensor.data.heights
  active_reference = (reference_command.snippet_id >= 0)[:, None].float()
  swing_mask = (
    (~reference_command.reference_contact).float()
    * straight_walk[:, None].float()
    * active_reference
  )
  in_range = ((foot_heights >= threshold_min) & (foot_heights <= threshold_max)).float()
  reward = torch.sum(in_range * swing_mask, dim=1)

  swing_count = torch.sum(swing_mask)
  mean_swing_clearance = torch.sum(foot_heights * swing_mask) / torch.clamp(
    swing_count, min=1.0
  )
  env.extras["log"]["Metrics/forward_reference_swing_clearance_mean"] = (
    mean_swing_clearance
  )
  return reward


def forward_scuff_cost(
  env: ManagerBasedRlEnv,
  reference_command_name: str,
  twist_command_name: str,
  height_sensor_name: str,
  asset_cfg: SceneEntityCfg,
  min_clearance: float = 0.04,
  forward_speed_threshold: float = 0.18,
  min_forward_speed: float = 0.4,
  max_lateral_command: float = 0.05,
  max_abs_yaw_command: float = 0.12,
  max_forward_velocity_lag: float = 0.0,
) -> torch.Tensor:
  """Penalize low-clearance forward sweep during straight walking.

  A foot should not move forward close to the ground when the active reference
  says that foot is in swing. This term is aimed at the visible scuff/catch
  that shows up every few steps in straight walking.
  """
  reference_command = _get_command(env, reference_command_name)
  twist_command = env.command_manager.get_command(twist_command_name)
  assert twist_command is not None

  straight_walk = _straight_walk_mask(
    twist_command,
    min_forward_speed=min_forward_speed,
    max_lateral_command=max_lateral_command,
    max_abs_yaw_command=max_abs_yaw_command,
  )

  height_sensor = env.scene[height_sensor_name]
  foot_heights = height_sensor.data.heights

  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, foot_vel_w.shape[1], -1
  )
  foot_vel_b = quat_apply_inverse(root_quat_w, foot_vel_w)
  forward_sweep = torch.relu(foot_vel_b[:, :, 0] - forward_speed_threshold)
  velocity_ready = _forward_velocity_ready_mask(
    twist_command,
    getattr(asset.data, "root_link_lin_vel_b", None),
    max_forward_velocity_lag=max_forward_velocity_lag,
  )

  active_reference = (reference_command.snippet_id >= 0)[:, None].float()
  swing_mask = (
    (~reference_command.reference_contact).float()
    * straight_walk[:, None].float()
    * velocity_ready[:, None].float()
    * active_reference
  )
  low_clearance = torch.relu(min_clearance - foot_heights)
  cost = torch.sum(low_clearance * forward_sweep * swing_mask, dim=1)

  scuff_risk = ((foot_heights < min_clearance) & (forward_sweep > 0.0)).float()
  risk_count = torch.sum(swing_mask)
  env.extras["log"]["Metrics/forward_scuff_risk_mean"] = torch.sum(
    scuff_risk * swing_mask
  ) / torch.clamp(risk_count, min=1.0)
  env.extras["log"]["Metrics/forward_velocity_ready_rate"] = torch.mean(
    (straight_walk & velocity_ready).float()
  )
  return cost


def forward_swing_recovery_cost(
  env: ManagerBasedRlEnv,
  reference_command_name: str,
  twist_command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  min_recovery_x: float = -0.06,
  min_air_time: float = 0.04,
  min_forward_speed: float = 0.4,
  max_lateral_command: float = 0.05,
  max_abs_yaw_command: float = 0.12,
  max_forward_velocity_lag: float = 0.0,
) -> torch.Tensor:
  """Penalize straight-walk swing feet that trail behind the body too long."""
  reference_command = _get_command(env, reference_command_name)
  twist_command = env.command_manager.get_command(twist_command_name)
  assert twist_command is not None

  straight_walk = _straight_walk_mask(
    twist_command,
    min_forward_speed=min_forward_speed,
    max_lateral_command=max_lateral_command,
    max_abs_yaw_command=max_abs_yaw_command,
  )

  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None

  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  root_pos_w = asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, foot_pos_w.shape[1], -1
  )
  foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)
  velocity_ready = _forward_velocity_ready_mask(
    twist_command,
    getattr(asset.data, "root_link_lin_vel_b", None),
    max_forward_velocity_lag=max_forward_velocity_lag,
  )

  active_reference = (reference_command.snippet_id >= 0)[:, None].float()
  swing_mask = (
    (~reference_command.reference_contact).float()
    * (current_air_time >= min_air_time).float()
    * straight_walk[:, None].float()
    * velocity_ready[:, None].float()
    * active_reference
  )
  trailing = torch.relu(min_recovery_x - foot_pos_b[:, :, 0])
  cost = torch.sum(trailing * swing_mask, dim=1)

  swing_count = torch.sum(swing_mask)
  env.extras["log"]["Metrics/forward_swing_trail_mean"] = torch.sum(
    trailing * swing_mask
  ) / torch.clamp(swing_count, min=1.0)
  env.extras["log"]["Metrics/forward_velocity_ready_rate"] = torch.mean(
    (straight_walk & velocity_ready).float()
  )
  return cost


def reference_swing_foot_position_cost(
  env: ManagerBasedRlEnv,
  reference_command_name: str,
  asset_cfg: SceneEntityCfg,
  twist_command_name: str = "",
  x_weight: float = 0.8,
  z_weight: float = 2.0,
  x_tolerance: float = 0.03,
  z_tolerance: float = 0.015,
  min_forward_speed: float = 0.4,
  max_lateral_command: float = 0.05,
  max_abs_yaw_command: float = 0.12,
  max_forward_velocity_lag: float = 0.0,
) -> torch.Tensor:
  """Penalize swing feet that stay lower or more trailing than the reference.

  Full XYZ tracking made the policy chase lateral and timing details too hard.
  This term only asks for two execution properties: lift at least as much as the
  reference, and avoid leaving the straight-walk swing foot behind the trunk.
  """
  reference_command = _get_command(env, reference_command_name)
  asset: Entity = env.scene[asset_cfg.name]
  actual_pos_b = _body_positions_in_root_frame(asset, asset_cfg.body_ids)
  reference_pos_b = reference_command.reference_foot_pos_b

  z_deficit = torch.relu(reference_pos_b[:, :, 2] - actual_pos_b[:, :, 2] - z_tolerance)
  x_trailing = torch.relu(
    reference_pos_b[:, :, 0] - actual_pos_b[:, :, 0] - x_tolerance
  )
  if twist_command_name:
    twist_command = env.command_manager.get_command(twist_command_name)
    assert twist_command is not None
    x_gate = _straight_walk_mask(
      twist_command,
      min_forward_speed=min_forward_speed,
      max_lateral_command=max_lateral_command,
      max_abs_yaw_command=max_abs_yaw_command,
    ).float()[:, None]
    velocity_ready = _forward_velocity_ready_mask(
      twist_command,
      getattr(asset.data, "root_link_lin_vel_b", None),
      max_forward_velocity_lag=max_forward_velocity_lag,
    ).float()[:, None]
    x_gate = x_gate * velocity_ready
  else:
    x_gate = torch.ones_like(x_trailing)

  weighted_error = z_weight * z_deficit + x_weight * x_trailing * x_gate
  swing_mask = (
    (~reference_command.reference_contact.bool())
    & (reference_command.snippet_id[:, None] >= 0)
    & reference_command.reference_foot_pos_valid[:, None]
  ).float()
  cost = torch.sum(weighted_error * swing_mask, dim=1)

  swing_count = torch.sum(swing_mask)
  env.extras["log"]["Metrics/reference_swing_foot_error_mean"] = torch.sum(
    weighted_error * swing_mask
  ) / torch.clamp(swing_count, min=1.0)
  env.extras["log"]["Metrics/reference_swing_foot_x_trailing_mean"] = torch.sum(
    x_trailing * x_gate * swing_mask
  ) / torch.clamp(swing_count, min=1.0)
  env.extras["log"]["Metrics/reference_swing_foot_z_deficit_mean"] = torch.sum(
    z_deficit * swing_mask
  ) / torch.clamp(swing_count, min=1.0)
  return cost


def turn_foot_separation_cost(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  max_separation: float = 0.22,
  min_abs_yaw_command: float = 0.12,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Penalize overly wide, unnatural stance during in-place turns."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  turn_gate = _turn_command_mask(
    command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  ).float()

  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  root_pos_w = asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, foot_pos_w.shape[1], -1
  )
  foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)

  separation = foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1]
  violation = torch.relu(separation - max_separation)
  env.extras["log"]["Metrics/turn_foot_separation_mean"] = torch.mean(separation)
  env.extras["log"]["Metrics/turn_foot_separation_violation_mean"] = torch.mean(
    violation * turn_gate
  )
  return violation * turn_gate
