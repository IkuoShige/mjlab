from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .rewards import straight_foot_lane_cost

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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


def _mean_over_mask(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  masked = value * mask.float()
  denom = mask.float().sum()
  return torch.sum(masked) / torch.clamp(denom, min=1.0)


def low_yaw_turn_active(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_abs_yaw_command: float = 0.12,
  max_abs_yaw_command: float = 0.25,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Fraction of envs currently sampling low-yaw in-place turn commands."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return _low_yaw_turn_command_mask(
    command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_abs_yaw_command=max_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  ).float()


def straight_foot_lane_violation(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  target_lateral_offset: float = 0.095,
  tolerance: float = 0.02,
  min_forward_speed: float = 0.3,
  max_lateral_command: float = 0.05,
  max_yaw_command: float = 0.1,
) -> torch.Tensor:
  return straight_foot_lane_cost(
    env,
    command_name=command_name,
    asset_cfg=asset_cfg,
    target_lateral_offset=target_lateral_offset,
    tolerance=tolerance,
    min_forward_speed=min_forward_speed,
    max_lateral_command=max_lateral_command,
    max_yaw_command=max_yaw_command,
  )


def turn_step_rate(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  min_abs_yaw_command: float = 0.2,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  turn_gate = _turn_command_mask(
    command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  ).float()
  return torch.sum(first_contact.float(), dim=1) * turn_gate


def low_yaw_turn_yaw_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_abs_yaw_command: float = 0.12,
  max_abs_yaw_command: float = 0.25,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Mean yaw-rate tracking error over low-yaw in-place turn commands only."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  mask = _low_yaw_turn_command_mask(
    command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_abs_yaw_command=max_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  )
  robot: Entity = env.scene["robot"]
  error = torch.abs(command[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
  return _mean_over_mask(error, mask).expand(env.num_envs)


def low_yaw_turn_step_rate(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  min_abs_yaw_command: float = 0.12,
  max_abs_yaw_command: float = 0.25,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Mean touchdown count over low-yaw in-place turn commands only."""
  sensor: ContactSensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  mask = _low_yaw_turn_command_mask(
    command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_abs_yaw_command=max_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  )
  steps = torch.sum(first_contact.float(), dim=1)
  return _mean_over_mask(steps, mask).expand(env.num_envs)


def low_yaw_turn_air_time_mean(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  min_abs_yaw_command: float = 0.12,
  max_abs_yaw_command: float = 0.25,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Mean foot air time over low-yaw in-place turn commands only."""
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None
  mask = _low_yaw_turn_command_mask(
    command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_abs_yaw_command=max_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  )
  active_air_time = current_air_time * (current_air_time > 0).float()
  per_env_air_time = torch.sum(active_air_time, dim=1) / torch.clamp(
    (active_air_time > 0).float().sum(dim=1), min=1.0
  )
  return _mean_over_mask(per_env_air_time, mask).expand(env.num_envs)


def low_yaw_turn_foot_separation(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  min_abs_yaw_command: float = 0.12,
  max_abs_yaw_command: float = 0.25,
  max_forward_speed: float = 0.25,
  max_lateral_command: float = 0.05,
) -> torch.Tensor:
  """Mean lateral foot separation over low-yaw in-place turn commands only."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  mask = _low_yaw_turn_command_mask(
    command,
    min_abs_yaw_command=min_abs_yaw_command,
    max_abs_yaw_command=max_abs_yaw_command,
    max_forward_speed=max_forward_speed,
    max_lateral_command=max_lateral_command,
  )
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  root_pos_w = asset.data.root_link_pos_w[:, None, :]
  root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
    -1, foot_pos_w.shape[1], -1
  )
  foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)
  separation = foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1]
  return _mean_over_mask(separation, mask).expand(env.num_envs)


class cadence_interval_cv:
  """Track cadence irregularity from touchdown intervals.

  Reports the coefficient of variation of per-foot touchdown intervals. Lower
  is better. The metric is updated only while the locomotion command is active
  and keeps the latest valid value so it can be reported with ``reduce="last"``.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    found = sensor.data.found
    assert found is not None
    num_feet = found.shape[1]
    self.step_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.last_touchdown_step = torch.full(
      (env.num_envs, num_feet),
      -1,
      dtype=torch.long,
      device=env.device,
    )
    self.interval_count = torch.zeros(
      (env.num_envs, num_feet),
      dtype=torch.long,
      device=env.device,
    )
    self.interval_mean = torch.zeros(
      (env.num_envs, num_feet),
      dtype=torch.float32,
      device=env.device,
    )
    self.interval_m2 = torch.zeros_like(self.interval_mean)
    self.current_cv = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    min_intervals_per_foot: int = 2,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    first_contact = sensor.compute_first_contact(dt=env.step_dt)

    command = env.command_manager.get_command(command_name)
    assert command is not None
    active = (
      torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      > command_threshold
    )

    step_index = self.step_index[:, None]
    touchdown_mask = first_contact & active[:, None]
    valid_prev = self.last_touchdown_step >= 0
    update_mask = touchdown_mask & valid_prev

    intervals = (step_index - self.last_touchdown_step).float() * env.step_dt
    count_old = self.interval_count.float()
    count_new = count_old + update_mask.float()
    safe_count_new = torch.clamp(count_new, min=1.0)
    delta = intervals - self.interval_mean
    updated_mean = self.interval_mean + delta / safe_count_new
    delta2 = intervals - updated_mean

    self.interval_mean = torch.where(update_mask, updated_mean, self.interval_mean)
    self.interval_m2 = torch.where(
      update_mask,
      self.interval_m2 + delta * delta2,
      self.interval_m2,
    )
    self.interval_count += update_mask.long()
    self.last_touchdown_step = torch.where(
      touchdown_mask, step_index, self.last_touchdown_step
    )

    enough = self.interval_count >= min_intervals_per_foot
    variance = torch.where(
      enough,
      self.interval_m2 / torch.clamp(self.interval_count.float() - 1.0, min=1.0),
      torch.zeros_like(self.interval_m2),
    )
    cv = torch.where(
      enough & (self.interval_mean > 1e-6),
      torch.sqrt(torch.clamp(variance, min=0.0)) / self.interval_mean,
      torch.zeros_like(variance),
    )
    valid_count = torch.sum(enough.float(), dim=1)
    mean_cv = torch.sum(cv * enough.float(), dim=1) / torch.clamp(valid_count, min=1.0)
    self.current_cv = torch.where(valid_count > 0, mean_cv, self.current_cv)
    self.step_index += 1
    return self.current_cv

  def reset(self, env_ids: torch.Tensor | None = None, env=None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.step_index.shape[0], device=self.step_index.device)
    if len(env_ids) == 0:
      return
    self.step_index[env_ids] = 0
    self.last_touchdown_step[env_ids] = -1
    self.interval_count[env_ids] = 0
    self.interval_mean[env_ids] = 0.0
    self.interval_m2[env_ids] = 0.0
    self.current_cv[env_ids] = 0.0
