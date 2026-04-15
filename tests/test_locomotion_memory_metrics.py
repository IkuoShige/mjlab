"""Tests for locomotion-memory evaluation metrics."""

from unittest.mock import Mock

import torch

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.locomotion_memory.mdp.metrics import (
  cadence_interval_cv,
  low_yaw_turn_active,
  low_yaw_turn_air_time_mean,
  low_yaw_turn_foot_separation,
  low_yaw_turn_step_rate,
  low_yaw_turn_yaw_error,
  straight_foot_lane_violation,
  turn_step_rate,
)


def test_straight_foot_lane_violation_matches_inward_sweep_penalty() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  robot = Mock()
  robot.data.site_pos_w = torch.tensor(
    [
      [[0.2, 0.03, 0.0], [0.2, -0.04, 0.0]],
      [[0.2, 0.095, 0.0], [0.2, -0.095, 0.0]],
    ]
  )
  robot.data.root_link_pos_w = torch.zeros(2, 3)
  robot.data.root_link_quat_w = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )
  env.scene = {"robot": robot}
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.6, 0.0, 0.0],
      [0.6, 0.0, 0.2],
    ]
  )

  metric = straight_foot_lane_violation(
    env,
    command_name="twist",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    target_lateral_offset=0.095,
    tolerance=0.02,
    min_forward_speed=0.3,
    max_lateral_command=0.05,
    max_yaw_command=0.1,
  )

  assert torch.allclose(metric, torch.tensor([0.08, 0.0]), atol=1e-6)


def test_turn_step_rate_counts_touchdowns_only_for_in_place_turn_commands() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.step_dt = 0.02

  sensor = Mock()
  sensor.compute_first_contact.return_value = torch.tensor(
    [
      [True, False],
      [True, True],
      [True, False],
    ]
  )
  env.scene = {"feet_ground_contact": sensor}
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.05, 0.0, 0.4],
      [0.4, 0.0, 0.4],
      [0.05, 0.0, 0.05],
    ]
  )

  metric = turn_step_rate(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    min_abs_yaw_command=0.2,
    max_forward_speed=0.25,
    max_lateral_command=0.05,
  )

  assert torch.equal(metric, torch.tensor([1.0, 0.0, 0.0]))


def test_low_yaw_turn_metrics_filter_small_in_place_turn_commands() -> None:
  env = Mock()
  env.num_envs = 4
  env.command_manager = Mock()
  env.step_dt = 0.02

  sensor = Mock()
  sensor.compute_first_contact.return_value = torch.tensor(
    [
      [True, False],
      [True, True],
      [True, True],
      [True, True],
    ]
  )
  sensor.data.current_air_time = torch.tensor(
    [
      [0.10, 0.0],
      [0.20, 0.20],
      [0.20, 0.20],
      [0.20, 0.20],
    ]
  )

  robot = Mock()
  robot.data.root_link_ang_vel_b = torch.tensor(
    [
      [0.0, 0.0, 0.10],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ]
  )
  robot.data.site_pos_w = torch.tensor(
    [
      [[0.0, 0.12, 0.0], [0.0, -0.10, 0.0]],
      [[0.0, 0.20, 0.0], [0.0, -0.20, 0.0]],
      [[0.0, 0.20, 0.0], [0.0, -0.20, 0.0]],
      [[0.0, 0.20, 0.0], [0.0, -0.20, 0.0]],
    ]
  )
  robot.data.root_link_pos_w = torch.zeros(4, 3)
  robot.data.root_link_quat_w = torch.tensor(
    [
      [1.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0],
    ]
  )

  env.scene = {"feet_ground_contact": sensor, "robot": robot}
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.05, 0.0, 0.15],
      [0.05, 0.0, 0.35],
      [0.30, 0.0, 0.15],
      [0.05, 0.0, 0.05],
    ]
  )
  assert torch.equal(
    low_yaw_turn_active(
      env,
      command_name="twist",
      min_abs_yaw_command=0.12,
      max_abs_yaw_command=0.25,
      max_forward_speed=0.25,
      max_lateral_command=0.05,
    ),
    torch.tensor([1.0, 0.0, 0.0, 0.0]),
  )
  assert torch.allclose(
    low_yaw_turn_yaw_error(
      env,
      command_name="twist",
      min_abs_yaw_command=0.12,
      max_abs_yaw_command=0.25,
      max_forward_speed=0.25,
      max_lateral_command=0.05,
    ),
    torch.full((4,), 0.05),
  )
  assert torch.allclose(
    low_yaw_turn_step_rate(
      env,
      sensor_name="feet_ground_contact",
      command_name="twist",
      min_abs_yaw_command=0.12,
      max_abs_yaw_command=0.25,
      max_forward_speed=0.25,
      max_lateral_command=0.05,
    ),
    torch.full((4,), 1.0),
  )
  assert torch.allclose(
    low_yaw_turn_air_time_mean(
      env,
      sensor_name="feet_ground_contact",
      command_name="twist",
      min_abs_yaw_command=0.12,
      max_abs_yaw_command=0.25,
      max_forward_speed=0.25,
      max_lateral_command=0.05,
    ),
    torch.full((4,), 0.10),
  )
  assert torch.allclose(
    low_yaw_turn_foot_separation(
      env,
      command_name="twist",
      asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
      min_abs_yaw_command=0.12,
      max_abs_yaw_command=0.25,
      max_forward_speed=0.25,
      max_lateral_command=0.05,
    ),
    torch.full((4,), 0.22),
  )


class _SequenceContactSensor:
  def __init__(self, sequence: list[torch.Tensor]):
    self.sequence = sequence
    self.index = 0
    self.data = Mock(found=torch.zeros_like(sequence[0], dtype=torch.float32))

  def compute_first_contact(self, dt: float) -> torch.Tensor:
    value = self.sequence[self.index]
    self.index += 1
    return value


def test_cadence_interval_cv_distinguishes_regular_and_irregular_touchdowns() -> None:
  env = Mock()
  env.num_envs = 2
  env.device = "cpu"
  env.step_dt = 0.1
  env.command_manager = Mock()
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.6, 0.0, 0.0],
      [0.6, 0.0, 0.0],
    ]
  )
  sensor = _SequenceContactSensor(
    [
      torch.tensor([[True, False], [True, False]]),
      torch.tensor([[False, True], [False, False]]),
      torch.tensor([[True, False], [False, True]]),
      torch.tensor([[False, True], [False, False]]),
      torch.tensor([[True, False], [True, False]]),
      torch.tensor([[False, True], [False, False]]),
      torch.tensor([[False, False], [False, True]]),
      torch.tensor([[False, False], [True, False]]),
    ]
  )
  env.scene = {"feet_ground_contact": sensor}

  metric = cadence_interval_cv(
    MetricsTermCfg(
      func=cadence_interval_cv,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
    env,
  )

  value = torch.zeros(2)
  for _ in range(8):
    value = metric(
      env,
      sensor_name="feet_ground_contact",
      command_name="twist",
      command_threshold=0.1,
      min_intervals_per_foot=2,
    )

  assert value[0].item() == 0.0
  assert value[1].item() > 0.15
