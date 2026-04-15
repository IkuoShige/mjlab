"""Unit tests for locomotion-memory reward helpers."""

from unittest.mock import Mock

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.locomotion_memory.mdp.rewards import (
  foot_midline_crossing_cost,
  forward_reference_swing_height_bonus,
  forward_scuff_cost,
  forward_swing_recovery_cost,
  low_yaw_turn_velocity_tracking_bonus,
  memory_cadence_jump_cost,
  memory_contact_validity,
  memory_phase_jump_cost,
  memory_phase_validity,
  memory_retrieval_switch_cost,
  memory_transition_bonus,
  reference_swing_foot_position_cost,
  straight_foot_lane_cost,
  straight_linear_velocity_tracking_bonus,
  straight_yaw_velocity_tracking_bonus,
  turn_feet_air_time_bonus,
  turn_foot_separation_cost,
  turn_reference_swing_height_bonus,
  turn_scuff_cost,
  turn_swing_clearance_cost,
  turn_yaw_velocity_tracking_bonus,
)


def _make_env_and_command() -> tuple[Mock, Mock]:
  env = Mock()
  env.command_manager = Mock()

  command = Mock()
  command.contact_valid = torch.tensor([True, False])
  command.phase_valid = torch.tensor([False, True])
  command.retrieval_switched = torch.tensor([True, False])
  command.transition_active = torch.tensor([True, True])
  command.phase_jump = torch.tensor([0.1, 0.3])
  command.cadence_jump = torch.tensor([0.2, 0.5])

  env.command_manager.get_term.return_value = command
  env.command_manager.get_command.return_value = torch.tensor(
    [[1.45, 0.0, 0.0], [0.4, 0.0, 0.0]]
  )
  return env, command


def test_memory_validity_rewards_follow_command_flags() -> None:
  env, _ = _make_env_and_command()

  assert torch.equal(memory_contact_validity(env, "memory"), torch.tensor([1.0, 0.0]))
  assert torch.equal(memory_phase_validity(env, "memory"), torch.tensor([0.0, 1.0]))
  assert torch.equal(
    memory_retrieval_switch_cost(env, "memory"), torch.tensor([1.0, 0.0])
  )
  assert torch.equal(memory_phase_jump_cost(env, "memory"), torch.tensor([0.1, 0.0]))
  assert torch.equal(memory_cadence_jump_cost(env, "memory"), torch.tensor([0.2, 0.0]))


def test_memory_transition_bonus_is_boundary_gated() -> None:
  env, _ = _make_env_and_command()

  bonus = memory_transition_bonus(
    env,
    command_name="memory",
    twist_command_name="twist",
    speed_threshold=1.5,
    speed_margin=0.2,
  )

  assert torch.equal(bonus, torch.tensor([1.0, 0.0]))


def test_straight_linear_velocity_tracking_bonus_is_straight_walk_gated() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.8, 0.0, 0.0],
      [0.8, 0.0, 0.0],
      [0.8, 0.0, 0.2],
    ]
  )

  robot = Mock()
  robot.data.root_link_lin_vel_b = torch.tensor(
    [
      [0.8, 0.0, 0.0],
      [0.2, 0.0, 0.0],
      [0.8, 0.0, 0.0],
    ]
  )
  env.scene = {"robot": robot}

  reward = straight_linear_velocity_tracking_bonus(
    env,
    command_name="twist",
    std=0.5,
    asset_cfg=SceneEntityCfg("robot"),
    min_forward_speed=0.4,
    max_lateral_command=0.05,
    max_abs_yaw_command=0.12,
  )

  expected = torch.tensor([1.0, torch.exp(torch.tensor(-1.44)), 0.0])
  assert torch.allclose(reward, expected, atol=1e-6)
  assert "Metrics/straight_linear_tracking_error_mean" in env.extras["log"]


def test_straight_yaw_velocity_tracking_bonus_penalizes_straight_yaw_drift() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.2],
    ]
  )

  robot = Mock()
  robot.data.root_link_ang_vel_b = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.18],
      [0.0, 0.0, 0.0],
    ]
  )
  env.scene = {"robot": robot}

  reward = straight_yaw_velocity_tracking_bonus(
    env,
    command_name="twist",
    std=0.18,
    asset_cfg=SceneEntityCfg("robot"),
    min_forward_speed=0.4,
    max_lateral_command=0.05,
    max_abs_yaw_command=0.12,
  )

  expected = torch.tensor([1.0, torch.exp(torch.tensor(-1.0)), 0.0])
  assert torch.allclose(reward, expected, atol=1e-6)
  assert "Metrics/straight_yaw_tracking_error_mean" in env.extras["log"]


def test_turn_yaw_velocity_tracking_bonus_is_in_place_turn_gated() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.05, 0.0, 0.15],
      [0.05, 0.0, 0.15],
      [0.30, 0.0, 0.15],
    ]
  )

  robot = Mock()
  robot.data.root_link_ang_vel_b = torch.tensor(
    [
      [0.0, 0.0, 0.15],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.15],
    ]
  )
  env.scene = {"robot": robot}

  reward = turn_yaw_velocity_tracking_bonus(
    env,
    command_name="twist",
    std=0.25,
    asset_cfg=SceneEntityCfg("robot"),
    min_abs_yaw_command=0.12,
    max_forward_speed=0.25,
    max_lateral_command=0.05,
  )

  expected = torch.tensor([1.0, torch.exp(torch.tensor(-0.36)), 0.0])
  assert torch.allclose(reward, expected, atol=1e-6)
  assert "Metrics/turn_yaw_tracking_error_mean" in env.extras["log"]


def test_low_yaw_turn_velocity_tracking_bonus_targets_low_yaw_only() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.05, 0.0, 0.15],
      [0.05, 0.0, -0.15],
      [0.05, 0.0, 0.35],
      [0.30, 0.0, 0.15],
    ]
  )

  robot = Mock()
  robot.data.root_link_ang_vel_b = torch.tensor(
    [
      [0.0, 0.0, 0.15],
      [0.0, 0.0, -0.03],
      [0.0, 0.0, 0.35],
      [0.0, 0.0, 0.15],
    ]
  )
  env.scene = {"robot": robot}

  reward = low_yaw_turn_velocity_tracking_bonus(
    env,
    command_name="twist",
    std=0.12,
    asset_cfg=SceneEntityCfg("robot"),
    min_abs_yaw_command=0.10,
    max_abs_yaw_command=0.25,
    max_forward_speed=0.25,
    max_lateral_command=0.05,
  )

  expected = torch.tensor([1.0, torch.exp(torch.tensor(-1.0)), 0.0, 0.0])
  assert torch.allclose(reward, expected, atol=1e-6)
  assert "Metrics/low_yaw_turn_tracking_error_mean" in env.extras["log"]
  assert "Metrics/low_yaw_turn_left_error_mean" in env.extras["log"]
  assert "Metrics/low_yaw_turn_right_error_mean" in env.extras["log"]


def test_foot_midline_crossing_cost_penalizes_crossing_and_narrow_steps() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  robot = Mock()
  robot.data.site_pos_w = torch.tensor(
    [
      [[0.1, 0.02, 0.0], [0.1, -0.01, 0.0]],
      [[0.1, 0.08, 0.0], [0.1, -0.08, 0.0]],
    ]
  )
  robot.data.root_link_pos_w = torch.zeros(2, 3)
  robot.data.root_link_quat_w = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )
  env.scene = {"robot": robot}
  env.command_manager.get_command.return_value = torch.tensor(
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
  )

  cost = foot_midline_crossing_cost(
    env,
    command_name="twist",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    min_lateral_offset=0.04,
    min_separation=0.12,
    command_threshold=0.05,
  )

  assert torch.allclose(cost, torch.tensor([0.14, 0.0]), atol=1e-6)
  assert "Metrics/foot_midline_violation_mean" in env.extras["log"]


def test_straight_foot_lane_cost_penalizes_inward_sweep_only_for_straight_walk() -> (
  None
):
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  robot = Mock()
  robot.data.site_pos_w = torch.tensor(
    [
      [[0.2, 0.03, 0.0], [0.2, -0.04, 0.0]],
      [[0.2, 0.095, 0.0], [0.2, -0.095, 0.0]],
      [[0.2, 0.03, 0.0], [0.2, -0.04, 0.0]],
    ]
  )
  robot.data.root_link_pos_w = torch.zeros(3, 3)
  robot.data.root_link_quat_w = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )
  env.scene = {"robot": robot}
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.6, 0.0, 0.0],
      [0.6, 0.0, 0.0],
      [0.6, 0.0, 0.2],
    ]
  )

  cost = straight_foot_lane_cost(
    env,
    command_name="twist",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    target_lateral_offset=0.095,
    tolerance=0.02,
    min_forward_speed=0.3,
    max_lateral_command=0.05,
    max_yaw_command=0.1,
  )

  assert torch.allclose(cost, torch.tensor([0.08, 0.0, 0.0]), atol=1e-6)
  assert "Metrics/foot_lane_violation_mean" in env.extras["log"]


def test_turn_feet_air_time_bonus_rewards_stepping_only_for_in_place_turns() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  sensor = Mock()
  sensor.data.current_air_time = torch.tensor(
    [
      [0.2, 0.0],
      [0.2, 0.2],
      [0.2, 0.2],
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

  reward = turn_feet_air_time_bonus(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    threshold_min=0.05,
    threshold_max=0.45,
    min_abs_yaw_command=0.2,
    max_forward_speed=0.25,
    max_lateral_command=0.05,
  )

  assert torch.equal(reward, torch.tensor([1.0, 0.0, 0.0]))
  assert "Metrics/turn_air_time_mean" in env.extras["log"]


def test_turn_reference_swing_height_bonus_rewards_v2_swing_clearance() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  command = Mock()
  command.turn_snippet_active = torch.tensor([True, True, False])
  command.reference_contact = torch.tensor(
    [
      [False, True],
      [False, True],
      [False, True],
    ]
  )
  env.command_manager.get_term.return_value = command
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.05, 0.0, 0.4],
      [0.30, 0.0, 0.4],
      [0.05, 0.0, 0.4],
    ]
  )

  robot = Mock()
  robot.data.site_pos_w = torch.tensor(
    [
      [[0.1, 0.0, 0.12], [0.1, 0.0, 0.00]],
      [[0.1, 0.0, 0.12], [0.1, 0.0, 0.00]],
      [[0.1, 0.0, 0.12], [0.1, 0.0, 0.00]],
    ]
  )
  env.scene = {"robot": robot}

  reward = turn_reference_swing_height_bonus(
    env,
    reference_command_name="memory",
    twist_command_name="twist",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    threshold_min=0.035,
    threshold_max=0.20,
    min_abs_yaw_command=0.2,
    max_forward_speed=0.25,
    max_lateral_command=0.05,
  )

  assert torch.equal(reward, torch.tensor([1.0, 0.0, 0.0]))
  assert "Metrics/turn_reference_swing_clearance_mean" in env.extras["log"]


def test_turn_swing_clearance_cost_penalizes_low_reference_swing_clearance() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  command = Mock()
  command.turn_snippet_active = torch.tensor([True, True, False])
  command.reference_contact = torch.tensor(
    [
      [False, True],
      [False, True],
      [False, True],
    ]
  )
  env.command_manager.get_term.return_value = command
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.05, 0.0, 0.15],
      [0.30, 0.0, 0.15],
      [0.05, 0.0, 0.15],
    ]
  )

  robot = Mock()
  robot.data.site_pos_w = torch.tensor(
    [
      [[0.1, 0.0, 0.02], [0.1, 0.0, 0.00]],
      [[0.1, 0.0, 0.02], [0.1, 0.0, 0.00]],
      [[0.1, 0.0, 0.02], [0.1, 0.0, 0.00]],
    ]
  )
  env.scene = {"robot": robot}

  cost = turn_swing_clearance_cost(
    env,
    reference_command_name="memory",
    twist_command_name="twist",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    min_clearance=0.055,
    min_abs_yaw_command=0.12,
    max_forward_speed=0.25,
    max_lateral_command=0.05,
  )

  assert torch.allclose(cost, torch.tensor([0.035, 0.0, 0.0]), atol=1e-6)
  assert "Metrics/turn_swing_clearance_deficit_mean" in env.extras["log"]


def test_turn_scuff_cost_penalizes_low_clearance_planar_sweep_only_for_turns() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  command = Mock()
  command.snippet_id = torch.tensor([3, 4, 5])
  command.turn_snippet_active = torch.tensor([True, True, False])
  command.reference_contact = torch.tensor(
    [
      [False, True],
      [False, True],
      [False, True],
    ]
  )
  env.command_manager.get_term.return_value = command
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.05, 0.0, 0.15],
      [0.30, 0.0, 0.15],
      [0.05, 0.0, 0.15],
    ]
  )

  height_sensor = Mock()
  height_sensor.data.heights = torch.tensor(
    [
      [0.01, 0.0],
      [0.01, 0.0],
      [0.01, 0.0],
    ]
  )
  robot = Mock()
  robot.data.site_lin_vel_w = torch.tensor(
    [
      [[0.0, 0.30, 0.0], [0.00, 0.0, 0.0]],
      [[0.0, 0.30, 0.0], [0.00, 0.0, 0.0]],
      [[0.0, 0.30, 0.0], [0.00, 0.0, 0.0]],
    ]
  )
  robot.data.root_link_quat_w = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )
  robot.data.root_link_lin_vel_b = torch.tensor(
    [
      [0.50, 0.0, 0.0],
      [0.80, 0.0, 0.0],
      [0.80, 0.0, 0.0],
    ]
  )
  env.scene = {"foot_height_scan": height_sensor, "robot": robot}

  cost = turn_scuff_cost(
    env,
    reference_command_name="memory",
    twist_command_name="twist",
    height_sensor_name="foot_height_scan",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    min_clearance=0.04,
    planar_speed_threshold=0.1,
    min_abs_yaw_command=0.12,
    max_forward_speed=0.25,
    max_lateral_command=0.05,
  )

  assert torch.allclose(cost, torch.tensor([0.006, 0.0, 0.0]), atol=1e-6)
  assert "Metrics/turn_scuff_risk_mean" in env.extras["log"]
  assert "Metrics/turn_planar_swing_speed_mean" in env.extras["log"]


def test_forward_reference_swing_height_bonus_rewards_straight_walk_swing_clearance() -> (
  None
):
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  command = Mock()
  command.snippet_id = torch.tensor([3, 4, -1])
  command.reference_contact = torch.tensor(
    [
      [False, True],
      [False, True],
      [False, True],
    ]
  )
  env.command_manager.get_term.return_value = command
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.8, 0.0, 0.0],
      [0.8, 0.0, 0.2],
      [0.8, 0.0, 0.0],
    ]
  )

  height_sensor = Mock()
  height_sensor.data.heights = torch.tensor(
    [
      [0.06, 0.0],
      [0.06, 0.0],
      [0.06, 0.0],
    ]
  )
  env.scene = {"foot_height_scan": height_sensor}

  reward = forward_reference_swing_height_bonus(
    env,
    reference_command_name="memory",
    twist_command_name="twist",
    height_sensor_name="foot_height_scan",
    threshold_min=0.04,
    threshold_max=0.15,
    min_forward_speed=0.4,
    max_lateral_command=0.05,
    max_abs_yaw_command=0.12,
  )

  assert torch.equal(reward, torch.tensor([1.0, 0.0, 0.0]))
  assert "Metrics/forward_reference_swing_clearance_mean" in env.extras["log"]


def test_forward_scuff_cost_penalizes_low_clearance_forward_sweep_only_in_straight_walk() -> (
  None
):
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  command = Mock()
  command.snippet_id = torch.tensor([3, 4, 5])
  command.reference_contact = torch.tensor(
    [
      [False, True],
      [False, True],
      [False, True],
    ]
  )
  env.command_manager.get_term.return_value = command
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.8, 0.0, 0.0],
      [0.8, 0.0, 0.2],
      [0.8, 0.0, 0.0],
    ]
  )

  height_sensor = Mock()
  height_sensor.data.heights = torch.tensor(
    [
      [0.01, 0.0],
      [0.01, 0.0],
      [0.05, 0.0],
    ]
  )

  robot = Mock()
  robot.data.site_lin_vel_w = torch.tensor(
    [
      [[0.30, 0.0, 0.0], [0.00, 0.0, 0.0]],
      [[0.30, 0.0, 0.0], [0.00, 0.0, 0.0]],
      [[0.30, 0.0, 0.0], [0.00, 0.0, 0.0]],
    ]
  )
  robot.data.root_link_quat_w = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )
  robot.data.root_link_lin_vel_b = torch.tensor(
    [
      [0.50, 0.0, 0.0],
      [0.80, 0.0, 0.0],
      [0.80, 0.0, 0.0],
    ]
  )
  env.scene = {"foot_height_scan": height_sensor, "robot": robot}

  cost = forward_scuff_cost(
    env,
    reference_command_name="memory",
    twist_command_name="twist",
    height_sensor_name="foot_height_scan",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    min_clearance=0.028,
    forward_speed_threshold=0.18,
    min_forward_speed=0.4,
    max_lateral_command=0.05,
    max_abs_yaw_command=0.12,
  )

  assert torch.allclose(cost, torch.tensor([0.00216, 0.0, 0.0]), atol=1e-6)
  assert "Metrics/forward_scuff_risk_mean" in env.extras["log"]

  gated_cost = forward_scuff_cost(
    env,
    reference_command_name="memory",
    twist_command_name="twist",
    height_sensor_name="foot_height_scan",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    min_clearance=0.028,
    forward_speed_threshold=0.18,
    min_forward_speed=0.4,
    max_lateral_command=0.05,
    max_abs_yaw_command=0.12,
    max_forward_velocity_lag=0.1,
  )
  assert torch.equal(gated_cost, torch.zeros(3))
  assert "Metrics/forward_velocity_ready_rate" in env.extras["log"]


def test_forward_swing_recovery_cost_penalizes_trailing_swing_foot() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  command = Mock()
  command.snippet_id = torch.tensor([3, 4, 5])
  command.reference_contact = torch.tensor(
    [
      [False, True],
      [False, True],
      [False, True],
    ]
  )
  env.command_manager.get_term.return_value = command
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.8, 0.0, 0.0],
      [0.8, 0.0, 0.2],
      [0.8, 0.0, 0.0],
    ]
  )

  sensor = Mock()
  sensor.data.current_air_time = torch.tensor(
    [
      [0.08, 0.0],
      [0.08, 0.0],
      [0.02, 0.0],
    ]
  )
  robot = Mock()
  robot.data.site_pos_w = torch.tensor(
    [
      [[-0.16, 0.0, 0.04], [0.0, 0.0, 0.0]],
      [[-0.16, 0.0, 0.04], [0.0, 0.0, 0.0]],
      [[-0.16, 0.0, 0.04], [0.0, 0.0, 0.0]],
    ]
  )
  robot.data.root_link_pos_w = torch.zeros(3, 3)
  robot.data.root_link_quat_w = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )
  robot.data.root_link_lin_vel_b = torch.tensor(
    [
      [0.50, 0.0, 0.0],
      [0.80, 0.0, 0.0],
      [0.80, 0.0, 0.0],
    ]
  )
  env.scene = {"feet_ground_contact": sensor, "robot": robot}

  cost = forward_swing_recovery_cost(
    env,
    reference_command_name="memory",
    twist_command_name="twist",
    sensor_name="feet_ground_contact",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    min_recovery_x=-0.10,
    min_air_time=0.04,
    min_forward_speed=0.4,
    max_lateral_command=0.05,
    max_abs_yaw_command=0.12,
  )

  assert torch.allclose(cost, torch.tensor([0.06, 0.0, 0.0]), atol=1e-6)
  assert "Metrics/forward_swing_trail_mean" in env.extras["log"]

  gated_cost = forward_swing_recovery_cost(
    env,
    reference_command_name="memory",
    twist_command_name="twist",
    sensor_name="feet_ground_contact",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    min_recovery_x=-0.10,
    min_air_time=0.04,
    min_forward_speed=0.4,
    max_lateral_command=0.05,
    max_abs_yaw_command=0.12,
    max_forward_velocity_lag=0.1,
  )
  assert torch.equal(gated_cost, torch.zeros(3))
  assert "Metrics/forward_velocity_ready_rate" in env.extras["log"]


def test_reference_swing_foot_position_cost_targets_lift_and_forward_recovery() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}

  command = Mock()
  command.snippet_id = torch.tensor([3, 4, -1])
  command.reference_contact = torch.tensor(
    [
      [False, True],
      [False, True],
      [False, True],
    ]
  )
  command.reference_foot_pos_valid = torch.tensor([True, False, True])
  command.reference_foot_pos_b = torch.tensor(
    [
      [[0.10, 0.10, -0.48], [0.00, -0.10, -0.52]],
      [[0.10, 0.10, -0.48], [0.00, -0.10, -0.52]],
      [[0.10, 0.10, -0.48], [0.00, -0.10, -0.52]],
    ]
  )
  env.command_manager.get_term.return_value = command
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.8, 0.0, 0.0],
      [0.8, 0.0, 0.0],
      [0.8, 0.0, 0.0],
    ]
  )

  robot = Mock()
  robot.data.body_link_pos_w = torch.tensor(
    [
      [[0.04, 0.12, 0.02], [0.00, -0.10, 0.00]],
      [[0.20, 0.12, 0.04], [0.00, -0.10, 0.00]],
      [[0.15, 0.12, 0.04], [0.00, -0.10, 0.00]],
    ]
  )
  robot.data.root_link_pos_w = torch.tensor(
    [
      [0.0, 0.0, 0.52],
      [0.0, 0.0, 0.52],
      [0.0, 0.0, 0.52],
    ]
  )
  robot.data.root_link_quat_w = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )
  env.scene = {"robot": robot}

  cost = reference_swing_foot_position_cost(
    env,
    reference_command_name="memory",
    asset_cfg=SceneEntityCfg("robot", body_ids=[0, 1]),
    twist_command_name="twist",
    x_weight=1.0,
    z_weight=2.0,
    x_tolerance=0.01,
    z_tolerance=0.005,
    min_forward_speed=0.4,
    max_lateral_command=0.05,
    max_abs_yaw_command=0.12,
  )

  assert torch.allclose(cost, torch.tensor([0.08, 0.0, 0.0]), atol=1e-6)
  assert "Metrics/reference_swing_foot_error_mean" in env.extras["log"]
  assert "Metrics/reference_swing_foot_x_trailing_mean" in env.extras["log"]
  assert "Metrics/reference_swing_foot_z_deficit_mean" in env.extras["log"]


def test_turn_foot_separation_cost_penalizes_wide_low_yaw_turn_stance() -> None:
  env = Mock()
  env.command_manager = Mock()
  env.extras = {"log": {}}
  env.command_manager.get_command.return_value = torch.tensor(
    [
      [0.05, 0.0, 0.15],
      [0.30, 0.0, 0.15],
      [0.05, 0.0, 0.05],
    ]
  )

  robot = Mock()
  robot.data.site_pos_w = torch.tensor(
    [
      [[0.0, 0.14, 0.0], [0.0, -0.12, 0.0]],
      [[0.0, 0.14, 0.0], [0.0, -0.12, 0.0]],
      [[0.0, 0.14, 0.0], [0.0, -0.12, 0.0]],
    ]
  )
  robot.data.root_link_pos_w = torch.zeros(3, 3)
  robot.data.root_link_quat_w = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
  )
  env.scene = {"robot": robot}

  cost = turn_foot_separation_cost(
    env,
    command_name="twist",
    asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]),
    max_separation=0.22,
    min_abs_yaw_command=0.12,
    max_forward_speed=0.25,
    max_lateral_command=0.05,
  )

  assert torch.allclose(cost, torch.tensor([0.04, 0.0, 0.0]), atol=1e-6)
  assert "Metrics/turn_foot_separation_mean" in env.extras["log"]
  assert "Metrics/turn_foot_separation_violation_mean" in env.extras["log"]
