"""Tests specific to locomotion-memory tasks."""

from typing import cast

import pytest

from mjlab.asset_zoo.robots import (
  K1_LOCOMOTION_ACTION_SCALE,
  K1_UPPER_BODY_JOINT_NAMES,
)
from mjlab.tasks.locomotion_memory.mdp import (
  LocomotionMemoryCommandCfg,
  ReferenceJointPositionActionCfg,
)
from mjlab.tasks.locomotion_memory.memory_db import default_k1_locomotion_memory_file
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


@pytest.fixture(scope="module")
def locomotion_memory_task_ids() -> list[str]:
  """Get all locomotion-memory task IDs."""
  return [task_id for task_id in list_tasks() if "LocomotionMemory" in task_id]


def test_locomotion_memory_tasks_have_required_commands(
  locomotion_memory_task_ids: list[str],
) -> None:
  """All locomotion-memory tasks should have twist and memory commands."""
  for task_id in locomotion_memory_task_ids:
    cfg = load_env_cfg(task_id)

    assert "twist" in cfg.commands, f"Task {task_id} missing 'twist' command"
    assert "memory" in cfg.commands, f"Task {task_id} missing 'memory' command"

    assert isinstance(cfg.commands["twist"], UniformVelocityCommandCfg)
    assert isinstance(cfg.commands["memory"], LocomotionMemoryCommandCfg)
    assert cfg.commands["memory"].memory_file == default_k1_locomotion_memory_file()
    assert cfg.commands["memory"].require_memory_file is True


def test_locomotion_memory_tasks_use_forward_only_velocity_ranges(
  locomotion_memory_task_ids: list[str],
) -> None:
  """Training commands should match the forward-focused K1 memory coverage."""
  for task_id in locomotion_memory_task_ids:
    cfg = load_env_cfg(task_id)
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    assert twist_cmd.heading_command is False
    assert twist_cmd.ranges.heading is None
    assert twist_cmd.rel_forward_envs == 0.15
    assert twist_cmd.rel_turn_in_place_envs == 0.2
    assert twist_cmd.turn_in_place_lin_vel_x_max == 0.12
    assert twist_cmd.turn_in_place_lin_vel_y_max == 0.03
    assert twist_cmd.turn_in_place_ang_vel_z_min == 0.10
    assert twist_cmd.ranges.lin_vel_x == (0.0, 1.0)
    assert twist_cmd.ranges.lin_vel_y == (0.0, 0.0)
    assert twist_cmd.ranges.ang_vel_z == (-0.6, 0.6)

    memory_cmd = cfg.commands["memory"]
    assert isinstance(memory_cmd, LocomotionMemoryCommandCfg)
    assert memory_cmd.event_gated_switching is True
    assert memory_cmd.min_snippet_hold_s == 0.25
    assert memory_cmd.max_snippet_hold_s == 0.6
    assert memory_cmd.blocked_resample_retry_s == 0.02
    assert memory_cmd.phase_boundary_tolerance == 0.08
    assert memory_cmd.cadence_weight == 0.4
    assert memory_cmd.cadence_scale == 0.25
    assert memory_cmd.task_weight_xy == 1.2
    assert memory_cmd.task_weight_yaw == 1.6
    assert memory_cmd.turn_intent_weight == 2.8
    assert memory_cmd.turn_intent_min_yaw_command == 0.10
    assert memory_cmd.turn_intent_max_forward_speed == 0.25
    assert memory_cmd.turn_intent_max_lateral_speed == 0.05
    assert memory_cmd.turn_intent_in_place_speed_scale == 0.12
    assert memory_cmd.turn_intent_yaw_rate_scale == 0.35
    assert memory_cmd.turn_intent_min_cadence == 1.3
    assert memory_cmd.prioritize_pivot_candidates is True
    assert memory_cmd.pivot_weight == 1.4
    assert memory_cmd.pivot_nonmatch_penalty == 0.75
    assert memory_cmd.prioritize_turn_in_place_v2_candidates is False
    assert memory_cmd.turn_in_place_v2_weight == 0.0
    assert memory_cmd.turn_in_place_v2_nonmatch_penalty == 0.75
    assert memory_cmd.straight_quality_weight == 0.8
    assert memory_cmd.straight_quality_min_forward_speed == 1.0
    assert memory_cmd.straight_quality_max_lateral_speed == 0.05
    assert memory_cmd.straight_quality_max_abs_yaw_command == 0.08
    assert memory_cmd.straight_quality_vx_scale == 0.25
    assert memory_cmd.straight_quality_yaw_rate_scale == 0.15
    assert memory_cmd.straight_quality_min_swing_lift == 0.03
    assert memory_cmd.straight_quality_lift_weight == 0.65
    assert memory_cmd.straight_quality_yaw_weight == 0.35

    command_vel = cfg.curriculum["command_vel"]
    velocity_stages = command_vel.params["velocity_stages"]
    assert velocity_stages == [
      {
        "step": 0,
        "lin_vel_x": (0.0, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.6, 0.6),
      },
      {
        "step": 8000 * 24,
        "lin_vel_x": (0.0, 1.6),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.6, 0.6),
      },
    ]


def test_locomotion_memory_tasks_have_proposal_observations(
  locomotion_memory_task_ids: list[str],
) -> None:
  """Locomotion-memory tasks should expose proposal observations."""
  for task_id in locomotion_memory_task_ids:
    cfg = load_env_cfg(task_id)
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms

    assert "memory_proposal" in actor_terms
    assert "memory_features" in actor_terms
    assert "memory_proposal" in critic_terms
    assert "memory_features" in critic_terms
    assert "memory_state" in critic_terms


def test_locomotion_memory_tasks_expose_turn_and_gait_quality_metrics(
  locomotion_memory_task_ids: list[str],
) -> None:
  for task_id in locomotion_memory_task_ids:
    cfg = load_env_cfg(task_id)
    assert "straight_foot_lane_violation" in cfg.metrics
    assert "turn_step_rate" in cfg.metrics
    assert "low_yaw_turn_active" in cfg.metrics
    assert "low_yaw_turn_yaw_error" in cfg.metrics
    assert "low_yaw_turn_step_rate" in cfg.metrics
    assert "low_yaw_turn_air_time_mean" in cfg.metrics
    assert "low_yaw_turn_foot_separation" in cfg.metrics
    assert "cadence_interval_cv" in cfg.metrics
    assert cfg.metrics["cadence_interval_cv"].reduce == "last"

    low_yaw_params = cfg.metrics["low_yaw_turn_yaw_error"].params
    assert low_yaw_params["min_abs_yaw_command"] == 0.10
    assert low_yaw_params["max_abs_yaw_command"] == 0.25
    assert low_yaw_params["max_forward_speed"] == 0.25
    assert low_yaw_params["max_lateral_command"] == 0.05


def test_locomotion_memory_tasks_lock_upper_body_residuals(
  locomotion_memory_task_ids: list[str],
) -> None:
  """Locomotion-memory tasks should zero upper-body residual scales."""
  for task_id in locomotion_memory_task_ids:
    cfg = load_env_cfg(task_id)
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, ReferenceJointPositionActionCfg)
    assert joint_pos_action.scale == K1_LOCOMOTION_ACTION_SCALE
    assert joint_pos_action.reference_command_name == "memory"
    assert joint_pos_action.turn_command_name == "twist"
    assert joint_pos_action.turn_min_abs_yaw_command == 0.10
    assert joint_pos_action.turn_max_forward_speed == 0.25
    assert joint_pos_action.turn_max_lateral_speed == 0.05
    assert joint_pos_action.turn_residual_scale == 0.9
    assert joint_pos_action.turn_residual_joint_names == (
      r".*Hip_Yaw.*",
      r".*Hip_Roll.*",
      r".*Ankle_Roll.*",
    )
    assert joint_pos_action.turn_snippet_residual_scale == 0.7
    assert joint_pos_action.turn_snippet_residual_joint_names == (
      r".*Hip_.*",
      r".*Knee.*",
      r".*Ankle.*",
    )
    assert joint_pos_action.turn_in_place_v2_residual_scale == 0.55
    assert joint_pos_action.turn_in_place_v2_residual_joint_names == (
      r".*Hip_.*",
      r".*Knee.*",
      r".*Ankle.*",
    )
    action_scale = cast(dict[str, float], joint_pos_action.scale)
    for joint_name in K1_UPPER_BODY_JOINT_NAMES:
      assert action_scale[joint_name] == 0.0


def test_locomotion_memory_tasks_have_memory_rewards(
  locomotion_memory_task_ids: list[str],
) -> None:
  """Locomotion-memory tasks should include retrieval-validity shaping rewards."""
  for task_id in locomotion_memory_task_ids:
    cfg = load_env_cfg(task_id)

    assert "memory_contact_validity" in cfg.rewards
    assert "memory_phase_validity" in cfg.rewards
    assert "memory_retrieval_switch_cost" in cfg.rewards
    assert "memory_phase_jump_cost" in cfg.rewards
    assert "memory_cadence_jump_cost" in cfg.rewards
    assert "memory_transition_bonus" in cfg.rewards
    assert "foot_midline_crossing" in cfg.rewards
    assert "straight_foot_lane" in cfg.rewards
    assert "turn_feet_air_time" in cfg.rewards
    assert "turn_reference_swing_height" in cfg.rewards
    assert "turn_swing_clearance_cost" in cfg.rewards
    assert "turn_scuff_cost" in cfg.rewards
    assert "forward_reference_swing_height" in cfg.rewards
    assert "forward_scuff_cost" in cfg.rewards
    assert "forward_swing_recovery" in cfg.rewards
    assert "turn_foot_separation" in cfg.rewards
    assert "straight_linear_velocity_tracking" in cfg.rewards
    assert "straight_yaw_velocity_tracking" in cfg.rewards
    assert "turn_yaw_velocity_tracking" in cfg.rewards
    assert "low_yaw_turn_velocity_tracking" in cfg.rewards

    assert cfg.rewards["memory_contact_validity"].weight > 0.0
    assert cfg.rewards["memory_phase_validity"].weight > 0.0
    assert cfg.rewards["memory_retrieval_switch_cost"].weight < 0.0
    assert cfg.rewards["memory_phase_jump_cost"].weight < 0.0
    assert cfg.rewards["memory_cadence_jump_cost"].weight < 0.0
    assert cfg.rewards["memory_transition_bonus"].weight > 0.0
    assert cfg.rewards["foot_midline_crossing"].weight < 0.0
    assert cfg.rewards["straight_foot_lane"].weight < 0.0
    assert cfg.rewards["turn_feet_air_time"].weight > 0.0
    assert cfg.rewards["turn_reference_swing_height"].weight > 0.0
    assert cfg.rewards["turn_swing_clearance_cost"].weight < 0.0
    assert cfg.rewards["turn_scuff_cost"].weight == 0.0
    assert cfg.rewards["forward_reference_swing_height"].weight > 0.0
    assert cfg.rewards["forward_scuff_cost"].weight < 0.0
    assert cfg.rewards["forward_swing_recovery"].weight < 0.0
    assert cfg.rewards["turn_foot_separation"].weight < 0.0
    assert cfg.rewards["straight_linear_velocity_tracking"].weight > 0.0
    assert cfg.rewards["straight_yaw_velocity_tracking"].weight > 0.0
    assert cfg.rewards["turn_yaw_velocity_tracking"].weight > 0.0
    assert cfg.rewards["low_yaw_turn_velocity_tracking"].weight > 0.0


def test_locomotion_memory_tasks_prioritize_task_rewards_over_smoothness(
  locomotion_memory_task_ids: list[str],
) -> None:
  """K1 locomotion-memory should optimize locomotion before auxiliary shaping."""
  for task_id in locomotion_memory_task_ids:
    cfg = load_env_cfg(task_id)
    assert cfg.rewards["track_linear_velocity"].weight == 3.6
    assert cfg.rewards["track_angular_velocity"].weight == 3.4
    assert cfg.rewards["upright"].weight == 1.5
    assert cfg.rewards["pose"].weight == 0.5
    assert cfg.rewards["action_rate_l2"].weight == -0.03
    assert cfg.rewards["self_collisions"].weight == -2.0
    assert cfg.rewards["foot_midline_crossing"].weight == -2.0
    assert cfg.rewards["straight_foot_lane"].weight == -2.0
    assert cfg.rewards["turn_feet_air_time"].weight == 0.55
    assert cfg.rewards["turn_reference_swing_height"].weight == 0.75
    assert cfg.rewards["turn_swing_clearance_cost"].weight == -1.0
    assert cfg.rewards["turn_scuff_cost"].weight == 0.0
    assert cfg.rewards["forward_reference_swing_height"].weight == 0.2
    assert cfg.rewards["forward_scuff_cost"].weight == -0.8
    assert cfg.rewards["forward_swing_recovery"].weight == -0.8
    assert cfg.rewards["reference_swing_foot_position"].weight == -0.6
    assert cfg.rewards["turn_foot_separation"].weight == -1.0
    assert cfg.rewards["straight_linear_velocity_tracking"].weight == 1.2
    assert cfg.rewards["straight_yaw_velocity_tracking"].weight == 1.0
    assert cfg.rewards["turn_yaw_velocity_tracking"].weight == 1.6
    assert cfg.rewards["low_yaw_turn_velocity_tracking"].weight == 1.0


def test_locomotion_memory_straight_walk_foot_rewards_match_human_like_lanes(
  locomotion_memory_task_ids: list[str],
) -> None:
  for task_id in locomotion_memory_task_ids:
    cfg = load_env_cfg(task_id)
    crossing_params = cfg.rewards["foot_midline_crossing"].params
    assert crossing_params["min_lateral_offset"] == 0.06
    assert crossing_params["min_separation"] == 0.15

    lane_params = cfg.rewards["straight_foot_lane"].params
    assert lane_params["target_lateral_offset"] == 0.095
    assert lane_params["tolerance"] == 0.02
    assert lane_params["min_forward_speed"] == 0.3
    assert lane_params["max_lateral_command"] == 0.05
    assert lane_params["max_yaw_command"] == 0.1

    turn_air_params = cfg.rewards["turn_feet_air_time"].params
    assert turn_air_params["threshold_min"] == 0.05
    assert turn_air_params["threshold_max"] == 0.45
    assert turn_air_params["min_abs_yaw_command"] == 0.10
    assert turn_air_params["max_forward_speed"] == 0.25
    assert turn_air_params["max_lateral_command"] == 0.05

    turn_ref_params = cfg.rewards["turn_reference_swing_height"].params
    assert turn_ref_params["reference_command_name"] == "memory"
    assert turn_ref_params["twist_command_name"] == "twist"
    assert turn_ref_params["threshold_min"] == 0.055
    assert turn_ref_params["threshold_max"] == 0.20
    assert turn_ref_params["min_abs_yaw_command"] == 0.10
    assert turn_ref_params["max_forward_speed"] == 0.25
    assert turn_ref_params["max_lateral_command"] == 0.05

    turn_swing_clearance_params = cfg.rewards["turn_swing_clearance_cost"].params
    assert turn_swing_clearance_params["reference_command_name"] == "memory"
    assert turn_swing_clearance_params["twist_command_name"] == "twist"
    assert turn_swing_clearance_params["min_clearance"] == 0.055
    assert turn_swing_clearance_params["min_abs_yaw_command"] == 0.10
    assert turn_swing_clearance_params["max_forward_speed"] == 0.25
    assert turn_swing_clearance_params["max_lateral_command"] == 0.05

    turn_scuff_params = cfg.rewards["turn_scuff_cost"].params
    assert turn_scuff_params["reference_command_name"] == "memory"
    assert turn_scuff_params["twist_command_name"] == "twist"
    assert turn_scuff_params["height_sensor_name"] == "foot_height_scan"
    assert turn_scuff_params["min_clearance"] == 0.04
    assert turn_scuff_params["planar_speed_threshold"] == 0.08
    assert turn_scuff_params["min_abs_yaw_command"] == 0.10
    assert turn_scuff_params["max_forward_speed"] == 0.25
    assert turn_scuff_params["max_lateral_command"] == 0.05

    forward_ref_params = cfg.rewards["forward_reference_swing_height"].params
    assert forward_ref_params["reference_command_name"] == "memory"
    assert forward_ref_params["twist_command_name"] == "twist"
    assert forward_ref_params["height_sensor_name"] == "foot_height_scan"
    assert forward_ref_params["threshold_min"] == 0.04
    assert forward_ref_params["threshold_max"] == 0.15
    assert forward_ref_params["min_forward_speed"] == 0.4
    assert forward_ref_params["max_lateral_command"] == 0.05
    assert forward_ref_params["max_abs_yaw_command"] == 0.12

    forward_scuff_params = cfg.rewards["forward_scuff_cost"].params
    assert forward_scuff_params["reference_command_name"] == "memory"
    assert forward_scuff_params["twist_command_name"] == "twist"
    assert forward_scuff_params["height_sensor_name"] == "foot_height_scan"
    assert forward_scuff_params["min_clearance"] == 0.04
    assert forward_scuff_params["forward_speed_threshold"] == 0.18
    assert forward_scuff_params["min_forward_speed"] == 0.4
    assert forward_scuff_params["max_lateral_command"] == 0.05
    assert forward_scuff_params["max_abs_yaw_command"] == 0.12
    assert forward_scuff_params["max_forward_velocity_lag"] == 0.18

    forward_recovery_params = cfg.rewards["forward_swing_recovery"].params
    assert forward_recovery_params["reference_command_name"] == "memory"
    assert forward_recovery_params["twist_command_name"] == "twist"
    assert forward_recovery_params["sensor_name"] == "feet_ground_contact"
    assert forward_recovery_params["min_recovery_x"] == -0.06
    assert forward_recovery_params["min_air_time"] == 0.04
    assert forward_recovery_params["min_forward_speed"] == 0.4
    assert forward_recovery_params["max_lateral_command"] == 0.05
    assert forward_recovery_params["max_abs_yaw_command"] == 0.12
    assert forward_recovery_params["max_forward_velocity_lag"] == 0.18

    turn_sep_params = cfg.rewards["turn_foot_separation"].params
    assert turn_sep_params["command_name"] == "twist"
    assert turn_sep_params["max_separation"] == 0.22
    assert turn_sep_params["min_abs_yaw_command"] == 0.10
    assert turn_sep_params["max_forward_speed"] == 0.25
    assert turn_sep_params["max_lateral_command"] == 0.05

    straight_yaw_params = cfg.rewards["straight_yaw_velocity_tracking"].params
    assert straight_yaw_params["command_name"] == "twist"
    assert straight_yaw_params["std"] == 0.18
    assert straight_yaw_params["min_forward_speed"] == 0.4
    assert straight_yaw_params["max_lateral_command"] == 0.05
    assert straight_yaw_params["max_abs_yaw_command"] == 0.12

    low_yaw_tracking_params = cfg.rewards["low_yaw_turn_velocity_tracking"].params
    assert low_yaw_tracking_params["command_name"] == "twist"
    assert low_yaw_tracking_params["std"] == 0.12
    assert low_yaw_tracking_params["min_abs_yaw_command"] == 0.10
    assert low_yaw_tracking_params["max_abs_yaw_command"] == 0.25
    assert low_yaw_tracking_params["max_forward_speed"] == 0.25
    assert low_yaw_tracking_params["max_lateral_command"] == 0.05


def test_locomotion_memory_pose_reward_targets_lower_body_only(
  locomotion_memory_task_ids: list[str],
) -> None:
  """Pose shaping should focus on lower-body gait quality for locomotion."""
  for task_id in locomotion_memory_task_ids:
    cfg = load_env_cfg(task_id)
    pose_params = cfg.rewards["pose"].params
    assert pose_params["asset_cfg"].joint_names == (
      r".*Hip_.*",
      r".*Knee.*",
      r".*Ankle.*",
    )
    assert pose_params["std_walking"] == {
      r".*Hip_Pitch.*": 0.3,
      r".*Hip_Roll.*": 0.1,
      r".*Hip_Yaw.*": 0.1,
      r".*Knee.*": 0.25,
      r".*Ankle_Pitch.*": 0.25,
      r".*Ankle_Roll.*": 0.08,
    }
    assert pose_params["std_running"] == {
      r".*Hip_Pitch.*": 0.5,
      r".*Hip_Roll.*": 0.15,
      r".*Hip_Yaw.*": 0.15,
      r".*Knee.*": 0.35,
      r".*Ankle_Pitch.*": 0.35,
      r".*Ankle_Roll.*": 0.1,
    }


def test_locomotion_memory_play_overrides() -> None:
  """Play mode should disable corruption and push events."""
  for task_id in [
    "Mjlab-LocomotionMemory-Flat-Booster-K1",
    "Mjlab-LocomotionMemory-Rough-Booster-K1",
  ]:
    cfg = load_env_cfg(task_id, play=True)
    assert cfg.episode_length_s >= 1e9
    assert cfg.observations["actor"].enable_corruption is False
    assert "push_robot" not in cfg.events
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    assert twist_cmd.heading_command is False
    assert twist_cmd.ranges.heading is None
    assert twist_cmd.ranges.lin_vel_x == (0.0, 1.6)
    assert twist_cmd.ranges.lin_vel_y == (0.0, 0.0)
    assert twist_cmd.ranges.ang_vel_z == (-0.6, 0.6)
    assert twist_cmd.rel_turn_in_place_envs == 0.2


def test_locomotion_memory_tasks_default_to_tensorboard_logging(
  locomotion_memory_task_ids: list[str],
) -> None:
  """Locomotion-memory research runs should train offline by default."""
  for task_id in locomotion_memory_task_ids:
    rl_cfg = load_rl_cfg(task_id)
    assert rl_cfg.logger == "tensorboard"
    assert rl_cfg.upload_model is False
