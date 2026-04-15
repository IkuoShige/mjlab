"""Booster K1 locomotion-memory environment configurations."""

from mjlab.asset_zoo.robots import (
  K1_LOCOMOTION_ACTION_SCALE,
  get_k1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.locomotion_memory import mdp as locomotion_memory_mdp
from mjlab.tasks.locomotion_memory.locomotion_memory_env_cfg import (
  make_locomotion_memory_env_cfg,
)
from mjlab.tasks.locomotion_memory.mdp import (
  LocomotionMemoryCommandCfg,
  ReferenceJointPositionActionCfg,
)
from mjlab.tasks.locomotion_memory.memory_db import (
  default_k1_locomotion_memory_file,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def _apply_forward_memory_command_profile(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
) -> None:
  """Align K1 locomotion-memory commands with the forward-focused memory DB."""
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_forward_envs = 0.15
  twist_cmd.rel_turn_in_place_envs = 0.2
  twist_cmd.turn_in_place_lin_vel_x_max = 0.12
  twist_cmd.turn_in_place_lin_vel_y_max = 0.03
  twist_cmd.turn_in_place_ang_vel_z_min = 0.10
  twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

  command_vel = cfg.curriculum.get("command_vel")
  if command_vel is not None:
    command_vel.params["velocity_stages"] = [
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

  if play:
    twist_cmd.ranges.lin_vel_x = (0.0, 1.6)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)


def _apply_task_first_reward_profile(cfg: ManagerBasedRlEnvCfg) -> None:
  """Prioritize stable command-tracking locomotion over auxiliary shaping."""
  cfg.rewards["track_linear_velocity"].weight = 3.6
  cfg.rewards["track_angular_velocity"].weight = 3.4
  cfg.rewards["upright"].weight = 1.5
  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["action_rate_l2"].weight = -0.03
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_swing_height"].weight = -0.15
  cfg.rewards["memory_contact_validity"].weight = 0.03
  cfg.rewards["memory_phase_validity"].weight = 0.02
  cfg.rewards["memory_retrieval_switch_cost"].weight = -0.01
  cfg.rewards["memory_transition_bonus"].weight = 0.01


def booster_k1_rough_locomotion_memory_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 rough-terrain locomotion-memory configuration."""
  cfg = make_locomotion_memory_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_k1_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "Trunk"
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = (
        ObjRef(type="site", name="left_foot", entity="robot"),
        ObjRef(type="site", name="right_foot", entity="robot"),
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_foot_link|right_foot_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  memory_cmd = cfg.commands["memory"]
  assert isinstance(memory_cmd, LocomotionMemoryCommandCfg)
  memory_cmd.memory_file = default_k1_locomotion_memory_file()
  memory_cmd.require_memory_file = True
  memory_cmd.event_gated_switching = True
  memory_cmd.min_snippet_hold_s = 0.25
  memory_cmd.max_snippet_hold_s = 0.6
  memory_cmd.blocked_resample_retry_s = 0.02
  memory_cmd.phase_boundary_tolerance = 0.08
  memory_cmd.cadence_weight = 0.4
  memory_cmd.cadence_scale = 0.25
  memory_cmd.task_weight_xy = 1.2
  memory_cmd.task_weight_yaw = 1.6
  memory_cmd.turn_intent_weight = 2.8
  memory_cmd.turn_intent_min_yaw_command = 0.10
  memory_cmd.turn_intent_max_forward_speed = 0.25
  memory_cmd.turn_intent_max_lateral_speed = 0.05
  memory_cmd.turn_intent_in_place_speed_scale = 0.12
  memory_cmd.turn_intent_yaw_rate_scale = 0.35
  memory_cmd.turn_intent_min_cadence = 1.3
  memory_cmd.prioritize_pivot_candidates = True
  memory_cmd.pivot_weight = 1.4
  memory_cmd.pivot_nonmatch_penalty = 0.75
  memory_cmd.prioritize_turn_in_place_v2_candidates = False
  memory_cmd.turn_in_place_v2_weight = 0.0
  memory_cmd.turn_in_place_v2_nonmatch_penalty = 0.75
  memory_cmd.straight_quality_weight = 0.8
  memory_cmd.straight_quality_min_forward_speed = 1.0
  memory_cmd.straight_quality_max_lateral_speed = 0.05
  memory_cmd.straight_quality_max_abs_yaw_command = 0.08
  memory_cmd.straight_quality_vx_scale = 0.25
  memory_cmd.straight_quality_yaw_rate_scale = 0.15
  memory_cmd.straight_quality_min_swing_lift = 0.03
  memory_cmd.straight_quality_lift_weight = 0.65
  memory_cmd.straight_quality_yaw_weight = 0.35

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, ReferenceJointPositionActionCfg)
  joint_pos_action.scale = K1_LOCOMOTION_ACTION_SCALE
  joint_pos_action.turn_command_name = "twist"
  joint_pos_action.turn_min_abs_yaw_command = 0.10
  joint_pos_action.turn_max_forward_speed = 0.25
  joint_pos_action.turn_max_lateral_speed = 0.05
  joint_pos_action.turn_residual_scale = 0.9
  joint_pos_action.turn_residual_joint_names = (
    r".*Hip_Yaw.*",
    r".*Hip_Roll.*",
    r".*Ankle_Roll.*",
  )
  joint_pos_action.turn_snippet_residual_scale = 0.7
  joint_pos_action.turn_snippet_residual_joint_names = (
    r".*Hip_.*",
    r".*Knee.*",
    r".*Ankle.*",
  )
  joint_pos_action.turn_in_place_v2_residual_scale = 0.55
  joint_pos_action.turn_in_place_v2_residual_joint_names = (
    r".*Hip_.*",
    r".*Knee.*",
    r".*Ankle.*",
  )

  cfg.viewer.body_name = "Trunk"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.0

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*Hip_Pitch.*": 0.3,
    r".*Hip_Roll.*": 0.1,
    r".*Hip_Yaw.*": 0.1,
    r".*Knee.*": 0.25,
    r".*Ankle_Pitch.*": 0.25,
    r".*Ankle_Roll.*": 0.08,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*Hip_Pitch.*": 0.5,
    r".*Hip_Roll.*": 0.15,
    r".*Hip_Yaw.*": 0.15,
    r".*Knee.*": 0.35,
    r".*Ankle_Pitch.*": 0.35,
    r".*Ankle_Roll.*": 0.1,
  }
  cfg.rewards["pose"].params["asset_cfg"].joint_names = (
    r".*Hip_.*",
    r".*Knee.*",
    r".*Ankle.*",
  )

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-2.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["memory_phase_jump_cost"] = RewardTermCfg(
    func=locomotion_memory_mdp.memory_phase_jump_cost,
    weight=-0.05,
    params={"command_name": "memory"},
  )
  cfg.rewards["memory_cadence_jump_cost"] = RewardTermCfg(
    func=locomotion_memory_mdp.memory_cadence_jump_cost,
    weight=-0.04,
    params={"command_name": "memory"},
  )
  cfg.rewards["foot_midline_crossing"] = RewardTermCfg(
    func=locomotion_memory_mdp.foot_midline_crossing_cost,
    weight=-2.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
      "min_lateral_offset": 0.06,
      "min_separation": 0.15,
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["straight_foot_lane"] = RewardTermCfg(
    func=locomotion_memory_mdp.straight_foot_lane_cost,
    weight=-2.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
      "target_lateral_offset": 0.095,
      "tolerance": 0.02,
      "min_forward_speed": 0.3,
      "max_lateral_command": 0.05,
      "max_yaw_command": 0.1,
    },
  )
  cfg.rewards["turn_feet_air_time"] = RewardTermCfg(
    func=locomotion_memory_mdp.turn_feet_air_time_bonus,
    weight=0.55,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "threshold_min": 0.05,
      "threshold_max": 0.45,
      "min_abs_yaw_command": 0.10,
      "max_forward_speed": 0.25,
      "max_lateral_command": 0.05,
    },
  )
  cfg.rewards["turn_reference_swing_height"] = RewardTermCfg(
    func=locomotion_memory_mdp.turn_reference_swing_height_bonus,
    weight=0.75,
    params={
      "reference_command_name": "memory",
      "twist_command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
      "threshold_min": 0.055,
      "threshold_max": 0.20,
      "min_abs_yaw_command": 0.10,
      "max_forward_speed": 0.25,
      "max_lateral_command": 0.05,
    },
  )
  cfg.rewards["turn_swing_clearance_cost"] = RewardTermCfg(
    func=locomotion_memory_mdp.turn_swing_clearance_cost,
    weight=-1.0,
    params={
      "reference_command_name": "memory",
      "twist_command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
      "min_clearance": 0.055,
      "min_abs_yaw_command": 0.10,
      "max_forward_speed": 0.25,
      "max_lateral_command": 0.05,
    },
  )
  cfg.rewards["turn_scuff_cost"] = RewardTermCfg(
    func=locomotion_memory_mdp.turn_scuff_cost,
    weight=0.0,
    params={
      "reference_command_name": "memory",
      "twist_command_name": "twist",
      "height_sensor_name": "foot_height_scan",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
      "min_clearance": 0.04,
      "planar_speed_threshold": 0.08,
      "min_abs_yaw_command": 0.10,
      "max_forward_speed": 0.25,
      "max_lateral_command": 0.05,
    },
  )
  cfg.rewards["forward_reference_swing_height"] = RewardTermCfg(
    func=locomotion_memory_mdp.forward_reference_swing_height_bonus,
    weight=0.2,
    params={
      "reference_command_name": "memory",
      "twist_command_name": "twist",
      "height_sensor_name": "foot_height_scan",
      "threshold_min": 0.04,
      "threshold_max": 0.15,
      "min_forward_speed": 0.4,
      "max_lateral_command": 0.05,
      "max_abs_yaw_command": 0.12,
    },
  )
  cfg.rewards["forward_scuff_cost"] = RewardTermCfg(
    func=locomotion_memory_mdp.forward_scuff_cost,
    weight=-0.8,
    params={
      "reference_command_name": "memory",
      "twist_command_name": "twist",
      "height_sensor_name": "foot_height_scan",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
      "min_clearance": 0.04,
      "forward_speed_threshold": 0.18,
      "min_forward_speed": 0.4,
      "max_lateral_command": 0.05,
      "max_abs_yaw_command": 0.12,
      "max_forward_velocity_lag": 0.18,
    },
  )
  cfg.rewards["forward_swing_recovery"] = RewardTermCfg(
    func=locomotion_memory_mdp.forward_swing_recovery_cost,
    weight=-0.8,
    params={
      "reference_command_name": "memory",
      "twist_command_name": "twist",
      "sensor_name": feet_ground_cfg.name,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
      "min_recovery_x": -0.06,
      "min_air_time": 0.04,
      "min_forward_speed": 0.4,
      "max_lateral_command": 0.05,
      "max_abs_yaw_command": 0.12,
      "max_forward_velocity_lag": 0.18,
    },
  )
  cfg.rewards["reference_swing_foot_position"] = RewardTermCfg(
    func=locomotion_memory_mdp.reference_swing_foot_position_cost,
    weight=-0.6,
    params={
      "reference_command_name": "memory",
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_foot_link", "right_foot_link"),
        preserve_order=True,
      ),
      "twist_command_name": "twist",
      "x_weight": 0.8,
      "z_weight": 2.0,
      "x_tolerance": 0.03,
      "z_tolerance": 0.015,
      "min_forward_speed": 0.4,
      "max_lateral_command": 0.05,
      "max_abs_yaw_command": 0.12,
      "max_forward_velocity_lag": 0.18,
    },
  )
  cfg.rewards["turn_foot_separation"] = RewardTermCfg(
    func=locomotion_memory_mdp.turn_foot_separation_cost,
    weight=-1.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
      "max_separation": 0.22,
      "min_abs_yaw_command": 0.10,
      "max_forward_speed": 0.25,
      "max_lateral_command": 0.05,
    },
  )
  cfg.rewards["straight_linear_velocity_tracking"] = RewardTermCfg(
    func=locomotion_memory_mdp.straight_linear_velocity_tracking_bonus,
    weight=1.2,
    params={
      "command_name": "twist",
      "std": 0.45,
      "asset_cfg": SceneEntityCfg("robot"),
      "min_forward_speed": 0.4,
      "max_lateral_command": 0.05,
      "max_abs_yaw_command": 0.12,
    },
  )
  cfg.rewards["straight_yaw_velocity_tracking"] = RewardTermCfg(
    func=locomotion_memory_mdp.straight_yaw_velocity_tracking_bonus,
    weight=1.0,
    params={
      "command_name": "twist",
      "std": 0.18,
      "asset_cfg": SceneEntityCfg("robot"),
      "min_forward_speed": 0.4,
      "max_lateral_command": 0.05,
      "max_abs_yaw_command": 0.12,
    },
  )
  cfg.rewards["turn_yaw_velocity_tracking"] = RewardTermCfg(
    func=locomotion_memory_mdp.turn_yaw_velocity_tracking_bonus,
    weight=1.6,
    params={
      "command_name": "twist",
      "std": 0.22,
      "asset_cfg": SceneEntityCfg("robot"),
      "min_abs_yaw_command": 0.10,
      "max_forward_speed": 0.25,
      "max_lateral_command": 0.05,
    },
  )
  cfg.rewards["low_yaw_turn_velocity_tracking"] = RewardTermCfg(
    func=locomotion_memory_mdp.low_yaw_turn_velocity_tracking_bonus,
    weight=1.0,
    params={
      "command_name": "twist",
      "std": 0.12,
      "asset_cfg": SceneEntityCfg("robot"),
      "min_abs_yaw_command": 0.10,
      "max_abs_yaw_command": 0.25,
      "max_forward_speed": 0.25,
      "max_lateral_command": 0.05,
    },
  )

  cfg.metrics["straight_foot_lane_violation"] = MetricsTermCfg(
    func=locomotion_memory_mdp.straight_foot_lane_violation,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
      "target_lateral_offset": 0.095,
      "tolerance": 0.02,
      "min_forward_speed": 0.3,
      "max_lateral_command": 0.05,
      "max_yaw_command": 0.1,
    },
  )
  cfg.metrics["turn_step_rate"] = MetricsTermCfg(
    func=locomotion_memory_mdp.turn_step_rate,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "min_abs_yaw_command": 0.10,
      "max_forward_speed": 0.25,
      "max_lateral_command": 0.05,
    },
  )
  low_yaw_turn_metric_params = {
    "command_name": "twist",
    "min_abs_yaw_command": 0.10,
    "max_abs_yaw_command": 0.25,
    "max_forward_speed": 0.25,
    "max_lateral_command": 0.05,
  }
  cfg.metrics["low_yaw_turn_active"] = MetricsTermCfg(
    func=locomotion_memory_mdp.low_yaw_turn_active,
    params={**low_yaw_turn_metric_params},
  )
  cfg.metrics["low_yaw_turn_yaw_error"] = MetricsTermCfg(
    func=locomotion_memory_mdp.low_yaw_turn_yaw_error,
    params={**low_yaw_turn_metric_params},
  )
  cfg.metrics["low_yaw_turn_step_rate"] = MetricsTermCfg(
    func=locomotion_memory_mdp.low_yaw_turn_step_rate,
    params={
      "sensor_name": feet_ground_cfg.name,
      **low_yaw_turn_metric_params,
    },
  )
  cfg.metrics["low_yaw_turn_air_time_mean"] = MetricsTermCfg(
    func=locomotion_memory_mdp.low_yaw_turn_air_time_mean,
    params={
      "sensor_name": feet_ground_cfg.name,
      **low_yaw_turn_metric_params,
    },
  )
  cfg.metrics["low_yaw_turn_foot_separation"] = MetricsTermCfg(
    func=locomotion_memory_mdp.low_yaw_turn_foot_separation,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
      **low_yaw_turn_metric_params,
    },
  )
  cfg.metrics["cadence_interval_cv"] = MetricsTermCfg(
    func=locomotion_memory_mdp.cadence_interval_cv,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.1,
      "min_intervals_per_foot": 2,
    },
    reduce="last",
  )

  _apply_forward_memory_command_profile(cfg, play=play)
  _apply_task_first_reward_profile(cfg)

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if (
      cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
    ):
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def booster_k1_flat_locomotion_memory_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 flat-terrain locomotion-memory configuration."""
  cfg = booster_k1_rough_locomotion_memory_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  return cfg
