"""Booster K1 soccer environment configurations."""

import glob
import math

from mjlab.asset_zoo.objects.soccer_ball import (
  SOCCER_BALL_RADIUS,
  get_soccer_ball_cfg,
)
from mjlab.asset_zoo.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from mjlab.asset_zoo.robots.booster_k1 import K1_FOOT_GEOM_REGEX
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.soccer import mdp as soccer_mdp
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, make_tracking_env_cfg

# Motion directory for K1 (retargeted via k1_retarget/scripts/soccer_npz_g1_to_k1.py).
_DEFAULT_MOTION_DIR = "motions/soccer-standard-mj-k1"

# K1 is a smaller, lighter robot than G1 so the soccer-kick reward thresholds
# and ball-placement ranges must scale accordingly. Values below are derived
# from measurements of the retargeted motions and mass ratios:
#   body_scale (K1/G1 standing height) = 0.678
#   mass ratio (K1/G1)                 = 0.590  (19.7 kg / 33.3 kg)
#   impact-force ratio  = mass * velocity ratio ≈ 0.4
# In the retargeted motions K1's peak kick-foot speed is ~3.4–4.0 m/s (G1 is
# ~5.2–5.6 m/s), so a 3.0 m/s gate would mostly silence kick rewards on K1.
_K1_MIN_FOOT_SPEED = 2.0
_K1_HORIZONTAL_FORCE_THRESHOLD = 15
_K1_BALL_VELOCITY_THRESHOLD = 0.35
_K1_BALL_SPEED_STD = 0.8
_K1_CURVE_RADIUS_OFFSET = 0.17


def _discover_motion_files(motion_dir: str = _DEFAULT_MOTION_DIR) -> list[str]:
  files = sorted(glob.glob(f"{motion_dir}/*.npz"))
  return files


def k1_flat_soccer_kick_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 flat terrain soccer kicking configuration."""
  cfg = make_tracking_env_cfg()

  # ------------------------------------------------------------------
  # Scene: robot + soccer ball + sensors
  # ------------------------------------------------------------------

  cfg.scene.entities = {
    "robot": get_k1_robot_cfg(),
    "soccer_ball": get_soccer_ball_cfg(),
  }

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  ball_contact_cfg = ContactSensorCfg(
    name="ball_contact",
    primary=ContactMatch(mode="body", pattern="ball_body", entity="soccer_ball"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg, ball_contact_cfg)

  # ------------------------------------------------------------------
  # Actions
  # ------------------------------------------------------------------

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = K1_ACTION_SCALE

  # ------------------------------------------------------------------
  # Commands: soccer motion with K1 bodies
  # ------------------------------------------------------------------

  def _foot_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      body_names=("left_foot_link", "right_foot_link"),
    )

  # K1 bodies analogous to G1_BODY_NAMES in soccer/config/g1/env_cfgs.py.
  # K1 has no separate torso body above a waist joint, so Trunk (which is
  # both root and upper-body attachment point) appears once. K1 has no
  # wrist joints; left/right_hand_link stands in for the wrist tip.
  K1_BODY_NAMES = (
    "Trunk",
    "Left_Hip_Roll",
    "Left_Shank",
    "left_foot_link",
    "Right_Hip_Roll",
    "Right_Shank",
    "right_foot_link",
    "Left_Arm_2",
    "Left_Arm_3",
    "left_hand_link",
    "Right_Arm_2",
    "Right_Arm_3",
    "right_hand_link",
  )

  cfg.commands = {
    "motion": soccer_mdp.SoccerMotionCommandCfg(
      entity_name="robot",
      soccer_ball_entity_name="soccer_ball",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      anchor_body_name="Trunk",
      body_names=K1_BODY_NAMES,
      motion_files=_discover_motion_files(),
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      curve_offset_range={
        "radius": (-_K1_CURVE_RADIUS_OFFSET, _K1_CURVE_RADIUS_OFFSET),
        "arc_angle": math.pi / 9,
        "height": SOCCER_BALL_RADIUS,
      },
      blind_distance_min_range=(0.2, 0.8),
      blind_distance_max_range=(1.8, 2.5),
      foot_body_names=("left_foot_link", "right_foot_link"),
    )
  }

  # ------------------------------------------------------------------
  # Observations
  # ------------------------------------------------------------------

  actor_terms = {
    "command": ObservationTermCfg(
      func=tracking_mdp.generated_commands,
      params={"command_name": "motion"},
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=tracking_mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=tracking_mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=tracking_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "joint_pos": ObservationTermCfg(
      func=tracking_mdp.joint_pos_rel,
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(func=tracking_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=tracking_mdp.last_action),
    "target_point_pos": ObservationTermCfg(
      func=soccer_mdp.blind_zone_target_point_pos,
      params={"command_name": "motion"},
    ),
    "target_destination_pos_local": ObservationTermCfg(
      func=soccer_mdp.target_destination_pos_local,
      params={"command_name": "motion"},
    ),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=tracking_mdp.generated_commands,
      params={"command_name": "motion"},
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=tracking_mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=tracking_mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
    ),
    "body_pos": ObservationTermCfg(
      func=tracking_mdp.robot_body_pos_b,
      params={"command_name": "motion"},
    ),
    "body_ori": ObservationTermCfg(
      func=tracking_mdp.robot_body_ori_b,
      params={"command_name": "motion"},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=tracking_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=tracking_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "joint_pos": ObservationTermCfg(func=tracking_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=tracking_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=tracking_mdp.last_action),
    "target_point_pos": ObservationTermCfg(
      func=soccer_mdp.constant_target_point_pos,
      params={"command_name": "motion"},
    ),
    "target_destination_pos_local": ObservationTermCfg(
      func=soccer_mdp.target_destination_pos_local,
      params={"command_name": "motion"},
    ),
  }

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # ------------------------------------------------------------------
  # Rewards
  # ------------------------------------------------------------------

  TRACKING_BODY_NAMES = [
    "Trunk",
    "Left_Hip_Roll",
    "Left_Shank",
    "Right_Hip_Roll",
    "Right_Shank",
    "Left_Arm_2",
    "Left_Arm_3",
    "left_hand_link",
    "Right_Arm_2",
    "Right_Arm_3",
    "right_hand_link",
  ]

  cfg.rewards = {
    # Motion tracking.
    "motion_global_root_pos": RewardTermCfg(
      func=tracking_mdp.motion_global_anchor_position_error_exp,
      weight=0.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=tracking_mdp.motion_global_anchor_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=tracking_mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.3,
        "body_names": TRACKING_BODY_NAMES,
      },
    ),
    "motion_body_ori": RewardTermCfg(
      func=tracking_mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.4,
        "body_names": TRACKING_BODY_NAMES,
      },
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=tracking_mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=tracking_mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    # motion_foot_pos weight is reduced from the G1 default of 1.0 because on
    # K1 the retargeted kick foot lands ~0.2 m short of the ball at the kick
    # frame (measured); a strong foot-tracking reward then prevents the policy
    # from deviating to reach the ball. 0.3 still anchors the swing foot to
    # the motion but leaves room for the RL policy to close the remaining gap.
    "motion_foot_pos": RewardTermCfg(
      func=soccer_mdp.motion_relative_foot_position_error_exp,
      weight=0.3,
      params={
        "command_name": "motion",
        "std": 0.3,
        "foot_body_names": ["left_foot_link", "right_foot_link"],
      },
    ),
    # Action penalties. K1 has no waist joints, so waist_action_rate_l2 is
    # omitted (G1-specific).
    "action_rate_l2": RewardTermCfg(func=tracking_mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=tracking_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=tracking_mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "foot_distance": RewardTermCfg(
      func=soccer_mdp.foot_distance,
      weight=0.2,
      params={"threshold": 0.24, "std": 0.5, "foot_cfg": _foot_cfg()},
    ),
    "target_point_proximity": RewardTermCfg(
      func=soccer_mdp.target_point_proximity,
      weight=1.0,
      params={"std": 4.0, "command_name": "motion"},
    ),
    "pelvis_orientation": RewardTermCfg(
      func=soccer_mdp.pelvis_orientation,
      weight=-1.0,
      params={"command_name": "motion"},
    ),
    "target_point_contact": RewardTermCfg(
      func=soccer_mdp.target_point_contact,
      weight=50.0,
      params={
        "command_name": "motion",
        "ball_sensor_name": "ball_contact",
        "horizontal_force_threshold": _K1_HORIZONTAL_FORCE_THRESHOLD,
        "min_foot_speed": _K1_MIN_FOOT_SPEED,
        "foot_cfg": _foot_cfg(),
      },
    ),
    "sideways_kick": RewardTermCfg(
      func=soccer_mdp.sideways_kick,
      weight=50.0,
      params={
        "command_name": "motion",
        "ball_sensor_name": "ball_contact",
        "horizontal_force_threshold": _K1_HORIZONTAL_FORCE_THRESHOLD,
        "min_foot_speed": _K1_MIN_FOOT_SPEED,
        "foot_cfg": _foot_cfg(),
      },
    ),
    "ball_velocity_direction_alignment": RewardTermCfg(
      func=soccer_mdp.ball_velocity_direction_alignment,
      weight=30.0,
      params={
        "command_name": "motion",
        "std": 0.8,
        "velocity_threshold": _K1_BALL_VELOCITY_THRESHOLD,
        "ball_sensor_name": "ball_contact",
        "horizontal_force_threshold": _K1_HORIZONTAL_FORCE_THRESHOLD,
        "min_foot_speed": _K1_MIN_FOOT_SPEED,
        "foot_cfg": _foot_cfg(),
      },
    ),
    "ball_speed_reward": RewardTermCfg(
      func=soccer_mdp.ball_speed_reward,
      weight=10.0,
      params={
        "command_name": "motion",
        "std": _K1_BALL_SPEED_STD,
        "velocity_threshold": _K1_BALL_VELOCITY_THRESHOLD,
        "ball_sensor_name": "ball_contact",
        "horizontal_force_threshold": _K1_HORIZONTAL_FORCE_THRESHOLD,
        "min_foot_speed": _K1_MIN_FOOT_SPEED,
        "foot_cfg": _foot_cfg(),
      },
    ),
    "ball_z_speed_penalty": RewardTermCfg(
      func=soccer_mdp.ball_z_speed_penalty_reward,
      weight=-0.0,
      params={
        "command_name": "motion",
        "std": 3,
        "velocity_threshold": _K1_BALL_VELOCITY_THRESHOLD,
      },
    ),
  }

  # ------------------------------------------------------------------
  # Terminations
  # ------------------------------------------------------------------

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=tracking_mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=tracking_mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=tracking_mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=tracking_mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (
          "left_foot_link",
          "right_foot_link",
          "left_hand_link",
          "right_hand_link",
        ),
      },
    ),
  }

  # ------------------------------------------------------------------
  # Events (domain randomization)
  # ------------------------------------------------------------------

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = K1_FOOT_GEOM_REGEX
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  # ------------------------------------------------------------------
  # Simulation
  # ------------------------------------------------------------------

  cfg.sim.nconmax = 50
  cfg.sim.njmax = 300

  # ------------------------------------------------------------------
  # Viewer
  # ------------------------------------------------------------------

  cfg.viewer.body_name = "Trunk"

  # ------------------------------------------------------------------
  # Play mode overrides
  # ------------------------------------------------------------------

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, soccer_mdp.SoccerMotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

  return cfg


def k1_flat_soccer_moving_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Soccer kick with moving ball."""
  cfg = k1_flat_soccer_kick_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, soccer_mdp.SoccerMotionCommandCfg)
  motion_cmd.enable_soccer_ball_init_vel = True
  motion_cmd.soccer_ball_init_lin_vel_range = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (0.0, 0.0),
  }
  return cfg


def k1_flat_soccer_teacher_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Teacher policy with privileged actor observations."""
  cfg = k1_flat_soccer_kick_env_cfg(play=play)

  actor = cfg.observations["actor"]
  actor.terms["motion_anchor_pos_b"] = ObservationTermCfg(
    func=tracking_mdp.motion_anchor_pos_b,
    params={"command_name": "motion"},
  )
  actor.terms["motion_anchor_ori_b"] = ObservationTermCfg(
    func=tracking_mdp.motion_anchor_ori_b,
    params={"command_name": "motion"},
  )
  actor.terms["body_pos"] = ObservationTermCfg(
    func=tracking_mdp.robot_body_pos_b,
    params={"command_name": "motion"},
  )
  actor.terms["body_ori"] = ObservationTermCfg(
    func=tracking_mdp.robot_body_ori_b,
    params={"command_name": "motion"},
  )
  actor.terms["base_lin_vel"] = ObservationTermCfg(
    func=tracking_mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
  )
  actor.terms["target_point_pos"] = ObservationTermCfg(
    func=soccer_mdp.constant_target_point_pos,
    params={"command_name": "motion"},
  )

  return cfg


def k1_flat_soccer_student_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Student policy with first-frame-only ball/destination observations."""
  cfg = k1_flat_soccer_kick_env_cfg(play=play)

  actor = cfg.observations["actor"]
  actor.terms["target_point_pos"] = ObservationTermCfg(
    func=soccer_mdp.target_point_pos_first_frame,
    params={"command_name": "motion"},
  )
  actor.terms["target_destination_pos_local"] = ObservationTermCfg(
    func=soccer_mdp.target_destination_pos_local_first_frame,
    params={"command_name": "motion"},
  )

  return cfg


def k1_flat_soccer_distill_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Distillation env (student obs = actor, teacher obs = critic)."""
  cfg = k1_flat_soccer_kick_env_cfg(play=play)

  actor = cfg.observations["actor"]
  actor.terms["target_point_pos"] = ObservationTermCfg(
    func=soccer_mdp.target_point_pos_first_frame,
    params={"command_name": "motion"},
  )
  actor.terms["target_destination_pos_local"] = ObservationTermCfg(
    func=soccer_mdp.target_destination_pos_local_first_frame,
    params={"command_name": "motion"},
  )

  critic = cfg.observations["critic"]
  critic.terms["body_pos"] = ObservationTermCfg(
    func=tracking_mdp.robot_body_pos_b,
    params={"command_name": "motion"},
  )
  critic.terms["body_ori"] = ObservationTermCfg(
    func=tracking_mdp.robot_body_ori_b,
    params={"command_name": "motion"},
  )
  critic.terms["base_lin_vel"] = ObservationTermCfg(
    func=tracking_mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
  )
  critic.terms["target_point_pos"] = ObservationTermCfg(
    func=soccer_mdp.constant_target_point_pos,
    params={"command_name": "motion"},
  )

  return cfg
