"""Unitree G1 soccer environment configurations."""

import glob
import math

from mjlab.asset_zoo.objects.soccer_ball import (
  SOCCER_BALL_RADIUS,
  get_soccer_ball_cfg,
)
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
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

# Default motion directory (symlinked from HumanoidSoccer).
_DEFAULT_MOTION_DIR = "motions/soccer-standard-mj"


def _discover_motion_files(motion_dir: str = _DEFAULT_MOTION_DIR) -> list[str]:
  """Find all .npz motion files in the given directory."""
  files = sorted(glob.glob(f"{motion_dir}/*.npz"))
  return files


def g1_flat_soccer_kick_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain soccer kicking configuration."""
  cfg = make_tracking_env_cfg()

  # ------------------------------------------------------------------
  # Scene: robot + soccer ball + sensors
  # ------------------------------------------------------------------

  cfg.scene.entities = {
    "robot": get_g1_robot_cfg(),
    "soccer_ball": get_soccer_ball_cfg(),
  }

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  ball_contact_cfg = ContactSensorCfg(
    name="ball_contact",
    primary=ContactMatch(mode="body", pattern="ball_body", entity="soccer_ball"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
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
  joint_pos_action.scale = G1_ACTION_SCALE

  # ------------------------------------------------------------------
  # Commands: replace base motion command with soccer multi-motion
  # ------------------------------------------------------------------

  def _foot_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      body_names=(
        "left_ankle_roll_link",
        "right_ankle_roll_link",
      ),
    )

  def _waist_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      joint_names=(
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
      ),
    )

  G1_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.commands = {
    "motion": soccer_mdp.SoccerMotionCommandCfg(
      entity_name="robot",
      soccer_ball_entity_name="soccer_ball",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      anchor_body_name="torso_link",
      body_names=G1_BODY_NAMES,
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
        "radius": (-0.25, 0.25),
        "arc_angle": math.pi / 9,
        "height": SOCCER_BALL_RADIUS,
      },
      blind_distance_min_range=(0.2, 0.8),
      blind_distance_max_range=(1.8, 2.5),
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
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "right_hip_roll_link",
    "right_knee_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
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
    "motion_foot_pos": RewardTermCfg(
      func=soccer_mdp.motion_relative_foot_position_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.3,
        "foot_body_names": [
          "left_ankle_roll_link",
          "right_ankle_roll_link",
        ],
      },
    ),
    # Action penalties.
    "action_rate_l2": RewardTermCfg(func=tracking_mdp.action_rate_l2, weight=-1e-1),
    "waist_action_rate_l2": RewardTermCfg(
      func=soccer_mdp.waist_action_rate_l2_clip,
      weight=-2.5e-1,
      params={"waist_cfg": _waist_cfg()},
    ),
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
    # Soccer: proximity and foot distance.
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
    # Soccer: kick contact and speed.
    # min_foot_speed filters slow approach contacts (foot touching ball
    # while walking). Only fast swing contacts count as kicks.
    "target_point_contact": RewardTermCfg(
      func=soccer_mdp.target_point_contact,
      weight=50.0,
      params={
        "command_name": "motion",
        "ball_sensor_name": "ball_contact",
        "horizontal_force_threshold": 30,
        "min_foot_speed": 3.0,
        "foot_cfg": _foot_cfg(),
      },
    ),
    "sideways_kick": RewardTermCfg(
      func=soccer_mdp.sideways_kick,
      weight=50.0,
      params={
        "command_name": "motion",
        "ball_sensor_name": "ball_contact",
        "horizontal_force_threshold": 30,
        "min_foot_speed": 3.0,
        "foot_cfg": _foot_cfg(),
      },
    ),
    "ball_velocity_direction_alignment": RewardTermCfg(
      func=soccer_mdp.ball_velocity_direction_alignment,
      weight=30.0,
      params={
        "command_name": "motion",
        "std": 0.8,
        "velocity_threshold": 0.5,
        "ball_sensor_name": "ball_contact",
        "horizontal_force_threshold": 30,
        "min_foot_speed": 3.0,
        "foot_cfg": _foot_cfg(),
      },
    ),
    "ball_speed_reward": RewardTermCfg(
      func=soccer_mdp.ball_speed_reward,
      weight=10.0,
      params={
        "command_name": "motion",
        "std": 1.2,
        "velocity_threshold": 0.5,
        "ball_sensor_name": "ball_contact",
        "horizontal_force_threshold": 30,
        "min_foot_speed": 3.0,
        "foot_cfg": _foot_cfg(),
      },
    ),
    "ball_z_speed_penalty": RewardTermCfg(
      func=soccer_mdp.ball_z_speed_penalty_reward,
      weight=-0.0,
      params={
        "command_name": "motion",
        "std": 3,
        "velocity_threshold": 0.5,
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
          "left_ankle_roll_link",
          "right_ankle_roll_link",
          "left_wrist_yaw_link",
          "right_wrist_yaw_link",
        ),
      },
    ),
  }

  # ------------------------------------------------------------------
  # Events (domain randomization)
  # ------------------------------------------------------------------

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # ------------------------------------------------------------------
  # Simulation: increase contact buffer for ball
  # ------------------------------------------------------------------

  cfg.sim.nconmax = 50
  cfg.sim.njmax = 300

  # ------------------------------------------------------------------
  # Viewer
  # ------------------------------------------------------------------

  cfg.viewer.body_name = "torso_link"

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


def g1_flat_soccer_moving_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Soccer kick with moving ball (ball has initial velocity)."""
  cfg = g1_flat_soccer_kick_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, soccer_mdp.SoccerMotionCommandCfg)
  motion_cmd.enable_soccer_ball_init_vel = True
  motion_cmd.soccer_ball_init_lin_vel_range = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (0.0, 0.0),
  }
  return cfg


def g1_flat_soccer_teacher_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Teacher policy with privileged actor observations."""
  cfg = g1_flat_soccer_kick_env_cfg(play=play)

  # Add privileged observations to actor (teacher sees full state).
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
  # Use constant (non-blind) ball observation for teacher.
  actor.terms["target_point_pos"] = ObservationTermCfg(
    func=soccer_mdp.constant_target_point_pos,
    params={"command_name": "motion"},
  )

  return cfg


def g1_flat_soccer_student_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Student policy with first-frame-only observations."""
  cfg = g1_flat_soccer_kick_env_cfg(play=play)

  actor = cfg.observations["actor"]
  # Student sees first-frame ball/destination only (no continuous updates).
  actor.terms["target_point_pos"] = ObservationTermCfg(
    func=soccer_mdp.target_point_pos_first_frame,
    params={"command_name": "motion"},
  )
  actor.terms["target_destination_pos_local"] = ObservationTermCfg(
    func=soccer_mdp.target_destination_pos_local_first_frame,
    params={"command_name": "motion"},
  )

  return cfg


def g1_flat_soccer_distill_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Distillation env: actor=student obs (limited), critic=teacher obs (privileged).

  RSL-RL Distillation maps obs_groups: student->actor, teacher->critic.
  The teacher network gets the critic obs group (privileged), and the
  student network gets the actor obs group (limited first-frame).
  """
  cfg = g1_flat_soccer_kick_env_cfg(play=play)

  # Actor = student obs (first-frame ball/destination).
  actor = cfg.observations["actor"]
  actor.terms["target_point_pos"] = ObservationTermCfg(
    func=soccer_mdp.target_point_pos_first_frame,
    params={"command_name": "motion"},
  )
  actor.terms["target_destination_pos_local"] = ObservationTermCfg(
    func=soccer_mdp.target_destination_pos_local_first_frame,
    params={"command_name": "motion"},
  )

  # Critic = teacher obs (privileged, matching Teacher actor).
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
  # Teacher uses constant (non-blind) ball observation.
  critic.terms["target_point_pos"] = ObservationTermCfg(
    func=soccer_mdp.constant_target_point_pos,
    params={"command_name": "motion"},
  )

  return cfg
