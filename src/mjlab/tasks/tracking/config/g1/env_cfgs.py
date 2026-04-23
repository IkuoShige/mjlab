"""Unitree G1 flat tracking environment configurations."""

import glob

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
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

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


# Directory of soccer NPZ clips for G1 multi-motion tracking (Stage 1).
_G1_MULTIMOTION_DIR = "motions/soccer-standard-mj"


def _discover_g1_motion_files(motion_dir: str = _G1_MULTIMOTION_DIR) -> list[str]:
  return sorted(glob.glob(f"{motion_dir}/*.npz"))


def unitree_g1_flat_multimotion_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  motion_files: list[str] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 multi-motion tracking configuration (HumanoidSoccer Stage 1).

  Loads all soccer kick clips and samples one per env per episode. No soccer
  ball / kick rewards — pure BeyondMimic-style motion tracking generalized
  across clips, intended as pre-training for Stage 2.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  single = cfg.commands["motion"]
  assert isinstance(single, MotionCommandCfg)

  files = motion_files if motion_files is not None else _discover_g1_motion_files()
  multi = MultiMotionCommandCfg(
    entity_name=single.entity_name,
    resampling_time_range=single.resampling_time_range,
    debug_vis=single.debug_vis,
    motion_files=files,
    anchor_body_name=single.anchor_body_name,
    body_names=single.body_names,
    pose_range=single.pose_range,
    velocity_range=single.velocity_range,
    joint_position_range=single.joint_position_range,
    sampling_mode=single.sampling_mode if play else "adaptive",
  )
  cfg.commands = {"motion": multi}

  return cfg
