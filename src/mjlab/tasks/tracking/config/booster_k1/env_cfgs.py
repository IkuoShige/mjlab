"""Booster K1 flat tracking environment configurations."""

import glob

from mjlab.asset_zoo.robots import (
  K1_ACTION_SCALE,
  get_k1_robot_cfg,
)
from mjlab.asset_zoo.robots.booster_k1 import K1_FOOT_GEOM_REGEX
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# K1 has no separate torso/waist body; Trunk is both the root and the
# upper-body attachment point. K1 has no wrist joints; the *_hand_link
# stands in for the wrist tip when computing relative body positions.
K1_TRACKING_BODY_NAMES: tuple[str, ...] = (
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


def booster_k1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_k1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = K1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "Trunk"
  motion_cmd.body_names = K1_TRACKING_BODY_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = K1_FOOT_GEOM_REGEX
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
  )

  cfg.viewer.body_name = "Trunk"

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


# Directory of retargeted K1 soccer NPZ clips, used for multi-motion tracking
# (HumanoidSoccer Stage 1 style). Generated via
# ../k1_retarget/scripts/soccer_npz_g1_to_k1.py.
_K1_MULTIMOTION_DIR = "motions/soccer-standard-mj-k1"


def _discover_k1_motion_files(motion_dir: str = _K1_MULTIMOTION_DIR) -> list[str]:
  return sorted(glob.glob(f"{motion_dir}/*.npz"))


def booster_k1_flat_multimotion_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  motion_files: list[str] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 multi-motion tracking configuration (HumanoidSoccer Stage 1).

  Loads all retargeted K1 kick clips and samples one per env per episode. No
  soccer ball / kick rewards — this is pure BeyondMimic-style motion tracking
  generalized across multiple clips, intended as pre-training for Stage 2.
  """
  cfg = booster_k1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  # Replace the single-motion command with a multi-motion one, preserving the
  # same sampling / randomization ranges.
  single = cfg.commands["motion"]
  assert isinstance(single, MotionCommandCfg)

  files = motion_files if motion_files is not None else _discover_k1_motion_files()
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
    sampling_mode=single.sampling_mode if play else "uniform",
  )
  cfg.commands = {"motion": multi}

  return cfg
