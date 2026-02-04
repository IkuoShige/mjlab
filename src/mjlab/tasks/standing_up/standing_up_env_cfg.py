"""Standing-up task configuration.

This module defines the base configuration for standing-up tasks.
Robot-specific configurations are located in the config/ directory.

Based on HoST (Humanoid Standing-up Control) from Isaac Gym.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.standing_up import mdp
from mjlab.tasks.standing_up.managers import (
  GaussianProductRewardManager,
  GroupedRewardTermCfg,
  UnactuatedMaskingObservationManager,
)
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_standing_up_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base standing-up task configuration.

  Returns:
    Base environment configuration that can be customized per robot.
  """

  ##
  # Observations
  ##

  # HoST observation: 43 dim × 6 history = 258 features.
  # Components: ang_vel(3), gravity(3), joint_pos(12), joint_vel(12), actions(12), action_scale(1).
  policy_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel_scaled,
      params={"scale": 0.25},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_scaled,
      params={"scale": 1.0},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_scaled,
      params={"scale": 0.05},
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "last_action": ObservationTermCfg(
      func=mdp.last_action,
    ),
    "action_rescale": ObservationTermCfg(
      func=mdp.action_rescale,
      params={"noise_scale": 0.05},
    ),
  }

  critic_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel_scaled,
      params={"scale": 0.25},
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_scaled,
      params={"scale": 1.0},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_scaled,
      params={"scale": 0.05},
    ),
    "last_action": ObservationTermCfg(
      func=mdp.last_action,
    ),
    "action_rescale": ObservationTermCfg(
      func=mdp.action_rescale,
      params={"noise_scale": 0.0},
    ),
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=6,
      flatten_history_dim=True,
      nan_policy="sanitize",  # Sanitize NaN/Inf values to prevent policy crash.
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=6,
      flatten_history_dim=True,
      nan_policy="sanitize",  # Sanitize NaN/Inf values to prevent policy crash.
    ),
  }

  ##
  # Actions
  ##

  # Note: HoST uses position offset action (target = current + action * scale).
  # This will be configured per-robot using a custom action type.
  actions: dict[str, ActionTermCfg] = {}  # Set per-robot.

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    # Initialize curriculum values at startup.
    "init_curriculum": EventTermCfg(
      func=mdp.init_curriculum_values,
      mode="startup",
      params={
        "initial_action_rescale": 1.0,
        "initial_force_magnitude": 12.0,
      },
    ),
    # Reset to random orientation from 4 directions.
    "reset_root_state_4dir": EventTermCfg(
      func=mdp.reset_root_state_4dir,
      mode="reset",
      params={
        "orientations": None,  # Use defaults (supine, prone, left, right).
        "weights": (0.25, 0.25, 0.25, 0.25),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # Apply pulling force to help standing.
    "apply_pulling_force": EventTermCfg(
      func=mdp.apply_pulling_force,
      mode="interval",
      interval_range_s=(0.0, 0.0),  # Every step.
      params={
        "force_magnitude": 12.0,
        "gravity_threshold": -0.8,
        "base_body_name": "base_link",
        "unactuated_steps": 30,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # Track head height every step for curriculum progression.
    "track_head_height": EventTermCfg(
      func=mdp.track_head_height,
      mode="interval",
      interval_range_s=(0.0, 0.0),  # Every step.
      params={
        "head_body_name": "keyframe_head_link",
        "feet_body_names": ("l_ankle_roll_link", "r_ankle_roll_link"),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # Update curriculum on reset (decrease force/action_scale when robot achieves target height).
    "update_curriculum": EventTermCfg(
      func=mdp.force_curriculum,
      mode="reset",
      params={
        "threshold_height": 0.37,
        "force_decrement": 20.0,
        "action_scale_decrement": 0.02,
        "min_force": 0.0,
        "min_action_scale": 0.25,
        "head_body_name": "keyframe_head_link",
        "feet_body_names": ("l_ankle_roll_link", "r_ankle_roll_link"),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
  }

  ##
  # Rewards
  ##

  # Note: These will use GaussianProductRewardManager with GroupedRewardTermCfg.
  # Configured per-robot in config/pi/env_cfgs.py.
  rewards: dict[str, GroupedRewardTermCfg] = {
    # Progress rewards (dense signal for early learning).
    "progress_upright": GroupedRewardTermCfg(
      func=mdp.upright_progress,
      weight=5.0,  # High weight for dense learning signal.
      group="style",  # Use additive aggregation.
    ),
    "progress_height": GroupedRewardTermCfg(
      func=mdp.base_height_progress,
      weight=5.0,  # High weight for dense learning signal.
      group="style",
      params={"target_height": 0.34},
    ),
    # Task rewards (product aggregation).
    "task_orientation": GroupedRewardTermCfg(
      func=mdp.orientation,
      weight=1.0,
      group="task",
      params={"threshold": 0.99, "phase1_height": 0.25},
    ),
    "task_head_height": GroupedRewardTermCfg(
      func=mdp.head_height,
      weight=1.0,
      group="task",
      params={
        "target_height": 0.37,
        "margin": 0.37,
        "head_body_name": "keyframe_head_link",
        "feet_body_names": ("l_ankle_roll_link", "r_ankle_roll_link"),
      },
    ),
    # Regularization rewards (additive).
    "regu_dof_acc": GroupedRewardTermCfg(
      func=mdp.dof_acc_l2,
      weight=-1e-8,  # Reduced from -2.5e-7 to allow more dynamic movement.
      group="regu",
    ),
    "regu_action_rate": GroupedRewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
      group="regu",
      params={"unactuated_steps": 30},
    ),
    "regu_smoothness": GroupedRewardTermCfg(
      func=mdp.smoothness,
      weight=-0.01,
      group="regu",
      params={"unactuated_steps": 30},
    ),
    "regu_torques": GroupedRewardTermCfg(
      func=mdp.torques_l2,
      weight=-2.5e-6,
      group="regu",
    ),
    "regu_joint_power": GroupedRewardTermCfg(
      func=mdp.joint_power,
      weight=-2.5e-5,
      group="regu",
    ),
    "regu_dof_vel": GroupedRewardTermCfg(
      func=mdp.dof_vel_l2,
      weight=-1e-3,
      group="regu",
    ),
    "regu_joint_tracking_error": GroupedRewardTermCfg(
      func=mdp.joint_tracking_error,
      weight=-0.00025,
      group="regu",
    ),
    "regu_dof_pos_limits": GroupedRewardTermCfg(
      func=mdp.dof_pos_limits,
      weight=-1.0,  # Reduced further to allow more exploration during early learning.
      group="regu",
    ),
    # Style rewards (additive).
    "style_hip_yaw_deviation": GroupedRewardTermCfg(
      func=mdp.hip_deviation,
      weight=-1.0,  # Reduced from -10.0.
      group="style",
      params={"joint_type": "yaw", "max_threshold": 1.4, "min_threshold": 0.9},
    ),
    "style_hip_roll_deviation": GroupedRewardTermCfg(
      func=mdp.hip_deviation,
      weight=-1.0,  # Reduced from -10.0 to allow exploration of different poses.
      group="style",
      params={"joint_type": "roll", "max_threshold": 1.4, "min_threshold": 0.9},
    ),
    "style_left_foot_displacement": GroupedRewardTermCfg(
      func=mdp.foot_displacement,
      weight=2.5,
      group="style",
      params={"side": "left", "sigma": -2.0, "clamp_min": 0.3},
    ),
    "style_right_foot_displacement": GroupedRewardTermCfg(
      func=mdp.foot_displacement,
      weight=2.5,
      group="style",
      params={"side": "right", "sigma": -2.0, "clamp_min": 0.3},
    ),
    "style_ground_parallel": GroupedRewardTermCfg(
      func=mdp.ground_parallel,
      weight=12.0,
      group="style",
      params={"decay_rate": 5.0},
    ),
    "style_feet_distance": GroupedRewardTermCfg(
      func=mdp.feet_distance,
      weight=-10.0,
      group="style",
      params={"threshold": 0.45},
    ),
    "style_feet_contact_balance": GroupedRewardTermCfg(
      func=mdp.feet_contact_balance,
      weight=5.0,
      group="style",
      params={"decay_rate": 5.0, "phase2_height": 0.25},
    ),
    "style_ankle_pitch_neutral": GroupedRewardTermCfg(
      func=mdp.ankle_pitch_neutral,
      weight=5.0,
      group="style",
      params={"target": -0.1, "decay_rate": 10.0},
    ),
    "style_soft_symmetry_action": GroupedRewardTermCfg(
      func=mdp.soft_symmetry_action,
      weight=-10.0,
      group="style",
    ),
    "style_soft_symmetry_body": GroupedRewardTermCfg(
      func=mdp.soft_symmetry_body,
      weight=2.5,
      group="style",
    ),
    # Target rewards (additive).
    "target_ang_vel_xy": GroupedRewardTermCfg(
      func=mdp.style_ang_vel_xy,
      weight=10.0,
      group="target",
      params={"decay_rate": 2.0, "phase1_height": 0.34},
    ),
    "target_lin_vel_xy": GroupedRewardTermCfg(
      func=mdp.lin_vel_xy_exp,
      weight=10.0,
      group="target",
      params={"decay_rate": 5.0, "phase3_height": 0.34},
    ),
    "target_feet_height_var": GroupedRewardTermCfg(
      func=mdp.feet_height_var,
      weight=2.5,
      group="target",
    ),
    "target_orientation": GroupedRewardTermCfg(
      func=mdp.target_orientation,
      weight=10.0,
      group="target",
      params={"decay_rate": 5.0, "phase3_height": 0.34},
    ),
    "target_base_height": GroupedRewardTermCfg(
      func=mdp.target_base_height,
      weight=10.0,
      group="target",
      params={"target_height": 0.34, "decay_rate": 20.0},
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "dof_vel_out": TerminationTermCfg(
      func=mdp.dof_vel_out,
      params={"limit": 300.0, "unactuated_steps": 30},
    ),
    "base_vel_out": TerminationTermCfg(
      func=mdp.base_vel_out,
      params={"limit": 20.0, "unactuated_steps": 30},
    ),
    "nan_in_physics": TerminationTermCfg(
      func=mdp.nan_in_physics,
    ),
  }

  ##
  # Assemble and return.
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainImporterCfg(terrain_type="plane"), num_envs=4096),
    observations=observations,
    observation_manager_cls=UnactuatedMaskingObservationManager,
    observation_manager_kwargs={"unactuated_steps": 30},
    actions=actions,
    commands={},  # No velocity commands for standing-up.
    events=events,
    rewards=rewards,  # type: ignore  # Uses GroupedRewardTermCfg.
    reward_manager_cls=GaussianProductRewardManager,  # Use Gaussian product for task rewards.
    reward_manager_kwargs={
      "group_names": ("task", "regu", "style", "target"),
      "group_weights": (2.5, 0.1, 1.0, 1.0),
    },
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=3.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=100,  # Per-env contact buffer (Pi robot needs ~50).
      njmax=300,  # Per-env constraint buffer (needs >= 268).
      mujoco=MujocoCfg(
        timestep=0.002,  # Reduced for stability.
        iterations=20,  # Increased for stability.
        ls_iterations=40,  # Increased for stability.
      ),
    ),
    decimation=10,  # Adjusted for timestep=0.002 to maintain 50Hz control (0.002*10=0.02s).
    episode_length_s=10.0,
  )
