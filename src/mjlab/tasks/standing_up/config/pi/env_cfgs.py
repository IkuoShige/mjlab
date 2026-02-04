"""Pi robot standing-up environment configuration."""

from mjlab.asset_zoo.robots.pi_12dof import PI_ACTION_SCALE, get_pi_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import events as mdp_events
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.standing_up import mdp
from mjlab.tasks.standing_up.standing_up_env_cfg import make_standing_up_env_cfg


def pi_standing_up_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Pi robot standing-up configuration.

  Args:
    play: If True, configure for playback/evaluation mode.

  Returns:
    Environment configuration for Pi standing-up task.
  """
  cfg = make_standing_up_env_cfg()

  # Set up robot entity.
  cfg.scene.entities = {"robot": get_pi_robot_cfg()}

  # Configure action (position offset).
  cfg.actions = {
    "joint_pos": mdp.JointPositionOffsetActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=PI_ACTION_SCALE,
      use_curriculum_scale=True,
    )
  }

  # Update events with domain randomization.
  cfg.events.update(
    {
      # Reset joints with randomization.
      "reset_joints_random": EventTermCfg(
        func=mdp.reset_joints_random,
        mode="reset",
        params={
          "position_scale_range": (0.9, 1.1),
          "position_offset_range": (-0.1, 0.1),
          "asset_cfg": SceneEntityCfg("robot"),
        },
      ),
      # Friction randomization.
      "randomize_friction": EventTermCfg(
        func=mdp_events.randomize_field,
        mode="startup",
        domain_randomization=True,
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
          "field": "geom_friction",
          "operation": "abs",
          "ranges": (0.1, 1.0),
          "shared_random": False,
        },
      ),
      # Base mass randomization.
      "randomize_base_mass": EventTermCfg(
        func=mdp_events.randomize_field,
        mode="startup",
        domain_randomization=True,
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
          "field": "body_mass",
          "operation": "scale",
          "ranges": (0.8, 1.2),
        },
      ),
      # COM displacement.
      "randomize_com": EventTermCfg(
        func=mdp_events.randomize_field,
        mode="startup",
        domain_randomization=True,
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
          "field": "body_ipos",
          "operation": "add",
          "ranges": {
            0: (-0.12, 0.12),  # x
            1: (-0.12, 0.12),  # y
            2: (-0.06, 0.06),  # z
          },
        },
      ),
      # PD gains randomization.
      "randomize_pd_gains": EventTermCfg(
        func=mdp_events.randomize_pd_gains,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", actuator_ids=slice(None)),
          "kp_range": (0.85, 1.15),
          "kd_range": (0.85, 1.15),
          "operation": "scale",
        },
      ),
    }
  )

  # Update robot-specific reward parameters.
  # Note: URDF body names have "_link" suffix.
  # Update task rewards with Pi-specific body names.
  if "task_head_height" in cfg.rewards:
    cfg.rewards["task_head_height"].params["head_body_name"] = "keyframe_head_link"
    cfg.rewards["task_head_height"].params["feet_body_names"] = (
      "l_ankle_roll_link",
      "r_ankle_roll_link",
    )

  # Update style rewards with Pi-specific parameters.
  for key in [
    "style_left_foot_displacement",
    "style_right_foot_displacement",
  ]:
    if key in cfg.rewards:
      cfg.rewards[key].params["foot_body_name"] = "l_ankle_pitch_link"
      cfg.rewards[key].params["phase3_height"] = 0.34

  if "style_ground_parallel" in cfg.rewards:
    cfg.rewards["style_ground_parallel"].params["left_foot_body"] = "l_ankle_pitch_link"
    cfg.rewards["style_ground_parallel"].params["right_foot_body"] = "r_ankle_pitch_link"
    cfg.rewards["style_ground_parallel"].params["phase3_height"] = 0.34

  if "style_feet_distance" in cfg.rewards:
    cfg.rewards["style_feet_distance"].params["left_foot_body"] = "l_ankle_pitch_link"
    cfg.rewards["style_feet_distance"].params["right_foot_body"] = "r_ankle_pitch_link"

  if "target_feet_height_var" in cfg.rewards:
    cfg.rewards["target_feet_height_var"].params["left_foot_body"] = "l_ankle_pitch_link"
    cfg.rewards["target_feet_height_var"].params["right_foot_body"] = "r_ankle_pitch_link"

  # Update target height parameters for Pi robot.
  pi_heights = {
    "phase1_height": 0.25,
    "phase2_height": 0.25,
    "phase3_height": 0.34,
    "target_height": 0.34,
    "target_head_height": 0.37,
  }

  # Apply height parameters to relevant rewards.
  height_param_mapping = {
    "task_orientation": ["phase1_height"],
    "style_feet_contact_balance": ["phase2_height"],
    "style_ankle_pitch_neutral": ["phase2_height"],
    "style_soft_symmetry_action": ["phase3_height"],
    "style_soft_symmetry_body": ["phase3_height"],
    "target_ang_vel_xy": ["phase1_height"],
    "target_lin_vel_xy": ["phase3_height"],
    "target_orientation": ["phase3_height"],
    "target_base_height": ["target_height", "phase3_height"],
  }

  for reward_name, param_names in height_param_mapping.items():
    if reward_name in cfg.rewards:
      for param_name in param_names:
        if param_name in pi_heights:
          cfg.rewards[reward_name].params[param_name] = pi_heights[param_name]

  # Update viewer.
  cfg.viewer.body_name = "base_link"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("apply_pulling_force", None)

    # Disable domain randomization events.
    for event_name in list(cfg.events.keys()):
      if "randomize" in event_name:
        cfg.events.pop(event_name, None)

  return cfg
