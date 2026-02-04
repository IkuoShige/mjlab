"""MDP components for standing-up task."""

from mjlab.tasks.standing_up.mdp.actions import (
  JointPositionOffsetAction,
  JointPositionOffsetActionCfg,
)
from mjlab.tasks.standing_up.mdp.curriculums import (
  force_curriculum,
)
from mjlab.tasks.standing_up.mdp.events import (
  apply_pulling_force,
  init_curriculum_values,
  reset_joints_random,
  reset_root_state_4dir,
  track_head_height,
)
from mjlab.tasks.standing_up.mdp.observations import (
  action_rescale,
  base_ang_vel_scaled,
  joint_pos_scaled,
  joint_vel_scaled,
  last_action,
  projected_gravity,
)
from mjlab.tasks.standing_up.mdp.rewards import (
  action_rate_l2,
  ankle_pitch_neutral,
  base_height_progress,
  dof_acc_l2,
  dof_pos_limits,
  dof_vel_l2,
  feet_contact_balance,
  feet_distance,
  feet_height_var,
  foot_displacement,
  ground_parallel,
  head_height,
  hip_deviation,
  joint_power,
  joint_tracking_error,
  lin_vel_xy_exp,
  orientation,
  smoothness,
  soft_symmetry_action,
  soft_symmetry_body,
  style_ang_vel_xy,
  target_base_height,
  target_orientation,
  torques_l2,
  upright_progress,
)
from mjlab.tasks.standing_up.mdp.terminations import (
  base_vel_out,
  dof_vel_out,
  nan_in_physics,
  time_out,
)

__all__ = [
  # Actions
  "JointPositionOffsetAction",
  "JointPositionOffsetActionCfg",
  # Observations
  "base_ang_vel_scaled",
  "projected_gravity",
  "joint_pos_scaled",
  "joint_vel_scaled",
  "last_action",
  "action_rescale",
  # Rewards
  "upright_progress",
  "base_height_progress",
  "orientation",
  "head_height",
  "dof_acc_l2",
  "action_rate_l2",
  "smoothness",
  "torques_l2",
  "joint_power",
  "dof_vel_l2",
  "joint_tracking_error",
  "dof_pos_limits",
  "hip_deviation",
  "foot_displacement",
  "ground_parallel",
  "feet_distance",
  "feet_contact_balance",
  "ankle_pitch_neutral",
  "style_ang_vel_xy",
  "soft_symmetry_action",
  "soft_symmetry_body",
  "target_orientation",
  "target_base_height",
  "lin_vel_xy_exp",
  "feet_height_var",
  # Terminations
  "time_out",
  "dof_vel_out",
  "base_vel_out",
  "nan_in_physics",
  # Events
  "reset_root_state_4dir",
  "reset_joints_random",
  "apply_pulling_force",
  "init_curriculum_values",
  "track_head_height",
  # Curriculums
  "force_curriculum",
]
