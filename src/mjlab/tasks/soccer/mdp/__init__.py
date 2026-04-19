"""Soccer MDP components."""

from mjlab.tasks.soccer.mdp.commands import (
  SoccerMotionCommand,
  SoccerMotionCommandCfg,
)
from mjlab.tasks.soccer.mdp.kick_detection import (
  ContactFootInfo,
  KickContactEvent,
  KickContactTracker,
)
from mjlab.tasks.soccer.mdp.observations import (
  blind_zone_target_point_pos,
  constant_target_point_pos,
  target_destination_pos_local,
  target_destination_pos_local_first_frame,
  target_point_pos_first_frame,
)
from mjlab.tasks.soccer.mdp.rewards import (
  ball_speed_reward,
  ball_velocity_direction_alignment,
  ball_z_speed_penalty_reward,
  foot_distance,
  motion_relative_foot_position_error_exp,
  pelvis_orientation,
  sideways_kick,
  target_point_contact,
  target_point_proximity,
  waist_action_rate_l2_clip,
)
from mjlab.tasks.soccer.mdp.terminations import motion_finished

__all__ = [
  "SoccerMotionCommand",
  "SoccerMotionCommandCfg",
  "KickContactEvent",
  "ContactFootInfo",
  "KickContactTracker",
  "constant_target_point_pos",
  "blind_zone_target_point_pos",
  "target_destination_pos_local",
  "target_point_pos_first_frame",
  "target_destination_pos_local_first_frame",
  "foot_distance",
  "target_point_proximity",
  "target_point_contact",
  "sideways_kick",
  "ball_velocity_direction_alignment",
  "ball_speed_reward",
  "ball_z_speed_penalty_reward",
  "pelvis_orientation",
  "waist_action_rate_l2_clip",
  "motion_relative_foot_position_error_exp",
  "motion_finished",
]
