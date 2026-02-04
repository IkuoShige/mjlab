"""Pi 12-DOF robot constants and configuration."""

from mjlab.asset_zoo.robots.pi_12dof.pi_constants import (
  PI_ARTICULATION,
  PI_DEFAULT_JOINT_POS,
  PI_JOINT_NAMES,
  PI_LEFT_LEG_JOINTS,
  PI_RIGHT_LEG_JOINTS,
  PI_STIFFNESS,
  PI_DAMPING,
  PI_ACTION_SCALE,
  get_pi_robot_cfg,
)

__all__ = [
  "PI_ARTICULATION",
  "PI_DEFAULT_JOINT_POS",
  "PI_JOINT_NAMES",
  "PI_LEFT_LEG_JOINTS",
  "PI_RIGHT_LEG_JOINTS",
  "PI_STIFFNESS",
  "PI_DAMPING",
  "PI_ACTION_SCALE",
  "get_pi_robot_cfg",
]
