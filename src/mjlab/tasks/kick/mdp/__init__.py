"""Kick task MDP components."""

from mjlab.tasks.kick.mdp import (
  commands,
  delayed_action,
  observations,
  perception,
  rewards,
  terminations,
)
from mjlab.tasks.kick.mdp.commands import KickTargetCommand, KickTargetCommandCfg
from mjlab.tasks.kick.mdp.delayed_action import (
  DelayedJointPositionAction,
  DelayedJointPositionActionCfg,
)
from mjlab.tasks.kick.mdp.perception import VirtualPerception, VirtualPerceptionCfg

__all__ = [
  "DelayedJointPositionAction",
  "DelayedJointPositionActionCfg",
  "KickTargetCommand",
  "KickTargetCommandCfg",
  "VirtualPerception",
  "VirtualPerceptionCfg",
  "commands",
  "delayed_action",
  "observations",
  "perception",
  "rewards",
  "terminations",
]
