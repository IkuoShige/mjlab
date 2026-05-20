"""Kick task MDP components."""

from mjlab.tasks.kick.mdp import (
  commands,
  observations,
  perception,
  rewards,
  terminations,
)
from mjlab.tasks.kick.mdp.commands import KickTargetCommand, KickTargetCommandCfg
from mjlab.tasks.kick.mdp.perception import VirtualPerception, VirtualPerceptionCfg

__all__ = [
  "KickTargetCommand",
  "KickTargetCommandCfg",
  "VirtualPerception",
  "VirtualPerceptionCfg",
  "commands",
  "observations",
  "perception",
  "rewards",
  "terminations",
]
