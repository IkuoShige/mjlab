"""Soccer-specific termination conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.soccer.mdp.commands import SoccerMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_finished(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Terminate when the motion sequence has ended."""
  command: SoccerMotionCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  return command.time_steps >= (command.motion_length - 1).clamp(min=0)
