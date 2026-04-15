from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import LocomotionMemoryCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_command(env: ManagerBasedRlEnv, command_name: str) -> LocomotionMemoryCommand:
  return cast(LocomotionMemoryCommand, env.command_manager.get_term(command_name))


def proposal_vector(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _get_command(env, command_name).command


def proposal_features(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _get_command(env, command_name)
  has_snippet = (command.snippet_id >= 0).to(dtype=torch.float32).unsqueeze(-1)
  phase = command.phase.unsqueeze(-1)
  elapsed_time = command.elapsed_time.unsqueeze(-1)
  current_score = torch.where(
    torch.isfinite(command.current_score),
    command.current_score,
    torch.zeros_like(command.current_score),
  ).unsqueeze(-1)
  contact_valid = command.contact_valid.to(dtype=torch.float32).unsqueeze(-1)
  phase_valid = command.phase_valid.to(dtype=torch.float32).unsqueeze(-1)
  transition_active = command.transition_active.to(dtype=torch.float32).unsqueeze(-1)
  return torch.cat(
    (
      has_snippet,
      phase,
      elapsed_time,
      current_score,
      contact_valid,
      phase_valid,
      transition_active,
    ),
    dim=-1,
  )


def proposal_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _get_command(env, command_name)
  snippet_id = command.snippet_id.to(dtype=torch.float32).unsqueeze(-1)
  phase = command.phase.unsqueeze(-1)
  contact_valid = command.contact_valid.to(dtype=torch.float32).unsqueeze(-1)
  phase_valid = command.phase_valid.to(dtype=torch.float32).unsqueeze(-1)
  transition_active = command.transition_active.to(dtype=torch.float32).unsqueeze(-1)
  return torch.cat(
    (snippet_id, phase, contact_valid, phase_valid, transition_active), dim=-1
  )
