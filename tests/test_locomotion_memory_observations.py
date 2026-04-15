"""Unit tests for locomotion-memory observation helpers."""

from unittest.mock import Mock

import torch

from mjlab.tasks.locomotion_memory.mdp.observations import (
  proposal_features,
  proposal_state,
  proposal_vector,
)


def _make_env_and_command() -> tuple[Mock, Mock]:
  env = Mock()
  env.command_manager = Mock()

  command = Mock()
  command.command = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
  command.snippet_id = torch.tensor([5, -1])
  command.phase = torch.tensor([0.25, 0.75])
  command.elapsed_time = torch.tensor([0.1, 0.2])
  command.current_score = torch.tensor([1.5, float("-inf")])
  command.contact_valid = torch.tensor([True, False])
  command.phase_valid = torch.tensor([False, True])
  command.transition_active = torch.tensor([True, False])

  env.command_manager.get_term.return_value = command
  return env, command


def test_proposal_vector_returns_command_tensor() -> None:
  env, command = _make_env_and_command()
  assert torch.equal(proposal_vector(env, "memory"), command.command)


def test_proposal_features_encode_current_retrieval_context() -> None:
  env, _ = _make_env_and_command()
  features = proposal_features(env, "memory")

  expected = torch.tensor(
    [
      [1.0, 0.25, 0.1, 1.5, 1.0, 0.0, 1.0],
      [0.0, 0.75, 0.2, 0.0, 0.0, 1.0, 0.0],
    ]
  )
  assert torch.allclose(features, expected)


def test_proposal_state_exposes_snippet_and_validity_flags() -> None:
  env, _ = _make_env_and_command()
  state = proposal_state(env, "memory")

  expected = torch.tensor(
    [
      [5.0, 0.25, 1.0, 0.0, 1.0],
      [-1.0, 0.75, 0.0, 1.0, 0.0],
    ]
  )
  assert torch.allclose(state, expected)
