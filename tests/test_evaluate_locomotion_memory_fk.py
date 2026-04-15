"""Tests for locomotion-memory FK evaluation helpers."""

import math

import pytest
import torch

from mjlab.scripts.evaluate_locomotion_memory_fk import (
  COMMAND_PRESETS,
  DEFAULT_COMMAND_SPECS,
  _mean_metric,
  concat_non_empty,
  parse_command_spec,
  parse_command_specs,
  quantile_tensor,
  resolve_command_specs,
  summarize_tensor,
)


def test_parse_command_spec_accepts_named_velocity_command() -> None:
  command = parse_command_spec("low_yaw_0.15:0.0,0.0,0.15")

  assert command.name == "low_yaw_0.15"
  assert command.velocity == (0.0, 0.0, 0.15)


def test_parse_command_spec_generates_name_for_bare_command() -> None:
  command = parse_command_spec("1.0,0.0,0.0")

  assert command.name == "1.0_0.0_0.0"
  assert command.velocity == (1.0, 0.0, 0.0)


def test_parse_command_specs_accepts_semicolon_delimited_commands() -> None:
  commands = parse_command_specs("forward:1.0,0.0,0.0;yaw:0.0,0.0,0.15")

  assert [command.name for command in commands] == ["forward", "yaw"]
  assert commands[1].velocity == (0.0, 0.0, 0.15)


def test_resolve_command_specs_uses_preset_when_commands_are_default() -> None:
  commands = resolve_command_specs(DEFAULT_COMMAND_SPECS, "yaw_sweep")

  assert [command.name for command in commands] == [
    "yaw_l_0.10",
    "yaw_r_0.10",
    "yaw_l_0.15",
    "yaw_r_0.15",
    "yaw_l_0.20",
    "yaw_r_0.20",
    "yaw_l_0.35",
    "yaw_r_0.35",
    "yaw_l_0.50",
    "yaw_r_0.50",
  ]


def test_resolve_command_specs_prefers_explicit_commands_over_preset() -> None:
  commands = resolve_command_specs("custom:0.2,0.0,0.1", "research")

  assert [command.name for command in commands] == ["custom"]
  assert commands[0].velocity == (0.2, 0.0, 0.1)


def test_research_command_preset_includes_forward_yaw_and_arc_commands() -> None:
  commands = parse_command_specs(COMMAND_PRESETS["research"])
  names = {command.name for command in commands}

  assert "forward_vx_1.0" in names
  assert "yaw_l_0.15" in names
  assert "yaw_r_0.15" in names
  assert "arc_vx1.0_yaw_l0.20" in names
  assert "arc_vx1.0_yaw_r0.20" in names


def test_parse_command_spec_rejects_wrong_dimension() -> None:
  with pytest.raises(ValueError, match="three"):
    parse_command_spec("bad:0.0,0.0")


def test_tensor_summaries_handle_empty_inputs() -> None:
  values = torch.empty(0)

  assert math.isnan(summarize_tensor(values))
  assert math.isnan(quantile_tensor(values, 0.5))


def test_concat_non_empty_skips_empty_tensors() -> None:
  values = concat_non_empty(
    [torch.empty(0), torch.tensor([1.0, 2.0]), torch.tensor([3.0])],
    device=torch.device("cpu"),
  )

  torch.testing.assert_close(values, torch.tensor([1.0, 2.0, 3.0]))


def test_mean_metric_ignores_nan_values() -> None:
  mean = _mean_metric(
    [
      {"scuff": float("nan")},
      {"scuff": 0.2},
      {"scuff": 0.4},
    ],
    "scuff",
  )

  assert mean == pytest.approx(0.3)
