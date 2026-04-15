from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import Mock

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class _FakeFolder:
  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    return False


class _FakeCheckbox:
  def __init__(self, initial_value: bool):
    self.value = initial_value


class _FakeSlider:
  def __init__(
    self,
    label: str,
    *,
    min: float,
    max: float,
    initial_value: float,
    step: float,
  ):
    assert max >= initial_value >= min
    self.label = label
    self.min = min
    self.max = max
    self.value = initial_value
    self.step = step
    self._on_update = None

  def on_update(self, callback):
    self._on_update = callback
    return callback

  def emit_update(self):
    assert self._on_update is not None
    self._on_update(None)


class _FakeButton:
  def __init__(self):
    self._on_click = None

  def on_click(self, callback):
    self._on_click = callback
    return callback


class _FakeGui:
  def __init__(self):
    self.sliders: list[_FakeSlider] = []

  def add_folder(self, _name: str):
    return _FakeFolder()

  def add_checkbox(self, _label: str, *, initial_value: bool):
    return _FakeCheckbox(initial_value)

  def add_slider(
    self,
    label: str,
    *,
    initial_value: float,
    step: float,
    min: float,
    max: float,
  ):
    slider = _FakeSlider(
      label,
      min=min,
      max=max,
      initial_value=initial_value,
      step=step,
    )
    self.sliders.append(slider)
    return slider

  def add_button(self, _label: str, *, icon):
    return _FakeButton()


@dataclass
class _FakeServer:
  gui: _FakeGui


def _make_command(
  *,
  lin_vel_x: tuple[float, float],
  lin_vel_y: tuple[float, float],
  ang_vel_z: tuple[float, float],
) -> UniformVelocityCommand:
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 1.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=lin_vel_x,
      lin_vel_y=lin_vel_y,
      ang_vel_z=ang_vel_z,
    ),
  )
  command = UniformVelocityCommand.__new__(UniformVelocityCommand)
  command.cfg = cfg
  command._joystick_enabled = None
  command._joystick_sliders = []
  command._joystick_get_env_idx = None
  return command


def _make_resample_command(
  *,
  num_envs: int,
  lin_vel_x: tuple[float, float],
  lin_vel_y: tuple[float, float],
  ang_vel_z: tuple[float, float],
  rel_turn_in_place_envs: float,
  turn_in_place_lin_vel_x_max: float,
  turn_in_place_lin_vel_y_max: float,
  turn_in_place_ang_vel_z_min: float,
) -> UniformVelocityCommand:
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 1.0),
    rel_turn_in_place_envs=rel_turn_in_place_envs,
    turn_in_place_lin_vel_x_max=turn_in_place_lin_vel_x_max,
    turn_in_place_lin_vel_y_max=turn_in_place_lin_vel_y_max,
    turn_in_place_ang_vel_z_min=turn_in_place_ang_vel_z_min,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=lin_vel_x,
      lin_vel_y=lin_vel_y,
      ang_vel_z=ang_vel_z,
    ),
  )
  command = UniformVelocityCommand.__new__(UniformVelocityCommand)
  command.cfg = cfg
  command._env = Mock(device="cpu", num_envs=num_envs)
  command.robot = cast(
    Any,
    type(
      "Robot",
      (),
      {"data": None, "write_root_state_to_sim": lambda *args, **kwargs: None},
    )(),
  )
  command.vel_command_b = torch.zeros(num_envs, 3)
  command.vel_command_w = torch.zeros(num_envs, 3)
  command.heading_target = torch.zeros(num_envs)
  command.heading_error = torch.zeros(num_envs)
  command.is_heading_env = torch.zeros(num_envs, dtype=torch.bool)
  command.is_standing_env = torch.zeros(num_envs, dtype=torch.bool)
  command.is_world_env = torch.zeros(num_envs, dtype=torch.bool)
  command.is_forward_env = torch.zeros(num_envs, dtype=torch.bool)
  command.is_turn_in_place_env = torch.zeros(num_envs, dtype=torch.bool)
  return command


def test_create_gui_supports_zero_width_axis_ranges():
  command = _make_command(
    lin_vel_x=(0.0, 1.6),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(-0.4, 0.4),
  )
  server = _FakeServer(gui=_FakeGui())

  command.create_gui("twist", cast(Any, server), get_env_idx=lambda: 0)

  max_y = next(
    slider for slider in server.gui.sliders if slider.label == "Max lin_vel_y"
  )
  y = next(slider for slider in server.gui.sliders if slider.label == "lin_vel_y")

  assert max_y.value == 0.0
  assert max_y.min == 0.0
  assert y.min == 0.0
  assert y.max == 0.0
  assert y.value == 0.0


def test_create_gui_preserves_forward_only_constraints_when_rescaled():
  command = _make_command(
    lin_vel_x=(0.0, 1.6),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(-0.4, 0.4),
  )
  server = _FakeServer(gui=_FakeGui())

  command.create_gui("twist", cast(Any, server), get_env_idx=lambda: 0)

  max_x = next(
    slider for slider in server.gui.sliders if slider.label == "Max lin_vel_x"
  )
  x = next(slider for slider in server.gui.sliders if slider.label == "lin_vel_x")
  yaw = next(slider for slider in server.gui.sliders if slider.label == "ang_vel_z")

  assert x.min == 0.0
  assert x.max == 1.6
  assert yaw.min == -0.4
  assert yaw.max == 0.4

  x.value = 1.4
  max_x.value = 0.8
  max_x.emit_update()

  assert x.min == 0.0
  assert x.max == 0.8
  assert x.value == 0.8


def test_resample_command_can_generate_turn_in_place_commands():
  torch.manual_seed(0)
  command = _make_resample_command(
    num_envs=32,
    lin_vel_x=(0.0, 1.0),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(-0.6, 0.6),
    rel_turn_in_place_envs=1.0,
    turn_in_place_lin_vel_x_max=0.12,
    turn_in_place_lin_vel_y_max=0.03,
    turn_in_place_ang_vel_z_min=0.25,
  )

  env_ids = torch.arange(command.num_envs)
  command._resample_command(env_ids)

  assert torch.all(command.is_turn_in_place_env)
  assert torch.all(command.vel_command_b[:, 0] >= 0.0)
  assert torch.all(command.vel_command_b[:, 0] <= 0.12 + 1e-6)
  assert torch.all(torch.abs(command.vel_command_b[:, 1]) <= 0.03 + 1e-6)
  assert torch.all(torch.abs(command.vel_command_b[:, 2]) >= 0.25 - 1e-6)
