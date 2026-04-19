"""Kick contact detection for soccer tasks.

Ported from HumanoidSoccer (Kong et al., 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.soccer.mdp.commands import SoccerMotionCommand


@dataclass
class KickContactEvent:
  """Results produced once per step by kick-contact detection."""

  new_contact: torch.Tensor
  kick_detected: torch.Tensor
  peak_force: torch.Tensor


@dataclass
class ContactFootInfo:
  """Resolved foot metadata for environments with an active kick contact."""

  env_ids: torch.Tensor
  body_indices: torch.Tensor
  sides: torch.Tensor
  expected: torch.Tensor


class KickContactTracker:
  """Shared kick-contact detection logic reusable across reward terms.

  All state tensors are stored on the owning ``SoccerMotionCommand``
  instance rather than on the env object to keep state management clean.
  """

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._env = env
    self._device = env.device
    self._num_envs = env.num_envs
    self._cache_valid = False
    self._cached_event: KickContactEvent | None = None
    self._foot_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    # Persistent state tensors.
    self.contact_awarded = torch.zeros(
      self._num_envs, dtype=torch.bool, device=self._device
    )
    self.kick_success = torch.zeros(
      self._num_envs, dtype=torch.bool, device=self._device
    )
    self.expected_kick_success = torch.zeros(
      self._num_envs, dtype=torch.bool, device=self._device
    )
    self.frozen_proximity_reward = torch.zeros(
      self._num_envs, dtype=torch.float32, device=self._device
    )

    # Reward window timers.
    self.dir_align_timer = torch.zeros(
      self._num_envs, dtype=torch.int32, device=self._device
    )
    self.speed_timer = torch.zeros(
      self._num_envs, dtype=torch.int32, device=self._device
    )
    self.z_speed_timer = torch.zeros(
      self._num_envs, dtype=torch.int32, device=self._device
    )
    self.z_speed_prev = torch.zeros(
      self._num_envs, dtype=torch.bool, device=self._device
    )

  def begin_step(self, command: SoccerMotionCommand) -> None:
    """Reset per-step cache and handle envs that just resampled."""
    self._cache_valid = False
    self._cached_event = None
    self._handle_resample(command)

  def detect(
    self,
    command: SoccerMotionCommand,
    ball_sensor_name: str,
    horizontal_force_threshold: float,
    min_foot_speed: float = 0.0,
  ) -> KickContactEvent:
    """Detect new kick contacts (cached per step).

    Args:
      min_foot_speed: Minimum foot speed (m/s) to count as a kick.
        Filters out slow approach-phase contacts where the foot
        incidentally touches the ball while walking.
    """
    if self._cache_valid and self._cached_event is not None:
      return self._cached_event

    sensor = self._env.scene.sensors.get(ball_sensor_name)
    if sensor is None:
      empty = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
      zero = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
      event = KickContactEvent(empty, empty, zero)
      self._cached_event = event
      self._cache_valid = True
      return event

    data = sensor.data
    # Use force_history if available, else instantaneous force.
    forces = data.force_history if data.force_history is not None else data.force
    if forces is None or forces.numel() == 0:
      empty = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
      zero = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
      event = KickContactEvent(empty, empty, zero)
      self._cached_event = event
      self._cache_valid = True
      return event

    # force_history shape: [B, N, H, 3], force shape: [B, N, 3]
    if forces.ndim == 4:
      force_norm = forces.norm(dim=-1).amax(dim=2)  # [B, N]
    elif forces.ndim == 3:
      force_norm = forces.norm(dim=-1)  # [B, N]
    else:
      force_norm = forces.norm(dim=-1, keepdim=True)

    peak_force = force_norm.amax(dim=-1) if force_norm.ndim > 1 else force_norm
    kick_detected = peak_force > horizontal_force_threshold

    # Foot velocity gate: reject contacts where both feet are moving slowly.
    if min_foot_speed > 0.0 and hasattr(command, "_foot_body_indices"):
      foot_vel = command.robot.data.body_link_lin_vel_w[:, command._foot_body_indices]
      max_foot_speed = foot_vel.norm(dim=-1).amax(dim=-1)  # [B]
      kick_detected = kick_detected & (max_foot_speed > min_foot_speed)

    new_contact = (~self.contact_awarded) & kick_detected
    if torch.any(new_contact):
      self.contact_awarded[new_contact] = True
      self.kick_success[new_contact] = True

    event = KickContactEvent(new_contact, kick_detected, peak_force)
    self._cached_event = event
    self._cache_valid = True
    return event

  def record_expected_success(
    self, mask: torch.Tensor, expected_mask: torch.Tensor
  ) -> None:
    """Store whether a detected kick matched the expected leg."""
    self.expected_kick_success[mask] = expected_mask[mask].to(torch.bool)

  def freeze_proximity(
    self, env_ids: torch.Tensor, reward_values: torch.Tensor
  ) -> None:
    """Freeze proximity reward at kick-contact moment."""
    self.frozen_proximity_reward[env_ids] = reward_values

  def resolve_contact_foot(
    self,
    command: SoccerMotionCommand,
    foot_cfg: SceneEntityCfg,
    mask: torch.Tensor,
  ) -> ContactFootInfo:
    """Determine which foot most likely produced the contact."""
    env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if env_ids.numel() == 0:
      empty = torch.zeros(0, dtype=torch.long, device=self._device)
      zeros_i8 = torch.zeros(0, dtype=torch.int8, device=self._device)
      return ContactFootInfo(empty, empty, zeros_i8, zeros_i8)

    body_indices, sides = self._get_foot_metadata(command, foot_cfg)
    robot = command.robot

    foot_pos = robot.data.body_link_pos_w[env_ids][:, body_indices]
    ball_pos = command.soccer_ball_pos[env_ids]
    env_origins = self._env.scene.env_origins
    ball_pos = ball_pos + env_origins[env_ids]

    diff = torch.norm(foot_pos - ball_pos.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(diff, dim=-1)
    selected_body = body_indices[closest_idx]
    hit_sides = sides[closest_idx]

    expected = command.kick_leg[env_ids].to(torch.int8).clamp(min=0)

    return ContactFootInfo(env_ids, selected_body, hit_sides, expected)

  def reset(self, env_ids: torch.Tensor) -> None:
    """Reset all state for the given environments."""
    self.contact_awarded[env_ids] = False
    self.kick_success[env_ids] = False
    self.expected_kick_success[env_ids] = False
    self.frozen_proximity_reward[env_ids] = 0.0
    self.dir_align_timer[env_ids] = 0
    self.speed_timer[env_ids] = 0
    self.z_speed_timer[env_ids] = 0
    self.z_speed_prev[env_ids] = False

  def _handle_resample(self, command: SoccerMotionCommand) -> None:
    resample_flags = command.motion_resampled
    if not torch.any(resample_flags):
      return

    # Log success rate before reset.
    eligible = resample_flags.clone()
    step_buf = self._env.episode_length_buf
    if step_buf is not None:
      eligible = eligible & (step_buf > 1)

    if eligible.any() and hasattr(command, "metrics"):
      if "kick_success_rate" not in command.metrics:
        command.metrics["kick_success_rate"] = torch.zeros(
          self._num_envs, device=self._device, dtype=torch.float32
        )
      if "expected_kick_success_rate" not in command.metrics:
        command.metrics["expected_kick_success_rate"] = torch.zeros(
          self._num_envs, device=self._device, dtype=torch.float32
        )
      command.metrics["kick_success_rate"].fill_(
        self.kick_success[eligible].float().mean().item()
      )
      command.metrics["expected_kick_success_rate"].fill_(
        self.expected_kick_success[eligible].float().mean().item()
      )

    self.reset(torch.where(resample_flags)[0])
    command.motion_resampled[resample_flags] = False

  def _get_foot_metadata(
    self,
    command: SoccerMotionCommand,
    foot_cfg: SceneEntityCfg,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self._foot_cache is not None:
      return self._foot_cache

    robot = self._env.scene[foot_cfg.name]
    indices = torch.as_tensor(
      robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self._device,
    )
    body_names = foot_cfg.body_names or ()
    sides = torch.tensor(
      [
        0 if "left" in name.lower() else 1 if "right" in name.lower() else -1
        for name in body_names
      ],
      dtype=torch.int8,
      device=self._device,
    )
    self._foot_cache = (indices, sides)
    return self._foot_cache
