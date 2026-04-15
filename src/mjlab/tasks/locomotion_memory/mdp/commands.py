from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.locomotion_memory.memory_db import (
  GAIT_NAME_TO_ID,
  LocomotionMemoryLoader,
  encode_proposals,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class _HasCommandTensor(Protocol):
  @property
  def command(self) -> torch.Tensor: ...


TURN_IN_PLACE_V2_PREFIX = "turn_in_place_v2_"
PIVOT_PREFIX = "pivot_"


def wrapped_phase_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Shortest absolute distance between two wrapped phases in ``[0, 1)``."""
  return torch.abs(torch.remainder(a - b + 0.5, 1.0) - 0.5)


def phase_boundary_mask(phase: torch.Tensor, *, tolerance: float) -> torch.Tensor:
  """Return True when phase is close to a gait boundary.

  The first version treats ``0.0``/``1.0`` and ``0.5`` as switch-friendly
  boundaries, which approximates support-change timing for the current
  left-right locomotion memory.
  """
  wrapped_phase = torch.remainder(phase, 1.0)
  dist_zero = torch.minimum(wrapped_phase, 1.0 - wrapped_phase)
  dist_half = torch.abs(wrapped_phase - 0.5)
  return torch.minimum(dist_zero, dist_half) <= tolerance


def cadence_continuity_score(
  current_cadence: torch.Tensor,
  candidate_cadence: torch.Tensor,
  *,
  cadence_scale: float,
) -> torch.Tensor:
  """Prefer candidates that preserve the current step frequency."""
  if cadence_scale <= 0.0:
    return torch.zeros_like(candidate_cadence)
  valid = (current_cadence > 1e-6) & (candidate_cadence > 1e-6)
  score = torch.exp(
    -torch.square((candidate_cadence - current_cadence) / cadence_scale)
  )
  return torch.where(valid, score, torch.zeros_like(score))


def snippet_source_prefix_mask(
  source_clip_names: tuple[str, ...],
  source_clip_id: torch.Tensor,
  *,
  prefix: str,
) -> torch.Tensor:
  """Map source-clip prefixes onto snippet ids."""
  if not source_clip_names:
    return torch.zeros_like(source_clip_id, dtype=torch.bool)
  clip_mask = torch.tensor(
    [name.startswith(prefix) for name in source_clip_names],
    dtype=torch.bool,
    device=source_clip_id.device,
  )
  return clip_mask[source_clip_id.long()]


def prioritize_prefixed_candidates(
  task_score: torch.Tensor,
  *,
  turn_intent: torch.Tensor,
  preferred_mask: torch.Tensor,
) -> torch.Tensor:
  """Restrict low-speed yaw retrieval to a preferred source subset when available."""
  if not torch.any(turn_intent) or not torch.any(preferred_mask):
    return task_score

  prioritized = task_score.clone()
  prioritized[turn_intent] = torch.where(
    preferred_mask[None, :],
    prioritized[turn_intent],
    torch.full_like(prioritized[turn_intent], -torch.inf),
  )
  return prioritized


def turn_intent_mask(
  twist_cmd: torch.Tensor,
  *,
  min_turn_yaw_command: float,
  max_turn_forward_speed: float,
  max_turn_lateral_speed: float,
) -> torch.Tensor:
  """Mask low-speed yaw commands that should behave like stepping turns."""
  abs_cmd_vx = torch.abs(twist_cmd[:, 0])
  abs_cmd_vy = torch.abs(twist_cmd[:, 1])
  abs_cmd_yaw = torch.abs(twist_cmd[:, 2])
  return (
    (abs_cmd_yaw >= min_turn_yaw_command)
    & (abs_cmd_vx <= max_turn_forward_speed)
    & (abs_cmd_vy <= max_turn_lateral_speed)
  )


def straight_intent_mask(
  twist_cmd: torch.Tensor,
  *,
  min_forward_speed: float,
  max_lateral_speed: float,
  max_abs_yaw_command: float,
) -> torch.Tensor:
  """Mask forward commands where the retrieved snippet should be straight."""
  return (
    (twist_cmd[:, 0] >= min_forward_speed)
    & (torch.abs(twist_cmd[:, 1]) <= max_lateral_speed)
    & (torch.abs(twist_cmd[:, 2]) <= max_abs_yaw_command)
  )


def reference_swing_lift_score(
  local_foot_pos_seq: torch.Tensor,
  contact_seq: torch.Tensor,
  *,
  min_swing_lift: float,
) -> torch.Tensor:
  """Score snippets by how often swing feet are lifted above their local support floor."""
  if min_swing_lift <= 0.0:
    return torch.ones(
      local_foot_pos_seq.shape[:-3],
      dtype=local_foot_pos_seq.dtype,
      device=local_foot_pos_seq.device,
    )

  foot_z = local_foot_pos_seq[..., 2]
  local_floor = torch.amin(foot_z, dim=-2, keepdim=True)
  swing_lift = foot_z - local_floor
  swing = (~contact_seq.bool()).float()
  swing_count = torch.sum(swing, dim=-2)
  low_lift_fraction = torch.sum((swing_lift < min_swing_lift).float() * swing, dim=-2)
  low_lift_fraction = low_lift_fraction / torch.clamp(swing_count, min=1.0)
  per_foot_score = 1.0 - low_lift_fraction
  valid_foot = swing_count > 0.0
  valid_count = torch.sum(valid_foot.float(), dim=-1)
  return torch.sum(per_foot_score * valid_foot.float(), dim=-1) / torch.clamp(
    valid_count, min=1.0
  )


def straight_reference_quality_score(
  twist_cmd: torch.Tensor,
  mean_vx: torch.Tensor,
  mean_vy: torch.Tensor,
  mean_yaw_rate: torch.Tensor,
  local_foot_pos_seq: torch.Tensor,
  contact_seq: torch.Tensor,
  *,
  min_forward_speed: float,
  max_lateral_speed: float,
  max_abs_yaw_command: float,
  vx_scale: float,
  yaw_rate_scale: float,
  min_swing_lift: float,
  lift_weight: float,
  yaw_weight: float,
) -> torch.Tensor:
  """Prefer straight high-speed snippets with low yaw drift and lifted swing feet."""
  straight = straight_intent_mask(
    twist_cmd,
    min_forward_speed=min_forward_speed,
    max_lateral_speed=max_lateral_speed,
    max_abs_yaw_command=max_abs_yaw_command,
  )[:, None]
  if not torch.any(straight):
    return torch.zeros_like(mean_yaw_rate)

  lift_score = reference_swing_lift_score(
    local_foot_pos_seq,
    contact_seq,
    min_swing_lift=min_swing_lift,
  )
  vx_score = torch.exp(
    -torch.square((mean_vx - twist_cmd[:, 0:1]) / max(vx_scale, 1e-6))
  )
  yaw_score = torch.exp(-torch.square(mean_yaw_rate / max(yaw_rate_scale, 1e-6)))
  lateral_score = torch.exp(-torch.square(mean_vy / max(max_lateral_speed, 1e-6)))
  total_weight = max(lift_weight + yaw_weight, 1e-6)
  score = (lift_weight * lift_score + yaw_weight * yaw_score) / total_weight
  score = 0.85 * score + 0.15 * lateral_score
  return score * vx_score * straight.float()


def prioritize_turn_in_place_candidates(
  task_score: torch.Tensor,
  *,
  turn_in_place_intent: torch.Tensor,
  turn_in_place_v2_mask: torch.Tensor,
) -> torch.Tensor:
  """Restrict low-speed yaw retrieval to in-place turn snippets when available."""
  if not torch.any(turn_in_place_intent) or not torch.any(turn_in_place_v2_mask):
    return task_score

  prioritized = task_score.clone()
  prioritized[turn_in_place_intent] = torch.where(
    turn_in_place_v2_mask[None, :],
    prioritized[turn_in_place_intent],
    torch.full_like(prioritized[turn_in_place_intent], -torch.inf),
  )
  return prioritized


def pivot_bias_score(
  twist_cmd: torch.Tensor,
  is_pivot: torch.Tensor,
  *,
  min_turn_yaw_command: float,
  max_turn_forward_speed: float,
  max_turn_lateral_speed: float,
  nonmatch_penalty: float,
) -> torch.Tensor:
  """Prefer pivot snippets for the current low-speed turn command range."""
  turn_intent = turn_intent_mask(
    twist_cmd,
    min_turn_yaw_command=min_turn_yaw_command,
    max_turn_forward_speed=max_turn_forward_speed,
    max_turn_lateral_speed=max_turn_lateral_speed,
  )[:, None]
  if not torch.any(turn_intent):
    return torch.zeros_like(is_pivot, dtype=torch.float32)

  preferred = torch.ones_like(is_pivot, dtype=torch.float32)
  fallback = torch.full_like(preferred, -nonmatch_penalty)
  bias = torch.where(is_pivot, preferred, fallback)
  return bias * turn_intent.float()


def cadence_bias_mask(
  *,
  twist_cmd: torch.Tensor,
  current_gait_id: torch.Tensor,
  candidate_gait_id: torch.Tensor,
  turn_gait_id: int,
  min_turn_yaw_command: float,
  max_turn_forward_speed: float,
  max_turn_lateral_speed: float,
) -> torch.Tensor:
  """Apply cadence continuity only for straight same-gait locomotion."""
  turn_mask = turn_intent_mask(
    twist_cmd,
    min_turn_yaw_command=min_turn_yaw_command,
    max_turn_forward_speed=max_turn_forward_speed,
    max_turn_lateral_speed=max_turn_lateral_speed,
  )
  same_gait = current_gait_id == candidate_gait_id
  non_turn = (current_gait_id != turn_gait_id) & (candidate_gait_id != turn_gait_id)
  return (~turn_mask[:, None]) & same_gait & non_turn


def turn_intent_bias_score(
  twist_cmd: torch.Tensor,
  gait_id: torch.Tensor,
  mean_vx: torch.Tensor,
  mean_vy: torch.Tensor,
  mean_yaw_rate: torch.Tensor,
  cadence: torch.Tensor,
  *,
  turn_gait_id: int,
  min_turn_yaw_command: float,
  max_turn_forward_speed: float,
  max_turn_lateral_speed: float,
  in_place_speed_scale: float,
  yaw_rate_scale: float,
  min_turn_cadence: float,
) -> torch.Tensor:
  """Bias retrieval toward stepping turn snippets for low-speed yaw commands."""
  abs_cmd_yaw = torch.abs(twist_cmd[:, 2:3])
  turn_intent = turn_intent_mask(
    twist_cmd,
    min_turn_yaw_command=min_turn_yaw_command,
    max_turn_forward_speed=max_turn_forward_speed,
    max_turn_lateral_speed=max_turn_lateral_speed,
  )[:, None]

  if not torch.any(turn_intent):
    return torch.zeros_like(mean_vx)

  is_turn = gait_id == turn_gait_id
  in_place_score = torch.exp(-torch.square(torch.abs(mean_vx) / in_place_speed_scale))
  lateral_score = torch.exp(-torch.square(torch.abs(mean_vy) / max_turn_lateral_speed))
  yaw_alignment = torch.exp(
    -torch.square((torch.abs(mean_yaw_rate) - abs_cmd_yaw) / yaw_rate_scale)
  )
  cadence_score = torch.clamp(cadence / min_turn_cadence, min=0.0, max=1.0)
  low_cadence = cadence < (0.75 * min_turn_cadence)

  cmd_yaw_sign = torch.sign(twist_cmd[:, 2:3])
  yaw_sign_match = (torch.sign(mean_yaw_rate) == cmd_yaw_sign) | (abs_cmd_yaw < 1e-6)

  turn_score = 0.4 * in_place_score + 0.15 * lateral_score + 0.3 * yaw_alignment
  turn_score = turn_score + 0.2 * cadence_score
  turn_score = torch.where(low_cadence, turn_score - 0.4, turn_score)
  turn_score = torch.where(yaw_sign_match, turn_score, turn_score - 1.0)

  non_turn_penalty = torch.full_like(turn_score, -0.4)
  bias = torch.where(is_turn, turn_score, non_turn_penalty)
  return bias * turn_intent.float()


def turn_in_place_v2_bias_score(
  twist_cmd: torch.Tensor,
  is_turn_in_place_v2: torch.Tensor,
  *,
  min_turn_yaw_command: float,
  max_turn_forward_speed: float,
  max_turn_lateral_speed: float,
  nonmatch_penalty: float,
) -> torch.Tensor:
  """Prefer turn_in_place_v2 snippets for low-speed yaw commands."""
  turn_in_place_intent = turn_intent_mask(
    twist_cmd,
    min_turn_yaw_command=min_turn_yaw_command,
    max_turn_forward_speed=max_turn_forward_speed,
    max_turn_lateral_speed=max_turn_lateral_speed,
  )[:, None]
  if not torch.any(turn_in_place_intent):
    return torch.zeros_like(is_turn_in_place_v2, dtype=torch.float32)

  preferred = torch.ones_like(is_turn_in_place_v2, dtype=torch.float32)
  fallback = torch.full_like(preferred, -nonmatch_penalty)
  bias = torch.where(is_turn_in_place_v2, preferred, fallback)
  return bias * turn_in_place_intent.float()


def switch_allowed_mask(
  elapsed_time: torch.Tensor,
  active_snippet: torch.Tensor,
  *,
  min_snippet_hold_s: float,
) -> torch.Tensor:
  """Allow retrieval switches only after the current snippet has been held long enough."""
  if min_snippet_hold_s <= 0.0:
    return torch.ones_like(active_snippet, dtype=torch.bool)
  return (~active_snippet) | (elapsed_time >= min_snippet_hold_s)


def event_synchronized_switch_mask(
  *,
  active_snippet: torch.Tensor,
  elapsed_time: torch.Tensor,
  phase: torch.Tensor,
  touchdown_event: torch.Tensor,
  min_snippet_hold_s: float,
  max_snippet_hold_s: float,
  phase_boundary_tolerance: float,
) -> torch.Tensor:
  """Allow retrieval switches near support/phase events, with stale fallback."""
  if not torch.any(active_snippet):
    return torch.ones_like(active_snippet, dtype=torch.bool)

  hold_ready = switch_allowed_mask(
    elapsed_time,
    active_snippet,
    min_snippet_hold_s=min_snippet_hold_s,
  )
  boundary_ready = phase_boundary_mask(phase, tolerance=phase_boundary_tolerance)
  event_ready = touchdown_event | boundary_ready
  if max_snippet_hold_s > 0.0:
    force_ready = elapsed_time >= max_snippet_hold_s
  else:
    force_ready = torch.zeros_like(active_snippet, dtype=torch.bool)
  return (~active_snippet) | (hold_ready & (event_ready | force_ready))


class LocomotionMemoryCommand(CommandTerm):
  """Proposal-command scaffold for future locomotion-memory retrieval."""

  cfg: LocomotionMemoryCommandCfg

  def __init__(self, cfg: LocomotionMemoryCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.proposal = torch.zeros(self.num_envs, cfg.proposal_dim, device=self.device)
    self.snippet_id = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self.snippet_frame = torch.zeros(
      self.num_envs,
      dtype=torch.long,
      device=self.device,
    )
    self.elapsed_time = torch.zeros(self.num_envs, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.contact_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self.phase_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self.reference_contact = torch.zeros(
      self.num_envs, 2, dtype=torch.bool, device=self.device
    )
    self.transition_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.turn_snippet_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.turn_in_place_v2_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.retrieval_switched = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.phase_jump = torch.zeros(self.num_envs, device=self.device)
    self.cadence_jump = torch.zeros(self.num_envs, device=self.device)
    self.reference_joint_pos = torch.zeros(
      self.num_envs,
      self.robot.num_joints,
      device=self.device,
    )
    self.reference_joint_vel = torch.zeros_like(self.reference_joint_pos)
    self.reference_foot_pos_b = torch.zeros(
      self.num_envs,
      2,
      3,
      device=self.device,
    )
    self.reference_foot_pos_valid = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.current_score = torch.full(
      (self.num_envs,),
      -torch.inf,
      device=self.device,
    )
    self._pending_snippet_id = torch.full_like(self.snippet_id, -1)
    self._pending_switch_count = torch.zeros(
      self.num_envs,
      dtype=torch.long,
      device=self.device,
    )
    self._pivot_snippet_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
    self._turn_in_place_v2_snippet_mask = torch.zeros(
      0, dtype=torch.bool, device=self.device
    )
    self._memory: LocomotionMemoryLoader | None = None
    if not cfg.memory_file:
      if cfg.require_memory_file:
        raise ValueError(
          "Locomotion-memory training requires env.commands.memory.memory_file. "
          "Build a database with `prepare-k1-locomotion-memory` or pass "
          "`--env.commands.memory.memory-file /path/to/memory.npz`."
        )
    else:
      memory_path = Path(cfg.memory_file)
      if not memory_path.is_file():
        raise FileNotFoundError(
          f"Locomotion memory file does not exist: {memory_path}. "
          "Build it with `prepare-k1-locomotion-memory` or override "
          "`--env.commands.memory.memory-file`."
        )
      self._memory = LocomotionMemoryLoader(str(memory_path), device=self.device)
      if self._memory.num_snippets <= 0:
        raise ValueError(
          f"Locomotion memory file {memory_path} does not contain any snippets."
        )
      if self._memory.proposal_dim != cfg.proposal_dim:
        raise ValueError(
          f"Locomotion memory proposal_dim={self._memory.proposal_dim} does not match "
          f"command cfg proposal_dim={cfg.proposal_dim}."
        )
      robot_joint_names = tuple(self.robot.joint_names)
      if self._memory.joint_names != robot_joint_names:
        raise ValueError(
          "Locomotion memory joint ordering does not match the active robot. "
          f"Memory has {self._memory.joint_names}, robot expects {robot_joint_names}."
        )
      self._pivot_snippet_mask = snippet_source_prefix_mask(
        self._memory.source_clip_names,
        self._memory.source_clip_id.long(),
        prefix=cfg.pivot_source_prefix,
      )
      self._turn_in_place_v2_snippet_mask = snippet_source_prefix_mask(
        self._memory.source_clip_names,
        self._memory.source_clip_id.long(),
        prefix=cfg.turn_in_place_v2_source_prefix,
      )

    self.metrics["retrieval_switches"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["contact_valid_rate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["phase_valid_rate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["transition_snippet_rate"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["turn_snippet_rate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["pivot_snippet_rate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["turn_in_place_v2_snippet_rate"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["phase_jump"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["cadence_jump"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.proposal

  @property
  def metadata(self) -> torch.Tensor:
    feature_dim = 14 + len(GAIT_NAME_TO_ID)
    metadata = torch.zeros(self.num_envs, feature_dim, device=self.device)
    metadata[:, 0] = torch.sin(2.0 * torch.pi * self.phase)
    metadata[:, 1] = torch.cos(2.0 * torch.pi * self.phase)
    active = self.snippet_id >= 0
    metadata[:, 2] = active.float()
    metadata[:, 3] = self.contact_valid.float()
    metadata[:, 4] = self.phase_valid.float()
    if self._memory is None or not torch.any(active):
      return metadata

    memory = self._get_memory()
    active_ids = self.snippet_id[active]
    active_frames = self.snippet_frame[active]
    metadata[active, 5] = memory.contact_seq[active_ids, active_frames, 0].float()
    metadata[active, 6] = memory.contact_seq[active_ids, active_frames, 1].float()
    metadata[active, 7] = memory.mean_vx[active_ids]
    metadata[active, 8] = memory.mean_vy[active_ids]
    metadata[active, 9] = memory.mean_yaw_rate[active_ids]
    metadata[active, 10] = memory.cadence[active_ids]
    metadata[active, 11] = memory.duty_factor[active_ids]
    metadata[active, 12] = memory.quality_score[active_ids]
    metadata[active, 13] = memory.transition_flag[active_ids].float()

    gait_ids = memory.gait_id[active_ids].long()
    gait_bins = torch.arange(len(GAIT_NAME_TO_ID), device=self.device)
    metadata[active, 14:] = (gait_ids[:, None] == gait_bins[None, :]).float()
    return metadata

  def _update_metrics(self) -> None:
    self.retrieval_switched.zero_()
    self.phase_jump.zero_()
    self.cadence_jump.zero_()
    self.metrics["contact_valid_rate"] += self.contact_valid.float()
    self.metrics["phase_valid_rate"] += self.phase_valid.float()
    if self._memory is not None:
      memory = self._memory
      active = self.snippet_id >= 0
      transition = torch.zeros(self.num_envs, device=self.device)
      turn = torch.zeros(self.num_envs, device=self.device)
      pivot = torch.zeros(self.num_envs, device=self.device)
      turn_in_place_v2 = torch.zeros(self.num_envs, device=self.device)
      transition[active] = memory.transition_flag[self.snippet_id[active]].float()
      turn[active] = (
        memory.gait_id[self.snippet_id[active]] == GAIT_NAME_TO_ID["turn"]
      ).float()
      pivot[active] = self._pivot_snippet_mask[self.snippet_id[active]].float()
      turn_in_place_v2[active] = self._turn_in_place_v2_snippet_mask[
        self.snippet_id[active]
      ].float()
      self.metrics["transition_snippet_rate"] += transition
      self.metrics["turn_snippet_rate"] += turn
      self.metrics["pivot_snippet_rate"] += pivot
      self.metrics["turn_in_place_v2_snippet_rate"] += turn_in_place_v2

  def compute(self, dt: float) -> None:
    self._update_metrics()
    self.time_left -= dt
    resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
    if len(resample_env_ids) > 0:
      allowed_env_ids, blocked_env_ids = self._split_resample_env_ids(resample_env_ids)
      if len(blocked_env_ids) > 0:
        self.time_left[blocked_env_ids] = self.cfg.blocked_resample_retry_s
      if len(allowed_env_ids) > 0:
        self._resample(allowed_env_ids)
    self._update_command()

  def _split_resample_env_ids(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self._memory is None or self._memory.num_snippets == 0:
      return env_ids, env_ids.new_empty((0,), dtype=torch.long)
    if not self.cfg.event_gated_switching:
      return env_ids, env_ids.new_empty((0,), dtype=torch.long)

    active_snippet = self.snippet_id[env_ids] >= 0
    if not torch.any(active_snippet):
      return env_ids, env_ids.new_empty((0,), dtype=torch.long)

    sensor = self._env.scene[self.cfg.contact_sensor_name]
    assert isinstance(sensor, ContactSensor)
    touchdown_event = sensor.compute_first_contact(dt=self._env.step_dt).any(dim=1)[
      env_ids
    ]
    switch_gate = event_synchronized_switch_mask(
      active_snippet=active_snippet,
      elapsed_time=self.elapsed_time[env_ids],
      phase=self.phase[env_ids],
      touchdown_event=touchdown_event,
      min_snippet_hold_s=self.cfg.min_snippet_hold_s,
      max_snippet_hold_s=self.cfg.max_snippet_hold_s,
      phase_boundary_tolerance=self.cfg.phase_boundary_tolerance,
    )
    return env_ids[switch_gate], env_ids[~switch_gate]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self._memory is None or self._memory.num_snippets == 0:
      self.proposal[env_ids] = 0.0
      self.snippet_id[env_ids] = -1
      self.snippet_frame[env_ids] = 0
      self.elapsed_time[env_ids] = 0.0
      self.phase[env_ids] = 0.0
      self.contact_valid[env_ids] = True
      self.phase_valid[env_ids] = True
      self.reference_contact[env_ids] = False
      self.reference_foot_pos_b[env_ids] = 0.0
      self.reference_foot_pos_valid[env_ids] = False
      self.transition_active[env_ids] = False
      self.turn_snippet_active[env_ids] = False
      self.turn_in_place_v2_active[env_ids] = False
      self.retrieval_switched[env_ids] = False
      self.current_score[env_ids] = -torch.inf
      return

    memory = self._get_memory()
    scores, candidate_ids, contact_ok, phase_ok = self._score_candidates(env_ids)
    candidate_scores, candidate_offsets = torch.max(scores, dim=1)
    best_ids = candidate_ids[
      torch.arange(len(env_ids), device=self.device), candidate_offsets
    ]
    current_valid = self.snippet_id[env_ids] >= 0
    current_scores = torch.full_like(candidate_scores, -torch.inf)
    if torch.any(current_valid):
      current_ids = self.snippet_id[env_ids][current_valid]
      current_scores[current_valid] = self._score_specific(
        env_ids[current_valid], current_ids
      )

    switch_allowed = switch_allowed_mask(
      self.elapsed_time[env_ids],
      current_valid,
      min_snippet_hold_s=self.cfg.min_snippet_hold_s,
    )
    switch_mask = ~current_valid
    better_mask = switch_allowed & (best_ids != self.snippet_id[env_ids])
    better_mask &= candidate_scores > current_scores + self.cfg.hysteresis_margin
    if torch.any(better_mask):
      same_pending = (
        self._pending_snippet_id[env_ids][better_mask] == best_ids[better_mask]
      )
      pending_counts = self._pending_switch_count[env_ids][better_mask]
      pending_counts = torch.where(
        same_pending, pending_counts + 1, torch.ones_like(pending_counts)
      )
      self._pending_switch_count[env_ids[better_mask]] = pending_counts
      self._pending_snippet_id[env_ids[better_mask]] = best_ids[better_mask]
      switch_mask[better_mask] = pending_counts >= self.cfg.hysteresis_steps
    reset_mask = ~better_mask
    self._pending_switch_count[env_ids[reset_mask]] = 0
    self._pending_snippet_id[env_ids[reset_mask]] = -1

    switch_env_ids = env_ids[switch_mask]
    if switch_env_ids.numel() > 0:
      switched_ids = best_ids[switch_mask]
      previously_active = self.snippet_id[switch_env_ids] >= 0
      switch_phase_jump = torch.zeros(
        switch_env_ids.shape[0], dtype=torch.float32, device=self.device
      )
      switch_cadence_jump = torch.zeros_like(switch_phase_jump)
      if torch.any(previously_active):
        prev_ids = self.snippet_id[switch_env_ids][previously_active]
        switch_phase_jump[previously_active] = wrapped_phase_distance(
          self.phase[switch_env_ids][previously_active],
          memory.phase_start[switched_ids[previously_active]],
        )
        switch_cadence_jump[previously_active] = torch.abs(
          memory.cadence[switched_ids[previously_active]] - memory.cadence[prev_ids]
        )
      self.metrics["retrieval_switches"][switch_env_ids[previously_active]] += 1.0
      self.metrics["phase_jump"][switch_env_ids[previously_active]] += (
        switch_phase_jump[previously_active]
      )
      self.metrics["cadence_jump"][switch_env_ids[previously_active]] += (
        switch_cadence_jump[previously_active]
      )
      self.retrieval_switched[switch_env_ids] = True
      self.phase_jump[switch_env_ids] = switch_phase_jump
      self.cadence_jump[switch_env_ids] = switch_cadence_jump
      self.snippet_id[switch_env_ids] = switched_ids
      self.snippet_frame[switch_env_ids] = 0
      self.elapsed_time[switch_env_ids] = 0.0
      self.current_score[switch_env_ids] = candidate_scores[switch_mask]
      self._pending_switch_count[switch_env_ids] = 0
      self._pending_snippet_id[switch_env_ids] = -1

    keep_mask = ~switch_mask
    if torch.any(keep_mask):
      self.current_score[env_ids[keep_mask]] = current_scores[keep_mask]

    chosen_contact_ok = contact_ok[
      torch.arange(len(env_ids), device=self.device), candidate_offsets
    ]
    chosen_phase_ok = phase_ok[
      torch.arange(len(env_ids), device=self.device), candidate_offsets
    ]
    self.contact_valid[env_ids] = chosen_contact_ok
    self.phase_valid[env_ids] = chosen_phase_ok

  def _update_command(self) -> None:
    if self._memory is None or self._memory.num_snippets == 0:
      self.proposal.zero_()
      self.reference_joint_pos.zero_()
      self.reference_joint_vel.zero_()
      self.reference_foot_pos_b.zero_()
      self.reference_foot_pos_valid.zero_()
      self.phase.zero_()
      self.transition_active.zero_()
      return

    memory = self._get_memory()
    active = self.snippet_id >= 0
    self.proposal.zero_()
    self.reference_joint_pos.zero_()
    self.reference_joint_vel.zero_()
    self.reference_foot_pos_b.zero_()
    self.reference_foot_pos_valid.zero_()
    self.phase.zero_()
    self.reference_contact.zero_()
    self.transition_active.zero_()
    self.turn_snippet_active.zero_()
    self.turn_in_place_v2_active.zero_()
    if not torch.any(active):
      return

    active_ids = self.snippet_id[active]
    self.snippet_frame[active] = torch.clamp(
      torch.round(self.elapsed_time[active] / memory.snippet_dt).long(),
      min=0,
      max=memory.snippet_length - 1,
    )

    self.reference_joint_pos[active] = memory.joint_pos_seq[
      active_ids,
      self.snippet_frame[active],
    ]
    self.reference_joint_vel[active] = memory.joint_vel_seq[
      active_ids,
      self.snippet_frame[active],
    ]
    self.reference_foot_pos_b[active] = memory.local_foot_pos_seq[
      active_ids,
      self.snippet_frame[active],
    ]
    self.reference_foot_pos_valid[active] = memory.has_local_foot_pos_seq
    self.phase[active] = memory.phase_seq[
      active_ids,
      self.snippet_frame[active],
    ]
    self.reference_contact[active] = memory.contact_seq[
      active_ids,
      self.snippet_frame[active],
    ]
    self.transition_active[active] = memory.transition_flag[active_ids] > 0
    self.turn_snippet_active[active] = (
      memory.gait_id[active_ids] == GAIT_NAME_TO_ID["turn"]
    )
    self.turn_in_place_v2_active[active] = self._turn_in_place_v2_snippet_mask[
      active_ids
    ]
    self.proposal[active] = encode_proposals(
      memory.local_root_pos_seq[active_ids],
      memory.local_root_yaw_seq[active_ids],
      memory.pelvis_height_seq[active_ids],
      memory.contact_seq[active_ids],
      self.snippet_frame[active],
      memory.horizon_frames,
    )
    self.elapsed_time[active] += self._env.step_dt

  def _score_candidates(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    memory = self._get_memory()
    twist_cmd = self._get_twist_command(env_ids)
    turn_in_place_intent = turn_intent_mask(
      twist_cmd,
      min_turn_yaw_command=self.cfg.turn_intent_min_yaw_command,
      max_turn_forward_speed=self.cfg.turn_intent_max_forward_speed,
      max_turn_lateral_speed=self.cfg.turn_intent_max_lateral_speed,
    )
    task_score = -(
      self.cfg.task_weight_xy
      * (
        (memory.mean_vx[None, :] - twist_cmd[:, 0:1]) ** 2
        + (memory.mean_vy[None, :] - twist_cmd[:, 1:2]) ** 2
      )
      + self.cfg.task_weight_yaw
      * (memory.mean_yaw_rate[None, :] - twist_cmd[:, 2:3]) ** 2
    )
    if self.cfg.prioritize_pivot_candidates:
      task_score = prioritize_prefixed_candidates(
        task_score,
        turn_intent=turn_in_place_intent,
        preferred_mask=self._pivot_snippet_mask,
      )
    if self.cfg.prioritize_turn_in_place_v2_candidates:
      task_score = prioritize_turn_in_place_candidates(
        task_score,
        turn_in_place_intent=turn_in_place_intent,
        turn_in_place_v2_mask=self._turn_in_place_v2_snippet_mask,
      )
    topk = min(self.cfg.topk_task_candidates, memory.num_snippets)
    candidate_scores, candidate_ids = torch.topk(task_score, k=topk, dim=1)

    current_phase = self.phase[env_ids][:, None]
    phase_start = memory.phase_start[candidate_ids]
    phase_score = torch.cos(2.0 * torch.pi * (current_phase - phase_start))
    phase_ok = phase_score > 0.0

    contact_score, contact_ok = self._contact_compatibility_score(
      env_ids, candidate_ids
    )

    current_joint_pos = self.robot.data.joint_pos[env_ids][:, None, :]
    current_joint_vel = self.robot.data.joint_vel[env_ids][:, None, :]
    candidate_joint_pos = memory.joint_pos_seq[candidate_ids, 0]
    candidate_joint_vel = memory.joint_vel_seq[candidate_ids, 0]
    connect_score = -(
      (candidate_joint_pos - current_joint_pos).pow(2).mean(dim=-1)
      + self.cfg.connect_velocity_scale
      * (candidate_joint_vel - current_joint_vel).pow(2).mean(dim=-1)
    )

    quality_score = memory.quality_score[candidate_ids]
    transition_bias = self._transition_bias(env_ids, candidate_ids)
    cadence_bias = self._cadence_continuity_bias(env_ids, candidate_ids)
    pivot_bias = pivot_bias_score(
      twist_cmd,
      self._pivot_snippet_mask[candidate_ids],
      min_turn_yaw_command=self.cfg.turn_intent_min_yaw_command,
      max_turn_forward_speed=self.cfg.turn_intent_max_forward_speed,
      max_turn_lateral_speed=self.cfg.turn_intent_max_lateral_speed,
      nonmatch_penalty=self.cfg.pivot_nonmatch_penalty,
    )
    turn_bias = turn_intent_bias_score(
      twist_cmd,
      memory.gait_id[candidate_ids],
      memory.mean_vx[candidate_ids],
      memory.mean_vy[candidate_ids],
      memory.mean_yaw_rate[candidate_ids],
      memory.cadence[candidate_ids],
      turn_gait_id=GAIT_NAME_TO_ID["turn"],
      min_turn_yaw_command=self.cfg.turn_intent_min_yaw_command,
      max_turn_forward_speed=self.cfg.turn_intent_max_forward_speed,
      max_turn_lateral_speed=self.cfg.turn_intent_max_lateral_speed,
      in_place_speed_scale=self.cfg.turn_intent_in_place_speed_scale,
      yaw_rate_scale=self.cfg.turn_intent_yaw_rate_scale,
      min_turn_cadence=self.cfg.turn_intent_min_cadence,
    )
    turn_in_place_v2_bias = turn_in_place_v2_bias_score(
      twist_cmd,
      self._turn_in_place_v2_snippet_mask[candidate_ids],
      min_turn_yaw_command=self.cfg.turn_intent_min_yaw_command,
      max_turn_forward_speed=self.cfg.turn_intent_max_forward_speed,
      max_turn_lateral_speed=self.cfg.turn_intent_max_lateral_speed,
      nonmatch_penalty=self.cfg.turn_in_place_v2_nonmatch_penalty,
    )
    straight_quality_bias = straight_reference_quality_score(
      twist_cmd,
      memory.mean_vx[candidate_ids],
      memory.mean_vy[candidate_ids],
      memory.mean_yaw_rate[candidate_ids],
      memory.local_foot_pos_seq[candidate_ids],
      memory.contact_seq[candidate_ids],
      min_forward_speed=self.cfg.straight_quality_min_forward_speed,
      max_lateral_speed=self.cfg.straight_quality_max_lateral_speed,
      max_abs_yaw_command=self.cfg.straight_quality_max_abs_yaw_command,
      vx_scale=self.cfg.straight_quality_vx_scale,
      yaw_rate_scale=self.cfg.straight_quality_yaw_rate_scale,
      min_swing_lift=self.cfg.straight_quality_min_swing_lift,
      lift_weight=self.cfg.straight_quality_lift_weight,
      yaw_weight=self.cfg.straight_quality_yaw_weight,
    )

    total_score = (
      self.cfg.task_weight * candidate_scores
      + self.cfg.phase_weight * phase_score
      + self.cfg.contact_weight * contact_score
      + self.cfg.connect_weight * connect_score
      + self.cfg.quality_weight * quality_score
      + self.cfg.transition_weight * transition_bias
      + self.cfg.cadence_weight * cadence_bias
      + self.cfg.pivot_weight * pivot_bias
      + self.cfg.turn_intent_weight * turn_bias
      + self.cfg.turn_in_place_v2_weight * turn_in_place_v2_bias
      + self.cfg.straight_quality_weight * straight_quality_bias
    )
    return total_score, candidate_ids, contact_ok, phase_ok

  def _score_specific(
    self, env_ids: torch.Tensor, snippet_ids: torch.Tensor
  ) -> torch.Tensor:
    memory = self._get_memory()
    twist_cmd = self._get_twist_command(env_ids)
    phase_score = torch.cos(
      2.0 * torch.pi * (self.phase[env_ids] - memory.phase_start[snippet_ids])
    )
    contact_score, _ = self._contact_compatibility_score(env_ids, snippet_ids[:, None])
    connect_score = -(
      (memory.joint_pos_seq[snippet_ids, 0] - self.robot.data.joint_pos[env_ids])
      .pow(2)
      .mean(dim=-1)
      + self.cfg.connect_velocity_scale
      * (memory.joint_vel_seq[snippet_ids, 0] - self.robot.data.joint_vel[env_ids])
      .pow(2)
      .mean(dim=-1)
    )
    task_score = -(
      self.cfg.task_weight_xy
      * (
        (memory.mean_vx[snippet_ids] - twist_cmd[:, 0]).pow(2)
        + (memory.mean_vy[snippet_ids] - twist_cmd[:, 1]).pow(2)
      )
      + self.cfg.task_weight_yaw
      * (memory.mean_yaw_rate[snippet_ids] - twist_cmd[:, 2]).pow(2)
    )
    transition_bias = self._transition_bias(env_ids, snippet_ids[:, None]).squeeze(1)
    cadence_bias = self._cadence_continuity_bias(env_ids, snippet_ids[:, None]).squeeze(
      1
    )
    pivot_bias = pivot_bias_score(
      twist_cmd,
      self._pivot_snippet_mask[snippet_ids][:, None],
      min_turn_yaw_command=self.cfg.turn_intent_min_yaw_command,
      max_turn_forward_speed=self.cfg.turn_intent_max_forward_speed,
      max_turn_lateral_speed=self.cfg.turn_intent_max_lateral_speed,
      nonmatch_penalty=self.cfg.pivot_nonmatch_penalty,
    ).squeeze(1)
    turn_bias = turn_intent_bias_score(
      twist_cmd,
      memory.gait_id[snippet_ids][:, None],
      memory.mean_vx[snippet_ids][:, None],
      memory.mean_vy[snippet_ids][:, None],
      memory.mean_yaw_rate[snippet_ids][:, None],
      memory.cadence[snippet_ids][:, None],
      turn_gait_id=GAIT_NAME_TO_ID["turn"],
      min_turn_yaw_command=self.cfg.turn_intent_min_yaw_command,
      max_turn_forward_speed=self.cfg.turn_intent_max_forward_speed,
      max_turn_lateral_speed=self.cfg.turn_intent_max_lateral_speed,
      in_place_speed_scale=self.cfg.turn_intent_in_place_speed_scale,
      yaw_rate_scale=self.cfg.turn_intent_yaw_rate_scale,
      min_turn_cadence=self.cfg.turn_intent_min_cadence,
    ).squeeze(1)
    turn_in_place_v2_bias = turn_in_place_v2_bias_score(
      twist_cmd,
      self._turn_in_place_v2_snippet_mask[snippet_ids][:, None],
      min_turn_yaw_command=self.cfg.turn_intent_min_yaw_command,
      max_turn_forward_speed=self.cfg.turn_intent_max_forward_speed,
      max_turn_lateral_speed=self.cfg.turn_intent_max_lateral_speed,
      nonmatch_penalty=self.cfg.turn_in_place_v2_nonmatch_penalty,
    ).squeeze(1)
    straight_quality_bias = straight_reference_quality_score(
      twist_cmd,
      memory.mean_vx[snippet_ids][:, None],
      memory.mean_vy[snippet_ids][:, None],
      memory.mean_yaw_rate[snippet_ids][:, None],
      memory.local_foot_pos_seq[snippet_ids][:, None],
      memory.contact_seq[snippet_ids][:, None],
      min_forward_speed=self.cfg.straight_quality_min_forward_speed,
      max_lateral_speed=self.cfg.straight_quality_max_lateral_speed,
      max_abs_yaw_command=self.cfg.straight_quality_max_abs_yaw_command,
      vx_scale=self.cfg.straight_quality_vx_scale,
      yaw_rate_scale=self.cfg.straight_quality_yaw_rate_scale,
      min_swing_lift=self.cfg.straight_quality_min_swing_lift,
      lift_weight=self.cfg.straight_quality_lift_weight,
      yaw_weight=self.cfg.straight_quality_yaw_weight,
    ).squeeze(1)
    return (
      self.cfg.task_weight * task_score
      + self.cfg.phase_weight * phase_score
      + self.cfg.contact_weight * contact_score.squeeze(1)
      + self.cfg.connect_weight * connect_score
      + self.cfg.quality_weight * memory.quality_score[snippet_ids]
      + self.cfg.transition_weight * transition_bias
      + self.cfg.cadence_weight * cadence_bias
      + self.cfg.pivot_weight * pivot_bias
      + self.cfg.turn_intent_weight * turn_bias
      + self.cfg.turn_in_place_v2_weight * turn_in_place_v2_bias
      + self.cfg.straight_quality_weight * straight_quality_bias
    )

  def _contact_compatibility_score(
    self,
    env_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    memory = self._get_memory()
    sensor = self._env.scene[self.cfg.contact_sensor_name]
    assert isinstance(sensor, ContactSensor)
    found = sensor.data.found
    assert found is not None
    current_contact = found[env_ids] > 0
    current_left = current_contact[:, 0:1]
    current_right = current_contact[:, 1:2]
    left_start = memory.left_contact_start[candidate_ids] > 0
    right_start = memory.right_contact_start[candidate_ids] > 0

    mismatch_count = (current_left != left_start).float() + (
      current_right != right_start
    ).float()
    score = 1.0 - 0.75 * mismatch_count
    double_support = current_left & current_right
    single_start = left_start ^ right_start
    score = torch.where(
      double_support & single_start, torch.full_like(score, -0.25), score
    )
    contact_ok = mismatch_count <= 1.0
    return score, contact_ok

  def _transition_bias(
    self,
    env_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
  ) -> torch.Tensor:
    memory = self._get_memory()
    twist_cmd = self._get_twist_command(env_ids)
    cmd_speed = torch.linalg.norm(twist_cmd[:, :2], dim=-1)
    near_boundary = torch.abs(cmd_speed - self.cfg.transition_speed_threshold) <= (
      self.cfg.transition_speed_margin
    )
    current_gait = torch.full_like(self.snippet_id[env_ids], GAIT_NAME_TO_ID["walk"])
    active = self.snippet_id[env_ids] >= 0
    current_gait[active] = memory.gait_id[self.snippet_id[env_ids][active]]
    gait_mismatch = (
      (cmd_speed >= self.cfg.transition_speed_threshold)
      & (current_gait == GAIT_NAME_TO_ID["walk"])
    ) | (
      (cmd_speed < self.cfg.transition_speed_threshold)
      & (current_gait == GAIT_NAME_TO_ID["run"])
    )
    boost = (near_boundary | gait_mismatch).float()[:, None]
    return boost * memory.transition_flag[candidate_ids].float()

  def _cadence_continuity_bias(
    self,
    env_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
  ) -> torch.Tensor:
    if self.cfg.cadence_weight <= 0.0:
      return torch.zeros_like(candidate_ids, dtype=torch.float32)

    active = self.snippet_id[env_ids] >= 0
    if not torch.any(active):
      return torch.zeros_like(candidate_ids, dtype=torch.float32)

    memory = self._get_memory()
    bias = torch.zeros_like(candidate_ids, dtype=torch.float32)
    current_ids = self.snippet_id[env_ids][active]
    score = cadence_continuity_score(
      memory.cadence[current_ids][:, None],
      memory.cadence[candidate_ids[active]],
      cadence_scale=self.cfg.cadence_scale,
    )
    mask = cadence_bias_mask(
      twist_cmd=self._get_twist_command(env_ids[active]),
      current_gait_id=memory.gait_id[current_ids][:, None],
      candidate_gait_id=memory.gait_id[candidate_ids[active]],
      turn_gait_id=GAIT_NAME_TO_ID["turn"],
      min_turn_yaw_command=self.cfg.turn_intent_min_yaw_command,
      max_turn_forward_speed=self.cfg.turn_intent_max_forward_speed,
      max_turn_lateral_speed=self.cfg.turn_intent_max_lateral_speed,
    )
    bias[active] = torch.where(mask, score, torch.zeros_like(score))
    return bias

  def _get_memory(self) -> LocomotionMemoryLoader:
    memory = self._memory
    assert memory is not None
    return memory

  def _get_twist_command(self, env_ids: torch.Tensor) -> torch.Tensor:
    twist_term = self._env.command_manager.get_term(self.cfg.twist_command_name)
    assert twist_term is not None
    twist_command_term = cast(_HasCommandTensor, twist_term)
    return twist_command_term.command[env_ids]


@dataclass(kw_only=True)
class LocomotionMemoryCommandCfg(CommandTermCfg):
  entity_name: str
  memory_file: str = ""
  require_memory_file: bool = False
  twist_command_name: str = "twist"
  contact_sensor_name: str = "feet_ground_contact"
  proposal_dim: int = 24
  topk_task_candidates: int = 256
  hysteresis_margin: float = 0.15
  hysteresis_steps: int = 2
  transition_speed_threshold: float = 1.5
  transition_speed_margin: float = 0.2
  task_weight: float = 1.0
  phase_weight: float = 0.8
  contact_weight: float = 1.5
  connect_weight: float = 1.2
  quality_weight: float = 0.3
  transition_weight: float = 0.7
  task_weight_xy: float = 1.0
  task_weight_yaw: float = 1.0
  connect_velocity_scale: float = 0.1
  min_snippet_hold_s: float = 0.0
  max_snippet_hold_s: float = 0.0
  blocked_resample_retry_s: float = 0.02
  phase_boundary_tolerance: float = 0.08
  event_gated_switching: bool = False
  turn_intent_weight: float = 0.0
  cadence_weight: float = 0.0
  cadence_scale: float = 0.25
  turn_intent_min_yaw_command: float = 0.2
  turn_intent_max_forward_speed: float = 0.25
  turn_intent_max_lateral_speed: float = 0.05
  turn_intent_in_place_speed_scale: float = 0.12
  turn_intent_yaw_rate_scale: float = 0.35
  turn_intent_min_cadence: float = 1.3
  prioritize_pivot_candidates: bool = False
  pivot_weight: float = 0.0
  pivot_nonmatch_penalty: float = 0.5
  pivot_source_prefix: str = PIVOT_PREFIX
  prioritize_turn_in_place_v2_candidates: bool = False
  turn_in_place_v2_weight: float = 0.0
  turn_in_place_v2_nonmatch_penalty: float = 0.5
  turn_in_place_v2_source_prefix: str = TURN_IN_PLACE_V2_PREFIX
  straight_quality_weight: float = 0.0
  straight_quality_min_forward_speed: float = 1.0
  straight_quality_max_lateral_speed: float = 0.05
  straight_quality_max_abs_yaw_command: float = 0.08
  straight_quality_vx_scale: float = 0.25
  straight_quality_yaw_rate_scale: float = 0.15
  straight_quality_min_swing_lift: float = 0.03
  straight_quality_lift_weight: float = 0.65
  straight_quality_yaw_weight: float = 0.35

  def build(self, env: ManagerBasedRlEnv) -> LocomotionMemoryCommand:
    return LocomotionMemoryCommand(self, env)
