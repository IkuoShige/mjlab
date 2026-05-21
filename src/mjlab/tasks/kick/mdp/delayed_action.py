"""Joint position action with per-env action-delay randomization (sim2real DR).

Wraps :class:`mjlab.envs.mdp.actions.JointPositionAction` with a ring buffer
that holds the last ``max_delay + 1`` raw actions. Each env independently
samples its delay (in policy steps) at reset; ``process_actions`` then
applies the action from ``delay`` steps ago rather than the current one.

Motivates: real K1's control loop has ~50 ms actuator latency that the
mjlab simulator doesn't model by default. Without it, the policy overfits
to the zero-delay setting and behaves erratically when deployed on either
real hardware or a standalone MuJoCo runtime that introduces its own
latency. Mirrors booster_amp_lab's ``DelayedImplicitActuator`` approach
but at the action-term level so any actuator behind it benefits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  """Joint position action with action-delay DR (per-env, sampled at reset)."""

  delay_steps_range: tuple[int, int] = (0, 0)
  """Inclusive ``(min, max)`` delay in policy steps. ``(0, 0)`` disables the
  buffer and reduces to plain :class:`JointPositionActionCfg`. Booster K1
  control cycle is ~20 ms; ``(2, 8)`` gives 40-160 ms latency window."""

  def build(self, env: ManagerBasedRlEnv) -> DelayedJointPositionAction:
    return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
  """JointPositionAction with a per-env action-delay ring buffer."""

  cfg: DelayedJointPositionActionCfg  # type: ignore[assignment]

  def __init__(
    self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    self._delay_min, self._delay_max = cfg.delay_steps_range
    if self._delay_min < 0 or self._delay_max < self._delay_min:
      raise ValueError(
        f"delay_steps_range must be a non-negative (min, max) with min<=max, "
        f"got {cfg.delay_steps_range}"
      )
    self._enabled = self._delay_max > 0
    if not self._enabled:
      return
    # Ring buffer of (max_delay + 1) snapshots so we can index back by 0..max.
    self._buffer_len = self._delay_max + 1
    self._buffer = torch.zeros(
      self._buffer_len, self.num_envs, self.action_dim, device=self.device
    )
    self._buffer_head = 0
    self._delay_per_env = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._env_idx = torch.arange(self.num_envs, device=self.device)
    self._sample_delay(slice(None))

  def _sample_delay(self, env_ids: torch.Tensor | slice) -> None:
    if self._delay_max == self._delay_min:
      self._delay_per_env[env_ids] = self._delay_min
      return
    n = (
      self.num_envs if isinstance(env_ids, slice) else int(env_ids.numel())  # type: ignore[arg-type]
    )
    self._delay_per_env[env_ids] = torch.randint(
      self._delay_min, self._delay_max + 1, (n,), device=self.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if not self._enabled:
      return
    if env_ids is None:
      env_ids = slice(None)
    self._buffer[:, env_ids] = 0.0
    self._sample_delay(env_ids)

  def process_actions(self, actions: torch.Tensor) -> None:
    if not self._enabled:
      super().process_actions(actions)
      return
    # Push the freshly arrived action, then read the (per-env) delayed slot.
    self._buffer[self._buffer_head] = actions
    indices = (self._buffer_head - self._delay_per_env) % self._buffer_len
    delayed = self._buffer[indices, self._env_idx]
    self._buffer_head = (self._buffer_head + 1) % self._buffer_len
    super().process_actions(delayed)
