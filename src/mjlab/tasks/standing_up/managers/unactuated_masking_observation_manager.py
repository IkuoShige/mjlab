"""Observation manager with unactuated phase masking for standing-up task.

During the unactuated phase (first N steps), observations are zeroed out
to match the Isaac Gym HoST implementation. This allows the robot to fall
and settle before starting to act.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationManager,
)
from mjlab.utils.noise import noise_cfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class UnactuatedMaskingObservationManager(ObservationManager):
  """Observation manager that masks observations during unactuated phase.

  This manager extends ObservationManager to zero out observations during
  the initial unactuated phase. This matches Isaac Gym HoST behavior where:
    current_obs *= episode_length_buf > unactuated_time

  The masking is applied BEFORE adding to history buffers, so the history
  also contains zeros during the unactuated phase. The robot falls passively
  during this phase, and the policy learns to output zero actions when
  receiving zero observations.
  """

  def __init__(
    self,
    cfg: dict[str, ObservationGroupCfg],
    env: ManagerBasedRlEnv,
    *,
    unactuated_steps: int = 30,
  ):
    """Initialize the unactuated masking observation manager.

    Args:
      cfg: Observation group configurations.
      env: The environment.
      unactuated_steps: Number of steps before observations are unmasked.
        Default 30 (matches Isaac Gym HoST).
    """
    super().__init__(cfg, env)
    self._unactuated_steps = unactuated_steps

  def compute_group(
    self, group_name: str, update_history: bool = False
  ) -> torch.Tensor | dict[str, torch.Tensor]:
    """Compute observations for a group with unactuated masking.

    This method applies masking BEFORE adding observations to history buffers,
    ensuring that the history also contains zeros during the unactuated phase.

    Args:
      group_name: Name of the observation group.
      update_history: Whether to update the history buffer.

    Returns:
      Observation tensor or dict of tensors for the group.
    """
    # Check if past unactuated phase.
    past_unactuated = self._env.episode_length_buf > self._unactuated_steps
    mask = past_unactuated.float().unsqueeze(-1)

    group_cfg = self.cfg[group_name]
    group_term_names = self._group_obs_term_names[group_name]
    group_obs: dict[str, torch.Tensor] = {}
    obs_terms = zip(
      group_term_names, self._group_obs_term_cfgs[group_name], strict=False
    )
    for term_name, term_cfg in obs_terms:
      obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
      if isinstance(term_cfg.noise, noise_cfg.NoiseCfg):
        obs = term_cfg.noise.apply(obs)
      elif isinstance(term_cfg.noise, noise_cfg.NoiseModelCfg):
        obs = self._group_obs_class_instances[term_name](obs)
      if term_cfg.clip:
        obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
      if term_cfg.scale is not None:
        scale = term_cfg.scale
        assert isinstance(scale, torch.Tensor)
        obs = obs.mul_(scale)

      # Check for NaN/Inf before delay/history buffers (per-term checking).
      if group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
        obs = self._check_and_handle_nans(
          obs, context=f"{group_name}/{term_name}", policy=group_cfg.nan_policy
        )

      # CRITICAL: Apply unactuated masking BEFORE adding to history/delay buffers.
      # This ensures the history also contains zeros during unactuated phase.
      obs = obs * mask

      if term_cfg.delay_max_lag > 0:
        delay_buffer = self._group_obs_term_delay_buffer[group_name][term_name]
        delay_buffer.append(obs)
        obs = delay_buffer.compute()
      if term_cfg.history_length > 0:
        circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
        if update_history or not circular_buffer.is_initialized:
          circular_buffer.append(obs)

        if term_cfg.flatten_history_dim:
          group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
        else:
          group_obs[term_name] = circular_buffer.buffer
      else:
        group_obs[term_name] = obs

    # Final NaN check for non-per-term checking.
    if not group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
      if self._group_obs_concatenate[group_name]:
        # Will check after concatenation below.
        pass
      else:
        for term_name in group_obs:
          group_obs[term_name] = self._check_and_handle_nans(
            group_obs[term_name],
            context=f"{group_name}/{term_name}",
            policy=group_cfg.nan_policy,
          )

    if self._group_obs_concatenate[group_name]:
      result = torch.cat(
        list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name]
      )
      # Final check for concatenated result (non-per-term checking).
      if not group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
        result = self._check_and_handle_nans(
          result, context=group_name, policy=group_cfg.nan_policy
        )
      return result
    return group_obs
