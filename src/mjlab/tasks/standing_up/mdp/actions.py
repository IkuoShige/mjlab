"""Custom action types for standing-up task.

HoST uses a position offset action where:
  target_position = current_position + action * action_scale

This differs from standard position actions that use:
  target_position = default_position + action * scale
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class JointPositionOffsetActionCfg(BaseActionCfg):
  """Configuration for joint position offset control.

  target = current_pos + action * action_scale

  Unlike standard position actions, this adds the action to the current
  joint position rather than the default position.
  """

  use_curriculum_scale: bool = True
  """Whether to use curriculum-based action scale from env.action_rescale."""

  unactuated_steps: int = 30
  """Number of steps before actions are applied. Robot falls passively during this phase."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> JointPositionOffsetAction:
    return JointPositionOffsetAction(self, env)


class JointPositionOffsetAction(BaseAction):
  """Control joints via position offset targets.

  target = current_position + action * action_scale

  This action type is used in HoST to allow the robot to move relative
  to its current configuration, which is important for getting up from
  various lying positions.
  """

  cfg: JointPositionOffsetActionCfg

  def __init__(self, cfg: JointPositionOffsetActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._use_curriculum_scale = cfg.use_curriculum_scale
    self._unactuated_steps = cfg.unactuated_steps

  def process_actions(self, actions: torch.Tensor):
    """Process raw actions.

    Unlike the base class, we don't apply offset here since we'll
    add to current position in apply_actions.
    """
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale

  def apply_actions(self) -> None:
    """Apply position offset actions.

    Computes target as: current_pos + processed_action * curriculum_scale
    Actions are zeroed during the unactuated phase to let the robot settle.
    """
    # Get current joint positions.
    current_pos = self._entity.data.joint_pos[:, self._target_ids]

    # Check if past unactuated phase (let robot fall and settle first).
    past_unactuated = self._env.episode_length_buf > self._unactuated_steps

    # Apply curriculum-based action scale if enabled.
    if self._use_curriculum_scale and hasattr(self._env, "action_rescale"):
      curriculum_scale: torch.Tensor = self._env.action_rescale  # type: ignore[attr-defined]
      if curriculum_scale.dim() == 2 and curriculum_scale.shape[-1] == 1:
        curriculum_scale = curriculum_scale.squeeze(-1)
      offset = self._processed_actions * curriculum_scale.unsqueeze(-1)
    else:
      offset = self._processed_actions

    # Mask actions during unactuated phase (zero offset = stay at current position).
    offset = offset * past_unactuated.unsqueeze(-1).float()

    target = current_pos + offset

    # Apply encoder bias correction.
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = target - encoder_bias

    # Set joint position target.
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset raw actions to zero for specified environments."""
    super().reset(env_ids)
