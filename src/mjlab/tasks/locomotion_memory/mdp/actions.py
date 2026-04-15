from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch

from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class _HasReferenceJointPose(Protocol):
  reference_joint_pos: torch.Tensor
  snippet_id: torch.Tensor
  turn_snippet_active: torch.Tensor
  turn_in_place_v2_active: torch.Tensor


def _turn_command_mask(
  command: torch.Tensor,
  *,
  min_abs_yaw_command: float,
  max_forward_speed: float,
  max_lateral_speed: float,
) -> torch.Tensor:
  return (
    (torch.abs(command[:, 2]) >= min_abs_yaw_command)
    & (torch.abs(command[:, 0]) <= max_forward_speed)
    & (torch.abs(command[:, 1]) <= max_lateral_speed)
  )


@dataclass(kw_only=True)
class ReferenceJointPositionActionCfg(BaseActionCfg):
  """Joint-position control around a locomotion-memory reference pose."""

  reference_command_name: str = "memory"
  fallback_to_default_pose: bool = True
  turn_command_name: str = ""
  turn_min_abs_yaw_command: float = 0.2
  turn_max_forward_speed: float = 0.25
  turn_max_lateral_speed: float = 0.05
  turn_residual_scale: float = 1.0
  turn_residual_joint_names: tuple[str, ...] = ()
  turn_snippet_residual_scale: float = 1.0
  turn_snippet_residual_joint_names: tuple[str, ...] = ()
  turn_in_place_v2_residual_scale: float = 1.0
  turn_in_place_v2_residual_joint_names: tuple[str, ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> ReferenceJointPositionAction:
    return ReferenceJointPositionAction(self, env)


class ReferenceJointPositionAction(BaseAction):
  """Apply residual joint actions around the active memory snippet pose."""

  def __init__(self, cfg: ReferenceJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    joint_names = [self._entity.joint_names[i] for i in self._target_ids]
    if cfg.turn_residual_joint_names:
      mask = [
        any(
          re.fullmatch(pattern, joint_name) for pattern in cfg.turn_residual_joint_names
        )
        for joint_name in joint_names
      ]
      self._turn_residual_joint_mask = torch.tensor(
        mask, dtype=torch.bool, device=self._env.device
      )
    else:
      self._turn_residual_joint_mask = torch.ones(
        len(joint_names), dtype=torch.bool, device=self._env.device
      )
    if cfg.turn_snippet_residual_joint_names:
      turn_snippet_mask = [
        any(
          re.fullmatch(pattern, joint_name)
          for pattern in cfg.turn_snippet_residual_joint_names
        )
        for joint_name in joint_names
      ]
      self._turn_snippet_residual_joint_mask = torch.tensor(
        turn_snippet_mask, dtype=torch.bool, device=self._env.device
      )
    else:
      self._turn_snippet_residual_joint_mask = torch.ones(
        len(joint_names), dtype=torch.bool, device=self._env.device
      )
    if cfg.turn_in_place_v2_residual_joint_names:
      v2_mask = [
        any(
          re.fullmatch(pattern, joint_name)
          for pattern in cfg.turn_in_place_v2_residual_joint_names
        )
        for joint_name in joint_names
      ]
      self._turn_in_place_v2_residual_joint_mask = torch.tensor(
        v2_mask, dtype=torch.bool, device=self._env.device
      )
    else:
      self._turn_in_place_v2_residual_joint_mask = torch.ones(
        len(joint_names), dtype=torch.bool, device=self._env.device
      )

  def _cfg(self) -> ReferenceJointPositionActionCfg:
    return cast(ReferenceJointPositionActionCfg, self.cfg)

  def _get_reference_term(self) -> _HasReferenceJointPose:
    reference_term = self._env.command_manager.get_term(
      self._cfg().reference_command_name
    )
    assert reference_term is not None
    return cast(_HasReferenceJointPose, reference_term)

  def apply_actions(self) -> None:
    reference_term = self._get_reference_term()
    reference_joint_pos = reference_term.reference_joint_pos[:, self._target_ids]
    if self._cfg().fallback_to_default_pose:
      fallback_joint_pos = self._entity.data.default_joint_pos[:, self._target_ids]
      active = reference_term.snippet_id >= 0
      reference_joint_pos = torch.where(
        active[:, None],
        reference_joint_pos,
        fallback_joint_pos,
      )

    residual = self._processed_actions
    if self._cfg().turn_command_name:
      command = self._env.command_manager.get_command(self._cfg().turn_command_name)
      assert command is not None
      turn_gate = _turn_command_mask(
        command,
        min_abs_yaw_command=self._cfg().turn_min_abs_yaw_command,
        max_forward_speed=self._cfg().turn_max_forward_speed,
        max_lateral_speed=self._cfg().turn_max_lateral_speed,
      )
      if self._cfg().turn_residual_scale < 1.0:
        attenuated = torch.where(
          self._turn_residual_joint_mask[None, :],
          residual * self._cfg().turn_residual_scale,
          residual,
        )
        residual = torch.where(
          turn_gate[:, None],
          attenuated,
          residual,
        )
      if self._cfg().turn_snippet_residual_scale < 1.0:
        turn_snippet_active = getattr(
          reference_term,
          "turn_snippet_active",
          torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device),
        )
        turn_snippet_gate = turn_gate & turn_snippet_active
        turn_snippet_attenuated = torch.where(
          self._turn_snippet_residual_joint_mask[None, :],
          residual * self._cfg().turn_snippet_residual_scale,
          residual,
        )
        residual = torch.where(
          turn_snippet_gate[:, None],
          turn_snippet_attenuated,
          residual,
        )
      if self._cfg().turn_in_place_v2_residual_scale < 1.0:
        turn_in_place_v2_active = getattr(
          reference_term,
          "turn_in_place_v2_active",
          torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device),
        )
        v2_gate = turn_gate & turn_in_place_v2_active
        v2_attenuated = torch.where(
          self._turn_in_place_v2_residual_joint_mask[None, :],
          residual * self._cfg().turn_in_place_v2_residual_scale,
          residual,
        )
        residual = torch.where(
          v2_gate[:, None],
          v2_attenuated,
          residual,
        )

    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = reference_joint_pos + residual - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
