"""Curriculum functions for standing-up task.

Implements force curriculum that decreases pulling force as the robot
learns to stand up better.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class ForceCurriculum:
  """Curriculum for decreasing pulling force and action scale.

  When the robot achieves a certain head height, the pulling force
  and action scale are decreased to make the task harder.
  """

  def __init__(
    self,
    initial_force: float = 12.0,
    initial_action_scale: float = 1.0,
    force_decrement: float = 20.0,
    action_scale_decrement: float = 0.02,
    min_force: float = 0.0,
    min_action_scale: float = 0.25,
    threshold_height: float = 0.37,
    head_body_name: str = "keyframe_head_link",
    feet_body_names: tuple[str, ...] = ("l_ankle_roll_link", "r_ankle_roll_link"),
  ):
    """Initialize the force curriculum.

    Args:
      initial_force: Initial pulling force in Newtons. Default 12.0.
      initial_action_scale: Initial action scale. Default 1.0.
      force_decrement: Force decrease amount per curriculum step. Default 20.0.
      action_scale_decrement: Action scale decrease per step. Default 0.02.
      min_force: Minimum force value. Default 0.0.
      min_action_scale: Minimum action scale. Default 0.25.
      threshold_height: Head height threshold to trigger curriculum. Default 0.37m.
      head_body_name: Name of head body for height measurement.
      feet_body_names: Names of feet bodies for relative height.
    """
    self.initial_force = initial_force
    self.initial_action_scale = initial_action_scale
    self.force_decrement = force_decrement
    self.action_scale_decrement = action_scale_decrement
    self.min_force = min_force
    self.min_action_scale = min_action_scale
    self.threshold_height = threshold_height
    self.head_body_name = head_body_name
    self.feet_body_names = feet_body_names

    # Buffers (initialized on first call).
    self._force: torch.Tensor | None = None
    self._action_scale: torch.Tensor | None = None
    self._max_head_height: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset curriculum state for specified environments."""
    if self._max_head_height is not None:
      if env_ids is None:
        env_ids = slice(None)
      self._max_head_height[env_ids] = 0.0

  @property
  def force(self) -> torch.Tensor | None:
    """Current force values."""
    return self._force

  @property
  def action_scale(self) -> torch.Tensor | None:
    """Current action scale values."""
    return self._action_scale

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    """Update curriculum based on achieved head height.

    Called on environment reset. Decreases force/action_scale if
    max head height exceeded threshold.

    Args:
      env: The environment.
      env_ids: Environment IDs being reset.
      asset_cfg: Asset configuration.
    """
    # Initialize buffers.
    if self._force is None:
      self._force = torch.full((env.num_envs,), self.initial_force, device=env.device)
    if self._action_scale is None:
      self._action_scale = torch.full(
        (env.num_envs,), self.initial_action_scale, device=env.device
      )
    if self._max_head_height is None:
      self._max_head_height = torch.zeros(env.num_envs, device=env.device)

    # Get head height for resetting envs.
    asset: Entity = env.scene[asset_cfg.name]

    head_ids, _ = asset.find_bodies(self.head_body_name)
    if not head_ids:
      return
    head_id = head_ids[0]

    feet_ids = []
    for name in self.feet_body_names:
      ids, _ = asset.find_bodies(name)
      if ids:
        feet_ids.extend(ids)

    if not feet_ids:
      return

    body_ids = asset.indexing.body_ids
    head_height = env.sim.data.xpos[env_ids, body_ids[head_id], 2]
    feet_heights = torch.stack(
      [env.sim.data.xpos[env_ids, body_ids[fid], 2] for fid in feet_ids], dim=-1
    )
    feet_height_mean = feet_heights.mean(dim=-1)
    relative_height = head_height - feet_height_mean

    # Update max head height.
    self._max_head_height[env_ids] = torch.max(
      self._max_head_height[env_ids], relative_height
    )

    # Check if threshold exceeded.
    exceeded = self._max_head_height[env_ids] > self.threshold_height

    # Decrease force and action scale for environments that exceeded threshold.
    if exceeded.any():
      new_force = (self._force[env_ids[exceeded]] - self.force_decrement).clamp(
        min=self.min_force
      )
      self._force[env_ids[exceeded]] = new_force

      new_action_scale = (
        self._action_scale[env_ids[exceeded]] - self.action_scale_decrement
      ).clamp(min=self.min_action_scale)
      self._action_scale[env_ids[exceeded]] = new_action_scale

    # Reset max head height for all resetting envs.
    self._max_head_height[env_ids] = 0.0

    # Store in env for use by other components.
    # Dynamic attributes are set on env for curriculum state sharing.
    env.force_magnitude = self._force  # type: ignore[attr-defined]
    env.action_rescale = self._action_scale.unsqueeze(-1)  # type: ignore[attr-defined]


def force_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  threshold_height: float = 0.37,
  force_decrement: float = 20.0,
  action_scale_decrement: float = 0.02,
  min_force: float = 0.0,
  min_action_scale: float = 0.25,
  head_body_name: str = "keyframe_head_link",
  feet_body_names: tuple[str, ...] = ("l_ankle_roll_link", "r_ankle_roll_link"),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Curriculum function that decreases force/action_scale when robot achieves target height.

  Called on environment reset. Computes the current head height for resetting
  environments and updates curriculum values if threshold is exceeded.

  Args:
    env: The environment.
    env_ids: Environment IDs being reset.
    threshold_height: Height threshold to trigger curriculum.
    force_decrement: Force decrease per step.
    action_scale_decrement: Action scale decrease per step.
    min_force: Minimum force.
    min_action_scale: Minimum action scale.
    head_body_name: Name of head body for height measurement.
    feet_body_names: Names of feet bodies for relative height.
    asset_cfg: Asset configuration.
  """
  # Initialize buffers in env if not present.
  if not hasattr(env, "force_magnitude"):
    env.force_magnitude = torch.full(  # type: ignore[attr-defined]
      (env.num_envs,), 12.0, device=env.device
    )
  if not hasattr(env, "action_rescale"):
    env.action_rescale = torch.ones(env.num_envs, 1, device=env.device)  # type: ignore[attr-defined]
  if not hasattr(env, "max_head_height"):
    env.max_head_height = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]

  # Get asset and compute current head height for resetting environments.
  asset: Entity = env.scene[asset_cfg.name]

  # Find head body.
  head_ids, _ = asset.find_bodies(head_body_name)
  if not head_ids:
    # Reset tracking and return if head body not found.
    env.max_head_height[env_ids] = 0.0  # type: ignore[attr-defined]
    return
  head_id = head_ids[0]

  # Find feet bodies.
  feet_ids = []
  for name in feet_body_names:
    ids, _ = asset.find_bodies(name)
    if ids:
      feet_ids.extend(ids)

  if not feet_ids:
    env.max_head_height[env_ids] = 0.0  # type: ignore[attr-defined]
    return

  # Compute relative head height.
  body_ids = asset.indexing.body_ids
  head_height = env.sim.data.xpos[env_ids, body_ids[head_id], 2]
  feet_heights = torch.stack(
    [env.sim.data.xpos[env_ids, body_ids[fid], 2] for fid in feet_ids], dim=-1
  )
  feet_height_mean = feet_heights.mean(dim=-1)
  current_height = head_height - feet_height_mean

  # Use max of tracked height and current height.
  max_height = torch.max(env.max_head_height[env_ids], current_height)  # type: ignore[attr-defined]

  # Check threshold.
  exceeded = max_height > threshold_height

  if exceeded.any():
    exceeded_ids = env_ids[exceeded]
    env.force_magnitude[exceeded_ids] = (  # type: ignore[attr-defined]
      env.force_magnitude[exceeded_ids] - force_decrement  # type: ignore[attr-defined]
    ).clamp(min=min_force)
    env.action_rescale[exceeded_ids] = (  # type: ignore[attr-defined]
      env.action_rescale[exceeded_ids] - action_scale_decrement  # type: ignore[attr-defined]
    ).clamp(min=min_action_scale)

  # Reset tracking for resetting environments.
  env.max_head_height[env_ids] = 0.0  # type: ignore[attr-defined]
