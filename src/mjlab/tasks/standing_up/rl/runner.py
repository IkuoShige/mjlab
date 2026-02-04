"""Custom runner for standing-up task.

Extends MjlabOnPolicyRunner to persist curriculum state
(force magnitude and action scale) across checkpoints.
"""

from __future__ import annotations

import torch

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper


class StandingUpRunner(MjlabOnPolicyRunner):
  """Runner that persists curriculum state for standing-up task.

  Saves and restores force_magnitude and action_rescale so that
  curriculum progress is maintained across training restarts.
  """

  env: RslRlVecEnvWrapper

  def log_curriculum_state(self) -> dict[str, float]:
    """Get curriculum state for logging.

    Returns:
      Dict of curriculum values for TensorBoard logging.
    """
    unwrapped = self.env.unwrapped
    log_dict: dict[str, float] = {}

    if hasattr(unwrapped, "force_magnitude") and unwrapped.force_magnitude is not None:
      force = unwrapped.force_magnitude
      if isinstance(force, torch.Tensor):
        log_dict["Curriculum/force_magnitude_mean"] = force.mean().item()
        log_dict["Curriculum/force_magnitude_min"] = force.min().item()

    if hasattr(unwrapped, "action_rescale") and unwrapped.action_rescale is not None:
      action_scale = unwrapped.action_rescale
      if isinstance(action_scale, torch.Tensor):
        log_dict["Curriculum/action_scale_mean"] = action_scale.mean().item()
        log_dict["Curriculum/action_scale_min"] = action_scale.min().item()

    if hasattr(unwrapped, "max_head_height") and unwrapped.max_head_height is not None:
      max_height = unwrapped.max_head_height
      if isinstance(max_height, torch.Tensor):
        log_dict["Curriculum/max_head_height_mean"] = max_height.mean().item()

    return log_dict

  def save(self, path: str, infos: dict | None = None) -> None:
    """Save checkpoint with curriculum state.

    Args:
      path: Path to save checkpoint.
      infos: Optional additional info to save.
    """
    curriculum_state = {}
    unwrapped = self.env.unwrapped

    # Save force magnitude if present.
    # Dynamic attributes are set by curriculum manager.
    if hasattr(unwrapped, "force_magnitude") and unwrapped.force_magnitude is not None:
      curriculum_state["force_magnitude"] = unwrapped.force_magnitude.cpu()  # type: ignore[attr-defined]

    # Save action rescale if present.
    if hasattr(unwrapped, "action_rescale") and unwrapped.action_rescale is not None:
      curriculum_state["action_rescale"] = unwrapped.action_rescale.cpu()  # type: ignore[attr-defined]

    # Save max head height tracking if present.
    if (
      hasattr(unwrapped, "_max_head_height") and unwrapped._max_head_height is not None
    ):
      curriculum_state["max_head_height"] = unwrapped._max_head_height.cpu()  # type: ignore[attr-defined]

    infos = {**(infos or {}), "curriculum_state": curriculum_state}
    super().save(path, infos)

  def load(
    self, path: str, load_optimizer: bool = True, map_location: str | None = None
  ) -> dict | None:
    """Load checkpoint and restore curriculum state.

    Args:
      path: Path to load checkpoint from.
      load_optimizer: Whether to load optimizer state.
      map_location: Device mapping for loading.

    Returns:
      Info dict from checkpoint if present.
    """
    infos = super().load(path, load_optimizer, map_location)

    if infos and "curriculum_state" in infos:
      curriculum_state = infos["curriculum_state"]
      unwrapped = self.env.unwrapped
      device = unwrapped.device

      # Restore force magnitude.
      # Dynamic attributes are set by curriculum manager.
      if "force_magnitude" in curriculum_state:
        force = curriculum_state["force_magnitude"].to(device)
        unwrapped.force_magnitude = force  # type: ignore[attr-defined]

      # Restore action rescale.
      if "action_rescale" in curriculum_state:
        action_scale = curriculum_state["action_rescale"].to(device)
        unwrapped.action_rescale = action_scale  # type: ignore[attr-defined]

      # Restore max head height tracking.
      if "max_head_height" in curriculum_state:
        max_height = curriculum_state["max_head_height"].to(device)
        unwrapped._max_head_height = max_height  # type: ignore[attr-defined]

    return infos
