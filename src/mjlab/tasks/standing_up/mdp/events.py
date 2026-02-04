"""Event functions for standing-up task.

Includes 4-direction reset and pulling force application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import normalize

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Initial orientation quaternions (w, x, y, z format for mjlab).
# Converted from Isaac Gym (x, y, z, w) format.
DEFAULT_ORIENTATIONS = {
  "supine": torch.tensor([0.7071068, 0.0, -0.7071068, 0.0]),  # Back down.
  "prone": torch.tensor([0.7071068, 0.0, 0.7071068, 0.0]),  # Face down.
  "left_side": torch.tensor([0.7071068, 0.7071068, 0.0, 0.0]),  # Left side down.
  "right_side": torch.tensor([0.7071068, -0.7071068, 0.0, 0.0]),  # Right side down.
}


def reset_root_state_4dir(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  orientations: dict[str, tuple[float, float, float, float]] | None = None,
  weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset root state with random orientation from 4 directions.

  Samples from supine, prone, left_side, and right_side orientations
  with specified probability weights.

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all.
    orientations: Dictionary mapping orientation names to quaternions (w, x, y, z).
      If None, uses default orientations.
    weights: Probability weights for each orientation. Default equal weights.
    asset_cfg: Asset configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]

  # Get default root state.
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_state = default_root_state[env_ids].clone()

  # Add env origins.
  root_state[:, 0:3] += env.scene.env_origins[env_ids]

  # Setup orientations.
  if orientations is None:
    quats = [
      DEFAULT_ORIENTATIONS[k].to(env.device)
      for k in ["supine", "prone", "left_side", "right_side"]
    ]
  else:
    quats = [
      torch.tensor(orientations[k], device=env.device) for k in orientations.keys()
    ]

  # Ensure we have exactly 4 orientations matching 4 weights.
  assert len(quats) == len(weights), (
    f"Number of orientations ({len(quats)}) must match weights ({len(weights)})"
  )

  # Sample orientation index for each environment.
  weight_tensor = torch.tensor(weights, device=env.device, dtype=torch.float)
  orientation_idx = torch.multinomial(
    weight_tensor.expand(len(env_ids), -1),
    num_samples=1,
    replacement=True,
  ).squeeze(-1)

  # Apply selected orientations.
  for i, quat in enumerate(quats):
    mask = orientation_idx == i
    if mask.any():
      root_state[mask, 3:7] = normalize(quat.unsqueeze(0))

  # Write to simulation.
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)


class ApplyPullingForce:
  """Apply vertical pulling force to help robot stand up.

  The force is only applied when the robot is roughly upright
  (projected gravity z < threshold).
  """

  def __init__(
    self,
    initial_force: float = 12.0,
    gravity_threshold: float = -0.8,
    base_body_name: str = "base_link",
  ):
    """Initialize the pulling force applicator.

    Args:
      initial_force: Initial upward force magnitude in Newtons. Default 12.0.
      gravity_threshold: Projected gravity z threshold for applying force.
        Force is only applied when proj_gravity_z < threshold. Default -0.8.
      base_body_name: Name of the body to apply force to.
    """
    self.initial_force = initial_force
    self.gravity_threshold = gravity_threshold
    self.base_body_name = base_body_name
    self._force_magnitude: torch.Tensor | None = None
    self._base_body_id: int | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset force magnitude for specified environments."""
    if self._force_magnitude is None:
      return
    if env_ids is None:
      env_ids = slice(None)
    self._force_magnitude[env_ids] = self.initial_force

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    """Apply pulling force to specified environments.

    Args:
      env: The environment.
      env_ids: Environment IDs to apply force to. If None, applies to all.
      asset_cfg: Asset configuration.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Initialize force magnitude buffer.
    if self._force_magnitude is None:
      self._force_magnitude = torch.full(
        (env.num_envs,), self.initial_force, device=env.device
      )

    # Find base body ID.
    if self._base_body_id is None:
      body_ids, _ = asset.find_bodies(self.base_body_name)
      if not body_ids:
        return
      self._base_body_id = body_ids[0]

    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    # Check if robot is upright enough to apply force.
    proj_gravity_z = asset.data.projected_gravity_b[env_ids, 2]
    apply_mask = proj_gravity_z < self.gravity_threshold

    # Create force tensor.
    forces = torch.zeros(len(env_ids), asset.num_bodies, 3, device=env.device)
    torques = torch.zeros_like(forces)

    # Apply vertical force to base body.
    forces[:, self._base_body_id, 2] = (
      self._force_magnitude[env_ids] * apply_mask.float()
    )

    # Write to simulation.
    asset.write_external_wrench_to_sim(forces, torques, env_ids=env_ids, body_ids=None)


def reset_joints_random(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_scale_range: tuple[float, float] = (0.9, 1.1),
  position_offset_range: tuple[float, float] = (-0.1, 0.1),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset joint positions with randomization.

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all.
    position_scale_range: (min, max) scale factor for default positions.
    position_offset_range: (min, max) offset to add to positions.
    asset_cfg: Asset configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  default_pos = asset.data.default_joint_pos
  assert default_pos is not None

  # Apply scale randomization.
  scale = torch.empty(len(env_ids), asset.num_joints, device=env.device).uniform_(
    position_scale_range[0], position_scale_range[1]
  )
  pos = default_pos[env_ids] * scale

  # Apply offset randomization.
  offset = torch.empty_like(pos).uniform_(
    position_offset_range[0], position_offset_range[1]
  )
  pos = pos + offset

  # Clamp to joint limits.
  soft_limits = asset.data.soft_joint_pos_limits
  assert soft_limits is not None
  pos = pos.clamp(soft_limits[env_ids, :, 0], soft_limits[env_ids, :, 1])

  # Zero velocity.
  vel = torch.zeros_like(pos)

  asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


def init_curriculum_values(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  initial_action_rescale: float = 1.0,
  initial_force_magnitude: float = 12.0,
) -> None:
  """Initialize curriculum values on the environment at startup.

  This ensures action_rescale and force_magnitude are available from
  the first step, preventing NaN issues in observations and actions.

  Args:
    env: The environment.
    env_ids: Not used (startup event applies to all).
    initial_action_rescale: Initial action scale value. Default 1.0.
    initial_force_magnitude: Initial pulling force magnitude. Default 12.0.
  """
  # Initialize action_rescale as (num_envs, 1) tensor for observation.
  if not hasattr(env, "action_rescale"):
    env.action_rescale = torch.full(  # type: ignore[attr-defined]
      (env.num_envs, 1), initial_action_rescale, device=env.device
    )

  # Initialize force_magnitude as (num_envs,) tensor for force application.
  if not hasattr(env, "force_magnitude"):
    env.force_magnitude = torch.full(  # type: ignore[attr-defined]
      (env.num_envs,), initial_force_magnitude, device=env.device
    )

  # Track max head height for curriculum progression.
  if not hasattr(env, "max_head_height"):
    env.max_head_height = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]


def track_head_height(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  head_body_name: str = "keyframe_head_link",
  feet_body_names: tuple[str, ...] = ("l_ankle_roll_link", "r_ankle_roll_link"),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Track maximum head height for curriculum progression.

  Called every step (interval event). Updates env.max_head_height with the
  maximum relative head height achieved during the episode.

  Args:
    env: The environment.
    env_ids: Environment IDs (not used, tracks all).
    head_body_name: Name of head body for height measurement.
    feet_body_names: Names of feet bodies for relative height.
    asset_cfg: Asset configuration.
  """
  # Initialize tracking buffer if needed.
  if not hasattr(env, "max_head_height"):
    env.max_head_height = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]

  asset: Entity = env.scene[asset_cfg.name]

  # Find head body.
  head_ids, _ = asset.find_bodies(head_body_name)
  if not head_ids:
    return
  head_id = head_ids[0]

  # Find feet bodies.
  feet_ids = []
  for name in feet_body_names:
    ids, _ = asset.find_bodies(name)
    if ids:
      feet_ids.extend(ids)

  if not feet_ids:
    return

  # Compute relative head height for all environments.
  body_ids = asset.indexing.body_ids
  head_height = env.sim.data.xpos[:, body_ids[head_id], 2]
  feet_heights = torch.stack(
    [env.sim.data.xpos[:, body_ids[fid], 2] for fid in feet_ids], dim=-1
  )
  feet_height_mean = feet_heights.mean(dim=-1)
  current_height = head_height - feet_height_mean

  # Update max head height.
  env.max_head_height = torch.max(env.max_head_height, current_height)  # type: ignore[attr-defined]


def apply_pulling_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  force_magnitude: float = 12.0,
  gravity_threshold: float = -0.8,
  base_body_name: str = "base_link",
  unactuated_steps: int = 30,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply vertical pulling force to help robot stand up.

  Simple function version (stateless) for use with EventTermCfg.

  Args:
    env: The environment.
    env_ids: Environment IDs (not used, force applied to all).
    force_magnitude: Upward force in Newtons. Default 12.0.
    gravity_threshold: Apply force when proj_gravity_z < threshold. Default -0.8.
    base_body_name: Name of body to apply force to.
    unactuated_steps: Don't apply force during unactuated phase.
    asset_cfg: Asset configuration.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Find base body ID.
  body_ids, _ = asset.find_bodies(base_body_name)
  if not body_ids:
    return
  base_body_id = body_ids[0]

  # Get force magnitude from env if curriculum is active.
  if hasattr(env, "force_magnitude"):
    force = env.force_magnitude
  else:
    force = torch.full((env.num_envs,), force_magnitude, device=env.device)

  # Check if robot is upright enough and past unactuated phase.
  proj_gravity_z = asset.data.projected_gravity_b[:, 2]
  apply_mask = proj_gravity_z < gravity_threshold
  past_unactuated = env.episode_length_buf > unactuated_steps
  apply_mask = apply_mask & past_unactuated

  # Create force tensor.
  forces = torch.zeros(env.num_envs, asset.num_bodies, 3, device=env.device)
  torques = torch.zeros_like(forces)

  # Apply vertical force to base body.
  forces[:, base_body_id, 2] = force * apply_mask.float()

  # Write to simulation.
  all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset.write_external_wrench_to_sim(
    forces, torques, env_ids=all_env_ids, body_ids=None
  )
