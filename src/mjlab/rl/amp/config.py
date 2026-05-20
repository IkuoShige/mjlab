"""AMP configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AMPCfg:
  """Top-level AMP config consumed by the runner.

  Combines motion buffer, state extractor, and discriminator training settings.
  """

  motion_files: list[str] = field(default_factory=list)
  """Paths to reference motion .npz files."""

  state_fields: tuple[str, ...] = ("joint_pos", "joint_vel")
  """Per-frame fields from the motion NPZ to include in the AMP state."""

  body_fields: tuple[str, ...] = ()
  """Per-body fields to include (e.g. "body_pos_w", "body_lin_vel_w").

  Body positions are converted to root-relative when included.
  """

  body_names: tuple[str, ...] = ()
  """Body names to include when ``body_fields`` is nonempty.

  Names must be valid in both the motion NPZ body ordering and the policy
  robot's body list.
  """

  body_indexes: tuple[int, ...] = ()
  """Indexes into the NPZ ``body_pos_w`` etc. tensors for the named bodies."""

  root_body_index: int = 0
  """Index of the root body in the motion NPZ (used for root-relative xforms)."""

  discriminator_hidden: tuple[int, ...] = (1024, 128)
  """Hidden layer sizes for the AMP discriminator MLP."""

  buffer_batch_size: int = 4096
  """Reference-motion transitions sampled per discriminator update."""

  policy_batch_size: int = 4096
  """Policy transitions sampled per discriminator update."""

  tanh_scale: float = 0.4
  """tanh saturation scale: D'(x) = tanh(tanh_scale * D(x))."""

  grad_penalty_coef: float = 50.0
  """Gradient penalty weight in the discriminator loss."""

  style_reward_coef: float = 0.3
  """Weight applied to AMP style reward when added to env rewards."""

  discriminator_lr: float = 1e-4
  """Learning rate for the discriminator optimizer (separate from policy)."""

  discriminator_update_steps: int = 1
  """Discriminator update steps per PPO iteration."""

  device: str = "cuda:0"
  """Torch device for discriminator + motion buffer tensors."""
