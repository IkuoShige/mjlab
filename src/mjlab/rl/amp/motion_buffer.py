"""AMP motion buffer: load reference motion clips and sample transitions.

State features are translation-invariant (root-relative for body fields)
so the discriminator generalizes across spawn positions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class AMPMotionBufferCfg:
  motion_files: list[str] = field(default_factory=list)
  """Paths to .npz motion files."""

  state_fields: tuple[str, ...] = ("joint_pos", "joint_vel")
  """Per-frame fields drawn directly from the NPZ as part of the AMP state."""

  body_fields: tuple[str, ...] = ()
  """Per-body fields. Supported: ``body_pos_w``, ``body_lin_vel_w``,
  ``body_ang_vel_w``. Position is converted to root-relative.
  """

  body_indexes: tuple[int, ...] = ()
  """Indexes into the NPZ body axis to include when ``body_fields`` is set."""

  root_body_index: int = 0
  """Index of the root body in the NPZ body axis (for root-relative xforms)."""

  device: str = "cuda:0"
  """Device tensors are loaded onto."""


_BODY_FIELD_KEYS: dict[str, str] = {
  "body_pos_w": "body_pos_w",
  "body_lin_vel_w": "body_lin_vel_w",
  "body_ang_vel_w": "body_ang_vel_w",
}


class AMPMotionBuffer:
  """Reference motion buffer that yields ``(s_t, s_{t+1})`` transitions."""

  def __init__(self, cfg: AMPMotionBufferCfg) -> None:
    if not cfg.motion_files:
      raise ValueError("AMPMotionBufferCfg.motion_files must not be empty")
    self.cfg = cfg
    self.device = torch.device(cfg.device)

    state_list: list[torch.Tensor] = []
    clip_lengths: list[int] = []
    clip_offsets: list[int] = [0]

    for path in cfg.motion_files:
      if not os.path.isfile(path):
        raise FileNotFoundError(f"AMP motion file missing: {path}")
      data = np.load(path)
      state = self._build_clip_state(data)
      state_list.append(state)
      clip_lengths.append(state.shape[0])
      clip_offsets.append(clip_offsets[-1] + state.shape[0])

    self._states = torch.cat(state_list, dim=0).contiguous()
    self._clip_lengths = torch.tensor(
      clip_lengths, dtype=torch.long, device=self.device
    )
    self._clip_offsets = torch.tensor(
      clip_offsets[:-1], dtype=torch.long, device=self.device
    )
    self._state_dim = self._states.shape[1]
    self._num_clips = len(clip_lengths)
    self._valid_pair_mask = self._build_valid_pair_mask()

  def _build_clip_state(self, data: np.lib.npyio.NpzFile) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    cfg = self.cfg

    for key in cfg.state_fields:
      arr = np.asarray(data[key], dtype=np.float32)
      if arr.ndim != 2:
        raise ValueError(
          f"AMP state field {key!r} must be 2D (T, D), got shape {arr.shape}"
        )
      parts.append(torch.as_tensor(arr, device=self.device))

    if cfg.body_fields:
      if not cfg.body_indexes:
        raise ValueError(
          "AMPMotionBufferCfg.body_indexes must be set when body_fields is nonempty"
        )
      body_idx = torch.as_tensor(cfg.body_indexes, dtype=torch.long, device=self.device)
      for key in cfg.body_fields:
        if key not in _BODY_FIELD_KEYS:
          raise ValueError(
            f"Unsupported AMP body field {key!r}; expected one of "
            f"{tuple(_BODY_FIELD_KEYS)}"
          )
        arr = np.asarray(data[_BODY_FIELD_KEYS[key]], dtype=np.float32)
        if arr.ndim != 3:
          raise ValueError(
            f"Body field {key!r} must be (T, N_bodies, 3), got {arr.shape}"
          )
        tens = torch.as_tensor(arr, device=self.device)
        selected = tens.index_select(dim=1, index=body_idx)
        if key == "body_pos_w":
          root = tens[:, cfg.root_body_index : cfg.root_body_index + 1]
          selected = selected - root
        parts.append(selected.flatten(start_dim=1))

    return torch.cat(parts, dim=1)

  def _build_valid_pair_mask(self) -> torch.Tensor:
    mask = torch.ones(self._states.shape[0], dtype=torch.bool, device=self.device)
    for offset, length in zip(
      self._clip_offsets.tolist(), self._clip_lengths.tolist(), strict=True
    ):
      mask[offset + length - 1] = False
    return mask

  @property
  def state_dim(self) -> int:
    return self._state_dim

  @property
  def num_clips(self) -> int:
    return self._num_clips

  def sample(self, batch_size: int) -> torch.Tensor:
    """Sample ``batch_size`` transitions.

    Returns:
      ``(B, 2 * state_dim)`` tensor of ``(s_t, s_{t+1})`` concatenations.
    """
    if batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {batch_size}")
    valid_indices = torch.nonzero(self._valid_pair_mask, as_tuple=False).squeeze(-1)
    sample_idx = valid_indices[
      torch.randint(0, valid_indices.numel(), (batch_size,), device=self.device)
    ]
    s_t = self._states.index_select(0, sample_idx)
    s_tp1 = self._states.index_select(0, sample_idx + 1)
    return torch.cat([s_t, s_tp1], dim=1)
