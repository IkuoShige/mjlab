"""Extract AMP state from a live policy environment.

Must produce features in the same space as ``AMPMotionBuffer`` so the
discriminator receives comparable inputs from real motion clips and
the policy's rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass
class AMPStateExtractorCfg:
  asset_name: str = "robot"
  """Scene entity to extract proprioception from."""

  state_fields: tuple[str, ...] = ("joint_pos", "joint_vel")
  """Per-frame proprioception keys to include."""

  body_fields: tuple[str, ...] = ()
  """Per-body fields (``body_pos_w``, ``body_lin_vel_w``, ``body_ang_vel_w``)."""

  body_names: tuple[str, ...] = field(default_factory=tuple)
  """Body names to include when ``body_fields`` is nonempty."""

  root_body_name: str = ""
  """Body used as the root for root-relative position xforms."""


class AMPStateExtractor:
  """Pulls AMP-state features off the current sim step."""

  def __init__(
    self,
    cfg: AMPStateExtractorCfg,
    env: "ManagerBasedRlEnv",
    device: str = "cuda:0",
  ) -> None:
    self.cfg = cfg
    self._env = env
    self._device = torch.device(device)

    robot: Entity = env.scene[cfg.asset_name]
    body_names = list(robot.body_names)
    self._body_idx: torch.Tensor | None = None
    if cfg.body_fields:
      missing = [n for n in cfg.body_names if n not in body_names]
      if missing:
        raise ValueError(
          f"AMP body names not in robot body list: {missing}; available: {body_names}"
        )
      self._body_idx = torch.tensor(
        [body_names.index(n) for n in cfg.body_names],
        dtype=torch.long,
        device=self._device,
      )
      if not cfg.root_body_name:
        raise ValueError("root_body_name must be set when body_fields is nonempty")
      if cfg.root_body_name not in body_names:
        raise ValueError(
          f"root_body_name {cfg.root_body_name!r} not in robot bodies {body_names}"
        )
      self._root_idx = body_names.index(cfg.root_body_name)
    else:
      self._root_idx = -1

    self._state_dim = self._probe_state_dim()

  def _probe_state_dim(self) -> int:
    return int(self.get_state().shape[1])

  @property
  def state_dim(self) -> int:
    return self._state_dim

  def get_state(self) -> torch.Tensor:
    """Return current ``(num_envs, state_dim)`` AMP state."""
    cfg = self.cfg
    robot = self._env.scene[cfg.asset_name]
    parts: list[torch.Tensor] = []

    for key in cfg.state_fields:
      tens = getattr(robot.data, key)
      if tens is None:
        raise AttributeError(f"robot.data has no field {key!r}")
      parts.append(tens)

    if cfg.body_fields and self._body_idx is not None:
      for key in cfg.body_fields:
        tens = getattr(robot.data, key)
        if tens is None:
          raise AttributeError(f"robot.data has no field {key!r}")
        selected = tens.index_select(dim=1, index=self._body_idx)
        if key == "body_pos_w":
          root = tens[:, self._root_idx : self._root_idx + 1]
          selected = selected - root
        parts.append(selected.flatten(start_dim=1))

    return torch.cat(parts, dim=1)
