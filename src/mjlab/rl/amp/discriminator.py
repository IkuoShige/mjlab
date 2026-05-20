"""AMP discriminator network.

LVDRS paper (arXiv:2511.03996) Appendix A Table 1 specifies the
encoder MLP as (1024, 128). We mirror the same shape for the
discriminator's trunk (it operates on a concatenated state transition).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class AMPDiscriminator(nn.Module):
  """WGAN discriminator over state transitions ``(s_t, s_{t+1})``.

  The network outputs a scalar score per sample. WGAN-style losses are
  applied externally (see :mod:`mjlab.rl.amp.loss`).
  """

  def __init__(
    self,
    state_dim: int,
    hidden_dims: Sequence[int] = (1024, 128),
    activation: type[nn.Module] = nn.ELU,
  ) -> None:
    super().__init__()
    self.state_dim = state_dim
    self.transition_dim = 2 * state_dim

    layers: list[nn.Module] = []
    last = self.transition_dim
    for h in hidden_dims:
      layers.append(nn.Linear(last, h))
      layers.append(activation())
      last = h
    layers.append(nn.Linear(last, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, transition: torch.Tensor) -> torch.Tensor:
    """Score a state transition.

    Args:
      transition: ``(B, 2 * state_dim)`` concatenated ``(s_t, s_{t+1})``.

    Returns:
      ``(B, 1)`` raw discriminator score (pre-saturation).
    """
    if transition.shape[-1] != self.transition_dim:
      raise ValueError(
        f"Expected transition with last dim {self.transition_dim}, "
        f"got {transition.shape[-1]}"
      )
    return self.net(transition)
