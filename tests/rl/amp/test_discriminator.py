"""Smoke tests for the AMP discriminator + loss + gradient penalty."""

from __future__ import annotations

import torch

from mjlab.rl.amp import (
  AMPDiscriminator,
  compute_gradient_penalty,
  discriminator_loss,
  style_reward,
)


def test_discriminator_forward_shape() -> None:
  state_dim = 44
  disc = AMPDiscriminator(state_dim=state_dim)
  x = torch.randn(8, 2 * state_dim)
  out = disc(x)
  assert out.shape == (8, 1)


def test_discriminator_loss_finite_and_decomposes() -> None:
  state_dim = 10
  batch = 16
  disc = AMPDiscriminator(state_dim=state_dim, hidden_dims=(32, 16))
  x_real = torch.randn(batch, 2 * state_dim)
  x_fake = torch.randn(batch, 2 * state_dim)
  d_real = disc(x_real)
  d_fake = disc(x_fake)
  _, grad_interp = compute_gradient_penalty(disc, x_real, x_fake)
  total, wgan, grad_p = discriminator_loss(d_real, d_fake, grad_interp)
  assert torch.isfinite(total)
  assert torch.isfinite(wgan)
  assert torch.isfinite(grad_p)
  assert torch.allclose(total, wgan + grad_p)


def test_style_reward_range() -> None:
  d_fake = torch.randn(32, 1) * 5.0
  r = style_reward(d_fake, tanh_scale=0.4)
  assert r.shape == (32,)
  assert r.abs().max() < 1.0  # tanh-bounded


def test_gradient_penalty_yields_unit_norm_target() -> None:
  state_dim = 8
  batch = 4
  disc = AMPDiscriminator(state_dim=state_dim, hidden_dims=(16, 8))
  x_real = torch.randn(batch, 2 * state_dim)
  x_fake = torch.randn(batch, 2 * state_dim)
  d_interp, grad_interp = compute_gradient_penalty(disc, x_real, x_fake)
  assert d_interp.shape == (batch, 1)
  assert grad_interp.shape == (batch, 2 * state_dim)
  # Penalty is well-defined.
  grad_norm = grad_interp.flatten(start_dim=1).norm(dim=1)
  assert torch.all(torch.isfinite(grad_norm))
