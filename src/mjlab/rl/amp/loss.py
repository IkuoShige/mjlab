"""WGAN-style discriminator loss + AMP style reward.

Reproduces LVDRS Appendix C:

  L_D     = -E[tanh(s * D(x_real))] + E[tanh(s * D(x_fake))]
  L_grad  = coef * E[(||grad D(x_hat)|| - 1)^2]    with x_hat ~ alpha-interp
  r_amp   = -tanh(s * D(x_fake))                   (added to env reward)
"""

from __future__ import annotations

import torch

from mjlab.rl.amp.discriminator import AMPDiscriminator


def discriminator_loss(
  d_real: torch.Tensor,
  d_fake: torch.Tensor,
  grad_interp: torch.Tensor,
  tanh_scale: float = 0.4,
  grad_penalty_coef: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Discriminator loss combining WGAN-tanh and gradient penalty.

  Args:
    d_real: ``(B, 1)`` scores on reference-motion transitions.
    d_fake: ``(B, 1)`` scores on policy-generated transitions.
    grad_interp: ``(B, transition_dim)`` gradient of ``D(x_hat)`` w.r.t.
      ``x_hat`` (interpolated input).
    tanh_scale: saturation scale ``s`` applied before ``tanh``.
    grad_penalty_coef: weight of the gradient penalty.

  Returns:
    ``(total, wgan, grad_penalty)`` — all scalar tensors. ``total`` is
    ``wgan + grad_penalty`` and is the value to backprop.
  """
  wgan = (
    -torch.tanh(tanh_scale * d_real).mean() + torch.tanh(tanh_scale * d_fake).mean()
  )
  grad_norm = grad_interp.flatten(start_dim=1).norm(dim=1)
  grad_penalty = grad_penalty_coef * ((grad_norm - 1.0) ** 2).mean()
  return wgan + grad_penalty, wgan, grad_penalty


def style_reward(d_fake: torch.Tensor, tanh_scale: float = 0.4) -> torch.Tensor:
  """Per-sample AMP style reward.

  Args:
    d_fake: ``(B, 1)`` discriminator scores on the policy's transitions.
    tanh_scale: saturation scale.

  Returns:
    Shape ``(B,)`` style reward in ``(-1, 1)``.
  """
  return -torch.tanh(tanh_scale * d_fake).squeeze(-1)


def compute_gradient_penalty(
  discriminator: AMPDiscriminator,
  x_real: torch.Tensor,
  x_fake: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sample interpolation point and compute gradient for the penalty.

  Args:
    discriminator: the AMPDiscriminator to query.
    x_real: ``(B, transition_dim)`` real transitions.
    x_fake: ``(B, transition_dim)`` fake transitions.

  Returns:
    ``(d_interp, grad_interp)``:
      - ``d_interp``: ``(B, 1)`` discriminator output on the interpolation.
      - ``grad_interp``: ``(B, transition_dim)`` gradient of the score
        w.r.t. the interpolation, ready for the gradient-penalty term.
  """
  if x_real.shape != x_fake.shape:
    raise ValueError(
      f"x_real shape {x_real.shape} must match x_fake shape {x_fake.shape}"
    )
  alpha = torch.rand(x_real.shape[0], 1, device=x_real.device, dtype=x_real.dtype)
  x_hat = alpha * x_real + (1.0 - alpha) * x_fake
  x_hat.requires_grad_(True)
  d_interp = discriminator(x_hat)
  grad_interp = torch.autograd.grad(
    outputs=d_interp.sum(),
    inputs=x_hat,
    create_graph=True,
    retain_graph=True,
  )[0]
  return d_interp, grad_interp
