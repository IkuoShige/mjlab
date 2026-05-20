"""Adversarial Motion Priors (AMP) infrastructure for mjlab.

Reproduces the AMP setup from the LVDRS paper (arXiv:2511.03996, Appendix C):
WGAN-style discriminator over state transitions with tanh saturation and
gradient penalty. Used to provide an implicit motion-style reward without
requiring explicit temporal alignment to reference motion clips.
"""

from __future__ import annotations

from mjlab.rl.amp.config import AMPCfg
from mjlab.rl.amp.discriminator import AMPDiscriminator
from mjlab.rl.amp.loss import (
  compute_gradient_penalty,
  discriminator_loss,
  style_reward,
)
from mjlab.rl.amp.motion_buffer import AMPMotionBuffer, AMPMotionBufferCfg
from mjlab.rl.amp.state_extractor import AMPStateExtractor, AMPStateExtractorCfg

__all__ = [
  "AMPCfg",
  "AMPDiscriminator",
  "AMPMotionBuffer",
  "AMPMotionBufferCfg",
  "AMPStateExtractor",
  "AMPStateExtractorCfg",
  "compute_gradient_penalty",
  "discriminator_loss",
  "style_reward",
]
