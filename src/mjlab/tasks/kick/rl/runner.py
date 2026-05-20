"""KickAMPRunner — PPO with AMP style discriminator.

Subclasses :class:`MjlabOnPolicyRunner` and overrides the rollout loop so
that each env step's reward is augmented with an AMP style reward computed
from a WGAN-style discriminator (see :mod:`mjlab.rl.amp`). After each
rollout, the discriminator is updated on policy-generated vs. reference-
motion state transitions.

Reference for the LVDRS pipeline: arXiv:2511.03996, Appendix C.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import torch

from mjlab.rl.amp import (
  AMPCfg,
  AMPDiscriminator,
  AMPMotionBuffer,
  AMPMotionBufferCfg,
  AMPStateExtractor,
  AMPStateExtractorCfg,
  compute_gradient_penalty,
  discriminator_loss,
  style_reward,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv


class KickAMPRunner(MjlabOnPolicyRunner):
  """PPO runner with AMP discriminator + style reward augmentation."""

  def __init__(
    self,
    env: "VecEnv",
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    amp_cfg_dict = train_cfg.pop("amp_cfg", None)
    super().__init__(env, train_cfg, log_dir, device)
    self._device = torch.device(device)

    if amp_cfg_dict is None:
      self.amp: _AMPRuntime | None = None
      return

    self.amp = _build_amp_runtime(amp_cfg_dict, env, self._device)

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if self.amp is None:
      super().learn(num_learning_iterations, init_at_random_ep_len)
      return

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self._device)
    self.alg.train_mode()

    if getattr(self, "is_distributed", False):
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    prev_state = self.amp.state_extractor.get_state().detach()

    for it in range(start_it, total_it):
      start = time.time()
      policy_transitions: list[torch.Tensor] = []

      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          obs = obs.to(self._device)
          rewards = rewards.to(self._device)
          dones = dones.to(self._device)

          cur_state = self.amp.state_extractor.get_state().detach()
          transition = torch.cat([prev_state, cur_state], dim=-1)
          d_fake = self.amp.discriminator(transition)
          r_amp = style_reward(d_fake, self.amp.cfg.tanh_scale)
          rewards = rewards + self.amp.cfg.style_reward_coef * r_amp

          self.alg.process_env_step(obs, rewards, dones, extras)
          self.logger.process_env_step(rewards, dones, extras, None)

          policy_transitions.append(transition.detach())
          # On done, the s_{t+1} is the FIRST frame of the new episode;
          # the transition just pushed crossed an episode boundary, but
          # mixing across episodes is acceptable for a stochastic prior.
          prev_state = cur_state

        collect_time = time.time() - start
        start = time.time()
        self.alg.compute_returns(obs)

      disc_loss_avg = self._update_discriminator(policy_transitions)

      loss_dict = self.alg.update()
      learn_time = time.time() - start
      self.current_learning_iteration = it

      loss_dict["amp/discriminator_loss"] = float(disc_loss_avg)

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=None,
      )

      log_dir = self.logger.log_dir or ""
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      log_dir = self.logger.log_dir or ""
      self.save(os.path.join(log_dir, f"model_{self.current_learning_iteration}.pt"))
      self.logger.stop_logging_writer()

  def _update_discriminator(self, policy_transitions: list[torch.Tensor]) -> float:
    assert self.amp is not None
    if not policy_transitions:
      return 0.0
    fake_all = torch.cat(policy_transitions, dim=0)
    losses: list[float] = []
    for _ in range(self.amp.cfg.discriminator_update_steps):
      idx = torch.randint(
        0, fake_all.shape[0], (self.amp.cfg.policy_batch_size,), device=self._device
      )
      x_fake = fake_all.index_select(0, idx)
      x_real = self.amp.motion_buffer.sample(self.amp.cfg.buffer_batch_size)
      # Match batch sizes by truncating to the smaller.
      b = min(x_fake.shape[0], x_real.shape[0])
      x_fake_b = x_fake[:b].requires_grad_(False)
      x_real_b = x_real[:b].requires_grad_(False)

      d_real = self.amp.discriminator(x_real_b)
      d_fake = self.amp.discriminator(x_fake_b)
      _, grad_interp = compute_gradient_penalty(
        self.amp.discriminator, x_real_b, x_fake_b
      )
      total, _, _ = discriminator_loss(
        d_real,
        d_fake,
        grad_interp,
        tanh_scale=self.amp.cfg.tanh_scale,
        grad_penalty_coef=self.amp.cfg.grad_penalty_coef,
      )
      self.amp.optimizer.zero_grad()
      total.backward()
      self.amp.optimizer.step()
      losses.append(total.item())
    return float(sum(losses) / max(1, len(losses)))


class _AMPRuntime:
  """Holds runtime AMP components built from cfg."""

  def __init__(
    self,
    cfg: AMPCfg,
    state_extractor: AMPStateExtractor,
    motion_buffer: AMPMotionBuffer,
    discriminator: AMPDiscriminator,
    optimizer: torch.optim.Optimizer,
  ) -> None:
    self.cfg = cfg
    self.state_extractor = state_extractor
    self.motion_buffer = motion_buffer
    self.discriminator = discriminator
    self.optimizer = optimizer


def _build_amp_runtime(
  amp_cfg_dict: dict, env: "VecEnv", device: torch.device
) -> _AMPRuntime:
  cfg = AMPCfg(**amp_cfg_dict)
  raw_env = getattr(env, "unwrapped", env)

  extractor_cfg = AMPStateExtractorCfg(
    asset_name="robot",
    state_fields=cfg.state_fields,
    body_fields=cfg.body_fields,
    body_names=cfg.body_names,
    root_body_name="Trunk" if cfg.body_fields else "",
  )
  from mjlab.envs import ManagerBasedRlEnv

  assert isinstance(raw_env, ManagerBasedRlEnv), (
    f"AMP runner requires ManagerBasedRlEnv (unwrapped), got {type(raw_env).__name__}"
  )
  extractor = AMPStateExtractor(extractor_cfg, raw_env, device=str(device))

  buffer_cfg = AMPMotionBufferCfg(
    motion_files=cfg.motion_files,
    state_fields=cfg.state_fields,
    body_fields=cfg.body_fields,
    body_indexes=cfg.body_indexes,
    root_body_index=cfg.root_body_index,
    device=str(device),
  )
  buffer = AMPMotionBuffer(buffer_cfg)

  if buffer.state_dim != extractor.state_dim:
    raise ValueError(
      f"AMP state dim mismatch: motion buffer={buffer.state_dim} vs "
      f"env extractor={extractor.state_dim}. Ensure state_fields and "
      f"body_fields produce identical features in both sides."
    )

  discriminator = AMPDiscriminator(
    state_dim=buffer.state_dim,
    hidden_dims=cfg.discriminator_hidden,
  ).to(device)
  optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.discriminator_lr)
  return _AMPRuntime(cfg, extractor, buffer, discriminator, optimizer)
