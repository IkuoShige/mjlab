"""RL runner configuration for Booster K1 LVDRS-style kick task."""

from __future__ import annotations

import glob

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.kick.config.booster_k1.symmetry import k1_data_augmentation

_K1_KICK_MOTIONS_DIR = "motions/soccer-standard-mj-k1"


def k1_kick_ppo_runner_cfg(
  mirror_loss_coeff: float = 10.0,
  use_data_augmentation: bool = True,
  use_mirror_loss: bool = True,
) -> RslRlOnPolicyRunnerCfg:
  """Feed-forward PPO with rsl-rl built-in mirror symmetry loss.

  Mirror symmetry follows LVDRS Appendix D — coefficient defaults to 10
  to encourage bilateral kicking and prevent unilateral specialization.

  AMP discriminator is NOT enabled in this baseline runner.
  Use ``k1_kick_amp_ppo_runner_cfg`` for the AMP variant.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.995,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      symmetry_cfg={
        "use_data_augmentation": use_data_augmentation,
        "use_mirror_loss": use_mirror_loss,
        "mirror_loss_coeff": mirror_loss_coeff,
        "data_augmentation_func": k1_data_augmentation,
      },
    ),
    experiment_name="k1_kick",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=20_000,
  )


def _discover_k1_kick_motions() -> list[str]:
  return sorted(glob.glob(f"{_K1_KICK_MOTIONS_DIR}/*.npz"))


def k1_kick_amp_ppo_runner_cfg(
  mirror_loss_coeff: float = 10.0,
  use_data_augmentation: bool = True,
  use_mirror_loss: bool = True,
  style_reward_coef: float = 0.3,
) -> RslRlOnPolicyRunnerCfg:
  """PPO + AMP discriminator + mirror symmetry (LVDRS-style).

  Adds an AMP style reward (WGAN discriminator over (s_t, s_{t+1}) joint
  state transitions) on top of the base PPO with mirror loss. Uses the
  10 K1-retargeted kick reference clips.
  """
  cfg = k1_kick_ppo_runner_cfg(
    mirror_loss_coeff=mirror_loss_coeff,
    use_data_augmentation=use_data_augmentation,
    use_mirror_loss=use_mirror_loss,
  )
  # Share experiment_name with baseline so we can warm-start from baseline
  # checkpoints under logs/rsl_rl/k1_kick/.
  cfg.experiment_name = "k1_kick"
  cfg.amp_cfg = {
    "motion_files": _discover_k1_kick_motions(),
    "state_fields": ("joint_pos", "joint_vel"),
    "body_fields": (),
    "body_names": (),
    "body_indexes": (),
    "root_body_index": 0,
    "discriminator_hidden": (1024, 128),
    "buffer_batch_size": 4096,
    "policy_batch_size": 4096,
    "tanh_scale": 0.4,
    "grad_penalty_coef": 50.0,
    "style_reward_coef": style_reward_coef,
    "discriminator_lr": 1e-4,
    "discriminator_update_steps": 1,
    "device": "cuda:0",
  }
  return cfg
