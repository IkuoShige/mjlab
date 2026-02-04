"""RL training configuration for Pi robot standing-up task."""

from mjlab.rl.config import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


def make_pi_standing_up_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL training config for Pi robot standing-up task.

  Returns:
    Training configuration with PPO hyperparameters matching HoST.
  """
  return RslRlOnPolicyRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=12000,
    save_interval=100,
    experiment_name="pi_standing_up",
    run_name="",
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=("standing_up", "pi"),
    obs_groups={
      "policy": ("policy",),
      "critic": ("critic",),
    },
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=0.8,
      noise_std_type="scalar",
      actor_obs_normalization=False,
      critic_obs_normalization=False,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.01,
      desired_kl=0.01,
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
    ),
  )


# Pre-built configuration for convenience.
PI_STANDING_UP_RL_CFG = make_pi_standing_up_rl_cfg()
