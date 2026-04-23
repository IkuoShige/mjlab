"""RL configuration for Booster K1 tracking task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def booster_k1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Feed-forward PPO config for single-motion K1 tracking (BeyondMimic-style)."""
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
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="k1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def booster_k1_multimotion_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Recurrent PPO config for multi-motion Stage 1 tracking.

  Mirrors HumanoidSoccer's G1FlatRecurrentPPORunnerCfg used by
  Tracking-Terrain-G1-RNN-v0: smaller MLP heads (128, 64, 32) plus a 2-layer
  128-dim LSTM. The hidden state captures temporal context so the policy can
  anticipate fast swing phases that a pure MLP collapses to an average.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 64, 32),
      activation="elu",
      obs_normalization=True,
      class_name="RNNModel",
      rnn_type="lstm",
      rnn_hidden_dim=128,
      rnn_num_layers=2,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 64, 32),
      activation="elu",
      obs_normalization=True,
      class_name="RNNModel",
      rnn_type="lstm",
      rnn_hidden_dim=128,
      rnn_num_layers=2,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="k1_tracking_rnn",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
