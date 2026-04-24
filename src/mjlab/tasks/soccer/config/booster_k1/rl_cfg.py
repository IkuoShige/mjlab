"""RL configuration for Booster K1 soccer task."""

from mjlab.rl import (
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def k1_soccer_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Feed-forward PPO (kept for non-RNN variants)."""
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
    experiment_name="k1_soccer",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


def k1_soccer_rnn_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """LSTM PPO matching HumanoidSoccer's G1FlatRecurrentPPORunnerCfg.

  Same arch as the Stage 1 multi-motion tracking runner so Stage 2 can
  resume from a Stage 1 checkpoint (both the MLP heads and the LSTM
  hidden-state weights transfer).
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
    experiment_name="k1_soccer_rnn",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


def k1_soccer_distillation_runner_cfg() -> RslRlDistillationRunnerCfg:
  return RslRlDistillationRunnerCfg(
    student=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.05,
        "std_type": "scalar",
      },
    ),
    teacher=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.05,
        "std_type": "scalar",
      },
    ),
    algorithm=RslRlDistillationAlgorithmCfg(
      num_learning_epochs=5,
      learning_rate=1.0e-3,
      gradient_length=24,
      max_grad_norm=1.0,
    ),
    obs_groups={
      "student": ("actor",),
      "teacher": ("critic",),
    },
    experiment_name="k1_soccer",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=50_000,
  )
