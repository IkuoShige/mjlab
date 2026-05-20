"""Verify K1 mirror data augmentation is idempotent."""

from __future__ import annotations

import os

os.environ.setdefault("MJLAB_DISABLE_CUDNN", "1")

import pytest
import torch
from tensordict import TensorDict


@pytest.fixture(scope="module")
def env():
  import mjlab.tasks  # noqa: F401 — registers tasks
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-Kick-Flat-Booster-K1")
  cfg.scene.num_envs = 2
  e = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  yield e
  e.close()


def test_double_mirror_is_identity(env) -> None:
  from mjlab.tasks.kick.config.booster_k1.symmetry import k1_data_augmentation

  obs, _ = env.reset()
  actor = obs["actor"]
  critic = obs["critic"]
  td = TensorDict({"actor": actor, "critic": critic}, batch_size=[actor.shape[0]])
  actions = torch.randn(actor.shape[0], 22)

  obs_aug, actions_aug = k1_data_augmentation(env, td, actions)
  assert obs_aug.batch_size[0] == 2 * actor.shape[0]
  assert actions_aug.shape[0] == 2 * actor.shape[0]

  mirrored_obs = TensorDict(
    {
      "actor": obs_aug["actor"][actor.shape[0] :],
      "critic": obs_aug["critic"][actor.shape[0] :],
    },
    batch_size=[actor.shape[0]],
  )
  mirrored_actions = actions_aug[actions.shape[0] :]

  obs_dm, actions_dm = k1_data_augmentation(env, mirrored_obs, mirrored_actions)
  dm_actions = actions_dm[actions.shape[0] :]
  dm_actor = obs_dm["actor"][actor.shape[0] :]
  dm_critic = obs_dm["critic"][actor.shape[0] :]

  assert torch.allclose(dm_actions, actions, atol=1e-6)
  assert torch.allclose(dm_actor, actor, atol=1e-6)
  assert torch.allclose(dm_critic, critic, atol=1e-6)
