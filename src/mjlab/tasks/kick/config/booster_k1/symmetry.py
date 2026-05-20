"""K1 mirror symmetry operators for the kick task.

Provides the ``data_augmentation_func`` consumed by rsl-rl's symmetry hook.
It produces left/right-mirrored copies of observations and actions so the
PPO mirror loss (LVDRS Appendix D, weight=10) can constrain the policy to
bilateral behavior.

Joint mirror table for K1 — see paper appendix and the K1 MJCF: pitch joints
swap without sign flip; roll/yaw joints swap with sign flip. Mid-plane
head_yaw flips sign in place; head_pitch is invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  pass

# Joint name → (mirror joint name, sign_flip).
K1_JOINT_MIRROR_MAP: dict[str, tuple[str, bool]] = {
  # Head (mid-plane).
  "AAHead_yaw": ("AAHead_yaw", True),
  "Head_pitch": ("Head_pitch", False),
  # Arms.
  "ALeft_Shoulder_Pitch": ("ARight_Shoulder_Pitch", False),
  "Left_Shoulder_Roll": ("Right_Shoulder_Roll", True),
  "Left_Elbow_Pitch": ("Right_Elbow_Pitch", False),
  "Left_Elbow_Yaw": ("Right_Elbow_Yaw", True),
  "ARight_Shoulder_Pitch": ("ALeft_Shoulder_Pitch", False),
  "Right_Shoulder_Roll": ("Left_Shoulder_Roll", True),
  "Right_Elbow_Pitch": ("Left_Elbow_Pitch", False),
  "Right_Elbow_Yaw": ("Left_Elbow_Yaw", True),
  # Legs.
  "Left_Hip_Pitch": ("Right_Hip_Pitch", False),
  "Left_Hip_Roll": ("Right_Hip_Roll", True),
  "Left_Hip_Yaw": ("Right_Hip_Yaw", True),
  "Left_Knee_Pitch": ("Right_Knee_Pitch", False),
  "Left_Ankle_Pitch": ("Right_Ankle_Pitch", False),
  "Left_Ankle_Roll": ("Right_Ankle_Roll", True),
  "Right_Hip_Pitch": ("Left_Hip_Pitch", False),
  "Right_Hip_Roll": ("Left_Hip_Roll", True),
  "Right_Hip_Yaw": ("Left_Hip_Yaw", True),
  "Right_Knee_Pitch": ("Left_Knee_Pitch", False),
  "Right_Ankle_Pitch": ("Left_Ankle_Pitch", False),
  "Right_Ankle_Roll": ("Left_Ankle_Roll", True),
}


# Each entry maps an obs-term name to the per-element mirror multiplier.
# Length must match the term's flat-vector width. None → unsupported term.
_OBS_TERM_AXIS_FLIPS: dict[str, tuple[int, ...]] = {
  "base_ang_vel": (-1, 1, -1),
  "base_lin_vel": (1, -1, 1),
  "projected_gravity": (1, -1, 1),
  "ball_pos_b": (1, -1),
  "ball_pos_b_perceived": (1, -1),
  "ball_vel_b": (1, -1, 1),
  "kick_target_dir_b": (1, -1),
  "ball_mask": (1,),
  "root_height": (1,),
  "ball_physics": (1, 1),
}

_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")


def _build_joint_perm_and_sign(
  joint_names: list[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
  """Build permutation/sign tensors aligned to the env's joint ordering."""
  unknown = [n for n in joint_names if n not in K1_JOINT_MIRROR_MAP]
  if unknown:
    raise ValueError(f"K1_JOINT_MIRROR_MAP missing entries for joints: {unknown}")
  perm = torch.empty(len(joint_names), dtype=torch.long, device=device)
  sign = torch.empty(len(joint_names), dtype=torch.float32, device=device)
  name_to_idx = {n: i for i, n in enumerate(joint_names)}
  for i, name in enumerate(joint_names):
    mirror_name, flip = K1_JOINT_MIRROR_MAP[name]
    if mirror_name not in name_to_idx:
      raise ValueError(
        f"Mirror joint {mirror_name!r} for {name!r} not in env joint set"
      )
    perm[i] = name_to_idx[mirror_name]
    sign[i] = -1.0 if flip else 1.0
  return perm, sign


def _slice_obs_term_layout(
  group_terms: dict[str, int],
) -> list[tuple[str, int, int]]:
  """Return ``[(term_name, start, end), ...]`` for a concatenated obs group."""
  layout: list[tuple[str, int, int]] = []
  offset = 0
  for name, dim in group_terms.items():
    layout.append((name, offset, offset + dim))
    offset += dim
  return layout


class _CachedMirrorOps:
  """Holds the precomputed permutation/sign tensors for an env's joint set."""

  joint_perm: torch.Tensor
  joint_sign: torch.Tensor
  actor_layout: list[tuple[str, int, int]]
  critic_layout: list[tuple[str, int, int]]

  def __init__(
    self,
    joint_perm: torch.Tensor,
    joint_sign: torch.Tensor,
    actor_layout: list[tuple[str, int, int]],
    critic_layout: list[tuple[str, int, int]],
  ) -> None:
    self.joint_perm = joint_perm
    self.joint_sign = joint_sign
    self.actor_layout = actor_layout
    self.critic_layout = critic_layout


_CACHE_KEY = "_kick_k1_mirror_cache"


def _get_or_build_cache(env) -> _CachedMirrorOps:
  # rsl-rl hands us either the raw env or a VecEnv wrapper; unwrap if needed.
  raw_env = getattr(env, "unwrapped", env)
  cache: _CachedMirrorOps | None = getattr(raw_env, _CACHE_KEY, None)
  if cache is not None:
    return cache

  robot = raw_env.scene["robot"]
  joint_names = (
    list(robot.data.joint_names)
    if hasattr(robot.data, "joint_names")
    else list(getattr(robot, "joint_names", []))
  )
  if not joint_names:
    raise RuntimeError("Robot must expose joint_names to build mirror cache")
  device = torch.device(raw_env.device)
  joint_perm, joint_sign = _build_joint_perm_and_sign(joint_names, device)

  obs_mgr = raw_env.observation_manager
  actor_dims = _term_dims(obs_mgr, "actor")
  critic_dims = _term_dims(obs_mgr, "critic")
  actor_layout = _slice_obs_term_layout(actor_dims)
  critic_layout = _slice_obs_term_layout(critic_dims)

  cache = _CachedMirrorOps(joint_perm, joint_sign, actor_layout, critic_layout)
  setattr(raw_env, _CACHE_KEY, cache)
  return cache


def _term_dims(obs_mgr, group: str) -> dict[str, int]:
  """Return ``{term_name: flat_dim}`` for a concatenated obs group."""
  term_names: list[str] = obs_mgr.active_terms[group]
  shapes = obs_mgr.group_obs_term_dim[group]
  result: dict[str, int] = {}
  for name, shape in zip(term_names, shapes, strict=True):
    flat = 1
    for d in shape:
      flat *= int(d)
    result[name] = flat
  return result


def _mirror_concatenated_obs(
  obs: torch.Tensor,
  layout: list[tuple[str, int, int]],
  joint_perm: torch.Tensor,
  joint_sign: torch.Tensor,
) -> torch.Tensor:
  out = torch.empty_like(obs)
  for name, start, end in layout:
    chunk = obs[..., start:end]
    if name in _JOINT_TERMS:
      mirrored = chunk.index_select(-1, joint_perm) * joint_sign
    elif name in _OBS_TERM_AXIS_FLIPS:
      flips = torch.tensor(
        _OBS_TERM_AXIS_FLIPS[name], device=chunk.device, dtype=chunk.dtype
      )
      if flips.numel() != (end - start):
        raise ValueError(
          f"Term {name!r} flip pattern {tuple(flips.tolist())} has "
          f"length {flips.numel()} but slice width is {end - start}"
        )
      mirrored = chunk * flips
    else:
      mirrored = chunk
    out[..., start:end] = mirrored
  return out


def k1_data_augmentation(
  env,
  obs,
  actions: torch.Tensor | None,
):
  """rsl-rl-compatible symmetry augmentation for K1 kick task.

  Concatenates the original batch with its mirrored copy along dim 0.
  Works on either a TensorDict (multi-group obs) or a plain tensor.
  Action augmentation skipped when ``actions`` is ``None`` (occurs when
  the runner only needs obs augmentation).

  ``env`` may be either ``ManagerBasedRlEnv`` or ``RslRlVecEnvWrapper``;
  the latter is unwrapped internally.
  """
  cache = _get_or_build_cache(env)

  if obs is None:
    obs_aug = None
  elif isinstance(obs, torch.Tensor):
    obs_aug = torch.cat(
      [
        obs,
        _mirror_concatenated_obs(
          obs, cache.actor_layout, cache.joint_perm, cache.joint_sign
        ),
      ],
      dim=0,
    )
  else:
    # TensorDict-like — handle both "actor" and "critic" entries. We rebuild
    # the container so its batch_size doubles correctly along dim 0.
    from typing import cast

    from tensordict import TensorDict

    out_entries: dict = {}
    new_batch: int | None = None
    for key in obs.keys():
      grp_tensor = obs[key]
      layout = None
      if key == "actor":
        layout = cache.actor_layout
      elif key == "critic":
        layout = cache.critic_layout
      if layout is None:
        out_entries[key] = torch.cat([grp_tensor, grp_tensor], dim=0)
      else:
        mirrored = _mirror_concatenated_obs(
          grp_tensor, layout, cache.joint_perm, cache.joint_sign
        )
        out_entries[key] = torch.cat([grp_tensor, mirrored], dim=0)
      new_batch = out_entries[key].shape[0]
    assert new_batch is not None
    obs_aug = TensorDict(
      cast(TensorDict, out_entries),
      batch_size=torch.Size((new_batch,)),
    )

  if actions is None:
    actions_aug = None
  else:
    mirrored_actions = actions.index_select(-1, cache.joint_perm) * cache.joint_sign
    actions_aug = torch.cat([actions, mirrored_actions], dim=0)

  return obs_aug, actions_aug
