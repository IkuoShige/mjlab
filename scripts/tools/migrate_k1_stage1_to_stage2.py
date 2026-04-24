"""Migrate a K1 Stage 1 (MultiMotion tracking) checkpoint to the Stage 2
(Soccer Kick-RNN) obs layout.

The Stage 1 multi-motion tracking task was trained with actor obs 125 dim and
critic obs 242 dim. The Stage 2 Kick-RNN actor expects 128 dim (+3 for
target_point_pos, +3 for target_destination_pos_local, -3 for the removed
base_lin_vel); the Stage 2 critic expects 248 dim (+6 for the same ball
observations, base_lin_vel is unchanged).

The LSTM/MLP body weights are identical between stages, so we only need to
reshape the three obs-normalizer running statistics tensors and the first
LSTM layer's weight_ih_l0. Matching obs indices are copied; the new
target-point indices are zero-filled for the input weight and initialized to
the standard-normal defaults (mean=0, var=1, std=1) for the normalizer.

Example:
  uv run python scripts/tools/migrate_k1_stage1_to_stage2.py \\
    --src logs/rsl_rl/k1_tracking_rnn/2026-04-23_20-08-32/model_9999.pt \\
    --dst logs/rsl_rl/k1_soccer_rnn/migrated/model_init.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Per-term dims / offsets established via mjlab.observation_manager on both
# the Stage 1 MultiMotion env and the Stage 2 Kick-RNN env. See the script
# docstring for context.
ACTOR_INDICES_TO_COPY: list[tuple[slice, slice]] = [
  # (src, dst): command + motion_anchor_{pos,ori}_b
  (slice(0, 53), slice(0, 53)),
  # base_ang_vel + joint_{pos,vel} + actions. Source skips base_lin_vel
  # (old [53:56]); destination skips the trailing 6 target-point dims.
  (slice(56, 125), slice(53, 122)),
]
ACTOR_NEW_INDICES: slice = slice(
  122, 128
)  # target_point_pos, target_destination_pos_local
OLD_ACTOR_DIM = 125
NEW_ACTOR_DIM = 128

# Critic keeps base_lin_vel in both stages, so only the trailing 6 dims are new.
CRITIC_INDICES_TO_COPY: list[tuple[slice, slice]] = [
  (slice(0, 242), slice(0, 242)),
]
CRITIC_NEW_INDICES: slice = slice(242, 248)
OLD_CRITIC_DIM = 242
NEW_CRITIC_DIM = 248


def _migrate_normalizer(
  state_dict: dict[str, torch.Tensor],
  copy_slices: list[tuple[slice, slice]],
  old_dim: int,
  new_dim: int,
) -> None:
  """Reshape obs_normalizer._{mean,var,std} from old_dim to new_dim."""
  defaults = {"_mean": 0.0, "_var": 1.0, "_std": 1.0}
  for field, default in defaults.items():
    key = f"obs_normalizer.{field}"
    old = state_dict[key]
    assert old.shape == (1, old_dim), (
      f"{key}: expected (1, {old_dim}), got {tuple(old.shape)}"
    )
    new = torch.full((1, new_dim), default, dtype=old.dtype, device=old.device)
    for src_slice, dst_slice in copy_slices:
      new[:, dst_slice] = old[:, src_slice]
    state_dict[key] = new


def _migrate_rnn_input(
  state_dict: dict[str, torch.Tensor],
  copy_slices: list[tuple[slice, slice]],
  old_dim: int,
  new_dim: int,
) -> None:
  """Reshape rnn.rnn.weight_ih_l0 from (..., old_dim) to (..., new_dim).

  New observation columns are zero-initialized so the fresh target-point
  inputs contribute zero to the LSTM input gate at step 0; the optimizer will
  pick up useful gradients once training resumes.
  """
  key = "rnn.rnn.weight_ih_l0"
  old = state_dict[key]
  assert old.shape[1] == old_dim, (
    f"{key}: expected (.., {old_dim}), got {tuple(old.shape)}"
  )
  new = torch.zeros((old.shape[0], new_dim), dtype=old.dtype, device=old.device)
  for src_slice, dst_slice in copy_slices:
    new[:, dst_slice] = old[:, src_slice]
  state_dict[key] = new


def _migrate_optimizer_state(
  opt_state: dict,
  actor_copy_slices: list[tuple[slice, slice]],
  critic_copy_slices: list[tuple[slice, slice]],
) -> None:
  """Reshape Adam moments (exp_avg, exp_avg_sq) for the input-weight tensors.

  The optimizer stores per-parameter moment buffers; the two parameters whose
  second dim is the old obs dim are the actor's and critic's
  ``rnn.rnn.weight_ih_l0``. We reshape their moments the same way as the
  weights themselves, so the first optimizer step after resume sees a
  consistent (param shape == moment shape) pair.
  """
  state = opt_state.get("state", {})
  for k, entry in state.items():
    if not isinstance(entry, dict):
      continue
    for field in ("exp_avg", "exp_avg_sq"):
      buf = entry.get(field)
      if buf is None or not hasattr(buf, "shape") or buf.ndim != 2:
        continue
      if buf.shape[1] == OLD_ACTOR_DIM:
        new = torch.zeros(
          (buf.shape[0], NEW_ACTOR_DIM), dtype=buf.dtype, device=buf.device
        )
        for src_slice, dst_slice in actor_copy_slices:
          new[:, dst_slice] = buf[:, src_slice]
        entry[field] = new
      elif buf.shape[1] == OLD_CRITIC_DIM:
        new = torch.zeros(
          (buf.shape[0], NEW_CRITIC_DIM), dtype=buf.dtype, device=buf.device
        )
        for src_slice, dst_slice in critic_copy_slices:
          new[:, dst_slice] = buf[:, src_slice]
        entry[field] = new


def migrate(src_path: Path, dst_path: Path) -> None:
  ckpt = torch.load(src_path, weights_only=False, map_location="cpu")

  actor = ckpt["actor_state_dict"]
  critic = ckpt["critic_state_dict"]

  _migrate_normalizer(actor, ACTOR_INDICES_TO_COPY, OLD_ACTOR_DIM, NEW_ACTOR_DIM)
  _migrate_rnn_input(actor, ACTOR_INDICES_TO_COPY, OLD_ACTOR_DIM, NEW_ACTOR_DIM)

  _migrate_normalizer(critic, CRITIC_INDICES_TO_COPY, OLD_CRITIC_DIM, NEW_CRITIC_DIM)
  _migrate_rnn_input(critic, CRITIC_INDICES_TO_COPY, OLD_CRITIC_DIM, NEW_CRITIC_DIM)

  if "optimizer_state_dict" in ckpt:
    _migrate_optimizer_state(
      ckpt["optimizer_state_dict"],
      ACTOR_INDICES_TO_COPY,
      CRITIC_INDICES_TO_COPY,
    )

  # Reset iter so the resumed run counts from 0 in the new experiment dir.
  ckpt["iter"] = 0

  dst_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(ckpt, dst_path)
  print(f"[migrate] wrote {dst_path}")
  print(f"  actor obs_normalizer._mean: {actor['obs_normalizer._mean'].shape}")
  print(f"  actor rnn.rnn.weight_ih_l0: {actor['rnn.rnn.weight_ih_l0'].shape}")
  print(f"  critic obs_normalizer._mean: {critic['obs_normalizer._mean'].shape}")
  print(f"  critic rnn.rnn.weight_ih_l0: {critic['rnn.rnn.weight_ih_l0'].shape}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--src", type=Path, required=True)
  parser.add_argument("--dst", type=Path, required=True)
  args = parser.parse_args()
  migrate(args.src, args.dst)


if __name__ == "__main__":
  main()
