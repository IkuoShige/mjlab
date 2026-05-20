"""Quantitative evaluation of the K1 kick policy's walking phase.

Plays one motion deterministically and reports foot-stride amplitude
(policy vs reference) plus trunk forward speed during the approach.
A "shuffle" failure mode shows up as small policy stride relative to
reference (e.g., policy ~10cm vs reference ~30cm) at the same trunk
forward speed.

Usage:
  uv run python scripts/tools/eval_k1_kick_walk.py \\
    --task Mjlab-Soccer-Kick-RNN-Flat-Booster-K1 \\
    --checkpoint logs/.../model_37500.pt \\
    --motion-idx 0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

# IMPORTANT: set env vars before any torch/cuda imports.
os.environ.setdefault("MJLAB_DISABLE_CUDNN", "1")

import numpy as np
import torch

import mjlab.tasks  # noqa: F401  - registers tasks.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.soccer.mdp.commands import SoccerMotionCommand
from mjlab.utils.torch import configure_torch_backends


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--motion-idx",
    type=int,
    default=0,
    help="Index of motion file to evaluate (0 = first clip).",
  )
  args = parser.parse_args()

  configure_torch_backends()

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = 1

  env_raw = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device="cuda:0")
  runner.load(str(args.checkpoint))
  policy = runner.alg.get_policy()
  policy.eval()

  # Pin to one motion clip + start from frame 0.
  motion_cmd = env_raw.command_manager.get_term("motion")
  assert isinstance(motion_cmd, SoccerMotionCommand)
  motion_cmd.motion_idx[:] = args.motion_idx
  motion_cmd.motion_length[:] = motion_cmd.motion.file_lengths[args.motion_idx]
  motion_cmd.time_steps[:] = 0

  motion_name = motion_cmd.motion.motion_name[args.motion_idx]
  T = int(motion_cmd.motion.file_lengths[args.motion_idx].item())
  fps = motion_cmd.motion.fps
  print(f"motion: {motion_name}  T={T}  fps={fps}")

  # K1 spec body indices for foot / trunk in the motion's body_pos_w
  # tensor (which is in spec order minus 'world').
  trunk_idx = 0  # Trunk
  lfoot_idx = 16  # left_foot_link
  rfoot_idx = 22  # right_foot_link

  obs = env.get_observations()

  records = []
  with torch.inference_mode():
    for _step in range(T):
      action = policy(obs)
      obs, _, _, _ = env.step(action)

      env_origins = env_raw.scene.env_origins[0].cpu().numpy()
      mi = int(motion_cmd.motion_idx[0].item())
      ti = int(motion_cmd.time_steps[0].item())
      ti = min(ti, T - 1)
      ref_trunk = motion_cmd.motion._body_pos_w[mi, ti, trunk_idx].cpu().numpy()
      ref_lfoot = motion_cmd.motion._body_pos_w[mi, ti, lfoot_idx].cpu().numpy()
      ref_rfoot = motion_cmd.motion._body_pos_w[mi, ti, rfoot_idx].cpu().numpy()

      robot = env_raw.scene["robot"]
      robot_trunk = (
        robot.data.body_link_pos_w[0, motion_cmd.robot_anchor_body_index].cpu().numpy()
      )
      lfoot_body_idx = robot.body_names.index("left_foot_link")
      rfoot_body_idx = robot.body_names.index("right_foot_link")
      robot_lfoot = robot.data.body_link_pos_w[0, lfoot_body_idx].cpu().numpy()
      robot_rfoot = robot.data.body_link_pos_w[0, rfoot_body_idx].cpu().numpy()

      records.append(
        {
          "ref_trunk": ref_trunk + env_origins,
          "ref_lfoot": ref_lfoot + env_origins,
          "ref_rfoot": ref_rfoot + env_origins,
          "policy_trunk": robot_trunk,
          "policy_lfoot": robot_lfoot,
          "policy_rfoot": robot_rfoot,
        }
      )

  ref_t = np.array([r["ref_trunk"] for r in records])
  ref_l = np.array([r["ref_lfoot"] for r in records])
  ref_r = np.array([r["ref_rfoot"] for r in records])
  pol_t = np.array([r["policy_trunk"] for r in records])
  pol_l = np.array([r["policy_lfoot"] for r in records])
  pol_r = np.array([r["policy_rfoot"] for r in records])

  print("\n=== Trunk forward distance (X span) ===")
  print(f"  reference: {ref_t[:, 0].max() - ref_t[:, 0].min():+.3f} m")
  print(f"  policy:    {pol_t[:, 0].max() - pol_t[:, 0].min():+.3f} m")

  # Stride amplitude: span of foot-X relative to trunk-X.
  ref_l_rel = ref_l[:, 0] - ref_t[:, 0]
  ref_r_rel = ref_r[:, 0] - ref_t[:, 0]
  pol_l_rel = pol_l[:, 0] - pol_t[:, 0]
  pol_r_rel = pol_r[:, 0] - pol_t[:, 0]
  print("\n=== Foot-X stride amplitude (foot.x − trunk.x range) ===")
  print(
    f"  reference left:  {ref_l_rel.min():+.3f}..{ref_l_rel.max():+.3f}  "
    f"(span {ref_l_rel.max() - ref_l_rel.min():.3f})"
  )
  print(
    f"  policy   left:   {pol_l_rel.min():+.3f}..{pol_l_rel.max():+.3f}  "
    f"(span {pol_l_rel.max() - pol_l_rel.min():.3f})"
  )
  print(
    f"  reference right: {ref_r_rel.min():+.3f}..{ref_r_rel.max():+.3f}  "
    f"(span {ref_r_rel.max() - ref_r_rel.min():.3f})"
  )
  print(
    f"  policy   right:  {pol_r_rel.min():+.3f}..{pol_r_rel.max():+.3f}  "
    f"(span {pol_r_rel.max() - pol_r_rel.min():.3f})"
  )

  # Foot Z range (lift amplitude).
  print("\n=== Foot-Z lift amplitude (world Z range) ===")
  print(
    f"  reference left:  {ref_l[:, 2].max() - ref_l[:, 2].min():.3f} m  "
    f"(peak {ref_l[:, 2].max():.3f})"
  )
  print(
    f"  policy   left:   {pol_l[:, 2].max() - pol_l[:, 2].min():.3f} m  "
    f"(peak {pol_l[:, 2].max():.3f})"
  )
  print(
    f"  reference right: {ref_r[:, 2].max() - ref_r[:, 2].min():.3f} m  "
    f"(peak {ref_r[:, 2].max():.3f})"
  )
  print(
    f"  policy   right:  {pol_r[:, 2].max() - pol_r[:, 2].min():.3f} m  "
    f"(peak {pol_r[:, 2].max():.3f})"
  )

  # Per-frame foot-X delta (trace of the walking phase). Print first 80 frames.
  print("\n=== Frames 0-80: foot X relative to trunk ===")
  print("frame  ref_l_x  pol_l_x  ref_r_x  pol_r_x")
  for i in range(0, min(80, T), 4):
    print(
      f"  {i:3d}  {ref_l_rel[i]:+.3f}  {pol_l_rel[i]:+.3f}   "
      f"{ref_r_rel[i]:+.3f}  {pol_r_rel[i]:+.3f}"
    )

  env.close()


if __name__ == "__main__":
  main()
