"""Per-quadrant kick success diagnosis for the K1 kick policy.

Loads a trained kick policy and runs N episodes per (distance × angle) bin
to identify where the 53% failure mode comes from. Output: a 2D success
grid printed to stdout.

Usage:
  uv run python scripts/tools/eval_kick_quadrants.py \\
      --checkpoint logs/rsl_rl/k1_kick/<run>/model_<n>.pt \\
      --episodes-per-bin 32
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MJLAB_DISABLE_CUDNN", "1")

import torch

import mjlab.tasks  # noqa: F401  - registers tasks.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.kick.mdp.commands import KickTargetCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

# Bins: distance and signed-angle (radians, 0 = front).
DIST_BINS: list[tuple[float, str]] = [
  (0.10, "very-close"),
  (0.25, "close"),
  (0.40, "mid"),
]
ANGLE_BINS: list[tuple[float, str]] = [
  (0.0, "front"),
  (math.pi / 4, "front-right"),
  (math.pi / 2, "right"),
  (3 * math.pi / 4, "back-right"),
  (math.pi, "back"),
  (-3 * math.pi / 4, "back-left"),
  (-math.pi / 2, "left"),
  (-math.pi / 4, "front-left"),
]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--task",
    default="Mjlab-Kick-AMP-Flat-Booster-K1",
    help="Task ID to load env/runner config (kick or kick-amp).",
  )
  parser.add_argument("--episodes-per-bin", type=int, default=24)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--num-envs",
    type=int,
    default=24,
    help="Parallel envs (1 env per episode in a bin).",
  )
  args = parser.parse_args()

  configure_torch_backends()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs

  agent_cfg = load_rl_cfg(args.task)
  if hasattr(agent_cfg, "amp_cfg") and agent_cfg.amp_cfg is not None:
    agent_cfg.amp_cfg = None  # disable AMP at eval time

  env_raw = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device=args.device)
  runner.load(str(args.checkpoint))
  policy = runner.alg.get_policy()
  policy.eval()

  cmd = env_raw.command_manager.get_term("kick_target")
  assert isinstance(cmd, KickTargetCommand)

  # Replace random resample with controlled placement: set a fixed
  # (distance, angle) before each reset.
  results: dict[tuple[str, str], dict[str, int]] = {}

  for dist, dist_label in DIST_BINS:
    for angle, angle_label in ANGLE_BINS:
      success_total = 0
      contacted_total = 0
      n_eps = 0
      eps_target = args.episodes_per_bin
      while n_eps < eps_target:
        obs, _ = env.reset()
        # Override ball spawn to the bin's (distance, angle) for all envs.
        with torch.no_grad():
          _force_spawn(cmd, dist, angle)

        ep_kicked = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        max_steps = int(env_raw.max_episode_length)
        with torch.inference_mode():
          for _ in range(max_steps):
            action = policy(obs)
            obs, _, dones, _ = env.step(action)
            ep_kicked = ep_kicked | cmd.kick_contact_awarded
            if dones.all():
              break

        batch_size = min(args.num_envs, eps_target - n_eps)
        contacted_total += int(ep_kicked[:batch_size].sum().item())
        success_total += int(ep_kicked[:batch_size].sum().item())
        n_eps += batch_size

      key = (dist_label, angle_label)
      results[key] = {
        "success": success_total,
        "contact": contacted_total,
        "total": n_eps,
      }
      print(
        f"  [{dist_label:>10}] [{angle_label:>11}] "
        f"contact={contacted_total}/{n_eps} "
        f"({100.0 * contacted_total / n_eps:.0f}%)"
      )

  print("\n=== Summary grid (% episodes with valid kick contact) ===")
  header = "dist \\ angle    " + "  ".join(f"{lbl:>11}" for _, lbl in ANGLE_BINS)
  print(header)
  for _, dist_label in DIST_BINS:
    row = [f"{dist_label:<15}"]
    for _, angle_label in ANGLE_BINS:
      r = results[(dist_label, angle_label)]
      pct = 100.0 * r["contact"] / r["total"]
      row.append(f"{pct:>10.0f}%")
    print("  ".join(row))

  env.close()


def _force_spawn(cmd: KickTargetCommand, distance: float, angle: float) -> None:
  """Override the just-resampled ball position to the controlled bin."""
  device = cmd.device
  n = cmd.num_envs
  # The command's _resample_command runs during env.reset; we override
  # AFTER, so all envs in this batch get identical (distance, angle).
  dist_t = torch.full((n,), distance, device=device)
  ang_t = torch.full((n,), angle, device=device)

  from mjlab.utils.lab_api.math import quat_apply, yaw_quat

  robot_pos = cmd.robot.data.root_link_pos_w
  yaw_only = yaw_quat(cmd.robot.data.root_link_quat_w)
  local = torch.zeros(n, 3, device=device)
  local[:, 0] = dist_t * torch.cos(ang_t)
  local[:, 1] = dist_t * torch.sin(ang_t)
  world = quat_apply(yaw_only, local)

  ball_pos = torch.zeros(n, 3, device=device)
  ball_pos[:, :2] = robot_pos[:, :2] + world[:, :2]
  ball_pos[:, 2] = cmd.cfg.ball_spawn_height

  state = torch.zeros(n, 13, device=device)
  state[:, :3] = ball_pos
  state[:, 3] = 1.0
  cmd.ball.write_root_state_to_sim(state)
  cmd._ball_pos_w = ball_pos
  cmd._ball_vel_w = torch.zeros(n, 3, device=device)
  cmd._target_dir_w = torch.stack([torch.cos(ang_t), torch.sin(ang_t)], dim=-1)
  cmd._initial_ball_progress = (
    ball_pos[:, 0] * cmd._target_dir_w[:, 0] + ball_pos[:, 1] * cmd._target_dir_w[:, 1]
  )
  cmd._prev_ball_progress = cmd._initial_ball_progress.clone()
  cmd._prev_robot_ball_distance = (robot_pos[:, :2] - ball_pos[:, :2]).norm(dim=-1)
  cmd.kick_contact_awarded.zero_()
  cmd.kick_contact_new.zero_()
  cmd.peak_kick_speed.zero_()
  if cmd.perception is not None:
    cmd.perception.reset(torch.arange(n, device=device, dtype=torch.long))


if __name__ == "__main__":
  main()
