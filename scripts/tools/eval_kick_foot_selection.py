"""Diagnose foot selection: does the policy kick with the RIGHT foot when
the ball spawns on the right side, and the LEFT foot when it spawns on the
left? This is a basic soccer-mechanics sanity check separate from
direction accuracy.

For each ball spawn angle in the sweep:
  1. Force ball spawn at that angle relative to robot forward.
  2. Force kick target direction = 0 (straight ahead).
  3. Run N episodes, record which foot first kicked (= foot closest to
     the ball at ``kick_contact_new``).
  4. Compare to the "expected" foot (right foot for positive spawn
     angles, left foot for negative; 0° = ambiguous, expect 50/50 split
     after the argmin tiebreak).

Usage:
  uv run --extra cu124 python scripts/tools/eval_kick_foot_selection.py \\
      --checkpoint <ckpt> \\
      --ball-angle-sweep="-45,-20,0,20,45" \\
      --episodes 4 --num-envs 64
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MJLAB_DISABLE_CUDNN", "1")

import torch

import mjlab.tasks  # noqa: F401 - registers tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.kick.mdp.commands import KickTargetCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

_FOOT_NAMES = ("left_foot_link", "right_foot_link")


def _force_spawn(cmd: KickTargetCommand, ball_angle_rad: float) -> None:
  """Lock ball spawn angle and kick target direction (=0, straight ahead)."""
  cmd.cfg.ball_spawn_angle_range = (ball_angle_rad, ball_angle_rad)
  cmd.cfg.ball_spawn_distance_range = (0.30, 0.30)
  cmd.cfg.target_dir_range = (0.0, 0.0)
  cmd.kick_contact_new.zero_()


def _run_angle(
  env: RslRlVecEnvWrapper,
  env_raw: ManagerBasedRlEnv,
  policy,
  cmd: KickTargetCommand,
  device: str,
  ball_angle_rad: float,
  episodes: int,
  num_envs: int,
) -> dict:
  robot = env_raw.scene["robot"]
  body_names = list(robot.body_names)
  foot_idx = torch.tensor(
    [body_names.index(n) for n in _FOOT_NAMES], dtype=torch.long, device=device
  )

  kicked_left = 0
  kicked_right = 0
  n_done = 0

  while n_done < episodes:
    with torch.inference_mode():
      obs, _ = env.reset()
      _force_spawn(cmd, ball_angle_rad)
      first_kick_foot = torch.full((num_envs,), -1, dtype=torch.long, device=device)
      max_steps = int(env_raw.max_episode_length)
      for _ in range(max_steps):
        action = policy(obs)
        obs, _, dones, _ = env.step(action)
        new_kick = cmd.kick_contact_new
        if new_kick.any():
          foot_pos = robot.data.body_link_pos_w[:, foot_idx]
          ball = cmd.ball_pos_w.unsqueeze(1)
          dist = (foot_pos - ball).norm(dim=-1)
          kf = dist.argmin(dim=-1)
          first_event = new_kick & (first_kick_foot < 0)
          first_kick_foot[first_event] = kf[first_event]
        if dones.all().item():
          break

    batch = min(num_envs, episodes - n_done)
    for i in range(batch):
      kf = int(first_kick_foot[i].item())
      if kf == 0:
        kicked_left += 1
      elif kf == 1:
        kicked_right += 1
    n_done += batch

  total = kicked_left + kicked_right
  return {
    "left_pct": 100.0 * kicked_left / total if total else 0.0,
    "right_pct": 100.0 * kicked_right / total if total else 0.0,
    "total_kicks": total,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--task", default="Mjlab-Kick-Flat-Booster-K1")
  parser.add_argument(
    "--ball-angle-sweep",
    type=str,
    default="-45,-20,0,20,45",
    help="Comma-separated ball spawn angles in degrees, relative to robot forward.",
  )
  parser.add_argument("--episodes", type=int, default=4)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--device", default="cuda")
  args = parser.parse_args()

  configure_torch_backends()
  device = args.device

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.episode_length_s = 5.0
  agent_cfg = load_rl_cfg(args.task)
  if hasattr(agent_cfg, "amp_cfg") and agent_cfg.amp_cfg is not None:
    agent_cfg.amp_cfg = None

  env_raw = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(
    str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.alg.get_policy().to(device).eval()

  cmd = env_raw.command_manager.get_term("kick_target")
  assert isinstance(cmd, KickTargetCommand)

  angles_deg = [float(a) for a in args.ball_angle_sweep.split(",")]
  print(f"\nFoot-selection sweep: ball at {angles_deg}° relative to robot forward.")
  print(
    "Target heading is 0° (straight) for every test — only ball position changes.\n"
  )
  print(
    f"{'ball_angle':>10} | {'L %':>6} | {'R %':>6} | {'total':>6} | {'expected':>10}"
  )
  print("-" * 56)
  for a_deg in angles_deg:
    a_rad = math.radians(a_deg)
    res = _run_angle(
      env=env,
      env_raw=env_raw,
      policy=policy,
      cmd=cmd,
      device=device,
      ball_angle_rad=a_rad,
      episodes=args.episodes,
      num_envs=args.num_envs,
    )
    if a_deg > 1:
      expected = "RIGHT"
    elif a_deg < -1:
      expected = "LEFT"
    else:
      expected = "either"
    print(
      f"{a_deg:>+10.1f} | {res['left_pct']:>5.1f}% | {res['right_pct']:>5.1f}% |"
      f" {res['total_kicks']:>6d} | {expected:>10}"
    )


if __name__ == "__main__":
  main()
