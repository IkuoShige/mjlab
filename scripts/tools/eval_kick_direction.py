"""Evaluate kick direction accuracy for the K1 kick policy.

Forces the target direction to a fixed heading (default = forward, +x in
body frame) for all envs, runs N episodes, and measures the angular
error between the post-kick ball velocity and the commanded direction.

Usage:
  uv run python scripts/tools/eval_kick_direction.py \\
      --checkpoint logs/rsl_rl/k1_kick/<run>/model_<n>.pt \\
      --episodes 64

  # Evaluate a different commanded heading (radians, body frame +x = 0):
  uv run python scripts/tools/eval_kick_direction.py \\
      --checkpoint <ckpt> --target-heading-rad 1.5708  # +90° (left)
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


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--task", default="Mjlab-Kick-Flat-Booster-K1")
  parser.add_argument(
    "--target-heading-rad",
    type=float,
    default=0.0,
    help="Commanded kick heading in body frame (0 = forward, π/2 = left).",
  )
  parser.add_argument(
    "--ball-distance",
    type=float,
    default=0.30,
    help="Fixed ball spawn distance from robot (m).",
  )
  parser.add_argument(
    "--ball-angle-rad",
    type=float,
    default=0.0,
    help="Fixed ball spawn angle in body frame (0 = directly in front).",
  )
  parser.add_argument("--episodes", type=int, default=32)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--measure-window",
    type=int,
    default=10,
    help="Steps post-kick over which to measure peak ball velocity.",
  )
  parser.add_argument(
    "--heading-sweep",
    type=str,
    default=None,
    help=(
      "Comma-separated heading values in DEGREES to sweep instead of running"
      " a single heading. Example: '-90,-45,0,45,90'. Runs --episodes per"
      " heading and prints a per-heading accuracy matrix."
    ),
  )
  args = parser.parse_args()

  configure_torch_backends()
  device = args.device

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  # play=True sets episode_length_s = 1e9 (infinite for viewer use). The
  # eval needs finite episodes that terminate on kick/fall/time_out so we
  # can advance to the next batch — restore the training-time cap.
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

  if args.heading_sweep:
    headings_deg = [float(h.strip()) for h in args.heading_sweep.split(",")]
    print(f"Sweep mode: {len(headings_deg)} headings × {args.episodes} episodes each.")
    results = []
    for h_deg in headings_deg:
      h_rad = math.radians(h_deg)
      print(f"\n--- heading {h_deg:+.0f}° ---")
      r = _run_heading(
        env=env,
        env_raw=env_raw,
        policy=policy,
        cmd=cmd,
        device=device,
        target_heading_rad=h_rad,
        ball_distance=args.ball_distance,
        ball_angle_rad=args.ball_angle_rad,
        episodes=args.episodes,
        num_envs=args.num_envs,
        measure_window=args.measure_window,
      )
      results.append((h_deg, r))
    _print_sweep_summary(args, results)
  else:
    r = _run_heading(
      env=env,
      env_raw=env_raw,
      policy=policy,
      cmd=cmd,
      device=device,
      target_heading_rad=args.target_heading_rad,
      ball_distance=args.ball_distance,
      ball_angle_rad=args.ball_angle_rad,
      episodes=args.episodes,
      num_envs=args.num_envs,
      measure_window=args.measure_window,
    )
    _print_single_summary(args, r)

  env.close()


def _run_heading(
  *,
  env,
  env_raw,
  policy,
  cmd,
  device,
  target_heading_rad: float,
  ball_distance: float,
  ball_angle_rad: float,
  episodes: int,
  num_envs: int,
  measure_window: int,
) -> dict:
  """Run ``episodes`` evals at a single commanded heading; return stats dict."""
  import time as _time

  errors_deg: list[float] = []
  speeds: list[float] = []
  successes = 0
  n_done = 0
  print(
    f"Running {episodes} episodes in batches of {num_envs}"
    f" (max {int(env_raw.max_episode_length)} steps/episode)...",
    flush=True,
  )
  while n_done < episodes:
    batch_t0 = _time.time()
    with torch.inference_mode():
      obs, _ = env.reset()
      _force_spawn(cmd, ball_distance, ball_angle_rad, target_heading_rad)

      peak_speed = torch.zeros(num_envs, device=device)
      best_dir = torch.zeros(num_envs, 2, device=device)
      kicked = torch.zeros(num_envs, dtype=torch.bool, device=device)
      measure_steps_left = torch.zeros(num_envs, dtype=torch.long, device=device)

      max_steps = int(env_raw.max_episode_length)
      for _ in range(max_steps):
        action = policy(obs)
        obs, _, dones, _ = env.step(action)

        new_kick = cmd.kick_contact_new
        if new_kick.any():
          measure_steps_left[new_kick] = measure_window
          kicked = kicked | new_kick

        active = measure_steps_left > 0
        if active.any():
          ball_vel_xy = cmd.ball_vel_w[:, :2]
          spd = ball_vel_xy.norm(dim=-1)
          improve = active & (spd > peak_speed)
          peak_speed[improve] = spd[improve]
          best_dir[improve] = ball_vel_xy[improve]
          measure_steps_left[active] -= 1

        if dones.all().item():
          break

    batch = min(num_envs, episodes - n_done)
    for i in range(batch):
      if not kicked[i].item():
        continue
      cmd_dir = cmd.kick_target_dir_w[i].cpu().numpy()
      actual = best_dir[i].cpu().numpy()
      actual_n = actual / max(float(peak_speed[i].item()), 1e-6)
      dot = float(actual_n[0] * cmd_dir[0] + actual_n[1] * cmd_dir[1])
      ang_err_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
      errors_deg.append(ang_err_deg)
      speeds.append(float(peak_speed[i].item()))
      if ang_err_deg < 10.0:
        successes += 1
    n_done += batch
    batch_dt = _time.time() - batch_t0
    kick_rate = len([1 for k in kicked[:batch].cpu().tolist() if k]) / max(1, batch)
    running_err = (
      "n/a" if not errors_deg else f"{sum(errors_deg) / len(errors_deg):.1f}°"
    )
    print(
      f"  [{n_done:>4}/{episodes}] batch_t={batch_dt:.1f}s"
      f"  batch_kick_rate={100 * kick_rate:.0f}%"
      f"  running mean err={running_err}",
      flush=True,
    )
  return {
    "errors_deg": errors_deg,
    "speeds": speeds,
    "successes": successes,
    "n_done": n_done,
  }


def _print_single_summary(args, r: dict) -> None:
  errors_deg = r["errors_deg"]
  print()
  print("=== Kick direction accuracy ===")
  print(f"Task: {args.task}")
  print(f"Checkpoint: {args.checkpoint}")
  print(
    f"Commanded heading (body frame): {math.degrees(args.target_heading_rad):+.1f}°"
  )
  print(
    f"Ball spawn: distance={args.ball_distance:.2f}m,"
    f" angle={math.degrees(args.ball_angle_rad):+.1f}°"
  )
  print(f"Episodes: {r['n_done']} (kicks registered: {len(errors_deg)})")
  if not errors_deg:
    print("No kicks registered.")
    return
  e = torch.tensor(errors_deg)
  s = torch.tensor(r["speeds"])
  print()
  print(f"Angular error: mean={e.mean():.1f}°  std={e.std():.1f}°")
  print(f"               median={e.median():.1f}°")
  print(
    f"               p25={e.kthvalue(int(len(e) * 0.25) + 1).values:.1f}°"
    f"  p75={e.kthvalue(int(len(e) * 0.75) + 1).values:.1f}°"
  )
  print(f"Ball speed:    mean={s.mean():.2f} m/s  std={s.std():.2f} m/s")
  print(
    f"Success (<10° err): {r['successes']}/{len(errors_deg)}"
    f" ({100 * r['successes'] / len(errors_deg):.1f}%)"
  )


def _print_sweep_summary(args, results: list) -> None:
  print()
  print("=== Heading sweep summary ===")
  print(f"Task: {args.task}")
  print(f"Checkpoint: {args.checkpoint}")
  print()
  print(
    f"{'heading':>9}  {'kicks':>5}/{'eps':<3}  "
    f"{'mean_err':>8}  {'median':>7}  {'p25':>5}  {'p75':>5}  "
    f"{'success%':>8}  {'speed':>6}"
  )
  print("-" * 75)
  for h_deg, r in results:
    errors_deg = r["errors_deg"]
    if not errors_deg:
      print(f"{h_deg:>+9.0f}  {0:>5}/{r['n_done']:<3}  (no kicks)")
      continue
    e = torch.tensor(errors_deg)
    s = torch.tensor(r["speeds"])
    print(
      f"{h_deg:>+9.0f}  {len(errors_deg):>5}/{r['n_done']:<3}  "
      f"{e.mean():>+8.1f}  {e.median():>+7.1f}  "
      f"{e.kthvalue(int(len(e) * 0.25) + 1).values:>+5.1f}  "
      f"{e.kthvalue(int(len(e) * 0.75) + 1).values:>+5.1f}  "
      f"{100 * r['successes'] / len(errors_deg):>7.1f}%  "
      f"{s.mean():>5.2f}"
    )


def _force_spawn(
  cmd: KickTargetCommand,
  ball_distance: float,
  ball_angle_rad: float,
  target_heading_rad: float,
) -> None:
  device = cmd.device
  n = cmd.num_envs
  env_origins = cmd._env.scene.env_origins
  ball_pos = torch.zeros(n, 3, device=device)
  ball_pos[:, 0] = env_origins[:, 0] + ball_distance * math.cos(ball_angle_rad)
  ball_pos[:, 1] = env_origins[:, 1] + ball_distance * math.sin(ball_angle_rad)
  ball_pos[:, 2] = cmd.cfg.ball_spawn_height
  ball_state = torch.zeros(n, 13, device=device)
  ball_state[:, :3] = ball_pos
  ball_state[:, 3] = 1.0
  cmd.ball.write_root_state_to_sim(ball_state)
  cmd._ball_pos_w = ball_pos
  cmd._ball_vel_w = torch.zeros(n, 3, device=device)
  cmd._target_dir_w[:, 0] = math.cos(target_heading_rad)
  cmd._target_dir_w[:, 1] = math.sin(target_heading_rad)
  cmd._initial_ball_progress = (
    ball_pos[:, 0] * cmd._target_dir_w[:, 0] + ball_pos[:, 1] * cmd._target_dir_w[:, 1]
  )
  cmd._prev_ball_progress = cmd._initial_ball_progress.clone()
  cmd._prev_robot_ball_distance = torch.full((n,), ball_distance, device=device)
  cmd.kick_contact_awarded.zero_()
  cmd.kick_contact_new.zero_()
  cmd.peak_kick_speed.zero_()
  cmd.steps_since_kick[:] = -1
  if cmd.perception is not None:
    cmd.perception.reset(torch.arange(n, device=device, dtype=torch.long))


if __name__ == "__main__":
  main()
