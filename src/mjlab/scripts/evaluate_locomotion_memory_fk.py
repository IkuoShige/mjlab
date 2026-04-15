from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.locomotion_memory.mdp import LocomotionMemoryCommandCfg
from mjlab.tasks.locomotion_memory.memory_db import (
  DEFAULT_K1_MEMORY_FILE,
  K1_MEMORY_FILE_ENV_VAR,
  discover_k1_memory_file,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends

DEFAULT_COMMAND_SPECS = (
  "forward_vx_1.0:1.0,0.0,0.0;low_yaw_0.15:0.0,0.0,0.15;yaw_0.35:0.0,0.0,0.35"
)
CommandPreset = Literal["core", "forward_sweep", "yaw_sweep", "research"]
COMMAND_PRESETS: dict[CommandPreset, str] = {
  "core": DEFAULT_COMMAND_SPECS,
  "forward_sweep": (
    "forward_vx_0.5:0.5,0.0,0.0;forward_vx_1.0:1.0,0.0,0.0;forward_vx_1.5:1.5,0.0,0.0"
  ),
  "yaw_sweep": (
    "yaw_l_0.10:0.0,0.0,0.10;"
    "yaw_r_0.10:0.0,0.0,-0.10;"
    "yaw_l_0.15:0.0,0.0,0.15;"
    "yaw_r_0.15:0.0,0.0,-0.15;"
    "yaw_l_0.20:0.0,0.0,0.20;"
    "yaw_r_0.20:0.0,0.0,-0.20;"
    "yaw_l_0.35:0.0,0.0,0.35;"
    "yaw_r_0.35:0.0,0.0,-0.35;"
    "yaw_l_0.50:0.0,0.0,0.50;"
    "yaw_r_0.50:0.0,0.0,-0.50"
  ),
  "research": (
    "forward_vx_0.5:0.5,0.0,0.0;"
    "forward_vx_1.0:1.0,0.0,0.0;"
    "forward_vx_1.5:1.5,0.0,0.0;"
    "yaw_l_0.10:0.0,0.0,0.10;"
    "yaw_r_0.10:0.0,0.0,-0.10;"
    "yaw_l_0.15:0.0,0.0,0.15;"
    "yaw_r_0.15:0.0,0.0,-0.15;"
    "yaw_l_0.20:0.0,0.0,0.20;"
    "yaw_r_0.20:0.0,0.0,-0.20;"
    "yaw_l_0.35:0.0,0.0,0.35;"
    "yaw_r_0.35:0.0,0.0,-0.35;"
    "yaw_l_0.50:0.0,0.0,0.50;"
    "yaw_r_0.50:0.0,0.0,-0.50;"
    "arc_vx0.5_yaw_l0.15:0.5,0.0,0.15;"
    "arc_vx0.5_yaw_r0.15:0.5,0.0,-0.15;"
    "arc_vx1.0_yaw_l0.20:1.0,0.0,0.20;"
    "arc_vx1.0_yaw_r0.20:1.0,0.0,-0.20"
  ),
}


@dataclass(frozen=True)
class FixedCommand:
  name: str
  velocity: tuple[float, float, float]


@dataclass(frozen=True)
class EvaluateLocomotionMemoryFkCfg:
  checkpoint_file: str
  task_id: str = "Mjlab-LocomotionMemory-Flat-Booster-K1"
  memory_file: str | None = None
  command_preset: CommandPreset = "core"
  commands: str = DEFAULT_COMMAND_SPECS
  num_envs: int = 64
  steps: int = 500
  warmup_steps: int = 100
  device: str | None = None
  output_file: str | None = None


def parse_command_spec(spec: str) -> FixedCommand:
  """Parse ``name:vx,vy,yaw`` or ``vx,vy,yaw`` command specifications."""
  if ":" in spec:
    name, raw_values = spec.split(":", maxsplit=1)
  else:
    raw_values = spec
    name = spec.replace(",", "_")
  values = tuple(float(part.strip()) for part in raw_values.split(","))
  if len(values) != 3:
    raise ValueError(
      f"Command spec must contain three comma-separated values, got: {spec!r}"
    )
  command_name = name.strip()
  if not command_name:
    raise ValueError(f"Command spec name cannot be empty: {spec!r}")
  return FixedCommand(command_name, (values[0], values[1], values[2]))


def parse_command_specs(specs: str) -> tuple[FixedCommand, ...]:
  commands = tuple(
    parse_command_spec(spec.strip()) for spec in specs.split(";") if spec.strip()
  )
  if not commands:
    raise ValueError("At least one command spec is required.")
  return commands


def resolve_command_specs(
  commands: str, command_preset: CommandPreset
) -> tuple[FixedCommand, ...]:
  """Resolve a command preset unless the user supplied an explicit command list."""
  if commands != DEFAULT_COMMAND_SPECS:
    return parse_command_specs(commands)
  return parse_command_specs(COMMAND_PRESETS[command_preset])


def summarize_tensor(values: torch.Tensor) -> float:
  values = values.detach().flatten()
  if values.numel() == 0:
    return float("nan")
  return float(values.mean().cpu())


def quantile_tensor(values: torch.Tensor, q: float) -> float:
  values = values.detach().flatten()
  if values.numel() == 0:
    return float("nan")
  return float(torch.quantile(values, q).cpu())


def concat_non_empty(
  values: list[torch.Tensor], *, device: torch.device
) -> torch.Tensor:
  non_empty = [value for value in values if value.numel() > 0]
  if not non_empty:
    return torch.empty(0, device=device)
  return torch.cat(non_empty)


def _set_fixed_twist_command(env: ManagerBasedRlEnv, command: FixedCommand) -> None:
  term = cast(Any, env.command_manager.get_term("twist"))
  velocity = torch.tensor(
    command.velocity,
    dtype=term.command.dtype,
    device=env.device,
  )
  term.vel_command_b[:] = velocity[None, :]
  term.vel_command_w[:] = velocity[None, :]
  term.time_left[:] = 1e9
  for attr in (
    "is_heading_env",
    "is_standing_env",
    "is_world_env",
    "is_forward_env",
    "is_turn_in_place_env",
  ):
    if hasattr(term, attr):
      getattr(term, attr)[:] = False


def _reference_foot_pos_w(
  env: ManagerBasedRlEnv,
  robot: Any,
  site_ids: torch.Tensor,
) -> torch.Tensor:
  """FK active memory reference joint poses without permanently changing sim state."""
  memory = cast(Any, env.command_manager.get_term("memory"))
  qpos = env.sim.data.qpos.clone()
  qvel = env.sim.data.qvel.clone()

  active = memory.snippet_id >= 0
  reference_joint_pos = torch.where(
    active[:, None],
    memory.reference_joint_pos,
    robot.data.joint_pos,
  )
  robot.write_joint_position_to_sim(reference_joint_pos)
  env.sim.forward()
  reference_foot_pos_w = robot.data.site_pos_w[:, site_ids, :].clone()

  env.sim.data.qpos.copy_(qpos)
  env.sim.data.qvel.copy_(qvel)
  env.sim.forward()
  return reference_foot_pos_w


def _active_prefixed_rate(memory: Any, mask_name: str) -> torch.Tensor:
  active = memory.snippet_id >= 0
  output = torch.zeros(memory.num_envs, device=memory.device)
  if not bool(torch.any(active)):
    return output
  mask = getattr(memory, mask_name)
  output[active] = mask[memory.snippet_id[active]].float()
  return output


def _body_frame_foot_pos(
  robot: Any,
  foot_pos_w: torch.Tensor,
) -> torch.Tensor:
  root_pos_w = robot.data.root_link_pos_w[:, None, :]
  root_quat_w = robot.data.root_link_quat_w[:, None, :].expand(
    -1, foot_pos_w.shape[1], -1
  )
  return quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)


def _append_step_metrics(
  env: ManagerBasedRlEnv,
  robot: Any,
  site_ids: torch.Tensor,
  values: dict[str, list[torch.Tensor]],
  touchdown_left: list[torch.Tensor],
  touchdown_right: list[torch.Tensor],
  prev_contact: torch.Tensor | None,
  dones: torch.Tensor,
) -> torch.Tensor:
  memory = cast(Any, env.command_manager.get_term("memory"))
  feet_sensor = env.scene["feet_ground_contact"]

  actual_foot_pos_w = robot.data.site_pos_w[:, site_ids, :].clone()
  actual_foot_pos_b = _body_frame_foot_pos(robot, actual_foot_pos_w)
  reference_foot_pos_w = _reference_foot_pos_w(env, robot, site_ids)
  reference_foot_pos_b = _body_frame_foot_pos(robot, reference_foot_pos_w)

  actual_foot_z = actual_foot_pos_w[:, :, 2]
  reference_foot_z = reference_foot_pos_w[:, :, 2]
  actual_clearance = actual_foot_z - actual_foot_z.min(dim=1, keepdim=True).values
  reference_clearance = (
    reference_foot_z - reference_foot_z.min(dim=1, keepdim=True).values
  )

  reference_contact = memory.reference_contact.bool()
  reference_swing = ~reference_contact
  actual_return_swing = reference_swing & (actual_foot_pos_b[:, :, 0] < 0.05)
  reference_return_swing = reference_swing & (reference_foot_pos_b[:, :, 0] < 0.05)

  current_air_time = feet_sensor.data.current_air_time
  assert current_air_time is not None
  current_contact_time = feet_sensor.data.current_contact_time
  assert current_contact_time is not None
  contact = current_contact_time > 0.0
  if prev_contact is not None:
    touchdown = (~prev_contact) & contact
    touchdown_left.append(touchdown[:, 0].float())
    touchdown_right.append(touchdown[:, 1].float())

  active_air_time = current_air_time[current_air_time > 0]
  foot_delta = actual_foot_pos_b - reference_foot_pos_b
  foot_separation = actual_foot_pos_b[:, 0, 1] - actual_foot_pos_b[:, 1, 1]
  midline_margin = torch.minimum(
    actual_foot_pos_b[:, 0, 1], -actual_foot_pos_b[:, 1, 1]
  )

  values["actual_vx"].append(robot.data.root_link_lin_vel_b[:, 0].clone())
  values["actual_yaw"].append(robot.data.root_link_ang_vel_b[:, 2].clone())
  values["root_height"].append(robot.data.root_link_pos_w[:, 2].clone())
  values["done"].append(dones.float().clone())
  values["actual_foot_z"].append(actual_foot_z.reshape(-1).clone())
  values["reference_foot_z"].append(reference_foot_z.reshape(-1).clone())
  values["actual_swing_foot_z"].append(actual_foot_z[reference_swing].clone())
  values["reference_swing_foot_z"].append(reference_foot_z[reference_swing].clone())
  values["actual_swing_clearance"].append(actual_clearance[reference_swing].clone())
  values["reference_swing_clearance"].append(
    reference_clearance[reference_swing].clone()
  )
  values["actual_return_swing_z"].append(actual_foot_z[actual_return_swing].clone())
  values["reference_return_swing_z"].append(
    reference_foot_z[reference_return_swing].clone()
  )
  values["actual_return_swing_scuff"].append(
    (actual_foot_z[actual_return_swing] < 0.03).float().clone()
  )
  values["reference_return_swing_scuff"].append(
    (reference_foot_z[reference_return_swing] < 0.03).float().clone()
  )
  values["actual_return_swing_trailing"].append(
    (actual_foot_pos_b[:, :, 0][actual_return_swing] < -0.10).float().clone()
  )
  values["reference_return_swing_trailing"].append(
    (reference_foot_pos_b[:, :, 0][reference_return_swing] < -0.10).float().clone()
  )
  values["foot_xyz_ref_error"].append(torch.linalg.norm(foot_delta, dim=-1).reshape(-1))
  values["foot_z_ref_error"].append(torch.abs(foot_delta[:, :, 2]).reshape(-1))
  values["air_time_active"].append(active_air_time.clone())
  values["foot_sep"].append(foot_separation.clone())
  values["midline_margin"].append(midline_margin.clone())
  values["lower_joint_abs_vel"].append(robot.data.joint_vel.abs().reshape(-1).clone())
  values["reference_swing"].append(reference_swing.float().reshape(-1))
  values["turn_snippet"].append(memory.turn_snippet_active.float().clone())
  values["pivot_snippet"].append(_active_prefixed_rate(memory, "_pivot_snippet_mask"))
  values["turn_in_place_v2"].append(memory.turn_in_place_v2_active.float().clone())
  return contact.clone()


def _make_value_buffers() -> dict[str, list[torch.Tensor]]:
  keys = (
    "actual_vx",
    "actual_yaw",
    "root_height",
    "done",
    "actual_foot_z",
    "reference_foot_z",
    "actual_swing_foot_z",
    "reference_swing_foot_z",
    "actual_swing_clearance",
    "reference_swing_clearance",
    "actual_return_swing_z",
    "reference_return_swing_z",
    "actual_return_swing_scuff",
    "reference_return_swing_scuff",
    "actual_return_swing_trailing",
    "reference_return_swing_trailing",
    "foot_xyz_ref_error",
    "foot_z_ref_error",
    "air_time_active",
    "foot_sep",
    "midline_margin",
    "lower_joint_abs_vel",
    "reference_swing",
    "turn_snippet",
    "pivot_snippet",
    "turn_in_place_v2",
  )
  return {key: [] for key in keys}


def _summarize_command(
  values: dict[str, list[torch.Tensor]],
  touchdown_left: list[torch.Tensor],
  touchdown_right: list[torch.Tensor],
  *,
  command: FixedCommand,
  device: torch.device,
) -> dict[str, float]:
  tensors = {
    key: concat_non_empty(value, device=device) for key, value in values.items()
  }
  actual_yaw = tensors["actual_yaw"]
  touchdown_left_mean = (
    float(torch.stack(touchdown_left).sum(dim=0).mean().cpu())
    if touchdown_left
    else float("nan")
  )
  touchdown_right_mean = (
    float(torch.stack(touchdown_right).sum(dim=0).mean().cpu())
    if touchdown_right
    else float("nan")
  )
  return {
    "command_vx": command.velocity[0],
    "command_vy": command.velocity[1],
    "command_yaw": command.velocity[2],
    "actual_vx_mean": summarize_tensor(tensors["actual_vx"]),
    "actual_yaw_mean": summarize_tensor(actual_yaw),
    "actual_yaw_p10": quantile_tensor(actual_yaw, 0.10),
    "actual_yaw_p90": quantile_tensor(actual_yaw, 0.90),
    "yaw_abs_error_mean": summarize_tensor(torch.abs(actual_yaw - command.velocity[2])),
    "root_height_mean": summarize_tensor(tensors["root_height"]),
    "done_rate": summarize_tensor(tensors["done"]),
    "actual_swing_clearance_p05": quantile_tensor(
      tensors["actual_swing_clearance"], 0.05
    ),
    "reference_swing_clearance_p05": quantile_tensor(
      tensors["reference_swing_clearance"], 0.05
    ),
    "actual_return_swing_z_p05": quantile_tensor(
      tensors["actual_return_swing_z"], 0.05
    ),
    "reference_return_swing_z_p05": quantile_tensor(
      tensors["reference_return_swing_z"], 0.05
    ),
    "actual_return_swing_scuff_frac_z_lt_0.03": summarize_tensor(
      tensors["actual_return_swing_scuff"]
    ),
    "reference_return_swing_scuff_frac_z_lt_0.03": summarize_tensor(
      tensors["reference_return_swing_scuff"]
    ),
    "actual_return_swing_trailing_frac_x_lt_-0.10": summarize_tensor(
      tensors["actual_return_swing_trailing"]
    ),
    "reference_return_swing_trailing_frac_x_lt_-0.10": summarize_tensor(
      tensors["reference_return_swing_trailing"]
    ),
    "foot_xyz_ref_error_mean": summarize_tensor(tensors["foot_xyz_ref_error"]),
    "foot_z_ref_error_mean": summarize_tensor(tensors["foot_z_ref_error"]),
    "mean_air_time_active": summarize_tensor(tensors["air_time_active"]),
    "touchdowns_left_per_env": touchdown_left_mean,
    "touchdowns_right_per_env": touchdown_right_mean,
    "foot_sep_mean": summarize_tensor(tensors["foot_sep"]),
    "foot_sep_p95": quantile_tensor(tensors["foot_sep"], 0.95),
    "midline_margin_p05": quantile_tensor(tensors["midline_margin"], 0.05),
    "lower_joint_abs_vel_mean": summarize_tensor(tensors["lower_joint_abs_vel"]),
    "reference_swing_rate": summarize_tensor(tensors["reference_swing"]),
    "turn_snippet_rate": summarize_tensor(tensors["turn_snippet"]),
    "pivot_snippet_rate": summarize_tensor(tensors["pivot_snippet"]),
    "turn_in_place_v2_rate": summarize_tensor(tensors["turn_in_place_v2"]),
  }


def _evaluate_command(
  env: RslRlVecEnvWrapper,
  raw_env: ManagerBasedRlEnv,
  policy: Any,
  robot: Any,
  site_ids: torch.Tensor,
  command: FixedCommand,
  cfg: EvaluateLocomotionMemoryFkCfg,
) -> dict[str, float]:
  obs, _ = env.reset()
  _set_fixed_twist_command(raw_env, command)
  obs = env.get_observations()

  values = _make_value_buffers()
  touchdown_left: list[torch.Tensor] = []
  touchdown_right: list[torch.Tensor] = []
  prev_contact: torch.Tensor | None = None

  for step in range(cfg.steps):
    with torch.inference_mode():
      actions = policy(obs)
    obs, _reward, dones, _extras = env.step(actions)
    _set_fixed_twist_command(raw_env, command)
    if step < cfg.warmup_steps:
      prev_contact = None
      continue
    prev_contact = _append_step_metrics(
      raw_env,
      robot,
      site_ids,
      values,
      touchdown_left,
      touchdown_right,
      prev_contact,
      dones,
    )

  return _summarize_command(
    values,
    touchdown_left,
    touchdown_right,
    command=command,
    device=torch.device(env.device),
  )


def _mean_metric(command_metrics: list[dict[str, float]], metric_name: str) -> float:
  values = [
    metrics[metric_name]
    for metrics in command_metrics
    if metric_name in metrics and not math.isnan(metrics[metric_name])
  ]
  if not values:
    return float("nan")
  return float(sum(values) / len(values))


def _summarize_suite(
  results: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
  forward = [
    metrics
    for metrics in results.values()
    if abs(metrics["command_yaw"]) < 1e-6 and metrics["command_vx"] > 0.0
  ]
  in_place_yaw = [
    metrics
    for metrics in results.values()
    if abs(metrics["command_yaw"]) > 1e-6
    and abs(metrics["command_vx"]) <= 0.25
    and abs(metrics["command_vy"]) <= 0.05
  ]
  arc = [
    metrics
    for metrics in results.values()
    if abs(metrics["command_yaw"]) > 1e-6 and abs(metrics["command_vx"]) > 0.25
  ]
  return {
    "all": {
      "done_rate_mean": _mean_metric(list(results.values()), "done_rate"),
      "foot_xyz_ref_error_mean": _mean_metric(
        list(results.values()), "foot_xyz_ref_error_mean"
      ),
      "scuff_frac_mean": _mean_metric(
        list(results.values()), "actual_return_swing_scuff_frac_z_lt_0.03"
      ),
      "air_time_active_mean": _mean_metric(
        list(results.values()), "mean_air_time_active"
      ),
    },
    "forward": {
      "vx_abs_error_mean": _mean_metric(
        [
          {
            **metrics,
            "vx_abs_error": abs(metrics["actual_vx_mean"] - metrics["command_vx"]),
          }
          for metrics in forward
        ],
        "vx_abs_error",
      ),
      "yaw_abs_error_mean": _mean_metric(forward, "yaw_abs_error_mean"),
      "scuff_frac_mean": _mean_metric(
        forward, "actual_return_swing_scuff_frac_z_lt_0.03"
      ),
      "trailing_frac_mean": _mean_metric(
        forward, "actual_return_swing_trailing_frac_x_lt_-0.10"
      ),
    },
    "in_place_yaw": {
      "yaw_abs_error_mean": _mean_metric(in_place_yaw, "yaw_abs_error_mean"),
      "actual_yaw_mean": _mean_metric(in_place_yaw, "actual_yaw_mean"),
      "scuff_frac_mean": _mean_metric(
        in_place_yaw, "actual_return_swing_scuff_frac_z_lt_0.03"
      ),
      "air_time_active_mean": _mean_metric(in_place_yaw, "mean_air_time_active"),
      "pivot_snippet_rate_mean": _mean_metric(in_place_yaw, "pivot_snippet_rate"),
    },
    "arc": {
      "vx_abs_error_mean": _mean_metric(
        [
          {
            **metrics,
            "vx_abs_error": abs(metrics["actual_vx_mean"] - metrics["command_vx"]),
          }
          for metrics in arc
        ],
        "vx_abs_error",
      ),
      "yaw_abs_error_mean": _mean_metric(arc, "yaw_abs_error_mean"),
      "scuff_frac_mean": _mean_metric(arc, "actual_return_swing_scuff_frac_z_lt_0.03"),
    },
  }


def _load_memory_override(cfg: EvaluateLocomotionMemoryFkCfg, env_cfg: Any) -> None:
  memory_cmd = env_cfg.commands["memory"]
  assert isinstance(memory_cmd, LocomotionMemoryCommandCfg)
  if cfg.memory_file is not None:
    memory_file = Path(cfg.memory_file)
    if not memory_file.is_file():
      raise FileNotFoundError(f"Memory file not found: {memory_file}")
    memory_cmd.memory_file = str(memory_file)
    return
  if memory_cmd.memory_file and Path(memory_cmd.memory_file).is_file():
    return
  discovered = discover_k1_memory_file()
  if discovered is not None:
    memory_cmd.memory_file = discovered
    return
  raise FileNotFoundError(
    "Locomotion-memory evaluation requires a memory database. "
    f"Provide --memory-file, set {K1_MEMORY_FILE_ENV_VAR}, or build {DEFAULT_K1_MEMORY_FILE}."
  )


def evaluate(cfg: EvaluateLocomotionMemoryFkCfg) -> dict[str, Any]:
  if cfg.steps <= cfg.warmup_steps:
    raise ValueError("steps must be greater than warmup_steps.")
  checkpoint_file = Path(cfg.checkpoint_file)
  if not checkpoint_file.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  _load_memory_override(cfg, env_cfg)
  agent_cfg = load_rl_cfg(cfg.task_id)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_file),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  robot = raw_env.scene["robot"]
  site_ids, site_names = robot.find_sites(
    ("left_foot", "right_foot"), preserve_order=True
  )
  site_id_tensor = torch.tensor(site_ids, device=env.device)

  commands = resolve_command_specs(cfg.commands, cfg.command_preset)
  results: dict[str, dict[str, float]] = {}
  try:
    for command in commands:
      results[command.name] = _evaluate_command(
        env,
        raw_env,
        policy,
        robot,
        site_id_tensor,
        command,
        cfg,
      )
  finally:
    raw_env.close()

  return {
    "task_id": cfg.task_id,
    "checkpoint_file": str(checkpoint_file),
    "num_envs": cfg.num_envs,
    "steps": cfg.steps,
    "warmup_steps": cfg.warmup_steps,
    "command_preset": cfg.command_preset
    if cfg.commands == DEFAULT_COMMAND_SPECS
    else "custom",
    "foot_site_names": site_names,
    "commands": results,
    "suite_summary": _summarize_suite(results),
  }


def _print_summary(results: dict[str, Any]) -> None:
  print(
    f"Evaluated {results['checkpoint_file']} "
    f"with {results['num_envs']} envs, steps={results['steps']}, "
    f"warmup={results['warmup_steps']}, preset={results['command_preset']}"
  )
  for name, metrics in results["commands"].items():
    print(f"\n{name}")
    for key in (
      "actual_vx_mean",
      "actual_yaw_mean",
      "yaw_abs_error_mean",
      "done_rate",
      "actual_swing_clearance_p05",
      "reference_swing_clearance_p05",
      "actual_return_swing_scuff_frac_z_lt_0.03",
      "reference_return_swing_scuff_frac_z_lt_0.03",
      "actual_return_swing_trailing_frac_x_lt_-0.10",
      "reference_return_swing_trailing_frac_x_lt_-0.10",
      "foot_xyz_ref_error_mean",
      "mean_air_time_active",
      "touchdowns_left_per_env",
      "touchdowns_right_per_env",
      "turn_snippet_rate",
      "pivot_snippet_rate",
    ):
      print(f"  {key}: {metrics[key]:.6g}")
  print("\nSuite summary")
  for group_name, metrics in results["suite_summary"].items():
    print(f"  {group_name}")
    for key, value in metrics.items():
      print(f"    {key}: {value:.6g}")


def main() -> None:
  cfg = tyro.cli(EvaluateLocomotionMemoryFkCfg, config=mjlab.TYRO_FLAGS)
  results = evaluate(cfg)
  _print_summary(results)
  if cfg.output_file is not None:
    output_file = Path(cfg.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved FK evaluation JSON to {output_file}")


if __name__ == "__main__":
  main()
