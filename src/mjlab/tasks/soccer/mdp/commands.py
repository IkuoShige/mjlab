"""Soccer multi-motion command term.

Ported from HumanoidSoccer (Kong et al., 2026).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.soccer.mdp.kick_detection import KickContactTracker
from mjlab.tasks.tracking.mdp.commands import MultiMotionLoader
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class SoccerMotionCommand(CommandTerm):
  """Multi-motion command with soccer ball placement and kick tracking."""

  cfg: SoccerMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: SoccerMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.soccer_ball: Entity | None = None
    try:
      self.soccer_ball = env.scene[cfg.soccer_ball_entity_name]
    except KeyError:
      pass

    self.kick_contact_tracker = KickContactTracker(env)

    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    # Cache foot body indices for velocity-gated kick detection.
    self._foot_body_indices = torch.tensor(
      self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MultiMotionLoader(
      cfg.motion_files, self.body_indexes, device=self.device
    )

    kick_leg_to_id = {"left": 0, "right": 1}
    self._kick_leg_id_to_name = {v: k for k, v in kick_leg_to_id.items()}
    self._kick_leg_id_to_name[-1] = "unknown"
    self.motion_kick_leg = torch.full(
      (self.motion.num_files,), -1, dtype=torch.int8, device=self.device
    )
    for idx, label in enumerate(self.motion.kick_leg_labels):
      normalized = label.lower() if isinstance(label, str) else None
      if normalized in kick_leg_to_id:
        self.motion_kick_leg[idx] = kick_leg_to_id[normalized]

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_length = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.motion_resampled = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Assign initial motions.
    if self.motion.num_files > 1:
      self.motion_idx = torch.randint(
        0,
        self.motion.num_files,
        (self.num_envs,),
        dtype=torch.long,
        device=self.device,
      )
    self.motion_length[:] = self.motion.file_lengths[self.motion_idx]

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Adaptive sampling.
    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.motion.num_files,
      self.bin_count,
      dtype=torch.float,
      device=self.device,
    )
    self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
    self.kernel = torch.tensor(
      [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    # Metrics.
    for key in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_anchor_lin_vel",
      "error_anchor_ang_vel",
      "error_body_pos",
      "error_body_rot",
      "error_joint_pos",
      "error_joint_vel",
    ):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    # Soccer ball state.
    self.target_point_pos = torch.zeros(
      self.num_envs, 3, dtype=torch.float32, device=self.device
    )
    self.soccer_ball_pos = torch.zeros_like(self.target_point_pos)
    self.target_destination_pos = torch.zeros_like(self.target_point_pos)
    self.initial_target_point_pos = torch.zeros_like(self.target_point_pos)

    # Blind zone state.
    self.blind_distance_min = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self.blind_distance_max = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self.last_visible_target_point_base = torch.zeros(
      self.num_envs, 3, dtype=torch.float32, device=self.device
    )
    self.is_in_blind_zone = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Destination generation.
    self.destination_height = 0.11
    self.destination_center = torch.tensor(
      [0.0, -5.0, self.destination_height], device=self.device
    )
    self.destination_length = 1.0
    self.destination_width = 0.5

    # Curve offset for ball placement.
    self.curve_radius_offset = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self._radius_offset_min: float | None = None
    self._radius_offset_max: float | None = None
    curve_cfg = cfg.curve_offset_range or {}
    radius_range = curve_cfg.get("radius")
    if isinstance(radius_range, tuple):
      self._radius_offset_min = float(radius_range[0])
      self._radius_offset_max = float(radius_range[1])
    elif radius_range is not None:
      v = float(radius_range)
      self._radius_offset_min = v
      self._radius_offset_max = v
    arc_angle = curve_cfg.get("arc_angle")
    self._target_arc_angle = (
      float(arc_angle) if isinstance(arc_angle, (int, float)) else math.pi / 18.0
    )
    height = curve_cfg.get("height")
    self._target_height = float(height) if isinstance(height, (int, float)) else 0.11

    # Initialize all envs.
    all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    self._sample_soccer_offset(all_ids)
    self._compute_soccer_ball_positions(all_ids)
    self._update_soccer_ball(all_ids)
    self._update_target_points(all_ids)

    # Ghost viz state (lazily built on first draw).
    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  # ------------------------------------------------------------------
  # Properties: motion reference data
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.motion_idx, self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.motion_idx, self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.motion_idx, self.time_steps]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.motion_idx, self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.motion_idx, self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.motion_idx, self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[
        self.motion_idx, self.time_steps, self.motion_anchor_body_index
      ]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[
      self.motion_idx, self.time_steps, self.motion_anchor_body_index
    ]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[
      self.motion_idx, self.time_steps, self.motion_anchor_body_index
    ]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[
      self.motion_idx, self.time_steps, self.motion_anchor_body_index
    ]

  # ------------------------------------------------------------------
  # Properties: robot state
  # ------------------------------------------------------------------

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_pelvis_pos_w(self) -> torch.Tensor:
    """Root/anchor body world position. Name kept for backwards compat with
    existing reward/observation functions ported from the G1 HumanoidSoccer
    codebase; for K1 this returns the Trunk pose."""
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_pelvis_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def kick_leg(self) -> torch.Tensor:
    return self.motion_kick_leg[self.motion_idx]

  # ------------------------------------------------------------------
  # Sampling
  # ------------------------------------------------------------------

  def _sample_soccer_offset(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    if self._radius_offset_min is None or self._radius_offset_max is None:
      self.curve_radius_offset[env_ids] = 0.0
      return
    if abs(self._radius_offset_max - self._radius_offset_min) < 1e-6:
      self.curve_radius_offset[env_ids] = self._radius_offset_min
      return
    rand = torch.rand(env_ids.numel(), device=self.device)
    span = self._radius_offset_max - self._radius_offset_min
    self.curve_radius_offset[env_ids] = self._radius_offset_min + rand * span

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    motion_indices = torch.randint(
      0, self.motion.num_files, (len(env_ids),), device=self.device
    )
    self.motion_idx[env_ids] = motion_indices
    self.motion_length[env_ids] = self.motion.file_lengths[motion_indices]
    # Start from frame 0.
    self.time_steps[env_ids] = 0

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    episode_failed = self._env.termination_manager.terminated[env_ids].to(
      device=self.device, dtype=torch.bool
    )
    self._current_bin_failed.zero_()

    if torch.any(episode_failed):
      failed_motion = self.motion_idx[env_ids][episode_failed]
      failed_lengths = self.motion_length[env_ids][episode_failed].clamp(min=1).float()
      failed_steps = self.time_steps[env_ids][episode_failed].float()
      failed_phase = failed_steps / (failed_lengths - 1.0 + 1e-6)
      failed_bins = torch.clamp(
        (failed_phase * self.bin_count).long(), 0, self.bin_count - 1
      )
      flat_idx = failed_motion * self.bin_count + failed_bins
      flat_size = int(self.motion.num_files * self.bin_count)
      flat_counts = torch.zeros(
        flat_size, dtype=self._current_bin_failed.dtype, device=self.device
      )
      if flat_idx.numel() > 0:
        flat_idx = flat_idx.to(self.device).long()
        ones = torch.ones_like(flat_idx, dtype=flat_counts.dtype, device=self.device)
        flat_counts.index_add_(0, flat_idx, ones)
      self._current_bin_failed[:] = flat_counts.float().view(
        self.motion.num_files, self.bin_count
      )

    M = max(1, int(self.motion.num_files))
    B = max(1, int(self.bin_count))
    uniform_per_pair = self.cfg.adaptive_uniform_ratio / float(M * B)
    probs = self.bin_failed_count + self._current_bin_failed + uniform_per_pair
    probs = torch.nn.functional.pad(
      probs.unsqueeze(1),
      (0, self.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).squeeze(1)

    probs = probs.view(-1)
    probs = probs / (probs.sum() + 1e-12)

    sampled_flat = torch.multinomial(probs, len(env_ids), replacement=True)
    sampled_motion = sampled_flat // self.bin_count
    sampled_bins = sampled_flat % self.bin_count

    self.motion_idx[env_ids] = sampled_motion
    self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
    rand_offset = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device).float()
    sampled_phase = (sampled_bins.float() + rand_offset) / float(self.bin_count)
    self.time_steps[env_ids] = (
      sampled_phase * (self.motion_length[env_ids].float() - 1)
    ).long()

  # ------------------------------------------------------------------
  # Soccer ball logic
  # ------------------------------------------------------------------

  def _compute_soccer_ball_positions(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    arc_limit = float(self._target_arc_angle)
    base_height = float(self._target_height)

    for env_id in env_ids:
      mi = int(self.motion_idx[env_id].item())
      ml = max(1, int(self.motion_length[env_id].item()))

      first_anchor = self.motion.get_first_frame_anchor_pos(
        mi, self.motion_anchor_body_index
      )
      last_anchor = self.motion.get_last_frame_anchor_pos(
        mi, self.motion_anchor_body_index, ml
      )

      radius_vec = last_anchor[:2] - first_anchor[:2]
      radius_sq = torch.dot(radius_vec, radius_vec)
      radius = (
        torch.sqrt(radius_sq)
        if float(radius_sq) > 1e-12
        else torch.tensor(0.0, device=self.device)
      )

      if arc_limit > 0.0 and float(radius_sq) > 1e-12:
        base_angle = torch.atan2(radius_vec[1], radius_vec[0])
        angle_offset = sample_uniform(
          -arc_limit, arc_limit, (1,), device=self.device
        ).squeeze(0)
        new_angle = base_angle + angle_offset
        direction = torch.stack((torch.cos(new_angle), torch.sin(new_angle)))
      elif float(radius_sq) > 1e-12:
        direction = radius_vec / radius
      else:
        direction = torch.tensor([1.0, 0.0], device=self.device)

      radius = torch.clamp(radius + self.curve_radius_offset[env_id], min=0.0)
      target_xy = first_anchor[:2] + radius * direction

      ball_pos = self.soccer_ball_pos.new_empty(3)
      ball_pos[:2] = target_xy
      ball_pos[2] = base_height
      self.soccer_ball_pos[env_id] = ball_pos

  def _update_target_points(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self.target_point_pos[env_ids] = self.soccer_ball_pos[env_ids]
    self.initial_target_point_pos[env_ids] = self.soccer_ball_pos[env_ids].clone()

  def _update_target_points_from_sim(self) -> None:
    """Read soccer-ball position from simulation each step."""
    if self.soccer_ball is None:
      return
    env_origins = self._env.scene.env_origins
    ball_world_pos = self.soccer_ball.data.root_link_pos_w
    self.soccer_ball_pos = ball_world_pos - env_origins
    self.target_point_pos = self.soccer_ball_pos.clone()

  def _update_destination_points(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    rand_x = (
      torch.rand(env_ids.numel(), device=self.device) - 0.5
    ) * self.destination_length
    rand_y = (
      torch.rand(env_ids.numel(), device=self.device) - 0.5
    ) * self.destination_width
    destination = self.destination_center.expand(env_ids.numel(), -1) + torch.stack(
      [rand_x, rand_y, torch.zeros_like(rand_x)], dim=1
    )
    self.target_destination_pos[env_ids] = destination

  def _update_soccer_ball(self, env_ids: torch.Tensor) -> None:
    if self.soccer_ball is None:
      return
    if env_ids.numel() == 0:
      return
    env_origins = self._env.scene.env_origins

    ball_pos = self.soccer_ball_pos[env_ids] + env_origins[env_ids]
    ball_quat = ball_pos.new_zeros((env_ids.numel(), 4))
    ball_quat[:, 0] = 1.0

    if self.cfg.enable_soccer_ball_init_vel:
      vel_range = self.cfg.soccer_ball_init_lin_vel_range or {}
      ranges_t = torch.tensor(
        [vel_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]],
        device=self.device,
      )
      ball_lin_vel = sample_uniform(
        ranges_t[:, 0], ranges_t[:, 1], (env_ids.numel(), 3), device=self.device
      )
    else:
      ball_lin_vel = ball_pos.new_zeros((env_ids.numel(), 3))
    ball_ang_vel = ball_pos.new_zeros((env_ids.numel(), 3))

    ball_state = torch.cat([ball_pos, ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
    self.soccer_ball.write_root_state_to_sim(ball_state, env_ids=env_ids)

  # ------------------------------------------------------------------
  # Resample / Update
  # ------------------------------------------------------------------

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    self._sample_soccer_offset(env_ids)
    self._uniform_sampling(env_ids)
    self._compute_soccer_ball_positions(env_ids)
    self._update_soccer_ball(env_ids)
    self._update_target_points(env_ids)
    self._update_destination_points(env_ids)

    # Blind zone.
    bmin_lo, bmin_hi = self.cfg.blind_distance_min_range
    bmax_lo, bmax_hi = self.cfg.blind_distance_max_range
    self.blind_distance_min[env_ids] = bmin_lo + torch.rand(
      env_ids.numel(), device=self.device
    ) * (bmin_hi - bmin_lo)
    self.blind_distance_max[env_ids] = bmax_lo + torch.rand(
      env_ids.numel(), device=self.device
    ) * (bmax_hi - bmax_lo)
    self.is_in_blind_zone[env_ids] = False
    self.last_visible_target_point_base[env_ids] = 0.0

    # Reset robot state from motion reference.
    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand[:, 0:3]
    orientations_delta = quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5])
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand[:, :3]
    root_ang_vel[env_ids] += rand[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()
    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_limits[:, :, 0], soft_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )
    self.robot.write_root_state_to_sim(
      torch.cat(
        [
          root_pos[env_ids],
          root_ori[env_ids],
          root_lin_vel[env_ids],
          root_ang_vel[env_ids],
        ],
        dim=-1,
      ),
      env_ids=env_ids,
    )
    self.robot.reset(env_ids=env_ids)

    self.motion_resampled[env_ids] = True

  def _update_command(self) -> None:
    self.kick_contact_tracker.begin_step(self)

    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion_length)[0]
    self._resample_command(env_ids)

    self._update_target_points_from_sim()

    # Keep initial_target_point_pos updated until contact occurs.
    contact_awarded = self.kick_contact_tracker.contact_awarded
    no_contact = ~contact_awarded
    if torch.any(no_contact):
      self.initial_target_point_pos[no_contact] = self.target_point_pos[no_contact]

    # Compute relative body poses.
    nb = len(self.cfg.body_names)
    anchor_pos_w_r = self.anchor_pos_w[:, None, :].expand(-1, nb, -1)
    anchor_quat_w_r = self.anchor_quat_w[:, None, :].expand(-1, nb, -1)
    robot_anchor_pos_w_r = self.robot_anchor_pos_w[:, None, :].expand(-1, nb, -1)
    robot_anchor_quat_w_r = self.robot_anchor_quat_w[:, None, :].expand(-1, nb, -1)

    delta_pos_w = robot_anchor_pos_w_r.clone()
    delta_pos_w[..., 2] = anchor_pos_w_r[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_r, quat_inv(anchor_quat_w_r)))

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_r
    )

    # EMA update of failure histogram.
    self.bin_failed_count = (
      self.cfg.adaptive_alpha * self._current_bin_failed
      + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
    )
    self._current_bin_failed.zero_()

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw a translucent ghost robot at the current reference motion pose."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode != "ghost":
      return

    if self._ghost_model is None:
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      for gi in range(self._ghost_model.ngeom):
        if (
          self._ghost_model.geom_contype[gi] != 0
          or self._ghost_model.geom_conaffinity[gi] != 0
        ):
          self._ghost_model.geom_rgba[gi, 3] = 0
        else:
          self._ghost_model.geom_rgba[gi] = self._ghost_color

    entity: Entity = self._env.scene[self.cfg.entity_name]
    indexing = entity.indexing
    free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = indexing.joint_q_adr.cpu().numpy()

    for batch in env_indices:
      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
      qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()
      visualizer.add_ghost_mesh(
        qpos,
        model=self._ghost_model,
        label=f"ghost_{batch}",
      )


@dataclass(kw_only=True)
class SoccerMotionCommandCfg(CommandTermCfg):
  """Configuration for the soccer motion command."""

  motion_files: list[str] = field(default_factory=list)
  anchor_body_name: str = ""
  body_names: tuple[str, ...] = ()
  entity_name: str = "robot"
  soccer_ball_entity_name: str = "soccer_ball"

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)

  adaptive_kernel_size: int = 3
  adaptive_lambda: float = 0.1
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.4

  curve_offset_range: dict[str, float | tuple[float, float]] | None = None

  enable_soccer_ball_init_vel: bool = False
  soccer_ball_init_lin_vel_range: dict[str, tuple[float, float]] | None = None

  blind_distance_min_range: tuple[float, float] = (0.3, 0.5)
  blind_distance_max_range: tuple[float, float] = (1.5, 2.0)

  foot_body_names: tuple[str, str] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> SoccerMotionCommand:
    return SoccerMotionCommand(self, env)
