"""KickTargetCommand: ball spawn + kick target direction (no motion reference).

Drives the kick-only LVDRS-style training task. The command:
  * Spawns the soccer ball at a random distance/angle around the robot.
  * Samples a random kick target direction (any heading).
  * Optionally feeds the ball position through a VirtualPerception model
    for sim2real-ready actor observations.
  * Tracks the first kick contact (latched) for reward gating.

It deliberately does NOT load reference motion clips — motion style comes
from the AMP discriminator running in the runner, not from per-frame
imitation rewards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.kick.mdp.perception import VirtualPerception, VirtualPerceptionCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


# Arms-down reset targets for K1. Critical V1.25 correction: K1's mesh has
# the upper arm extending laterally at Shoulder_Roll=0, so Shoulder_Roll=0
# IS the T-pose. To bring arms down, the LEFT arm needs Shoulder_Roll
# negative and the RIGHT arm needs Shoulder_Roll positive (mirror), each
# combined with Elbow_Pitch ≈ 1.5 rad to fold the forearm. Grid-search
# yielded hand_z ≈ 0.10 m with this combination — i.e., true arms-down.
#
# Head_pitch is set to the joint's upper limit (+0.855 rad ≈ +49°, head
# fully tilted DOWN — positive pitch looks down per K1 convention). This
# matches the real K1 power-on stance with the camera angled toward the
# feet, so the policy doesn't OOD on the initial observation at deploy
# time. The head is actuated so the policy can re-aim mid-episode.
_INITIAL_POSE_RESET_TARGETS_RAD: dict[str, float] = {
  "ALeft_Shoulder_Pitch": 0.0,
  "Left_Shoulder_Roll": -1.0,
  "Left_Elbow_Pitch": 1.5,
  "Left_Elbow_Yaw": 0.0,
  "ARight_Shoulder_Pitch": 0.0,
  "Right_Shoulder_Roll": 1.0,
  "Right_Elbow_Pitch": 1.5,
  "Right_Elbow_Yaw": 0.0,
  "Head_pitch": 0.855,
}


@dataclass(kw_only=True)
class KickTargetCommandCfg(CommandTermCfg):
  asset_name: str = "robot"
  """Robot entity name in the scene."""

  ball_name: str = "soccer_ball"
  """Ball entity name in the scene."""

  ball_sensor_name: str | None = "ball_contact"
  """Contact sensor name for foot-ball contact detection. ``None`` disables
  contact tracking (rewards that depend on it will return zero).
  """

  foot_body_names: tuple[str, ...] = ("left_foot_link", "right_foot_link")
  """Foot link names used to gate kick contact by foot speed."""

  ball_spawn_distance_range: tuple[float, float] = (0.05, 1.5)
  """Min/max radial distance from robot for ball spawn (m)."""

  ball_spawn_angle_range: tuple[float, float] = (-math.pi, math.pi)
  """Angular range (radians) around robot's forward axis for ball spawn."""

  target_dir_range: tuple[float, float] = (-math.pi, math.pi)
  """Kick target direction range in world frame (radians)."""

  ball_spawn_height: float = 0.11
  """Ball spawn height (ball radius)."""

  robot_spawn_height: float = 0.55
  """Robot root (Trunk) z spawned at this height every reset. The K1
  MJCF's worldbody default is ``pos="0 0 1.0"`` which causes a 50cm
  mid-air drop; 0.55m matches the HOME_KEYFRAME ground-contact trunk z.
  """

  bias_close: bool = True
  """If True, sample log-uniform in distance (favor close spawns)."""

  min_foot_speed: float = 3.0
  """Minimum foot speed (m/s) for a contact to count as a kick."""

  reset_head_pitch_down: bool = True
  """If True, override Head_pitch on reset to ``_INITIAL_POSE_RESET_TARGETS_RAD``
  (looking at feet, sim2real-ready). Set False at play time to play
  pre-V1.43 checkpoints (which were trained with Head_pitch=0 init)."""

  ball_moving_prob: float = 0.0
  """Probability that a reset ball spawns with a non-zero linear velocity.
  V1.48 sets this to 0.3 so 30% of episodes have a slowly moving ball —
  forces the policy to keep tracking the ball with its head/perception
  up to the last moment before contact."""

  ball_init_speed_range: tuple[float, float] = (0.0, 0.5)
  """``(min, max)`` linear-speed magnitude in m/s for the moving-ball
  branch of `ball_moving_prob`. Direction is sampled uniformly in xy."""

  horizontal_force_threshold: float = 10.0
  """Contact force threshold (N) for kick detection."""

  perception: VirtualPerceptionCfg | None = None
  """Virtual perception configuration. ``None`` disables perception sim and
  the policy receives ground-truth ball state.
  """

  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  """Episodes drive resampling; default disables timed resampling."""

  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
    return KickTargetCommand(self, env)


class KickTargetCommand(CommandTerm):
  """Ball spawn + kick direction sampler with optional perception sim."""

  cfg: KickTargetCommandCfg

  def __init__(self, cfg: KickTargetCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.asset_name]
    self.ball: Entity = env.scene[cfg.ball_name]

    body_names = list(self.robot.body_names)
    missing_feet = [n for n in cfg.foot_body_names if n not in body_names]
    if missing_feet:
      raise ValueError(
        f"Foot bodies not on robot: {missing_feet}; available: {body_names}"
      )
    self._foot_body_indices = torch.tensor(
      [body_names.index(n) for n in cfg.foot_body_names],
      dtype=torch.long,
      device=self.device,
    )

    self._ball_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._ball_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._target_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
    self._initial_ball_progress = torch.zeros(self.num_envs, device=self.device)
    self._prev_robot_ball_distance = torch.zeros(self.num_envs, device=self.device)
    self._prev_ball_progress = torch.zeros(self.num_envs, device=self.device)

    self.kick_contact_awarded = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.kick_contact_new = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.peak_kick_speed = torch.zeros(self.num_envs, device=self.device)
    # Counts how many steps have elapsed since the first valid kick contact
    # in the current episode (-1 before the kick happens). Used by the
    # ``kick_completed`` termination to end episodes shortly after the strike.
    self.steps_since_kick = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )

    self.perception: VirtualPerception | None = None
    if cfg.perception is not None:
      self.perception = VirtualPerception(
        cfg.perception, self.num_envs, env.step_dt, self.device
      )

    # V1.21: cache reset-override joint indices and target angles so we can
    # override the K1 HOME_KEYFRAME's defaults on every reset.
    # Covers the arm joints (HOME_KEYFRAME has them raised — V1.25 fixed
    # this to true arms-down) and Head_pitch (set to look at feet,
    # matching the real K1 power-on stance for sim2real).
    pose_targets = dict(_INITIAL_POSE_RESET_TARGETS_RAD)
    if not cfg.reset_head_pitch_down:
      pose_targets.pop("Head_pitch", None)
    pose_idx: list[int] = []
    pose_vals: list[float] = []
    robot_joint_names = list(self.robot.joint_names)
    for joint_name, target in pose_targets.items():
      if joint_name in robot_joint_names:
        pose_idx.append(robot_joint_names.index(joint_name))
        pose_vals.append(target)
    if pose_idx:
      self._pose_reset_indices: torch.Tensor | None = torch.tensor(
        pose_idx, dtype=torch.long, device=self.device
      )
      self._pose_reset_targets: torch.Tensor = torch.tensor(
        pose_vals, dtype=torch.float32, device=self.device
      )
    else:
      self._pose_reset_indices = None
      self._pose_reset_targets = torch.empty(0, device=self.device)

    self.metrics["kick_success_rate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["peak_kick_speed"] = torch.zeros(self.num_envs, device=self.device)

    all_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
    self._resample_command(all_ids)

  # -- CommandTerm API --

  @property
  def command(self) -> torch.Tensor:
    """Concatenated (target_dir_b, ball_pos_b) for default logging."""
    return torch.cat([self.kick_target_dir_b, self.ball_pos_b], dim=-1)

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    cfg = self.cfg
    n = env_ids.numel()

    if cfg.bias_close:
      log_min = math.log(max(cfg.ball_spawn_distance_range[0], 1e-3))
      log_max = math.log(cfg.ball_spawn_distance_range[1])
      log_d = torch.rand(n, device=self.device) * (log_max - log_min) + log_min
      distance = torch.exp(log_d)
    else:
      lo, hi = cfg.ball_spawn_distance_range
      distance = torch.rand(n, device=self.device) * (hi - lo) + lo

    angle_lo, angle_hi = cfg.ball_spawn_angle_range
    spawn_angle = torch.rand(n, device=self.device) * (angle_hi - angle_lo) + angle_lo

    # V1.11: forward-biased target_dir sampling.
    # V1.33: tightened to 95/5/0 to break the bimodal direction local
    # optimum. +0° eval went 69° → 24.8° error.
    # V1.35: widened to 70/25/5 for arbitrary direction. Eval at iter
    # 32000 (1800 iters in) showed bimodal local optimum returned at
    # ±20° (14-19° → 68-69° mean), even though forward (+0°) held
    # roughly steady. Target widening was too aggressive — 5× more
    # side training collapsed the forward mode the policy had learned.
    # V1.36: reverted to 95/5/0 after V1.35's 70/25/5 broke ±20°. Forward
    # then got even better than V1.33 with the V1.35 perception fixes —
    # +0° eval median 2.6°, 50% success.
    # V1.37 (90/8/2): even 3% step regressed +0° eval from V1.36 median
    # 2.6° to 19.2°. Continued training (V1.37b, 1780 iters at 90/8/2)
    # did NOT recover forward — suggests the 2% rear targets are
    # destabilizing forward sharpness, not the 8% side targets.
    # V1.38b (92/8/0, 1400 iters): sides became exceptional (median 4-10°
    # all directions inc. left), but +0° forward collapsed (median 33°,
    # 0% success) — L/R foot indecision emerges at exact 0° when policy
    # learns to handle both sides.
    # V1.47/V1.48: refocus on the forward ±60° cone where the user wants
    # tight precision, while keeping a small slice of training beyond ±60°
    # so the policy remains *functional* (kicks happen, doesn't fall) at
    # any heading. The precision tier is the inner cones; the 5% outside
    # is just to prevent the policy from forgetting that arbitrary
    # headings exist.
    # Mixture (V1.48):
    # - 70% in [-π/6, π/6]   (±30°, forward core — precision tier)
    # - 25% in [±π/6, ±π/3]  (±30° to ±60°, mid-cone — precision tier)
    # -  5% in [±π/3, ±π]    (±60° to ±180°, functional coverage)
    bin_probs = torch.tensor([0.70, 0.25, 0.05], device=self.device).expand(n, -1)
    bin_idx = torch.multinomial(bin_probs, num_samples=1).squeeze(-1)
    bin_ranges = torch.tensor(
      [
        [-math.pi / 6, math.pi / 6],
        [math.pi / 6, math.pi / 3],
        [math.pi / 3, math.pi],
      ],
      device=self.device,
    )
    selected = bin_ranges[bin_idx]
    target_angle = (
      torch.rand(n, device=self.device) * (selected[:, 1] - selected[:, 0])
      + selected[:, 0]
    )
    # For side and rear bins, flip half to negative angles so both sides
    # are sampled symmetrically.
    flip = (torch.rand(n, device=self.device) < 0.5) & (bin_idx > 0)
    target_angle = torch.where(flip, -target_angle, target_angle)

    # Reset robot root state AND joint positions to HOME pose every episode.
    # InitialStateCfg only applies at first instantiation; without explicit
    # writes on each reset, the K1 MJCF defaults take over: trunk z=1.0
    # (50cm mid-air) and all joints at 0 (arms straight, knees locked) —
    # both produce the "weird initial pose" observed in viser. The soccer
    # task masks this by writing the motion-frame-0 state on every reset.
    env_origins = self._env.scene.env_origins[env_ids]
    robot_state = torch.zeros(n, 13, device=self.device)
    robot_state[:, 0:2] = env_origins[:, 0:2]
    robot_state[:, 2] = env_origins[:, 2] + cfg.robot_spawn_height
    robot_state[:, 3] = 1.0  # quat w identity
    self.robot.write_root_state_to_sim(robot_state, env_ids=env_ids)

    # V1.21: reset joints to HOME defaults but OVERRIDE the arm joints to
    # an arms-down pose. K1 HOME_KEYFRAME has Shoulder_Pitch=+17°,
    # Elbow_Pitch=+29° (arms tilted forward) — every reset the arms start
    # raised, and arm_deviation_l2 alone wasn't strong enough to pull
    # them down within an episode. Resetting to arms-down means the
    # policy maintains an already-down pose instead of fighting it.
    starting_jp = self.robot.data.default_joint_pos[env_ids].clone()
    default_jv = self.robot.data.default_joint_vel[env_ids]
    if self._pose_reset_indices is not None:
      starting_jp[:, self._pose_reset_indices] = self._pose_reset_targets
    self.robot.write_joint_state_to_sim(starting_jp, default_jv, env_ids=env_ids)

    # Robot was just written to env_origin with identity quat (forward = +x
    # world). Use env_origins directly for the ball anchor rather than
    # robot.data.root_link_pos_w, which is stale until the next sim step
    # refreshes the data buffer. Identity quat means local body axes
    # coincide with world axes — no rotation needed.
    local_offset = torch.stack(
      [distance * torch.cos(spawn_angle), distance * torch.sin(spawn_angle)],
      dim=-1,
    )

    ball_pos = torch.zeros(n, 3, device=self.device)
    ball_pos[:, :2] = env_origins[:, :2] + local_offset
    ball_pos[:, 2] = cfg.ball_spawn_height

    ball_state = torch.zeros(n, 13, device=self.device)
    ball_state[:, :3] = ball_pos
    ball_state[:, 3] = 1.0  # quat w

    # V1.48: moving-ball DR. Sample per-env whether the ball starts
    # moving, and if so give it a uniformly-random xy direction with a
    # random speed in ``ball_init_speed_range``. Forces the policy to
    # keep tracking the ball with its head/perception right up to the
    # kick contact rather than committing to a memorized swing trajectory.
    init_vel_w = torch.zeros(n, 3, device=self.device)
    if cfg.ball_moving_prob > 0.0:
      moving = torch.rand(n, device=self.device) < cfg.ball_moving_prob
      lo, hi = cfg.ball_init_speed_range
      speed = torch.rand(n, device=self.device) * (hi - lo) + lo
      dir_angle = torch.rand(n, device=self.device) * (2.0 * math.pi)
      init_vel_w[:, 0] = moving.float() * speed * torch.cos(dir_angle)
      init_vel_w[:, 1] = moving.float() * speed * torch.sin(dir_angle)
    ball_state[:, 7:10] = init_vel_w

    self.ball.write_root_state_to_sim(ball_state, env_ids=env_ids)

    self._ball_pos_w[env_ids] = ball_pos
    self._ball_vel_w[env_ids] = init_vel_w
    self._target_dir_w[env_ids] = torch.stack(
      [torch.cos(target_angle), torch.sin(target_angle)], dim=-1
    )

    target_dir = self._target_dir_w[env_ids]
    progress = ball_pos[:, 0] * target_dir[:, 0] + ball_pos[:, 1] * target_dir[:, 1]
    self._initial_ball_progress[env_ids] = progress
    self._prev_ball_progress[env_ids] = progress
    # Robot is at env_origins (just written), so distance = ball offset magnitude.
    rb_distance = local_offset.norm(dim=-1)
    self._prev_robot_ball_distance[env_ids] = rb_distance

    self.kick_contact_awarded[env_ids] = False
    self.kick_contact_new[env_ids] = False
    self.peak_kick_speed[env_ids] = 0.0
    self.steps_since_kick[env_ids] = -1

    if self.perception is not None:
      self.perception.reset(env_ids)

  def _update_command(self) -> None:
    self._ball_pos_w = self.ball.data.root_link_pos_w.clone()
    self._ball_vel_w = self.ball.data.root_link_lin_vel_w.clone()
    if self.perception is not None:
      self.perception.update(self.robot, self._ball_pos_w)
    self._detect_kick_contact()

  def _detect_kick_contact(self) -> None:
    self.kick_contact_new.zero_()
    if self.cfg.ball_sensor_name is None:
      return
    sensor = self._env.scene.sensors.get(self.cfg.ball_sensor_name)
    if sensor is None:
      return
    data = sensor.data
    forces = data.force_history if data.force_history is not None else data.force
    if forces is None or forces.numel() == 0:
      return
    if forces.ndim == 4:
      force_norm = forces.norm(dim=-1).amax(dim=2)
    elif forces.ndim == 3:
      force_norm = forces.norm(dim=-1)
    else:
      force_norm = forces.norm(dim=-1, keepdim=True)
    peak = force_norm.amax(dim=-1) if force_norm.ndim > 1 else force_norm
    contact = peak > self.cfg.horizontal_force_threshold

    foot_vel = self.robot.data.body_link_lin_vel_w[:, self._foot_body_indices]
    foot_speed = foot_vel.norm(dim=-1).amax(dim=-1)
    contact = contact & (foot_speed > self.cfg.min_foot_speed)
    self.peak_kick_speed = torch.maximum(self.peak_kick_speed, foot_speed)

    new = contact & ~self.kick_contact_awarded
    self.kick_contact_new = new
    self.kick_contact_awarded = self.kick_contact_awarded | new
    # Tick the post-kick counter. Set to 0 on the first kick step;
    # increment on subsequent steps. Remains -1 until the first kick.
    started = new & (self.steps_since_kick < 0)
    self.steps_since_kick = torch.where(
      started,
      torch.zeros_like(self.steps_since_kick),
      torch.where(
        self.steps_since_kick >= 0,
        self.steps_since_kick + 1,
        self.steps_since_kick,
      ),
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
      # Record before super().reset() zeros the per-env metric tensors.
      self.metrics["kick_success_rate"][env_ids] = self.kick_contact_awarded[
        env_ids
      ].float()
      self.metrics["peak_kick_speed"][env_ids] = self.peak_kick_speed[env_ids]
    return super().reset(env_ids)

  # -- Properties for observations / rewards --

  @property
  def ball_pos_w(self) -> torch.Tensor:
    return self._ball_pos_w

  @property
  def ball_vel_w(self) -> torch.Tensor:
    return self._ball_vel_w

  @property
  def ball_pos_b(self) -> torch.Tensor:
    """Ground-truth ball xy in robot body yaw frame (privileged)."""
    return self._world_to_body_xy(self._ball_pos_w[:, :2])

  @property
  def ball_vel_b(self) -> torch.Tensor:
    """Ground-truth ball velocity in robot body yaw frame (privileged)."""
    yq = yaw_quat(self.robot.data.root_link_quat_w)
    return quat_apply_inverse(yq, self._ball_vel_w)

  @property
  def ball_pos_b_perceived(self) -> torch.Tensor:
    if self.perception is not None:
      return self.perception.ball_pos_b
    return self.ball_pos_b

  @property
  def ball_mask(self) -> torch.Tensor:
    if self.perception is not None:
      return self.perception.ball_mask
    return torch.ones(self.num_envs, device=self.device)

  @property
  def kick_target_dir_w(self) -> torch.Tensor:
    return self._target_dir_w

  @property
  def kick_target_dir_b(self) -> torch.Tensor:
    """Target direction (cos, sin) rotated into robot body yaw frame."""
    dir_w = torch.zeros(self.num_envs, 3, device=self.device)
    dir_w[:, :2] = self._target_dir_w
    yq = yaw_quat(self.robot.data.root_link_quat_w)
    rotated = quat_apply_inverse(yq, dir_w)
    return rotated[:, :2]

  @property
  def ball_progress(self) -> torch.Tensor:
    """Signed projection of ball xy onto target direction (world frame)."""
    return (
      self._ball_pos_w[:, 0] * self._target_dir_w[:, 0]
      + self._ball_pos_w[:, 1] * self._target_dir_w[:, 1]
    )

  @property
  def ball_progress_delta(self) -> torch.Tensor:
    """Change in progress since last step (positive = ball moved toward target)."""
    delta = self.ball_progress - self._prev_ball_progress
    self._prev_ball_progress = self.ball_progress.clone()
    return delta

  @property
  def robot_ball_distance(self) -> torch.Tensor:
    robot_xy = self.robot.data.root_link_pos_w[:, :2]
    return (robot_xy - self._ball_pos_w[:, :2]).norm(dim=-1)

  @property
  def robot_ball_distance_delta(self) -> torch.Tensor:
    """Negative when robot is approaching ball (good)."""
    d = self.robot_ball_distance
    delta = d - self._prev_robot_ball_distance
    self._prev_robot_ball_distance = d.clone()
    return delta

  def _world_to_body_xy(self, xy_w: torch.Tensor) -> torch.Tensor:
    rel_w = torch.zeros(xy_w.shape[0], 3, device=self.device)
    rel_w[:, :2] = xy_w - self.robot.data.root_link_pos_w[:, :2]
    yq = yaw_quat(self.robot.data.root_link_quat_w)
    rel_b = quat_apply_inverse(yq, rel_w)
    return rel_b[:, :2]

  def _debug_vis_impl(self, visualizer) -> None:
    """Render debug overlays each frame.

    - Green arrow at the ball pointing in the commanded kick direction.
    - Yellow camera FOV frustum from the head-mounted virtual camera, so
      it is visually clear which region the policy can actually see.
    - Sphere at the policy's perceived ball position (cyan when the ball
      is detected this step, magenta when ``ball_mask = 0``).
    """
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    arrow_length = 0.5
    for i in env_indices:
      ball_xy = self._ball_pos_w[i].detach().cpu().numpy()
      direction = self._target_dir_w[i].detach().cpu().numpy()
      start = np.asarray([float(ball_xy[0]), float(ball_xy[1]), 0.15], dtype=np.float32)
      end = start + np.asarray(
        [
          float(direction[0]) * arrow_length,
          float(direction[1]) * arrow_length,
          0.0,
        ],
        dtype=np.float32,
      )
      visualizer.add_arrow(
        start=start,
        end=end,
        color=(0.1, 0.85, 0.1, 0.95),
        width=0.02,
        label=f"kick_target_dir/env_{i}",
      )

    if self.perception is None:
      return

    self._draw_perception_overlays(visualizer, env_indices)

  def _draw_perception_overlays(self, visualizer, env_indices) -> None:
    """FOV frustum + perceived-ball sphere for the play-mode viewer."""
    perception = self.perception
    assert perception is not None
    cam_pos = perception._cam_pos_w.detach().cpu().numpy()
    cam_quat = perception._cam_quat_w.detach().cpu().numpy()
    fov_h = perception._fov_h_per_env.detach().cpu().numpy()
    fov_v = perception._fov_v_per_env.detach().cpu().numpy()
    ball_pos_b = perception.ball_pos_b.detach().cpu().numpy()
    ball_mask = perception.ball_mask.detach().cpu().numpy()
    robot_pos = self.robot.data.root_link_pos_w.detach().cpu().numpy()
    robot_quat = self.robot.data.root_link_quat_w.detach().cpu().numpy()

    frustum_range = 2.0
    for i in env_indices:
      tan_h = math.tan(float(fov_h[i]))
      tan_v = math.tan(float(fov_v[i]))
      corners_cam = np.array(
        [
          [frustum_range, +frustum_range * tan_h, +frustum_range * tan_v],
          [frustum_range, -frustum_range * tan_h, +frustum_range * tan_v],
          [frustum_range, -frustum_range * tan_h, -frustum_range * tan_v],
          [frustum_range, +frustum_range * tan_h, -frustum_range * tan_v],
        ],
        dtype=np.float32,
      )
      corners_w = np.stack(
        [_rotate_by_quat_np(cam_quat[i], c) + cam_pos[i] for c in corners_cam]
      )
      origin = cam_pos[i].astype(np.float32)
      fov_color = (0.95, 0.85, 0.15, 0.85)
      for k, corner in enumerate(corners_w):
        visualizer.add_arrow(
          start=origin,
          end=corner.astype(np.float32),
          color=fov_color,
          width=0.005,
          label=f"fov_edge/env_{i}_{k}",
        )
      for k in range(4):
        a = corners_w[k].astype(np.float32)
        b = corners_w[(k + 1) % 4].astype(np.float32)
        visualizer.add_cylinder(
          start=a,
          end=b,
          radius=0.005,
          color=fov_color,
          label=f"fov_rim/env_{i}_{k}",
        )

      detected = float(ball_mask[i]) > 0.5
      sphere_color = (0.2, 0.95, 0.95, 0.95) if detected else (0.95, 0.2, 0.85, 0.5)
      yaw = _yaw_from_quat(robot_quat[i])
      cos_y, sin_y = math.cos(yaw), math.sin(yaw)
      bx_b, by_b = float(ball_pos_b[i, 0]), float(ball_pos_b[i, 1])
      perceived_xy_w = np.array(
        [
          robot_pos[i, 0] + cos_y * bx_b - sin_y * by_b,
          robot_pos[i, 1] + sin_y * bx_b + cos_y * by_b,
        ],
        dtype=np.float32,
      )
      perceived_xyz = np.array(
        [perceived_xy_w[0], perceived_xy_w[1], 0.11], dtype=np.float32
      )
      visualizer.add_sphere(
        center=perceived_xyz,
        radius=0.08,
        color=sphere_color,
        label=f"perceived_ball/env_{i}",
      )


def _rotate_by_quat_np(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
  """Rotate a 3-vector by a wxyz quaternion (numpy, debug-viz only)."""
  w, x, y, z = (
    float(quat_wxyz[0]),
    float(quat_wxyz[1]),
    float(quat_wxyz[2]),
    float(quat_wxyz[3]),
  )
  vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  rx = vx + w * tx + (y * tz - z * ty)
  ry = vy + w * ty + (z * tx - x * tz)
  rz = vz + w * tz + (x * ty - y * tx)
  return np.array([rx, ry, rz], dtype=np.float32)


def _yaw_from_quat(quat_wxyz: np.ndarray) -> float:
  """Extract yaw (rotation about world Z) from a wxyz quaternion (numpy)."""
  w, x, y, z = (
    float(quat_wxyz[0]),
    float(quat_wxyz[1]),
    float(quat_wxyz[2]),
    float(quat_wxyz[3]),
  )
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  return math.atan2(siny_cosp, cosy_cosp)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  from mjlab.utils.lab_api.math import quat_apply as _qa

  return _qa(quat, vec)
