"""Virtual perception system for the kick task.

Reproduces the LVDRS sim2real perception model (arXiv:2511.03996, Section 4.3
+ Appendix E): the policy receives noisy/intermittent ball detections that
match the characteristics of an on-robot camera + detection pipeline rather
than ground-truth ball state. Defaults are tuned for Booster K1 (RealSense
D435i mounted on Head_2).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass
class VirtualPerceptionCfg:
  """Parameters for the simulated head-camera ball detector."""

  camera_body_name: str = "Head_2"
  """Robot body the camera is rigidly attached to (post-yaw, post-pitch)."""

  camera_offset_pos: tuple[float, float, float] = (0.05, 0.04553, 0.10)
  """Camera optical-frame origin offset in the camera body's local frame (m).

  Head_2 body origin is at the head_pitch joint (≈ neck base on K1). The
  RealSense D435i is forehead-mounted, so the offset puts the optical
  origin +5cm forward, +4.5cm to the side (measured from the actual K1
  CAD), +10cm up (forehead, in line with Head_2 inertial COM at +8cm)."""

  camera_offset_quat: tuple[float, float, float, float] = (
    0.0054,
    0.9986,
    -0.0028,
    0.0511,
  )
  """Camera optical-frame orientation relative to the camera body (wxyz)."""

  fov_h: float = math.radians(105.12) * 0.5
  """Horizontal FOV half-angle (radians). Default: K1 RealSense D435i."""

  fov_v: float = math.radians(94.17) * 0.5
  """Vertical FOV half-angle (radians)."""

  detection_prob_in_fov: float = 0.9
  """Bernoulli detection probability when the ball is inside FOV and range."""

  max_detection_range: float = 7.0
  """Distance beyond which detection probability decays to zero (m)."""

  range_decay: float = 2.0
  """Soft falloff distance for range-based detection probability (m)."""

  noise_a: float = 0.05
  """Linear coefficient in distance-dependent noise model: sigma(d) = a*d + b.

  V1.35: reduced 0.124 → 0.05. Original LVDRS-paper value made the
  perceived ball jitter by ±27cm at 1m, far worse than a real RealSense
  D435i + ball detector (typical ±5cm at 1m). The previous value bought
  unrealistic robustness while making the displayed perception look
  obviously wrong. Real sensor + 2× sim2real margin gives a*d+b ≈
  0.05*d + 0.08."""

  noise_b: float = 0.08
  """Offset coefficient: sigma(d) = a*d + b (meters).

  V1.35: reduced 0.149 → 0.08 (see noise_a comment)."""

  latency_mean_s: float = 0.116
  """Mean detection-to-policy latency (seconds)."""

  latency_std_s: float = 0.018
  """Std of latency Gaussian (seconds)."""

  update_hz_mean: float = 25.36
  """Mean detector update frequency (Hz)."""

  update_hz_std: float = 1.06
  """Std of update frequency Gaussian (Hz)."""

  buffer_size: int = 16
  """Latency ring buffer length in control steps. Should cover
  ``ceil((latency_mean + 3*latency_std) / dt)``.
  """

  hold_last_on_miss: bool = True
  """When detection fails, hold the most recent valid value (true) or output
  zeros (false). ``ball_mask`` is set to 0 either way.
  """

  # ---------------------------------------------------------------- DR knobs
  # All ranges are MULTIPLICATIVE scaling factors sampled per-env at reset,
  # except detection_prob_in_fov_range which replaces the base value with a
  # per-env absolute probability. The base values above stay as the central
  # / nominal values around which the per-env values vary.
  noise_a_range: tuple[float, float] = (0.7, 1.5)
  """Per-env multiplicative range for ``noise_a`` (distance-dependent stdev
  slope). Cameras with worse focus / depth estimation get larger values."""

  noise_b_range: tuple[float, float] = (0.7, 1.5)
  """Per-env multiplicative range for ``noise_b`` (constant noise offset)."""

  detection_prob_in_fov_range: tuple[float, float] = (0.70, 0.95)
  """Per-env absolute range for in-FOV detection probability. Cameras with
  bad exposure / partial occlusion miss balls more often."""

  fov_scale_range: tuple[float, float] = (0.85, 1.0)
  """Per-env multiplicative scaling on the FOV half-angles. Modeling
  miscalibrated / lower-quality cameras with effectively smaller FOV."""

  latency_mean_range: tuple[float, float] = (0.080, 0.160)
  """Per-env range for the latency Gaussian mean (seconds). Replaces
  ``latency_mean_s`` with a per-env value."""

  update_hz_mean_range: tuple[float, float] = (20.0, 30.0)
  """Per-env range for the update-frequency Gaussian mean (Hz). Replaces
  ``update_hz_mean`` with a per-env value."""


@dataclass
class _PerceptionState:
  buffer: deque[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=deque)
  """Per-env ring buffer of (ball_pos_b, mask) pairs, newest at right."""


class VirtualPerception:
  """Stateful simulator of head-camera ball detection.

  Each ``update`` step:
    1. Compute the ball position in the camera optical frame.
    2. Check FOV (angular limits + ball must be forward).
    3. Apply a distance-attenuated Bernoulli detection probability.
    4. Inject Gaussian noise scaled by distance.
    5. Push (pos, mask) into per-env latency buffer.
    6. Read the buffer entry corresponding to each env's sampled latency.
    7. Decimate to the sampled detector update rate (hold otherwise).
  """

  def __init__(
    self,
    cfg: VirtualPerceptionCfg,
    num_envs: int,
    dt: float,
    device: torch.device | str,
  ) -> None:
    self.cfg = cfg
    self.num_envs = num_envs
    self.dt = dt
    self.device = torch.device(device)

    self._cam_offset_pos = torch.tensor(
      cfg.camera_offset_pos, dtype=torch.float32, device=self.device
    )
    self._cam_offset_quat = torch.tensor(
      cfg.camera_offset_quat, dtype=torch.float32, device=self.device
    )

    self._buffer_pos = torch.zeros(
      cfg.buffer_size, num_envs, 2, dtype=torch.float32, device=self.device
    )
    self._buffer_mask = torch.zeros(
      cfg.buffer_size, num_envs, dtype=torch.float32, device=self.device
    )
    self._buffer_head = 0

    self._latency_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    self._update_period_steps = torch.ones(
      num_envs, dtype=torch.long, device=self.device
    )
    self._steps_since_update = torch.zeros(
      num_envs, dtype=torch.long, device=self.device
    )
    self._last_pos = torch.zeros(num_envs, 2, dtype=torch.float32, device=self.device)
    self._last_mask = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

    self._ball_pos_b = torch.zeros_like(self._last_pos)
    self._ball_mask = torch.zeros_like(self._last_mask)

    # Cached camera pose from the most recent update() — used by the
    # play-time debug visualizer (see KickTargetCommand._debug_vis_impl).
    self._cam_pos_w = torch.zeros(num_envs, 3, dtype=torch.float32, device=self.device)
    self._cam_quat_w = torch.zeros(num_envs, 4, dtype=torch.float32, device=self.device)
    self._cam_quat_w[:, 0] = 1.0  # identity quat (wxyz)

    # Per-env DR coefficients (V1.23). Sampled per env at reset; gives a
    # heterogeneous "camera fleet" so the policy generalizes to deploy-
    # time camera variations rather than overfitting to the paper's
    # single nominal sensor model.
    self._noise_a_per_env = torch.full(
      (num_envs,), cfg.noise_a, dtype=torch.float32, device=self.device
    )
    self._noise_b_per_env = torch.full(
      (num_envs,), cfg.noise_b, dtype=torch.float32, device=self.device
    )
    self._detection_prob_per_env = torch.full(
      (num_envs,),
      cfg.detection_prob_in_fov,
      dtype=torch.float32,
      device=self.device,
    )
    self._fov_h_per_env = torch.full(
      (num_envs,), cfg.fov_h, dtype=torch.float32, device=self.device
    )
    self._fov_v_per_env = torch.full(
      (num_envs,), cfg.fov_v, dtype=torch.float32, device=self.device
    )

    self._sample_latency_and_rate(
      torch.arange(num_envs, dtype=torch.long, device=self.device)
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self._buffer_pos[:, env_ids] = 0.0
    self._buffer_mask[:, env_ids] = 0.0
    self._steps_since_update[env_ids] = 0
    self._last_pos[env_ids] = 0.0
    self._last_mask[env_ids] = 0.0
    self._ball_pos_b[env_ids] = 0.0
    self._ball_mask[env_ids] = 0.0
    self._sample_latency_and_rate(env_ids)

  def _sample_latency_and_rate(self, env_ids: torch.Tensor) -> None:
    cfg = self.cfg
    n = env_ids.numel()
    if n == 0:
      return

    # V1.23 DR: per-env coefficient sampling.
    def _uniform(lo: float, hi: float) -> torch.Tensor:
      return torch.rand(n, device=self.device) * (hi - lo) + lo

    self._noise_a_per_env[env_ids] = cfg.noise_a * _uniform(*cfg.noise_a_range)
    self._noise_b_per_env[env_ids] = cfg.noise_b * _uniform(*cfg.noise_b_range)
    self._detection_prob_per_env[env_ids] = _uniform(*cfg.detection_prob_in_fov_range)
    self._fov_h_per_env[env_ids] = cfg.fov_h * _uniform(*cfg.fov_scale_range)
    self._fov_v_per_env[env_ids] = cfg.fov_v * _uniform(*cfg.fov_scale_range)

    latency_mean = _uniform(*cfg.latency_mean_range)
    latency = (
      latency_mean + torch.randn(n, device=self.device) * cfg.latency_std_s
    ).clamp_min(0.0)
    self._latency_steps[env_ids] = (
      (latency / self.dt).round().long().clamp_(0, cfg.buffer_size - 1)
    )

    hz_mean = _uniform(*cfg.update_hz_mean_range)
    hz = (hz_mean + torch.randn(n, device=self.device) * cfg.update_hz_std).clamp_min(
      1.0
    )
    period_s = 1.0 / hz
    self._update_period_steps[env_ids] = (
      (period_s / self.dt).round().long().clamp_min_(1)
    )

  @torch.no_grad()
  def update(
    self,
    robot: "Entity",
    ball_pos_w: torch.Tensor,
  ) -> None:
    """Advance one control step.

    Args:
      robot: scene entity holding the camera body.
      ball_pos_w: ``(num_envs, 3)`` ball world position.
    """
    cfg = self.cfg
    body_names = robot.body_names
    if cfg.camera_body_name not in body_names:
      raise ValueError(
        f"camera_body_name {cfg.camera_body_name!r} not in robot bodies "
        f"{list(body_names)}"
      )
    cam_body_idx = body_names.index(cfg.camera_body_name)
    body_pos = robot.data.body_link_pos_w[:, cam_body_idx]
    body_quat = robot.data.body_link_quat_w[:, cam_body_idx]

    cam_pos_w = body_pos + _quat_apply(body_quat, self._cam_offset_pos)
    cam_quat_w = quat_mul(
      body_quat,
      self._cam_offset_quat.expand(body_quat.shape[0], -1),
    )
    self._cam_pos_w.copy_(cam_pos_w)
    self._cam_quat_w.copy_(cam_quat_w)

    ball_in_cam = quat_apply_inverse(cam_quat_w, ball_pos_w - cam_pos_w)
    bx, by, bz = ball_in_cam.unbind(dim=-1)
    distance = ball_in_cam.norm(dim=-1)

    forward = bx > 1e-3
    yaw = torch.atan2(by, bx)
    pitch = torch.atan2(bz, bx)
    # V1.23: use per-env FOV / detection probability / noise coefficients
    # so each parallel env sees a slightly different "camera". The cfg
    # constants are only the nominal centers; actual values vary across
    # envs (sampled at reset in _sample_latency_and_rate).
    in_fov = (
      forward & (yaw.abs() < self._fov_h_per_env) & (pitch.abs() < self._fov_v_per_env)
    )

    range_prob = torch.where(
      distance < cfg.max_detection_range,
      torch.ones_like(distance),
      torch.clamp(
        1.0 - (distance - cfg.max_detection_range) / max(cfg.range_decay, 1e-6),
        min=0.0,
      ),
    )
    p_detect = self._detection_prob_per_env * range_prob * in_fov.float()
    detected = torch.bernoulli(p_detect).to(torch.bool)

    sigma = self._noise_a_per_env * distance + self._noise_b_per_env
    noise = torch.randn_like(ball_pos_w[:, :2]) * sigma.unsqueeze(-1)
    ball_xy_world = ball_pos_w[:, :2] + noise

    # Express noisy ball xy in robot body yaw frame (z-axis ignored — flat task).
    robot_pos = robot.data.root_link_pos_w
    robot_quat = robot.data.root_link_quat_w
    rel_xy = ball_xy_world - robot_pos[:, :2]
    rel_xy_b = _yaw_rotate_inverse(robot_quat, rel_xy)

    detected_f = detected.float()
    new_pos = torch.where(detected.unsqueeze(-1), rel_xy_b, self._last_pos)

    self._steps_since_update += 1
    do_update = (self._steps_since_update >= self._update_period_steps) | (
      self._last_mask < 0.5
    )
    self._last_pos = torch.where(
      do_update.unsqueeze(-1) & detected.unsqueeze(-1), new_pos, self._last_pos
    )
    self._last_mask = torch.where(
      do_update,
      detected_f,
      self._last_mask,
    )
    self._steps_since_update = torch.where(
      do_update,
      torch.zeros_like(self._steps_since_update),
      self._steps_since_update,
    )

    self._buffer_head = (self._buffer_head + 1) % cfg.buffer_size
    self._buffer_pos[self._buffer_head] = self._last_pos
    self._buffer_mask[self._buffer_head] = self._last_mask

    env_idx = torch.arange(self.num_envs, device=self.device)
    read_head = (self._buffer_head - self._latency_steps) % cfg.buffer_size
    out_pos = self._buffer_pos[read_head, env_idx]
    out_mask = self._buffer_mask[read_head, env_idx]

    if not cfg.hold_last_on_miss:
      out_pos = out_pos * out_mask.unsqueeze(-1)

    self._ball_pos_b = out_pos
    self._ball_mask = out_mask

  @property
  def ball_pos_b(self) -> torch.Tensor:
    return self._ball_pos_b

  @property
  def ball_mask(self) -> torch.Tensor:
    return self._ball_mask


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Apply quaternion to a vector. Broadcasts ``vec`` across the batch."""
  from mjlab.utils.lab_api.math import quat_apply as _qa

  if vec.dim() == 1:
    vec = vec.expand(quat.shape[0], -1)
  return _qa(quat, vec)


def _yaw_rotate_inverse(quat: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
  """Rotate an xy vector into the body-yaw frame.

  Uses only the yaw component of ``quat`` so the result is stable under
  body pitch/roll (flat-ground task).
  """
  from mjlab.utils.lab_api.math import yaw_quat

  yq = yaw_quat(quat)
  vec3 = torch.zeros(xy.shape[0], 3, device=xy.device, dtype=xy.dtype)
  vec3[:, :2] = xy
  rotated = quat_apply_inverse(yq, vec3)
  return rotated[:, :2]
