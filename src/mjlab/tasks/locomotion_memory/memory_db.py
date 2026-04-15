from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

GAIT_NAME_TO_ID = {
  "walk": 0,
  "run": 1,
  "turn": 2,
  "start": 3,
  "stop": 4,
}
TRANSITION_NAME_TO_ID = {
  "none": 0,
  "walk_to_run": 1,
  "run_to_walk": 2,
}
DEFAULT_PROPOSAL_HORIZONS_S = (0.2, 0.4, 0.6, 0.8)
DEFAULT_K1_SOURCE_DIR = Path("/workspace/k1_retarget/motions_k1/k1_fixed")
DEFAULT_K1_DATA_DIR = (
  Path(__file__).resolve().parents[4] / "artifacts" / "locomotion_memory" / "booster_k1"
)
DEFAULT_K1_CONVERTED_DIR = DEFAULT_K1_DATA_DIR / "converted"
DEFAULT_K1_MEMORY_FILE = DEFAULT_K1_DATA_DIR / "locomotion_memory.npz"
K1_MEMORY_FILE_ENV_VAR = "MJLAB_K1_LOCOMOTION_MEMORY_FILE"
DEFAULT_K1_LOCOMOTION_INCLUDE_GLOBS = (
  "walk_forward_*.csv",
  "walk_fast_steady_*.csv",
  "jog_*.csv",
  "run_forward_*.csv",
  "run_steady_*.csv",
  "steady_*.csv",
  "walk_start_stop_*.csv",
  "walk_accel_*.csv",
  "walk_decel_*.csv",
  "pivot_*.csv",
  "turn_*.csv",
)
DEFAULT_K1_LOCOMOTION_EXCLUDE_GLOBS = (
  "*backward*",
  "*lateral*",
  "*shuffle*",
  "stand_*.csv",
  "turn_in_place_left_*.csv",
  "turn_in_place_right_*.csv",
)


@dataclass(frozen=True)
class MotionClip:
  path: Path
  fps: float
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  body_pos_w: np.ndarray
  body_quat_w: np.ndarray
  body_lin_vel_w: np.ndarray
  body_ang_vel_w: np.ndarray
  joint_names: tuple[str, ...]
  body_names: tuple[str, ...]


class LocomotionMemoryLoader:
  """Loads a locomotion-memory NPZ file onto the requested device."""

  proposal_dim: int
  snippet_dt: float
  horizon_frames: torch.Tensor
  joint_names: tuple[str, ...]
  source_clip_names: tuple[str, ...]
  num_snippets: int
  snippet_length: int
  has_local_foot_pos_seq: bool
  joint_pos_seq: torch.Tensor
  joint_vel_seq: torch.Tensor
  local_root_pos_seq: torch.Tensor
  local_root_yaw_seq: torch.Tensor
  local_foot_pos_seq: torch.Tensor
  pelvis_height_seq: torch.Tensor
  contact_seq: torch.Tensor
  phase_seq: torch.Tensor
  mean_vx: torch.Tensor
  mean_vy: torch.Tensor
  mean_yaw_rate: torch.Tensor
  phase_start: torch.Tensor
  phase_end: torch.Tensor
  phase_start_sin: torch.Tensor
  phase_start_cos: torch.Tensor
  phase_end_sin: torch.Tensor
  phase_end_cos: torch.Tensor
  left_contact_ratio: torch.Tensor
  right_contact_ratio: torch.Tensor
  left_contact_start: torch.Tensor
  right_contact_start: torch.Tensor
  duty_factor: torch.Tensor
  cadence: torch.Tensor
  step_length: torch.Tensor
  com_height_mean: torch.Tensor
  pelvis_bounce_amp: torch.Tensor
  arm_swing_amp: torch.Tensor
  quality_score: torch.Tensor
  gait_id: torch.Tensor
  transition_type: torch.Tensor
  transition_flag: torch.Tensor
  source_clip_id: torch.Tensor

  def __init__(self, memory_file: str, device: str | torch.device = "cpu") -> None:
    data = np.load(memory_file)
    self.proposal_dim = int(np.asarray(data["proposal_dim"]).item())
    self.snippet_dt = float(np.asarray(data["snippet_dt"]).item())
    self.horizon_frames = torch.tensor(
      data["horizon_frames"], dtype=torch.long, device=device
    )
    self.joint_names = tuple(str(name) for name in data["joint_names"])
    self.source_clip_names = tuple(str(name) for name in data["source_clip_names"])
    self.num_snippets = int(data["joint_pos_seq"].shape[0])
    self.snippet_length = int(data["joint_pos_seq"].shape[1])

    required_sequence_keys = (
      "joint_pos_seq",
      "joint_vel_seq",
      "local_root_pos_seq",
      "local_root_yaw_seq",
      "contact_seq",
      "phase_seq",
    )
    for key in required_sequence_keys:
      if key == "contact_seq":
        value = torch.tensor(data[key], dtype=torch.bool, device=device)
      else:
        value = torch.tensor(data[key], device=device)
      setattr(self, key, value)

    self.has_local_foot_pos_seq = "local_foot_pos_seq" in data
    if self.has_local_foot_pos_seq:
      self.local_foot_pos_seq = torch.tensor(data["local_foot_pos_seq"], device=device)
    else:
      self.local_foot_pos_seq = torch.zeros(
        self.num_snippets,
        self.snippet_length,
        2,
        3,
        dtype=self.joint_pos_seq.dtype,
        device=device,
      )

    for key in (
      "pelvis_height_seq",
      "mean_vx",
      "mean_vy",
      "mean_yaw_rate",
      "phase_start",
      "phase_end",
      "phase_start_sin",
      "phase_start_cos",
      "phase_end_sin",
      "phase_end_cos",
      "left_contact_ratio",
      "right_contact_ratio",
      "left_contact_start",
      "right_contact_start",
      "duty_factor",
      "cadence",
      "step_length",
      "com_height_mean",
      "pelvis_bounce_amp",
      "arm_swing_amp",
      "quality_score",
      "gait_id",
      "transition_type",
      "transition_flag",
      "source_clip_id",
    ):
      setattr(self, key, torch.tensor(data[key], device=device))


def default_k1_locomotion_memory_file() -> str:
  return os.getenv(K1_MEMORY_FILE_ENV_VAR, str(DEFAULT_K1_MEMORY_FILE))


def discover_k1_memory_file() -> str | None:
  memory_file = Path(default_k1_locomotion_memory_file())
  if memory_file.is_file():
    return str(memory_file)
  return None


def select_k1_locomotion_csv_files(
  source_dir: str | Path = DEFAULT_K1_SOURCE_DIR,
  include_globs: tuple[str, ...] = DEFAULT_K1_LOCOMOTION_INCLUDE_GLOBS,
  exclude_globs: tuple[str, ...] = DEFAULT_K1_LOCOMOTION_EXCLUDE_GLOBS,
) -> list[str]:
  root = Path(source_dir)
  if not root.is_dir():
    raise FileNotFoundError(f"K1 source motion directory does not exist: {root}")

  included: dict[Path, None] = {}
  for pattern in include_globs:
    for candidate in sorted(root.glob(pattern)):
      included[candidate] = None
  for pattern in exclude_globs:
    for candidate in root.glob(pattern):
      included.pop(candidate, None)
  return [str(path) for path in sorted(included)]


def encode_proposals(
  local_root_pos_seq: torch.Tensor,
  local_root_yaw_seq: torch.Tensor,
  pelvis_height_seq: torch.Tensor,
  contact_seq: torch.Tensor,
  frame_indices: torch.Tensor,
  horizon_frames: torch.Tensor,
) -> torch.Tensor:
  """Encode proposal vectors from snippet trajectories.

  Args:
    local_root_pos_seq: [B, T, 2] local x/y positions in snippet-start frame.
    local_root_yaw_seq: [B, T] local yaw relative to snippet start.
    pelvis_height_seq: [B, T]
    contact_seq: [B, T, 2]
    frame_indices: [B]
    horizon_frames: [H]
  Returns:
    Proposal tensor of shape [B, H * 6].
  """
  batch_size, snippet_len, _ = local_root_pos_seq.shape
  horizon_count = int(horizon_frames.numel())
  batch_ids = torch.arange(batch_size, device=local_root_pos_seq.device)
  current_idx = torch.clamp(frame_indices, 0, snippet_len - 1)
  future_idx = torch.clamp(
    current_idx[:, None] + horizon_frames[None, :], 0, snippet_len - 1
  )

  current_pos = local_root_pos_seq[batch_ids, current_idx]
  future_pos = local_root_pos_seq[batch_ids[:, None], future_idx]
  delta_pos = future_pos - current_pos[:, None, :]

  current_yaw = local_root_yaw_seq[batch_ids, current_idx]
  future_yaw = local_root_yaw_seq[batch_ids[:, None], future_idx]
  cos_yaw = torch.cos(current_yaw)[:, None]
  sin_yaw = torch.sin(current_yaw)[:, None]

  dx = cos_yaw * delta_pos[..., 0] + sin_yaw * delta_pos[..., 1]
  dy = -sin_yaw * delta_pos[..., 0] + cos_yaw * delta_pos[..., 1]
  dyaw = torch.atan2(
    torch.sin(future_yaw - current_yaw[:, None]),
    torch.cos(future_yaw - current_yaw[:, None]),
  )
  pelvis_h = pelvis_height_seq[batch_ids[:, None], future_idx]
  contacts = contact_seq[batch_ids[:, None], future_idx].to(dtype=dx.dtype)

  proposal = torch.stack(
    (dx, dy, dyaw, pelvis_h, contacts[..., 0], contacts[..., 1]),
    dim=-1,
  )
  return proposal.reshape(batch_size, horizon_count * 6)


def build_memory_database(
  input_files: list[str],
  *,
  anchor_body_name: str = "Trunk",
  left_foot_body_name: str = "left_foot_link",
  right_foot_body_name: str = "right_foot_link",
  left_hand_body_name: str = "left_hand_link",
  right_hand_body_name: str = "right_hand_link",
  snippet_length_s: float = 0.8,
  stride_frames: int = 6,
  contact_height_threshold: float = 0.03,
  contact_velocity_threshold: float = 0.15,
  transition_speed_threshold: float = 1.5,
  min_quality_score: float = 0.05,
  proposal_horizons_s: tuple[float, ...] = DEFAULT_PROPOSAL_HORIZONS_S,
) -> dict[str, np.ndarray]:
  """Build a structured locomotion-memory database from converted motion clips."""
  motion_files = [Path(path) for path in input_files]
  if not motion_files:
    raise ValueError("No motion files were provided.")

  clips = [_load_motion_clip(path) for path in motion_files]
  _validate_clips(clips)

  fps = clips[0].fps
  snippet_length = int(round(snippet_length_s * fps))
  if snippet_length <= 1:
    raise ValueError("snippet_length_s must produce at least 2 frames.")
  horizon_frames = np.array(
    [min(int(round(h * fps)), snippet_length - 1) for h in proposal_horizons_s],
    dtype=np.int64,
  )

  joint_names = np.asarray(clips[0].joint_names, dtype=str)
  source_clip_names = np.asarray([clip.path.stem for clip in clips], dtype=str)

  records: dict[str, list[np.ndarray]] = {
    "joint_pos_seq": [],
    "joint_vel_seq": [],
    "local_root_pos_seq": [],
    "local_root_yaw_seq": [],
    "local_foot_pos_seq": [],
    "pelvis_height_seq": [],
    "contact_seq": [],
    "phase_seq": [],
    "mean_vx": [],
    "mean_vy": [],
    "mean_yaw_rate": [],
    "phase_start": [],
    "phase_end": [],
    "phase_start_sin": [],
    "phase_start_cos": [],
    "phase_end_sin": [],
    "phase_end_cos": [],
    "left_contact_ratio": [],
    "right_contact_ratio": [],
    "left_contact_start": [],
    "right_contact_start": [],
    "duty_factor": [],
    "cadence": [],
    "step_length": [],
    "com_height_mean": [],
    "pelvis_bounce_amp": [],
    "arm_swing_amp": [],
    "quality_score": [],
    "gait_id": [],
    "transition_type": [],
    "transition_flag": [],
    "source_clip_id": [],
  }

  for clip_id, clip in enumerate(clips):
    anchor_idx = clip.body_names.index(anchor_body_name)
    left_foot_idx = clip.body_names.index(left_foot_body_name)
    right_foot_idx = clip.body_names.index(right_foot_body_name)
    left_hand_idx = clip.body_names.index(left_hand_body_name)
    right_hand_idx = clip.body_names.index(right_hand_body_name)

    root_pos = clip.body_pos_w[:, anchor_idx]
    root_quat = clip.body_quat_w[:, anchor_idx]
    root_yaw = _yaw_from_quat_wxyz(root_quat)
    foot_pos = clip.body_pos_w[:, [left_foot_idx, right_foot_idx], :]
    foot_vel = clip.body_lin_vel_w[:, [left_foot_idx, right_foot_idx], :]
    ground_height = float(np.min(foot_pos[..., 2]))
    foot_height = foot_pos[..., 2] - ground_height
    foot_speed = np.linalg.norm(foot_vel, axis=-1)
    contact = _annotate_contacts(
      foot_height,
      foot_speed,
      contact_height_threshold,
      contact_velocity_threshold,
    )
    phase = _compute_phase(contact[:, 0])
    hand_pos = clip.body_pos_w[:, [left_hand_idx, right_hand_idx], :]

    for start in range(0, clip.joint_pos.shape[0] - snippet_length + 1, stride_frames):
      end = start + snippet_length
      local_root_pos_seq = _make_local_root_pos_seq(
        root_pos[start:end], root_yaw[start]
      )
      local_root_yaw_seq = _wrap_to_pi(root_yaw[start:end] - root_yaw[start])
      local_foot_pos_seq = _make_local_body_pos_seq(
        foot_pos[start:end],
        root_pos[start:end],
        root_yaw[start:end],
      )
      joint_pos_seq = clip.joint_pos[start:end]
      joint_vel_seq = clip.joint_vel[start:end]
      pelvis_height_seq = root_pos[start:end, 2].astype(np.float32)
      contact_seq = contact[start:end].astype(np.int8)
      phase_seq = phase[start:end].astype(np.float32)

      duration = max((snippet_length - 1) / fps, 1e-6)
      mean_vx = local_root_pos_seq[-1, 0] / duration
      mean_vy = local_root_pos_seq[-1, 1] / duration
      mean_yaw_rate = local_root_yaw_seq[-1] / duration
      left_contact_ratio = float(contact_seq[:, 0].mean())
      right_contact_ratio = float(contact_seq[:, 1].mean())
      duty_factor = 0.5 * (left_contact_ratio + right_contact_ratio)
      cadence = _touchdown_count(contact_seq) / duration
      step_length = float(
        np.linalg.norm(local_root_pos_seq[-1] - local_root_pos_seq[0])
      )
      com_height_mean = float(pelvis_height_seq.mean())
      pelvis_bounce_amp = float(np.ptp(pelvis_height_seq))
      arm_swing_amp = float(
        _compute_arm_swing_amp(hand_pos[start:end], root_yaw[start])
      )
      quality_score = float(
        _compute_quality_score(
          local_root_pos_seq,
          pelvis_height_seq,
          foot_height[start:end],
          fps,
          contact_height_threshold,
        )
      )

      if not np.isfinite(quality_score) or quality_score < min_quality_score:
        continue

      gait_id = np.int64(_infer_gait_id(clip.path.stem))
      transition_type = np.int64(
        _infer_transition_type(
          clip.path.stem,
          local_root_pos_seq,
          fps,
          transition_speed_threshold,
        )
      )
      transition_flag = np.int8(transition_type != TRANSITION_NAME_TO_ID["none"])
      phase_start = float(phase_seq[0])
      phase_end = float(phase_seq[-1])

      records["joint_pos_seq"].append(joint_pos_seq.astype(np.float32))
      records["joint_vel_seq"].append(joint_vel_seq.astype(np.float32))
      records["local_root_pos_seq"].append(local_root_pos_seq.astype(np.float32))
      records["local_root_yaw_seq"].append(local_root_yaw_seq.astype(np.float32))
      records["local_foot_pos_seq"].append(local_foot_pos_seq.astype(np.float32))
      records["pelvis_height_seq"].append(pelvis_height_seq.astype(np.float32))
      records["contact_seq"].append(contact_seq)
      records["phase_seq"].append(phase_seq)
      records["mean_vx"].append(np.array(mean_vx, dtype=np.float32))
      records["mean_vy"].append(np.array(mean_vy, dtype=np.float32))
      records["mean_yaw_rate"].append(np.array(mean_yaw_rate, dtype=np.float32))
      records["phase_start"].append(np.array(phase_start, dtype=np.float32))
      records["phase_end"].append(np.array(phase_end, dtype=np.float32))
      records["phase_start_sin"].append(
        np.array(np.sin(2.0 * np.pi * phase_start), dtype=np.float32)
      )
      records["phase_start_cos"].append(
        np.array(np.cos(2.0 * np.pi * phase_start), dtype=np.float32)
      )
      records["phase_end_sin"].append(
        np.array(np.sin(2.0 * np.pi * phase_end), dtype=np.float32)
      )
      records["phase_end_cos"].append(
        np.array(np.cos(2.0 * np.pi * phase_end), dtype=np.float32)
      )
      records["left_contact_ratio"].append(
        np.array(left_contact_ratio, dtype=np.float32)
      )
      records["right_contact_ratio"].append(
        np.array(right_contact_ratio, dtype=np.float32)
      )
      records["left_contact_start"].append(np.array(contact_seq[0, 0], dtype=np.int8))
      records["right_contact_start"].append(np.array(contact_seq[0, 1], dtype=np.int8))
      records["duty_factor"].append(np.array(duty_factor, dtype=np.float32))
      records["cadence"].append(np.array(cadence, dtype=np.float32))
      records["step_length"].append(np.array(step_length, dtype=np.float32))
      records["com_height_mean"].append(np.array(com_height_mean, dtype=np.float32))
      records["pelvis_bounce_amp"].append(np.array(pelvis_bounce_amp, dtype=np.float32))
      records["arm_swing_amp"].append(np.array(arm_swing_amp, dtype=np.float32))
      records["quality_score"].append(np.array(quality_score, dtype=np.float32))
      records["gait_id"].append(np.array(gait_id, dtype=np.int64))
      records["transition_type"].append(np.array(transition_type, dtype=np.int64))
      records["transition_flag"].append(np.array(transition_flag, dtype=np.int8))
      records["source_clip_id"].append(np.array(clip_id, dtype=np.int64))

  if not records["joint_pos_seq"]:
    raise ValueError("No valid snippets were produced from the provided clips.")

  output: dict[str, np.ndarray] = {
    "proposal_dim": np.array(len(horizon_frames) * 6, dtype=np.int64),
    "snippet_dt": np.array(1.0 / fps, dtype=np.float32),
    "horizon_frames": horizon_frames,
    "joint_names": joint_names,
    "source_clip_names": source_clip_names,
  }
  for key, values in records.items():
    output[key] = np.stack(values, axis=0)
  return output


def resolve_motion_files(
  input_paths: list[str], include_glob: str = "*.npz"
) -> list[str]:
  resolved: list[str] = []
  for input_path in input_paths:
    path = Path(input_path)
    if path.is_dir():
      resolved.extend(str(candidate) for candidate in sorted(path.glob(include_glob)))
    elif path.is_file():
      resolved.append(str(path))
    else:
      raise FileNotFoundError(f"Input path does not exist: {path}")
  return resolved


def _load_motion_clip(path: Path) -> MotionClip:
  data = np.load(path)
  required = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
  }
  missing = sorted(required.difference(data.files))
  if missing:
    missing_str = ", ".join(missing)
    raise ValueError(
      f"Motion file {path} is missing required fields: {missing_str}. "
      "Rebuild it with the updated csv_to_npz converter that stores metadata."
    )
  return MotionClip(
    path=path,
    fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
    joint_pos=data["joint_pos"].astype(np.float32),
    joint_vel=data["joint_vel"].astype(np.float32),
    body_pos_w=data["body_pos_w"].astype(np.float32),
    body_quat_w=data["body_quat_w"].astype(np.float32),
    body_lin_vel_w=data["body_lin_vel_w"].astype(np.float32),
    body_ang_vel_w=data["body_ang_vel_w"].astype(np.float32),
    joint_names=tuple(str(name) for name in data["joint_names"]),
    body_names=tuple(str(name) for name in data["body_names"]),
  )


def _validate_clips(clips: list[MotionClip]) -> None:
  first = clips[0]
  for clip in clips[1:]:
    if clip.joint_names != first.joint_names:
      raise ValueError("All clips must share the same joint ordering.")
    if clip.body_names != first.body_names:
      raise ValueError("All clips must share the same body ordering.")
    if abs(clip.fps - first.fps) > 1e-6:
      raise ValueError("All clips must share the same FPS.")


def _make_local_root_pos_seq(root_pos: np.ndarray, yaw0: float) -> np.ndarray:
  delta_pos = root_pos[:, :2] - root_pos[0, :2]
  return _rotate_into_frame(delta_pos, yaw0).astype(np.float32)


def _make_local_body_pos_seq(
  body_pos: np.ndarray,
  root_pos: np.ndarray,
  root_yaw: np.ndarray,
) -> np.ndarray:
  delta_xy = body_pos[..., :2] - root_pos[:, None, :2]
  cos_yaw = np.cos(root_yaw)[:, None]
  sin_yaw = np.sin(root_yaw)[:, None]
  local_x = cos_yaw * delta_xy[..., 0] + sin_yaw * delta_xy[..., 1]
  local_y = -sin_yaw * delta_xy[..., 0] + cos_yaw * delta_xy[..., 1]
  local_z = body_pos[..., 2] - root_pos[:, None, 2]
  return np.stack((local_x, local_y, local_z), axis=-1).astype(np.float32)


def _rotate_into_frame(xy: np.ndarray, yaw: float) -> np.ndarray:
  cos_yaw = np.cos(yaw)
  sin_yaw = np.sin(yaw)
  x_local = cos_yaw * xy[:, 0] + sin_yaw * xy[:, 1]
  y_local = -sin_yaw * xy[:, 0] + cos_yaw * xy[:, 1]
  return np.stack((x_local, y_local), axis=-1)


def _yaw_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
  w, x, y, z = quat.T
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  return np.arctan2(siny_cosp, cosy_cosp).astype(np.float32)


def _annotate_contacts(
  foot_height: np.ndarray,
  foot_speed: np.ndarray,
  height_threshold: float,
  velocity_threshold: float,
) -> np.ndarray:
  contact = np.zeros_like(foot_height, dtype=bool)
  height_release = height_threshold * 1.5
  velocity_release = velocity_threshold * 1.5
  for foot_idx in range(foot_height.shape[1]):
    in_contact = bool(
      (foot_height[0, foot_idx] < height_threshold)
      and (foot_speed[0, foot_idx] < velocity_threshold)
    )
    contact[0, foot_idx] = in_contact
    for frame_idx in range(1, foot_height.shape[0]):
      if in_contact:
        in_contact = bool(
          (foot_height[frame_idx, foot_idx] < height_release)
          and (foot_speed[frame_idx, foot_idx] < velocity_release)
        )
      else:
        in_contact = bool(
          (foot_height[frame_idx, foot_idx] < height_threshold)
          and (foot_speed[frame_idx, foot_idx] < velocity_threshold)
        )
      contact[frame_idx, foot_idx] = in_contact
  return contact


def _compute_phase(left_contact: np.ndarray) -> np.ndarray:
  touchdown_events = np.flatnonzero(left_contact[1:] & ~left_contact[:-1]) + 1
  if left_contact[0]:
    touchdown_events = np.concatenate((np.array([0]), touchdown_events))
  num_frames = int(left_contact.shape[0])
  if touchdown_events.size < 2:
    return np.linspace(0.0, 1.0, num_frames, endpoint=False, dtype=np.float32)

  phase = np.zeros(num_frames, dtype=np.float32)
  for start, end in zip(touchdown_events[:-1], touchdown_events[1:], strict=True):
    phase[start:end] = np.linspace(0.0, 1.0, end - start, endpoint=False)

  cycle = max(int(touchdown_events[-1] - touchdown_events[-2]), 1)
  phase[touchdown_events[-1] :] = (
    np.arange(num_frames - touchdown_events[-1], dtype=np.float32) / cycle
  ) % 1.0
  if touchdown_events[0] > 0:
    phase[: touchdown_events[0]] = np.mod(
      np.arange(-touchdown_events[0], 0, dtype=np.float32) / cycle,
      1.0,
    )
  return phase.astype(np.float32)


def _compute_quality_score(
  local_root_pos_seq: np.ndarray,
  pelvis_height_seq: np.ndarray,
  foot_height_seq: np.ndarray,
  fps: float,
  contact_height_threshold: float,
) -> float:
  dt = 1.0 / fps
  root_vel = np.gradient(local_root_pos_seq, dt, axis=0)
  root_acc = np.gradient(root_vel, dt, axis=0)
  accel_cost = float(np.linalg.norm(root_acc, axis=-1).mean())
  height_cost = float(np.ptp(pelvis_height_seq))
  penetration = float(np.maximum(-foot_height_seq.min(), 0.0))
  raw = np.exp(
    -0.05 * accel_cost
    - 2.0 * height_cost
    - 5.0 * penetration / max(contact_height_threshold, 1e-6)
  )
  return float(np.clip(raw, 0.0, 1.0))


def _compute_arm_swing_amp(hand_pos_seq: np.ndarray, yaw0: float) -> float:
  left_local = _rotate_into_frame(hand_pos_seq[:, 0, :2] - hand_pos_seq[0, 0, :2], yaw0)
  right_local = _rotate_into_frame(
    hand_pos_seq[:, 1, :2] - hand_pos_seq[0, 1, :2],
    yaw0,
  )
  return float(0.5 * (np.ptp(left_local[:, 0]) + np.ptp(right_local[:, 0])))


def _touchdown_count(contact_seq: np.ndarray) -> float:
  contact_bool = contact_seq.astype(bool)
  transitions = (contact_bool[1:] & ~contact_bool[:-1]).sum()
  return float(transitions)


def _infer_gait_id(stem: str) -> int:
  name = stem.lower()
  if "turn" in name or "pivot" in name:
    return GAIT_NAME_TO_ID["turn"]
  if "start" in name:
    return GAIT_NAME_TO_ID["start"]
  if "stop" in name or "stand" in name:
    return GAIT_NAME_TO_ID["stop"]
  if "run" in name or "jog" in name:
    return GAIT_NAME_TO_ID["run"]
  return GAIT_NAME_TO_ID["walk"]


def _infer_transition_type(
  stem: str,
  local_root_pos_seq: np.ndarray,
  fps: float,
  transition_speed_threshold: float,
) -> int:
  third = max(local_root_pos_seq.shape[0] // 3, 1)
  start_speed = np.linalg.norm(
    local_root_pos_seq[third - 1] - local_root_pos_seq[0]
  ) / (third / fps)
  end_speed = np.linalg.norm(local_root_pos_seq[-1] - local_root_pos_seq[-third]) / (
    third / fps
  )
  name = stem.lower()
  if "steady" in name:
    return TRANSITION_NAME_TO_ID["none"]
  if start_speed < transition_speed_threshold <= end_speed or "accel" in name:
    return TRANSITION_NAME_TO_ID["walk_to_run"]
  if start_speed >= transition_speed_threshold > end_speed or "decel" in name:
    return TRANSITION_NAME_TO_ID["run_to_walk"]
  if "start_stop" in name:
    return TRANSITION_NAME_TO_ID["walk_to_run"]
  return TRANSITION_NAME_TO_ID["none"]


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
  return ((angles + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
