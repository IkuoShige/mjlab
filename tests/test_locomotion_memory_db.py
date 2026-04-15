"""Tests for locomotion-memory database building and proposal encoding."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.tasks.locomotion_memory.memory_db import (
  DEFAULT_K1_LOCOMOTION_EXCLUDE_GLOBS,
  DEFAULT_K1_LOCOMOTION_INCLUDE_GLOBS,
  GAIT_NAME_TO_ID,
  LocomotionMemoryLoader,
  _infer_gait_id,
  build_memory_database,
  default_k1_locomotion_memory_file,
  encode_proposals,
  select_k1_locomotion_csv_files,
)


def _make_synthetic_motion_file(path: Path) -> None:
  num_frames = 30
  num_joints = 4
  joint_names = np.asarray(["j0", "j1", "j2", "j3"], dtype=str)
  body_names = np.asarray(
    ["Trunk", "left_foot_link", "right_foot_link", "left_hand_link", "right_hand_link"],
    dtype=str,
  )

  time = np.arange(num_frames, dtype=np.float32) / 30.0
  joint_pos = np.stack([0.1 * np.sin(time + i) for i in range(num_joints)], axis=-1)
  joint_vel = np.gradient(joint_pos, 1.0 / 30.0, axis=0)

  body_pos_w = np.zeros((num_frames, len(body_names), 3), dtype=np.float32)
  body_pos_w[:, 0, 0] = time
  body_pos_w[:, 0, 2] = 0.54 + 0.01 * np.sin(2.0 * np.pi * time)

  left_contact = ((np.arange(num_frames) // 4) % 2 == 0).astype(np.float32)
  right_contact = 1.0 - left_contact
  body_pos_w[:, 1, 0] = time - 0.05
  body_pos_w[:, 1, 1] = 0.1
  body_pos_w[:, 1, 2] = np.where(left_contact > 0, 0.0, 0.06)
  body_pos_w[:, 2, 0] = time + 0.05
  body_pos_w[:, 2, 1] = -0.1
  body_pos_w[:, 2, 2] = np.where(right_contact > 0, 0.0, 0.06)
  body_pos_w[:, 3, 0] = time
  body_pos_w[:, 3, 1] = 0.2 + 0.03 * np.sin(2.0 * np.pi * time)
  body_pos_w[:, 3, 2] = 0.8
  body_pos_w[:, 4, 0] = time
  body_pos_w[:, 4, 1] = -0.2 - 0.03 * np.sin(2.0 * np.pi * time)
  body_pos_w[:, 4, 2] = 0.8

  body_quat_w = np.zeros((num_frames, len(body_names), 4), dtype=np.float32)
  body_quat_w[..., 0] = 1.0
  body_lin_vel_w = np.gradient(body_pos_w, 1.0 / 30.0, axis=0)
  body_ang_vel_w = np.zeros_like(body_lin_vel_w)

  np.savez(
    path,
    fps=np.array([30.0], dtype=np.float32),
    joint_names=joint_names,
    body_names=body_names,
    joint_pos=joint_pos.astype(np.float32),
    joint_vel=joint_vel.astype(np.float32),
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
  )


def test_build_memory_database_produces_expected_shapes(tmp_path: Path) -> None:
  motion_file = tmp_path / "walk_forward_normal_00.npz"
  _make_synthetic_motion_file(motion_file)

  memory = build_memory_database([str(motion_file)])

  assert int(memory["proposal_dim"]) == 24
  assert memory["joint_pos_seq"].shape[1] == 24
  assert memory["local_root_pos_seq"].shape[-1] == 2
  assert memory["local_foot_pos_seq"].shape[2:] == (2, 3)
  np.testing.assert_allclose(memory["local_foot_pos_seq"][0, 0, 0, :2], [-0.05, 0.1])
  assert memory["contact_seq"].shape[-1] == 2
  assert memory["phase_seq"].shape == memory["pelvis_height_seq"].shape
  assert memory["quality_score"].shape[0] == memory["joint_pos_seq"].shape[0]


def test_locomotion_memory_loader_normalizes_contact_seq_to_bool(
  tmp_path: Path,
) -> None:
  memory_file = tmp_path / "memory.npz"
  np.savez(
    memory_file,
    proposal_dim=np.array(24, dtype=np.int64),
    snippet_dt=np.array(1.0 / 30.0, dtype=np.float32),
    horizon_frames=np.array([1, 2], dtype=np.int64),
    joint_names=np.asarray(["j0", "j1"], dtype=str),
    source_clip_names=np.asarray(["turn_in_place_v2_left_00"], dtype=str),
    joint_pos_seq=np.zeros((1, 4, 2), dtype=np.float32),
    joint_vel_seq=np.zeros((1, 4, 2), dtype=np.float32),
    local_root_pos_seq=np.zeros((1, 4, 2), dtype=np.float32),
    local_root_yaw_seq=np.zeros((1, 4), dtype=np.float32),
    pelvis_height_seq=np.zeros((1, 4), dtype=np.float32),
    contact_seq=np.array([[[1, 0], [0, 1], [1, 1], [0, 0]]], dtype=np.int8),
    phase_seq=np.zeros((1, 4), dtype=np.float32),
    mean_vx=np.zeros((1,), dtype=np.float32),
    mean_vy=np.zeros((1,), dtype=np.float32),
    mean_yaw_rate=np.zeros((1,), dtype=np.float32),
    phase_start=np.zeros((1,), dtype=np.float32),
    phase_end=np.zeros((1,), dtype=np.float32),
    phase_start_sin=np.zeros((1,), dtype=np.float32),
    phase_start_cos=np.ones((1,), dtype=np.float32),
    phase_end_sin=np.zeros((1,), dtype=np.float32),
    phase_end_cos=np.ones((1,), dtype=np.float32),
    left_contact_ratio=np.zeros((1,), dtype=np.float32),
    right_contact_ratio=np.zeros((1,), dtype=np.float32),
    left_contact_start=np.zeros((1,), dtype=np.int8),
    right_contact_start=np.zeros((1,), dtype=np.int8),
    duty_factor=np.zeros((1,), dtype=np.float32),
    cadence=np.zeros((1,), dtype=np.float32),
    step_length=np.zeros((1,), dtype=np.float32),
    com_height_mean=np.zeros((1,), dtype=np.float32),
    pelvis_bounce_amp=np.zeros((1,), dtype=np.float32),
    arm_swing_amp=np.zeros((1,), dtype=np.float32),
    quality_score=np.ones((1,), dtype=np.float32),
    gait_id=np.zeros((1,), dtype=np.int64),
    transition_type=np.zeros((1,), dtype=np.int64),
    transition_flag=np.zeros((1,), dtype=np.int8),
    source_clip_id=np.zeros((1,), dtype=np.int64),
  )

  loader = LocomotionMemoryLoader(str(memory_file), device="cpu")

  assert loader.contact_seq.dtype == torch.bool
  assert not loader.has_local_foot_pos_seq
  assert torch.equal(loader.local_foot_pos_seq, torch.zeros(1, 4, 2, 3))
  assert torch.equal(
    loader.contact_seq[0, 0],
    torch.tensor([True, False]),
  )


def test_encode_proposals_returns_horizon_stacked_features() -> None:
  local_root_pos_seq = torch.tensor(
    [[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.1]]], dtype=torch.float32
  )
  local_root_yaw_seq = torch.tensor([[0.0, 0.0, 0.1, 0.2]], dtype=torch.float32)
  pelvis_height_seq = torch.tensor([[0.5, 0.51, 0.52, 0.53]], dtype=torch.float32)
  contact_seq = torch.tensor([[[1, 0], [1, 0], [0, 1], [0, 1]]], dtype=torch.int64)
  frame_indices = torch.tensor([0], dtype=torch.long)
  horizon_frames = torch.tensor([1, 3], dtype=torch.long)

  proposal = encode_proposals(
    local_root_pos_seq,
    local_root_yaw_seq,
    pelvis_height_seq,
    contact_seq,
    frame_indices,
    horizon_frames,
  )

  assert proposal.shape == (1, 12)
  np.testing.assert_allclose(proposal[0, 0].item(), 0.2, atol=1e-6)
  np.testing.assert_allclose(proposal[0, 6].item(), 0.6, atol=1e-6)


def test_default_k1_locomotion_memory_file_respects_env_override(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  override = "/tmp/custom_k1_memory.npz"
  monkeypatch.setenv("MJLAB_K1_LOCOMOTION_MEMORY_FILE", override)
  assert default_k1_locomotion_memory_file() == override


def test_select_k1_locomotion_csv_files_prefers_turn_in_place_v2_and_excludes_v1(
  tmp_path: Path,
) -> None:
  for name in (
    "walk_forward_slow_00.csv",
    "walk_fast_steady_01_00.csv",
    "jog_slow_00.csv",
    "run_steady_01_00.csv",
    "steady_1.5mps_run_steady_02_01.csv",
    "pivot_gentle_left_00.csv",
    "pivot_quarter_right_00.csv",
    "turn_left_90_00.csv",
    "turn_in_place_left_00.csv",
    "turn_in_place_right_00.csv",
    "turn_in_place_v2_left_00.csv",
    "turn_in_place_v2_right_00.csv",
    "walk_accel_00.csv",
    "walk_decel_00.csv",
    "walk_start_stop_00.csv",
    "lateral_left_fast_00.csv",
    "shuffle_left_00.csv",
    "walk_backward_slow_00.csv",
    "stand_idle_00.csv",
  ):
    (tmp_path / name).write_text("0,0,0,0,0,0,1\n", encoding="utf-8")

  selected = select_k1_locomotion_csv_files(
    source_dir=tmp_path,
    include_globs=DEFAULT_K1_LOCOMOTION_INCLUDE_GLOBS,
    exclude_globs=DEFAULT_K1_LOCOMOTION_EXCLUDE_GLOBS,
  )

  selected_names = [Path(path).name for path in selected]
  assert selected_names == [
    "jog_slow_00.csv",
    "pivot_gentle_left_00.csv",
    "pivot_quarter_right_00.csv",
    "run_steady_01_00.csv",
    "steady_1.5mps_run_steady_02_01.csv",
    "turn_in_place_v2_left_00.csv",
    "turn_in_place_v2_right_00.csv",
    "turn_left_90_00.csv",
    "walk_accel_00.csv",
    "walk_decel_00.csv",
    "walk_fast_steady_01_00.csv",
    "walk_forward_slow_00.csv",
    "walk_start_stop_00.csv",
  ]


def test_infer_gait_id_treats_pivot_as_turn() -> None:
  assert _infer_gait_id("pivot_slow_left_00") == GAIT_NAME_TO_ID["turn"]


def test_steady_cutouts_are_not_marked_as_speed_transitions() -> None:
  local_root_pos_seq = np.zeros((24, 2), dtype=np.float32)
  local_root_pos_seq[:, 0] = np.linspace(0.0, 1.2, 24)

  from mjlab.tasks.locomotion_memory import memory_db

  transition_type = memory_db._infer_transition_type(
    "steady_1.8mps_walk_decel_00",
    local_root_pos_seq,
    fps=30.0,
    transition_speed_threshold=1.5,
  )

  assert transition_type == 0
