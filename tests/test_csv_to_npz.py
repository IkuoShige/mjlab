"""Tests for CSV-to-NPZ motion conversion helpers."""

from pathlib import Path

import numpy as np
import torch

from mjlab.scripts.csv_to_npz import MotionLoader


def _write_motion_csv(
  path: Path, quat_rows: list[tuple[float, float, float, float]]
) -> None:
  rows = []
  for quat in quat_rows:
    row = np.zeros(29, dtype=np.float32)
    row[2] = 0.5
    row[3:7] = np.asarray(quat, dtype=np.float32)
    rows.append(row)
  np.savetxt(path, np.asarray(rows), delimiter=",")


def test_motion_loader_preserves_wxyz_quaternions(tmp_path: Path) -> None:
  csv_path = tmp_path / "wxyz.csv"
  _write_motion_csv(
    csv_path,
    [
      (1.0, 0.0, 0.0, 0.0),
      (0.999, 0.01, 0.02, 0.03),
      (0.997, 0.02, 0.03, 0.04),
    ],
  )

  loader = MotionLoader(
    motion_file=str(csv_path),
    input_fps=30,
    output_fps=30,
    device="cpu",
    quat_order="wxyz",
  )

  assert loader.source_quat_order == "wxyz"
  torch.testing.assert_close(
    loader.motion_base_rots_input[0],
    torch.tensor([1.0, 0.0, 0.0, 0.0]),
  )


def test_motion_loader_converts_xyzw_quaternions_to_wxyz(tmp_path: Path) -> None:
  csv_path = tmp_path / "xyzw.csv"
  _write_motion_csv(
    csv_path,
    [
      (0.0, 0.0, 0.0, 1.0),
      (0.01, 0.02, 0.03, 0.999),
      (0.02, 0.03, 0.04, 0.997),
    ],
  )

  loader = MotionLoader(
    motion_file=str(csv_path),
    input_fps=30,
    output_fps=30,
    device="cpu",
    quat_order="xyzw",
  )

  assert loader.source_quat_order == "xyzw"
  torch.testing.assert_close(
    loader.motion_base_rots_input[0],
    torch.tensor([1.0, 0.0, 0.0, 0.0]),
  )


def test_motion_loader_auto_detects_wxyz_input(tmp_path: Path) -> None:
  csv_path = tmp_path / "auto.csv"
  _write_motion_csv(
    csv_path,
    [
      (0.9995, 0.01, -0.02, 0.005),
      (0.9980, 0.015, -0.03, 0.006),
      (0.9970, 0.02, -0.035, 0.007),
    ],
  )

  loader = MotionLoader(
    motion_file=str(csv_path),
    input_fps=30,
    output_fps=30,
    device="cpu",
    quat_order="auto",
  )

  assert loader.source_quat_order == "wxyz"
  torch.testing.assert_close(
    loader.motion_base_rots_input[0],
    torch.tensor([0.9995, 0.01, -0.02, 0.005]),
  )
