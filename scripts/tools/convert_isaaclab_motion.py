#!/usr/bin/env python3
"""Convert Isaac Lab motion files from BFS body/joint ordering to MuJoCo DFS ordering.

Isaac Lab (PhysX) stores bodies and joints in breadth-first search (BFS) order
from the articulation root, while MuJoCo uses depth-first search (DFS) order
matching the URDF/MJCF tree traversal.

This script reorders the body and joint arrays in .npz motion files so they
can be used directly by mjlab's MotionLoader / SoccerMotionCommand.

Usage:
    uv run python scripts/tools/convert_isaaclab_motion.py \\
        --input motions/soccer-standard \\
        --output motions/soccer-standard-mj

    # Single file:
    uv run python scripts/tools/convert_isaaclab_motion.py \\
        --input motion.npz --output motion_mj.npz
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# Isaac Lab BFS ordering -> MuJoCo DFS ordering for Unitree G1 (30 bodies).
# MJ_TO_ISAAC_BODY[mj_idx] = isaac_idx
# fmt: off
G1_MJ_TO_ISAAC_BODY = [
    0,   # pelvis
    1,   # left_hip_pitch_link
    4,   # left_hip_roll_link
    7,   # left_hip_yaw_link
    10,  # left_knee_link
    14,  # left_ankle_pitch_link
    18,  # left_ankle_roll_link
    2,   # right_hip_pitch_link
    5,   # right_hip_roll_link
    8,   # right_hip_yaw_link
    11,  # right_knee_link
    15,  # right_ankle_pitch_link
    19,  # right_ankle_roll_link
    3,   # waist_yaw_link
    6,   # waist_roll_link
    9,   # torso_link
    12,  # left_shoulder_pitch_link
    16,  # left_shoulder_roll_link
    20,  # left_shoulder_yaw_link
    22,  # left_elbow_link
    24,  # left_wrist_roll_link
    26,  # left_wrist_pitch_link
    28,  # left_wrist_yaw_link
    13,  # right_shoulder_pitch_link
    17,  # right_shoulder_roll_link
    21,  # right_shoulder_yaw_link
    23,  # right_elbow_link
    25,  # right_wrist_roll_link
    27,  # right_wrist_pitch_link
    29,  # right_wrist_yaw_link
]

# Same BFS->DFS pattern for joints (29 joints, excluding free joint).
G1_MJ_TO_ISAAC_JOINT = [
    0,   # left_hip_pitch_joint
    3,   # left_hip_roll_joint
    6,   # left_hip_yaw_joint
    9,   # left_knee_joint
    13,  # left_ankle_pitch_joint
    17,  # left_ankle_roll_joint
    1,   # right_hip_pitch_joint
    4,   # right_hip_roll_joint
    7,   # right_hip_yaw_joint
    10,  # right_knee_joint
    14,  # right_ankle_pitch_joint
    18,  # right_ankle_roll_joint
    2,   # waist_yaw_joint
    5,   # waist_roll_joint
    8,   # waist_pitch_joint
    11,  # left_shoulder_pitch_joint
    15,  # left_shoulder_roll_joint
    19,  # left_shoulder_yaw_joint
    21,  # left_elbow_joint
    23,  # left_wrist_roll_joint
    25,  # left_wrist_pitch_joint
    27,  # left_wrist_yaw_joint
    12,  # right_shoulder_pitch_joint
    16,  # right_shoulder_roll_joint
    20,  # right_shoulder_yaw_joint
    22,  # right_elbow_joint
    24,  # right_wrist_roll_joint
    26,  # right_wrist_pitch_joint
    28,  # right_wrist_yaw_joint
]
# fmt: on


def convert_motion(
  src: str,
  dst: str,
  body_map: list[int] = G1_MJ_TO_ISAAC_BODY,
  joint_map: list[int] = G1_MJ_TO_ISAAC_JOINT,
) -> None:
  """Convert a single .npz motion file from Isaac Lab BFS to MuJoCo DFS ordering."""
  m = np.load(src)

  save: dict[str, np.ndarray] = {
    "fps": m["fps"],
    "joint_pos": m["joint_pos"][:, joint_map],
    "joint_vel": m["joint_vel"][:, joint_map],
    "body_pos_w": m["body_pos_w"][:, body_map],
    "body_quat_w": m["body_quat_w"][:, body_map],
    "body_lin_vel_w": m["body_lin_vel_w"][:, body_map],
    "body_ang_vel_w": m["body_ang_vel_w"][:, body_map],
  }
  if "kick_leg" in m.files:
    save["kick_leg"] = m["kick_leg"]

  np.savez(dst, **save)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input", required=True, help="Input .npz file or directory of .npz files"
  )
  parser.add_argument("--output", required=True, help="Output .npz file or directory")
  args = parser.parse_args()

  if os.path.isfile(args.input):
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    convert_motion(args.input, args.output)
    print(f"  {args.input} -> {args.output}")
  elif os.path.isdir(args.input):
    os.makedirs(args.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input, "*.npz")))
    if not files:
      print(f"No .npz files found in {args.input}")
      return
    for fpath in files:
      out_path = os.path.join(args.output, os.path.basename(fpath))
      convert_motion(fpath, out_path)
      print(f"  {os.path.basename(fpath)} -> OK")
  else:
    parser.error(f"Input not found: {args.input}")

  print("Done.")


if __name__ == "__main__":
  main()
