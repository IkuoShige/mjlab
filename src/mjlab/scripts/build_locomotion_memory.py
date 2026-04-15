from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import tyro

import mjlab
from mjlab.tasks.locomotion_memory.memory_db import (
  build_memory_database,
  resolve_motion_files,
)


@dataclass
class BuildLocomotionMemoryCfg:
  input_paths: list[str]
  output_file: str
  include_glob: str = "*.npz"
  anchor_body_name: str = "Trunk"
  left_foot_body_name: str = "left_foot_link"
  right_foot_body_name: str = "right_foot_link"
  left_hand_body_name: str = "left_hand_link"
  right_hand_body_name: str = "right_hand_link"
  snippet_length_s: float = 0.8
  stride_frames: int = 6
  contact_height_threshold: float = 0.03
  contact_velocity_threshold: float = 0.15
  transition_speed_threshold: float = 1.5
  min_quality_score: float = 0.05


def _save_memory_database(output_file: str, memory: dict[str, np.ndarray]) -> None:
  savez = cast(Any, np.savez)
  savez(output_file, **memory)


def main() -> None:
  cfg = tyro.cli(BuildLocomotionMemoryCfg, config=mjlab.TYRO_FLAGS)
  motion_files = resolve_motion_files(cfg.input_paths, include_glob=cfg.include_glob)
  if not motion_files:
    raise ValueError("No input motion files matched the provided paths/glob.")

  memory = build_memory_database(
    motion_files,
    anchor_body_name=cfg.anchor_body_name,
    left_foot_body_name=cfg.left_foot_body_name,
    right_foot_body_name=cfg.right_foot_body_name,
    left_hand_body_name=cfg.left_hand_body_name,
    right_hand_body_name=cfg.right_hand_body_name,
    snippet_length_s=cfg.snippet_length_s,
    stride_frames=cfg.stride_frames,
    contact_height_threshold=cfg.contact_height_threshold,
    contact_velocity_threshold=cfg.contact_velocity_threshold,
    transition_speed_threshold=cfg.transition_speed_threshold,
    min_quality_score=cfg.min_quality_score,
  )

  _save_memory_database(cfg.output_file, memory)
  num_snippets = int(memory["joint_pos_seq"].shape[0])
  proposal_dim = int(np.asarray(memory["proposal_dim"]).item())
  print(
    f"Saved locomotion memory with {num_snippets} snippets "
    f"(proposal_dim={proposal_dim}) to {cfg.output_file}"
  )


if __name__ == "__main__":
  main()
