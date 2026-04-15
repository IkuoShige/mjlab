from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import tyro

import mjlab
from mjlab.scripts.csv_to_npz import main as convert_csv_to_npz
from mjlab.tasks.locomotion_memory.memory_db import (
  DEFAULT_K1_CONVERTED_DIR,
  DEFAULT_K1_MEMORY_FILE,
  DEFAULT_K1_SOURCE_DIR,
  build_memory_database,
  select_k1_locomotion_csv_files,
)


@dataclass
class PrepareK1LocomotionMemoryCfg:
  source_dir: str = str(DEFAULT_K1_SOURCE_DIR)
  converted_dir: str = str(DEFAULT_K1_CONVERTED_DIR)
  memory_file: str = str(DEFAULT_K1_MEMORY_FILE)
  input_fps: float = 30.0
  output_fps: float = 30.0
  device: str = "cuda:0"
  overwrite_converted: bool = False
  overwrite_memory: bool = False
  max_clips: int | None = None


def _save_memory_npz(output_file: str, memory: dict[str, np.ndarray]) -> None:
  np.savez(output_file, **cast(dict[str, Any], memory))


def main() -> None:
  cfg = tyro.cli(PrepareK1LocomotionMemoryCfg, config=mjlab.TYRO_FLAGS)

  selected_csvs = select_k1_locomotion_csv_files(cfg.source_dir)
  if cfg.max_clips is not None:
    selected_csvs = selected_csvs[: cfg.max_clips]
  if not selected_csvs:
    raise ValueError("No K1 locomotion CSV files matched the first-pass selection.")

  converted_dir = Path(cfg.converted_dir or DEFAULT_K1_CONVERTED_DIR)
  memory_file = Path(cfg.memory_file or DEFAULT_K1_MEMORY_FILE)
  converted_dir.mkdir(parents=True, exist_ok=True)

  converted_paths: list[str] = []
  for csv_path in selected_csvs:
    output_npz = converted_dir / f"{Path(csv_path).stem}.npz"
    if cfg.overwrite_converted or not output_npz.is_file():
      convert_csv_to_npz(
        input_file=csv_path,
        output_file=str(output_npz),
        robot="k1",
        input_fps=cfg.input_fps,
        output_fps=cfg.output_fps,
        device=cfg.device,
        upload_wandb=False,
        render=False,
        quat_order="wxyz",
      )
    converted_paths.append(str(output_npz))

  if cfg.overwrite_memory or not memory_file.is_file():
    memory = build_memory_database(converted_paths)
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    _save_memory_npz(str(memory_file), memory)

  print(
    f"Prepared K1 locomotion memory from {len(converted_paths)} clips at {memory_file}"
  )


if __name__ == "__main__":
  main()
