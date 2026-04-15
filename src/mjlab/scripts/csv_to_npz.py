from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.booster_k1.env_cfgs import (
  booster_k1_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


def _save_motion_npz(output_file: str, log: dict[str, Any]) -> None:
  output_path = Path(output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **log)


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    quat_order: Literal["auto", "xyzw", "wxyz"] = "auto",
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.quat_order = quat_order
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    header_rows = 1 if self._has_header() else 0
    if self.line_range is None:
      motion = torch.from_numpy(
        np.loadtxt(self.motion_file, delimiter=",", skiprows=header_rows)
      )
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=header_rows + self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    # motion[:, 2] -= 0.05
    self.motion_base_poss_input = motion[:, :3]
    root_quat_input = motion[:, 3:7]
    self.source_quat_order = self._resolve_quat_order(root_quat_input)
    if self.source_quat_order == "xyzw":
      self.motion_base_rots_input = root_quat_input[:, [3, 0, 1, 2]]
    else:
      self.motion_base_rots_input = root_quat_input
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _resolve_quat_order(
    self, root_quat_input: torch.Tensor
  ) -> Literal["xyzw", "wxyz"]:
    if self.quat_order == "xyzw":
      return "xyzw"
    if self.quat_order == "wxyz":
      return "wxyz"

    first_col_mean = torch.mean(torch.abs(root_quat_input[:, 0])).item()
    last_col_mean = torch.mean(torch.abs(root_quat_input[:, 3])).item()

    # In typical locomotion clips the scalar term dominates. Use that to infer
    # whether the input already stores quaternions as wxyz.
    if first_col_mean > last_col_mean * 2.0:
      return "wxyz"
    if last_col_mean > first_col_mean * 2.0:
      return "xyzw"

    print(
      "[WARNING]: Could not confidently infer quaternion order from input; "
      "defaulting to xyzw. Pass --quat-order explicitly if replay looks wrong."
    )
    return "xyzw"

  def _has_header(self) -> bool:
    """Returns True if the CSV file has a text header row."""
    with open(self.motion_file, encoding="utf-8") as file:
      first_line = file.readline()
    if not first_line.strip():
      return False
    try:
      for value in first_line.strip().split(","):
        float(value)
      return False
    except ValueError:
      return True

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_file,
  artifact_name,
  upload_wandb,
  render,
  quat_order,
  line_range,
  renderer: OffscreenRenderer | None = None,
):
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    quat_order=quat_order,
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
  expected_dof = len(joint_names)
  csv_dof = motion.motion_dof_poss_input.shape[1]
  if csv_dof != expected_dof:
    raise ValueError(
      f"CSV joint dimension mismatch: CSV has {csv_dof} DOF, "
      f"but robot expects {expected_dof} DOF. "
      f"If CSV is for K1 robot (22 DOF), use --robot k1. "
      f"If CSV is for G1 robot (29 DOF), use --robot g1."
    )

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_names": np.asarray(joint_names, dtype=str),
    "body_names": np.asarray(robot.body_names, dtype=str),
    "source_quat_order": np.asarray([motion.source_quat_order], dtype=str),
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # Create progress bar
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)

        print(f"Saving to {output_file}...")
        _save_motion_npz(output_file, log)

        if upload_wandb:
          import wandb

          collection = artifact_name or Path(output_file).stem
          print("Uploading to Weights & Biases...")
          run = wandb.init(project="csv_to_npz", name=collection)
          print(f"[INFO]: Logging motion to wandb: {collection}")
          registry = "motions"
          logged_artifact = run.log_artifact(
            artifact_or_path=output_file,
            name=collection,
            type=registry,
          )
          run.link_artifact(
            artifact=logged_artifact,
            target_path=f"wandb-registry-{registry}/{collection}",
          )
          print(f"[INFO]: Motion saved to wandb registry: {registry}/{collection}")

          if render:
            import mediapy as media

            video_path = str(Path(output_file).with_suffix(".mp4"))
            print("Creating video...")
            media.write_video(video_path, frames, fps=output_fps)

            print("Logging video to wandb...")
            wandb.log({"motion_video": wandb.Video(video_path, format="mp4")})

          wandb.finish()


G1_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

K1_JOINT_NAMES = [
  "AAHead_yaw",
  "Head_pitch",
  "ALeft_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "ARight_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
  "Left_Hip_Pitch",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Left_Knee_Pitch",
  "Left_Ankle_Pitch",
  "Left_Ankle_Roll",
  "Right_Hip_Pitch",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Right_Knee_Pitch",
  "Right_Ankle_Pitch",
  "Right_Ankle_Roll",
]


def main(
  input_file: str,
  output_file: str,
  robot: Literal["g1", "k1"] = "g1",
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  artifact_name: str | None = None,
  upload_wandb: bool = False,
  render: bool = False,
  quat_order: Literal["auto", "xyzw", "wxyz"] = "auto",
  line_range: tuple[int, int] | None = None,
):
  """Replay motion from CSV file and output to npz file.

  Args:
    input_file: Path to the input CSV file.
    output_file: Path to the output npz file.
    robot: Robot type to use, either "g1" or "k1".
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    artifact_name: Optional W&B artifact name to use when uploading.
    upload_wandb: Whether to upload the converted motion as a W&B artifact.
    render: Whether to render the simulation and save a video.
    quat_order: Root quaternion order in the CSV. Use ``wxyz`` for
      ``/workspace/k1_retarget/motions_k1/k1_fixed`` and ``xyzw`` for raw Booster
      CSV exports. ``auto`` infers from the data.
    line_range: Range of lines to process from the CSV file.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  if robot == "k1":
    env_cfg = booster_k1_flat_tracking_env_cfg()
    joint_names = K1_JOINT_NAMES
  else:
    env_cfg = unitree_g1_flat_tracking_env_cfg()
    joint_names = G1_JOINT_NAMES

  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  run_sim(
    sim=sim,
    scene=scene,
    joint_names=joint_names,
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_file=output_file,
    artifact_name=artifact_name,
    upload_wandb=upload_wandb,
    render=render,
    quat_order=quat_order,
    line_range=line_range,
    renderer=renderer,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
