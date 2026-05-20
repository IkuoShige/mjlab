"""Reward functions for the kick task.

LVDRS-style reward decomposition (paper Table 3, adapted to kick-only scope):

Goal-related (becomes the goal critic in the multi-critic runner):
  - alive: constant survival reward
  - terminated: large penalty for non-timeout terminations
  - ball_approach: potential-based shaping toward ball
  - target_progress: potential-based shaping for ball-in-target-direction
  - kick_success: one-shot bonus when ball flies in target direction

Auxiliary (becomes the auxiliary critic):
  - sideways_kick_aligned: reward foot lateral motion aligned with target
  - forward_kick_penalty: discourage toe-poke when foot contacts ball
  - foot_proximity: penalize feet collapsing together
  - pelvis_orientation: mild trunk-upright stability
  - non_foot_collision: penalize contacts on non-foot bodies
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.kick.mdp.commands import KickTargetCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


_FOOT_BODIES = ("left_foot_link", "right_foot_link")
_DEFAULT_FOOT_CFG = SceneEntityCfg("robot", body_names=_FOOT_BODIES)
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def _argmin_random_tiebreak(dist: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
  """``argmin`` along the last dim, breaking ties uniformly at random.

  Plain ``torch.argmin`` returns the lowest index on ties, which biases
  the foot-selection rewards toward the left foot (index 0 in
  ``_FOOT_BODIES``) whenever the geometry is left/right symmetric — e.g.
  a forward target with the ball directly in front. That bias survives
  the mirror-symmetry data augmentation and shows up at eval +0° as a
  bimodal direction distribution.

  Adding noise of magnitude ``eps`` (default 1e-4 m, much smaller than
  the typical foot-to-ball distance of 0.01-0.5 m) only changes the
  result when distances are within ``eps`` of each other.
  """
  noise = (torch.rand_like(dist) * 2.0 - 1.0) * eps
  return (dist + noise).argmin(dim=-1)


def _get_command(env: ManagerBasedRlEnv, name: str) -> KickTargetCommand:
  term = env.command_manager.get_term(name)
  if not isinstance(term, KickTargetCommand):
    raise TypeError(
      f"Command {name!r} is {type(term).__name__}, expected KickTargetCommand"
    )
  return term


# ---------------------------------------------------------------------------
# Goal-related rewards
# ---------------------------------------------------------------------------


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1.0 when episode just ended via a non-timeout termination."""
  term_mgr = env.termination_manager
  if term_mgr is None:
    return torch.zeros(env.num_envs, device=env.device)
  dones = term_mgr.terminated.float()
  return dones


def ball_approach_potential(
  env: ManagerBasedRlEnv, command_name: str = "kick_target"
) -> torch.Tensor:
  """Reward when robot is approaching ball (potential-based shaping).

  Positive when ``-Δ(robot-ball distance) > 0`` (distance decreased).
  """
  cmd = _get_command(env, command_name)
  return -cmd.robot_ball_distance_delta


def ball_proximity_continuous(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  std: float = 0.3,
) -> torch.Tensor:
  """Per-step continuous reward for being near the ball (Gaussian shape).

  Returns ``exp(-d²/std²)`` with ``d = ||robot_xy - ball_xy||``. Unlike
  the delta-based ``ball_approach_potential``, this provides a constant
  pull toward the ball even when the robot is stationary or oscillating —
  helps the policy learn to *walk* to balls that spawn beyond ``std``.
  """
  cmd = _get_command(env, command_name)
  d = cmd.robot_ball_distance
  return torch.exp(-(d**2) / (std**2))


def target_progress_potential(
  env: ManagerBasedRlEnv, command_name: str = "kick_target"
) -> torch.Tensor:
  """Reward for the ball moving along the target direction.

  Positive when projection onto ``target_dir`` increases between steps.
  """
  cmd = _get_command(env, command_name)
  return cmd.ball_progress_delta


def support_foot_proximity_at_kick(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
  ideal_distance: float = 0.20,
  std: float = 0.10,
) -> torch.Tensor:
  """Reward natural plant-foot mechanics: at kick contact, the NON-kicking
  ("support") foot should be planted next to the ball, not far behind.

  At the first valid kick contact step, identify which foot made contact
  (closest to ball) → the OTHER foot is the support. Reward a Gaussian
  shape peaked at ``ideal_distance`` between support foot and ball:
  ``exp(-((d_support - ideal)/std)^2)``. Fires only on ``kick_contact_new``.
  """
  cmd = _get_command(env, command_name)
  fire = cmd.kick_contact_new
  if not fire.any():
    return torch.zeros(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  body_names = list(asset.body_names)
  foot_names = tuple(asset_cfg.body_names or ())
  if len(foot_names) != 2:
    raise ValueError(
      f"support_foot_proximity expects 2 foot body_names, got {foot_names}"
    )
  l_idx, r_idx = body_names.index(foot_names[0]), body_names.index(foot_names[1])
  foot_pos = asset.data.body_link_pos_w[:, [l_idx, r_idx]]
  ball_pos = cmd.ball_pos_w.unsqueeze(1)
  dist = (foot_pos - ball_pos).norm(dim=-1)
  kick_foot = _argmin_random_tiebreak(dist)
  support_dist = dist.gather(1, (1 - kick_foot).unsqueeze(-1)).squeeze(-1)
  shaped = torch.exp(-((support_dist - ideal_distance) ** 2) / (std**2))
  return shaped * fire.float()


def ball_velocity_alignment_post_kick(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  window_steps: int = 10,
  std_rad: float = 0.3,
) -> torch.Tensor:
  """Sharp post-kick direction-accuracy reward (exp form, V1.26-V1.28).

  Retained for reference. Replaced in V1.29 by ``kick_angle_error_l2``
  (linear form) — the exp form gave near-zero gradient at the policy's
  large-angle operating regime, so no learning occurred despite weight
  and std-rad sweeps.
  """
  cmd = _get_command(env, command_name)
  ssk = cmd.steps_since_kick
  active = (ssk >= 0) & (ssk < window_steps)
  if not active.any():
    return torch.zeros(env.num_envs, device=env.device)
  ball_vel_xy = cmd.ball_vel_w[:, :2]
  speed = ball_vel_xy.norm(dim=-1).clamp_min(1e-6)
  target = cmd.kick_target_dir_w
  cos_angle = ((ball_vel_xy * target).sum(dim=-1) / speed).clamp(-1.0, 1.0)
  angle_sq = torch.acos(cos_angle).pow(2)
  return torch.exp(-angle_sq / (std_rad**2)) * speed * active.float()


def kick_angle_error_l2(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  window_steps: int = 10,
  min_speed: float = 1.0,
) -> torch.Tensor:
  """Linear post-kick direction-error penalty (V1.29/V1.30).

  Returns ``-angle² × ball_speed`` for the ``window_steps`` immediately
  following kick contact, where ``angle`` is the angle between ball xy
  velocity and the commanded target direction.

  Linear in angle (unlike ``ball_velocity_alignment_post_kick``'s exp
  form): the gradient ``-2·angle × speed`` is monotonically larger for
  larger errors, so the policy gets stronger signal exactly where it
  needs to correct most. V1.18 lesson: linear penalties beat exp forms
  when the policy starts far from the target value.

  Multiplied by ball speed so that fast misaligned kicks are penalized
  more than weak misaligned kicks — keeps the policy from "playing
  safe" by reducing kick force. V1.29 (no speed mult) gave only
  ~0.003 episode contribution and direction reward flatlined; V1.30
  adds the multiplier to bring magnitude up to ~0.2 (in line with the
  V1.27 exp reward at w=5) while preserving the linear gradient shape.

  Speed-gated >min_speed to filter out negligible nudges whose large
  angle errors would otherwise dominate the gradient.
  """
  cmd = _get_command(env, command_name)
  ssk = cmd.steps_since_kick
  active = (ssk >= 0) & (ssk < window_steps)
  if not active.any():
    return torch.zeros(env.num_envs, device=env.device)
  ball_vel_xy = cmd.ball_vel_w[:, :2]
  speed = ball_vel_xy.norm(dim=-1).clamp_min(1e-6)
  speed_gate = (speed > min_speed).float()
  target = cmd.kick_target_dir_w
  cos_angle = ((ball_vel_xy * target).sum(dim=-1) / speed).clamp(-1.0, 1.0)
  angle_sq = torch.acos(cos_angle).pow(2)
  return -angle_sq * speed * active.float() * speed_gate


def pre_kick_body_yaw_alignment(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
) -> torch.Tensor:
  """Pre-kick body-yaw alignment penalty (V1.31).

  Returns ``-angle²`` where angle is between the robot's body-forward
  direction and the commanded target direction. Active ONLY before kick
  contact (``steps_since_kick < 0``).

  Motivation: V1.26-V1.30 reward shaping on post-kick ball direction
  couldn't break the bimodal local optimum revealed by the V1.30 eval —
  the +0° target case got 69° mean error and +45° went from 5° (V1.28)
  to 66° (V1.30). The structural issue is that kick direction is largely
  determined by approach geometry (body yaw + foot choice), and a
  post-kick penalty can't fix the kick-time geometry.

  This reward forces the policy to physically rotate the robot to face
  the target before kicking. Combined with the existing
  `target_progress` (approach) and `kick_angle_error_l2` (direction),
  this gives the policy three coordinated signals: get close, face
  target, then kick straight.
  """
  cmd = _get_command(env, command_name)
  target_dir_b = cmd.kick_target_dir_b
  cos_angle = target_dir_b[:, 0].clamp(-1.0, 1.0)
  angle = torch.acos(cos_angle)
  pre_kick = (cmd.steps_since_kick < 0).float()
  return -angle.pow(2) * pre_kick


def kick_success_bonus(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  speed_threshold: float = 2.0,
) -> torch.Tensor:
  """One-shot bonus when ball velocity in target direction exceeds threshold."""
  cmd = _get_command(env, command_name)
  ball_vel_xy = cmd.ball_vel_w[:, :2]
  proj = (ball_vel_xy * cmd.kick_target_dir_w).sum(dim=-1)
  fired = proj > speed_threshold
  return fired.float()


# ---------------------------------------------------------------------------
# Auxiliary rewards
# ---------------------------------------------------------------------------


def sideways_kick_aligned(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
) -> torch.Tensor:
  """Reward foot lateral velocity aligned with target direction during contact.

  Encourages an arch-style kick (foot sweeps sideways through the ball) rather
  than a toe-poke.
  """
  cmd = _get_command(env, command_name)
  if not cmd.kick_contact_new.any():
    return torch.zeros(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  body_names = list(asset.body_names)
  foot_idx = torch.tensor(
    [body_names.index(n) for n in asset_cfg.body_names or ()],
    dtype=torch.long,
    device=env.device,
  )
  foot_vel = asset.data.body_link_lin_vel_w[:, foot_idx]
  ball_pos = cmd.ball_pos_w
  foot_pos = asset.data.body_link_pos_w[:, foot_idx]
  dist = (foot_pos - ball_pos.unsqueeze(1)).norm(dim=-1)
  closest = _argmin_random_tiebreak(dist)
  vel = foot_vel.gather(1, closest.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
  vel_xy = vel[:, :2]
  alignment = (vel_xy * cmd.kick_target_dir_w).sum(dim=-1)
  lateral_norm = vel_xy.norm(dim=-1).clamp_min(1e-6)
  cos_align = (alignment / lateral_norm).clamp(-1.0, 1.0)
  reward = cos_align.clamp_min(0.0) * lateral_norm
  return reward * cmd.kick_contact_new.float()


def forward_kick_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
) -> torch.Tensor:
  """Penalize toe-poke style contact (foot forward motion exceeds lateral)."""
  cmd = _get_command(env, command_name)
  if not cmd.kick_contact_new.any():
    return torch.zeros(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  body_names = list(asset.body_names)
  foot_idx = torch.tensor(
    [body_names.index(n) for n in asset_cfg.body_names or ()],
    dtype=torch.long,
    device=env.device,
  )
  foot_pos = asset.data.body_link_pos_w[:, foot_idx]
  ball_pos = cmd.ball_pos_w
  dist = (foot_pos - ball_pos.unsqueeze(1)).norm(dim=-1)
  closest = _argmin_random_tiebreak(dist)
  foot_vel = asset.data.body_link_lin_vel_w[:, foot_idx]
  vel = foot_vel.gather(1, closest.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
  vel_xy = vel[:, :2]
  # Robot-frame "forward" component (project onto root forward axis).
  from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

  yq = yaw_quat(asset.data.root_link_quat_w)
  vel_xy_3 = torch.zeros(env.num_envs, 3, device=env.device)
  vel_xy_3[:, :2] = vel_xy
  vel_b = quat_apply_inverse(yq, vel_xy_3)
  forward = vel_b[:, 0].clamp_min(0.0)
  lateral = vel_b[:, 1].abs()
  excess_forward = (forward - lateral).clamp_min(0.0)
  return excess_forward * cmd.kick_contact_new.float()


def foot_proximity(
  env: ManagerBasedRlEnv,
  min_distance: float = 0.18,
  asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
) -> torch.Tensor:
  """Penalty when feet are closer than ``min_distance``. Returns positive
  magnitude of the violation (apply a negative weight in the reward manager)."""
  asset: Entity = env.scene[asset_cfg.name]
  body_names = list(asset.body_names)
  names_seq = tuple(asset_cfg.body_names or ())
  if len(names_seq) != 2:
    raise ValueError(f"foot_proximity expects exactly 2 body_names, got {names_seq}")
  left_name, right_name = names_seq
  l_idx, r_idx = body_names.index(left_name), body_names.index(right_name)
  pos = asset.data.body_link_pos_w
  d = (pos[:, l_idx, :2] - pos[:, r_idx, :2]).norm(dim=-1)
  return (min_distance - d).clamp_min(0.0)


def pelvis_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Magnitude of trunk gravity-vector horizontal component (apply negative weight).

  Encourages trunk to stay upright.
  """
  asset: Entity = env.scene[asset_cfg.name]
  g = asset.data.projected_gravity_b
  return (g[:, 0] ** 2 + g[:, 1] ** 2).clamp_min(0.0)


def non_foot_collision(
  env: ManagerBasedRlEnv,
  sensor_name: str = "self_collision",
  threshold: float = 10.0,
) -> torch.Tensor:
  """Penalty count for non-foot collision contacts exceeding force threshold."""
  sensor = env.scene.sensors.get(sensor_name)
  if sensor is None:
    return torch.zeros(env.num_envs, device=env.device)
  data = sensor.data
  forces = data.force_history if data.force_history is not None else data.force
  if forces is None or forces.numel() == 0:
    return torch.zeros(env.num_envs, device=env.device)
  if forces.ndim == 4:
    norm = forces.norm(dim=-1).amax(dim=2)
  elif forces.ndim == 3:
    norm = forces.norm(dim=-1)
  else:
    norm = forces.norm(dim=-1, keepdim=True)
  return (norm > threshold).float().sum(dim=-1)


# ---------------------------------------------------------------------------
# Anti-spin / anti-float (V1.9 / V1.12): discourage spin kicks + body lift
# ---------------------------------------------------------------------------


def post_kick_motion_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  grace_steps: int = 15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Linear penalty for body motion in the standstill window after kick.

  Active only after ``steps_since_kick >= grace_steps`` so the policy can
  take a brief settling step right after impact (~0.3 s by default).
  Returns ``|base_lin_vel|^2 + 0.05 * |joint_vel|^2``; apply with a
  negative weight. Linear (vs. exp) keeps gradient effective at running
  speeds.
  """
  cmd = _get_command(env, command_name)
  active = cmd.steps_since_kick >= grace_steps
  if not active.any():
    return torch.zeros(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  base_speed_sq = asset.data.root_link_lin_vel_w.pow(2).sum(dim=-1)
  jv_sq = asset.data.joint_vel.pow(2).mean(dim=-1)
  return (base_speed_sq + 0.05 * jv_sq) * active.float()


# Arms-down target — must match _ARM_DOWN_RESET_TARGETS_RAD in commands.py.
# V1.25 critical fix: K1's Shoulder_Roll=0 is the T-pose; arms-down needs
# Shoulder_Roll = ±1.0 (negative for left, positive for right) plus
# Elbow_Pitch ≈ 1.5 to fold the forearm. Earlier versions targeted 0 on
# Shoulder_Roll which reinforced T-pose against the user's intent.
_ARM_DOWN_TARGETS_RAD: dict[str, float] = {
  "ALeft_Shoulder_Pitch": 0.0,
  "Left_Shoulder_Roll": -1.0,
  "Left_Elbow_Pitch": 1.5,
  "Left_Elbow_Yaw": 0.0,
  "ARight_Shoulder_Pitch": 0.0,
  "Right_Shoulder_Roll": 1.0,
  "Right_Elbow_Pitch": 1.5,
  "Right_Elbow_Yaw": 0.0,
}


def arm_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot",
    joint_names=tuple(_ARM_DOWN_TARGETS_RAD),
  ),
) -> torch.Tensor:
  """Per-step linear penalty for arm joint deviation from arms-down pose.

  Targets an arms-down pose (shoulder pitch/roll = 0, elbow lightly bent)
  rather than the K1 HOME_KEYFRAME which has shoulders pitched forward;
  the user wants arms hanging down at rest. Applies every step so the
  policy keeps arms folded across approach + kick + stand-still.
  """
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  if not isinstance(jnt_ids, (list, tuple, torch.Tensor)) or len(jnt_ids) == 0:
    return torch.zeros(env.num_envs, device=env.device)
  joint_names_ordered = tuple(asset_cfg.joint_names or ())
  targets = torch.tensor(
    [_ARM_DOWN_TARGETS_RAD[n] for n in joint_names_ordered],
    device=env.device,
    dtype=torch.float32,
  )
  jp = asset.data.joint_pos[:, jnt_ids]
  return (jp - targets).pow(2).mean(dim=-1)


def shoulder_roll_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot",
    joint_names=("Left_Shoulder_Roll", "Right_Shoulder_Roll"),
  ),
) -> torch.Tensor:
  """Dedicated heavy penalty for shoulder-roll deviation from zero.

  Apart from the general `arm_deviation_l2`, shoulder roll is the SINGLE
  joint that produces the user-observed T-pose (shoulder rolled out to
  ±π/2). A focused per-joint L2 penalty here is more cost-effective than
  bumping arm_deviation_l2 globally because it targets the specific
  failure mode without over-constraining the other arm joints (which
  can still swing pitch for balance).
  """
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  if not isinstance(jnt_ids, (list, tuple, torch.Tensor)) or len(jnt_ids) == 0:
    return torch.zeros(env.num_envs, device=env.device)
  jp = asset.data.joint_pos[:, jnt_ids]
  return jp.pow(2).mean(dim=-1)


def post_kick_homing_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  grace_steps: int = 15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Linear penalty for deviation from HOME_KEYFRAME pose after kick.

  Active only after ``steps_since_kick >= grace_steps``. Returns
  ``mean((joint_pos - default_joint_pos)^2)``. Apply with a NEGATIVE
  weight. Linear (not exp) so the gradient stays useful even when the
  pose is far from HOME (e.g. immediately after a kick swing); the
  exp-form version we tried first saturated to ~0 at typical post-kick
  joint deviation and gave no usable gradient.
  """
  cmd = _get_command(env, command_name)
  active = cmd.steps_since_kick >= grace_steps
  if not active.any():
    return torch.zeros(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  jp_err_sq = (asset.data.joint_pos - asset.data.default_joint_pos).pow(2).mean(dim=-1)
  return jp_err_sq * active.float()


def base_lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """L2 of the robot root's world-frame vertical velocity.

  Apply with a NEGATIVE weight. V1.11 viser playback showed the policy
  generating kick force by hopping ("体が浮いてしまっている") — both
  unphysical for K1 on the real robot and a sign the policy is gaming
  the per-step target_progress signal by lobbing the ball with a body
  jump rather than a clean leg swing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w[:, 2].pow(2)


def base_yaw_rate_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """L2 of the robot's z-axis (yaw) angular velocity in the body frame.

  Apply with a negative weight to penalize whole-body spin moves where the
  policy generates kick force by rotating the torso instead of swinging
  the leg with the foot arch. Observed in V1.8 viser playback as "rotation
  kicks" rather than human-style arch kicks.
  """
  asset: Entity = env.scene[asset_cfg.name]
  omega_z = asset.data.root_link_ang_vel_b[:, 2]
  return omega_z.pow(2)


# ---------------------------------------------------------------------------
# Head alignment (active perception)
# ---------------------------------------------------------------------------


def head_yaw_alignment(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  yaw_joint_name: str = "AAHead_yaw",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """L2 mismatch between head yaw joint and the ball's azimuth in body frame.

  Encourages active perception: the policy turns its head to keep the ball
  near the camera's horizontal center. Apply a small NEGATIVE weight (LVDRS
  uses -0.5).
  """
  cmd = _get_command(env, command_name)
  asset: Entity = env.scene[asset_cfg.name]
  joint_names = list(asset.joint_names)
  if yaw_joint_name not in joint_names:
    return torch.zeros(env.num_envs, device=env.device)
  j_idx = joint_names.index(yaw_joint_name)
  head_yaw = asset.data.joint_pos[:, j_idx]
  ball_xy = cmd.ball_pos_b
  desired_yaw = torch.atan2(ball_xy[:, 1], ball_xy[:, 0].clamp_min(1e-4))
  return (head_yaw - desired_yaw).pow(2)


def head_pitch_alignment(
  env: ManagerBasedRlEnv,
  command_name: str = "kick_target",
  pitch_joint_name: str = "Head_pitch",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  camera_height_offset: float = 0.45,
) -> torch.Tensor:
  """L2 mismatch between head pitch and the elevation needed to look at ball.

  Encourages the policy to pitch the head downward to keep ball-at-foot
  inside the camera's vertical FOV. ``camera_height_offset`` approximates
  camera height above the root (m).
  """
  cmd = _get_command(env, command_name)
  asset: Entity = env.scene[asset_cfg.name]
  joint_names = list(asset.joint_names)
  if pitch_joint_name not in joint_names:
    return torch.zeros(env.num_envs, device=env.device)
  j_idx = joint_names.index(pitch_joint_name)
  head_pitch = asset.data.joint_pos[:, j_idx]
  ball_b = cmd.ball_pos_b
  ground_distance = ball_b.norm(dim=-1).clamp_min(1e-3)
  # Camera looks forward at zero pitch. Positive pitch = looking DOWN.
  # Desired pitch ≈ atan(camera_height / ground_distance).
  desired_pitch = torch.atan(
    torch.full_like(ground_distance, camera_height_offset) / ground_distance
  )
  return (head_pitch - desired_pitch).pow(2)
