"""Booster K1 environment configuration for the LVDRS-style kick task."""

from __future__ import annotations

import math

from mjlab.asset_zoo.objects.soccer_ball import (
  SOCCER_BALL_RADIUS,
  get_soccer_ball_cfg,
)
from mjlab.asset_zoo.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from mjlab.asset_zoo.robots.booster_k1 import K1_FOOT_GEOM_REGEX
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.kick.kick_env_cfg import make_kick_env_cfg
from mjlab.tasks.kick.mdp.commands import KickTargetCommandCfg
from mjlab.tasks.kick.mdp.perception import VirtualPerceptionCfg

_K1_CAMERA_BODY = "Head_2"
_K1_CAMERA_OFFSET_POS = (0.00124, 0.04553, -0.01582)
_K1_CAMERA_OFFSET_QUAT = (0.0054, 0.9986, -0.0028, 0.0511)


def k1_kick_env_cfg(
  play: bool = False,
  enable_perception: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Compose the K1 kick task config."""
  cfg = make_kick_env_cfg()

  # ----- Scene: robot + ball + sensors -----
  cfg.scene.entities = {
    "robot": get_k1_robot_cfg(),
    "soccer_ball": get_soccer_ball_cfg(),
  }
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
    ContactSensorCfg(
      name="ball_contact",
      primary=ContactMatch(mode="body", pattern="ball_body", entity="soccer_ball"),
      secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
      history_length=4,
    ),
  )

  # ----- Actions -----
  # V1.10: shrink arm action scale 4x (0.44 → 0.10) to suppress the wild
  # arm flailing observed in viser. Arms still move (needed for balance
  # during kick swing) but at a quarter of the velocity. Mirror-symmetric
  # so left/right stay paired.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    **K1_ACTION_SCALE,
    # V1.24: shoulder/elbow pitch + yaw action scale restored to allow the
    # policy to actually reach arms-down. The previous 0.05 was too tight:
    # with `use_default_offset=True` and K1 HOME `Shoulder_Pitch=0.3`,
    # the minimum reachable target was 0.3 − 0.05 = 0.25 rad ≈ 14° — the
    # policy could not bring arms down no matter what. Shoulder roll stays
    # narrow to keep T-pose suppressed (arm_deviation_l2 -100 +
    # shoulder_roll_l2 -50 still apply).
    "ALeft_Shoulder_Pitch": 0.5,
    "Left_Shoulder_Roll": 0.05,
    "Left_Elbow_Pitch": 0.5,
    "Left_Elbow_Yaw": 0.3,
    "ARight_Shoulder_Pitch": 0.5,
    "Right_Shoulder_Roll": 0.05,
    "Right_Elbow_Pitch": 0.5,
    "Right_Elbow_Yaw": 0.3,
  }

  # ----- Commands (attach K1 perception + foot details) -----
  # V1.7: hold_last_on_miss=False — out-of-FOV ball MUST give zero obs.
  # Previously stale ball positions persisted and let the policy act as if
  # it could see balls outside the camera frustum (visible in viser).
  perception_cfg = (
    VirtualPerceptionCfg(
      camera_body_name=_K1_CAMERA_BODY,
      camera_offset_pos=_K1_CAMERA_OFFSET_POS,
      camera_offset_quat=_K1_CAMERA_OFFSET_QUAT,
      hold_last_on_miss=False,
    )
    if enable_perception
    else None
  )
  cfg.commands["kick_target"] = KickTargetCommandCfg(
    asset_name="robot",
    ball_name="soccer_ball",
    ball_sensor_name="ball_contact",
    foot_body_names=("left_foot_link", "right_foot_link"),
    # V1.10: ball-at-foot scope, but outside the foot stance (K1 hip width
    # ~0.18m, foot tip ~0.10m forward). Below 0.20m the ball spawned
    # BETWEEN the feet, under the robot body — unphysical and not what
    # "ball at foot" means.
    # V1.34 distance curriculum (0.50 → 0.60 → 1.00m attempts) paused
    # in V1.35 — user direction shifted to "perfect close-ball kick in
    # arbitrary direction first, distance curriculum later (possibly
    # via multi-critic for search / approach / kick decomposition)".
    # Reverted to V1.33's at-foot distribution while re-introducing
    # wider target_dir sampling (see commands.py:218).
    ball_spawn_distance_range=(0.20, 0.40),
    # V1.7: tight front cone (~±45°) so the ball is always inside the K1
    # camera horizontal FOV (~±52°) when the policy aligns head yaw. Without
    # this, balls behind/beside the robot were "kicked" with no real
    # perception — visible as unnatural behavior in viser playback that
    # would never work at deploy time.
    ball_spawn_angle_range=(-math.pi / 4, math.pi / 4),
    # Kick target direction remains 360° — the user wants "kick from front,
    # any direction" (e.g., side-foot pass to lateral target). Mirror
    # symmetry handles left vs right foot selection.
    target_dir_range=(-math.pi, math.pi),
    ball_spawn_height=SOCCER_BALL_RADIUS,
    min_foot_speed=3.0,
    horizontal_force_threshold=15.0,
    perception=perception_cfg,
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
  )

  # ----- Events: K1 body/foot wiring -----
  base_com_event = cfg.events["base_com"]
  base_com_event.params["asset_cfg"] = SceneEntityCfg("robot", body_names=("Trunk",))

  foot_friction_event = cfg.events["foot_friction"]
  foot_friction_event.params["asset_cfg"] = SceneEntityCfg(
    "robot", geom_names=(K1_FOOT_GEOM_REGEX,)
  )

  # ----- Viewer -----
  cfg.viewer.body_name = "Trunk"

  # ----- Play overrides -----
  if play:
    cfg.episode_length_s = 1.0e9
    cfg.events.pop("push_robot", None)
    cfg.scene.num_envs = max(cfg.scene.num_envs, 1)
    # V1.35: play distance matches training (0.20-0.40m). Show the
    # upper end so the robot still takes a step.
    cmd_term = cfg.commands["kick_target"]
    cmd_term.ball_spawn_distance_range = (0.30, 0.40)

    # Two opt-in play modes, switched by env vars:
    # - default ("clean" play): disable DR so a pre-V1.44 checkpoint still
    #   works and the user sees the policy's intent without noise/delay.
    # - MJLAB_KICK_PLAY_FULL_DR=1: keep training-time DR (action delay,
    #   corruption, head_pitch_down init). Use this to evaluate
    #   V1.43+/V1.44+ checkpoints under their own training distribution
    #   (sim2sim/sim2real preview).
    import os

    from mjlab.tasks.kick.mdp.delayed_action import DelayedJointPositionActionCfg

    full_dr_play = os.environ.get("MJLAB_KICK_PLAY_FULL_DR", "0") == "1"
    if not full_dr_play:
      for group in cfg.observations.values():
        group.enable_corruption = False
      cmd_term.reset_head_pitch_down = False
      joint_pos_action = cfg.actions.get("joint_pos")
      if isinstance(joint_pos_action, DelayedJointPositionActionCfg):
        joint_pos_action.delay_steps_range = (0, 0)

  return cfg
