"""Base environment configuration for the LVDRS-style kick-only task.

Robot-specific configs (in ``config/``) compose on top of this base by
setting the robot entity, foot/camera body names, action scales, and
event/termination thresholds.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.kick import mdp
from mjlab.tasks.kick.mdp.commands import KickTargetCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_kick_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base kick task configuration (robot-agnostic)."""

  # --------------------------------------------------------------------
  # Observations
  # --------------------------------------------------------------------

  # V1.44: sim2real noise upgrade — switch from Unoise (bounded uniform)
  # to Gnoise (Gaussian) on proprioceptive obs to match real-sensor
  # statistics. Standard devs picked so 3σ ≈ old uniform range max,
  # matching booster_amp_lab's per-step noise model. Per-env calibration
  # bias drift is added on top via NoiseModelWithAdditiveBias in the
  # observation group (see config/booster_k1/env_cfgs.py override).
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Gnoise(std=0.07),
    ),
    "projected_gravity": ObservationTermCfg(
      func=base_mdp.projected_gravity,
      noise=Gnoise(std=0.02),
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      params={"biased": True},
      noise=Gnoise(std=0.005),
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel,
      noise=Gnoise(std=0.2),
    ),
    "actions": ObservationTermCfg(func=base_mdp.last_action),
    "ball_pos_b_perceived": ObservationTermCfg(
      func=mdp.observations.ball_pos_b_perceived,
      params={"command_name": "kick_target"},
    ),
    "ball_mask": ObservationTermCfg(
      func=mdp.observations.ball_mask,
      params={"command_name": "kick_target"},
    ),
    "kick_target_dir_b": ObservationTermCfg(
      func=mdp.observations.kick_target_dir_b,
      params={"command_name": "kick_target"},
    ),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=base_mdp.last_action),
    "ball_pos_b": ObservationTermCfg(
      func=mdp.observations.ball_pos_b,
      params={"command_name": "kick_target"},
    ),
    "ball_vel_b": ObservationTermCfg(
      func=mdp.observations.ball_vel_b,
      params={"command_name": "kick_target"},
    ),
    "kick_target_dir_b": ObservationTermCfg(
      func=mdp.observations.kick_target_dir_b,
      params={"command_name": "kick_target"},
    ),
    "root_height": ObservationTermCfg(func=mdp.observations.root_height),
    "ball_physics": ObservationTermCfg(
      func=mdp.observations.ball_physics_normalized,
      params={"command_name": "kick_target"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # --------------------------------------------------------------------
  # Actions (full joint position; per-robot overrides scales)
  # --------------------------------------------------------------------

  # V1.44/V1.45: action delay DR. K1 ZeroErr motors + EtherCAT loop have
  # ~30-60 ms round-trip latency that mjlab doesn't model by default.
  # Without it, the policy overfits to zero-delay and breaks at deploy.
  # V1.44 used (2, 8) = 40-160 ms which made adaptation hard in 3000 iters
  # (direction accuracy degraded; some robots fell after whiff). V1.45
  # narrows to (1, 4) = 20-80 ms — still brackets the real K1 ~50 ms
  # latency but is much easier for the policy to compensate for.
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.DelayedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
      delay_steps_range=(1, 4),
    )
  }

  # --------------------------------------------------------------------
  # Commands
  # --------------------------------------------------------------------

  commands: dict[str, CommandTermCfg] = {
    "kick_target": KickTargetCommandCfg(
      asset_name="robot",
      ball_name="soccer_ball",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=False,
    ),
  }

  # --------------------------------------------------------------------
  # Events (domain randomization + perturbations)
  # --------------------------------------------------------------------

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=base_mdp.push_by_setting_velocity,
      mode="interval",
      # V1.23: more frequent + stronger pushes for sim2real robustness.
      interval_range_s=(1.5, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.15, 0.15),
          "roll": (-0.35, 0.35),
          "pitch": (-0.35, 0.35),
          "yaw": (-0.5, 0.5),
        }
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        # V1.20: lower bound 0.3 → 0.6. User reported visible foot sliding
        # in viser; 0.3 contact friction is borderline slippery for K1.
        # 0.6-1.2 covers typical indoor surfaces (rubber, vinyl, hardwood)
        # while keeping enough variation for sim2real robustness.
        "ranges": (0.6, 1.2),
        "shared_random": True,
      },
    ),
    "ball_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("soccer_ball", geom_names=(".*",)),
        "operation": "abs",
        "ranges": (0.5, 1.5),
        "shared_random": True,
      },
    ),
    "ball_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("soccer_ball", body_names=(".*",)),
        "operation": "scale",
        "ranges": (0.85, 1.15),
      },
    ),
    # V1.20: sim2real domain randomization expanded.
    "trunk_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
        "operation": "scale",
        "ranges": (0.85, 1.15),
      },
    ),
    "joint_damping": EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "scale",
        "ranges": (0.8, 1.2),
      },
    ),
    # V1.23: extra robot-side DR for sim2real.
    "all_body_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "operation": "scale",
        "ranges": (0.90, 1.10),
      },
    ),
    "joint_friction": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "abs",
        "ranges": (0.0, 0.05),
      },
    ),
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "scale",
        "ranges": (0.9, 1.1),
      },
    ),
  }

  # --------------------------------------------------------------------
  # Rewards (LVDRS Table 3 adapted to kick-only scope)
  # --------------------------------------------------------------------

  rewards: dict[str, RewardTermCfg] = {
    # ----- Goal-related (group: "goal") -----
    # V1.1: alive ↑ (live-long bias), terminated ↓ (saner scale relative to
    # other rewards — original -1000 was too coarse for PPO).
    "alive": RewardTermCfg(func=mdp.rewards.alive, weight=5.0),
    "terminated": RewardTermCfg(func=mdp.rewards.terminated, weight=-200.0),
    "ball_approach": RewardTermCfg(
      func=mdp.rewards.ball_approach_potential,
      weight=50.0,
      params={"command_name": "kick_target"},
    ),
    # V1.2: continuous "near ball" pull — gives the policy a per-step gradient
    # toward the ball even when stationary. Fixes the iter~7000 plateau where
    # only close-spawned balls (~0.18m) were kicked because there was no
    # gradient pulling the robot toward farther-spawned balls.
    "ball_proximity": RewardTermCfg(
      func=mdp.rewards.ball_proximity_continuous,
      weight=5.0,
      params={"command_name": "kick_target", "std": 0.3},
    ),
    # V1.5 reverted: dropping target_progress from 500 to 100 caused total
    # behavioral collapse (kick_success_rate 0.47 → 0.00, peak_kick_speed
    # 4.97 → 2.97 m/s, episode length 167 → 578 — policy "stood next to
    # ball" until timeout). The weight-500 target_progress signal IS the
    # gradient that pulls the policy into kicking. Keep at 500.
    "target_progress": RewardTermCfg(
      func=mdp.rewards.target_progress_potential,
      weight=500.0,
      params={"command_name": "kick_target"},
    ),
    # V1.1: kick_success ↓ to discourage "kick-and-fall" local optimum.
    "kick_success": RewardTermCfg(
      func=mdp.rewards.kick_success_bonus,
      weight=3.0,
      params={"command_name": "kick_target", "speed_threshold": 2.0},
    ),
    # V1.31/V1.32: pre-kick body yaw alignment. Forces the robot to
    # physically rotate to face the target before kicking. Addresses the
    # bimodal local optimum from V1.30 eval (51.3° mean, +0° case at 69°
    # error) that 5 versions of post-kick reward shaping could not break.
    #
    # V1.31 (w=-0.1): policy refused to turn — yaw_alignment plateaued at
    # 0.0065 episode_reward with no improvement over 1000 iters.
    # `base_yaw_rate_l2` (w=-0.5) penalty was dominating the trade-off,
    # so the policy chose to stand still and accept the smaller
    # misalignment penalty over the larger yaw-rate penalty.
    # V1.32 (w=-0.5): 5× bump (within safety band). Now matches the
    # `base_yaw_rate_l2` scale, so the policy must trade off
    # `-0.5·angle²` vs `-0.5·yaw_rate²` — turning becomes worthwhile.
    "pre_kick_body_yaw_alignment": RewardTermCfg(
      func=mdp.rewards.pre_kick_body_yaw_alignment,
      weight=-0.5,
      params={"command_name": "kick_target"},
    ),
    # V1.29/V1.30: linear -angle² penalty for direction accuracy.
    # V1.26-V1.28 used exp form; never learned because exp saturates to
    # ~0 gradient at large angles (eval showed bimodal distribution with
    # +45°=5° err vs -45°=68° err — exp gave zero gradient on the bad
    # mode kicks).
    #
    # V1.29 (w=-1.0, no speed mult): episode contribution stayed at
    # ~0.003. Linear gradient is monotonic and does pull, but signal
    # magnitude is ~50× too weak relative to other rewards (target_progress
    # = 16.5 episode contrib).
    # V1.30 (w=-3.0, with speed mult): adds ball-speed multiplier so
    # episode contribution lands around ~3 (≈18% of target_progress,
    # similar to V1.28's failed exp attempt but with proper linear
    # gradient). Weight bump 1→3 stays inside the "≤5× change" safety
    # band (V1.5/V1.8 lessons).
    "kick_angle_error_l2": RewardTermCfg(
      func=mdp.rewards.kick_angle_error_l2,
      weight=-3.0,
      params={
        "command_name": "kick_target",
        "window_steps": 10,
        "min_speed": 1.0,
      },
    ),
    # ----- Auxiliary (group: "aux") -----
    "sideways_kick": RewardTermCfg(
      func=mdp.rewards.sideways_kick_aligned,
      weight=20.0,
      params={"command_name": "kick_target"},
    ),
    # V1.11: reward planting the non-kicking foot next to the ball at
    # kick contact ("support foot beside ball, swing foot strikes"). This
    # is the natural human kick mechanic.
    "support_foot_proximity": RewardTermCfg(
      func=mdp.rewards.support_foot_proximity_at_kick,
      weight=15.0,
      params={"command_name": "kick_target", "ideal_distance": 0.20, "std": 0.10},
    ),
    # V1.18: phase-aware post-kick shaping, linear-penalty form for both
    # the motion AND homing terms (the V1.17 exp-form homing reward
    # saturated to ~0 at post-kick joint deviation, leaving the policy
    # with no gradient toward HOME). Now: kick → 0.3s grace → strong
    # penalty for any joint deviation from HOME and any base motion.
    "post_kick_motion_penalty": RewardTermCfg(
      func=mdp.rewards.post_kick_motion_penalty,
      weight=-10.0,
      params={"command_name": "kick_target", "grace_steps": 15},
    ),
    "post_kick_homing_penalty": RewardTermCfg(
      func=mdp.rewards.post_kick_homing_penalty,
      weight=-10.0,
      params={"command_name": "kick_target", "grace_steps": 15},
    ),
    "forward_kick_penalty": RewardTermCfg(
      func=mdp.rewards.forward_kick_penalty,
      weight=-20.0,
      params={"command_name": "kick_target"},
    ),
    "foot_proximity": RewardTermCfg(
      func=mdp.rewards.foot_proximity,
      weight=-5.0,
    ),
    # V1.45: pelvis_orientation weight -1 → -3 (3× bump, within safe band).
    # Under V1.44's action delay some envs lost balance during the kick
    # swing and fell after whiff. A stronger continuous upright penalty
    # gives a per-step gradient against leaning regardless of whether the
    # foot contacted the ball — addresses "robot must not fall even when
    # missing the ball" requirement.
    "pelvis_orientation": RewardTermCfg(
      func=mdp.rewards.pelvis_orientation,
      weight=-3.0,
    ),
    # V1.11: head alignment weight -0.5 → -2.0 (4x). At -0.5 the policy
    # barely moved its head — the per-step penalty was too small relative
    # to other rewards. For deploy the head MUST be pre-aligned with the
    # ball before kick (ball-at-foot is only visible at near-max head
    # pitch). Stronger weight forces active perception throughout the
    # approach phase, not just at impact.
    "head_yaw_alignment": RewardTermCfg(
      func=mdp.rewards.head_yaw_alignment,
      weight=-2.0,
      params={"command_name": "kick_target"},
    ),
    "head_pitch_alignment": RewardTermCfg(
      func=mdp.rewards.head_pitch_alignment,
      weight=-2.0,
      params={"command_name": "kick_target"},
    ),
    # V1.8: sim2real-friendly regularizers. Without these, the policy
    # exploits unrealistic body postures (e.g., wild arm flails, extreme
    # torso twist) that a real Booster K1 can't reproduce. Penalize
    # deviation from the home pose for the upper body and any non-trivial
    # joint velocity / acceleration.
    # V1.10: arm_posture (exp +0.15) — kept as a small bonus.
    "arm_posture": RewardTermCfg(
      func=base_mdp.posture,
      weight=0.15,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            "ALeft_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "ARight_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
          ),
        ),
        "std": {".*": 0.3},
      },
    ),
    # V1.22: weight -15 → -100. V1.21 didn't suppress T-pose — policy
    # actively rolled the shoulders to 90° during play. Need a dominant
    # penalty plus the dedicated shoulder_roll_l2 term below.
    "arm_deviation_l2": RewardTermCfg(
      func=mdp.rewards.arm_deviation_l2,
      weight=-100.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            "ALeft_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "ARight_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
          ),
        ),
      },
    ),
    # V1.22 shoulder_roll_l2 REMOVED in V1.25. It penalized |shoulder_roll|²,
    # which for K1 actually means "penalize bringing arm down" because
    # Shoulder_Roll=0 is the T-pose and arms-down requires ±1.0 rad. The
    # broader arm_deviation_l2 now targets the geometrically correct
    # arms-down values for every arm joint, including shoulder roll.
    # V1.9: anti-spin penalty — the viser playback of warm-started V1.8
    # showed the policy generating kick force by torso rotation rather than
    # leg swing. Penalize base yaw rate to keep kicks human-like.
    "base_yaw_rate_l2": RewardTermCfg(
      func=mdp.rewards.base_yaw_rate_l2,
      weight=-0.5,
    ),
    # V1.12: anti-float penalty — viser playback showed the policy hopping
    # to launch the ball ("体が浮いている") with target_progress rising
    # but kick_success_rate dropping. Strong vertical velocity penalty to
    # ground the kick.
    "base_lin_vel_z_l2": RewardTermCfg(
      func=mdp.rewards.base_lin_vel_z_l2,
      weight=-2.0,
    ),
    "joint_vel_l2": RewardTermCfg(
      func=base_mdp.joint_vel_l2,
      weight=-1.0e-4,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=base_mdp.joint_acc_l2,
      weight=-1.0e-7,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "non_foot_collision": RewardTermCfg(
      func=mdp.rewards.non_foot_collision,
      weight=-100.0,
      params={"sensor_name": "self_collision", "threshold": 10.0},
    ),
    # V1.1: tighten action_rate to suppress kamikaze swing-then-fall pattern.
    "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-3.0),
    "joint_limit": RewardTermCfg(
      func=base_mdp.joint_pos_limits,
      weight=-100.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  # --------------------------------------------------------------------
  # Terminations
  # --------------------------------------------------------------------

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.terminations.time_out, time_out=True),
    "robot_fell": TerminationTermCfg(
      func=mdp.terminations.robot_fell_height,
      params={"min_height": 0.3},
    ),
    "robot_tilted": TerminationTermCfg(
      func=mdp.terminations.robot_fell_orientation,
      # V1.1: loosen tilt threshold so transient kick-swing lean doesn't
      # terminate; only real falls do.
      params={"max_tilt": 1.3},
    ),
    # V1.15: kick_completed removed entirely. Replacing it with the
    # post_kick_stillness reward — policy now learns to be still after
    # kick rather than terminating immediately, matching the user's
    # "kick then stand still" requirement.
    "ball_out_of_range": TerminationTermCfg(
      func=mdp.terminations.ball_out_of_range,
      params={"command_name": "kick_target", "max_distance": 8.0},
    ),
  }

  # --------------------------------------------------------------------
  # Assemble
  # --------------------------------------------------------------------

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    # V1.13: shortened back to 5.0 — ball-at-foot spawn (0.20-0.40m) doesn't
    # need walking time. 12s episodes let the policy wander pre-kick which
    # the user observed as "蹴ったあとボール追いかけている" (post-kick chase
    # was already ≤ 4 steps after V1.11's kick_completed fix). Tight 5s
    # budget forces the policy to commit to a kick early.
    episode_length_s=5.0,
  )
