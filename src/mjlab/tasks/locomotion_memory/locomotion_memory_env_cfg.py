"""Locomotion-memory task configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.locomotion_memory import mdp
from mjlab.tasks.locomotion_memory.mdp import LocomotionMemoryCommandCfg
from mjlab.tasks.locomotion_memory.memory_db import default_k1_locomotion_memory_file
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def make_locomotion_memory_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create a base locomotion-memory environment configuration.

  The first version keeps the execution stack aligned with the velocity task and
  adds a dedicated proposal command/observation surface for future retrieval.
  """
  cfg = make_velocity_env_cfg()

  memory_command: CommandTermCfg = LocomotionMemoryCommandCfg(
    entity_name="robot",
    memory_file=default_k1_locomotion_memory_file(),
    twist_command_name="twist",
    contact_sensor_name="feet_ground_contact",
    resampling_time_range=(0.1, 0.1),
    proposal_dim=24,
    debug_vis=False,
  )
  cfg.commands["memory"] = memory_command

  cfg.actions["joint_pos"] = mdp.ReferenceJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=0.5,
    reference_command_name="memory",
  )

  cfg.observations["actor"].terms["memory_proposal"] = ObservationTermCfg(
    func=mdp.proposal_vector,
    params={"command_name": "memory"},
  )
  cfg.observations["actor"].terms["memory_features"] = ObservationTermCfg(
    func=mdp.proposal_features,
    params={"command_name": "memory"},
  )
  cfg.observations["critic"].terms["memory_proposal"] = ObservationTermCfg(
    func=mdp.proposal_vector,
    params={"command_name": "memory"},
  )
  cfg.observations["critic"].terms["memory_features"] = ObservationTermCfg(
    func=mdp.proposal_features,
    params={"command_name": "memory"},
  )
  cfg.observations["critic"].terms["memory_state"] = ObservationTermCfg(
    func=mdp.proposal_state,
    params={"command_name": "memory"},
  )

  cfg.rewards["memory_contact_validity"] = RewardTermCfg(
    func=mdp.memory_contact_validity,
    weight=0.05,
    params={"command_name": "memory"},
  )
  cfg.rewards["memory_phase_validity"] = RewardTermCfg(
    func=mdp.memory_phase_validity,
    weight=0.03,
    params={"command_name": "memory"},
  )
  cfg.rewards["memory_retrieval_switch_cost"] = RewardTermCfg(
    func=mdp.memory_retrieval_switch_cost,
    weight=-0.02,
    params={"command_name": "memory"},
  )
  cfg.rewards["memory_transition_bonus"] = RewardTermCfg(
    func=mdp.memory_transition_bonus,
    weight=0.02,
    params={
      "command_name": "memory",
      "twist_command_name": "twist",
      "speed_threshold": 1.5,
      "speed_margin": 0.2,
    },
  )

  return cfg
