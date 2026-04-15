"""Tests for actions."""

from pathlib import Path
from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device, load_fixture_xml

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import BuiltinMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  RelativeJointPositionActionCfg,
  SiteEffortActionCfg,
  TendonEffortActionCfg,
  TendonLengthActionCfg,
  TendonVelocityActionCfg,
)
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.locomotion_memory.mdp import ReferenceJointPositionActionCfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def fixtures_dir():
  return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tendon_finger_entity(fixtures_dir, device):
  return make_entity(
    fixtures_dir / "tendon_finger.xml",
    ("finger_tendon",),
    TransmissionType.TENDON,
    device,
    from_file=True,
  )


@pytest.fixture(scope="module")
def fixed_base_entity(fixtures_dir, device):
  return make_entity(
    fixtures_dir / "fixed_base_articulated.xml",
    ("joint.*",),
    TransmissionType.JOINT,
    device,
    from_file=True,
  )


@pytest.fixture(scope="module")
def floating_base_entity(device):
  return make_entity(
    load_fixture_xml("floating_base_articulated"),
    ("joint.*",),
    TransmissionType.JOINT,
    device,
    from_file=False,
  )


def make_entity(xml_or_path, target_expr, transmission_type, device, from_file=False):
  """Create and initialize entity."""

  def spec_fn():
    if from_file:
      return mujoco.MjSpec.from_file(str(xml_or_path))
    return mujoco.MjSpec.from_string(xml_or_path)

  cfg = EntityCfg(
    spec_fn=spec_fn,
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=target_expr,
          transmission_type=transmission_type,
          effort_limit=10.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=4, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity


def make_env(entity, name, device):
  """Create mock environment."""
  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = 4
  env.device = device
  env.scene = {name: entity}
  return env


def test_base_action_applies_scale_and_offset(tendon_finger_entity, device):
  """BaseAction: processed = raw * scale + offset."""
  entity = tendon_finger_entity
  env = make_env(entity, "finger", device)

  cfg = TendonLengthActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
    scale=2.0,
    offset=0.5,
  )
  action = cfg.build(env)

  raw = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
  action.process_actions(raw)

  assert torch.allclose(action._processed_actions, raw * 2.0 + 0.5)


def test_base_action_reset_zeros_specific_envs(tendon_finger_entity, device):
  """BaseAction.reset() zeros raw_action for specified env_ids only."""
  entity = tendon_finger_entity
  env = make_env(entity, "finger", device)

  cfg = TendonLengthActionCfg(entity_name="finger", actuator_names=("finger_tendon",))
  action = cfg.build(env)

  action.process_actions(torch.ones(4, 1, device=device))
  action.reset(env_ids=torch.tensor([0, 2], device=device))

  assert torch.all(action.raw_action[[0, 2]] == 0.0)
  assert torch.all(action.raw_action[[1, 3]] == 1.0)


@pytest.mark.parametrize(
  "cfg_cls,target_attr,fixture,target_expr,transmission,entity_name",
  [
    # Joints.
    (
      JointPositionActionCfg,
      "joint_pos_target",
      "floating_base_articulated",
      ("joint.*",),
      TransmissionType.JOINT,
      "robot",
    ),
    # Tendons.
    (
      TendonLengthActionCfg,
      "tendon_len_target",
      "tendon_finger.xml",
      ("finger_tendon",),
      TransmissionType.TENDON,
      "finger",
    ),
    (
      TendonVelocityActionCfg,
      "tendon_vel_target",
      "tendon_finger.xml",
      ("finger_tendon",),
      TransmissionType.TENDON,
      "finger",
    ),
    (
      TendonEffortActionCfg,
      "tendon_effort_target",
      "tendon_finger.xml",
      ("finger_tendon",),
      TransmissionType.TENDON,
      "finger",
    ),
    # Sites.
    (
      SiteEffortActionCfg,
      "site_effort_target",
      "quadcopter.xml",
      ("rotor_.*",),
      TransmissionType.SITE,
      "drone",
    ),
  ],
)
def test_action_sets_entity_target(
  fixtures_dir,
  device,
  cfg_cls,
  target_attr,
  fixture,
  target_expr,
  transmission,
  entity_name,
):
  """Each action type writes to correct entity.data field."""
  if fixture.endswith(".xml"):
    entity = make_entity(
      fixtures_dir / fixture, target_expr, transmission, device, from_file=True
    )
  else:
    entity = make_entity(
      load_fixture_xml(fixture), target_expr, transmission, device, from_file=False
    )

  env = make_env(entity, entity_name, device)
  cfg = cfg_cls(entity_name=entity_name, actuator_names=target_expr)
  action = cfg.build(env)

  target = torch.arange(4 * action.action_dim, device=device, dtype=torch.float32)
  target = target.reshape(4, action.action_dim) * 0.1

  action.process_actions(target)
  action.apply_actions()

  entity_target = getattr(entity.data, target_attr)
  assert torch.allclose(entity_target, target)


def test_base_action_clip(fixed_base_entity, device):
  """BaseAction: clip clamps only matched actuators; others stay unclipped."""
  entity = fixed_base_entity
  env = make_env(entity, "robot", device)

  # Clip only joint1; joint2 should remain unclipped.
  cfg = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    scale=1.0,
    use_default_offset=False,
    clip={"joint1": (-0.5, 0.5)},
  )
  action = cfg.build(env)

  # joint1=2.0 should be clipped to 0.5, joint2=2.0 should pass through.
  raw = torch.tensor([[2.0, 2.0]], device=device).expand(4, -1)
  action.process_actions(raw)

  assert torch.allclose(
    action._processed_actions[:, 0], torch.tensor(0.5, device=device)
  )
  assert torch.allclose(
    action._processed_actions[:, 1], torch.tensor(2.0, device=device)
  )


def test_relative_joint_position_zero_action(floating_base_entity, device):
  """With zero action, targets equal current_pos."""
  entity = floating_base_entity
  env = make_env(entity, "robot", device)

  cfg = RelativeJointPositionActionCfg(
    entity_name="robot", actuator_names=("joint.*",), scale=1.0
  )
  action = cfg.build(env)

  current_pos = entity.data.joint_pos[:, action.target_ids].clone()

  action.process_actions(torch.zeros(4, action.action_dim, device=device))
  action.apply_actions()

  assert torch.allclose(entity.data.joint_pos_target[:, action.target_ids], current_pos)


def test_relative_joint_position_nonzero_action(floating_base_entity, device):
  """With nonzero action, targets shift by action * scale from current."""
  entity = floating_base_entity
  env = make_env(entity, "robot", device)

  scale = 0.1
  cfg = RelativeJointPositionActionCfg(
    entity_name="robot", actuator_names=("joint.*",), scale=scale
  )
  action = cfg.build(env)

  current_pos = entity.data.joint_pos[:, action.target_ids].clone()

  raw = torch.ones(4, action.action_dim, device=device)
  action.process_actions(raw)
  action.apply_actions()

  expected = current_pos + raw * scale
  assert torch.allclose(entity.data.joint_pos_target[:, action.target_ids], expected)


def test_relative_joint_position_ignores_encoder_bias(floating_base_entity, device):
  """Encoder bias must not affect the target: target = current_pos + delta."""
  entity = floating_base_entity
  env = make_env(entity, "robot", device)

  cfg = RelativeJointPositionActionCfg(
    entity_name="robot", actuator_names=("joint.*",), scale=1.0
  )
  action = cfg.build(env)

  entity.data.encoder_bias[:, action.target_ids] = 0.05

  current_pos = entity.data.joint_pos[:, action.target_ids].clone()
  delta = 0.1
  raw = torch.full((4, action.action_dim), delta, device=device)
  action.process_actions(raw)
  action.apply_actions()

  assert torch.allclose(
    entity.data.joint_pos_target[:, action.target_ids], current_pos + delta
  )

  # Reset bias so this shared fixture doesn't affect other tests.
  entity.data.encoder_bias[:, action.target_ids] = 0.0


def test_relative_joint_position_offset_raises(floating_base_entity, device):
  """Setting offset on RelativeJointPositionActionCfg raises ValueError."""
  with pytest.raises(ValueError, match="offset"):
    RelativeJointPositionActionCfg(
      entity_name="robot", actuator_names=("joint.*",), offset=0.5
    )


def test_reference_joint_position_action_uses_memory_reference(
  floating_base_entity, device
):
  """Residual actions should be applied around the active snippet pose."""
  entity = floating_base_entity
  env = make_env(entity, "robot", device)
  env.command_manager = Mock()

  cfg = ReferenceJointPositionActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    scale=0.5,
  )
  action = cfg.build(env)

  reference_joint_pos = torch.full_like(entity.data.joint_pos, 0.25)
  env.command_manager.get_term = Mock(
    return_value=Mock(
      reference_joint_pos=reference_joint_pos,
      snippet_id=torch.zeros(env.num_envs, dtype=torch.long, device=device),
    )
  )

  raw = torch.full((env.num_envs, action.action_dim), 0.2, device=device)
  action.process_actions(raw)
  action.apply_actions()

  expected = reference_joint_pos[:, action.target_ids] + raw * 0.5
  assert torch.allclose(entity.data.joint_pos_target[:, action.target_ids], expected)


def test_reference_joint_position_action_falls_back_to_default_pose_when_inactive(
  floating_base_entity, device
):
  """Inactive retrieval should fall back to the robot default pose."""
  entity = floating_base_entity
  env = make_env(entity, "robot", device)
  env.command_manager = Mock()

  cfg = ReferenceJointPositionActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    scale=1.0,
  )
  action = cfg.build(env)

  env.command_manager.get_term = Mock(
    return_value=Mock(
      reference_joint_pos=torch.full_like(entity.data.joint_pos, 0.9),
      snippet_id=torch.full((env.num_envs,), -1, dtype=torch.long, device=device),
    )
  )

  entity.data.encoder_bias[:, action.target_ids] = 0.05
  action.process_actions(torch.zeros(env.num_envs, action.action_dim, device=device))
  action.apply_actions()

  expected = (
    entity.data.default_joint_pos[:, action.target_ids]
    - entity.data.encoder_bias[:, action.target_ids]
  )
  assert torch.allclose(entity.data.joint_pos_target[:, action.target_ids], expected)

  entity.data.encoder_bias[:, action.target_ids] = 0.0


def test_reference_joint_position_action_scales_residuals_down_for_turn_commands(
  floating_base_entity, device
):
  """In-place turn commands should reduce residual authority around the reference."""
  entity = floating_base_entity
  env = make_env(entity, "robot", device)
  env.command_manager = Mock()

  cfg = ReferenceJointPositionActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    scale=1.0,
    turn_command_name="twist",
    turn_residual_scale=0.25,
    turn_residual_joint_names=("joint1",),
    turn_min_abs_yaw_command=0.2,
    turn_max_forward_speed=0.25,
    turn_max_lateral_speed=0.05,
  )
  action = cfg.build(env)

  reference_joint_pos = torch.full_like(entity.data.joint_pos, 0.15)
  env.command_manager.get_term = Mock(
    return_value=Mock(
      reference_joint_pos=reference_joint_pos,
      snippet_id=torch.zeros(env.num_envs, dtype=torch.long, device=device),
    )
  )
  env.command_manager.get_command = Mock(
    return_value=torch.tensor(
      [
        [0.05, 0.0, 0.4],
        [0.40, 0.0, 0.4],
        [0.05, 0.0, 0.05],
        [0.05, 0.0, -0.4],
      ],
      device=device,
    )
  )

  raw = torch.full((env.num_envs, action.action_dim), 0.2, device=device)
  action.process_actions(raw)
  action.apply_actions()

  expected = reference_joint_pos[:, action.target_ids] + 0.2
  expected[[0, 3], 0] = reference_joint_pos[[0, 3], 0] + 0.05
  assert torch.allclose(entity.data.joint_pos_target[:, action.target_ids], expected)


def test_reference_joint_position_action_further_scales_turn_in_place_v2_residuals(
  floating_base_entity, device
):
  """turn_in_place_v2 snippets should follow the reference more tightly."""
  entity = floating_base_entity
  env = make_env(entity, "robot", device)
  env.command_manager = Mock()

  cfg = ReferenceJointPositionActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    scale=1.0,
    turn_command_name="twist",
    turn_residual_scale=1.0,
    turn_in_place_v2_residual_scale=0.25,
    turn_in_place_v2_residual_joint_names=("joint1",),
    turn_min_abs_yaw_command=0.2,
    turn_max_forward_speed=0.25,
    turn_max_lateral_speed=0.05,
  )
  action = cfg.build(env)

  reference_joint_pos = torch.full_like(entity.data.joint_pos, 0.15)
  env.command_manager.get_term = Mock(
    return_value=Mock(
      reference_joint_pos=reference_joint_pos,
      snippet_id=torch.zeros(env.num_envs, dtype=torch.long, device=device),
      turn_in_place_v2_active=torch.tensor([True, False, True, False], device=device),
    )
  )
  env.command_manager.get_command = Mock(
    return_value=torch.tensor(
      [
        [0.05, 0.0, 0.4],
        [0.05, 0.0, 0.4],
        [0.40, 0.0, 0.4],
        [0.05, 0.0, 0.05],
      ],
      device=device,
    )
  )

  raw = torch.full((env.num_envs, action.action_dim), 0.2, device=device)
  action.process_actions(raw)
  action.apply_actions()

  expected = reference_joint_pos[:, action.target_ids] + 0.2
  expected[0, 0] = reference_joint_pos[0, 0] + 0.05
  assert torch.allclose(entity.data.joint_pos_target[:, action.target_ids], expected)


def test_reference_joint_position_action_scales_turn_snippet_residuals(
  floating_base_entity, device
):
  """Turn snippets should follow the reference more tightly than straight snippets."""
  entity = floating_base_entity
  env = make_env(entity, "robot", device)
  env.command_manager = Mock()

  cfg = ReferenceJointPositionActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    scale=1.0,
    turn_command_name="twist",
    turn_residual_scale=1.0,
    turn_snippet_residual_scale=0.5,
    turn_snippet_residual_joint_names=("joint1",),
    turn_min_abs_yaw_command=0.2,
    turn_max_forward_speed=0.25,
    turn_max_lateral_speed=0.05,
  )
  action = cfg.build(env)

  reference_joint_pos = torch.full_like(entity.data.joint_pos, 0.15)
  env.command_manager.get_term = Mock(
    return_value=Mock(
      reference_joint_pos=reference_joint_pos,
      snippet_id=torch.zeros(env.num_envs, dtype=torch.long, device=device),
      turn_snippet_active=torch.tensor([True, False, True, False], device=device),
      turn_in_place_v2_active=torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
      ),
    )
  )
  env.command_manager.get_command = Mock(
    return_value=torch.tensor(
      [
        [0.05, 0.0, 0.4],
        [0.05, 0.0, 0.4],
        [0.40, 0.0, 0.4],
        [0.05, 0.0, 0.05],
      ],
      device=device,
    )
  )

  raw = torch.full((env.num_envs, action.action_dim), 0.2, device=device)
  action.process_actions(raw)
  action.apply_actions()

  expected = reference_joint_pos[:, action.target_ids] + 0.2
  expected[0, 0] = reference_joint_pos[0, 0] + 0.1
  assert torch.allclose(entity.data.joint_pos_target[:, action.target_ids], expected)
