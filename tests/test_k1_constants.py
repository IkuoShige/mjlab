"""Tests for k1_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.booster_k1 import k1_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def k1_entity() -> Entity:
  return Entity(k1_constants.get_k1_robot_cfg())


@pytest.fixture(scope="module")
def k1_model(k1_entity: Entity) -> mujoco.MjModel:
  return k1_entity.spec.compile()


# fmt: off
@pytest.mark.parametrize(
  "actuator_config,stiffness,damping",
  [
    (k1_constants.K1_ACTUATOR_HIP_PITCH, k1_constants.STIFFNESS_HIP_PITCH, k1_constants.DAMPING_HIP_PITCH),
    (k1_constants.K1_ACTUATOR_HIP_ROLL, k1_constants.STIFFNESS_HIP_ROLL, k1_constants.DAMPING_HIP_ROLL),
    (k1_constants.K1_ACTUATOR_HIP_YAW, k1_constants.STIFFNESS_HIP_YAW, k1_constants.DAMPING_HIP_YAW),
    (k1_constants.K1_ACTUATOR_KNEE, k1_constants.STIFFNESS_KNEE, k1_constants.DAMPING_KNEE),
    (k1_constants.K1_ACTUATOR_ANKLE, k1_constants.STIFFNESS_ANKLE, k1_constants.DAMPING_ANKLE),
    (k1_constants.K1_ACTUATOR_ARM, k1_constants.STIFFNESS_ARM, k1_constants.DAMPING_ARM),
    (k1_constants.K1_ACTUATOR_HEAD, k1_constants.STIFFNESS_HEAD, k1_constants.DAMPING_HEAD),
  ],
)
# fmt: on
def test_actuator_parameters(k1_model, actuator_config, stiffness, damping):
  for i in range(k1_model.nu):
    actuator = k1_model.actuator(i)
    matches = actuator.name in actuator_config.target_names_expr
    if matches:
      assert actuator.gainprm[0] == stiffness
      assert actuator.biasprm[1] == -stiffness
      assert actuator.biasprm[2] == -damping
      assert actuator.forcerange[0] == -actuator_config.effort_limit
      assert actuator.forcerange[1] == actuator_config.effort_limit


def test_keyframe_base_position(k1_model) -> None:
  data = mujoco.MjData(k1_model)
  mujoco.mj_resetDataKeyframe(k1_model, data, 0)
  mujoco.mj_forward(k1_model, data)
  np.testing.assert_array_equal(data.qpos[:3], k1_constants.HOME_KEYFRAME.pos)
  np.testing.assert_array_equal(data.qpos[3:7], k1_constants.HOME_KEYFRAME.rot)


def test_keyframe_joint_positions(k1_entity, k1_model) -> None:
  key = k1_model.key("init_state")
  expected_joint_pos = k1_constants.HOME_KEYFRAME.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, k1_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    k1_entity.joint_names, expected_values, strict=True
  ):
    joint = k1_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch: expected {expected_value}, got {actual_value}",
    )


def test_foot_collision_geoms(k1_model) -> None:
  foot_pattern = r"^(left|right)_foot[1-4]_collision$"
  for i in range(k1_model.ngeom):
    geom = k1_model.geom(i)
    if re.match(foot_pattern, geom.name):
      assert geom.contype == 1
      assert geom.conaffinity == 1
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6


def test_collision_geom_count(k1_model) -> None:
  foot_pattern = r"^(left|right)_foot[1-4]_collision$"
  foot_geoms = [
    k1_model.geom(i).name
    for i in range(k1_model.ngeom)
    if re.match(foot_pattern, k1_model.geom(i).name)
  ]
  assert len(foot_geoms) == 8


def test_upper_body_collision_proxies_exist(k1_model) -> None:
  for geom_name in (
    "left_forearm_collision",
    "left_hand_collision",
    "right_forearm_collision",
    "right_hand_collision",
  ):
    geom = k1_model.geom(geom_name)
    assert geom.contype[0] == 1
    assert geom.conaffinity[0] == 1


def test_hip_pitch_offsets_match_booster_assets_update(k1_model) -> None:
  np.testing.assert_allclose(k1_model.body("Left_Hip_Pitch").pos, [0.0, 0.096, -0.062])
  np.testing.assert_allclose(
    k1_model.body("Right_Hip_Pitch").pos, [0.0, -0.096, -0.062]
  )


def test_locomotion_action_scale_locks_upper_body() -> None:
  for joint_name in k1_constants.K1_UPPER_BODY_JOINT_NAMES:
    assert k1_constants.K1_ACTION_SCALE[joint_name] > 0.0
    assert k1_constants.K1_LOCOMOTION_ACTION_SCALE[joint_name] == 0.0
  assert (
    k1_constants.K1_LOCOMOTION_ACTION_SCALE["Left_Hip_Pitch"]
    == k1_constants.K1_ACTION_SCALE["Left_Hip_Pitch"]
  )


def test_k1_entity_creation(k1_entity) -> None:
  assert k1_entity.num_actuators == 22
  assert k1_entity.num_joints == 22
  assert k1_entity.is_actuated
  assert not k1_entity.is_fixed_base
