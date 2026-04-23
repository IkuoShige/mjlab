"""Booster K1 constants."""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

K1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "booster_k1" / "xmls" / "k1.xml"
)
assert K1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(K1_XML))


##
# Actuator config.
##

ARMATURE_HIP_PITCH = 0.0478125
ARMATURE_HIP_ROLL = 0.0339552
ARMATURE_HIP_YAW = 0.0282528
ARMATURE_KNEE = 0.095625
ARMATURE_ANKLE = 0.0565
ARMATURE_ARM = 0.001
ARMATURE_HEAD = 0.002

EFFORT_HIP_PITCH = 30.0
EFFORT_HIP_ROLL = 35.0
EFFORT_HIP_YAW = 20.0
EFFORT_KNEE = 40.0
EFFORT_ANKLE = 20.0
EFFORT_ARM = 14.0
EFFORT_HEAD = 6.0

NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0

STIFFNESS_HIP_PITCH = ARMATURE_HIP_PITCH * NATURAL_FREQ**2
STIFFNESS_HIP_ROLL = ARMATURE_HIP_ROLL * NATURAL_FREQ**2
STIFFNESS_HIP_YAW = ARMATURE_HIP_YAW * NATURAL_FREQ**2
STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ**2
STIFFNESS_ANKLE = ARMATURE_ANKLE * NATURAL_FREQ**2
STIFFNESS_ARM = ARMATURE_ARM * NATURAL_FREQ**2
STIFFNESS_HEAD = ARMATURE_HEAD * NATURAL_FREQ**2

DAMPING_HIP_PITCH = 2.0 * DAMPING_RATIO * ARMATURE_HIP_PITCH * NATURAL_FREQ
DAMPING_HIP_ROLL = 2.0 * DAMPING_RATIO * ARMATURE_HIP_ROLL * NATURAL_FREQ
DAMPING_HIP_YAW = 2.0 * DAMPING_RATIO * ARMATURE_HIP_YAW * NATURAL_FREQ
DAMPING_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE * NATURAL_FREQ
DAMPING_ARM = 2.0 * DAMPING_RATIO * ARMATURE_ARM * NATURAL_FREQ
DAMPING_HEAD = 2.0 * DAMPING_RATIO * ARMATURE_HEAD * NATURAL_FREQ

K1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("Left_Hip_Pitch", "Right_Hip_Pitch"),
  stiffness=STIFFNESS_HIP_PITCH,
  damping=DAMPING_HIP_PITCH,
  effort_limit=EFFORT_HIP_PITCH,
  armature=ARMATURE_HIP_PITCH,
)
K1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("Left_Hip_Roll", "Right_Hip_Roll"),
  stiffness=STIFFNESS_HIP_ROLL,
  damping=DAMPING_HIP_ROLL,
  effort_limit=EFFORT_HIP_ROLL,
  armature=ARMATURE_HIP_ROLL,
)
K1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("Left_Hip_Yaw", "Right_Hip_Yaw"),
  stiffness=STIFFNESS_HIP_YAW,
  damping=DAMPING_HIP_YAW,
  effort_limit=EFFORT_HIP_YAW,
  armature=ARMATURE_HIP_YAW,
)
K1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=("Left_Knee_Pitch", "Right_Knee_Pitch"),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=EFFORT_KNEE,
  armature=ARMATURE_KNEE,
)
K1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
  ),
  stiffness=STIFFNESS_ANKLE,
  damping=DAMPING_ANKLE,
  effort_limit=EFFORT_ANKLE,
  armature=ARMATURE_ANKLE,
)
K1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
  ),
  stiffness=STIFFNESS_ARM,
  damping=DAMPING_ARM,
  effort_limit=EFFORT_ARM,
  armature=ARMATURE_ARM,
)
K1_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("AAHead_yaw", "Head_pitch"),
  stiffness=STIFFNESS_HEAD,
  damping=DAMPING_HEAD,
  effort_limit=EFFORT_HEAD,
  armature=ARMATURE_HEAD,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.54),
  joint_pos={
    "Left_Hip_Pitch": -0.2,
    "Right_Hip_Pitch": -0.2,
    "Left_Knee_Pitch": 0.4,
    "Right_Knee_Pitch": 0.4,
    "Left_Ankle_Pitch": -0.2,
    "Right_Ankle_Pitch": -0.2,
    "ALeft_Shoulder_Pitch": 0.3,
    "ARight_Shoulder_Pitch": 0.3,
    "Left_Elbow_Pitch": 0.5,
    "Right_Elbow_Pitch": 0.5,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

K1_FOOT_GEOM_REGEX = r"^(left|right)_foot[1-4]_collision$"
K1_UPPER_BODY_JOINT_NAMES = (
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
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(K1_FOOT_GEOM_REGEX,),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  disable_other_geoms=False,
)

##
# Final config.
##

K1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    K1_ACTUATOR_HIP_PITCH,
    K1_ACTUATOR_HIP_ROLL,
    K1_ACTUATOR_HIP_YAW,
    K1_ACTUATOR_KNEE,
    K1_ACTUATOR_ANKLE,
    K1_ACTUATOR_ARM,
    K1_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_k1_robot_cfg() -> EntityCfg:
  """Get a fresh K1 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=K1_ARTICULATION,
  )


# Manually tuned action scales matching G1's working values
# (G1 uses 0.35 rad for knee, 0.55 rad for hip pitch, 0.44 rad for ankles, etc.).
# The generic ``0.25 * effort / stiffness`` formula gives pathologically small
# scales on K1's legs (knee 1.5°, ankle 1.3°) because K1's stiffness-to-effort
# ratio is ~13x G1's — the formula couples a physical PD property to an RL
# hyperparameter, which breaks here. Motion-capture kick swings need ~20°+ of
# knee extension, so any scale <10° makes the policy physically unable to
# match the reference motion regardless of reward/architecture tuning.
K1_ACTION_SCALE: dict[str, float] = {
  # Legs: mirror G1 values for the analogous joints.
  "Left_Hip_Pitch": 0.55,
  "Right_Hip_Pitch": 0.55,
  "Left_Hip_Roll": 0.35,
  "Right_Hip_Roll": 0.35,
  "Left_Hip_Yaw": 0.35,
  "Right_Hip_Yaw": 0.35,
  "Left_Knee_Pitch": 0.35,
  "Right_Knee_Pitch": 0.35,
  "Left_Ankle_Pitch": 0.44,
  "Right_Ankle_Pitch": 0.44,
  "Left_Ankle_Roll": 0.44,
  "Right_Ankle_Roll": 0.44,
  # Arms: mirror G1 shoulder/elbow scales; cap K1's extreme 50.8° shoulder.
  "ALeft_Shoulder_Pitch": 0.44,
  "ARight_Shoulder_Pitch": 0.44,
  "Left_Shoulder_Roll": 0.44,
  "Right_Shoulder_Roll": 0.44,
  "Left_Elbow_Pitch": 0.44,
  "Right_Elbow_Pitch": 0.44,
  "Left_Elbow_Yaw": 0.44,
  "Right_Elbow_Yaw": 0.44,
  # Head: keep a modest range.
  "AAHead_yaw": 0.19,
  "Head_pitch": 0.19,
}


K1_LOCOMOTION_ACTION_SCALE = dict(K1_ACTION_SCALE)
for joint_name in K1_UPPER_BODY_JOINT_NAMES:
  K1_LOCOMOTION_ACTION_SCALE[joint_name] = 0.0


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_k1_robot_cfg())

  viewer.launch(robot.spec.compile())
