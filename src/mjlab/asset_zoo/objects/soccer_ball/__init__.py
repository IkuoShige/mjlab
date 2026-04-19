"""Soccer ball asset for mjlab."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

SOCCER_BALL_RADIUS = 0.11
SOCCER_BALL_XML: Path = Path(__file__).parent / "soccer_ball.xml"


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(SOCCER_BALL_XML))


def get_soccer_ball_cfg() -> EntityCfg:
  """Get a fresh soccer ball entity configuration."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.7, 0.0, SOCCER_BALL_RADIUS),
    ),
    spec_fn=_get_spec,
  )
