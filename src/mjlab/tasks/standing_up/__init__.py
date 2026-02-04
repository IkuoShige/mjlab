"""Standing-up task for humanoid robots.

This module implements the HoST (Humanoid Standing-up Control) task,
ported from Isaac Gym to mjlab.
"""

# Import config subpackages to register tasks.
# This ensures tasks are registered when this module is imported.
import mjlab.tasks.standing_up.config.pi  # noqa: F401
from mjlab.tasks.standing_up.standing_up_env_cfg import make_standing_up_env_cfg

__all__ = ["make_standing_up_env_cfg"]
