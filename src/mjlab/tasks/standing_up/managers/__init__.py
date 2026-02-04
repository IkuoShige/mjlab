"""Custom managers for standing-up task."""

from mjlab.tasks.standing_up.managers.gaussian_product_reward_manager import (
  GaussianProductRewardManager,
  GroupedRewardTermCfg,
)
from mjlab.tasks.standing_up.managers.unactuated_masking_observation_manager import (
  UnactuatedMaskingObservationManager,
)

__all__ = [
  "GaussianProductRewardManager",
  "GroupedRewardTermCfg",
  "UnactuatedMaskingObservationManager",
]
