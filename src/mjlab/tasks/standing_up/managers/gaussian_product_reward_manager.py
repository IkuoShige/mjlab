"""Gaussian Product Reward Manager for standing-up task.

HoST uses a specific reward structure:
- Task rewards: Product aggregation (Gaussian product).
- Constraint/style rewards: Additive aggregation.
- Groups: ['task', 'regu', 'style', 'target'] with configurable weights.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class GroupedRewardTermCfg(ManagerTermBaseCfg):
  """Configuration for a reward term with group assignment."""

  func: Any
  """The callable that computes this reward term's value."""

  weight: float
  """Weight multiplier for this reward term."""

  group: str = "task"
  """Reward group: 'task' (product), 'regu', 'style', or 'target' (additive)."""


class GaussianProductRewardManager(ManagerBase):
  """Reward manager using Gaussian product for task rewards.

  This manager implements the HoST reward structure:
  - 'task' group: Rewards are multiplied together (Gaussian product).
  - Other groups ('regu', 'style', 'target'): Rewards are summed.
  - Final reward = sum(group_weight[g] * group_reward[g] for all groups).

  Attributes:
    group_names: List of reward group names.
    group_weights: Dictionary mapping group names to weights.
  """

  _env: ManagerBasedRlEnv

  def __init__(
    self,
    cfg: dict[str, GroupedRewardTermCfg],
    env: ManagerBasedRlEnv,
    *,
    group_names: tuple[str, ...] = ("task", "regu", "style", "target"),
    group_weights: tuple[float, ...] = (2.5, 0.1, 1.0, 1.0),
    scale_by_dt: bool = True,
  ):
    """Initialize the Gaussian Product Reward Manager.

    Args:
      cfg: Dictionary mapping term names to configurations.
      env: The environment.
      group_names: Names of reward groups. Default: ('task', 'regu', 'style', 'target').
      group_weights: Weights for each group. Default: (2.5, 0.1, 1.0, 1.0).
      scale_by_dt: Whether to scale rewards by timestep. Default True.
    """
    self._term_names: list[str] = []
    self._term_cfgs: list[GroupedRewardTermCfg] = []
    self._class_term_cfgs: list[GroupedRewardTermCfg] = []
    self._scale_by_dt = scale_by_dt

    self.group_names = group_names
    self.group_weights = dict(zip(group_names, group_weights, strict=False))

    self.cfg = deepcopy(cfg)
    super().__init__(env=env)

    # Episode sums per term.
    self._episode_sums: dict[str, torch.Tensor] = {}
    for term_name in self._term_names:
      self._episode_sums[term_name] = torch.zeros(
        self.num_envs, dtype=torch.float, device=self.device
      )

    # Episode sums per group.
    self._group_episode_sums: dict[str, torch.Tensor] = {}
    for group_name in self.group_names:
      self._group_episode_sums[group_name] = torch.zeros(
        self.num_envs, dtype=torch.float, device=self.device
      )

    self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    self._step_reward = torch.zeros(
      (self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device
    )
    self._group_rewards = torch.zeros(
      (self.num_envs, len(self.group_names)), dtype=torch.float, device=self.device
    )

  def __str__(self) -> str:
    msg = (
      f"<GaussianProductRewardManager> contains {len(self._term_names)} active terms.\n"
    )
    msg += (
      f"Groups: {self.group_names} with weights {list(self.group_weights.values())}\n"
    )

    table = PrettyTable()
    table.title = "Active Reward Terms"
    table.field_names = ["Index", "Name", "Group", "Weight"]
    table.align["Name"] = "l"
    table.align["Weight"] = "r"
    for index, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      table.add_row([index, name, term_cfg.group, term_cfg.weight])
    msg += table.get_string()
    msg += "\n"
    return msg

  @property
  def active_terms(self) -> list[str]:
    """List of active term names."""
    return self._term_names

  def reset(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> dict[str, torch.Tensor]:
    """Reset episode sums and return logged metrics.

    Args:
      env_ids: Environment IDs to reset. If None, resets all.

    Returns:
      Dictionary of episode metrics.
    """
    if env_ids is None:
      env_ids = slice(None)

    extras: dict[str, torch.Tensor] = {}

    # Log per-term episode rewards.
    for key in self._episode_sums.keys():
      episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
      extras["Episode_Reward/" + key] = (
        episodic_sum_avg / self._env.max_episode_length_s
      )
      self._episode_sums[key][env_ids] = 0.0

    # Log per-group episode rewards.
    for group_name in self.group_names:
      group_sum_avg = torch.mean(self._group_episode_sums[group_name][env_ids])
      extras[f"Episode_Reward/group_{group_name}"] = (
        group_sum_avg / self._env.max_episode_length_s
      )
      self._group_episode_sums[group_name][env_ids] = 0.0

    # Reset class-based term instances.
    for term_cfg in self._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)

    return extras

  def compute(self, dt: float) -> torch.Tensor:
    """Compute total reward using Gaussian product for task group.

    Args:
      dt: Timestep duration.

    Returns:
      Total reward tensor of shape (num_envs,).
    """
    scale = dt if self._scale_by_dt else 1.0

    # Initialize group rewards.
    # Task group starts at 1 (product), others start at 0 (sum).
    group_rewards: dict[str, torch.Tensor] = {}
    for group_name in self.group_names:
      if group_name == "task":
        group_rewards[group_name] = torch.ones(
          self.num_envs, dtype=torch.float, device=self.device
        )
      else:
        group_rewards[group_name] = torch.zeros(
          self.num_envs, dtype=torch.float, device=self.device
        )

    # Compute each term and aggregate by group.
    for term_idx, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      if term_cfg.weight == 0.0:
        self._step_reward[:, term_idx] = 0.0
        continue

      # Compute raw reward value.
      value = term_cfg.func(self._env, **term_cfg.params)

      # Handle NaN/Inf.
      value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

      # Clip raw values to prevent overflow.
      value = value.clamp(-1e6, 1e6)

      # Aggregate by group.
      group = term_cfg.group
      if group == "task":
        # Product aggregation: multiply raw values (weight is ignored for task group).
        # Task rewards should be in [0, 1] range for Gaussian product.
        # The group_weight is applied after the product.
        group_rewards[group] *= value.clamp(0.0, 1e4)
        # Store raw value for logging.
        self._step_reward[:, term_idx] = value
        self._episode_sums[name] += value * scale
      else:
        # Apply weight for non-task groups.
        weighted_value = value * term_cfg.weight
        # Clip weighted values as additional safety.
        weighted_value = weighted_value.clamp(-1e4, 1e4)
        # Additive aggregation with dt scaling.
        group_rewards[group] += weighted_value * scale
        # Store for logging.
        self._step_reward[:, term_idx] = weighted_value
        self._episode_sums[name] += weighted_value * scale

    # Compute total reward as weighted sum of groups.
    self._reward_buf[:] = 0.0
    for group_idx, group_name in enumerate(self.group_names):
      group_weight = self.group_weights[group_name]
      group_value = group_rewards[group_name]

      # Clip group values to prevent overflow.
      group_value = group_value.clamp(-1e4, 1e4)

      self._reward_buf += group_weight * group_value
      self._group_rewards[:, group_idx] = group_value
      self._group_episode_sums[group_name] += group_weight * group_value

    # Final clip on total reward.
    self._reward_buf = self._reward_buf.clamp(-100.0, 100.0)

    return self._reward_buf

  def get_active_iterable_terms(self, env_idx: int) -> list[tuple[str, list[float]]]:
    """Get active terms for visualization.

    Args:
      env_idx: Environment index.

    Returns:
      List of (name, [value]) tuples.
    """
    terms = []
    for idx, name in enumerate(self._term_names):
      terms.append((name, [self._step_reward[env_idx, idx].cpu().item()]))
    return terms

  def get_term_cfg(self, term_name: str) -> GroupedRewardTermCfg:
    """Get configuration for a specific term.

    Args:
      term_name: Name of the term.

    Returns:
      Term configuration.

    Raises:
      ValueError: If term not found.
    """
    if term_name not in self._term_names:
      raise ValueError(f"Term '{term_name}' not found in active terms.")
    return self._term_cfgs[self._term_names.index(term_name)]

  def _prepare_terms(self) -> None:
    """Prepare reward terms from configuration."""
    for term_name, term_cfg in self.cfg.items():
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue

      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)

      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
        self._class_term_cfgs.append(term_cfg)
