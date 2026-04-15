"""Unit tests for locomotion-memory retrieval helper logic."""

import torch

from mjlab.tasks.locomotion_memory.mdp.commands import (
  cadence_bias_mask,
  cadence_continuity_score,
  event_synchronized_switch_mask,
  phase_boundary_mask,
  pivot_bias_score,
  prioritize_prefixed_candidates,
  prioritize_turn_in_place_candidates,
  reference_swing_lift_score,
  snippet_source_prefix_mask,
  straight_reference_quality_score,
  switch_allowed_mask,
  turn_in_place_v2_bias_score,
  turn_intent_bias_score,
  turn_intent_mask,
  wrapped_phase_distance,
)


def test_turn_intent_bias_prefers_stepping_turn_snippets_for_low_speed_yaw_commands():
  twist_cmd = torch.tensor(
    [
      [0.05, 0.0, 0.4],
      [0.6, 0.0, 0.4],
    ]
  )
  gait_id = torch.tensor(
    [
      [2, 0, 2],
      [2, 0, 2],
    ]
  )
  mean_vx = torch.tensor(
    [
      [0.03, 0.55, 0.04],
      [0.03, 0.55, 0.04],
    ]
  )
  mean_vy = torch.zeros_like(mean_vx)
  mean_yaw_rate = torch.tensor(
    [
      [0.45, 0.08, 0.45],
      [0.45, 0.08, 0.45],
    ]
  )
  cadence = torch.tensor(
    [
      [1.3, 1.3, 0.0],
      [1.3, 1.3, 0.0],
    ]
  )

  bias = turn_intent_bias_score(
    twist_cmd,
    gait_id,
    mean_vx,
    mean_vy,
    mean_yaw_rate,
    cadence,
    turn_gait_id=2,
    min_turn_yaw_command=0.2,
    max_turn_forward_speed=0.25,
    max_turn_lateral_speed=0.05,
    in_place_speed_scale=0.12,
    yaw_rate_scale=0.35,
    min_turn_cadence=1.3,
  )

  assert bias[0, 0] > 0.0
  assert bias[0, 0] > bias[0, 1]
  assert bias[0, 0] > bias[0, 2]
  assert torch.allclose(bias[1], torch.zeros_like(bias[1]))


def test_switch_allowed_mask_requires_minimum_hold_time_for_active_snippets():
  allowed = switch_allowed_mask(
    elapsed_time=torch.tensor([0.1, 0.3, 0.1]),
    active_snippet=torch.tensor([True, True, False]),
    min_snippet_hold_s=0.25,
  )

  assert torch.equal(allowed, torch.tensor([False, True, True]))


def test_phase_boundary_mask_targets_zero_and_half_cycle_boundaries():
  mask = phase_boundary_mask(
    torch.tensor([0.01, 0.48, 0.31, 0.91]),
    tolerance=0.03,
  )

  assert torch.equal(mask, torch.tensor([True, True, False, False]))


def test_event_synchronized_switch_mask_waits_for_touchdown_or_boundary():
  mask = event_synchronized_switch_mask(
    active_snippet=torch.tensor([True, True, True, False]),
    elapsed_time=torch.tensor([0.30, 0.30, 0.65, 0.05]),
    phase=torch.tensor([0.20, 0.49, 0.20, 0.10]),
    touchdown_event=torch.tensor([False, False, False, False]),
    min_snippet_hold_s=0.25,
    max_snippet_hold_s=0.6,
    phase_boundary_tolerance=0.03,
  )

  assert torch.equal(mask, torch.tensor([False, True, True, True]))


def test_event_synchronized_switch_mask_allows_touchdown_triggered_switch():
  mask = event_synchronized_switch_mask(
    active_snippet=torch.tensor([True, True]),
    elapsed_time=torch.tensor([0.30, 0.10]),
    phase=torch.tensor([0.20, 0.20]),
    touchdown_event=torch.tensor([True, True]),
    min_snippet_hold_s=0.25,
    max_snippet_hold_s=0.6,
    phase_boundary_tolerance=0.03,
  )

  assert torch.equal(mask, torch.tensor([True, False]))


def test_wrapped_phase_distance_uses_shortest_distance_on_unit_cycle():
  distance = wrapped_phase_distance(
    torch.tensor([0.95, 0.10, 0.25]),
    torch.tensor([0.05, 0.45, 0.75]),
  )

  assert torch.allclose(distance, torch.tensor([0.10, 0.35, 0.50]), atol=1e-6)


def test_cadence_continuity_score_prefers_close_cadence_matches():
  score = cadence_continuity_score(
    current_cadence=torch.tensor([[1.6], [1.6]]),
    candidate_cadence=torch.tensor([[1.58, 1.10], [0.0, 1.45]]),
    cadence_scale=0.2,
  )

  assert score[0, 0] > score[0, 1]
  assert score[1, 1] > score[1, 0]
  assert score[1, 0].item() == 0.0


def test_reference_swing_lift_score_penalizes_low_swing_lift():
  local_foot_pos_seq = torch.tensor(
    [
      [
        [[0.0, 0.0, 0.00], [0.0, 0.0, 0.00]],
        [[0.0, 0.0, 0.06], [0.0, 0.0, 0.00]],
        [[0.0, 0.0, 0.07], [0.0, 0.0, 0.00]],
      ],
      [
        [[0.0, 0.0, 0.00], [0.0, 0.0, 0.00]],
        [[0.0, 0.0, 0.01], [0.0, 0.0, 0.00]],
        [[0.0, 0.0, 0.02], [0.0, 0.0, 0.00]],
      ],
    ]
  )
  contact_seq = torch.tensor(
    [
      [[True, True], [False, True], [False, True]],
      [[True, True], [False, True], [False, True]],
    ]
  )

  score = reference_swing_lift_score(
    local_foot_pos_seq,
    contact_seq,
    min_swing_lift=0.03,
  )

  assert score[0] > score[1]
  assert torch.allclose(score, torch.tensor([1.0, 0.0]))


def test_straight_reference_quality_score_gates_and_prefers_lifted_low_yaw_snippets():
  twist_cmd = torch.tensor(
    [
      [1.5, 0.0, 0.0],
      [0.2, 0.0, 0.2],
    ]
  )
  mean_vy = torch.zeros(2, 2)
  mean_yaw_rate = torch.tensor(
    [
      [0.02, 0.20],
      [0.02, 0.20],
    ]
  )
  mean_vx = torch.tensor(
    [
      [1.48, 1.48],
      [1.48, 1.48],
    ]
  )
  local_foot_pos_seq = torch.zeros(2, 2, 3, 2, 3)
  local_foot_pos_seq[:, 0, 1:, 0, 2] = 0.06
  local_foot_pos_seq[:, 1, 1:, 0, 2] = 0.01
  contact_seq = torch.ones(2, 2, 3, 2, dtype=torch.bool)
  contact_seq[:, :, 1:, 0] = False

  score = straight_reference_quality_score(
    twist_cmd,
    mean_vx,
    mean_vy,
    mean_yaw_rate,
    local_foot_pos_seq,
    contact_seq,
    min_forward_speed=1.0,
    max_lateral_speed=0.05,
    max_abs_yaw_command=0.08,
    vx_scale=0.25,
    yaw_rate_scale=0.15,
    min_swing_lift=0.03,
    lift_weight=0.65,
    yaw_weight=0.35,
  )

  assert score[0, 0] > score[0, 1]
  assert torch.equal(score[1], torch.zeros_like(score[1]))


def test_turn_intent_mask_matches_low_speed_yaw_commands_only():
  mask = turn_intent_mask(
    torch.tensor(
      [
        [0.05, 0.0, 0.4],
        [0.30, 0.0, 0.4],
        [0.05, 0.0, 0.05],
      ]
    ),
    min_turn_yaw_command=0.2,
    max_turn_forward_speed=0.25,
    max_turn_lateral_speed=0.05,
  )

  assert torch.equal(mask, torch.tensor([True, False, False]))


def test_snippet_source_prefix_mask_maps_clip_prefixes_to_snippet_mask():
  mask = snippet_source_prefix_mask(
    source_clip_names=(
      "walk_forward_normal_00",
      "turn_in_place_v2_left_00",
      "turn_left_90_00",
    ),
    source_clip_id=torch.tensor([0, 1, 1, 2]),
    prefix="turn_in_place_v2_",
  )

  assert torch.equal(mask, torch.tensor([False, True, True, False]))


def test_prioritize_turn_in_place_candidates_masks_non_v2_snippets_for_turn_intent():
  task_score = torch.tensor(
    [
      [0.9, 0.8, 0.7],
      [0.9, 0.8, 0.7],
    ]
  )
  prioritized = prioritize_turn_in_place_candidates(
    task_score,
    turn_in_place_intent=torch.tensor([True, False]),
    turn_in_place_v2_mask=torch.tensor([False, True, False]),
  )

  assert torch.isneginf(prioritized[0, 0])
  assert prioritized[0, 1] == task_score[0, 1]
  assert torch.isneginf(prioritized[0, 2])
  assert torch.equal(prioritized[1], task_score[1])


def test_prioritize_prefixed_candidates_masks_non_pivot_snippets_for_turn_intent():
  task_score = torch.tensor(
    [
      [0.9, 0.8, 0.7],
      [0.9, 0.8, 0.7],
    ]
  )
  prioritized = prioritize_prefixed_candidates(
    task_score,
    turn_intent=torch.tensor([True, False]),
    preferred_mask=torch.tensor([True, False, True]),
  )

  assert prioritized[0, 0] == task_score[0, 0]
  assert torch.isneginf(prioritized[0, 1])
  assert prioritized[0, 2] == task_score[0, 2]
  assert torch.equal(prioritized[1], task_score[1])


def test_turn_in_place_v2_bias_score_prefers_v2_snippets_for_low_speed_yaw_commands():
  bias = turn_in_place_v2_bias_score(
    twist_cmd=torch.tensor(
      [
        [0.05, 0.0, 0.4],
        [0.5, 0.0, 0.4],
      ]
    ),
    is_turn_in_place_v2=torch.tensor(
      [
        [True, False, False],
        [True, False, False],
      ]
    ),
    min_turn_yaw_command=0.2,
    max_turn_forward_speed=0.25,
    max_turn_lateral_speed=0.05,
    nonmatch_penalty=0.5,
  )

  assert bias[0, 0] > 0.0
  assert bias[0, 1] < 0.0
  assert bias[0, 2] < 0.0
  assert torch.equal(bias[1], torch.zeros_like(bias[1]))


def test_pivot_bias_score_prefers_pivot_snippets_for_low_speed_yaw_commands():
  bias = pivot_bias_score(
    twist_cmd=torch.tensor(
      [
        [0.05, 0.0, 0.35],
        [0.5, 0.0, 0.35],
      ]
    ),
    is_pivot=torch.tensor(
      [
        [True, False, True],
        [True, False, True],
      ]
    ),
    min_turn_yaw_command=0.2,
    max_turn_forward_speed=0.25,
    max_turn_lateral_speed=0.05,
    nonmatch_penalty=0.5,
  )

  assert bias[0, 0] > 0.0
  assert bias[0, 1] < 0.0
  assert bias[0, 2] > 0.0
  assert torch.equal(bias[1], torch.zeros_like(bias[1]))


def test_cadence_bias_mask_applies_only_to_same_gait_non_turn_commands():
  mask = cadence_bias_mask(
    twist_cmd=torch.tensor(
      [
        [0.6, 0.0, 0.0],
        [0.05, 0.0, 0.4],
      ]
    ),
    current_gait_id=torch.tensor([[0], [0]]),
    candidate_gait_id=torch.tensor([[0, 1, 2], [0, 0, 2]]),
    turn_gait_id=2,
    min_turn_yaw_command=0.2,
    max_turn_forward_speed=0.25,
    max_turn_lateral_speed=0.05,
  )

  assert torch.equal(
    mask,
    torch.tensor(
      [
        [True, False, False],
        [False, False, False],
      ]
    ),
  )
