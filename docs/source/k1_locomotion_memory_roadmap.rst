.. _k1-locomotion-memory-roadmap:

Booster K1 Locomotion-Memory Research Roadmap
=============================================

Snapshot date: April 15, 2026

This page describes the research roadmap for the Booster K1
locomotion-memory project. The operational restart guide is in
:doc:`k1_locomotion_memory_handoff`; the chronological implementation status
is in :doc:`k1_locomotion_memory_status`.

Final target
------------

The target outcome is a Booster K1 policy that can be evaluated as a research
result, not only as a task-specific controller:

- it follows commanded forward velocity and yaw velocity;
- it walks naturally at slow, normal, and fast speeds;
- it turns in place by stepping with visible foot lift;
- it supports smooth transitions between standing, walking, arc walking, and
  turning;
- it uses human-like retargeted motion as a reusable structured memory;
- it remains robust enough that small friction or terrain changes do not break
  the learned behavior.

Method invariants
-----------------

These design constraints should not be changed casually:

- The task is velocity-commanded locomotion, not hard reference tracking.
- The memory proposal is a structured suggestion and should remain optional
  enough for the policy to adapt.
- Retrieval should stay feasibility-aware: command fit alone is not enough.
- The current primary robot is Booster K1.
- The current primary terrain is flat ground until the core gait is reliable.
- Reward tuning must not sacrifice command tracking just to make foot motion
  look cleaner.

Current phase
-------------

The project is between the first usable K1 locomotion-memory stack and the
first publishable-quality gait.

Completed foundations:

- K1 asset integration and task registration.
- K1 motion conversion from retargeted CSVs.
- Structured locomotion memory with snippet metadata.
- Proposal-conditioned K1 locomotion-memory task.
- Viser play workflow.
- W&B training workflow.
- Fixed-command FK evaluation.
- First high-speed steady-forward memory expansion.

Current open control problems:

- straight-command yaw drift at higher forward speeds;
- return-swing scuff during forward walking;
- weak right-side low-yaw in-place turning;
- insufficient foot lift during some in-place turns;
- occasional cadence irregularity;
- remaining visual unnaturalness around hip yaw, knee usage, and foot lanes.

Milestone 1: Train and evaluate v4
----------------------------------

Goal:

Train the currently implemented ``tracking_balance_v4`` code and determine
whether straight-reference retrieval bias plus yaw-specific rewards solve the
main v3 regressions.

Command:

.. code-block:: bash

   uv run train Mjlab-LocomotionMemory-Flat-Booster-K1 \
     --agent.logger wandb \
     --agent.upload-model False \
     --env.scene.num-envs 4096 \
     --agent.run-name k1_locomotion_memory_tracking_balance_v4

Acceptance criteria:

- ``forward.vx_abs_error_mean`` stays close to or below the v3 value of about
  ``0.082``.
- ``forward_vx_1.5`` executed speed remains at least around ``1.35 m/s``.
- ``forward.yaw_abs_error_mean`` improves from the v3 regression.
- ``forward.scuff_frac_mean`` improves from the v3 regression.
- ``yaw_r_0.10`` and ``yaw_r_0.15`` move in the commanded direction instead of
  remaining near zero.
- ``Train/mean_reward`` and fall rate do not regress enough to indicate that
  auxiliary gait shaping has suppressed locomotion.

Decision:

- If v4 improves yaw drift and low-yaw turning while preserving forward speed,
  keep it as the new baseline.
- If v4 improves appearance but hurts tracking, reduce auxiliary weights before
  adding new shaping terms.
- If v4 does not change behavior, inspect retrieval statistics and memory
  selection before changing rewards.

Milestone 2: Separate memory quality from execution quality
-----------------------------------------------------------

Goal:

Avoid tuning rewards blindly. For each bad behavior, decide whether the memory
reference is bad, the retriever selected the wrong snippet, or the execution
policy failed to follow a good proposal.

Required checks:

- Compare executed FK foot paths against ``reference_swing_foot_position``.
- Inspect selected snippet labels and command fit for each fixed command.
- Check whether high-speed straight commands retrieve straight high-speed
  snippets or curved/transition snippets.
- Check whether low-yaw turn commands retrieve pivot or turn-in-place snippets.
- Compare left and right low-yaw turn outcomes separately.

Expected outcome:

Each failure should be assigned to one of three buckets:

- memory corpus problem: add or recut K1 motions;
- retrieval problem: adjust scoring, gates, or snippet eligibility;
- execution problem: adjust observations, residual limits, reward terms, or
  curriculum.

Milestone 3: Improve forward naturalness without losing speed
-------------------------------------------------------------

Goal:

Fix the forward-walking scuff and foot-return issue while preserving the
high-speed improvement from the steady-forward memory update.

Likely changes:

- Prefer high-speed straight snippets with better swing clearance and low yaw
  rate.
- Penalize return-swing scuff only when commanded speed is already being met.
- Shape the foot return path so the foot comes closer to the hip after push-off
  instead of dragging forward from behind the body.
- Keep foot-lane shaping active for straight commands, but avoid overconstraining
  arc walking and turning.

Acceptance criteria:

- ``vx=1.5`` still executes near the v3/v4 speed target.
- Forward scuff metrics improve.
- Viser no longer shows repeated toe catches during return swing.
- Foot lanes remain visually straight during straight walking.

Milestone 4: Make low-yaw turning real stepping
-----------------------------------------------

Goal:

For small in-place yaw commands, K1 should step with visible foot lift instead
of twisting or sliding in place.

Likely changes:

- Ensure small-yaw commands retrieve ``pivot_*`` snippets instead of low-speed
  walk snippets.
- Add command-range-specific metrics for ``abs(wz)`` around ``0.10`` to
  ``0.20``.
- Add retrieval eligibility or score bonuses for pivot clips under near-zero
  forward speed.
- Add symmetric left/right diagnostics because right low-yaw turning has been
  weaker than left.
- If execution follows flat-foot references, improve or recut the pivot motion
  corpus rather than forcing lift with rewards alone.

Acceptance criteria:

- ``yaw_l_0.10`` and ``yaw_r_0.10`` both produce visible stepping.
- ``yaw_l_0.15`` and ``yaw_r_0.15`` track the commanded sign and magnitude
  better than v3.
- Foot lift is visible in Viser without making the stance unnaturally wide or
  jerky.
- Behavior remains stable under small friction changes in later robustness
  tests.

Milestone 5: Formal baselines and ablations
-------------------------------------------

Goal:

Turn the engineering result into a research result by comparing the method
against clear alternatives.

Minimum baselines:

- command-only K1 velocity policy with no memory;
- hard reference tracking policy using converted K1 motions;
- retrieval-only or low-residual hard-proposal policy;
- locomotion-memory without feasibility terms;
- locomotion-memory without proposal observations;
- locomotion-memory with command-fit-only retrieval.

Evaluation axes:

- task success: command tracking, fall rate, episode length;
- naturalness: foot lanes, cadence, scuff, angular momentum, joint usage;
- executability: collisions, joint limits, action smoothness, contact forces;
- memory validity: command fit, phase/contact validity, switch count, transition
  quality.

Milestone 6: Broaden the motion vocabulary
------------------------------------------

Goal:

After straight walking and in-place turning are reliable, expand expressivity.

Candidate additions:

- lateral stepping;
- backward walking;
- starts and stops;
- walk-run transitions;
- broader speed and style variants;
- rough-terrain transfer.

Gate:

Do not broaden the memory pool until the retriever reliably selects the right
kind of snippet for the current command. More motions can make behavior worse
if retrieval is not selective enough.

Evaluation protocol
-------------------

Every serious run should produce:

- a W&B run name with the intended intervention in the name;
- a final checkpoint;
- a ``research`` preset FK evaluation JSON;
- a short Viser qualitative note;
- an update to :doc:`k1_locomotion_memory_status`.

Recommended comparison table fields:

- run name;
- memory version;
- key code change;
- ``Train/mean_reward``;
- ``Episode_Termination/fell_over``;
- ``forward.vx_abs_error_mean``;
- ``forward.yaw_abs_error_mean``;
- ``forward.scuff_frac_mean``;
- ``yaw_l_0.10`` actual yaw;
- ``yaw_r_0.10`` actual yaw;
- qualitative result.

Research stop criteria
----------------------

The current line of work is ready to transition from engineering iteration to
research write-up when:

- velocity tracking is competitive with a command-only baseline;
- straight gait looks natural at least up to ``vx = 1.5``;
- low-yaw and high-yaw in-place turns use actual stepping;
- the memory method beats at least one non-memory baseline on naturalness or
  robustness without losing task success;
- the evaluation protocol is reproducible from a fresh checkout using this
  documentation.
