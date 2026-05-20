# K1 Kick Policy Training History

LVDRS-style kick-only training for Booster K1. Branch: `feat/humanoid-soccer`.

## Scope (V1)

- Ball spawns near the foot, in front cone (±π/4).
- Robot must kick the ball in any target direction (left/right via mirror symmetry).
- Sim2real-ready (virtual perception, head FOV alignment, sim2real regularizers).
- Post-kick: brief settling step, then return to upright HOME pose and stand
  still (clean handoff state for a separate locomotion policy).

## Task ID

- `Mjlab-Kick-Flat-Booster-K1` — baseline PPO + mirror symmetry.
- `Mjlab-Kick-AMP-Flat-Booster-K1` — adds AMP discriminator (V2 work,
  currently underperforming due to discriminator saturation).

## File map

- `src/mjlab/tasks/kick/kick_env_cfg.py` — env config, reward weights.
- `src/mjlab/tasks/kick/mdp/commands.py` — `KickTargetCommand`, virtual
  perception, kick contact tracking, debug visualization.
- `src/mjlab/tasks/kick/mdp/rewards.py` — task-specific reward functions.
- `src/mjlab/tasks/kick/mdp/terminations.py` — termination conditions.
- `src/mjlab/tasks/kick/mdp/perception.py` — `VirtualPerception` (FOV,
  noise, latency, update freq for sim2real).
- `src/mjlab/tasks/kick/config/booster_k1/env_cfgs.py` — K1 wiring
  (camera offsets, action scales, ball spawn).
- `src/mjlab/tasks/kick/config/booster_k1/symmetry.py` — K1 joint mirror
  table + `data_augmentation` for rsl-rl symmetry loss.
- `src/mjlab/tasks/kick/config/booster_k1/rl_cfg.py` — PPO + AMP configs.
- `src/mjlab/rl/amp/` — AMP discriminator, WGAN loss, motion buffer.
- `src/mjlab/tasks/kick/rl/runner.py` — `KickAMPRunner` (PPO + AMP).

## Reward composition (current V1.17)

Goal-related:
- `alive` (+5), `terminated` (-200), `ball_approach` (+50, potential),
  `ball_proximity` (+5, continuous), `target_progress` (+500, potential),
  `kick_success` (+3, one-shot).

Auxiliary (style + sim2real):
- `sideways_kick_aligned` (+20), `forward_kick_penalty` (-20),
  `support_foot_proximity` (+15, at kick),
  `post_kick_motion_penalty` (-5, after 0.3s grace),
  `post_kick_homing` (+10, after 0.3s grace),
  `foot_proximity` (-5),
  `pelvis_orientation` (-1),
  `head_yaw_alignment` (-2), `head_pitch_alignment` (-2),
  `arm_posture` (+0.05),
  `joint_vel_l2` (-1e-4), `joint_acc_l2` (-1e-7),
  `action_rate_l2` (-3),
  `base_yaw_rate_l2` (-0.5), `base_lin_vel_z_l2` (-2),
  `non_foot_collision` (-100), `joint_limit` (-100).

Terminations:
- `time_out`, `robot_fell_height`, `robot_fell_orientation`,
  `ball_out_of_range`. (`kick_completed` was removed in V1.15.)

## Iteration history

The 2 critical infrastructure bugs found and fixed (applied to all V1.x
from V1.10 onward) — without these, **every** earlier run was training
from a 50 cm mid-air drop + zero-joint pose:

1. `KickTargetCommand._resample_command` now `write_root_state_to_sim`
   to `env_origins + (0, 0, robot_spawn_height)` so the robot lands at
   the K1 HOME_KEYFRAME `pos=(0,0,0.54)` every reset. (K1 MJCF worldbody
   default is `pos="0 0 1.0"`.)
2. `write_joint_state_to_sim(default_jp, default_jv)` so all 22 joints
   match HOME_KEYFRAME on reset (otherwise all zero → arms straight,
   knees locked, ankles flat).

### V1.0 — initial scratch
Single-critic PPO + mirror symmetry + virtual perception. All weights
copied from LVDRS paper. Quickly converged to self-destructive "suicide
kick" local optimum (kick once then fall).

### V1.1 — anti-self-destruct
Tightened `action_rate_l2` (-3), softened tilt threshold, restored alive
weight. Broke the suicide-kick pattern; policy started living longer
but stopped kicking ("standing still 47% of close-ball spawns").

### V1.2 — added `ball_proximity_continuous`
Continuous near-ball reward to give a constant pull when stationary.
Episode 8 s → 12 s.

### V1.3 — close-spawn curriculum
`ball_spawn_distance_range` 0.05–1.5 m → 0.05–0.5 m. Kick rate jumped
0.37 → 0.47 instantly (eval artifact: tighter task distribution), then
plateaued at 0.47 for thousands of iters.

### V1.4 — front-cone hypothesis (failed)
Restricted `ball_spawn_angle_range` to front cone + bumped
`kick_success` to 15. Hypothesis was that 53% of failures were
back-spawned balls. ksr dropped -1.8σ → hypothesis wrong, reverted.

### V1.5 — target_progress saturation removal (failed)
Cut `target_progress` weight 500 → 100 to "remove saturation". Caused
total behavioral collapse: ksr 0.47 → 0.00, peak_kick_speed 4.97 → 2.97,
policy stood still beside the ball until timeout. Reverted.
Lesson: `target_progress @ 500` is the load-bearing gradient for the
kick. Don't touch.

### V1.6 / V2 — AMP first attempt (no breakthrough)
Built `src/mjlab/rl/amp/` (WGAN discriminator + tanh + gradient
penalty + motion buffer + state extractor + custom runner). Reference
data = 10 K1-retargeted kick clips. AMP state = `(joint_pos, joint_vel)`
(44 D). Discriminator saturated at -1.985 by iter ~25k (no walking
clips means it just learns "policy frames look unlike kick frames" and
emits a constant style penalty). No effect on ksr.

### V1.7 — first sim2real-honest config (still pre-fix)
- Front-cone spawn (±π/4) so the ball is always inside the camera FOV
  (~±52° hor, ~±47° ver) when the head aligns.
- `kick_completed` termination after 10 steps post-strike.
- `VirtualPerceptionCfg.hold_last_on_miss = False` — out-of-FOV ball
  must give zero observation (no stale-position cheating).
- Head `yaw_alignment` / `pitch_alignment` rewards (-0.5 each).
- Episode 12 s.

### V1.8 — sim2real regularizers (broke V1.7)
Added `arm_posture` (+0.3), `joint_vel_l2`, `joint_acc_l2`. Too much at
once: ksr 0.85 → 0.27 in 900 iters. Reverted to V1.7.

### V1.9 — restart from scratch + anti-spin
`base_yaw_rate_l2` (-0.5) to deter the "rotation kick" pattern. Reduced
`arm_posture` to +0.05. Scratch start (no warm-start) — but still
landed in mid-air every episode (the 2 init bugs above were undiagnosed).

### V1.10 — fix mid-air spawn + fix zero joints
Discovered `write_root_state_to_sim` and `write_joint_state_to_sim`
were never called on per-episode reset. Robot was spawning at
`Trunk pos="0 0 1.0"` (K1 MJCF default), then falling 50 cm with all
joints at 0. After fix: trunk at 0.55, feet at 0.05, HOME joints
applied. **Every V1.x policy before this was training on a fall-recover-
then-kick task.** Standing emerged cleanly at iter 2500 (eplen 317,
robot_fell 1.8/iter).

### V1.11 — ball-spawn-between-feet fix + multi-feature shaping (regressed)
- Ball was spawning between the feet (≤ 0.18 m forward = inside K1
  stance). Moved range to 0.20–0.40 m, in front of foot tips.
- Arm action scale 0.44 → 0.10 (4× tighter); `arm_posture` 0.05 → 0.15.
- `kick_completed` hold 10 → 4 to shorten chase phase.
- Forward-biased `target_dir` sampling (60% front, 30% side, 10% rear).
- `support_foot_proximity_at_kick` (+15) for natural plant-and-strike.
- Head alignment weights -0.5 → -2.0 (4×).
Result: ksr 0.14 → 0.09 (regression). Too many simultaneous shifts.

### V1.12 — anti-float
Added `base_lin_vel_z_l2` (-2). Viser playback showed body lifting
during kick. Tested as warm-start; ksr still 0.04 at iter 5100.

### V1.13 — 5 s episode cap (worse)
Cut episode 12 s → 5 s to force quick kicks. Mostly time-outs (15/iter,
ksr 0.04). User identified the warm-start checkpoint distribution as
the root cause: every V1.11+ warm-start was carrying the "kick-while-
floating" mode from V1.10.

### V1.14 — full scratch with all V1.11–V1.13 features
Scratch start with all rewards in place. At iter 3000: ksr 0.14 again
(plateau-level), mean reward +49, eplen 317, robot_fell 1.8.
Visually OK but not yet kicking reliably.

### V1.15 — replace `kick_completed` with `post_kick_stillness`
Removed `kick_completed` termination; added per-step bonus
`exp(-base_vel² / 0.3²) · exp(-joint_vel² / 2²)` for post-kick stillness
(weight +10). Warm-started from V1.14 iter 3000. **kick_success_rate
jumped to 0.91 at iter 4468** — the kick task is solved. But viser
showed the policy chasing the ball post-strike; `post_kick_stillness`
reward was 0.0004 / ep — exp(-vel²) saturated to ~0 at running speeds,
so no useful gradient.

### V1.16 — linear motion penalty
Replaced the exp-shape stillness with `post_kick_motion_penalty` =
`base_vel² + 0.05 · joint_vel²` (weight -5). Linear in velocity² so
gradient remains effective even at 2 m/s. ksr 0.92 maintained.

### V1.17 — phase-aware homing (exp form, saturated)
User requested: kick → 1 settling step → upright HOME standstill,
suitable for handoff to a separate locomotion policy.
- `post_kick_motion_penalty` now gates on `steps_since_kick ≥ 15`
  (0.3 s grace window so the settling step is unpenalized).
- New `post_kick_homing` reward (+10): for `steps_since_kick ≥ 15`,
  reward `exp(-mean(joint_pos − default_joint_pos)² / 0.25²) ·
  exp(-base_vel² / 0.2²)` — pulls posture back to HOME_KEYFRAME and
  velocity to 0.
- Warm-started from V1.16 iter 8000.
At iter 9591: ksr 0.994, but `post_kick_homing` ≈ +0.0018 — the exp
shape saturated to ~0 at typical post-kick joint deviation, leaving
no gradient. Same failure mode as V1.15's exp-shaped stillness reward.

### V1.18 — linear homing penalty
- Replaced `post_kick_homing` (exp +10) with `post_kick_homing_penalty`
  (linear -10): `mean((joint_pos − default_joint_pos)^2)` for
  `steps_since_kick ≥ 15`. Linear → gradient stays useful at any
  deviation magnitude.
- Bumped `post_kick_motion_penalty` weight -5 → -10 to keep it on the
  same order as the homing penalty.
- Warm-started from V1.17 iter 9500 (kick skill preserved, ksr 1.00 at
  iter 1).
By iter 11000 the policy was reliably standing still after kick (user
confirmation). post_kick_motion_penalty -0.80 → -0.39, homing_penalty
-0.15 → -0.05.

### V1.19 — arm folding (target was HOME default — still raised)
User observed arms in T-pose during the whole episode. `arm_posture`
(exp +0.15) was paying ~0 at T-pose (`exp(-(π/4)² / 0.3²) ≈ 0.001`),
so the policy had no incentive to fold them.
- New `arm_deviation_l2` (-5): linear per-step penalty
  `mean((arm_jp − default_arm_jp)^2)`. Applies every step (not gated
  on post-kick) so arms stay folded throughout approach + kick + stand.
- Kept the small `arm_posture` exp bonus too.
- Warm-started from V1.18 iter 11000.
By iter 12000 arms had converged to HOME default but the user still
called this "raised" — K1 HOME has `Shoulder_Pitch=0.3` (17° forward)
and `Elbow_Pitch=0.5` (29° bent), which is not "arms down".

### V1.20 — arms-down target + DR expansion (initial pose still raised)
- `arm_deviation_l2` target changed from HOME default to an explicit
  arms-down pose: shoulder pitch/roll = 0, elbow_pitch = 0.2, others = 0.
  Hard-coded in `_ARM_DOWN_TARGETS_RAD` so the task doesn't depend on
  the shared K1 HOME_KEYFRAME (which other tasks rely on).
- Diagnosis caught: the default `asset_cfg` on the reward function was
  not being resolved by the reward manager → empty joint_ids → constant
  zero penalty. Fix: pass `asset_cfg` explicitly in the env cfg.
- Sim2real DR strengthened:
  - `foot_friction` range 0.3-1.2 → 0.6-1.2 (user reported visible foot
    sliding; 0.3 was too slippery for any realistic indoor surface).
  - New `trunk_mass`: ±15% mass scaling.
  - New `joint_damping`: ±20% damping scaling.
- Warm-started from V1.19 iter 12000 (kick skill ksr 0.99 preserved).
At iter 13500 the policy still played with raised arms. Diagnosis: K1
HOME_KEYFRAME has `Shoulder_Pitch = +17°` and `Elbow_Pitch = +29°`, so
every reset starts the arms raised; the per-step `arm_deviation_l2` at
weight -5 wasn't strong enough to pull them down within the episode.

### V1.21 — arms-down reset + stronger penalty (T-pose still emerged)
- `KickTargetCommand._resample_command` now overrides arm joints to the
  arms-down target (`Shoulder_Pitch=0`, `Elbow_Pitch=0.2`,
  `Shoulder_Roll=0`, `Elbow_Yaw=0`) on every episode reset. HOME-K1's
  raised-arm pose is no longer the starting state.
- `arm_deviation_l2` weight -5 → -15 to keep the policy from drifting
  back up across the episode.
- Warm-started from V1.20 iter 13500 (kick skill preserved, ksr 0.99
  at iter 1).
At iter 16500 the viser playback showed the shoulders rolled out to
~90° (T-pose) despite the arms-down reset; per-step `arm_deviation_l2`
at weight -15 was still being out-competed by the target_progress and
alive rewards.

### V1.22 — heavy + focused T-pose suppression
- `arm_deviation_l2` weight -15 → -100 (6.7×).
- New `shoulder_roll_l2` (weight -50): focused L2 penalty on the two
  shoulder roll joints, which are the single failure-mode source for
  T-pose (`shoulder_roll → ±π/2`). Diagnosed the same default-asset_cfg
  bug as V1.20 — fixed by passing `asset_cfg` explicitly in env cfg.
- Arm action scale halved: shoulder/elbow pitch & yaw 0.10 → 0.05,
  shoulder roll 0.10 → 0.03 (tighter on the T-pose joint).
- Warm-started from V1.21 iter 16500.

### V1.23 — sim2real DR expansion (perception + robot)
**Completed at iter 21000.** Final: ksr 0.987, peak_kick_speed 6.14 m/s,
shoulder_roll_l2 -0.23 (still some deviation residual), ball_out_of_range
41/iter (strong kicks). Checkpoint: `2026-05-20_04-00-26/model_20999.pt`.

Detailed setup:
User requested stricter sim2real domain randomization, especially on
the virtual perception side. The LVDRS paper uses fixed empirical
perception parameters; we add per-env variation so the policy sees a
heterogeneous "camera fleet" instead of one nominal sensor.

Perception DR (per-env at episode reset):
- `noise_a_range` multiplier ×[0.7, 1.5] → effective slope 0.087–0.186.
- `noise_b_range` multiplier ×[0.7, 1.5] → effective offset 0.104–0.224.
- `detection_prob_in_fov_range` absolute [0.70, 0.95].
- `fov_scale_range` multiplier ×[0.85, 1.0] → 89–105° / 80–94° FOV.
- `latency_mean_range` [80, 160] ms (was 116 ± 18 ms fixed).
- `update_hz_mean_range` [20, 30] Hz (was 25.36 ± 1.06 Hz fixed).

The base values in `VirtualPerceptionCfg` are interpreted as the
nominal centers; the ranges multiply / replace them per env. Verified
by sampling 8 envs and confirming distinct values for noise, detection
probability, FOV, latency.

Robot-side DR:
- `push_robot`: interval 2–4 s → 1.5–3.0 s, velocity range ~1.5× wider.
- New `all_body_mass`: ±10% mass scaling on every body (not just Trunk).
- New `joint_friction`: 0–0.05 (sim friction has a wide tolerance).
- New `joint_armature`: ±10% scaling.

Warm-started from V1.22 iter 16500.

### V1.24 (current) — fix the action-scale-vs-offset trap
**Root cause for "arms still up" at V1.23 iter 21000:** the
`JointPositionActionCfg` uses `use_default_offset=True`, so the PD
target is `default_joint_pos + action * scale`. K1 HOME has
`Shoulder_Pitch = +0.3` and V1.22's `arm action scale = 0.05` meant
the MINIMUM reachable target was `0.3 + (-1) * 0.05 = 0.25 rad ≈ 14°`.
No matter how negative the policy commanded, the joint could only
swing down to 14°. My arms-down reset at the start of each episode
was being overwritten by the PD controller within a single step.
- Action scales restored to allow reaching arms-down:
  - Shoulder pitch 0.05 → 0.5 (now reaches `0.3 − 0.5 = −0.2 rad`)
  - Elbow pitch 0.05 → 0.5
  - Elbow yaw 0.05 → 0.3
  - Shoulder roll 0.03 → 0.05 (still narrow to prevent T-pose)
- `arm_deviation_l2 -100` and `shoulder_roll_l2 -50` kept; they now
  have effective room to actually pull the arms down.
- Warm-started from V1.23 iter 20999.
Initial iters show large -23 `arm_deviation_l2` and ksr drop 0.99 → 0.3
because the policy's pre-existing action outputs (tuned for scale 0.05)
now produce 10× joint motion. Re-converged within 500 iter (ksr 0.99,
arm_deviation_l2 -0.74). User still reported T-pose.

### V1.25 (current) — CORRECT arms-down target for K1
**Root cause for persistent T-pose:** K1's mesh has the upper arm
extending laterally at `Shoulder_Roll = 0` — so `Shoulder_Roll = 0` IS
the T-pose. My V1.20–V1.24 `arm_deviation_l2` target had
`Shoulder_Roll = 0` and the dedicated `shoulder_roll_l2` penalized
`shoulder_roll²`. Both rewards together TAUGHT the policy to keep
shoulders rolled to 0 = T-pose. I had the geometry inverted.

Diagnosis via grid-search over `(Shoulder_Pitch, Shoulder_Roll,
Elbow_Pitch)`:
- `Shoulder_Pitch = 0, Shoulder_Roll = -1.0, Elbow_Pitch = 1.5` (LEFT)
- mirror for RIGHT (`Shoulder_Roll = +1.0`)
- → `hand_z ≈ 0.10 m` (near ground, true arms-down)

Fixes:
- `_ARM_DOWN_RESET_TARGETS_RAD` updated to the correct values.
- `_ARM_DOWN_TARGETS_RAD` (for `arm_deviation_l2`) updated the same way.
- `shoulder_roll_l2` reward REMOVED — it was enforcing T-pose by
  penalizing exactly the joint values we now want.
- Warm-started from V1.24 iter 21500. `arm_deviation_l2` immediately
  jumped to -2.5 (policy at T-pose is now penalty-dominant) and ksr
  dropped 0.99 → 0.21. Policy needs to RE-LEARN arm position from
  this state. May take 2000–4000 iter; if it stalls, scratch.

**V1.25 outcome (iter 21500 → 23700, 2026-05-20):** ksr recovered to
0.99 within 200 iter, `arm_deviation_l2` improved from -2.5 to -0.97
and plateaued at that level from iter 23138 onward (six consecutive
500-iter milestones all -0.96…-0.98 — clear plateau, not still
descending). The remaining residual is the policy still raising the
arm slightly mid-kick swing before bringing it back to home; refining
that further would need separate phase-aware shaping, not more
training time on the existing reward stack. Stopping V1.25 here and
moving direction accuracy forward instead.

### V1.26 — sharp post-kick direction alignment
**Motivation:** `scripts/tools/eval_kick_direction.py --heading-sweep`
against V1.25 showed 49.6° mean direction error in the single-front
test and 21–28% success-rate across the 9-direction sweep with a
bimodal distribution. `target_progress_potential` (weight 500) gives
*any* cos>0 ball motion the same gradient signal, so the policy banks
partial credit for any roughly-forward kick — there is no reward
gradient between a 5° kick and a 30° kick.

Fix: added `ball_velocity_alignment_post_kick` (rewards.py L141):

```
exp(-angle² / std_rad²) × ball_speed × (steps_since_kick ∈ [0, 10))
```

with `std_rad = 0.3` (≈17°). Kicks within ±17° get near-full reward;
kicks beyond ~35° get near zero. Active only for the 10 steps
(≈200 ms at dt=0.02) immediately after the contact event, so
`target_progress` still owns the approach phase and this reward owns
post-contact accuracy.

Weight 5.0: per-episode contribution ≈ 75–120 (raw integral ~15–24 if
well-aligned at ~3 m/s post-kick speed), which is 5–7× the current
`target_progress` episode contribution of 16.6 but does not dwarf
the rest of the reward stack. Conservative start — V1.27 can push
to 10 or 20 if direction error stays above 20°.

Plan: warm-start from V1.25 latest checkpoint once V1.25 stops
(iter ~25k or natural end), run 2000–4000 iters, then re-run the
9-direction eval sweep. Watch for ksr regression (this reward is
contact-conditioned, not approach-conditioned, so it shouldn't
distort approach behavior, but watch anyway).

**V1.26 outcome (iter 23500 → 24516, 1016 iters, std_rad=0.3, weight 5.0):**
- Direction reward gained only ~5% (1.07 → 1.12). Computing the implied
  angle from raw reward magnitude: avg post-kick angle ~37° (raw per
  step ≈ 0.028 = exp(-(0.65)²/0.09) × 3 m/s ≈ 0.0093 × 3).
- At std_rad=0.3 the gradient at 37° is vanishing — `exp(-(0.65)²/0.09)`
  is essentially zero, so the policy has almost no signal pulling it
  toward zero angle.
- No regression: ksr stayed 0.99, arm_deviation drifted briefly to -1.11
  then recovered to -0.95 (noise, not a trend), `target_progress` and
  `post_kick_motion` stayed within their V1.25 noise bands.
- Conclusion: the reward shape is too peaked for the policy's current
  state. Need wider std to give gradient at the operating angle.

### V1.27 — widen std_rad 0.3 → 0.5 for gradient at large angles
At std_rad=0.5, the per-step reward at the current ~37° angle is
`exp(-(0.65)²/0.25) ≈ 0.186 × 3 m/s ≈ 0.56`, vs `0.028` at std_rad=0.3
— **20× more gradient** at the policy's operating angle. The peak
signal at angle=0 is unchanged at `1.0 × speed` in both, so this just
flattens the curve away from peak, not the peak itself.

Weight stays at 5.0 (single-variable change vs V1.26). Warm-start from
V1.26 `model_24500.pt`. Plan: 2000 iters, then re-eval. If mean angle
drops below 0.5 rad (≈29°), narrow std back to 0.3 in V1.28 for sharp
peak refinement.

**V1.27 outcome (iter 24500 → 25571, 1071 iters, std_rad=0.5, w=5.0):**
- Immediate +18% reward bump at iter 24500 (1.07 → 1.27) from the shape
  change alone — no policy update yet, just the same angle scoring
  higher under wider std. Confirms the wider gradient was missing.
- But then *flat* at 1.25–1.27 over the next 945 iters. No policy
  learning despite the better gradient.
- Other metrics: ksr steady 0.99, `peak_kick_speed` 6.5 m/s, no
  regression elsewhere. So the reward is firing but isn't strong
  enough to redirect the gradient.
- Diagnosis: at w=5 the episode contribution (~1.26) is only ~7.6% of
  `target_progress` (16.5). PPO's gradient on direction is dominated
  by the larger reward, which gives positive signal for *any* cos>0
  ball motion. The new sharper reward is being drowned out.

### V1.28 — bump weight 5 → 15 (3× incremental)
Single change vs V1.27: weight from 5 → 15. New episode contribution
~3.78, about 23% of `target_progress` — still smaller but now large
enough to meaningfully tilt the policy gradient.

Kept ≤ 3× change to stay inside the "never ≥5× weight change on warm-
started policy" rule from V1.5/V1.8 lessons. Warm-start from V1.27
`model_25500.pt`. Plan: 2000 iters, then re-eval. If still flat,
the issue is structural (kick foot determines deflection angle, not
swing direction) — V1.29 will explore body-yaw-alignment-before-kick
rather than more reward shaping.

**V1.28 outcome (iter 25500 → 25948, 448 iters, std_rad=0.5, w=15):**
- Direction reward instantly tripled (1.27 → 3.85) from the weight bump
  alone — pure shape effect, no learning.
- Then flat at 3.85 for the next 448 iters, but with cost: arm
  `-1.0 → -1.11`, `target_progress` `16.33 → 16.23`,
  `post_kick_motion_penalty` `-0.70 → -0.80`. Other rewards regressing
  while direction stayed put = cost without benefit.
- Stopped at iter 25948 to run an eval and check ground truth.

**Direction eval on V1.28 `model_26000.pt` (5-heading sweep, 4 ep × 64 envs):**

| heading | mean err | median | success% | speed (m/s) |
|---------|----------|--------|----------|-------------|
| -90°    | 41.6°    | +0.8°  | 50%      | 10.5        |
| -45°    | 68.0°    | +53.7° | 0%       | 10.9        |
| +0°     | 47.0°    | +28.8° | 25%      | 10.3        |
| +45°    | **5.2°** | +2.3°  | **75%**  | 10.5        |
| +90°    | 77.8°    | +65.2° | 0%       | 10.6        |

Mean across headings: 47.9°, 30% success. Two diagnostic findings:
1. Severe L/R asymmetry — +45° has 5° error while -45° has 68° error,
   despite the mirror-symmetry config + wiring both being correct.
2. `-90°` is bimodal: median 0.8° but mean 41.6° and p75 86.5° — so
   half the kicks are essentially perfect and half go +90° opposite.
   The policy has at least two modes, one that follows the command
   and one that defaults to ~+30°.

Implication: the exp(-angle²/σ²) reward saturates to ~zero at any large
angle (`exp(-(π/2)²/0.25) ≈ 1.4e-5`) so it gave zero gradient at the
broken-mode kicks — the policy could not learn to fix the cases that
needed correcting most.

### V1.29 — linear -angle² penalty (replaces exp form)
Drop `ball_velocity_alignment_post_kick` entirely (exp form, kept in
rewards.py for history). Add `kick_angle_error_l2`:

```
-angle² × (steps_since_kick ∈ [0,10)) × (speed > 1 m/s)
```

Gradient is `-2·angle`, monotonically larger at larger errors — the
opposite shape from exp, exactly designed for the broken-mode case.
Speed gate filters meaningless tiny kicks. Weight `-1.0`: at the
policy's worst-case angle ≈ π/2, episode contribution ≈ -2.47 × 10 active
steps = -24.7, comparable to `arm_deviation_l2` (-100 weight, episode
~-100 × 0.01 = -1 noted contribution actually). Strong enough to drive
learning, not strong enough to dwarf the kick mechanic rewards.

Warm-start from V1.28 `model_26000.pt`, 3000 iters. Plan: re-eval at
iter ~28000 to check whether mean error drops below 30° and L/R
asymmetry closes.

**V1.29 outcome (iter 26000 → 26920):** linear gradient pulled but
signal magnitude too weak (raw episode integral 0.003) — direction
metric showed only 11% reduction over 800 iters. Other rewards
recovered to V1.25 baseline (arm_dev -0.86, best of run at the time).

### V1.30 / V1.31 / V1.32 — three failed attempts to boost the signal
- **V1.30** (linear + speed multiplier, w=-3): raw magnitude 60× larger,
  but reward bounced 0.031-0.042 with no trend over 1500 iters. Eval
  at iter 28000: mean 51.3° / 25% success across [-90,-45,0,+45,+90],
  WORSE than V1.28 baseline (47.9° / 30%). +0° went 47° → 69°.
- **V1.31** (added pre_kick_body_yaw_alignment, w=-0.1): reward got
  *worse* (more misalignment) over 1000 iters — policy refused to
  turn because `base_yaw_rate_l2` (w=-0.5) was the dominant trade-off.
- **V1.32** (yaw_alignment w=-0.1 → -0.5, matching yaw_rate scale):
  still flat. Raw penalty barely changed. The policy was using
  non-yaw strategies (foot selection at contact) that the reward
  could not perturb.

Conclusion: 6 versions of reward shaping (V1.26-V1.32) could not
break the bimodal local optimum. The V1.30 eval median values were
revealing — +90° had median 1° error but mean 35° (bimodal split
between perfect and broken kicks). The capacity for accurate kicks
exists; the policy just inconsistently picks the right mode.

### V1.33 — curriculum on target distribution (the actual fix)
Pivoted from reward shaping to *changing what the policy trains on*.
Edited `commands.py:218` so target_angle sampling shifted from
`60/30/10` to `95/5/0`:
- 95% in [-π/4, π/4]   (front cone — primary use case)
- 5% in side cones    ([π/4, π/2] ∪ [-π/2, -π/4])
- 0% in rear arc      (removed)

Hypothesis: training on a mix of easy/medium/hard targets was
splitting the policy's capacity across modes. Forward-focused
distribution lets the gradient consolidate around one tight mode.

Warm-start from V1.32 `model_29500.pt`, 1500 iters.

**V1.33 outcome (iter 29500 → 30980):**

Training metrics (vs V1.32 plateau):
- `pre_kick_body_yaw_alignment`: 0.037 → **0.010** (-73%)
- `kick_angle_error_l2`: 0.037 → **0.027** (-27%)
- `target_progress`: 16.65 → 16.96 (+1.9%)
- `arm_deviation_l2`: -0.91 → **-0.78** (best of any run)
- `mean_reward`: 54.6 → **60.6** (+11%)

Eval on `model_30500.pt` (sweep `-45,-20,0,+20,+45`, 256 samples each):

| heading | V1.30 mean | V1.33 mean | V1.33 median | success% |
|---------|------------|------------|--------------|----------|
| -45°    | 64.6°      | 31.1°      | 15.6°        | 25%      |
| -20°    | (not run)  | 18.9°      | 8.0°         | **75%**  |
| +0°     | **69.3°**  | **24.8°**  | **18.2°**    | 0%       |
| +20°    | (not run)  | 14.2°      | 9.8°         | 50%      |
| +45°    | 65.8°      | 42.3°      | 19.4°        | 25%      |

**+0° improvement: 69.3° → 24.8° mean (-64%), 18° median.** The
curriculum is the real fix. Bimodality persists (medians are much
better than means) but the good mode now dominates.

Note: -45° / +45° asymmetry mostly closed (was V1.28's +45°=5° vs
-45°=68°, V1.30's +45°=65° vs -45°=64°, now V1.33 +45°=42° vs
-45°=31° — within noise band). L/R is symmetric again.

### V1.33b — extend curriculum for tighter forward convergence
Continue 95/5/0 sampling for another 2000 iters from `model_30500.pt`.
Goal: drive +0° mean error below 15°, median below 8°. After
convergence, V1.34 will gradually expand target range back toward
±π.

### V1.34–V1.41 — distance + arbitrary direction iteration (single-critic limit)
Distance curriculum (V1.34a–c) hit shortcut behavior at 0.6m+ (wind-up
kick reaches without stepping). Pivoted to focus on close-ball
arbitrary direction.

Perception fixes in V1.35 (camera Z offset -0.016 → +0.10 m to forehead;
noise_a 0.124 → 0.05, noise_b 0.149 → 0.08 — matches real RealSense
D435i + 2× sim2real margin) plus V1.36 reverted to 95/5/0 produced the
**single best forward** result of the project: +0° eval median 2.6°,
50% success at <10° threshold (V1.36 m31000).

V1.37 (90/8/2 with rear) → V1.38b (92/8/0 no rear) → V1.39 (94/6/0) →
V1.40 (99/1/0 from V1.38b) → V1.41 (3× direction reward boost) all
explored the same target-distribution + reward-weight axis. Each
landed on a different operating point in a **single-critic forward/side
trade-off**:

| version | +0° med | sides med (mean) | character |
|---------|---------|-------------------|-----------|
| V1.36 (95/5/0) | **2.6°** | ~40° | forward-specialized |
| **V1.38 m31500** | 10° | **14°** | **best balanced — current deliverable** |
| V1.38b m32000 | 33° ✗ | **6°** | side-specialized, forward broken |
| V1.39 (94/6/0) | 2.6° | 36° | bouncing between modes |
| V1.40 (99/1/0 short) | 14° | 27° | mixed |
| V1.41 (boost) | 15° | 20° | reward boost just shifts operating point |

Structural conclusion: the single PPO critic + small MLP (77→512→256→128)
cannot simultaneously specialize for forward (foot committed) and side
(foot selection based on target). The L/R foot indecision shows up at
exactly +0° heading when the policy has learned both modes.

**V1.38 m31500 (`logs/rsl_rl/k1_kick/2026-05-20_22-47-07/model_31500.pt`)
is the current canonical deliverable** for close-ball arbitrary-direction
kicking. Forward median 10°, sides median 7–32° within ±π/4. Eval
results in V1.38 outcome section above.

Next architectural step (deferred): multi-critic PPO (task #9), with
separate value heads for forward vs lateral kick zones, would let the
policy specialize in both regimes simultaneously.

## Diagnostic discipline

Established after multiple early misreads:
- Never declare a trend from 2 consecutive milestone point samples.
  `kick_success_rate` per-iter std ≈ 0.012; differences below ~3 σ
  (~0.04) over 500–1000 iters are noise.
- For verdicts use 500-iter window means via the tensorboard sub-agent
  (see `scripts/tools/` and per-iter dispatch via the milestone
  Monitor).
- Reward weight changes ≥ 5× should never be applied to a warm-started
  policy without a follow-up assessment — they can collapse learned
  skills (V1.5, V1.8).

## Open issues

- AMP discriminator saturation (V2). Likely needs richer state features
  (`body_lin_vel_w`, foot positions relative to root) so the
  discriminator can't trivially separate policy vs. reference frames.
  No walking reference clips on disk; the existing
  `motions/soccer-standard-mj-k1/*.npz` are walk-kick combinations and
  should be usable once state features are improved.
- Body float during kick swing (`base_lin_vel_z_l2` is firing but
  visible in viser). May resolve with more iterations on the V1.17
  reward stack.
- The `kick_completed` removal in V1.15 means episodes always run to
  the full 5 s — this is intentional so the homing phase has time to
  accumulate, but it does increase per-iter wall-clock cost.

## Tooling

- Training: `tmux new-session -d -s kick-v1 "WANDB_MODE=disabled
  MJLAB_DISABLE_CUDNN=1 uv run --extra cu124 train
  Mjlab-Kick-Flat-Booster-K1 --env.scene.num-envs 4096 ..."`.
- Play (viser, port 8080): `uv run --extra cu124 play
  Mjlab-Kick-Flat-Booster-K1 --checkpoint-file <path> --viewer viser
  --num-envs 1`. Kick target direction is rendered as a green arrow
  on the ball (`KickTargetCommand._debug_vis_impl`).
- Monitor: per-iter milestone notifications via the `Monitor` tool on
  the `kick-v1` tmux session, polling every 120 s and emitting on each
  500-iter step + on `Traceback|FAILED|OOM` patterns.
- Tensorboard dirs at `logs/rsl_rl/k1_kick/<timestamp>/`.

## Update policy

This document is updated whenever:
- A new V1.x or V2.x iteration is launched (record the change diff and
  expected effect).
- A run completes or is stopped due to regression (record the
  observed effect and the next decision).
- A new task ID or runner class is registered.
