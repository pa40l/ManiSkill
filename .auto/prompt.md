# Autoresearch: TakeItBack planner to 50/50

## Objective

Improve planner geometry and motion strategy until `myrobocasa_takeitback_planner` succeeds on all 50 seeds. Prefer structural geometric changes over threshold-only patches.

## Metrics

- Primary: success_count (count, higher is better), target 50/50.
- Secondary: failure categories, terminal stage, false grasp count, placement metric failures.

## How to Run

`NO_VIDEO=1 bash skills/maniskill-remote-planner-analysis/scripts/remote_analyze.sh myrobocasa_takeitback_planner 50 10`

## Files in Scope

- `planners/myrobocasa_takeitback_planner.py`: TakeItBack motion strategy.
- `utils/planners_utils.py`: reusable base-motion geometry helpers.
- `my_scenes/my_robocasa_takeitback.py`: task evaluation only when metric semantics require correction.

## Off Limits

Do not modify upstream ManiSkill files or files outside `my_scenes/`, `planners/`, and `utils/` except autoresearch session metadata. Do not add dependencies. Preserve physical safety and evaluation semantics.

## Baseline

50 seeds with 10 workers, no video: 37/50 (74%). Failures: 1,2,4,12,21,28,35,40,41,45,46,47,48.

## Failure map

- Stage 3 pre-grasp geometry/IK: 12,41,45,47. TCP often 5.7-9.9 cm away; orientation error 120-162 degrees.
- False grasp / lift slip: 2,46,48. `is_grasping` true while cup does not follow TCP.
- Release displacement / regrasp: 1,4,28. Cup shifts 5-28 cm after release; stale target and collisions.
- Transport discontinuity: 40. Cup-TCP gap grows 3.7 cm -> 35 cm during abrupt motion.
- Return joint limit: 21. Stage 10 rejects j5=1.63; base remains ~48 cm from start.
- Metric/settling: 35. Placement latch remains false; terminal angular velocity ~0.317 rad/s.

## Strategy order

1. Make full 6-D pre-grasp geometry reachable; do not reuse arbitrary current TCP orientation.
2. Validate grasp physically with stationary hold and micro-lift before transport.
3. Use slow bounded transport with live cup-TCP monitoring.
4. Verify tray placement and settle, then re-localize cup before regrasp.
5. Use joint-limit-safe carry posture and waypoint return.
6. Add fail-fast recovery; never spend thousands of steps repeating stuck/IK attempts.

## What's been tried

- Existing baseline already contains straight-arm geometry, torso ramps, base correction, grasp/lift checks, and fallback regrasp. It still fails 13/50, so next experiments must change geometry/phase contracts rather than add more blind retries.
