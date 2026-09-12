"""A scripted solution skeleton that matches the upstream contract. Copy this one.

All 17 solutions in ManiSkill are `solve(env, seed=None, debug=False, vis=False)`
returning either the sentinel `-1` or the gym 5-tuple. Matching that is not
pedantry — it is what makes `mani_skill/examples/motionplanning/panda/run.py`
usable, and run.py already does the seed sweep this repo keeps deferring: it loops
seeds, wraps each attempt in try/except, and reports success_rate,
failed_motion_plan_rate and episode lengths. It also separates "the planner found
no path" from "the plan ran and the task was not achieved", which is a distinction
a bare success flag cannot make.

The inherited `myrobocasa_planner.planning()` returns a tensor and takes different
arguments, so none of that machinery works with it. See docs/review-inherited-code.md.

This file is a skeleton, not a working solution: the stages below are the shape to
fill in, and it has never been run against the simulator.
"""

from __future__ import annotations

import numpy as np
import sapien

from utils.mikasa_oracle.motionplanning.fetch.extand import MikasaFetchSolver
from utils.mikasa_oracle.motionplanning.fetch.utils import compute_box_grasp_thin_side_info
from utils.mikasa.seeding import seed_everything

# Distance the fingers close over, used to sink the grasp into the object.
FINGER_LENGTH = 0.025


def hold_still(env, planner, n_steps: int):
    """Hold the current configuration for n_steps sim steps.

    A memory task's cue phase must be WAITED OUT, not acted through: the oracle
    reads the answer from the sim before an honest agent could know it, and if
    it starts moving early the recording shows behaviour that no cue-watching
    policy can reproduce. MIKASA-Robo's oracles idle through the cue exactly
    like this; RoboMME instead runs the cue as scripted subgoals before the
    policy acts.

    Delegates to `planner.idle_steps`, the solver's own "hold current qpos"
    primitive (extand.py:1498) — one implementation of the hold action, and the
    stub planner mirrors it, so an oracle's cue wait is testable without mplib.
    Do not hand-build the action here: `pd_joint_pos` takes absolute targets, so
    the layout must match the controller and a stray zero drops the torso.

    Args:
        env: the (possibly wrapped) env; only used to keep the signature stable.
        planner: a `MikasaFetchSolver` (or the stub).
        n_steps: how many control steps to hold.

    Returns:
        The last `(obs, reward, terminated, truncated, info)` tuple, or None if
        `n_steps <= 0`.

    Example:
        >>> phase_end = int(env.unwrapped.cue_steps.max().item())
        >>> res = hold_still(env, planner, phase_end - int(env.unwrapped.elapsed_steps.max().item()))
    """
    n = int(n_steps)
    if n <= 0:
        return None
    return planner.idle_steps(t=n)


def solve(env, seed=None, debug=False, vis=False):
    """Solve one episode. Returns the gym 5-tuple, or -1 if a plan failed.

    The contract, spelled out because every part of it is load-bearing:

    - **The solution owns the reset.** `env.reset(seed=seed)` is the first
      statement; the caller does not reset beforehand. run.py relies on this to
      drive one seed per attempt.
    - **Return `-1` when a plan fails**, and the 5-tuple otherwise. run.py counts
      `-1` as a failed motion plan and reads `res[-1]["success"]` otherwise, so
      returning anything else silently breaks both statistics.
    - **When to return `-1`, exactly** (the rule this repo settled on, D6, after
      the first three oracles were measured): `-1` means *the oracle never got
      to attempt the task* — a planner that found no path, an IK refusal, a
      grasp that would not close. Everything that happens *after* the attempt
      is a **miss**, not a no-plan: the cup settled on the wrong burner, the
      distractor got moved, the latch never fired, the pour was never held.
      Those return the **last 5-tuple** (whose `info["success"]` is False), so
      `utils.mikasa_oracle.evaluate_planner.classify_result` books them as `missed`
      and the sweep line reads
      `success h/N  (missed m, no plan p, errored e, truncated t)`.
      The reason to keep them apart: `no plan` is a motor/solver defect and
      `missed` is a task outcome — collapsing them makes a blind arm's honest
      wrong answers look like a broken planner, and the floor comparison stops
      meaning anything. An episode cut short by the horizon is a third thing
      again: return the truncated 5-tuple (`oracle_common.stopped_by_horizon`
      is the check) rather than reading a horizon cut as a failed plan.
    - **Do not catch exceptions.** run.py wraps the call and converts any exception
      into `-1`. Swallowing them here hides real errors as ordinary failures.
    - **Do not decide success.** The caller reads it out of `info`.
    - **Verify state after a stage, not just its return code.** A `-1` means the
      planner found no path; a non-`-1` only means the path was executed. Whether
      the cup came along is a question for `agent.is_grasping`, and whether it
      landed where intended is a question for the last step's `info`. Check,
      retry once with an alternative, then fail loudly — see the grasp stage.
    """
    env.reset(seed=seed)

    # Determinism: mplib's C++ RNG (std::srand, OMPL's RRT sampler, FCL) is not
    # covered by any Python-side seeding, and left unseeded it makes the same
    # seed return different outcomes — verified: four divergent runs without
    # this call, bitwise-identical runs with it (docs/lab-journal.md, 2026-08-10).
    # Seed it PER ATTEMPT, next to the reset, so an episode's outcome is a
    # property of its seed and not of how much RNG earlier episodes consumed.
    # `seed_everything` is the one place that knows the list (random, numpy,
    # torch, mplib) — do not add a second, hand-rolled mplib call here.
    if seed is not None:
        seed_everything(seed)

    # Upstream asserts this: a planner emits joint targets, so a delta or
    # end-effector control mode would silently mean something else.
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    planner = MikasaFetchSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped

    # For a task with cue phases (e.g. MikasaMemoryExample-v0), wait them out
    # BEFORE touching anything, or the demo leaks the answer through timing:
    #
    #   if hasattr(env, "cue_steps"):
    #       phase_end = int((env.cue_steps + env.delay_steps).max().item())
    #       hold_still(env, planner, phase_end - int(env.elapsed_steps.max().item()))

    # ----------------------------------------------------------------- grasp --
    mesh = env.cup.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        return -1
    obb = mesh.bounding_box_oriented

    tcp_pos = env.agent.tcp.pose.p[0].cpu().numpy()
    ee_direction = obb.center_mass - tcp_pos
    ee_direction = ee_direction / np.linalg.norm(ee_direction)
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        ee_direction=ee_direction,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
        ortho=True,
    )
    grasp_pose = env.agent.build_grasp_pose(
        grasp_info["approaching"], grasp_info["closing"], grasp_info["center"]
    )

    # ----------------------------------------------------------------- reach --
    # Every stage is guarded. The terse upstream solutions drop these return
    # values, which is how a failed reach turns into a confusing failure three
    # stages later — the inherited planner does the same at its alignment step.
    res = planner.static_manipulation(grasp_pose * sapien.Pose([0, 0, -0.1]))
    if res == -1:
        return res

    res = planner.static_manipulation(grasp_pose)
    if res == -1:
        return res
    planner.close_gripper()

    # ---------------------------------------------------- verify the grasp --
    # A return code says the plan executed, not that the object came along.
    # Check the STATE after the stage, and print what you checked: this stdout
    # is what the next debugging turn (human or coding agent) reads — a stage
    # that fails silently costs a week of plausible wrong sweep numbers, a stage
    # that fails loudly costs a minute (docs/coding-agent-primer.md §9).
    # One retry with the opposite closing direction, then give up honestly.
    if not bool(env.agent.is_grasping(env.cup).any()):
        print("[solve] grasp check failed: cup not in gripper; retrying with flipped closing")
        planner.open_gripper()
        grasp_info = compute_box_grasp_thin_side_info(
            obb,
            ee_direction=ee_direction,
            target_closing=-target_closing,
            depth=FINGER_LENGTH,
            ortho=True,
        )
        grasp_pose = env.agent.build_grasp_pose(
            grasp_info["approaching"], grasp_info["closing"], grasp_info["center"]
        )
        res = planner.static_manipulation(grasp_pose * sapien.Pose([0, 0, -0.1]))
        if res == -1:
            return res
        res = planner.static_manipulation(grasp_pose)
        if res == -1:
            return res
        planner.close_gripper()
        if not bool(env.agent.is_grasping(env.cup).any()):
            print("[solve] grasp check failed twice: giving up (returning -1)")
            return -1

    # ------------------------------------------------------------------ lift --
    res = planner.static_manipulation(
        sapien.Pose(grasp_pose.p + np.array([0.0, 0.0, 0.15]), grasp_pose.q)
    )
    if res == -1:
        return res

    # ----------------------------------------------------------------- place --
    # Fill in: drive the base with planner.drive_base / move_base_forward, then
    # lower and release. Keep guarding each stage the same way.

    res = planner.open_gripper()
    return res
