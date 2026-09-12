"""Oracle for MikasaBurner-v0, and its blind twin.

Written against `tools/stub_planner.py` on a Mac (mplib has no macOS wheel) and,
since 2026-08-18, run for real in the amd64 CPU container
(`tools/docker/planner.sh run …`) — the geometry below is what the container
measured, not a guess. The shared stage kit lives in `oracle_common.py` (D7);
every public name here stays as a thin wrapper so this file remains the staging
that ships and the one the offline tests read.

What the oracle does, in order, and why each choice is what it is:

1. **Holds still to the end of the CUE phase, not the delay.** An honest agent may
   drive as soon as the marker is gone — the road to the stove is what eats the
   delay. Reading `task.cue_steps` to know when that is uses privileged timing;
   `template_planner.py` allows the oracle exactly that. It does *not* read
   `cue_id` before the cue is over, so the demonstration shows nothing a
   cue-watching policy could not also do.
2. **Reads the answer once**, after the cue: `int(task.cue_id[0])`. The `--blind`
   arm replaces this single line with a random draw and changes nothing else, so
   the two arms differ only in memory. Blind must land at ~0.25 x the sighted
   arm's motor rate; higher means the cue leaks into the act phase.
3. **Grasps the cup** with the geometry of `myrobocasa_planner.py`: OBB thin side,
   flip if the reach pose is on the far side of the cup, fall back to the opposite
   closing direction if the reach plan fails. Then *verifies* the grasp with
   `agent.is_grasping(cup)`; one retry with the opposite closing; then `-1`,
   with the reason printed. A missed grasp is "never started", which belongs in
   failed_motion_plan_rate and not in the success rate.
4. **Lifts with the torso frozen** (`disable_lift_joint=True`), **tucks the cup
   into a carry pose over the base** (`oracle_common.carry_pose`, K52: the TCP
   0.25 m ahead of the base centre at its current height, the cup upright — its
   orientation held, else the hand yawed about world z; torso frozen) and
   **drives to the stove dock**, slid along the stove's face to the *target
   burner's column* (`dock_for_target`), turning to face the stove.
   `rotate_base_z` checks each turn's swept arc with the attached cup (K51), so
   a `rotation sweep hits …` refusal after the tuck is a real obstacle to the
   base itself and an honest `-1`. (D13 — the tuck only as a recovery after a
   refusal — was the interim rule while mplib's attached-body frame error made
   the tuck unplannable; K53 fixed the frame on 2026-08-18 and the carry is a
   full stage again.) After the drive it logs `d_dock` and `|dyaw|` (K26 — how
   well the base parks is what decides the station task's zone mode, D5).
5. **Hovers over the target burner**, lowers, opens, retracts, settles. After the
   settle it reads `info["on_burner_id"]` from the last step; a cup that settled
   on the wrong burner is a **physical miss** and returns that last 5-tuple (D6)
   — the sweep books it as `missed`, and `-1` stays reserved for "the planner
   gave up before a decision". Every stage checks `stopped_by_horizon(planner)`
   before interpreting its result (K23), so an episode cut by the horizon comes
   back as the truncated tuple, never as a fake `-1`.

Contract, as `template_planner.py:32-45`: the solution owns the reset; return `-1`
on a failed plan and the gym 5-tuple otherwise; do not catch exceptions; do not
decide success. And do not call `env.evaluate()` — read `info` from the tuple.
"""

from __future__ import annotations

import argparse
import math
import sys

import gymnasium as gym
import numpy as np
import sapien

from mani_skill.utils.wrappers import RecordEpisode

from utils.mikasa_oracle.planners import oracle_common as common
from utils.mikasa.seeding import seed_everything

# The mplib-dependent imports stay deferred inside oracle_common's factories, so
# this module imports on a Mac and tests/test_burner.py can run the *real* solve()
# against tools/stub_planner.py.

WHO = "burner_planner"

# Distance the fingers close over, used to sink the grasp into the object.
FINGER_LENGTH = common.FINGER_LENGTH

# Where the cup origin goes over the burner, relative to the burner top: hover,
# then release height. cup_2 rests with its origin ~0.054 m above the surface
# (measured on CPU), so releasing at RELEASE_ABOVE = 0.11 drops it about 6 cm —
# high enough that the open fingers clear the rim, low enough that it lands
# inside the 8 cm zone.
HOVER_ABOVE = 0.10 + 0.054
RELEASE_ABOVE = 0.11

def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def say(env, stage: str, **extra) -> None:
    """`oracle_common.say` tagged `[burner_planner]` — stdout + a `phase` event."""
    common.say(env, WHO, stage, **extra)


def fail(env, stage: str, **extra):
    """Return -1 for a failed stage, saying which one — never a silent -1."""
    return common.fail(env, WHO, stage, **extra)


def default_planner_factory(env, debug: bool, vis: bool):
    """The real solver at the oracle refinement cap (60). Needs mplib."""
    return common.default_planner_factory(env, debug, vis)


def default_grasp_info(obb, ee_direction, target_closing) -> dict:
    """OBB thin-side grasp frame, `utils.mikasa_oracle.motionplanning.fetch.utils`. Needs mplib."""
    return common.default_grasp_info(obb, ee_direction, target_closing, finger_length=FINGER_LENGTH)


def wait_cue(env, planner, info):
    """Idle until `elapsed_steps >= cue_steps` — the CUE phase only, not the delay."""
    return common.wait_cue(env, planner, info, who=WHO)


def choose_target(task, blind: bool, rng: np.random.Generator) -> int:
    """The one line the blind arm replaces. Called only after wait_cue."""
    if blind:
        return int(rng.integers(task.cfg.n_burners))
    return int(_np(task.cue_id)[0])


def grasp_geometry(task, obb, ee_direction, target_closing, grasp_info=default_grasp_info):
    """(grasp_pose, reach_pose) for the cup, with the far-side flip of myrobocasa_planner.py:100-120."""
    return common.grasp_geometry(task, obb, ee_direction, target_closing, grasp_info)


def try_grasp(env, planner, task, grasp, reach):
    """reach -> grasp -> close. Returns (res, grasped) with res == -1 on a failed plan."""
    return common.try_grasp(env, planner, task, task.cup, grasp, reach)


def burner_target(task, target: int) -> np.ndarray:
    return _np(task.burner_pos)[0, target].astype(np.float64)


def cup_pose_over(burner_xyz: np.ndarray, above: float, cup_q) -> sapien.Pose:
    return common.pose_over(burner_xyz, above, cup_q)


def hold_cup_in_planner(env, planner, task, held: bool) -> None:
    """Tell the planning world that the cup is (or no longer is) part of the hand.

    `oracle_common.hold_object_in_planner` on `task.cup` — see its docstring for
    why the attach exists and why the touch links are spelled out.
    """
    common.hold_object_in_planner(env, planner, task, task.cup, held, who=WHO)


def dock_for_target(dock_xyyaw, burner_xyz):
    """`common.dock_for_target`, kept under this name for the tests that cite it."""
    return common.dock_for_target(dock_xyyaw, burner_xyz)


def solve(
    env,
    seed=None,
    debug=False,
    vis=False,
    blind=False,
    *,
    planner_factory=default_planner_factory,
    grasp_info=default_grasp_info,
):
    """Solve one episode. `-1` on a failed plan, the gym 5-tuple otherwise.

    The two keyword-only hooks exist for the offline test: they default to the
    real solver and the real grasp geometry, and `run.py`-style drivers never
    pass them.
    """
    obs, info = env.reset(seed=seed)

    # Seeds Python, numpy, torch AND mplib's C++ RNG; without the last one the same
    # seed returns different outcomes (utils.mikasa/seeding.py, measured). Per
    # attempt, next to the reset, so an episode belongs to its seed.
    if seed is not None:
        seed_everything(seed)

    assert env.unwrapped.control_mode in (
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ), env.unwrapped.control_mode

    planner = planner_factory(env, debug, vis)
    task = env.unwrapped
    rng = np.random.default_rng(seed)

    # -- STAGE 0: sit through the cue -------------------------------------------
    info = wait_cue(env, planner, info)
    if info == -1:
        return info

    # -- STAGE 1: the privileged read (or the blind draw) -------------------------
    target = choose_target(task, blind, rng)
    say(env, "target chosen", target=target, blind=bool(blind))

    # -- STAGE 2: grasp the cup -----------------------------------------------------
    say(env, "grasp the cup")
    mesh = task.cup.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        return fail(env, "grasp the cup: no collision mesh")
    obb = mesh.bounding_box_oriented

    tcp_pos = _np(task.agent.tcp.pose.p)[0]
    ee_direction = obb.center_mass - tcp_pos
    ee_direction = ee_direction / np.linalg.norm(ee_direction)
    target_closing = _np(task.agent.tcp.pose.to_transformation_matrix())[0, :3, 1]

    grasp, reach = grasp_geometry(task, obb, ee_direction, target_closing, grasp_info)
    res, grasped = try_grasp(env, planner, task, grasp, reach)
    # K23: the horizon first, before reading a cut-off grasp as a failure or
    # retrying a grasp the episode no longer has time for.
    if res != -1 and common.stopped_by_horizon(planner):
        say(env, "stopped by the horizon during the grasp")
        return res
    if res == -1 or not grasped:
        # Once more with the opposite closing direction (myrobocasa_planner.py:125-140),
        # whether the plan failed or the fingers closed on air.
        say(env, "grasp retry with flipped closing", plan_failed=bool(res == -1), grasped=grasped)
        planner.open_gripper()
        planner.planner.update_from_simulation()
        grasp, reach = grasp_geometry(task, obb, ee_direction, -target_closing, grasp_info)
        res, grasped = try_grasp(env, planner, task, grasp, reach)
        if res != -1 and common.stopped_by_horizon(planner):
            say(env, "stopped by the horizon during the grasp retry")
            return res
        if res == -1:
            return fail(env, "grasp the cup (retry)")
        if not grasped:
            return fail(env, "grasp the cup: fingers closed but agent.is_grasping(cup) is False "
                             "after both closing directions")
    planner.planner.update_from_simulation()
    hold_cup_in_planner(env, planner, task, held=True)

    # -- STAGE 3: lift, torso frozen ------------------------------------------------
    # disable_lift_joint=True: the lift must not spend the torso — the hover is
    # where the torso lever matters (K19), and the carry pose below re-freezes it
    # anyway.
    say(env, "lift")
    res = planner.static_manipulation(
        sapien.Pose(grasp.p + np.array([0.0, 0.0, 0.15]), grasp.q), disable_lift_joint=True
    )
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "lift")
    planner.planner.update_from_simulation()
    # The cup's orientation relative to the base, before the tuck turns the hand:
    # the hover after the drive asks for the cup in this same relation to the
    # parked base (hand along the base's facing), not as the tuck left it.
    cup_rel = common.object_pose_in_base(task, task.cup)

    # -- STAGE 4: carry pose — the cup tucked in over the base (K52) ------------------
    # `rotate_base_z` sweeps each turn's arc with the attached cup (K51); with the
    # arm stretched out after the lift the cup can genuinely sweep through
    # furniture, so it is pulled in first. `carry_pose` tries the held orientation,
    # then the hand yawed about world z (cup upright under each) and executes the
    # first that plans; none → an honest -1 (D6). A full stage again since K53
    # (before it, mplib's attached-body phantom made the tuck unplannable and D13
    # kept it as a recovery only).
    res = common.carry_pose(env, planner, task, task.cup, who=WHO)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return res  # carry_pose said why
    planner.planner.update_from_simulation()

    # -- STAGE 5: drive to the stove dock -----------------------------------------------
    dock = _np(task._stove_dock_np)[0]
    burner = burner_target(task, target)
    dock_xyz, face = dock_for_target(dock, burner)
    say(env, "drive to stove dock", dock=[round(float(v), 3) for v in dock_xyz], target=target)
    res = planner.drive_base(target_pos=dock_xyz, target_view_vec=face)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "drive to stove dock")
    planner.planner.update_from_simulation()
    # K26: how well the base parked, in the trace — this number decides the
    # station task's zone mode (D5: nearest if max d_dock > 0.10 m over T3's runs).
    d_dock, dyaw = common.dock_error(task, (dock_xyz[0], dock_xyz[1], math.atan2(face[1], face[0])))
    say(env, "parked at the dock", d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1))

    # World-frame, so it must be read *after* the drive: rotate_base_z turns the held
    # cup with the base, and the counter dock and the stove dock differ by 90 degrees
    # in every L/U/G-shaped kitchen (compute_robot_base_placement_pose returns
    # base_fixture.rot + pi/2, so the groups that carry a group_z_rot move the dock
    # with them). Composed here, where the robot is parked at the dock: the cup in
    # its pre-tuck relation to the base (`cup_rel`, hand along the facing) with
    # the base as it stands — not the cup as the carry pose left it (the tuck may
    # have yawed the hand 90 degrees; a hover inheriting that asked the wrist for a
    # sideways reach and failed IK on seed 3, measured after K53).
    cup_q = common.object_q_from_base(task, cup_rel)
    # The grasp transform too — "cup origin at X" becomes "TCP at X * T_tcp_cup.inv()".
    # It was first captured right after the lift, on the theory that a rigid grasp
    # makes it invariant to the drive; the emulated run showed the cup shifting in
    # the fingers during the base rotations, so read it here, from the same still
    # moment as cup_q.
    T_tcp_cup = (task.agent.tcp.pose[0].inv() * task.cup.pose[0]).sp

    # -- STAGE 6: hover, lower, release, retract -----------------------------------
    hover = cup_pose_over(burner, HOVER_ABOVE, cup_q) * T_tcp_cup.inv()
    lower = cup_pose_over(burner, RELEASE_ABOVE, cup_q) * T_tcp_cup.inv()

    say(env, "hover over burner", target=target, hover=[round(float(v), 3) for v in hover.p])
    res = planner.static_manipulation(hover, disable_lift_joint=False)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "hover over burner")
    say(env, "lower")
    res = planner.static_manipulation(lower, disable_lift_joint=False)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "lower")
    say(env, "release")
    res = planner.open_gripper()
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "release")
    hold_cup_in_planner(env, planner, task, held=False)
    planner.planner.update_from_simulation()
    # Straight up, and higher than the hover: the released cup is an obstacle again
    # (convex hull, wider than the cup) right between the open fingers, and a plan
    # back to the hover pose failed IK on the emulated run. Not fatal either way —
    # the cup is already released, and what decides the episode is where it rests.
    say(env, "retract")
    retract = sapien.Pose(p=hover.p + np.array([0.0, 0.0, 0.12]), q=hover.q)
    res = planner.static_manipulation(retract, disable_lift_joint=False)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        say(env, "retract failed; leaving the arm where it is (cup already released)")

    # Let the cup settle so the verdict can latch, then check the aim.
    say(env, "settle")
    res = planner.idle_steps(t=15)
    if res != -1 and common.stopped_by_horizon(planner):
        say(env, "stopped by the horizon during the settle")
        return res
    if res == -1:
        return fail(env, "settle")
    landed = int(_np(res[-1]["on_burner_id"])[0])
    say(env, "landed", target=target, on_burner_id=landed)
    if landed != target:
        # D6: a physical miss returns the episode as it stands — the plan ran, the
        # cup is where it is, and the sweep books it as `missed`. `-1` would claim
        # the planner never got to a decision, which is false here.
        say(env, f"missed: aimed at burner {target}, cup is on {landed} (-1 = none); "
                 "a placement failure, not a plan failure")
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--scene-idx", type=int, default=0)
    p.add_argument("--output-dir", default="videos/burner")
    p.add_argument("--render-mode", default="rgb_array")
    p.add_argument("--render-width", type=int, default=512)
    p.add_argument("--render-height", type=int, default=512)
    p.add_argument("--max-steps-per-video", type=int, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-trajectory", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--blind",
        action="store_true",
        help="memory-free control: pick a random burner instead of reading the cue",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    env = gym.make(
        "MikasaBurner-v0",
        num_envs=1,
        render_mode=args.render_mode,
        robot_uids="mikasa_ds_fetch",
        control_mode="pd_joint_pos",
        scene_idx=args.scene_idx,
        human_render_camera_configs=dict(width=args.render_width, height=args.render_height),
    )
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_video=not args.no_video,
        save_trajectory=not args.no_trajectory,
        video_fps=30,
        save_on_reset=True,
        max_steps_per_video=args.max_steps_per_video,
    )

    res = solve(env, seed=args.seed, debug=args.debug, vis=False, blind=args.blind)
    if res == -1:
        print("failed_motion_plan")
    else:
        info = res[-1]
        print(
            "success:", bool(_np(info["success"])[0]),
            "| cue_id:", int(_np(info["cue_id"])[0]),
            "| on_burner_id:", int(_np(info["on_burner_id"])[0]),
            "| wrong_burner_settled:", bool(_np(info["wrong_burner_settled"])[0]),
            "| elapsed:", int(_np(info["elapsed"])[0]),
        )
    env.close()
    return res


if __name__ == "__main__":
    sys.exit(0 if main() != -1 else 1)
