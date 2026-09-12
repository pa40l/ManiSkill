"""Scripted oracle for MikasaSameDrawer-v0.

Close the open drawer, put the apple on the plate at the far counter, come back and
re-open the same drawer. The answer (which drawer) is read at t=0, when the open
drawer is honestly visible in the observation — reading it then is perception, not
privilege (same argument season_dish documents for reading its cue during the cue
phase). What memory contributes is *retaining* it across the apple round trip, which
is exactly what `--blind` severs: the blind arm closes the true target (it can see
it) and re-opens a RANDOM drawer from the answer space, so over a sweep it must land
at 1/len(drawer_choices) of the sighted rate (1/3 in v1) — the measured floor.

Drawer manipulation, both directions, is verified by STATE (the joint's open amount
read from the env), never by a return code:

- CLOSE is a push: the fingers close into a fist and `static_manipulation` drives the
  TCP through the handle bar toward the cabinet until the joint reads closed. No
  grasp is needed to push.
- OPEN is a grasp-and-pull: the bar is a horizontal cylinder (r ~13 mm, 128 mm wide)
  standing 55 mm proud of the front — an easy grasp for a 100 mm aperture. Close on
  it, pull straight back along the slide axis, release.

`-1` is returned only for a planning/grasp refusal (D6); a physical miss that leaves
the world scoreable returns the last 5-tuple.
"""

from __future__ import annotations

import argparse
import math

import gymnasium as gym
import numpy as np
import sapien

from utils.mikasa_oracle.planners import oracle_common as common
from utils.mikasa_oracle.planners.oracle_common import default_planner_factory

from my_scenes.same_drawer import DRAWER_ART_SUFFIX, DRAWER_FIXTURES


def _art_name(drawer: int) -> str:
    """The planning-world articulation this drawer lives in."""
    return f"{DRAWER_FIXTURES[drawer]}{DRAWER_ART_SUFFIX}"
from utils.mikasa.seeding import seed_everything

WHO = "same_drawer_planner"

#: This oracle's RRT wall-clock budget, scoped via `common.planning_budget` exactly as
#: season_dish scopes its own (the solver default of 2 s leaves `RRTConnect Failed.
#: Approximate solution` refusals on reaches that plan cleanly at 6 s — measured on the
#: drawer-2 reach while bringing this oracle up; same finding as K79).
PLANNING_TIME_S = 6.0

#: Metres the TCP starts behind the handle bar before a push or a pull leg. 0.22, not
#: 0.12: at 0.12 the arm-only plan to the TOP drawer's bar (z 0.785) is refused while
#: 0.22 plans for all four drawers (orientation-corridor probe, 2026-08-28).
BAR_STANDOFF = 0.22

#: IK seed configurations for the bar reach/push/pull legs. mplib's default of 20
#: leaves marginal-corridor refusals that flip run to run; these legs are on the
#: failure path of nothing else, so the K79m caution (globally worse) does not apply.
REACH_N_INIT = 40

#: The push drives the TCP this far PAST the bar's closed-position centre, so the
#: drawer bottoms out and the detent (task-side) clicks it to exactly zero.
PUSH_PAST_CLOSED = 0.01

#: The pull target overshoots `cfg.open_success` by this much, because the release
#: lets the drawer settle back a few millimetres.
PULL_EXTRA = 0.05

#: The apple is released this far above the plate surface: high enough for the open
#: fingers to clear the rim, low enough that a rolling fruit stays aboard.
APPLE_HOVER = 0.09
APPLE_RELEASE = 0.06
RETRACT_UP = 0.12

#: `drive_to`'s squaring retry fires above these — metres and DEGREES, in the units
#: `oracle_common.dock_error` actually returns (`oracle_common.py:1548`; every other
#: oracle prints its second value as `dyaw_deg` unconverted). This one compared that
#: value against `0.20` and then ran it through `math.degrees` a second time, so the
#: gate read as 0.20 DEGREES and the trace printed a heading error 57x too large.
#: 11.5 deg is the 0.20 rad the comparison was written for. Measured-neutral: over
#: EVAL 0-29 the base parks within 0.012 m and 0.044 deg of the one dock this task has
#: (runs/2026-09-03-fix-e/A, 30 seeds), so neither gate has ever fired — this is an
#: instrument fix and a latent horizon trap removed, not a success-rate change.
DOCK_SQUARE_M = 0.10
DOCK_SQUARE_DEG = 11.5


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def say(env, stage: str, **extra) -> None:
    """`oracle_common.say` tagged with this oracle's name."""
    common.say(env, WHO, stage, **extra)


def fail(env, stage: str, **extra):
    """`oracle_common.fail` tagged with this oracle's name."""
    return common.fail(env, WHO, stage, **extra)


def bar_grasp_poses(task, drawer: int, open_amt: float,
                    topdown: bool = False) -> tuple[sapien.Pose, sapien.Pose]:
    """TCP grasp pose on the handle bar, and its standoff, for `drawer` as it stands.

    The bar's world centre travels with the joint as `home + (0, -open_amt, 0)`
    (`same_drawer.py` stores the closed homes). Approach is +y — straight at the
    cabinet — and the fingers close vertically across the bar's diameter.

    Example:
        >>> class _T:  # doctest: +SKIP
        ...     handle_home = ...
        >>> g, r = bar_grasp_poses(task, 2, 0.11)          # doctest: +SKIP
        >>> bool(g.p[1] < r.p[1])                          # reach is behind the grasp
        True
    """
    home = _np(task.handle_home)[0][drawer].astype(np.float64)
    centre = home + np.array([0.0, -float(open_amt), 0.0])
    if topdown:
        # Hook frame: approach straight down, fingers closing along y so they land
        # fore and aft of the bar (bar axis is x); the reach hovers above it.
        grasp = task.agent.build_grasp_pose(
            np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), centre)
        reach = sapien.Pose(p=centre + np.array([0.0, 0.0, BAR_STANDOFF]), q=grasp.q)
        return grasp, reach
    approaching = np.array([0.0, 1.0, 0.0])
    closing = np.array([0.0, 0.0, 1.0])
    grasp = task.agent.build_grasp_pose(approaching, closing, centre)
    # The reach sits at a FIXED depth behind the closed-bar home, not at an offset
    # from the moving bar: the plannable corridor at the top drawer is a band around
    # world y ~ home_y - 0.22 (probed both ways on 2026-08-28 — an offset from an
    # open bar leaves it, and 0.12 from a closed bar leaves it the other way). With
    # init_open <= 0.16 this depth is always behind the bar.
    reach = sapien.Pose(p=home + np.array([0.0, -BAR_STANDOFF, 0.0]), q=grasp.q)
    return grasp, reach


def target_amount(task, drawer: int) -> float:
    """The drawer's open amount, read from the joint — the only truth this oracle
    trusts about the drawers."""
    return float(_np(task.drawer_open_amounts())[0][drawer])


def stow_arm(env, planner, task, z: float = 1.10, aheads=(0.70,)) -> None:
    """Pull the empty hand in over the base before a drive.

    The drives carry whatever posture the last manipulation left, and
    `rotate_base_z` sweeps the turn's arc for collisions — with the arm still
    stretched at drawer height, the final turn at the apple dock was refused
    `rotation sweep hits ...stack_3...` (measured). The colliding furniture is all
    under-counter (tops <= 0.89), so the fix is HEIGHT, not a tuck: raise the fist
    above ~1.1 and the sweep clears everything; `static_manipulation` (with its
    RRT fallback) does it from any posture, where a pure vertical screw hits limits. Non-fatal: a refusal still tries the drive (refusals cost no steps)."""
    planner.close_gripper()
    tcp = _np(task.agent.tcp.pose.p).reshape(-1)[:3]
    if len(aheads) == 1 and abs(tcp[2] - z) <= 0.10:
        return  # already at drive height, and no tighter tuck was asked for
    base = _np(task.agent.base_link.pose.p).reshape(-1)[:3]
    q = _np(task.agent.base_link.pose.raw_pose).reshape(-1)[3:]
    yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
    face = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    tcp_q = _np(task.agent.tcp.pose.raw_pose).reshape(-1)[3:]
    # A ladder over the pull-in distance, tightest first: how close the hand can
    # tuck depends on the wrist family the last stage left (with the vertical-closing
    # bar wrist, IK starts at ~0.70 ahead; the top-down apple wrist tucks closer).
    # The tightest tuck that plans wins — near the left wall the arm's overhang is
    # the difference between a drivable dock approach and `wrist<->wall_left_room`.
    for ahead in aheads:
        target = sapien.Pose(p=base + face * ahead + np.array([0.0, 0.0, z]), q=tcp_q)
        if planner.static_manipulation(target, disable_lift_joint=False,
                                       n_init_qpos=REACH_N_INIT) != -1:
            say(env, "hand stowed for the drive", ahead=ahead)
            return
    say(env, "stow refused; driving with the hand as it is", tcp_z=round(float(tcp[2]), 2))


def unwind_base(env, planner, task) -> None:
    """Unwind the base yaw joint before a long drive.

    `rotate_base_z` always takes the locally short way and nothing tracks the
    winding (K54/D14): after the outbound drive, the slide and the release, the
    `root_z_rotation_joint` can sit on its ±2π stop, where every further turn
    reports `jammed=True` and moves nothing — the return drive then silently
    no-ops (measured: three jammed rotations, base parked 1.57 m from the dock).
    The solver reports the jam and leaves recovery to the caller; this is the
    caller's half: two opposite-sign half-turns bring the joint back near zero.
    A refused hop is non-fatal — the drive is attempted regardless."""
    if not callable(getattr(planner.robot, "get_qpos", None)):
        return  # the offline stub has no joints to unwind
    for _ in range(3):
        yaw_j = float(_np(planner.robot.get_qpos()).reshape(-1)[2])
        if abs(yaw_j) < 2.0:
            return
        say(env, "unwinding the base yaw", joint=round(yaw_j, 2))
        if planner.rotate_z_delta(delta=-math.copysign(math.pi, yaw_j)) == -1:
            say(env, "unwind hop refused")
            return


def drive_to(env, planner, task, dock_xyyaw, label: str):
    """`drive_base` to a dock, verified by `dock_error`; one squaring retry."""
    d = np.asarray(dock_xyyaw, dtype=np.float64).reshape(-1)
    face = np.array([math.cos(d[2]), math.sin(d[2]), 0.0])
    pos = np.array([d[0], d[1], 0.0])
    res = planner.drive_base(target_pos=pos, target_view_vec=face, freeze_arm=True)
    if res == -1:
        return fail(env, f"drive to {label}"), False
    dist, dyaw = common.dock_error(task, (d[0], d[1], d[2]))  # metres, degrees
    if dist > DOCK_SQUARE_M or dyaw > DOCK_SQUARE_DEG:
        res = planner.drive_base(target_pos=pos, target_view_vec=face, freeze_arm=True)
        if res == -1:
            return fail(env, f"drive to {label} (retry)"), False
        dist, dyaw = common.dock_error(task, (d[0], d[1], d[2]))
    say(env, f"parked at {label}", d_dock=round(dist, 3), dyaw_deg=round(dyaw, 1))
    return res, dist <= 0.15


def close_drawer(env, planner, task, drawer: int, cfg) -> tuple:
    """Push the drawer shut; verified by the joint. Three attempts before giving up
    (a refused plan inside an attempt continues to the next — RRT is randomized)."""
    frontal_progress = True
    for attempt in range(4):
        amt = target_amount(task, drawer)
        if amt <= cfg.closed_tol:
            say(env, "drawer already closed", drawer=drawer, attempt=attempt)
            return 0, True
        # Stay frontal while it is MAKING PROGRESS (a push that executes with
        # tracking error closes partway — seed 0 halved the gap and the old ladder
        # threw the working family away); switch to the hook only when frontal
        # stalls or refuses outright.
        if attempt == 0:
            frontal_progress = True
        topdown = (attempt >= 1) and not frontal_progress
        grasp, reach = bar_grasp_poses(task, drawer, amt, topdown=topdown)
        # Frontal: a closed fist pushes through the bar. Hook (top-down): the OPEN
        # fingers straddle the bar fore-and-aft, and the rear finger drags it — the
        # floor-band family WaterPlants proves for low grasps; a level fist at the
        # bottom drawer's z=0.155 refused its reach three times running.
        res = planner.close_gripper() if not topdown else planner.open_gripper()
        if res == -1:
            return res, False
        # The stroke's whole point is contact, so the target drawer leaves the
        # planning world for these legs (contact_stroke, K100); the other three
        # drawers stay in as obstacles.
        with common.contact_stroke(planner, [_art_name(drawer)]):
            res = planner.static_manipulation(reach, disable_lift_joint=False,
                                              n_init_qpos=REACH_N_INIT)
            if res == -1:
                # Arm-only refused; the refusals here read `RRTConnect Failed.
                # Approximate solution` — IK exists, the tree cannot connect from
                # rest. Base freedom bridges exactly this (every probe that planned
                # these poses was 15-dof), and nothing is held yet: let the base
                # shuffle. World-frame targets are unaffected by the shift.
                res = planner.move_to_pose_with_RRTConnect(reach, n_init_qpos=REACH_N_INIT)
            if res == -1:
                say(env, "close: reach refused; another draw", attempt=attempt)
                frontal_progress = topdown and frontal_progress
                continue
            if common.stopped_by_horizon(planner):
                return res, False
            if topdown:
                res = planner.static_manipulation(grasp, disable_lift_joint=False,
                                                  n_init_qpos=REACH_N_INIT)
                if res == -1:
                    say(env, "close: hook descend refused; another draw", attempt=attempt)
                    continue
            # Through the bar to just past its closed-position centre: the drawer
            # bottoms out and the task's detent clicks it to zero.
            push = sapien.Pose(p=grasp.p + np.array([0.0, float(amt) + PUSH_PAST_CLOSED, 0.0]),
                               q=grasp.q)
            res = planner.static_manipulation(push, disable_lift_joint=False,
                                              n_init_qpos=REACH_N_INIT)
            if res == -1:
                # The stroke too may need the base to creep with the hand (a human
                # leans in); world-frame target, same bridge as the reach.
                res = planner.move_to_pose_with_RRTConnect(push, n_init_qpos=REACH_N_INIT)
        planner.planner.update_from_simulation()
        if res == -1:
            say(env, "close: push refused; another draw", attempt=attempt)
            frontal_progress = topdown and frontal_progress
            continue
        if common.stopped_by_horizon(planner):
            return res, False
        amt_before = amt
        amt = target_amount(task, drawer)
        frontal_progress = (not topdown) and (amt_before - amt > 0.02)
        say(env, "push done", drawer=drawer, attempt=attempt, open_amt=round(amt, 4))
        if amt <= cfg.closed_tol:
            off = np.array([0.0, 0.0, BAR_STANDOFF]) if topdown else (
                np.array([0.0, -BAR_STANDOFF, 0.0]))
            back = sapien.Pose(p=push.p + off, q=push.q)
            res = planner.static_manipulation(back, disable_lift_joint=False)
            return (res if res != -1 else 0), True
    return fail(env, "close: drawer would not close"), False


def open_drawer(env, planner, task, drawer: int, cfg) -> tuple:
    """Grasp the bar and pull; verified by the joint. Three attempts before giving up
    (refused plans inside an attempt continue to the next — RRT is randomized)."""
    res = planner.open_gripper()
    if res == -1:
        return res, False
    for attempt in range(3):
        amt = target_amount(task, drawer)
        if amt >= cfg.open_success:
            return 0, True
        grasp, reach = bar_grasp_poses(task, drawer, amt)
        # The grasp goal sits ON the bar and the pull drags the drawer with the
        # hand — all of it is deliberate contact with this one drawer, so it leaves
        # the planning world for these legs (contact_stroke, K100).
        with common.contact_stroke(planner, [_art_name(drawer)]):
            res = planner.static_manipulation(reach, disable_lift_joint=False,
                                              n_init_qpos=REACH_N_INIT)
            if res == -1:
                res = planner.move_to_pose_with_RRTConnect(reach, n_init_qpos=REACH_N_INIT)
            if res == -1:
                say(env, "open: reach refused; another draw", attempt=attempt)
                continue
            res = planner.static_manipulation(grasp, disable_lift_joint=False,
                                              n_init_qpos=REACH_N_INIT)
            if res == -1:
                say(env, "open: grasp leg refused; another draw", attempt=attempt)
                continue
            res = planner.close_gripper()
            if res == -1:
                return res, False
            if common.stopped_by_horizon(planner):
                return res, False
            pull = float(cfg.open_success) + PULL_EXTRA - amt
            pull_pose = sapien.Pose(p=grasp.p + np.array([0.0, -pull, 0.0]), q=grasp.q)
            res = planner.static_manipulation(pull_pose, disable_lift_joint=False)
            if res == -1:
                res = planner.move_to_pose_with_RRTConnect(pull_pose, n_init_qpos=REACH_N_INIT)
            if res == -1:
                # The bar is in the fingers; let go before any retry re-plan.
                planner.open_gripper()
                return fail(env, "open: the pull leg"), False
            res = planner.open_gripper()
            if res == -1:
                return res, False
            back = sapien.Pose(p=pull_pose.p + np.array([0.0, -0.08, 0.10]), q=pull_pose.q)
            res_b = planner.static_manipulation(back, disable_lift_joint=False)
        planner.planner.update_from_simulation()
        amt = target_amount(task, drawer)
        say(env, "pull done", drawer=drawer, attempt=attempt, open_amt=round(amt, 4))
        if amt >= cfg.open_success:
            return (res_b if res_b != -1 else res), True
    return res, False


def place_apple(env, planner, task) -> tuple:
    """Grasp the apple and set it on the plate — the burner stage-6 pattern."""
    mesh = task.apple.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        return fail(env, "apple: no collision mesh"), False
    obb = mesh.bounding_box_oriented
    base_p = _np(task.agent.base_link.pose.p).reshape(-1)[:3]
    ee_dir = np.asarray(obb.center_mass, dtype=np.float64) - base_p
    ee_dir[2] = 0.0
    ee_dir /= max(np.linalg.norm(ee_dir), 1e-9)
    closing = np.cross(ee_dir, [0.0, 0.0, 1.0])
    # The season ladder, compact: families x closings x grip heights x draws, with a
    # retreat between rungs (a failed rung leaves the arm where IK seeds die — the
    # arc lesson). One golden frame kept flaking seed to seed; the ladder is what
    # took SeasonDish to its measured rate, and this is that shape.
    rungs = []
    for dz in (0.0, 0.03):
        rungs += [("frontal", +1.0, dz), ("frontal", -1.0, dz)]
        rungs += [("top", az, dz) for az in (0.0, 45.0, 90.0, -45.0)]
    res, grasped = -1, False
    for k, (kind, par, dz) in enumerate(rungs):
        live = task.apple.get_first_collision_mesh(to_world_frame=True)
        if live is None:
            return fail(env, "apple: no collision mesh"), False
        obb = live.bounding_box_oriented
        centre = np.asarray(obb.center_mass, dtype=np.float64)
        base_p = _np(task.agent.base_link.pose.p).reshape(-1)[:3]
        ee_dir = centre - base_p
        ee_dir[2] = 0.0
        ee_dir /= max(np.linalg.norm(ee_dir), 1e-9)
        closing = np.cross(ee_dir, [0.0, 0.0, 1.0])
        if kind == "frontal":
            grasp, reach = common.grasp_geometry(task, obb, ee_dir, par * closing)
            z = max(float(centre[2]), float(live.bounds[1][2]) - 0.03) + dz
            dzz = z - float(grasp.p[2])
            grasp = sapien.Pose(p=np.asarray(grasp.p) + [0, 0, dzz], q=grasp.q)
            reach = sapien.Pose(p=np.asarray(reach.p) + [0, 0, dzz + 0.04], q=reach.q)
        else:
            a = np.deg2rad(par)
            c, sn = np.cos(a), np.sin(a)
            cl = np.array([closing[0] * c - closing[1] * sn,
                           closing[0] * sn + closing[1] * c, 0.0])
            gp = centre + np.array([0.0, 0.0, dz])
            grasp = task.agent.build_grasp_pose(np.array([0.0, 0.0, -1.0]), cl, gp)
            reach = grasp * sapien.Pose([0, 0, -0.10])
        for _draw in range(2):
            res, grasped = common.try_grasp(env, planner, task, task.apple, grasp, reach,
                                            resync_before_grasp=True, close_on_contact=True,
                                            n_init_qpos=REACH_N_INIT)
            if res != -1:
                break
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res != -1 and not grasped:
            # `is_grasping`'s antipodal cone dislikes a sphere pinched off-equator:
            # the flag reads False on physically held apples. Verify by STATE — the
            # drawers' own rule: micro-lift and ask whether the apple came along.
            z0 = float(_np(task.apple.pose.p).reshape(-1)[2])
            if planner.lift_hand(delta_h=0.05) != -1:
                z1 = float(_np(task.apple.pose.p).reshape(-1)[2])
                grasped = (z1 - z0) > 0.02
        if res != -1 and grasped:
            say(env, "apple grasped", rung=k, kind=kind, dz=dz)
            break
        # Retreat before the next family so its IK is not seeded from a buried pose.
        planner.open_gripper()
        if planner.lift_hand(delta_h=0.10) == -1:
            tcp_now = task.agent.tcp.pose.sp
            back = sapien.Pose(p=tcp_now.p, q=tcp_now.q) * sapien.Pose([0, 0, -0.12])
            planner.static_manipulation(back, disable_lift_joint=False)
        planner.planner.update_from_simulation()
    if res == -1 or not grasped:
        return (res if res == -1 else fail(env, "apple: the ladder is exhausted")), False
    common.hold_object_in_planner(env, planner, task, task.apple, True, who=WHO)

    lift = sapien.Pose(p=_np(task.agent.tcp.pose.p).reshape(-1)[:3] + [0, 0, 0.12],
                       q=_np(task.agent.tcp.pose.raw_pose).reshape(-1)[3:])
    res = planner.static_manipulation(lift, disable_lift_joint=False)
    if res == -1:
        res = planner.lift_hand(delta_h=0.12)
    if res == -1:
        return fail(env, "apple: lift"), False

    # Release by BASE SLIDE, not by arm hover: every marginal-IK fight today was an
    # arm reach, and the base never refused once. The apple hangs from the TCP; slide
    # the base along the counter until the apple stands over the plate, open, retract.
    apple_p = _np(task.apple.pose.p).reshape(-1)[:3]
    plate_p = _np(task.plate.pose.p).reshape(-1)[:3]
    base_p = _np(task.agent.base_link.pose.p).reshape(-1)[:3]
    q = _np(task.agent.base_link.pose.raw_pose).reshape(-1)[3:]
    yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
    face = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    shift = plate_p[:2] - apple_p[:2]
    target = base_p[:2] + shift
    res = -1
    for _ in range(2):
        res = planner.drive_base(target_pos=[float(target[0]), float(target[1]), 0.0],
                                 target_view_vec=face, freeze_arm=True)
        if res != -1:
            break
    if res == -1:
        # Plan B: the burner hover — marginal here, but better than giving up.
        tcp = task.agent.tcp.pose.sp
        T_tcp_obj = tcp.inv() * task.apple.pose.sp
        obj_q = task.apple.pose.sp.q
        hover = sapien.Pose(p=plate_p + [0, 0, APPLE_HOVER], q=obj_q) * T_tcp_obj.inv()
        res = planner.static_manipulation(hover, disable_lift_joint=False,
                                          n_init_qpos=REACH_N_INIT)
    if res == -1:
        return fail(env, "apple: neither slide nor hover reached the plate"), False
    if common.stopped_by_horizon(planner):
        return res, False
    res = planner.open_gripper()
    common.hold_object_in_planner(env, planner, task, task.apple, False, who=WHO)
    planner.planner.update_from_simulation()
    tcp = task.agent.tcp.pose.sp
    retract = sapien.Pose(p=np.asarray(tcp.p) + [0, 0, RETRACT_UP], q=tcp.q)
    res_r = planner.static_manipulation(retract, disable_lift_joint=False)
    planner.idle_steps(t=10)
    apple_now = _np(task.apple.pose.p).reshape(-1)
    on = float(np.linalg.norm(apple_now[:2] - plate_p[:2]))
    say(env, "apple released", apple_to_plate=round(on, 3))
    return (res_r if res_r != -1 else res), on <= float(task.cfg.plate_radius)


def solve(env, seed=None, debug=False, vis=False, blind=False, *,
          planner_factory=default_planner_factory):
    """Solve one episode; `-1` on a planning/grasp refusal, the gym 5-tuple otherwise.

    Runs under this oracle's own RRT budget (`PLANNING_TIME_S`), scoped so the other
    oracles keep the solver default."""
    with common.planning_budget(PLANNING_TIME_S):
        return _solve(env, seed=seed, debug=debug, vis=vis, blind=blind,
                      planner_factory=planner_factory)


def _solve(env, seed=None, debug=False, vis=False, blind=False, *,
           planner_factory=default_planner_factory):
    obs, info = env.reset(seed=seed)
    if seed is not None:
        seed_everything(seed)

    assert env.unwrapped.control_mode in (
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ), env.unwrapped.control_mode

    planner = planner_factory(env, debug, vis)
    task = env.unwrapped
    cfg = task.cfg
    rng = np.random.default_rng(seed)

    # The answer, read at t=0 — when it is honestly visible as the one open drawer.
    # Blindness (the floor's definition) is forgetting it across the apple trip, so
    # the blind arm still CLOSES the true target and only re-opens at random.
    target = int(_np(task.target_drawer).reshape(-1)[0])
    reopen = int(rng.choice(np.asarray(cfg.drawer_choices))) if blind else target
    say(env, "episode", blind=bool(blind), target=target, reopen=reopen,
        init_open=round(target_amount(task, target), 3))

    # -- STAGE 1: close the open drawer --------------------------------------------
    say(env, "close the drawer")
    res, closed = close_drawer(env, planner, task, target, cfg)
    if res == -1:
        return res
    if common.stopped_by_horizon(planner):
        say(env, "stopped by the horizon during the close")
        return res
    if not closed:
        say(env, "MISSED: the drawer did not close")
        return res if res != 0 else planner.idle_steps(t=1)

    # -- STAGE 2: the apple round trip ----------------------------------------------
    say(env, "drive to the apple counter")
    stow_arm(env, planner, task, aheads=(0.70, 0.55, 0.85))
    unwind_base(env, planner, task)
    dock = _np(task.apple_dock).reshape(-1, 3)[0]
    res, parked = drive_to(env, planner, task, dock, "the apple dock")
    if res == -1:
        return res
    if common.stopped_by_horizon(planner):
        return res

    say(env, "apple onto the plate")
    # Face the apple dead-on first: the dock is drawn at a fixed along-offset while
    # the apple jitters, so the ray can be diagonal — and every diagonal family today
    # planned worse than the straight one (K59's finding, re-confirmed all morning).
    # The base micro-slide is the one primitive that has not refused once.
    apple_x = float(_np(task.apple.pose.p).reshape(-1)[0])
    base_now = _np(task.agent.base_link.pose.p).reshape(-1)[:3]
    if abs(apple_x - base_now[0]) > 0.03:
        planner.drive_base(target_pos=[apple_x, float(base_now[1]), 0.0],
                           target_view_vec=[0.0, 1.0, 0.0], freeze_arm=True)
    planner.open_gripper()
    res, placed = place_apple(env, planner, task)
    if res == -1:
        return res
    if common.stopped_by_horizon(planner):
        return res
    if not placed:
        say(env, "MISSED: the apple is not on the plate")
        return res

    # -- STAGE 3: back, and open the remembered drawer -------------------------------
    say(env, "drive back to the drawer column")
    # Tightest-first ladder HERE only: near the left wall the arm's overhang decides
    # whether the dock approach plans. The outbound stow stays pinned at 0.70 — the
    # apple rungs are calibrated to start from exactly that raised state.
    stow_arm(env, planner, task, aheads=(0.35, 0.50, 0.70))
    unwind_base(env, planner, task)
    start = _np(task._robot_start).reshape(-1, 7)[0]
    yaw = 2.0 * math.atan2(float(start[6]), float(start[3]))
    # Approach from the SOUTH: driving straight to the dock means facing the travel
    # direction (-x), which points the arm into the left room wall for the final
    # stretch (measured: `forearm/gripper/wrist<->wall_left_room` with 0.177 of the
    # twist left). Via a waypoint below the dock, the last leg travels +y — the arm
    # faces the drawers, and the wall is never faced.
    # Waypoint at (1.0, -2.3), not straight below the dock: the column stands 0.4 m
    # from the left wall, and any WESTWARD-facing arrival puts the raised arm through
    # the wall plane (measured twice). From (1.0, -2.3) the final leg's bearing is
    # ~119 deg — mostly north — and the arm tip stays east of the wall the whole way.
    res = planner.drive_base(target_pos=[1.0, -2.3, 0.0],
                             target_view_vec=[0.0, 1.0, 0.0], freeze_arm=True)
    if res == -1:
        return fail(env, "drive to the return waypoint")
    if common.stopped_by_horizon(planner):
        return res
    res, parked = drive_to(env, planner, task, (start[0], start[1], yaw), "the drawer dock")
    if res == -1:
        return res
    if common.stopped_by_horizon(planner):
        return res

    say(env, "open the drawer", drawer=reopen)
    res, opened = open_drawer(env, planner, task, reopen, cfg)
    if res == -1:
        return res
    if common.stopped_by_horizon(planner):
        return res
    if not opened:
        say(env, "MISSED: the drawer did not open far enough")
        return res

    # Terminal hold: the predicate wants `hold_steps` settled steps.
    say(env, "hold", steps=int(cfg.hold_steps) + 5)
    out = planner.idle_steps(t=int(cfg.hold_steps) + 5)
    return out if out is not None else res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    env = gym.make("MikasaSameDrawer-v0", scene_idx=0, obs_mode="state",
                   robot_uids="mikasa_ds_fetch", control_mode="pd_joint_pos",
                   render_mode="rgb_array")
    res = solve(env, seed=args.seed, debug=args.debug, blind=args.blind)
    if res == -1:
        print("no plan found (-1)")
    else:
        print("final info:", {k: _np(v).tolist() for k, v in res[4].items()})
    env.close()


if __name__ == "__main__":
    main()
