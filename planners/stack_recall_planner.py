"""Scripted oracle for MikasaStackRecall-v0: unstack, deliver the bottom cube
east, and restack the movables on the home plinth in the remembered order.

Every motor form here is the W17 measurement (17 instrumented runs, journal
K109) and nothing else:

- **side grasp of a cube**, approach +y / closing x, TCP `GRASP_Z_OFF` above
  the cube centre (the centre grasp is dead: pad bottoms end ~7 mm over the
  support and the planner refuses the margin; A's centre grasp ploughed the
  cube below 316 mm);
- **grasps only at plinth heights** (1.4885+): the shelf floor is a measured
  dead zone twice over — the face-frame lip pops a cube parked south of
  y~-0.21 back north 3.7 cm, and every lone-cube grasp at 1.4425 dies on
  `wrist<->cab` (8 W17 rungs);
- **seat-with-stroke**: a cube is released 4 mm over its support with the
  support taken out of the planning world for the leg (a 15 mm gap refuses
  the attached-cube margin; a 25 mm drop bounces the cube 58 mm off) and the
  seat goal carries GRASP_Z_OFF (without it the held cube pressed 6 mm into
  the support and PhysX popped it sideways);
- **the retreat after a seat goes FLAT first** (the up-and-back diagonal
  clipped the seated cube's top edge and kicked it 72 mm);
- **the east leg is DRIVEN** (a base teleport with cargo drops the cargo —
  probe-measured; the oracle drives anyway).

The SCRAMBLE is the oracle's friend: when the target lands, the env teleports
both movables to the side plinth slots (rest error 3 mm, W17) — sloppy
staging during the unstack is normalized by the task itself, so the staging
seats are best-effort (a refused seat parks nothing and the episode goes on).

Return contract (D6): -1 only for a planning/grasp refusal before physics has
committed (the first successful cube grasp is the commit); after it, misses
return the last 5-tuple.
"""

from __future__ import annotations

import types
from contextlib import contextmanager

import numpy as np
import sapien
import torch

from utils.mikasa_oracle.planners import oracle_common as common
from utils.mikasa_oracle.planners.oracle_common import fail as _fail
from utils.mikasa_oracle.planners.oracle_common import say as _say
from utils.mikasa.seeding import seed_everything

WHO = "stack_recall_planner"

GRASP_Z_OFF = 0.010
"""TCP height above the grasped cube's CENTRE (W17: the centre grasp is dead,
+10 mm gives the pads ~17 mm of support clearance with 12+ mm of cube held)."""
SEAT_DROP = 0.004
"Release gap over the support inside a seat stroke (W17: 15 refuses, 25 bounces)."
LIFT_DZ = 0.03
RETRACT_REACH = 0.55
"W13/K105: the retract is a REACH from the base, not a world y."
WORK_DOCK_Y = -1.05
EAST_DOCK_DY = -0.75
"The place dock sits this far south of the place point (SeasonDish reach band)."
GRASP_TRIES = 3
"""Per-cube grasp attempts: W17 measured close-on-air/knock as a per-binary
coin toss (fingers 0.0, cube slid a few cm) — detectable and retryable from a
re-read pose."""


def say(env, stage: str, **extra) -> None:
    """`oracle_common.say` with this oracle's tag.

    Example:
        >>> say(env, "unstack the top cube")                      # doctest: +SKIP
    """
    _say(env, WHO, stage, **extra)


def fail(env, stage: str, **extra):
    """`oracle_common.fail` with this oracle's tag; returns -1.

    Example:
        >>> return fail(env, "no grasp held")                     # doctest: +SKIP
    """
    return _fail(env, WHO, stage, **extra)


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


@contextmanager
def actor_stroke(planner, name_substr: str):
    """contact_stroke for an ACTOR: remove the named body from the planning
    world for a leg that must touch or seat onto it (K100 logic; W17 measured
    the seat forms dead without it). Restored on every exit path BEFORE the
    caller's next `update_from_simulation` — a sync with an entity missing
    raises (the K106 lesson).

    Example:
        >>> with actor_stroke(planner, "plinth_0"):               # doctest: +SKIP
        ...     res = planner.static_manipulation(seat_goal)
    """
    world = getattr(getattr(planner, "planner", None), "planning_world", None)
    if world is None:
        yield
        return
    removed = []
    try:
        for full in list(world.get_object_names()):
            if name_substr in full:
                obj = world.get_object(full)
                world.remove_object(full)
                removed.append((full, obj))
        yield
    finally:
        for full, obj in removed:
            try:
                world.add_object(full, obj)
            except TypeError:
                try:
                    world.add_object(obj)
                except Exception:
                    pass
            except Exception:
                pass


def cube_p(task, ci) -> np.ndarray:
    return _np(task.cubes[ci].pose.p).reshape(-1)


def grasp_pose(task, centre):
    """The W13/W17 side grasp: approach +y, closing x."""
    return task.agent.build_grasp_pose(
        np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        np.asarray(centre, dtype=np.float64))


def held(task, ci) -> bool:
    return bool(_np(task.agent.is_grasping(task.cubes[ci])).reshape(-1)[0])


def grasp_cube(env, planner, task, ci, *, stage: str, pre_offset: float = 0.15):
    """Grasp cube `ci` where it stands NOW, with the W17 retry ladder: the
    pose is re-read on every attempt (a knocked cube slides a few cm and
    stays graspable), the final leg keeps the plain form (the touch-stop
    fired early on the cube's top and closed on air — W17 run 8).

    Returns the executor result, or -1 when every attempt missed."""
    res = -1
    z_ladder = (GRASP_Z_OFF, 0.014, 0.012)
    pre_lifts = (0.0, 0.05, 0.08)
    for attempt in range(GRASP_TRIES):
        planner.planner.update_from_simulation()
        p = cube_p(task, ci)
        goal = grasp_pose(task, [p[0], p[1],
                                 p[2] + z_ladder[attempt % len(z_ladder)]])
        pre = goal * sapien.Pose([0, 0, -float(pre_offset)])
        plift = pre_lifts[attempt % len(pre_lifts)]
        if plift:
            # the W17 diagonal entry, counter edition: a lifted pre is
            # better-conditioned than the horizontal one at counter heights
            # (sweep 13: `joint limit [7, 11]` IK death on the flat pre)
            pre = sapien.Pose(p=[pre.p[0], pre.p[1], pre.p[2] + plift],
                              q=pre.q)
        r = planner.static_manipulation(pre)
        if r == -1:
            r = planner.static_manipulation(pre)
        if r == -1:
            say(env, f"{stage}: pre-grasp refused", attempt=attempt)
            continue
        r = planner.static_manipulation(goal)
        if r == -1:
            say(env, f"{stage}: grasp leg refused", attempt=attempt)
            continue
        res = planner.close_gripper(t=12)
        if held(task, ci):
            common.hold_object_in_planner(env, planner, task, task.cubes[ci],
                                          held=True, who=WHO)
            say(env, f"{stage}: cube in the gripper", cube=ci, attempt=attempt)
            return res
        say(env, f"{stage}: close missed", attempt=attempt)
        planner.open_gripper(t=6)
    return -1


def carry_out(env, planner, task, ci):
    """The W13 exit with the cube attached: +0.03 lift, retract by reach.
    Non-fatal legs (a refused retract leaves the arm where it is; the next
    stage plans from there)."""
    tcp = task.agent.tcp.pose.sp
    planner.static_manipulation(
        sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + LIFT_DZ], q=tcp.q))
    base_y = float(_np(task.agent.robot.get_qpos()).reshape(-1)[1])
    tcp = task.agent.tcp.pose.sp
    planner.static_manipulation(
        sapien.Pose(p=[tcp.p[0], base_y + RETRACT_REACH, tcp.p[2]], q=tcp.q))


def seat_cube(env, planner, task, ci, support_name, x, y, z_top, *, stage: str):
    """Seat the HELD cube onto a support: goal `SEAT_DROP` over the support's
    top with the support stroked out of the planning world, release, detach,
    retreat FLAT then up (every number is a W17 casualty — see the module
    docstring). Returns True when the seat leg planned; the release happens
    either way only after a planned approach (never a mid-air drop)."""
    ok = False
    for drop, app, plift in ((SEAT_DROP, -0.10, 0.0), (SEAT_DROP, -0.12, 0.05),
                             (0.012, -0.16, 0.0), (SEAT_DROP, -0.14, 0.08)):
        # LIVE in-hand compensation: a cube gripped by its top band hangs
        # PENDULUM-swung against the palm — its centre rides 40-46 mm SOUTH
        # of the TCP (the seat trace, sweep 19) — so aim the TCP where the
        # CUBE must land, offset by the measured in-hand displacement.
        tcp_now = task.agent.tcp.pose.sp
        c_now = cube_p(task, ci)
        off = c_now - np.asarray(tcp_now.p, dtype=np.float64).reshape(-1)
        # HIGH approach: the horizontal leg runs 20+ mm over the support, so
        # the pendulum-swung cube's leading bottom corner cannot drag on it
        # (sweep 20: at a 4 mm clearance the drag ate the whole compensation
        # — the cube landed 41-50 mm south of aim regardless).
        goal_hi = grasp_pose(task, [x - float(off[0]), y - float(off[1]),
                                    z_top + task.cfg.cube_half + drop + 0.020
                                    - float(off[2])])
        pre = goal_hi * sapien.Pose([0, 0, app])
        if plift:
            pre = sapien.Pose(p=[pre.p[0], pre.p[1], pre.p[2] + plift], q=pre.q)
        planner.static_manipulation(pre)
        with actor_stroke(planner, support_name):
            r2 = planner.static_manipulation(goal_hi)
            if r2 == -1:
                r2 = planner.static_manipulation(goal_hi)
            if r2 != -1:
                # fresh pendulum read at the hover, then a PURE-VERTICAL
                # 20 mm descent — no horizontal component, no drag
                tcp_now = task.agent.tcp.pose.sp
                c_now = cube_p(task, ci)
                off = c_now - np.asarray(tcp_now.p,
                                         dtype=np.float64).reshape(-1)
                goal_lo = grasp_pose(
                    task, [x - float(off[0]), y - float(off[1]),
                           z_top + task.cfg.cube_half + drop - float(off[2])])
                r3 = planner.static_manipulation(goal_lo)
                if r3 == -1:
                    planner.static_manipulation(goal_lo)
        if r2 != -1:
            ok = True
            break
    if ok:
        c0 = cube_p(task, ci)
        planner.open_gripper(t=8)
        common.hold_object_in_planner(env, planner, task, task.cubes[ci],
                                      held=False, who=WHO)
        c1 = cube_p(task, ci)
        tcp = task.agent.tcp.pose.sp
        planner.static_manipulation(
            sapien.Pose(p=[tcp.p[0], tcp.p[1] - 0.20, tcp.p[2]], q=tcp.q))
        tcp = task.agent.tcp.pose.sp
        planner.static_manipulation(
            sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + 0.05], q=tcp.q))
        c2 = cube_p(task, ci)
        landed = float(np.hypot(c2[0] - x, c2[1] - y))
        if landed > 0.020:
            say(env, f"{stage}: seat landed off aim",
                off_mm=round(landed * 1000, 1),
                at_open=[round(float(v), 3) for v in c0[:2]],
                after_retreat=[round(float(v), 3) for v in c2[:2]])
    say(env, f"{stage}: seat {'ok' if ok else 'REFUSED'}", cube=ci)
    return ok


def drop_anywhere(env, planner, task, ci, *, stage: str):
    """Last-resort staging release when a seat refused: lower the cube over
    the counter band south of the dock and let go — the SCRAMBLE re-seats
    every movable on the slots when the target lands, so a rough drop costs
    nothing but dignity (W17: slot rest error 3 mm regardless of staging)."""
    base = _np(task.agent.robot.get_qpos()).reshape(-1)
    ctop = float(_np(task._counter_top_z).reshape(-1)[0])
    goal = grasp_pose(task, [float(base[0]) - 0.15, -0.60,
                             ctop + task.cfg.cube_half + 0.04 + GRASP_Z_OFF])
    r = planner.static_manipulation(goal)
    planner.open_gripper(t=8)
    common.hold_object_in_planner(env, planner, task, task.cubes[ci],
                                  held=False, who=WHO)
    say(env, f"{stage}: rough drop", planned=r != -1)


def plan_joints(env, planner, task, targets: dict, *, label: str, tries: int = 2):
    """Plan and execute a joint-space move, straight line first, RRT second.

    The SECOND trimmed copy of water_plants_planner.plan_to_joint_targets
    (cabinet_retrieval_planner carries the first) — a third copy promotes the
    helper into oracle_common, per the repo rule.

    Example:
        >>> res = plan_joints(env, planner, task, {"torso_lift_joint": 0.2},
        ...                   label="duck")                       # doctest: +SKIP
    """
    p = planner.planner
    robot = task.agent.robot
    for i in range(int(tries)):
        cur = _np(robot.get_qpos()).reshape(-1).astype(np.float64)
        goal = cur.copy()
        for n, v in targets.items():
            goal[robot.active_joints_map[n].active_index[0].item()] = float(v)
        line = p.plan_qpos_line(goal, cur, time_step=task.control_timestep,
                                ref_yaw=float(cur[2]))
        if line["status"] == "Success" and len(np.asarray(line["position"])) > 1:
            say(env, f"{label}: joint line",
                knots=int(np.asarray(line["position"]).shape[0]))
            return planner.follow_forward_path_w_refinement(line, refine=True)
        result = p.plan_qpos(
            [p.fold_qpos(p.pad_move_group_qpos(goal))],
            p.fold_qpos(p.pad_move_group_qpos(cur)),
            time_step=task.control_timestep, planning_time=8.0, rrt_range=0.1,
            simplify=True, fixed_joint_indices=[0, 1, 2], ref_yaw=float(cur[2]),
        )
        if result["status"] == "Success":
            return planner.follow_forward_path_w_refinement(result, refine=True)
        say(env, f"{label}: plan refused", status=result["status"], draw=i + 1)
    return -1


def fold_to_rest(env, planner, task):
    """Fold the arm to the REST pose (wrist off its stop) before a drive —
    K106's lesson verbatim, re-measured here: freeze_arm keeps the arm out of
    the base screws, but the drive plan still carries the extended-arm
    profile, and solo run 1's west drive refused 3/3 with the place-retreat
    arm still stretched east. Non-fatal.

    Example:
        >>> fold_to_rest(env, planner, task)                      # doctest: +SKIP
    """
    rest_q = np.asarray(task.agent.keyframes["rest"].qpos,
                        dtype=np.float64).reshape(-1)
    jm = task.agent.robot.active_joints_map
    arm_names = list(task.agent.controller.controllers["arm"].config.joint_names)
    targets = {n: float(rest_q[jm[n].active_index[0].item()]) for n in arm_names}
    targets["torso_lift_joint"] = float(
        rest_q[jm["torso_lift_joint"].active_index[0].item()])
    # The PURE rest keyframe — no duck: the rest TCP rides at z 1.25, the one
    # corridor above the stove top (1.08) and below the microwave box (1.37),
    # and the initial dock drive from rest passed all 17 solo runs. The
    # K105-style duck (torso 0.2) drops the boom to z~1.05 where the west
    # screws clip the counter AND the rotate sweep hallucinates (K109).
    return plan_joints(env, planner, task, targets, label="fold to rest", tries=3)


def plinth_xyz(task, pi):
    p = _np(task.plinths[pi].pose.p).reshape(-1)
    return float(p[0]), float(p[1]), float(p[2]) + task.cfg.cube_half


def drive_dock(env, planner, task, x, y, *, stage: str, tries=2):
    """drive_base with the arm frozen (the K106 BASE_PLAN_MASK lesson) and a
    short redraw ladder."""
    for t in range(tries):
        # Sync BEFORE every attempt: sweep 2 measured rotate_base_z's sweep
        # verdict frozen across 0.65 m of executed base motion — the check
        # reads the planning world, and a freeze_arm screw leaves it stale.
        planner.planner.update_from_simulation()
        tcp = task.agent.tcp.pose.sp
        base = _np(task.agent.robot.get_qpos()).reshape(-1)
        say(env, f"{stage}: PRE-ROTATE DIAG",
            tcp=[round(float(v), 3) for v in tcp.p],
            base=[round(float(v), 3) for v in base[:4]])
        r = planner.drive_base(target_pos=np.array([x, y, 0.0]),
                               target_view_vec=np.array([0.0, 1.0, 0.0]),
                               freeze_arm=True)
        if r != -1:
            normalize_base_yaw(env, planner, task)
            return r
        # A "refused" drive often ARRIVES anyway: the screw executes and only
        # the final view rotate dies — on the measured sweep-hallucination
        # (solo 13: the rotate check names obstacles metres away, invariant
        # to 0.65 m of base motion; filed for its own probe). Accept a dock
        # within 8 cm and 45 deg of north — the 7-DOF arm absorbs the yaw.
        base = _np(task.agent.robot.get_qpos()).reshape(-1)
        d = float(np.hypot(base[0] - x, base[1] - y))
        dyaw = float((base[2] - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi)
        if d <= 0.08 and abs(dyaw) <= np.pi / 4:
            say(env, f"{stage}: refused rotate but ARRIVED",
                d=round(d, 3), dyaw_deg=round(np.degrees(dyaw), 1))
            normalize_base_yaw(env, planner, task)
            return planner.idle_steps(t=1)
        say(env, f"{stage}: drive refused, one more draw", attempt=t,
            d=round(d, 3))
    return -1


def normalize_base_yaw(env, planner, task):
    """Rewrite the base yaw qpos into (-pi, pi] — the SAME physical pose of a
    continuous joint, a representation change and not a teleport (the K55
    unwrap precedent). Solo 13/14 measured the wound representation (yaw
    7.85+ after long-way turns) feeding the rotate sweep hallucinated
    obstacles metres away and killing downstream IK; filed for its own probe.

    Example:
        >>> normalize_base_yaw(env, planner, task)                # doctest: +SKIP
    """
    q = task.agent.robot.get_qpos()
    q = q.clone() if torch.is_tensor(q) else torch.as_tensor(q).clone()
    yaw = float(q.reshape(-1)[2])
    wrapped = (yaw + np.pi) % (2 * np.pi) - np.pi
    if abs(wrapped - yaw) > 1e-9:
        q.reshape(-1)[2] = wrapped
        task.agent.robot.set_qpos(q)
        planner.planner.update_from_simulation()
        say(env, "base yaw normalized", was=round(yaw, 3), now=round(wrapped, 3))


def back_off(env, planner, task, delta, *, stage: str):
    """A CHECKED, arm-frozen backward screw. `move_forward_delta` plans with
    the default mask and sweep 1 measured it dying silently on
    `joint limit [7]` at 95% of the twist — the screw bent the arm instead of
    driving the base (the BASE_PLAN_MASK disease), so no back-off ever
    happened and the whole rung above it was fiction.

    Example:
        >>> back_off(env, planner, task, -0.35, stage="off the alcove")  # doctest: +SKIP
    """
    cur = task.agent.base_link.pose.sp
    direction = cur.to_transformation_matrix()[:3, 0]
    direction[2] = 0.0
    target = np.asarray(cur.p, dtype=np.float64) + direction * float(delta)
    r = planner.move_base_forward(target, n_init_qpos=100, freeze_arm=True)
    say(env, f"{stage}: back-off {'ok' if r != -1 else 'REFUSED'}",
        delta=delta)
    return r


def align_dock(env, planner, task, x, *, stage: str, with_cargo: bool):
    """Dock the base under column `x` with the measured posture discipline:
    empty-handed drives go in the DUCK (fold_to_rest — solo 4/9: the rest
    gripper clips the counter edge on the screw), and the arm returns to REST
    after parking (mplib IK seeds from the current config — W13/solo 5); a
    CARGO drive keeps the carry arm as it is (solo 1/2: the extended carry
    drives clean, the folded carry sweeps the cube through the counter).

    Example:
        >>> align_dock(env, planner, task, 2.66, stage="x", with_cargo=False)  # doctest: +SKIP
    """
    if not with_cargo:
        fold_to_rest(env, planner, task)
        planner.planner.update_from_simulation()
    r = drive_dock(env, planner, task, x, WORK_DOCK_Y, stage=stage, tries=3)
    if r == -1:
        return -1
    if not with_cargo:
        rest_q = np.asarray(task.agent.keyframes["rest"].qpos,
                            dtype=np.float64).reshape(-1)
        jm = task.agent.robot.active_joints_map
        arm_names = list(
            task.agent.controller.controllers["arm"].config.joint_names)
        targets = {n: float(rest_q[jm[n].active_index[0].item()])
                   for n in arm_names}
        targets["torso_lift_joint"] = float(
            rest_q[jm["torso_lift_joint"].active_index[0].item()])
        plan_joints(env, planner, task, targets, label="back to rest", tries=3)
    planner.planner.update_from_simulation()
    return r


def solve(env, seed=None, debug=False, vis=False, blind=False,
          planner_factory=common.default_planner_factory):
    """Solve one episode. `-1` on a pre-commit refusal, the gym 5-tuple otherwise.

    Args:
        env: the (possibly wrapped) MikasaStackRecall-v0 env.
        seed: episode seed; the solution owns the reset.
        debug, vis: passed to the solver factory.
        blind: the design blank's control arm — the restack order is a random
            coin instead of the remembered one; everything else is identical.
            Must land at 0.5 x the sighted rate over a sweep (2 orders).
        planner_factory: seam for `tools/stub_planner.py`.

    Returns:
        -1, or the last gym 5-tuple.

    Example:
        >>> res = solve(env, seed=0)                              # doctest: +SKIP
    """
    obs, info = env.reset(seed=seed)
    if seed is not None:
        seed_everything(seed)
    assert env.unwrapped.control_mode in ("pd_joint_pos", "pd_joint_pos_vel"), \
        env.unwrapped.control_mode
    task = env.unwrapped
    planner = planner_factory(env, debug, vis)

    # The answer: the original level order. Privileged but honest — the stack
    # order is readable off the cube heights in the very first observation
    # (the design blank's point 7). The blind arm flips a coin instead.
    order = [int(v) for v in _np(task.original_order).reshape(-1).tolist()]
    bottom, mid, top = order[0], order[1], order[2]
    if blind:
        flip = bool(np.random.randint(0, 2))
        restack = (top, mid) if flip else (mid, top)
        say(env, "BLIND: restack order is a coin", flip=flip)
    else:
        restack = (mid, top)
    say(env, "episode", order=order, blind=bool(blind))

    hx, hy, htop = plinth_xyz(task, 0)

    # -- 0: duck and dock at the shelf ----------------------------------------
    res = common.duck_torso(env, planner, task) \
        if hasattr(common, "duck_torso") else -1
    r = drive_dock(env, planner, task, hx, WORK_DOCK_Y, stage="dock at the shelf")
    if r == -1:
        return fail(env, "drive to the work dock")
    res = r

    # -- 1: unstack the top cube ----------------------------------------------
    r = grasp_cube(env, planner, task, top, stage="unstack top")
    if r == -1:
        return fail(env, "grasp the top cube")
    res = r
    if common.stopped_by_horizon(planner):
        return res
    carry_out(env, planner, task, top)
    # Staging is a RELEASE at the retract pose SHIFTED WEST: the cube drops
    # to the counter and the SCRAMBLE re-seats it later, so tidiness buys
    # nothing — but the retract column IS the place column, and sweep 11
    # measured the litter blocking the target's descent
    # (`fingers<->dropped cube` at 97%%). West, away from the place disc.
    tcp = task.agent.tcp.pose.sp
    planner.static_manipulation(
        sapien.Pose(p=[tcp.p[0] - 0.18, tcp.p[1], tcp.p[2]], q=tcp.q))
    planner.open_gripper(t=8)
    common.hold_object_in_planner(env, planner, task, task.cubes[top],
                                  held=False, who=WHO)
    say(env, "stage top: released over the counter")
    planner.planner.update_from_simulation()

    # -- 2: unstack the mid cube ----------------------------------------------
    r = grasp_cube(env, planner, task, mid, stage="unstack mid")
    if r == -1:
        say(env, "MISSED: the mid cube never came off (physics committed)")
        return res
    res = r
    if common.stopped_by_horizon(planner):
        return res
    carry_out(env, planner, task, mid)
    tcp = task.agent.tcp.pose.sp
    planner.static_manipulation(
        sapien.Pose(p=[tcp.p[0] - 0.33, tcp.p[1], tcp.p[2]], q=tcp.q))
    planner.open_gripper(t=8)
    common.hold_object_in_planner(env, planner, task, task.cubes[mid],
                                  held=False, who=WHO)
    say(env, "stage mid: released over the counter")
    planner.planner.update_from_simulation()

    # -- 3: the target, off the home plinth -----------------------------------
    r = grasp_cube(env, planner, task, bottom, stage="take the target")
    if r == -1:
        say(env, "MISSED: the target never came off")
        return res
    res = r
    if common.stopped_by_horizon(planner):
        return res
    carry_out(env, planner, task, bottom)

    # -- 4: place the target on the counter below, let the scramble fire ------
    # The v0 place shape: back the base off to the reach band, place from
    # there. (The far-east place died of the east minefield — K109.)
    tgt = _np(task.place_target).reshape(-1)
    back_off(env, planner, task, -0.30, stage="to the place band")
    planner.planner.update_from_simulation()
    # The Burner counter-work posture: duck before the low work (sweep 12:
    # even the hover->descent's last centimetres die on `joint limit [3]`
    # from the tall carry posture; the restack fetches at the same height
    # live in the duck).
    plan_joints(env, planner, task, {"torso_lift_joint": 0.2},
                label="duck for the place")
    planner.planner.update_from_simulation()
    # The v0 place shape verbatim: HOVER over the point first, then the
    # short descent (sweep 10: a one-leg place from the carry is a 0.83 m
    # down-reach and dies on `joint limit [3]`).
    r2 = -1
    for hz, nx in ((1.10, 0.0), (1.16, 0.02), (1.06, -0.02)):
        hover = grasp_pose(task, [float(tgt[0]) + nx, float(tgt[1]), hz])
        r = planner.static_manipulation(hover)
        if r == -1:
            r = planner.static_manipulation(hover)
        if r == -1:
            continue
        goal = grasp_pose(task, [float(tgt[0]) + nx, float(tgt[1]),
                                 float(tgt[2]) + task.cfg.cube_half + 0.015
                                 + GRASP_Z_OFF])
        r2 = planner.static_manipulation(goal)
        if r2 == -1:
            r2 = planner.static_manipulation(goal)
        if r2 != -1:
            break
        say(env, "place rung refused; next hover rung", hz=hz)
    if r2 == -1:
        say(env, "MISSED: the place leg refused")
        return res
    res = r2
    planner.open_gripper(t=8)
    common.hold_object_in_planner(env, planner, task, task.cubes[bottom],
                                  held=False, who=WHO)
    tcp = task.agent.tcp.pose.sp
    planner.static_manipulation(
        sapien.Pose(p=[tcp.p[0], tcp.p[1] - 0.20, tcp.p[2]], q=tcp.q))
    r = planner.idle_steps(t=20)
    if r != -1:
        res = r
    info = res[-1] if res != -1 else {}
    scrambled = bool(_np(info.get("scrambled", torch.zeros(1))).reshape(-1)[0])
    say(env, "target placed", scrambled=scrambled)
    if not scrambled:
        say(env, "MISSED: the scramble never fired (target not accepted)")
        return res
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()

    # -- 5: back west, restack in the (remembered or guessed) order -----------
    # -- 5: restack from the west scramble stack — Hanoi on plinth pegs ------
    # The scramble parks the movables STACKED on the west plinth in a seeded
    # order; every move below is a measured-solid shelf form (W17 A/B grasps
    # at 1.49-1.56, D seats) from the single rest dock — the counter fetch
    # dance that flaked a per-binary coin across sweeps 10-14 is gone.
    fold_to_rest(env, planner, task)
    planner.planner.update_from_simulation()
    r = drive_dock(env, planner, task, hx, WORK_DOCK_Y,
                   stage="back to the shelf dock", tries=3)
    if r == -1:
        say(env, "MISSED: the shelf dock drive refused")
        return res
    res = r
    if common.stopped_by_horizon(planner):
        return res

    first, second = restack
    z1 = float(cube_p(task, first)[2])
    z2 = float(cube_p(task, second)[2])
    tx, ty = float(_np(task.plinths[2].pose.p).reshape(-1)[0]), \
        float(_np(task.plinths[2].pose.p).reshape(-1)[1])
    temp_top = float(_np(task.plinths[2].pose.p).reshape(-1)[2]) \
        + task.cfg.temp_col_half_h

    def correct_home_seat(ci):
        """One measured correction pass when the home seat lands off the
        plinth (sweeps 5-17: driven seats drift 45-75 mm; the predicate
        allows 35)."""
        off = float(np.linalg.norm(
            cube_p(task, ci)[:2] - np.array([hx, hy])))
        if off <= 0.030:
            return
        say(env, "home seat off; correction pass", off_mm=round(off * 1000, 1))
        rr = grasp_cube(env, planner, task, ci, stage="home correction")
        if rr == -1:
            return
        tcp = task.agent.tcp.pose.sp
        planner.static_manipulation(
            sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + 0.05], q=tcp.q))
        seat_cube(env, planner, task, ci, "plinth_0", hx, hy, htop,
                  stage="home correction")
        planner.planner.update_from_simulation()

    def move_cube(ci, support_name, x, y, z_top, *, stage):
        r = grasp_cube(env, planner, task, ci, stage=stage)
        if r == -1:
            return -1
        tcp = task.agent.tcp.pose.sp
        planner.static_manipulation(
            sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + 0.05], q=tcp.q))
        if not seat_cube(env, planner, task, ci, support_name, x, y, z_top,
                         stage=stage):
            return -1
        planner.planner.update_from_simulation()
        return 0

    if z1 > z2:
        # the remembered bottom is the scramble TOP: two moves
        say(env, "hanoi: two moves (m1 on top)")
        if move_cube(first, "plinth_0", hx, hy, htop,
                     stage="move m1 home") == -1:
            say(env, "MISSED: move m1 home")
            return res
        correct_home_seat(first)
        fp = cube_p(task, first)
        if move_cube(second, f"cube_{first}", float(fp[0]), float(fp[1]),
                     float(fp[2]) + task.cfg.cube_half,
                     stage="move m2 onto m1") == -1:
            say(env, "MISSED: move m2 onto m1")
            return res
    else:
        # the remembered bottom is UNDER: three moves via the tall temp peg
        say(env, "hanoi: three moves via the temp peg")
        if move_cube(second, "plinth_2", tx, ty, temp_top,
                     stage="park m2 on the temp") == -1:
            say(env, "MISSED: park m2")
            return res
        if move_cube(first, "plinth_0", hx, hy, htop,
                     stage="move m1 home") == -1:
            say(env, "MISSED: move m1 home")
            return res
        correct_home_seat(first)
        fp = cube_p(task, first)
        if move_cube(second, f"cube_{first}", float(fp[0]), float(fp[1]),
                     float(fp[2]) + task.cfg.cube_half,
                     stage="move m2 onto m1") == -1:
            say(env, "MISSED: move m2 onto m1")
            return res
    # Seat correction on the pair base: the driven seat can land off the
    # plinth (sweeps 5/9: 46-51 mm); one measured pass.
    off = float(np.linalg.norm(
        cube_p(task, first)[:2] - np.array([hx, hy])))
    if off > 0.030:
        say(env, "the pair sits off the plinth", off_mm=round(off * 1000, 1))
    if common.stopped_by_horizon(planner):
        return res

    r = planner.idle_steps(t=30)
    if r == -1:
        return fail(env, "settle wait")
    res = r
    info = res[-1]
    ok = bool(_np(info["success"]).reshape(-1)[0])
    say(env, "episode over", success=ok,
        pair_correct=bool(_np(info["pair_correct"]).reshape(-1)[0]),
        wrong_pair=bool(_np(info["wrong_pair"]).reshape(-1)[0]),
        on_home=bool(_np(info["on_home_plinth"]).reshape(-1)[0]),
        home_xy_mm=round(float(_np(info["home_xy_dist"]).reshape(-1)[0]) * 1000, 1))
    if not ok:
        say(env, "MISSED: the final predicate did not latch")
    return res


if __name__ == "__main__":
    import argparse

    import gymnasium as gym

    import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind", action="store_true")
    args = ap.parse_args()
    env = gym.make("MikasaStackRecall-v0", num_envs=1, robot_uids="mikasa_ds_fetch",
                   control_mode="pd_joint_pos", obs_mode="state", scene_idx=0,
                   sim_backend="cpu")
    out = solve(env, seed=args.seed, blind=args.blind)
    print("result:", "no plan (-1)" if out == -1 else out[-1].get("success"))
