"""The scripted oracle for MikasaDepthRecall-v0 (design blank H).

The route, leg by leg — every form is a K109/W18/W19 measurement or a
measured-adjacent transfer (the staging-line seats themselves were never
probed in isolation; they are validated in vivo by the K112 sweeps):

1. dock at the row column (rest posture drives — K109);
2. disassemble FRONT-first (physics refuses any deeper grasp behind an
   occupied slot — W18 B1, W19 D1/D2) and STAGE EACH PROP ON THE SHELF
   ITSELF — there is no teleport erasure any more (the user's call): the
   front prop is seated at the staging line's DEEP point (west of the
   row, y -0.145), the middle prop at its SOUTH point (y -0.20). The
   y-line composes with physics as a LIFO stack: seating deep-first is
   the only order the corridors allow, and fetching south-first on the
   way back is the only order they allow again — which is exactly the
   restore's own mid-slot-first requirement;
3. the target (deep) prop: carry out, back the base off to the reach band,
   duck (the K109 counter posture), hover-ladder place on `place_target`
   with LIVE in-hand compensation, release, retreat;
4. fold (TCP screw to the rest point + a branch line to the rest config,
   roll joints wrapped toward rest), dock at the row ONCE, and restore:
   fetch the SOUTH staged prop, seat it at the row's MIDDLE slot; fetch
   the (now unblocked) DEEP staged prop, seat it at the FRONT slot — all
   legs straight-frame from the single sx dock;
5. the blind arm (the design blank's control) assigns the fetched props
   to slots by a coin instead of the remembered order; a wrong coin walks
   into the physics wall (the second seat lands behind an occupied slot
   and refuses) — the honest 0.5x floor.

Tall-prop deltas from the K109 cube forms (measured, W19):
- the grasp aims at the prop's TOP BAND (centre +0.045/0.055): the CENTRE
  grasp is measured dead at depth (the palm behind the pads arrives where
  the prop's upper 6 cm stands — W19/W19b), and every manipulation leg
  masks the torso (`disable_lift_joint=True` — the screws die on the
  torso limit [3] otherwise);
- the top-band hold hangs the prop ~44 mm under the TCP (W19b) — the K109
  pendulum compensation machinery absorbs it, fed by live reads;
- a tall prop released from the hover lands UPRIGHT (aspect 12/4.5
  survives a ~24 mm fall — W19 A2).

Interpreter/runner: container only (mplib) — see AGENTS.md.

Run:
  tools/docker/planner.sh run --timeout 1800 python -m utils.mikasa_oracle.evaluate_planner \\
      -s MikasaDepthRecall-v0 -p depth_recall_planner -n 10 --start-seed 0 \\
      --scene-idx 0 --obs-mode state --sim-backend cpu --log-dir logs/depth-recall
"""

from __future__ import annotations

import numpy as np
import sapien
import torch

from utils.mikasa_oracle.planners import oracle_common as common
from utils.mikasa.seeding import seed_everything

WHO = "depth_recall"

GRASP_TRIES = 3
LIFT_DZ = 0.03
RETRACT_REACH = 0.55
WORK_DOCK_Y = -1.05
"The measured shelf work dock (W13/W17/K109)."
GRASP_Z_LADDER = (0.045, 0.055, 0.035)
"""TOP-BAND rungs over the prop centre (the prop top is +0.060): W19
measured the CENTRE grasp dead at the -0.145/-0.09 slots (`joint limit
[3]` in the screw's last centimetres — the palm behind the pads arrives
where the prop's upper 6 cm stands), and W19b measured the top band
alive: +0.045/+0.055 held at -0.145, all of +0.030/0.045/0.055 at -0.09.
The in-hand hang that comes with a top-band hold (-44 mm z, +10 mm y —
W19b TS) is the K109 pendulum in miniature; the live compensation in the
seat already absorbs it (TS landed 1.6 mm off aim)."""
PRE_LIFTS = (0.0, 0.05, 0.08)
SEAT_LADDER = ((0.004, -0.10, 0.0), (0.012, -0.12, 0.05), (0.020, -0.16, 0.0))
"(drop, approach, pre-lift) rungs; a refused descent falls to the next gap."
STAGE_DX = -0.13
"""The staging line's x offset west of the row column: the measured motor
band (W19c fetch alive at the 13-16 cm west-of-dock offset; the closed
left door's finger wall sits at ~2.33, and sx_min - 0.13 = 2.33
exactly)."""
STAGE_Y_DEEP = -0.145
STAGE_Y_SOUTH = -0.20
"""The staging line's two y points, same 5.5 cm pitch as the row: the
DEEP point is seated first (its corridor needs the south point empty),
the SOUTH point second — and fetched back first, by the same physics
(W19 D2: any reach behind an occupied slot refuses)."""
REST_TORSO = common.REST_TORSO
"The rest keyframe's torso lift (the duck is 0.2); one number with the promoted fold."


# The dock-hop phantom saga (solos 13-18) lives in docs/lab-journal.md
# K111: base drives to a second dock die on planning-world phantoms
# (`forearm <-> hingerightdoor/microwave` with the base a metre away; the
# ACM unblocks the rotate sweep but not the screw check). The teleport-free
# route needs no second dock at all — the whole staging line and the row
# share the single sx dock.


def say(env, stage: str, **extra) -> None:
    print(f"[solve] {stage}" + (f" {extra}" if extra else ""))


def fail(env, stage: str, **extra):
    say(env, f"FAILED: {stage}", **extra)
    return -1


def _np(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def prop_p(task, pi) -> np.ndarray:
    return _np(task.props[pi].pose.p).reshape(-1)


def held(task, pi) -> bool:
    return bool(_np(task.agent.is_grasping(task.props[pi])).reshape(-1)[0])


def grasp_pose(task, centre):
    """The W13/W17 side grasp: approach +y, closing x. (The 10-degree
    yawed variant that revived the abandoned east staging leg lives in
    tools/probes/w19h_yawed_east.py and the K111 journal entry — the
    teleport-free route is all straight-frame and does not need it.)"""
    return task.agent.build_grasp_pose(
        np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        np.asarray(centre, dtype=np.float64))


def grasp_prop(env, planner, task, pi, *, stage: str, pre_offset: float = 0.15):
    """The K109 retry-ladder grasp at the prop's TOP BAND (GRASP_Z_LADDER
    over the centre), W19 tall-prop form: pose re-read per attempt, lifted
    pres on retries.

    Returns the executor result, or -1 when every attempt missed."""
    for attempt in range(GRASP_TRIES):
        if common.stopped_by_horizon(planner):
            break  # let the caller's horizon check classify it (K23)
        planner.planner.update_from_simulation()
        p = prop_p(task, pi)
        goal = grasp_pose(task, [p[0], p[1],
                                 p[2] + GRASP_Z_LADDER[attempt]])
        pre = goal * sapien.Pose([0, 0, -float(pre_offset)])
        if PRE_LIFTS[attempt]:
            pre = sapien.Pose(p=[pre.p[0], pre.p[1],
                                 pre.p[2] + PRE_LIFTS[attempt]], q=pre.q)
        r = planner.static_manipulation(pre, disable_lift_joint=True)
        if r == -1:
            r = planner.static_manipulation(pre, disable_lift_joint=True)
        if r == -1:
            say(env, f"{stage}: pre-grasp refused", attempt=attempt)
            continue
        r = planner.static_manipulation(goal, disable_lift_joint=True)
        if r == -1:
            say(env, f"{stage}: grasp leg refused", attempt=attempt)
            continue
        # verify the REACH before committing the close: solo 3 measured an
        # RRT goal leg landing 38 mm off (reached=False) and the blind close
        # SHOVING the prop over — every later attempt then aims at a tipped
        # prop. A missed reach is a failed attempt, not a close.
        tcp_now = np.asarray(task.agent.tcp.pose.sp.p,
                             dtype=np.float64).reshape(-1)
        reach_err = float(np.linalg.norm(tcp_now - np.asarray(goal.p)))
        if reach_err > 0.020:
            say(env, f"{stage}: goal leg missed by "
                     f"{round(reach_err * 1000, 1)} mm — not closing",
                attempt=attempt)
            continue
        res = planner.close_gripper(t=12)
        if held(task, pi):
            common.hold_object_in_planner(env, planner, task, task.props[pi],
                                          held=True, who=WHO)
            say(env, f"{stage}: prop in the gripper", prop=pi, attempt=attempt)
            return res
        say(env, f"{stage}: close missed", attempt=attempt)
        planner.open_gripper(t=6)
    return -1


def carry_out(env, planner, task):
    """The W13 exit with the prop attached: +0.03 lift, retract by reach.
    Non-fatal legs (a refused retract leaves the arm where it is)."""
    tcp = task.agent.tcp.pose.sp
    planner.static_manipulation(
        sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + LIFT_DZ], q=tcp.q),
        disable_lift_joint=True)
    base_y = float(_np(task.agent.robot.get_qpos()).reshape(-1)[1])
    tcp = task.agent.tcp.pose.sp
    planner.static_manipulation(
        sapien.Pose(p=[tcp.p[0], base_y + RETRACT_REACH, tcp.p[2]], q=tcp.q),
        disable_lift_joint=True)


def seat_prop(env, planner, task, pi, x, y, base_z, *, stage: str):
    """The W19 bare-shelf seat: aim the prop's CENTRE at
    base_z + prop_half_h + drop with LIVE in-hand compensation, +20 mm
    hover, fresh re-read, PURE-VERTICAL descent; a refused descent falls
    through to the next rung's bigger gap, and after the last rung the prop
    goes from the hover (W19 A2: a tall prop lands UPRIGHT from ~24 mm,
    21.5 mm off aim — inside the 35 mm slot tolerance). No support stroke —
    the shelf is scene, not an actor. Returns True when released over the
    slot, False when no hover ever planned (the prop stays in hand)."""
    half_h = float(task.cfg.prop_half_h)
    hovered = False
    last_hover = None
    for drop, app, plift in SEAT_LADDER:
        if common.stopped_by_horizon(planner):
            break
        tcp_now = task.agent.tcp.pose.sp
        off = prop_p(task, pi) - np.asarray(tcp_now.p,
                                            dtype=np.float64).reshape(-1)
        goal_hi = grasp_pose(task, [x - float(off[0]), y - float(off[1]),
                                    base_z + half_h + drop + 0.020
                                    - float(off[2])])
        pre = goal_hi * sapien.Pose([0, 0, app])
        if plift:
            pre = sapien.Pose(p=[pre.p[0], pre.p[1], pre.p[2] + plift],
                              q=pre.q)
        planner.static_manipulation(pre, disable_lift_joint=True)
        r2 = planner.static_manipulation(goal_hi, disable_lift_joint=True)
        if r2 == -1:
            r2 = planner.static_manipulation(goal_hi, disable_lift_joint=True)
        if r2 == -1:
            say(env, f"{stage}: hover refused", drop=drop)
            continue
        hovered = True
        last_hover = goal_hi
        tcp_now = task.agent.tcp.pose.sp
        off = prop_p(task, pi) - np.asarray(tcp_now.p,
                                            dtype=np.float64).reshape(-1)
        goal_lo = grasp_pose(task, [x - float(off[0]), y - float(off[1]),
                                    base_z + half_h + drop - float(off[2])])
        r3 = planner.static_manipulation(goal_lo, disable_lift_joint=True)
        if r3 == -1:
            r3 = planner.static_manipulation(goal_lo, disable_lift_joint=True)
        if r3 == -1:
            say(env, f"{stage}: descent refused, next rung", drop=drop)
            continue
        break
    if not hovered:
        say(env, f"{stage}: seat REFUSED (no hover)")
        return False
    # the fallback release must happen OVER the slot, not wherever the last
    # refused rung's pre left the arm (review corner): re-plan the last
    # good hover; if even that refuses now, release anyway and say so
    tcp_now = np.asarray(task.agent.tcp.pose.sp.p, dtype=np.float64).reshape(-1)
    if float(np.linalg.norm(tcp_now - np.asarray(last_hover.p))) > 0.03:
        r = planner.static_manipulation(last_hover, disable_lift_joint=True)
        if r == -1:
            r = planner.static_manipulation(last_hover, disable_lift_joint=True)
        if r == -1:
            say(env, f"{stage}: hover re-plan refused before the fallback "
                     f"release — releasing off-hover")
    c0 = prop_p(task, pi)
    planner.open_gripper(t=8)
    common.hold_object_in_planner(env, planner, task, task.props[pi],
                                  held=False, who=WHO)
    tcp = task.agent.tcp.pose.sp
    planner.static_manipulation(
        sapien.Pose(p=[tcp.p[0], tcp.p[1] - 0.20, tcp.p[2]], q=tcp.q),
        disable_lift_joint=True)
    tcp = task.agent.tcp.pose.sp
    planner.static_manipulation(
        sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + 0.05], q=tcp.q),
        disable_lift_joint=True)
    c2 = prop_p(task, pi)
    landed = float(np.hypot(c2[0] - x, c2[1] - y))
    if landed > 0.020:
        say(env, f"{stage}: seat landed off aim",
            off_mm=round(landed * 1000, 1),
            at_open=[round(float(v), 3) for v in c0[:2]])
    say(env, f"{stage}: seat done", prop=pi)
    return True


def plan_joints(env, planner, task, targets: dict, *, label: str, tries: int = 2):
    """`oracle_common.plan_joints` under this oracle's tag — the copy that lived
    here was promoted (it was the third). The K111 caveat travels with it: the
    RRT branch is a measured no-op; only the line branch executes.

    Example:
        >>> res = plan_joints(env, planner, task, {"torso_lift_joint": 0.2},
        ...                   label="duck")                       # doctest: +SKIP
    """
    return common.plan_joints(env, planner, task, targets, label=label,
                              tries=tries, who=WHO)


def fold_to_rest(env, planner, task):
    """Fold the arm to the PURE rest pose before a drive (K106/K109: the
    rest TCP rides the 1.08-1.37 free corridor; the duck poisons rotate
    sweeps). Non-fatal."""
    rest_q = np.asarray(task.agent.keyframes["rest"].qpos,
                        dtype=np.float64).reshape(-1)
    jm = task.agent.robot.active_joints_map
    arm_names = list(task.agent.controller.controllers["arm"].config.joint_names)
    targets = {n: float(rest_q[jm[n].active_index[0].item()]) for n in arm_names}
    targets["torso_lift_joint"] = float(
        rest_q[jm["torso_lift_joint"].active_index[0].item()])
    return plan_joints(env, planner, task, targets, label="fold to rest",
                       tries=3)


def fold_via_tcp(env, planner, task, rest_tcp, *, stage: str):
    """`oracle_common.fold_via_tcp` under this oracle's tag and rest torso —
    promoted from here (K111: the only fold verified end to end). Non-fatal.

    Example:
        >>> fold_via_tcp(env, planner, task, rest_tcp, stage="fold")  # doctest: +SKIP
    """
    return common.fold_via_tcp(env, planner, task, rest_tcp, stage=stage,
                               who=WHO, rest_torso=REST_TORSO)


def normalize_continuous_arm_joints(env, planner, task):
    """`oracle_common.normalize_continuous_arm_joints` under this oracle's tag —
    promoted from here (the K55 precedent, arm edition; solo 9/19).

    Example:
        >>> normalize_continuous_arm_joints(env, planner, task)   # doctest: +SKIP
    """
    return common.normalize_continuous_arm_joints(env, planner, task, who=WHO)


def normalize_base_yaw(env, planner, task):
    """Moved to `oracle_common.normalize_base_yaw` on 2026-09-07 (shared with
    CabinetSearch); kept here by name for the probes that call `drp.normalize_base_yaw`.
    Same body, same `[solve]` line."""
    common.normalize_base_yaw(env, planner, task, say=say)


def drive_dock(env, planner, task, x, y, *, stage: str, tries=2):
    """drive_base with the arm frozen and the K109 ARRIVED-despite-refusal
    acceptance (the rotate sweep hallucination is filed for its own probe)."""
    for t in range(tries):
        normalize_base_yaw(env, planner, task)
        planner.planner.update_from_simulation()
        r = planner.drive_base(target_pos=np.array([x, y, 0.0]),
                               target_view_vec=np.array([0.0, 1.0, 0.0]),
                               freeze_arm=True)
        if r != -1:
            normalize_base_yaw(env, planner, task)
            return r
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


def back_off(env, planner, task, delta, *, stage: str):
    """A CHECKED, arm-frozen backward screw (the BASE_PLAN_MASK disease —
    K109 sweep 1)."""
    cur = task.agent.base_link.pose.sp
    direction = cur.to_transformation_matrix()[:3, 0]
    direction[2] = 0.0
    target = np.asarray(cur.p, dtype=np.float64) + direction * float(delta)
    r = planner.move_base_forward(target, n_init_qpos=100, freeze_arm=True)
    say(env, f"{stage}: back-off {'ok' if r != -1 else 'REFUSED'}",
        delta=delta)
    return r


def solve(env, seed=None, debug=False, vis=False, blind=False,
          planner_factory=common.default_planner_factory):
    """Solve one episode. `-1` on a pre-commit refusal, the gym 5-tuple
    otherwise.

    Args:
        env: the (possibly wrapped) MikasaDepthRecall-v0 env.
        seed: episode seed; the solution owns the reset.
        debug, vis: passed to the solver factory.
        blind: the design blank's control arm — the restore's slot
            assignment is a coin instead of the remembered one; everything
            else is identical. Must land at 0.5 x the sighted rate.
        planner_factory: seam for `tools/stub_planner.py`.

    Returns:
        -1, or the last gym 5-tuple.

    Example:
        >>> res = solve(env, seed=0)                              # doctest: +SKIP
    """
    obs, info = env.reset(seed=seed)
    if seed is not None:
        seed_everything(seed)
    assert env.unwrapped.control_mode in ("pd_joint_pos", "pd_joint_pos_vel", "pd_joint_delta_pos"), \
        env.unwrapped.control_mode
    task = env.unwrapped
    try:
        # The W19 probes measured every leg with max_refine_steps=200 —
        # their screws finish at goal_error 0.000 (refine 3-21 steps). The
        # factory default of 60 left the oracle's legs at refine=0 and
        # goal_error 8-14 mm, which is exactly a pad-width of shove on a
        # 45 mm prop (solo 2-5: the close nudges the prop out and misses).
        planner = planner_factory(env, debug, vis, max_refine_steps=200)
    except TypeError:
        planner = planner_factory(env, debug, vis)  # the stub seam

    # The answer: which movable stood at which slot. Privileged but honest —
    # the disassembly itself shows the honest agent the row order (front
    # comes off first, middle second); the occlusion hides it only BEFORE
    # the work starts. The blind arm swaps the assignment on a coin.
    slots = [int(v) for v in _np(task.original_slots).reshape(-1).tolist()]
    front, mid, deep = slots
    if blind:
        # the coin swaps WHICH slot each fetched prop goes to; the FETCH
        # order stays physical (south staged first — anything else is
        # blocked), so a wrong coin seats the south prop at the FRONT slot
        # and then walks into the wall: the mid seat behind an occupied
        # front slot refuses (W19 D2's physics) — the honest 0.5x failure
        flip = bool(np.random.randint(0, 2))
        assign = ((mid, 0), (front, 1)) if flip else ((mid, 1), (front, 0))
        say(env, "BLIND: slot assignment is a coin", flip=flip)
    else:
        assign = ((mid, 1), (front, 0))
    say(env, "episode", slots=slots, blind=bool(blind))

    sx = float(_np(task.place_target).reshape(-1)[0])
    ys = task.cfg.row_slots_y
    shelf = float(task.cfg.shelf_top_z)
    # the rest TCP in the base frame, while the arm still sits on the reset
    # keyframe — the fold target for fold_via_tcp
    rest_tcp = task.agent.base_link.pose.sp.inv() * task.agent.tcp.pose.sp

    # -- 0: dock at the row column --------------------------------------------
    r = drive_dock(env, planner, task, sx, WORK_DOCK_Y, stage="dock at the row")
    if r == -1:
        return fail(env, "drive to the work dock")
    res = r

    # -- 1: the front prop off the row ----------------------------------------
    # NO fold before the row grasps: solos 2-6 measured the disassembly
    # clean (attempt 0 on all three) with every mid-episode fold refusing —
    # i.e. with no fold at all — and solo 7 measured a WORKING un-duck
    # shifting the RNG stream and flaking the target grasp (the K72
    # per-draw coin). The one place the arm genuinely arrives contorted is
    # after the place — the restore loop folds there.
    r = grasp_prop(env, planner, task, front, stage="clear front")
    if r == -1:
        if common.stopped_by_horizon(planner):
            return res
        return fail(env, "grasp the front prop")
    res = r
    if common.stopped_by_horizon(planner):
        return res
    carry_out(env, planner, task)
    # the robot stages the prop ITSELF (no teleport erasure): the front
    # prop goes to the staging line's DEEP point while its south point is
    # still empty — the only order the corridors allow
    ok = seat_prop(env, planner, task, front, sx + STAGE_DX, STAGE_Y_DEEP,
                   shelf, stage="stage front (deep point)")
    if not ok:
        say(env, "MISSED: the front staging seat refused with the prop "
                 "in hand")
        return res
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()

    # -- 2: the middle prop ----------------------------------------------------
    r = grasp_prop(env, planner, task, mid, stage="clear middle")
    if r == -1:
        say(env, "MISSED: the middle prop never came off (physics committed)")
        return res
    res = r
    if common.stopped_by_horizon(planner):
        return res
    carry_out(env, planner, task)
    ok = seat_prop(env, planner, task, mid, sx + STAGE_DX, STAGE_Y_SOUTH,
                   shelf, stage="stage middle (south point)")
    if not ok:
        say(env, "MISSED: the middle staging seat refused with the prop "
                 "in hand")
        return res
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()

    # -- 3: the target ---------------------------------------------------------
    r = grasp_prop(env, planner, task, deep, stage="take the target")
    if r == -1:
        say(env, "MISSED: the target never came off")
        return res
    res = r
    if common.stopped_by_horizon(planner):
        return res
    carry_out(env, planner, task)

    # -- 4: place the target on the counter ------------------------------------
    tgt = _np(task.place_target).reshape(-1)
    half_h = float(task.cfg.prop_half_h)
    back_off(env, planner, task, -0.30, stage="to the place band")
    planner.planner.update_from_simulation()
    plan_joints(env, planner, task, {"torso_lift_joint": 0.2},
                label="duck for the place")
    planner.planner.update_from_simulation()
    r2 = -1
    for hz, nx in ((1.10, 0.0), (1.16, 0.02), (1.06, -0.02)):
        hover = grasp_pose(task, [float(tgt[0]) + nx, float(tgt[1]), hz])
        r = planner.static_manipulation(hover, disable_lift_joint=True)
        if r == -1:
            r = planner.static_manipulation(hover, disable_lift_joint=True)
        if r == -1:
            continue
        # LIVE in-hand compensation at the hover (W19 C: the tall prop rides
        # millimetres off the TCP in xy — the read costs nothing)
        tcp_now = task.agent.tcp.pose.sp
        off = prop_p(task, deep) - np.asarray(tcp_now.p,
                                              dtype=np.float64).reshape(-1)
        goal = grasp_pose(task, [float(tgt[0]) + nx - float(off[0]),
                                 float(tgt[1]) - float(off[1]),
                                 float(tgt[2]) + half_h + 0.015
                                 - float(off[2])])
        r2 = planner.static_manipulation(goal, disable_lift_joint=True)
        if r2 == -1:
            r2 = planner.static_manipulation(goal, disable_lift_joint=True)
        if r2 != -1:
            break
        say(env, "place rung refused; next hover rung", hz=hz)
    if r2 == -1:
        say(env, "MISSED: the place leg refused")
        return res
    res = r2
    planner.open_gripper(t=8)
    common.hold_object_in_planner(env, planner, task, task.props[deep],
                                  held=False, who=WHO)
    tcp = task.agent.tcp.pose.sp
    planner.static_manipulation(
        sapien.Pose(p=[tcp.p[0], tcp.p[1] - 0.20, tcp.p[2]], q=tcp.q),
        disable_lift_joint=True)
    r = planner.idle_steps(t=20)
    if r != -1:
        res = r
    info = res[-1] if res != -1 else {}
    placed = bool(_np(info.get("target_placed",
                                torch.zeros(1))).reshape(-1)[0])
    say(env, "target placed", accepted=placed)
    if not placed:
        say(env, "MISSED: the place was not accepted")
        return res
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()

    # -- 5: restore, all straight-frame from the single sx dock ----------------
    # The fetch order is physical, not chosen: the south staged prop (the
    # middle one) unblocks first, and its slot (mid) must be seated first
    # anyway — the LIFO line and the row's physics agree. `assign` carries
    # the MEMORY (or the blind coin): which fetched prop belongs to which
    # slot.
    first_leg = True
    for pi, slot_k in assign:
        fold_via_tcp(env, planner, task, rest_tcp, stage="fold")
        planner.planner.update_from_simulation()
        if first_leg:
            r = drive_dock(env, planner, task, sx, WORK_DOCK_Y,
                           stage="back to the row dock", tries=3)
            if r == -1:
                say(env, "MISSED: the return dock drive refused")
                return res
            res = r
            first_leg = False
            if common.stopped_by_horizon(planner):
                return res
        say(env, f"PRE-FETCH DIAG prop {pi}",
            base=[round(float(v), 3)
                  for v in _np(task.agent.robot.get_qpos()).reshape(-1)[:3]],
            staged=[round(float(v), 3) for v in prop_p(task, pi)])
        r = grasp_prop(env, planner, task, pi,
                       stage=f"fetch staged prop {pi}")
        if r == -1:
            say(env, f"MISSED: staged prop {pi} never came up")
            return res
        res = r
        if common.stopped_by_horizon(planner):
            return res
        carry_out(env, planner, task)
        ok = seat_prop(env, planner, task, pi, sx, float(ys[slot_k]), shelf,
                       stage=f"seat slot {slot_k}")
        if not ok:
            say(env, f"MISSED: the slot-{slot_k} seat refused with the prop "
                     f"in hand")
            return res
        if common.stopped_by_horizon(planner):
            return res
        planner.planner.update_from_simulation()

    # -- 6: settle and read the verdict ----------------------------------------
    r = planner.idle_steps(t=40)
    if r != -1:
        res = r
    info = res[-1] if res != -1 else {}
    say(env, "verdict",
        success=bool(_np(info.get("success", False)).reshape(-1)[0]),
        front_right=bool(_np(info.get("front_right", False)).reshape(-1)[0]),
        mid_right=bool(_np(info.get("mid_right", False)).reshape(-1)[0]),
        wrong_assign=bool(_np(info.get("wrong_assign", False)).reshape(-1)[0]))
    return res
