"""Oracle for MikasaCabinetStow-v0 — the retrieval straight flow, run backwards.

`cabinet_retrieval_planner`'s straight flow (200/200 on 1100-1299 in both control
modes, 2026-09-09) carries a cup from the shelf to the counter. This carries one
from the counter to the shelf. It is deliberately built by REUSING that flow's
pieces — `side_grasp_pose`, `plan_joints`, `ready_by_line`, the measured docks and
`GRASP_LIFT` — rather than by re-deriving geometry: the two directions are the same
two places, so anything re-derived here would be a second copy of a measured
number, which is how the two drift apart.

Three things do NOT reverse, and they are the whole reason this is a separate file
rather than a flag on the other one:

1. **The torso rises with the cup in hand.** In retrieval the hand is empty at the
   raise (it grasps after driving in); here the cup is already held, so the raise is
   collision-checked against the cup's own hull and can refuse where the empty
   raise plans. Hence the rung ladder and the retreat-and-retry below.
2. **The loaded drive.** `drive_straight` plans nothing — it re-commands the arm at
   its measured qpos each step. All of its collision safety lives in
   `ready_by_line`'s drive pre-check, so the cup must be ATTACHED to the planning
   world (`hold_object_in_planner`) BEFORE the ready posture is chosen. Attaching
   after, as retrieval does, would check an empty hand and thread an 11.5 cm cup
   through the cabinet mouth on trust.
3. **The retreat comes before any torso motion.** Retrieval lifts the hand out of
   the cup with a 0.12 m torso rise before backing off; inside the cabinet the torso
   is already at its 0.386 stop, so there is no rise left. The withdrawal is instead
   a straight screw back along the approach axis — which is the exact reverse of the
   approach, so the pads leave the cup the way they came rather than dragging across
   it.

Contract: `solve(env, seed=None, debug=False, vis=False)` returning -1 (no plan) or
the gym 5-tuple, per `template_planner.py`. -1 means the oracle never got to attempt
the chore; a physical miss returns the last 5-tuple.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import sapien

from planners import cabinet_retrieval_planner as retr
from utils.mikasa_oracle.planners import oracle_common as common
from utils.mikasa.seeding import seed_everything

WHO = "cabinet_stow_planner"

#: Torso heights tried for the COUNTER grasp, in order. The counter top sits near
#: z = 0.90 and the side grasp lands ~0.078 m above the cup's base, so the reach is
#: low; these are retrieval's own place-descent rungs (its `(PLACE_TORSO, 0.10,
#: 0.20)`), which is the same reach measured from the same dock.
GRASP_TORSO_RUNGS = (0.0, 0.10, 0.20)

#: Torso heights tried for the loaded rise toward the shelf, in order. The first is
#: the joint's stop, which is what retrieval reaches the shelf from.
RISE_TORSO_RUNGS = (retr.READY_TORSO, 0.34, 0.30)

#: How far the cup rides ABOVE its final resting height while the base drives into
#: the cabinet. The shelf's own front edge is at z = 1.42 where the cabinet box
#: starts (y = -0.37): entering at the placement height would cross that edge with
#: only `PLACE_DROP` of clearance, so the flow enters high and descends inside. The
#: mirror of retrieval's `UNSHELVE_DZ`.
ENTRY_RISE = float(os.environ.get("MIKASA_STOW_ENTRY_RISE", "0.04"))

#: Steps idled after the entry joint line so the arm reaches its loaded steady state
#: before the sag is measured. Short: the PD error settles in a few control steps.
ENTRY_SETTLE_STEPS = int(os.environ.get("MIKASA_STOW_ENTRY_SETTLE", "6"))

#: Straight-up clearance taken off the counter right after the grasp, before
#: anything else moves. Retrieval's `UNSHELVE_DZ`, same job on the other plane.
LIFT_OFF_DZ = float(os.environ.get("MIKASA_STOW_LIFT_OFF", "0.03"))

#: Metres the TCP retreats along -y when a loaded torso rise refuses from where the
#: grasp left it. Not a first resort: it costs a screw and the rise usually plans.
PRE_RISE_BACK = float(os.environ.get("MIKASA_STOW_PRE_RISE_BACK", "0.12"))

#: Cup mesh bottom above `place_target[2]` when the pads part, metres. Retrieval's
#: `STRAIGHT_PLACE_DROP` is 0.005 on a counter; 8 mm here because the shelf is the
#: surface an under-shot drops the cup onto from inside a box.
PLACE_DROP = float(os.environ.get("MIKASA_STOW_PLACE_DROP", "0.008"))

#: Metres the TCP rises STRAIGHT UP off the released cup, before anything else
#: moves; the rungs are tried in order.
#:
#: Measured, not chosen (2026-09-10, seeds 0-9): the cup is 7.44 cm across and the
#: pads open to 10 cm, so a withdrawal that slides the fingers back along the
#: approach axis has 1.28 cm of clearance a side — and a screw is a least-norm
#: Jacobian walk, not a pure translation, so the small wrist rotation it carries
#: eats that. Measured on the first version, which withdrew along -y: the cup was
#: left tilted 86.4 deg (seed 8, lying on the shelf) and 6.8 deg (seed 6), clean
#: only on seed 7. The side grasp sits GRASP_LIFT (0.02) above the cup's mid, i.e.
#: 3.75 cm below its rim, so 6 cm of rise clears the rim by 2.25 cm and the pads
#: cannot touch the cup again whatever the base then does.
#:
#: This is retrieval's own lesson in the other direction ("the hand lifts off the
#: cup before the base moves", 2026-09-09 round 4, which took its 100/100 from
#: hollow to honest); there the lift was the torso's, which here is already at its
#: 0.386 stop, so the arm has to do it.
RELEASE_LIFT_RUNGS = tuple(
    float(v) for v in os.environ.get("MIKASA_STOW_RELEASE_LIFT", "0.06,0.045,0.03").split(",")
)

#: Fallback only: metres the TCP withdraws along the approach axis when NO rise
#: plans. Kept because a refused rise with no fallback would leave the pads
#: straddling the cup for the base's exit, which is worse.
WITHDRAW_BACK = float(os.environ.get("MIKASA_STOW_WITHDRAW", "0.12"))

#: Tilt (degrees from upright) above which the stowed cup is called knocked over in
#: the trace. Not a verdict: the family's `evaluate()` latches success once the five
#: place predicates hold (the base's chosen semantics), so this is the line a sweep
#: greps to tell an honest placement from a hollow one.
TILT_WARN_DEG = 15.0

#: How far the TCP may sit from the entry pose after the joint line before the flow
#: corrects it with a screw, metres.
#:
#: Measured 2026-09-10, and the reason this stage exists: a joint line lands in JOINT
#: space and refines there, so a heavy object leaves the arm sagging with a steady PD
#: error the delta controller cannot integrate away. At the entry the TCP came in
#: **4.4 cm low** with the 243 g prop against **2 mm** with the 8 g cup. The drive that
#: follows plans nothing, so an entry that low carries the object INTO the shelf's front
#: edge rather than over it: on prop seed 2 it arrived already tilted 21.6 deg in the
#: fingers and fell flat the moment they opened. A screw refines in TCP space and lands
#: to ~3 mm under the same load, so one corrective screw buys the clearance back.
ENTRY_SAG_TOL = float(os.environ.get("MIKASA_STOW_ENTRY_SAG_TOL", "0.005"))

#: Probe offset for the ready posture's placement pre-check: 5 mm BELOW the
#: placement, so a solution that only just reaches is refused before the drive.
#: Retrieval probes 15 mm deeper into the shelf for the same reason.
PLACE_PROBE_SLACK = (0.0, 0.0, -0.005)


def say(env, stage: str, **extra):
    """Trace one stage to stdout and to the episode log. See `oracle_common.say`."""
    return common.say(env, WHO, stage, **extra)


def fail(env, stage: str, **extra):
    """Print the refusal and return -1 — the no-plan sentinel of the contract."""
    return common.fail(env, WHO, stage, **extra)


def _np(x) -> np.ndarray:
    """A torch tensor or numpy array as numpy, on the host."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _b(info, key: str) -> bool:
    """One batched boolean out of `info`, for num_envs=1."""
    return bool(_np(info[key]).reshape(-1)[0])


def _needle(task) -> str:
    """The carried object's name in the planning world, for `common.touchable`.

    `oracle_common.touchable` matches by name substring, so the literal "cup" is
    wrong the moment the family carries anything else — the box variant names its
    actor "prop" and a hard-coded needle would silently make every `touchable`
    block a no-op, which shows up as a refused pre-check rather than as an error.

    Example:
        >>> with common.touchable(planner, _needle(task)):   # doctest: +SKIP
        ...     res = _straight_move(env, planner, up, stage="lift off")
    """
    return str(getattr(task.cfg, "object_name", "cup"))


def _held(task) -> dict:
    """Where the carried object is and how far it leans, for a stage trace.

    A tilt that appears between two stages names the leg that caused it; a tilt only
    read at the end names nothing.

    Example:
        >>> say(env, "lifted off", **_held(task))   # doctest: +SKIP
    """
    return {
        "obj": [round(float(v), 3) for v in _np(task.cup.pose.p).reshape(-1)[:3]],
        "tilt": round(common._tilt_deg(task.cup.pose.sp.q), 1),
    }


def _straight_move(env, planner, pose, *, stage: str, tries: int = 1):
    """`arm_move` restricted to the straight channels: the screw with the torso held,
    never an RRT path. -1 or the 5-tuple.

    Args:
        pose: the TCP target.
        stage: what to call this leg in the trace.

    Returns:
        -1 when nothing planned (nothing is stepped), else the gym 5-tuple.

    Example:
        >>> res = _straight_move(env, planner, up, stage="lift off the counter")  # doctest: +SKIP
    """
    return common.arm_move(env, planner, pose, who=WHO, stage=stage, tries=tries,
                           disable_lift_joint=True, max_knots=1, knot_refuse=True)


def place_pose_for(task, *, rise: float = 0.0, obj=None, target=None,
                   compensate_xy: bool = False):
    """The TCP pose that stands the HELD cup on `task.place_target`, `rise` metres high.

    The arithmetic is retrieval's descent (`take_and_place_straight`), unchanged: keep
    the hand's current orientation, snap x and y to the target, and lower z by however
    much the cup's LIVE mesh bottom overshoots the target plane. Reading the live mesh
    rather than a cached rest-lift is what makes it correct for a cup that is tilted in
    the fingers.

    Args:
        task: the unwrapped env.
        rise: metres above the final placement (the drive-in clearance).
        obj: the held actor; None = `task.cup`.
        target: the (x, y, z) the object's BASE must end on; None = `task.place_target`.
        compensate_xy: aim so the OBJECT lands on the target, not the TCP.

            Default False, which is this family's measured behaviour and correct for a
            cup grasped about its axis. It is wrong for anything that hangs off-axis in
            the fingers: measured 2026-09-10 on `MikasaDepthRecall-v1`, a prop aimed at
            a row slot came to rest 1.7-2.9 cm away from it, against a 3.5 cm slot
            tolerance — inside, but with almost nothing left. With this on, the live
            offset between the object and the TCP is subtracted from the goal, which is
            what `depth_recall_planner.seat_prop` has always done (W19).

    Returns:
        `sapien.Pose` for the TCP, or None when the cup has no collision mesh.

    Example:
        >>> place = place_pose_for(task)                      # doctest: +SKIP
        >>> entry = place_pose_for(task, rise=ENTRY_RISE)     # doctest: +SKIP
    """
    obj = task.cup if obj is None else obj
    mesh = obj.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        return None
    bottom = float(np.asarray(mesh.bounds)[0][2])
    target = (_np(task.place_target).reshape(-1, 3)[0] if target is None
              else np.asarray(target, dtype=np.float64).reshape(3))
    tcp = task.agent.tcp.pose.sp
    drop = bottom - (float(target[2]) + PLACE_DROP)
    x, y = float(target[0]), float(target[1])
    if compensate_xy:
        off = _np(obj.pose.p).reshape(-1)[:3] - np.asarray(tcp.p, dtype=np.float64).reshape(3)
        x -= float(off[0])
        y -= float(off[1])
    return sapien.Pose(p=[x, y, float(tcp.p[2]) - drop + float(rise)], q=tcp.q)


def grasp_off_the_counter(env, planner, task):
    """Torso down, posture by joint line, screw, close. `(res, ok)`.

    `ok` False with `res == -1` is a refusal; `ok` False with a 5-tuple is a miss.
    """
    # -- torso to the counter reach ------------------------------------------
    lowered = -1
    for torso in GRASP_TORSO_RUNGS:
        lowered = retr.plan_joints(env, planner, task, {"torso_lift_joint": torso},
                                   label=f"lower the torso to {torso}", line_only=True)
        if lowered != -1 and common.stopped_by_horizon(planner):
            return lowered, False
        if lowered != -1:
            say(env, "torso down", torso=torso)
            break
    if lowered == -1:
        return fail(env, "lower the torso to the counter", rungs=list(GRASP_TORSO_RUNGS)), False
    planner.planner.update_from_simulation()

    # -- the grasp, over the depth ladder ------------------------------------
    # The posture line is chosen with drive=0: the cup stands on the counter in
    # front of the dock, so unlike retrieval there is nothing to drive through
    # before the grasp. The screw pre-check is retrieval's own (15 mm deeper, cup
    # out of the planning world) and its defaults say so.
    res = -1
    for depth in retr.SIDE_GRASP_DEPTHS:
        grasp, pre = retr.side_grasp_pose(task, depth, lift=retr.GRASP_LIFT)
        if grasp is None:
            return fail(env, "grasp: the cup has no collision mesh"), False
        say(env, "grasp posture", depth=depth, pre=[round(float(v), 3) for v in pre.p])
        res = retr.ready_by_line(env, planner, task, pre, drive=0.0, grasp=grasp,
                                 label=f"grasp posture (depth {depth})",
                                 touch_needle=_needle(task))
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res == -1:
            planner.planner.update_from_simulation()
            continue
        planner.planner.update_from_simulation()

        say(env, "grasp stroke", depth=depth, grasp=[round(float(v), 3) for v in grasp.p])
        res = _straight_move(env, planner, grasp, stage=f"grasp (depth {depth})")
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res == -1:
            say(env, "grasp stroke refused", depth=depth)
            planner.planner.update_from_simulation()
            continue

        res = planner.close_gripper(t=12)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if bool(task.agent.is_grasping(task.cup).any()):
            say(env, "cup in the gripper", depth=depth, **_held(task))
            planner.planner.update_from_simulation()
            return res, True
        say(env, "close missed", depth=depth)
        planner.open_gripper()
        planner.planner.update_from_simulation()

    return fail(env, "grasp the cup off the counter",
                tried=list(retr.SIDE_GRASP_DEPTHS)), False


def rise_with_the_cup(env, planner, task):
    """The loaded torso rise toward the shelf: the rung ladder, then a retreat and
    the ladder again. -1 or the 5-tuple.

    This is the stage that has no counterpart in retrieval (there the raise happens
    with an empty hand), so it is also the one that carries the fallback.
    """
    for attempt in ("as grasped", "after a retreat"):
        for torso in RISE_TORSO_RUNGS:
            res = retr.plan_joints(env, planner, task, {"torso_lift_joint": torso},
                                   label=f"raise the torso to {torso}", line_only=True)
            if res != -1 and common.stopped_by_horizon(planner):
                return res
            if res != -1:
                say(env, "torso up", torso=torso, how=attempt)
                planner.planner.update_from_simulation()
                return res
        if attempt != "as grasped":
            break
        # Nothing planned from where the grasp left the hand. Withdraw along the
        # approach axis — south, away from the counter — and try the ladder again.
        tcp = task.agent.tcp.pose.sp
        back = sapien.Pose(p=[tcp.p[0], tcp.p[1] - PRE_RISE_BACK, tcp.p[2]], q=tcp.q)
        say(env, "the loaded rise refused; retreating first", back=PRE_RISE_BACK)
        moved = _straight_move(env, planner, back, stage="retreat before the rise", tries=2)
        if moved != -1 and common.stopped_by_horizon(planner):
            return moved
        if moved == -1:
            return fail(env, "retreat before the rise")
        planner.planner.update_from_simulation()
    return fail(env, "raise the torso with the cup", rungs=list(RISE_TORSO_RUNGS))


def stow_the_cup(env, planner, task):
    """Counter -> shelf, after the dock. `(res, done)` like retrieval's own stage
    function: `done=True` means the episode is over and `res` is what solve returns.
    """
    # -- 1. off the counter --------------------------------------------------
    res, ok = grasp_off_the_counter(env, planner, task)
    if not ok:
        return res, True

    # The cup joins the planning world NOW, before any posture is chosen: every
    # later pre-check, and the unplanned drive that rests on it, has to see it.
    common.hold_object_in_planner(env, planner, task, task.cup, held=True, who=WHO)

    tcp = task.agent.tcp.pose.sp
    up = sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + LIFT_OFF_DZ], q=tcp.q)
    res = _straight_move(env, planner, up, stage="lift off the counter")
    if res == -1:
        res = common.arm_move(env, planner, up, who=WHO,
                              stage="lift off the counter (any plan)", tries=2)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, True
    if res == -1:
        return fail(env, "lift the cup off the counter"), True
    if not _b(res[-1], "is_grasped"):
        say(env, "MISSED: the cup left the gripper at the lift-off")
        return res, True
    say(env, "lifted off the counter", **_held(task))
    planner.planner.update_from_simulation()

    # -- 2. up to the shelf's height, cup in hand ----------------------------
    res = rise_with_the_cup(env, planner, task)
    if res == -1:
        return res, True
    if common.stopped_by_horizon(planner):
        return res, True
    if not _b(res[-1], "is_grasped"):
        say(env, "MISSED: the cup left the gripper on the way up")
        return res, True
    say(env, "risen with the object", **_held(task))

    # -- 3. the posture the loaded drive ends in -----------------------------
    entry = place_pose_for(task, rise=ENTRY_RISE)
    if entry is None:
        return fail(env, "stow: the cup has no collision mesh"), True
    base_p = task.agent.base_link.pose.p
    drive = float(retr.WORK_DOCK_Y - _np(base_p).reshape(-1)[1])
    ready = sapien.Pose(p=[entry.p[0], entry.p[1] - drive, entry.p[2]], q=entry.q)
    say(env, "entry posture", entry=[round(float(v), 3) for v in entry.p],
        drive=round(drive, 3))
    # The probe is the DESCENT from where the drive ends, and the held cup stays in
    # the planning world for it (touch_needle=None) — it is exactly what must not
    # hit the shelf.
    res = retr.ready_by_line(env, planner, task, ready, drive=drive, grasp=entry,
                             label="entry posture", slack=PLACE_PROBE_SLACK,
                             touch_needle=None)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, True
    if res == -1:
        return fail(env, "entry posture at the dock"), True
    planner.planner.update_from_simulation()
    if not _b(res[-1], "is_grasped"):
        say(env, "MISSED: the cup left the gripper at the entry posture")
        return res, True
    say(env, "at the entry posture", **_held(task))

    # Let the arm come to rest under the load, then correct what the joint line could
    # not see (see ENTRY_SAG_TOL). Non-fatal: a refused correction still leaves a
    # usable entry, just a lower one.
    settle = planner.idle_steps(t=ENTRY_SETTLE_STEPS)
    if settle != -1:
        res = settle
        if common.stopped_by_horizon(planner):
            return res, True
    tcp_now = task.agent.tcp.pose.sp
    sag = float(ready.p[2]) - float(tcp_now.p[2])
    if abs(sag) > ENTRY_SAG_TOL:
        say(env, "the entry sagged under the load; correcting", sag=round(sag, 4),
            tcp_z=round(float(tcp_now.p[2]), 3), asked_z=round(float(ready.p[2]), 3))
        fix = _straight_move(env, planner, ready, stage="correct the entry height", tries=2)
        if fix != -1:
            res = fix
            if common.stopped_by_horizon(planner):
                return res, True
            planner.planner.update_from_simulation()
            say(env, "entry corrected",
                tcp_z=round(float(task.agent.tcp.pose.sp.p[2]), 3),
                left=round(float(ready.p[2]) - float(task.agent.tcp.pose.sp.p[2]), 4))
        else:
            say(env, "entry correction refused; driving in as we are")

    # -- 4. the loaded drive into the cabinet --------------------------------
    # `hold_pose=True` where the solver has it: the compliant hold lets a loaded arm
    # creep down (3.9 cm with the prop, see the kwarg's docstring), and this drive is
    # the one leg that carries the object past the shelf's edge. Probed rather than
    # assumed, so the stub and older trees still work.
    drive_kw = {}
    if "hold_pose" in inspect.signature(planner.drive_straight).parameters:
        drive_kw["hold_pose"] = True
    say(env, "drive into the cabinet, loaded", distance=round(drive, 3), **drive_kw)
    res = planner.drive_straight(drive, v=0.10, **drive_kw)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, True
    if res == -1:
        return fail(env, "drive into the cabinet"), True
    if not _b(res[-1], "is_grasped"):
        say(env, "MISSED: the cup left the gripper on the way in")
        return res, True
    planner.planner.update_from_simulation()
    say(env, "over the shelf", tcp=[round(float(v), 3) for v in task.agent.tcp.pose.sp.p],
        **_held(task))

    # -- 5. down onto the shelf ----------------------------------------------
    place = place_pose_for(task)
    if place is None:
        return fail(env, "stow: the cup has no collision mesh"), True
    say(env, "descend onto the shelf", place=[round(float(v), 3) for v in place.p])
    res = _straight_move(env, planner, place, stage="descend onto the shelf", tries=2)
    if res == -1:
        higher = sapien.Pose(p=[place.p[0], place.p[1], place.p[2] + 0.005], q=place.q)
        res = _straight_move(env, planner, higher, stage="descend onto the shelf (5 mm up)")
    if res == -1:
        res = common.arm_move(env, planner, place, who=WHO,
                              stage="descend onto the shelf (any plan)", tries=2)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, True
    if res == -1:
        return fail(env, "descend onto the shelf"), True
    planner.planner.update_from_simulation()

    # -- 6. release, withdraw along the approach, then the base --------------
    # Before the fingers part: where the object actually IS, and how far the arm
    # sagged under it. The two together separate "fell on the way down" from "fell
    # when let go", which want different cures.
    mesh_now = task.cup.get_first_collision_mesh(to_world_frame=True)
    tcp_now = task.agent.tcp.pose.sp
    say(env, "at the shelf, before the release",
        obj=[round(float(v), 3) for v in _np(task.cup.pose.p).reshape(-1)[:3]],
        tilt_deg=round(common._tilt_deg(task.cup.pose.sp.q), 1),
        bottom=round(float(np.asarray(mesh_now.bounds)[0][2]), 4) if mesh_now is not None else None,
        tcp_z=round(float(tcp_now.p[2]), 3), asked_z=round(float(place.p[2]), 3),
        sag=round(float(place.p[2]) - float(tcp_now.p[2]), 4),
        grasped=bool(task.agent.is_grasping(task.cup).any()))
    res = planner.open_gripper(t=retr.RELEASE_RAMP + 6, ramp=retr.RELEASE_RAMP)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, True
    common.hold_object_in_planner(env, planner, task, task.cup, held=False, who=WHO)
    say(env, "released", cup=[round(float(v), 3) for v in _np(task.cup.pose.p).reshape(-1)[:3]],
        tilt_deg=round(common._tilt_deg(task.cup.pose.sp.q), 1))
    planner.planner.update_from_simulation()

    # The hand comes STRAIGHT UP off the cup before anything else moves, and the
    # horizontal withdrawal is then the base's — a rigid translation of the whole
    # arm, which carries no wrist rotation and so cannot scrape the 1.28 cm of
    # clearance a side. See RELEASE_LIFT_RUNGS for the measurement that forced this.
    tcp = task.agent.tcp.pose.sp
    back = -1
    for dz in RELEASE_LIFT_RUNGS:
        with common.touchable(planner, _needle(task)):
            back = _straight_move(
                env, planner,
                sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + dz], q=tcp.q),
                stage=f"lift the hand off the cup ({dz} m)", tries=2)
        if back != -1 and common.stopped_by_horizon(planner):
            return back, True
        if back != -1:
            res = back
            say(env, "hand off the cup", rise=dz,
                cup=[round(float(v), 3) for v in _np(task.cup.pose.p).reshape(-1)[:3]],
                tilt_deg=round(common._tilt_deg(task.cup.pose.sp.q), 1))
            break
    if back == -1:
        # No rise planned. Sliding back along the approach axis is second best and
        # measurably risky (RELEASE_LIFT_RUNGS), but leaving the pads around the cup
        # for the base's exit is worse.
        say(env, "no rise planned; withdrawing along the approach instead")
        with common.touchable(planner, _needle(task)):
            back = _straight_move(
                env, planner,
                sapien.Pose(p=[place.p[0], place.p[1] - WITHDRAW_BACK, place.p[2]], q=place.q),
                stage="withdraw from the shelf", tries=2)
        if back != -1:
            res = back
            if common.stopped_by_horizon(planner):
                return res, True
        else:
            say(env, "withdraw REFUSED; backing the base out anyway")
    planner.planner.update_from_simulation()

    out = planner.drive_straight(-drive, v=0.10)
    if out != -1:
        res = out
        if common.stopped_by_horizon(planner):
            return res, True
    say(env, "backed out", cup=[round(float(v), 3) for v in _np(task.cup.pose.p).reshape(-1)[:3]],
        tilt_deg=round(common._tilt_deg(task.cup.pose.sp.q), 1))
    return res, False


def solve(env, seed=None, debug=False, vis=False,
          planner_factory=common.default_planner_factory):
    """Stand the cup from the counter on the open cabinet's shelf.

    Contract: -1 when a plan or a grasp refused (nothing attempted), otherwise the
    last gym 5-tuple. The caller reads success from `info`.

    Example:
        >>> res = solve(env, seed=0)   # doctest: +SKIP
    """
    obs, info = env.reset(seed=seed)
    if seed is not None:
        seed_everything(seed)
    assert env.unwrapped.control_mode in (
        "pd_joint_pos", "pd_joint_pos_vel", "pd_joint_delta_pos"
    ), env.unwrapped.control_mode
    task = env.unwrapped
    planner = planner_factory(env, debug, vis)

    cup_p = _np(task.cup.pose.p).reshape(-1)[:3]
    say(env, "episode", cup=[round(float(v), 3) for v in cup_p],
        target=[round(float(v), 3) for v in _np(task.place_target).reshape(-1)[:3]],
        door_rad=float(task.cfg.door_open_rad))

    # Duck for the drive — retrieval's stage 1, unchanged.
    res = retr.plan_joints(env, planner, task,
                           {"torso_lift_joint": retr.TORSO_DRIVE, "wrist_flex_joint": 1.7},
                           label="duck the torso")
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "duck the torso for the drive")
    planner.planner.update_from_simulation()

    # A closed door is retrieval's measured opening stage; this variant starts open,
    # so the branch is here for the closed sibling rather than for today.
    if float(task.cfg.door_open_rad) < retr.ARM_PASS_RAD:
        opened = retr.open_the_door(env, planner, task)
        if opened == -1:
            return opened
        if retr.door_rad_now(task) < retr.ARM_PASS_RAD:
            say(env, "MISSED: the door did not open enough")
            return opened
        planner.planner.update_from_simulation()

    dock = np.array([float(cup_p[0]), retr.READY_DOCK_Y, 0.0])
    say(env, "dock at the counter", dock=[round(float(v), 3) for v in dock])
    res = planner.drive_base(target_pos=dock, target_view_vec=np.array([0.0, 1.0, 0.0]))
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "drive to the dock")
    planner.planner.update_from_simulation()
    d_dock, dyaw = common.dock_error(task, (float(dock[0]), float(dock[1]), np.pi / 2))
    say(env, "parked", d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1))

    res, done = stow_the_cup(env, planner, task)
    if done:
        return res

    settled = planner.idle_steps(t=retr.SETTLE_STEPS)
    if settled == -1:
        return fail(env, "settle wait")
    res = settled
    info = res[-1]
    tilt = common._tilt_deg(task.cup.pose.sp.q)
    if tilt > TILT_WARN_DEG:
        # The family's evaluate() latches success at the moment the place predicates
        # hold, so a cup knocked over afterwards still reads SUCCESS. Say so loudly:
        # a sweep that greps this line is what tells an honest 200/200 from a hollow
        # one (retrieval, 2026-09-09).
        say(env, "KNOCKED OVER after the place", tilt_deg=round(tilt, 1),
            cup=[round(float(v), 3) for v in _np(task.cup.pose.p).reshape(-1)[:3]])
    if _b(info, "success"):
        say(env, "episode over", success=True, tilt_deg=round(tilt, 1),
            steps=int(getattr(planner, "elapsed_steps", -1)))
    else:
        say(env, "MISSED: the cup did not end standing on the shelf",
            on_target=_b(info, "on_counter"), settled=_b(info, "settled"),
            above_floor=_b(info, "above_floor"),
            xy=round(float(_np(info["xy_distance"]).reshape(-1)[0]), 3),
            height=round(float(_np(info["height_above"]).reshape(-1)[0]), 3))
    return res


if __name__ == "__main__":
    import argparse

    import gymnasium as gym

    import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    e = gym.make("MikasaCabinetStow-v0", num_envs=1, robot_uids="mikasa_ds_fetch",
                 control_mode="pd_joint_pos", obs_mode="state", sim_backend="cpu")
    out = solve(e, seed=args.seed)
    print("no_plan" if isinstance(out, int) else f"success={_b(out[-1], 'success')}")
    e.close()
