"""Oracle for MikasaStationChecklist-v0, and its blind twin.

Written against `tools/stub_planner.py` on a Mac (mplib has no macOS wheel) and,
since 2026-08-18 (T4), run for real in the amd64 CPU container
(`tools/docker/planner.sh run …`) on kitchen 102 — the staging below is what the
container measured, not a guess. The shared stage kit is `oracle_common.py`.

Spec §6 asks for a privileged-access oracle that visits stations in a deterministic
order. Two things there are load-bearing and easy to get wrong:

- **Order by station index, never by distance.** "Go to the nearest one" makes the
  visit order a function of the layout, and the layout correlates with nothing the
  task wants to teach — the order would leak straight into the demonstration data
  (spec trap 3). Station indices are assigned alphabetically by fixture name in
  `station_candidates`, precisely so that iterating them means nothing spatial.
- **The environment plays the gesture, not the planner.** After a commitment the
  env owns the arm for `gesture_steps + settle_steps` steps and ignores what the
  planner sends. So the planner must idle through it, and must call
  `planner.planner.update_from_simulation()` afterwards — its internal copy of the
  world has been standing still while the real robot moved.

What the oracle does per station, in order (T4):

1. `drive_base(dock, target_view_vec=facing)` — the base to the station's dock,
   turned to face the fixture; if the direct drive is refused (on 102 the drawer
   dock sits 0.5 m from the left wall and the rest arm, leading the base by 0.5 m,
   would end inside it), **once more from a waypoint** diagonally behind the dock
   on the room side (`approach_waypoint`); then `parked at the dock d_dock=…
   dyaw_deg=… in_zone_any=…` in the trace (K26/D5, via `oracle_common.dock_error`).
2. **Ten settle steps, then read `budget_left`.** The commitment latches inside
   the env the moment base and TCP are both in place; at the dock the rest-pose
   TCP is already ~0.4 m from the service point and inside the z band for the
   counter-height stations, so the commit fires *on arrival* and the env takes
   the arm. That is the **drive-first** path: no arm plan at all, the oracle idles
   through gesture + settle + 5 and moves on. (The settle matters: the env-owned
   gesture zeroes the base and arm actions, and the solver must have finished
   the drive before it starts.)
3. Otherwise the **fallback**: one `static_manipulation` to
   `activation_pose_for(...)` — the TCP in free space *in front of* the fixture's
   face (0.25 m out from the service point along −across; +0.45 m above a top ≤
   1.0 m, −0.30 m below a taller one, K33), orientation kept (a translation-only
   move), torso frozen (`disable_lift_joint=True`) — then 45 idle steps and the
   budget again.
4. If nothing latched even there, the episode is returned as it stands — the
   last 5-tuple with a `MISSED: reached the pose and nothing latched d_tcp=… dz=…
   d_dock=… dyaw=… in_zone_any=…` line (D6/K33: a parking miss and an arm miss
   read differently); the sweep books it as `missed`. `-1` is reserved for a
   failed plan or drive. Every primitive is followed by
   `stopped_by_horizon(planner)` (K23).

`--blind` is the memory-free control. It services stations 0, 1, 2 without ever
reading the answer; everything else is identical. Two different floors follow, and
the spec conflates them:

    first decision   P(station 0 is a target) = C(5,3)/C(6,3) = 0.5
    episode success  P(exactly the right three) = 1/C(6,3) = 0.05

Quote both. An oracle at 90% episode success against a "0.5 floor" reads as a much
smaller achievement once the right floor, 5%, is on the page.

Contract, as `template_planner.py:32-45`: the solution owns the reset; return `-1`
on a failed plan and the gym 5-tuple otherwise; do not catch exceptions; do not
decide success. And do not call `env.evaluate()` — it mutates. Read `info`.
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import numpy as np
import sapien

from mani_skill.utils.wrappers import RecordEpisode

from utils.mikasa_oracle.planners import oracle_common as common
from utils.mikasa.seeding import seed_everything

# The mplib-dependent import stays deferred inside oracle_common's factory, so this
# module imports on a Mac and tests/test_station_checklist.py runs the *real*
# solve() against tools/stub_planner.py.

WHO = "station_checklist_planner"

# Free-space activation pose (K33): standoff from the service point along the
# outward normal of the fixture face, and the vertical offset by station height.
# +0.45 for low tops, not the planned +0.30: from the dock the rest TCP sits at
# z 1.249 (torso 0.386, frozen), and the drawer (top 0.47) at +0.30 = 0.77 m was
# refused — `joint limit at index [11]` (wrist_flex) / `IK Failed` (ladder run 4,
# journal 2026-08-18); +0.45 keeps the pose 0.10 m inside `gesture_z_high` (0.55)
# and 0.33 m closer to the rest height. The orientation is the TCP's *current*
# one (a translation-only screw, like `carry_pose`), not a fixed straight-down
# frame: the down frame added a ~70° wrist re-orientation to the same refusal.
ACTIVATION_AHEAD = 0.25
ACTIVATION_DZ_LOW_TOP = 0.45
ACTIVATION_DZ_HIGH_TOP = -0.30
ACTIVATION_HIGH_TOP_Z = 1.0

# Settle after the drive before reading whether the commit latched on arrival; the
# env-owned gesture zeroes base and arm, so the drive must be over first.
SETTLE_AFTER_DRIVE = 10


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _info_bool(info: dict, key: str, idx: int = 0) -> bool:
    return bool(_np(info[key]).reshape(-1)[idx])


def _info_int(info: dict, key: str, idx: int = 0) -> int:
    return int(_np(info[key]).reshape(-1)[idx])


def say(env, stage: str, **extra) -> None:
    """`oracle_common.say` tagged `[station_checklist_planner]` — stdout + a `phase` event."""
    common.say(env, WHO, stage, **extra)


def fail(env, stage: str, **extra):
    """Return -1 for a failed stage, saying which one — never a silent -1."""
    return common.fail(env, WHO, stage, **extra)


def default_planner_factory(env, debug: bool, vis: bool):
    """The real Fetch solver at the oracle refinement cap (60). Needs mplib."""
    return common.default_planner_factory(env, debug, vis)


def wait_cue(env, planner, info):
    """Idle until `elapsed_steps >= cue_steps` — `oracle_common.wait_cue`, `cfg.cue_steps` (K36).

    Nothing to steer: the env replaces the whole action while `t < cue_steps`,
    which is what makes the window exactly `cue_steps` long for every policy.
    """
    return common.wait_cue(env, planner, info, who=WHO)


def _facing_vec(task, s: int) -> np.ndarray:
    """Unit vector the base should face at station `s`, with z forced to zero.

    `rotate_base_z` asserts the direction is horizontal (extand.py) and an
    assertion failure is not a `-1`, so the zeroing is not cosmetic.
    """
    yaw = float(_np(task._dock_yaw)[0, s])
    v = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return v / np.linalg.norm(v)


def down_quat(across_out: np.ndarray) -> np.ndarray:
    """`(w, x, y, z)` of the TCP frame that points the Fetch gripper straight down.

    The ds_fetch TCP is `gripper_link`, whose +x is the finger direction (at the
    rest keyframe it points forward and 18° down; measured from the rest TCP
    quaternion on 102). Down = +x → −z world; +y (the closing axis, fingers open
    along it) = lateral to the fixture face; +z = x × y = back toward the robot.

    Args:
        across_out: unit outward normal of the fixture face (from the fixture
            toward the robot), z = 0.

    Example:
        >>> q = down_quat(np.array([0.0, -1.0, 0.0]))
        >>> R = sapien.Pose(q=q).to_transformation_matrix()[:3, :3]
        >>> np.round(R[:, 0], 6).tolist()   # +x of the TCP: straight down
        [0.0, 0.0, -1.0]
    """
    n = np.asarray(across_out, dtype=np.float64)
    n = np.array([n[0], n[1], 0.0])
    n = n / np.linalg.norm(n)
    x = np.array([0.0, 0.0, -1.0])
    facing = -n  # robot looks at the fixture
    y = np.array([facing[1], -facing[0], 0.0])  # lateral (robot's right)
    z = np.cross(x, y)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    return np.asarray(sapien.Pose(T).q, dtype=np.float64)


def activation_pose_for(
    act: np.ndarray,
    across: np.ndarray,
    top_z: float,
    q=None,
    *,
    ahead: float = ACTIVATION_AHEAD,
    dz_low_top: float = ACTIVATION_DZ_LOW_TOP,
    dz_high_top: float = ACTIVATION_DZ_HIGH_TOP,
    high_top_z: float = ACTIVATION_HIGH_TOP_Z,
) -> sapien.Pose:
    """TCP pose in free space in front of the station's face that satisfies `at_station`.

    Pure numpy (K33). The point is `act + across_out · ahead` — `ahead` metres
    out from the service point along the outward normal of the fixture face
    (`across_out = −across`; `act` itself is `beacon_inset` = 0.10 m *inside* the
    face, so 0.25 m out is 0.15 m of clear air in front of it — never inside the
    cabinet, which is what the old `_activation_pose` asked for). The height is
    `act_z + dz_low_top` for tops ≤ `high_top_z` and `act_z + dz_high_top` above:
    on 1.3–1.6 m tops a positive offset would put the TCP at 1.6–1.9 m with the
    torso frozen. Orientation `q`: pass the TCP's current quaternion (the oracle
    does — a translation-only move, measured reachable where the straight-down
    frame `down_quat` was refused; see the constants above); None = `down_quat`.

    By construction, against `StationChecklistTask.evaluate`'s `at_station`:
    `d_tcp = ahead = 0.25 ≤ gesture_xy_radius (0.45)` and
    `dz ∈ {+0.45, −0.30} ⊂ [gesture_z_low −0.35, gesture_z_high 0.55]`.

    Args:
        act: the station's service point (`task._act_pt[0, s]`), world xyz.
        across: the fixture's `across` unit vector (near edge → centre; from
            `fixture_frame`, cached as `task._station_across_np[0, s]`).
        top_z: the fixture's own top z (`task._station_top_z_np[0, s]`).
        q: TCP quaternion `(w, x, y, z)`; None = `down_quat(across_out)`.
        ahead, dz_low_top, dz_high_top, high_top_z: the constants above.

    Example:
        >>> p = activation_pose_for(np.array([0.5, -0.5, 0.47]), np.array([0.0, 1.0, 0.0]), 0.47, [1, 0, 0, 0])
        >>> np.round(p.p.astype(np.float64), 3).tolist()   # drawer, top 0.47: +0.45
        [0.5, -0.75, 0.92]
        >>> p = activation_pose_for(np.array([1.25, -0.475, 1.085]), np.array([0.0, 1.0, 0.0]), 1.085, [1, 0, 0, 0])
        >>> np.round(p.p.astype(np.float64), 3).tolist()   # sink, top 1.085: -0.30
        [1.25, -0.725, 0.785]
    """
    act = np.asarray(act, dtype=np.float64)
    across = np.asarray(across, dtype=np.float64)
    out = -np.array([across[0], across[1], 0.0])
    out = out / np.linalg.norm(out)
    dz = dz_low_top if float(top_z) <= high_top_z else dz_high_top
    p = act + out * ahead + np.array([0.0, 0.0, dz])
    if q is None:
        q = down_quat(out)
    return sapien.Pose(p=p, q=np.asarray(q, dtype=np.float64))


def _station_geometry(task, s: int):
    """`(act, across, top_z)` for station `s`, from the task's cached arrays."""
    act = _np(task._act_pt)[0, s].astype(np.float64)
    across = _np(task._station_across_np)[0, s].astype(np.float64)
    top_z = float(_np(task._station_top_z_np)[0, s])
    return act, across, top_z


def _tcp_error(task, act) -> tuple[float, float]:
    """`(d_tcp, dz)` of the TCP against a service point — the `at_station` numbers."""
    tcp = _np(task.agent.tcp.pose.p).reshape(-1, 3)[0]
    return float(np.linalg.norm(tcp[:2] - act[:2])), float(tcp[2] - act[2])


def choose_targets(task, info: dict, blind: bool, rng: np.random.Generator) -> list[int]:
    """The stations to service, sorted by index. The one read the blind arm replaces.

    The sighted arm reads `uncompleted_mask` — privileged, as a demonstrator may.
    The blind arm draws **a uniformly random k − m subset of the k stations** from
    `np.random.default_rng(seed)`, the convention burner_planner and
    season_dish_planner already follow, and changes nothing else, so the two arms
    differ only in memory.

    Until 2026-08-21 the blind branch returned `range(k − m)` — the *budget count*
    used where a list of station indices was meant. It yielded `[0, 1, 2]` on every
    seed, and nothing crashed because the list is the right length. Its numbers were
    therefore not an estimate of anything: over eval seeds 0–9 station 0 happens to
    be uncompleted on 7 and the answer is exactly {0,1,2} on 2, which is precisely
    the `first_decision_ok 7/10` and `success 2/10` that got published. **Those two
    numbers came from the constant arm and are not a floor estimate**; the arm below
    has not been swept yet (journal, 2026-08-21).

    Sorted order is kept for both arms: the sighted one services by ascending index,
    so a matched control must too. It costs nothing in floor terms — the answer set
    is drawn independently of the guess, so any particular station is uncompleted
    with probability (k − m)/k = 0.5 whatever order the guess is serviced in, and
    the episode still needs the whole set right, 1/C(6,3) = 5%.

    Example:
        >>> targets = choose_targets(task, info, blind=False, rng=rng)  # doctest: +SKIP
        >>> say(env, "targets chosen", targets=targets)                 # doctest: +SKIP
    """
    if blind:
        budget = task.cfg.k_stations - task.cfg.m_completed
        drawn = rng.choice(task.cfg.k_stations, size=budget, replace=False)
        return sorted(int(s) for s in drawn)
    uncompleted = _np(info["uncompleted_mask"]).reshape(-1)
    return sorted(int(s) for s in np.flatnonzero(uncompleted))


# Recovery approach (measured on 102, T4 ladder runs 2-4): the waypoint sits this
# far behind the dock (away from the fixture) and this far to the side, toward the
# other stations, so the last leg comes in at ~63° from the counter line. The rest
# arm's farthest link leads the base by ~0.61 m (run 2: the direct −x approach hit
# the wall with 0.152 of a 0.75 m twist left); at 45° (run 3, 0.6/0.6) the wrist
# still ended 0.07 m from the wall and was refused; at 63° it ends 0.23 m clear.
APPROACH_BACK = 0.8
APPROACH_SIDE = 0.4


def approach_waypoint(dock_xyz, yaw: float, room_centre_xy, *, back: float = APPROACH_BACK,
                      side: float = APPROACH_SIDE) -> np.ndarray:
    """A waypoint diagonally behind the dock, on the room side, for a second drive attempt.

    Why: `drive_base` turns toward the target and moves straight, and the Fetch's
    rest arm leads the base by ~0.5 m. On kitchen 102 the drawer dock (x = 0.5) is
    0.5 m from the left wall, so any approach along the counter (heading −x) ends
    with the gripper inside the wall — `move_base_forward`'s screw plan refuses:
    `collision …forearm_roll_link<->wall_left_room` (ladder run 2, journal
    2026-08-18). From `dock − back·face + side·lateral`, `lateral` pointing along
    the counter toward the room centre (the dock centroid), the last leg comes in
    at ~63°: the arm's farthest link (≈0.61 m ahead) ends ≈0.27 m short of the
    dock's x, i.e. clear of the wall, and the final turn to `face` swings the arm
    away from it (45° was not enough — the wrist ended 0.07 m from the wall).

    Args:
        dock_xyz: the dock `(x, y, z)`.
        yaw: the dock's yaw (the base faces `(cos, sin)`).
        room_centre_xy: where the other stations are, e.g. `task._room_centre`.
        back, side: metres behind / beside the dock.

    Returns:
        float64 `(x, y, 0)`.

    Example:
        >>> approach_waypoint(np.array([0.5, -1.4, 0.0]), np.pi / 2, np.array([2.234, -1.41])).round(3).tolist()
        [0.9, -2.2, 0.0]
        >>> approach_waypoint(np.array([3.868, -1.4, 0.0]), np.pi / 2, np.array([2.234, -1.41])).round(3).tolist()
        [3.468, -2.2, 0.0]
    """
    d = np.asarray(dock_xyz, dtype=np.float64)
    face = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    along = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    c = np.asarray(room_centre_xy, dtype=np.float64)
    sgn = 1.0 if float(np.dot(np.array([c[0] - d[0], c[1] - d[1], 0.0]), along)) >= 0 else -1.0
    w = np.array([d[0], d[1], 0.0]) - back * face + side * sgn * along
    return w


def service_station(env, planner, task, s: int, budget_before: int):
    """Drive to station `s`, commit on arrival or from the activation pose.

    Returns `(res, committed)`: `res` is `-1` on a failed drive/plan (already said
    why), else the last 5-tuple; `committed` is whether `budget_left` dropped. A
    horizon cut is returned as the truncated tuple (K23), `committed` as read.

    Example:
        >>> res, ok = service_station(env, planner, task, 3, budget_before=3)  # doctest: +SKIP
        >>> if res == -1: return res                                            # doctest: +SKIP
    """
    planner.planner.update_from_simulation()
    dock = _np(task._dock_pos)[0, s].astype(np.float64)
    yaw = float(_np(task._dock_yaw)[0, s])
    face = _facing_vec(task, s)
    name = task.station_names[s] if getattr(task, "station_names", None) else str(s)
    say(env, "drive to station dock", station=s, name=name, dock=[round(float(v), 3) for v in dock])
    res = planner.drive_base(target_pos=dock, target_view_vec=face)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if (
        res == -1
        and int(_np(task.budget_left).reshape(-1)[0]) >= budget_before
        and not common.stopped_by_horizon(planner)
    ):
        # Once more, from the room side (measured on 102: the direct approach to
        # the drawer dock 0.5 m from the left wall runs the rest arm into the
        # wall — `screw plan failed: collision …forearm_roll_link<->wall_left_room`).
        waypoint = approach_waypoint(dock, yaw, _np(task._room_centre))
        say(env, "drive refused; approaching from the room side", station=s,
            waypoint=[round(float(v), 3) for v in waypoint])
        planner.planner.update_from_simulation()
        res = planner.drive_base(target_pos=waypoint)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res != -1:
            planner.planner.update_from_simulation()
            res = planner.drive_base(target_pos=dock, target_view_vec=face)
            if res != -1 and common.stopped_by_horizon(planner):
                return res, False
    d_dock, dyaw = common.dock_error(task, (dock[0], dock[1], yaw))
    if res == -1:
        # The commit may have latched mid-drive (the base static and in the zone
        # before the final turn settled): the env then owns base and arm and the
        # solver's last refinement reads as a refusal. Privileged read of the
        # budget, not evaluate(): a decision was made, so this is not a no-plan.
        if int(_np(task.budget_left).reshape(-1)[0]) < budget_before:
            say(env, "commit latched during the drive; the env owns the arm", station=s,
                d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1))
            res = planner.idle_steps(t=task.cfg.gesture_steps + task.cfg.settle_steps + 5)
            if res != -1 and common.stopped_by_horizon(planner):
                return res, True
            if res == -1:
                return fail(env, "idle through the gesture", station=s), True
            planner.planner.update_from_simulation()
            return res, True
        return fail(env, "drive to station dock", station=s, d_dock=round(d_dock, 3),
                    dyaw_deg=round(dyaw, 1)), False
    planner.planner.update_from_simulation()

    # Settle, then read: did the commit fire on arrival (drive-first)?
    res = planner.idle_steps(t=SETTLE_AFTER_DRIVE)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "settle at the dock", station=s), False
    info = res[-1]
    in_zone_any = _info_bool(info, "in_zone_any")
    say(env, "parked at the dock", station=s, d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1),
        in_zone_any=in_zone_any)
    if _info_int(info, "budget_left") < budget_before:
        say(env, "committed on arrival (drive-first)", station=s)
        res = planner.idle_steps(t=task.cfg.gesture_steps + task.cfg.settle_steps + 5)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, True
        if res == -1:
            return fail(env, "idle through the gesture", station=s), True
        planner.planner.update_from_simulation()
        return res, True

    # Fallback: the TCP to free space in front of the face, torso frozen.
    act, across, top_z = _station_geometry(task, s)
    tcp_q = _np(task.agent.tcp.pose.q).reshape(-1, 4)[0].astype(np.float64)
    pose = activation_pose_for(act, across, top_z, tcp_q)
    say(env, "activation pose", station=s, target=[round(float(v), 3) for v in pose.p],
        top_z=round(top_z, 3))
    res = planner.static_manipulation(pose, disable_lift_joint=True)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        d_tcp, dz = _tcp_error(task, act)
        return fail(env, "activation pose unreachable", station=s, d_tcp=round(d_tcp, 3),
                    dz=round(dz, 3), d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1),
                    in_zone_any=in_zone_any), False
    res = planner.idle_steps(t=task.cfg.gesture_steps + task.cfg.settle_steps + 5)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "idle at the activation pose", station=s), False
    info = res[-1]
    planner.planner.update_from_simulation()
    if _info_int(info, "budget_left") < budget_before:
        say(env, "committed at the activation pose", station=s)
        return res, True
    # D6/K33: reached the pose and nothing latched — a physical miss, not a plan
    # failure. The five numbers tell a parking miss from an arm miss.
    d_tcp, dz = _tcp_error(task, act)
    say(env, "MISSED: reached the pose and nothing latched", station=s, d_tcp=round(d_tcp, 3),
        dz=round(dz, 3), d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1),
        in_zone_any=_info_bool(info, "in_zone_any"))
    return res, False


def solve(env, seed=None, debug=False, vis=False, blind=False, *, planner_factory=default_planner_factory):
    """Solve one episode. `-1` on a failed plan/drive, the gym 5-tuple otherwise.

    `planner_factory` exists for the offline test: it defaults to the real solver
    and `run.py`-style drivers never pass it.
    """
    obs, info = env.reset(seed=seed)
    # Seeds Python, numpy, torch AND mplib's C++ RNG (utils.mikasa/seeding.py).
    if seed is not None:
        seed_everything(seed)

    assert env.unwrapped.control_mode in (
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ), env.unwrapped.control_mode

    planner = planner_factory(env, debug, vis)
    task = env.unwrapped
    rng = np.random.default_rng(seed)

    # -- STAGE 0: sit through the encoding window ---------------------------------
    info = wait_cue(env, planner, info)
    if info == -1:
        return info

    # -- STAGE 1: the privileged read, and the one line the blind arm replaces ------
    targets = choose_targets(task, info, blind, rng)
    say(env, "targets chosen", targets=targets, blind=bool(blind))

    # -- STAGE 2: one station after another, by index -----------------------------
    res = None
    for s in targets:
        budget_before = _info_int(info, "budget_left")
        res, committed = service_station(env, planner, task, s, budget_before)
        if res == -1:
            return res
        if common.stopped_by_horizon(planner):
            say(env, "stopped by the horizon", station=s)
            return res
        info = res[-1]
        if not committed:
            # service_station already said MISSED with the diagnostic fields;
            # a silent continue would burn the budget on the remaining stations
            # and report a memory failure that never happened.
            return res
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--scene-idx", type=int, default=0)
    p.add_argument("--output-dir", default="videos/station_checklist")
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
        help="memory-free control: service stations 0,1,2 without reading the answer",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    env = gym.make(
        "MikasaStationChecklist-v0",
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
            "success:", _info_bool(info, "success"),
            "| first_decision_ok:", _info_bool(info, "first_decision_ok"),
            "| decisions:", _info_int(info, "decision_correct_n"),
            "/", _info_int(info, "decision_total_n"),
            "| double:", _info_int(info, "double_service_count"),
            "| omission:", _info_int(info, "omission_count"),
        )
    env.close()
    return res


if __name__ == "__main__":
    sys.exit(0 if main() != -1 else 1)
