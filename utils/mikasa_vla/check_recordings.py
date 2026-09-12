"""The pre-collection check of native `pd_joint_delta_pos` recordings (the supervisor's note,
2026-09-08), plus the replay that proves the recorded actions ARE the episode.

    python tools/vla/check_recordings.py <dir> [--limit 5] [--no-replay]

For every `seed_*/<stamp>.h5` under <dir>:
  1. the action array is (T, 13) and every entry is within [-1, 1] (a forgotten absolute
     slot shows up here as a value over 1) and, when the replay below builds an env, within
     the bounds that env itself declares;
  2. the stream is in the env's UNITS: the arm channels reach at least `--min-arm-span` of
     their bound and the gripper reaches ±1. A stream recorded in raw radians instead of the
     normalised action that went into `env.step` is ten times smaller, sits comfortably
     inside [-1, 1] and passes check 1 in silence — see `ARM_SPAN_MIN`;
  3. the recording carries its own `sim_config.control_freq`, so a converter cannot stamp a
     default rate onto a dataset nobody recorded at it;
  4. `obs/agent/qpos` is (T+1, 15);
  5. an rgb recording carries exactly the two robot-mounted cameras and no other;
     a recorded episode is a success (both checkable off, see --allow-failures);
  6. the share of steps with an arm channel at the clip (|a| > 0.999) is printed — nonzero
     means the arm was more than one controller step (0.1 rad) from its target on those
     steps (measured 2026-09-08: the PD lag, not the plan's step, see docs/vla-data.md);
  7. the recorded actions are fed back into a fresh env with the same seed, the same
     control mode and `physx_cpu`, and the episode's success must come back, with the
     final `qpos` within `--tol` of the recording (bit-for-bit is what one expects on the
     same machine; the tolerance is for a different build of PhysX).

Run from `mikasa/` of the tree that made the recordings (`python -m` / cwd on sys.path).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.getcwd())

ACTION_DIM, STATE_DIM = 13, 15

#: What an UNMODIFIED Fetch could hold, per `qpos` column: the three arm roll joints and
#: the virtual base spin. Our URDF widens the rolls to +-6.28 (mplib refuses a
#: `continuous` joint) and the base spin to +-50, so a recording can legally contain a
#: posture that upstream ManiSkill's own Fetch cannot reach — replay it there and the
#: delta controller clamps the target at the limit and the arm ends up somewhere else.
#: This is a property of the recording worth printing, not a defect in it: the check
#: does not fail on it, and `--stock-strict` is for when the caller wants it to.
STOCK_LIMITS = {2: ("root_z_rotation", 6.28), 8: ("upperarm_roll", 3.141),
                10: ("forearm_roll", 3.141), 12: ("wrist_roll", 3.141)}


#: How much of its own range a NATIVE stream uses, and why that is a check.
#:
#: The note's own bounds assert (`|a| <= 1`) was written against a forgotten absolute slot —
#: a value too BIG. It cannot see the opposite mistake, a stream recorded without the
#: normalisation ManiSkill does itself (the note's "Что просьба не трогать": no hand
#: scaling, no raw radians "alongside, just in case"). Raw radians hold the same motion in
#: numbers ten times SMALLER, every one of them inside [-1, 1], and the check passes in
#: silence. The replay catches it — but only when there is an env to replay in, which is
#: not the case for a foreign recording or under `--no-replay`.
#:
#: What separates the two, measured 2026-09-12 over 402 of our recordings (SeasonDish
#: 3800-3999, Retrieval 1100-1299, gate5): the arm channels reach at least **0.741** of
#: their bound (p10 0.766, max 0.821) and the gripper reaches exactly 1.000 in every single
#: one of them. A raw-radian stream would reach about 0.1 on the arm and 0.05 on the
#: gripper. The thresholds sit a factor of three below what we measure and a factor of
#: three above what the mistake produces; `--min-arm-span` moves the first one for a task
#: whose oracle really does creep.
ARM_SPAN_MIN, GRIPPER_SPAN_MIN = 0.25, 0.5


def past_stock_limits(q: np.ndarray) -> list:
    """`[(joint, peak, limit), ...]` for the columns an unmodified Fetch could not hold."""
    out = []
    for col, (name, lim) in STOCK_LIMITS.items():
        if q.shape[1] > col:
            peak = float(np.abs(q[:, col]).max())
            if peak > lim:
                out.append((name, round(peak, 3), lim))
    return out


def has_episode(path: str) -> bool:
    """Is there an episode in this recording at all?

    A seed the sweep LOST leaves the directory behind with an 800-byte h5 and a json whose
    `episodes` is `[]` — the sweep writes the file and keeps no trajectory (the owner's rule:
    a failed episode never reaches the data). That is not a bad recording, it is the absence
    of one. Until 2026-09-12 the check read `episodes[0]` of it and died with an IndexError,
    which meant a pool with a single lost seed could not be checked at all (found on
    DepthRecall's 199 of 200, 2026-09-11).

    Example:
        >>> has_episode("pool10/seed_158/20260911_160418.h5")      # doctest: +SKIP
        False
    """
    try:
        with open(path.replace(".h5", ".json")) as fh:
            return bool(json.load(fh).get("episodes"))
    except (OSError, ValueError, KeyError):
        return False


#: What an rgb recording must carry and nothing else: the two cameras that sit ON the
#: robot (2026-09-10, the owner: "камеру не пишем - только те, что на роботе сидят").
#: A world-anchored camera in this set is a task that still declares one in
#: `_default_sensor_configs`; a missing one is a recording made with
#: MIKASA_SHOULDER_CAMERAS=1 or an older agent.
ROBOT_CAMERAS = {"fetch_head", "fetch_hand"}


def qpos_of(traj) -> np.ndarray:
    """`obs/agent/qpos` (rgb / dict observations) or the robot's qpos out of the
    recorded env states (a flat `state` observation has no such key)."""
    obs = traj["obs"]
    if isinstance(obs, h5py.Group) and "agent" in obs:
        return np.array(obs["agent"]["qpos"], dtype=np.float64)
    art = traj["env_states"]["articulations"]
    key = next(k for k in art.keys() if "fetch" in k)
    st = np.array(art[key], dtype=np.float64)           # pose 7 + vel 6 + qpos + qvel
    nq = (st.shape[1] - 13) // 2
    return st[:, 13:13 + nq]


def one(path: str, *, replay: bool, tol: float, obs_mode: str | None,
        require_success: bool = True, min_arm_span: float = ARM_SPAN_MIN) -> dict:
    with open(path.replace(".h5", ".json")) as fh:
        meta = json.load(fh)
    kw = dict(meta["env_info"]["env_kwargs"]); env_id = meta["env_info"]["env_id"]
    ep = meta["episodes"][0]
    h = h5py.File(path, "r"); t = h["traj_0"]
    a = np.array(t["actions"], dtype=np.float64)
    q = qpos_of(t)
    out = dict(file=os.path.relpath(path), env=env_id, seed=ep.get("episode_seed"), steps=int(a.shape[0]),
               control_mode=kw.get("control_mode"), success=bool(np.array(t["success"])[-1]))
    problems = []
    if a.shape[1:] != (ACTION_DIM,):
        problems.append(f"actions {a.shape}, want (T, {ACTION_DIM})")
    if np.abs(a).max() > 1.0 + 1e-6:
        over = np.where(np.abs(a).max(0) > 1.0 + 1e-6)[0].tolist()
        problems.append(f"|a| max {np.abs(a).max():.3f} on channels {over}")
    if q.shape[1:] != (STATE_DIM,):
        problems.append(f"qpos {q.shape}, want (T+1, {STATE_DIM})")
    if kw.get("control_mode") != "pd_joint_delta_pos":
        problems.append(f"control_mode {kw.get('control_mode')!r}")
    # The units, which the bounds check above cannot see (see ARM_SPAN_MIN).
    out["arm_span"] = round(float(np.abs(a[:, :7]).max()), 3) if a.shape[1] >= 7 else None
    out["gripper_span"] = round(float(np.abs(a[:, 7]).max()), 3) if a.shape[1] >= 8 else None
    if out["arm_span"] is not None and out["arm_span"] < min_arm_span:
        problems.append(
            f"the arm channels reach only {out['arm_span']} of their bound (native recordings: "
            f"0.741 at the lowest over 402 of ours) — this looks like physical units (radians) "
            f"rather than the array that went into env.step. Nothing here scales an action: the "
            f"env's own space is [-1, 1] (normalize_action=True) and the recording holds what was "
            f"handed to it")
    # The gripper's span only says anything when the gripper is USED. A task that grasps
    # nothing — a drive, a push, a door opened with the wrist — holds that channel at one
    # value for the whole episode, and a constant channel means "unused", not "raw units"
    # (the peer session's catch, 2026-09-12). It is reported either way as a zero-variance
    # channel, and the arm's span carries the check.
    grip_used = out["gripper_span"] is not None and float(np.std(a[:, 7])) > 0.0
    if grip_used and out["gripper_span"] < GRIPPER_SPAN_MIN:
        problems.append(
            f"the gripper channel moves but reaches only {out['gripper_span']}; it is exactly 1.0 "
            f"in every native recording (the note: index 7 takes only ±1)")
    # The rate has to be IN the recording: `h5_to_lerobot` falls back to 20 Hz when it is
    # not, and the dataset then claims a rate nobody recorded — MIKASA-Robo's own trap,
    # where the metadata says 10 and the step is 50 ms (2026-09-10).
    out["control_freq"] = (kw.get("sim_config") or {}).get("control_freq")
    if out["control_freq"] is None:
        problems.append("no sim_config.control_freq in the recording; a converter would stamp "
                        "its own default and the dataset would claim a rate nobody recorded")
    out["arm_clip_frac"] = round(float((np.abs(a[:, :7]) > 0.999).any(1).mean()), 3) if a.shape[1] >= 7 else None
    out["body_clip_frac"] = round(float((np.abs(a[:, 8:11]) > 0.999).any(1).mean()), 3) if a.shape[1] >= 11 else None
    out["zero_var_channels"] = [int(i) for i in range(min(ACTION_DIM, a.shape[1])) if a[:, i].std() < 1e-12]
    out["past_stock_limits"] = past_stock_limits(q)
    # Only rgb recordings carry pixels; a state recording has no sensor_data and this
    # says nothing about it either way.
    obs = t["obs"]
    cams = set(obs["sensor_data"].keys()) if isinstance(obs, h5py.Group) and "sensor_data" in obs else None
    out["cameras"] = sorted(cams) if cams is not None else None
    if cams is not None and cams != ROBOT_CAMERAS:
        problems.append(f"cameras {sorted(cams)}, want {sorted(ROBOT_CAMERAS)}")
    if require_success and not out["success"]:
        problems.append("a failed episode in the dataset (the sweep should have dropped it)")
    if replay and not problems:
        import gymnasium as gym
        import torch
        import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)
        kw.pop("render_mode", None); kw.pop("human_render_camera_configs", None)
        kw["sim_backend"] = "physx_cpu"; kw["num_envs"] = 1
        if obs_mode:
            kw["obs_mode"] = obs_mode
        env = gym.make(env_id, **kw)
        # The bounds the ENV declares, instead of the [-1, 1] assumed above. The note's
        # assert stands in for this check because a file on its own cannot know the space;
        # once there is an env, ask it. (Ours is [-1, 1] on all 13, so this agrees with the
        # assert — it is the foreign recording, or a mode with other bounds, that it is for.)
        space = getattr(env, "action_space", None)
        lo = np.asarray(getattr(space, "low", []), dtype=np.float64).reshape(-1)
        hi = np.asarray(getattr(space, "high", []), dtype=np.float64).reshape(-1)
        if lo.shape == a.shape[1:] and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            out["action_space"] = [round(float(lo.min()), 3), round(float(hi.max()), 3)]
            outside = np.where((a < lo - 1e-6).any(0) | (a > hi + 1e-6).any(0))[0].tolist()
            if outside:
                problems.append(f"actions leave the env's own action space on channels {outside}")
        # A plain `reset(seed)` on the freshly built env — exactly what the oracle did — and
        # NOT `options=dict(reconfigure=True)`: a reconfigure re-runs `_load_scene`, where
        # RoboCasa's `get_fixture` draws the station counter from the episode RNG, so the
        # same seed lands the condiments on another counter segment (measured 2026-09-09,
        # SeasonDish 3608: shaker x 2.142 with reset(seed), 2.386 with reconfigure — the
        # replay then grasps air and "fails"). The canonical layout of a seed is the one
        # built at construction plus reset(seed).
        env.reset(seed=int(ep["episode_seed"]))
        u = env.unwrapped
        succ = False
        for i in range(a.shape[0]):
            _, _, _, _, info = env.step(torch.as_tensor(a[i], dtype=torch.float32)[None])
            succ = succ or bool(np.asarray(info["success"]).reshape(-1)[0])
        q_end = u.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float64)
        out["replay_success"] = succ
        out["replay_qpos_err"] = round(float(np.abs(q_end - q[a.shape[0]]).max()), 5)
        if succ != out["success"]:
            problems.append(f"replay success {succ} vs recorded {out['success']}")
        if out["replay_qpos_err"] > tol:
            problems.append(f"final qpos differs by {out['replay_qpos_err']} > {tol}")
        env.close()
    out["problems"] = problems
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--glob", default="seed_*/*.h5")
    ap.add_argument("--limit", type=int, default=5,
                    help="check this many recordings, spread evenly over the whole "
                         "directory rather than taken off the front — the head of a "
                         "sweep is its lowest seeds, and a fault that starts later "
                         "would never be sampled. 0 checks every file")
    ap.add_argument("--allow-failures", action="store_true",
                    help="do not treat a recorded episode whose success is False as a "
                         "problem. Off by default: the sweep drops failures, so one in "
                         "the file means the gate leaked")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-3, help="max |qpos| difference at the end of the replay (measured 1e-3 on the same machine)")
    ap.add_argument("--replay-obs-mode", default="state", help="obs mode for the replay env (state is faster; the episode is the same)")
    ap.add_argument("--stock-strict", action="store_true",
                    help="fail a recording that holds a posture an unmodified Fetch could "
                         "not (see STOCK_LIMITS); off by default, it is reported either way")
    ap.add_argument("--min-arm-span", type=float, default=ARM_SPAN_MIN,
                    help="the arm channels must reach this much of their bound, or the stream "
                         "is reported as raw units rather than the normalised action that went "
                         "into env.step (measured: 0.741 at the lowest over 402 native "
                         "recordings; a raw-radian stream reaches about 0.1)")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not files and args.glob == "seed_*/*.h5":
        # `--traj-dir <dir>` writes `<dir>/<stamp>.h5`; the per-seed layout only appears
        # when the sweep is given `--traj-dir <dir>/seed_<N>` (what the pools do). Accept
        # both rather than making the caller know which one they have.
        files = sorted(glob.glob(os.path.join(args.dir, "*.h5"))) \
            or sorted(glob.glob(os.path.join(args.dir, "*", "*.h5")))
    lost = [f for f in files if not has_episode(f)]
    if lost:
        # A lost seed is not a recording to check; say which ones and carry on.
        files = [f for f in files if f not in set(lost)]
        print(f"skipping {len(lost)} recording(s) with no episode (lost seeds): "
              + ", ".join(os.path.basename(os.path.dirname(f)) for f in lost[:5])
              + (" ..." if len(lost) > 5 else ""), flush=True)
    if args.limit and len(files) > args.limit:
        # evenly spaced over the sorted list, ends included
        step = (len(files) - 1) / (args.limit - 1) if args.limit > 1 else 0
        files = [files[round(i * step)] for i in range(args.limit)]
    if not files:
        raise SystemExit(f"no recordings under {args.dir}/{args.glob}")
    bad = stock = 0
    for f in files:
        # Every file here is a recording: the lost seeds were filtered out above by
        # `has_episode`, which is the one place that knows what "lost" looks like.
        r = one(f, replay=not args.no_replay, tol=args.tol, obs_mode=args.replay_obs_mode,
                require_success=not args.allow_failures, min_arm_span=args.min_arm_span)
        past = r["past_stock_limits"]
        if past and args.stock_strict:
            r["problems"].append("past the stock Fetch's limits: "
                                 + ", ".join(f"{n} {v} > {lim}" for n, v, lim in past))
        flag = "OK " if not r["problems"] else "BAD"
        bad += bool(r["problems"])
        stock += bool(past)
        print(f"{flag} {r['env']} seed={r['seed']} steps={r['steps']} success={r['success']} "
              f"arm_clip={r['arm_clip_frac']} arm_span={r['arm_span']} hz={r['control_freq']} "
              f"body_clip={r['body_clip_frac']} zero_var={r['zero_var_channels']}"
              + (f" cameras={r['cameras']}" if r["cameras"] else "")
              + (f" replay_success={r['replay_success']} qpos_err={r['replay_qpos_err']}" if "replay_success" in r else "")
              + (" stock_fetch=ok" if not past
                 else " stock_fetch=" + ",".join(f"{n}:{v}" for n, v, _ in past))
              + (f"  <- {'; '.join(r['problems'])}" if r["problems"] else ""), flush=True)
    n = len(files)
    # "within the stock Fetch's limits", not "replays on an unmodified Fetch": the stock
    # check reads joint limits only, and a replay of a ds_fetch recording on the stock
    # Fetch does drift (its hand's mass sits elsewhere — measured 2026-09-11).
    print(f"{n - bad} of {n} recordings pass"
          + (f"; {n - stock} of {n} within the stock Fetch's limits" if n else "")
          + (f"; {len(lost)} lost seed(s) without a recording" if lost else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
