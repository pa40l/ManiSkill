"""The pre-collection check of native `pd_joint_delta_pos` recordings (the supervisor's note,
2026-09-08), plus the replay that proves the recorded actions ARE the episode.

    python tools/vla/check_recordings.py <dir> [--limit 5] [--no-replay]

For every `seed_*/<stamp>.h5` under <dir>:
  1. the action array is (T, 13) and every entry is within [-1, 1] (a forgotten absolute
     slot shows up here as a value over 1);
  2. `obs/agent/qpos` is (T+1, 15);
  3. the share of steps with an arm channel at the clip (|a| > 0.999) is printed — nonzero
     means the arm was more than one controller step (0.1 rad) from its target on those
     steps (measured 2026-09-08: the PD lag, not the plan's step, see docs/vla-data.md);
  4. the recorded actions are fed back into a fresh env with the same seed, the same
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


def one(path: str, *, replay: bool, tol: float, obs_mode: str | None) -> dict:
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
    out["arm_clip_frac"] = round(float((np.abs(a[:, :7]) > 0.999).any(1).mean()), 3) if a.shape[1] >= 7 else None
    out["body_clip_frac"] = round(float((np.abs(a[:, 8:11]) > 0.999).any(1).mean()), 3) if a.shape[1] >= 11 else None
    out["zero_var_channels"] = [int(i) for i in range(min(ACTION_DIM, a.shape[1])) if a[:, i].std() < 1e-12]
    if replay and not problems:
        import gymnasium as gym
        import torch
        import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)
        kw.pop("render_mode", None); kw.pop("human_render_camera_configs", None)
        kw["sim_backend"] = "physx_cpu"; kw["num_envs"] = 1
        if obs_mode:
            kw["obs_mode"] = obs_mode
        env = gym.make(env_id, **kw)
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
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-3, help="max |qpos| difference at the end of the replay (measured 1e-3 on the same machine)")
    ap.add_argument("--replay-obs-mode", default="state", help="obs mode for the replay env (state is faster; the episode is the same)")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not files and args.glob == "seed_*/*.h5":
        # `--traj-dir <dir>` writes `<dir>/<stamp>.h5`; the per-seed layout only appears
        # when the sweep is given `--traj-dir <dir>/seed_<N>` (what the pools do). Accept
        # both rather than making the caller know which one they have.
        files = sorted(glob.glob(os.path.join(args.dir, "*.h5"))) \
            or sorted(glob.glob(os.path.join(args.dir, "*", "*.h5")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no recordings under {args.dir}/{args.glob}")
    bad = 0
    for f in files:
        r = one(f, replay=not args.no_replay, tol=args.tol, obs_mode=args.replay_obs_mode)
        flag = "OK " if not r["problems"] else "BAD"
        bad += bool(r["problems"])
        print(f"{flag} {r['env']} seed={r['seed']} steps={r['steps']} success={r['success']} "
              f"arm_clip={r['arm_clip_frac']} body_clip={r['body_clip_frac']} zero_var={r['zero_var_channels']}"
              + (f" replay_success={r['replay_success']} qpos_err={r['replay_qpos_err']}" if "replay_success" in r else "")
              + (f"  <- {'; '.join(r['problems'])}" if r["problems"] else ""), flush=True)
    print(f"{len(files) - bad} of {len(files)} recordings pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
