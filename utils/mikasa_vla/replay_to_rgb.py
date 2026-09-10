"""Turn STATE recordings into RGB recordings by replaying their actions.

    python tools/vla/replay_to_rgb.py <state_dir> <out_dir> [--limit N] [--tol 5e-3]

The supervisor's suggestion (2026-09-10): *"если у вас есть motion planner, вы можете на
стейтах записать действия, потом среду вызвать в режиме rgb image и в ней прокрутить
просто действия"*. This is that second half — the first half is any ordinary sweep with
`--obs-mode state --traj-dir …`.

Why it is worth having, and what it does NOT buy (measured, 2026-09-10, Retrieval seed
1100 on one core):

| path (Retrieval seed 1100, 10 Hz, one core) | wall clock | peak RSS |
|---|---|---|
| plan + step, `--obs-mode state` | 5.6 s | 1.34 GB |
| plan + step, `--obs-mode rgb` — the direct recording | 15.2 s | 1.99 GB |
| this replay of the state recording into rgb | 8.4 s | 2.6 GB |

So state-then-replay costs about what recording rgb directly costs (5.6 + 8.4 = 14.0 s
against 15.2 s): the rendering is the bill and it is paid either way. What it buys is that
the **planning is paid once**:

* the cameras, their resolution and the observation mode become a rendering choice made
  after the fact — 2026-09-10 the robot's cameras changed (the shoulder outriggers gone,
  `fetch_head` in) and 100 recorded episodes lost their images while their actions stayed
  perfectly good;
* a failed seed costs no rendering at all — plan a large pool on state, replay only the
  episodes that succeeded;
* a re-render of an existing pool costs 8.4 s an episode instead of 15.2 — 45 % less;
* the two halves have different appetites (1.34 GB against 2.6 GB per process here), so a
  planning pool runs wider than a rendering one under the container's 62 GB cgroup.

The replay is exact, not approximate: the recorded actions are fed to a fresh env of the
same seed and control frequency, and the episode is only written when its success flag and
its final `qpos` come back (`--tol`, 5 mm by default; on this machine the error is 0.0).
That is the same guarantee `check_recordings.py` verifies, so a replayed recording is as
honest as a directly recorded one — the actions in it are what the env executed.

Run from `mikasa/` (`python -m` / cwd on sys.path).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.getcwd())

ACTION_DIM, STATE_DIM = 13, 15


def replay_one(path: str, out_dir: str, *, tol: float, obs_mode: str, keep_failures: bool) -> dict:
    """Replay one state recording into `out_dir` as an rgb recording; a verdict dict."""
    import gymnasium as gym
    import torch
    from mani_skill.utils.wrappers.record import RecordEpisode

    import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)

    with open(path.replace(".h5", ".json")) as fh:
        meta = json.load(fh)
    env_id = meta["env_info"]["env_id"]
    kw = dict(meta["env_info"]["env_kwargs"])
    ep = meta["episodes"][0]
    seed = int(ep["episode_seed"])

    with h5py.File(path, "r") as h:
        t = h["traj_0"]
        actions = np.array(t["actions"], dtype=np.float64)
        recorded_success = bool(np.array(t["success"])[-1])
        art = t["env_states"]["articulations"]
        key = next(k for k in art.keys() if "fetch" in k)
        st = np.array(art[key], dtype=np.float64)
        nq = (st.shape[1] - 13) // 2
        qpos_end = st[actions.shape[0], 13:13 + nq]

    if not recorded_success and not keep_failures:
        return dict(file=os.path.relpath(path), seed=seed, skipped="the recorded episode failed")

    # The observation mode is the only thing that changes; everything else — the control
    # mode, the control frequency (`sim_config`), the scene index, the robot — is taken
    # from the recording, so the replay is the same episode with cameras switched on.
    kw.pop("render_mode", None)
    kw["obs_mode"] = obs_mode
    kw["num_envs"] = 1

    env = gym.make(env_id, **kw)
    recorder = RecordEpisode(
        env,
        output_dir=out_dir,
        save_trajectory=True,
        save_video=False,
        save_on_reset=False,
        source_type="motionplanning-replay",
        source_desc=f"replayed from {os.path.relpath(path)} by tools/vla/replay_to_rgb.py",
    )
    t0 = time.time()
    # A plain `reset(seed)` — never `reconfigure=True`, which redraws the layout for the
    # same seed (2026-09-09, SeasonDish 3608).
    recorder.reset(seed=seed)
    success = False
    for i in range(actions.shape[0]):
        _, _, _, _, info = recorder.step(torch.as_tensor(actions[i], dtype=torch.float32)[None])
        success = success or bool(np.asarray(info["success"]).reshape(-1)[0])
    q_end = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float64)
    err = float(np.abs(q_end - qpos_end).max())
    wall = time.time() - t0

    problems = []
    if success != recorded_success:
        problems.append(f"replay success {success} vs recorded {recorded_success}")
    if err > tol:
        problems.append(f"final qpos differs by {err:.5f} > {tol}")
    if not problems:
        recorder.flush_trajectory(save=True, ignore_empty_transition=False)
    recorder.close()
    return dict(file=os.path.relpath(path), seed=seed, steps=int(actions.shape[0]),
                success=success, qpos_err=round(err, 6), wall_s=round(wall, 1),
                problems=problems)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state_dir", help="the sweep's --traj-dir of STATE recordings")
    ap.add_argument("out_dir", help="where the rgb recordings go")
    ap.add_argument("--glob", default="seed_*/*.h5")
    ap.add_argument("--limit", type=int, default=0, help="0 = every recording found")
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="max |qpos| difference at the end of the replay (0.0 on this machine)")
    ap.add_argument("--obs-mode", default="rgb")
    ap.add_argument("--keep-failures", action="store_true",
                    help="replay the episodes the oracle lost too (off: they cost no rendering)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.state_dir, args.glob)))
    if not files and args.glob == "seed_*/*.h5":
        # `--traj-dir <dir>` writes `<dir>/<stamp>.h5`; the per-seed layout only appears
        # when the sweep is given `--traj-dir <dir>/seed_<N>` (what the pools do). Accept
        # both rather than making the caller know which one they have.
        files = sorted(glob.glob(os.path.join(args.state_dir, "*.h5"))) \
            or sorted(glob.glob(os.path.join(args.state_dir, "*", "*.h5")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no recordings under {args.state_dir}/{args.glob}")
    os.makedirs(args.out_dir, exist_ok=True)

    bad = skipped = 0
    for f in files:
        # Named after the SEED, whatever the input layout was: `<out>/seed_<N>/<stamp>.h5`,
        # the same shape a pool writes, so the tools downstream find it either way.
        with open(f.replace(".h5", ".json")) as fh:
            seed_of = json.load(fh)["episodes"][0]["episode_seed"]
        seed_dir = os.path.join(args.out_dir, f"seed_{int(seed_of)}")
        r = replay_one(f, seed_dir, tol=args.tol, obs_mode=args.obs_mode,
                       keep_failures=args.keep_failures)
        if "skipped" in r:
            skipped += 1
            print(f"SKIP seed={r['seed']} {r['skipped']}", flush=True)
            continue
        bad += bool(r["problems"])
        print(f"{'OK ' if not r['problems'] else 'BAD'} seed={r['seed']} steps={r['steps']} "
              f"success={r['success']} qpos_err={r['qpos_err']} {r['wall_s']}s"
              + (f"  <- {'; '.join(r['problems'])}" if r["problems"] else ""), flush=True)
    print(f"{len(files) - bad - skipped} of {len(files)} replayed ({skipped} skipped, {bad} bad)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
