"""ManiSkill h5 recordings (native `pd_joint_delta_pos`, rgb obs, ds_fetch rig) -> LeRobot dataset for openpi.

Run inside the openpi venv:
    uv run --project /workspace/openpi python h5_to_lerobot.py --raw <dir with seed_*/...h5> --repo mikasa/<name>

The recordings are the oracle's own steps, taken with `evaluate_planner ... --control-mode
pd_joint_delta_pos --obs-mode rgb --traj-dir <dir>`: `RecordEpisode` keeps the array that went
into `env.step`, so the `actions` here are exactly what the env executed, no conversion
anywhere (the supervisor's format note, 2026-09-08). A recording in any other control mode or
observation mode is refused, not converted.

Features (names are what `MikasaInputs` in openpi repacks):
  image        fetch_head rgb 256x256 (the camera ON the head)  -> base_0_rgb
  wrist_image  fetch_hand rgb 128x128 (the wrist camera)        -> left_wrist_0_rgb
  (no third image since 2026-09-10: the robot has the real Fetch's two cameras; openpi's
   right_wrist_0_rgb slot is filled with zeros and masked off in `MikasaInputs`)
  state        obs["agent"]["qpos"], 15 floats, nothing else (no obs["extra"]: task truth lives
               there under obs_mode="state", and the base/TCP poses are functions of qpos)
  actions      13 floats, all in [-1, 1]: see FORMAT below / the card written next to the data
  task         the env's language instruction (INSTRUCTIONS of the scene module)

The format card `meta/mikasa_format.json` (and `README.md`) goes next to the dataset: neither
the h5 nor LeRobot's own metadata records the control mode, and two datasets in different modes
are indistinguishable 13-float vectors otherwise.
"""
import argparse, glob, json, os, shutil, subprocess, sys
import h5py, numpy as np

# The instruction texts, per env id, copied from the scene modules (the openpi venv cannot
# import utils.mikasa). `tests/test_vla_tools.py` at home checks they still match.
INSTRUCTIONS = {
    "MikasaCabinetRetrieval-v0": (
        "Take the cup out of the open wall cabinet and set it down on the counter.",
        "Get the cup down from the cabinet shelf onto the counter.",
        "Fetch the cup from the open cabinet and put it on the counter below.",
    ),
    "MikasaSeasonDish-v0": (
        "At the start a yellow ball hovers over a condiment on the counter and then "
        "disappears. Remember which condiment it marked, pick that condiment up, find the "
        "bowl standing somewhere on this counter, and hold the condiment tipped over the bowl "
        "to season the dish. Leave the other condiment where it is.",
        "A yellow ball briefly marks a condiment, then vanishes. Take the condiment it "
        "marked, carry it to the bowl on the counter and tip it over the bowl. Do not touch "
        "the other condiment.",
        "Season the dish in the bowl with the condiment the yellow ball marked at the start: "
        "pick it up, bring it over the bowl and hold it tipped. The other condiment stays "
        "put.",
    ),
    # CabinetSearch under the touch terminal (cfg.terminal == "nudge", the default)
    "MikasaCabinetSearch-v0": (
        "A red cube is hidden in a wall cabinet. Start from the yellow mark on the floor. "
        "Open a cabinet and look inside; if the cube is there, give it a push. If not, close "
        "that cabinet, go back to the yellow mark and try another. Never open the same "
        "cabinet again.",
        "Find the cube hidden in the wall cabinets and nudge it when you see it. Before "
        "opening any cabinet, stand on the yellow floor mark; close each cabinet you open and "
        "return to the mark before opening the next. Do not open a cabinet you have already "
        "opened.",
        "Search the wall cabinets for the hidden red cube, starting from the yellow mark on "
        "the floor: open a cabinet, look, push the cube if it is there; otherwise close the "
        "door, come back to the mark and choose a different cabinet. A cabinet you have "
        "opened must not be opened again.",
    ),
}

CONTROL_MODE = "pd_joint_delta_pos"
OBS_MODE = "rgb"
STATE_DIM = 15
ACTION_DIM = 13

#: The action vector, channel by channel. `-1` and `+1` map to `lo` and `hi`; `kind` says how
#: the env reads the number. This IS `ds_fetch`'s `pd_joint_delta_pos` controller layout.
FORMAT = {
    "control_mode": CONTROL_MODE,
    "control_freq_hz": 20,
    "sim_freq_hz": 100,
    "robot_uid": "ds_fetch (mikasa_ds_fetch in jezvgg/ManiSkill)",
    "urdf": "utils.mikasa/agents/ds_fetch/fetch.urdf (roll joints +-6.28, base spin +-50 rad)",
    "sim_backend": "physx_cpu",
    "obs_mode": OBS_MODE,
    "action": [
        {"index": 0, "group": "arm", "name": "shoulder_pan_joint", "kind": "delta", "lo": -0.1, "hi": 0.1, "unit": "rad/step"},
        {"index": 1, "group": "arm", "name": "shoulder_lift_joint", "kind": "delta", "lo": -0.1, "hi": 0.1, "unit": "rad/step"},
        {"index": 2, "group": "arm", "name": "upperarm_roll_joint", "kind": "delta", "lo": -0.1, "hi": 0.1, "unit": "rad/step"},
        {"index": 3, "group": "arm", "name": "elbow_flex_joint", "kind": "delta", "lo": -0.1, "hi": 0.1, "unit": "rad/step"},
        {"index": 4, "group": "arm", "name": "forearm_roll_joint", "kind": "delta", "lo": -0.1, "hi": 0.1, "unit": "rad/step"},
        {"index": 5, "group": "arm", "name": "wrist_flex_joint", "kind": "delta", "lo": -0.1, "hi": 0.1, "unit": "rad/step"},
        {"index": 6, "group": "arm", "name": "wrist_roll_joint", "kind": "delta", "lo": -0.1, "hi": 0.1, "unit": "rad/step"},
        {"index": 7, "group": "gripper", "name": "finger opening (mimic, both fingers)", "kind": "absolute", "lo": -0.01, "hi": 0.05, "unit": "m"},
        {"index": 8, "group": "body", "name": "head_pan_joint", "kind": "delta", "lo": -0.1, "hi": 0.1, "unit": "rad/step"},
        {"index": 9, "group": "body", "name": "head_tilt_joint", "kind": "delta", "lo": -0.1, "hi": 0.1, "unit": "rad/step"},
        {"index": 10, "group": "body", "name": "torso_lift_joint", "kind": "delta", "lo": -0.1, "hi": 0.1, "unit": "m/step"},
        {"index": 11, "group": "base", "name": "forward velocity", "kind": "velocity", "lo": -1.0, "hi": 1.0, "unit": "m/s"},
        {"index": 12, "group": "base", "name": "yaw velocity", "kind": "velocity", "lo": -3.14, "hi": 3.14, "unit": "rad/s"},
    ],
    "delta_anchor": "the MEASURED joint position of the step the action is applied on (use_target=False)",
    "state": [
        "root_x_axis_joint (base x, world, m)", "root_y_axis_joint (base y, world, m)",
        "root_z_rotation_joint (base yaw, world, rad)", "torso_lift_joint (m)",
        "head_pan_joint", "shoulder_pan_joint", "head_tilt_joint", "shoulder_lift_joint",
        "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint",
        "wrist_roll_joint", "r_gripper_finger_joint (m)", "l_gripper_finger_joint (m)",
    ],
    "state_note": "obs['agent']['qpos'] as is; NOT the action's order (head_pan sits between torso and shoulder_pan)",
    "images": {"image": "fetch_head 256x256 (on the head, turns with head_pan/head_tilt)", "wrist_image": "fetch_hand 128x128"},
    "gripper_note": "index 7 takes only -1 (closed) or +1 (open) in the oracle's recordings",
    "normalization_note": "the data are the env's own normalized actions; any per-channel statistic normalization "
                          "for training must be undone before env.step; channels listed under inactive_channels "
                          "have zero variance in this dataset and near_inactive_channels a std under 1e-3 — "
                          "do not divide either by its std (treat as constant, or clamp the std)",
}


def env_info(h5_path: str) -> dict:
    with open(h5_path.replace(".h5", ".json")) as fh:
        return json.load(fh)


def check_recording(meta: dict, path: str):
    kw = meta["env_info"]["env_kwargs"]
    if kw.get("control_mode") != CONTROL_MODE:
        raise SystemExit(f"{path}: control_mode={kw.get('control_mode')!r}, need {CONTROL_MODE!r} — "
                         "record natively, do not convert")
    if kw.get("obs_mode") != OBS_MODE:
        raise SystemExit(f"{path}: obs_mode={kw.get('obs_mode')!r}, need {OBS_MODE!r}")


def control_freq_of(h5_path: str) -> int:
    """The control frequency the recording was made at, from its `.json` — the env kwargs
    carry `sim_config.control_freq` when the sweep set one (`--control-freq`), and the
    tasks' own default is 20 Hz. One recorded action is one control step, so this is the
    dataset's frame rate.

    Example:
        >>> control_freq_of("seed_1100/20260910_101112.h5")   # doctest: +SKIP
        10
    """
    with open(h5_path.replace(".h5", ".json")) as fh:
        meta = json.load(fh)
    return int(meta["env_info"]["env_kwargs"].get("sim_config", {}).get("control_freq", 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory with seed_*/<stamp>.h5 native recordings")
    ap.add_argument("--repo", default="mikasa/retrieval_jdp")
    ap.add_argument("--glob", default="seed_*/*.h5")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--fps", type=int, default=None,
                    help="dataset frame rate; default: the control frequency the episodes "
                         "were RECORDED at, read from the first recording's env kwargs "
                         "(`sim_config.control_freq`, 20 when it says nothing). The VLA "
                         "data are collected at 10 (`evaluate_planner --control-freq 10`), "
                         "and the frame rate must be the rate the actions were applied at: "
                         "one recorded action is one control step.")
    ap.add_argument("--keep-failures", action="store_true", help="also keep episodes whose success is False")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.raw, args.glob)))
    if not files:
        raise SystemExit(f"no recordings under {args.raw}/{args.glob}")
    if args.fps is None:
        args.fps = control_freq_of(files[0])
        print(f"fps {args.fps} (the control frequency the episodes were recorded at)")
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    out = HF_LEROBOT_HOME / args.repo
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=args.repo, robot_type="mikasa_ds_fetch", fps=args.fps,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=8, image_writer_processes=4,
    )
    n_ep = n_frames = 0
    env_ids, sources = set(), []
    col_sum = np.zeros(ACTION_DIM); col_sq = np.zeros(ACTION_DIM)
    for f in files:
        meta = env_info(f); check_recording(meta, f)
        env_id = meta["env_info"]["env_id"]; env_ids.add(env_id)
        h = h5py.File(f, "r"); t = h["traj_0"]
        ok = bool(np.array(t["success"])[-1])
        if not ok and not args.keep_failures:
            continue
        acts = np.array(t["actions"], dtype=np.float32); n = acts.shape[0]
        assert acts.shape[1] == ACTION_DIM, acts.shape
        assert np.abs(acts).max() <= 1.0 + 1e-6, f"{f}: |action| max {np.abs(acts).max()}"
        qpos = np.array(t["obs"]["agent"]["qpos"], dtype=np.float32)[:n]
        assert qpos.shape[1] == STATE_DIM, qpos.shape
        imgs = {k: np.array(t["obs"]["sensor_data"][k]["rgb"])[:n]
                for k in ("fetch_head", "fetch_hand")}
        ep = meta["episodes"][0]
        texts = INSTRUCTIONS[env_id]
        idx = 0
        if "instruction_idx" in t["env_states"]:
            idx = int(np.array(t["env_states"]["instruction_idx"])[0])
        task = texts[idx % len(texts)]
        for i in range(n):
            ds.add_frame({
                "image": imgs["fetch_head"][i], "wrist_image": imgs["fetch_hand"][i],
                "state": qpos[i], "actions": acts[i], "task": task,
            })
        ds.save_episode(); n_ep += 1; n_frames += n
        col_sum += acts.sum(0); col_sq += (acts ** 2).sum(0)
        sources.append({"file": os.path.relpath(f, args.raw), "seed": ep.get("episode_seed"), "steps": int(n), "success": ok})
        print(f"episode {n_ep}: {os.path.basename(os.path.dirname(f))} {n} frames, task='{task[:40]}...'", flush=True)
        if args.max_episodes and n_ep >= args.max_episodes:
            break
    # the card
    mean = col_sum / max(n_frames, 1); var = col_sq / max(n_frames, 1) - mean ** 2
    inactive = [int(i) for i in range(ACTION_DIM) if var[i] < 1e-12]
    # near-inactive: a std under 1e-3 (head pan in Retrieval: 4e-4 — the drives park the
    # head at 0 from a rest tilt of a few thousandths); dividing by such a std at training
    # time amplifies noise, so the card names them too
    near_inactive = [int(i) for i in range(ACTION_DIM) if 1e-12 <= var[i] < 1e-6]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
    except Exception:
        commit = None
    card = dict(FORMAT)
    card.update({
        "env_ids": sorted(env_ids), "episodes": n_ep, "frames": n_frames,
        "inactive_channels": inactive,
        "near_inactive_channels": near_inactive,
        "action_channel_mean": [round(float(v), 6) for v in mean],
        "action_channel_std": [round(float(v), 6) for v in np.sqrt(np.maximum(var, 0))],
        "source_commit": commit, "sources": sources,
    })
    meta_dir = out / "meta"; meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "mikasa_format.json", "w") as fh:
        json.dump(card, fh, indent=2)
    with open(out / "README.md", "w") as fh:
        fh.write(f"# {args.repo}\n\nMIKASA recordings of {', '.join(sorted(env_ids))}: {n_ep} episodes, {n_frames} frames, "
                 f"{args.fps} Hz.\n\n`actions` (13) are the env's own `{CONTROL_MODE}` actions, every channel in [-1, 1], fed "
                 f"straight to `env.step` — see `meta/mikasa_format.json` for the channel table, the state layout and the "
                 f"inactive channels {inactive} (zero variance here; do not normalize them by their std).\n"
                 f"`state` (15) is `obs['agent']['qpos']` and nothing else.\n")
    print(f"done: {n_ep} episodes, {n_frames} frames -> {out}; inactive channels {inactive}; card {meta_dir / 'mikasa_format.json'}")


if __name__ == "__main__":
    main()
