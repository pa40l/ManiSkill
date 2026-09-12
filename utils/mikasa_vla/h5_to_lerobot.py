"""ManiSkill h5 recordings (native `pd_joint_delta_pos`, rgb obs, ds_fetch rig) -> LeRobot dataset for openpi.

Run inside the openpi venv:
    uv run --project /workspace/openpi python h5_to_lerobot.py --raw <dir with seed_*/...h5> --repo mikasa/<name>

The recordings are the oracle's own steps, taken with `evaluate_planner ... --control-mode
pd_joint_delta_pos --obs-mode rgb --traj-dir <dir>`: `RecordEpisode` keeps the array that went
into `env.step`, so the `actions` here are exactly what the env executed, no conversion
anywhere (the supervisor's format note, 2026-09-08). A recording in any other control mode or
observation mode is refused, not converted.

Features (names are what `MikasaInputs` in openpi repacks):
  image        fetch_head rgb (the camera ON the head)  -> base_0_rgb
  wrist_image  fetch_hand rgb (the wrist camera)        -> left_wrist_0_rgb
  (both are 224x224 as of 2026-09-12, but the size is read from each recording rather than
   written here: the robot's cameras are the robot's business, see `camera_shapes_of`)
  (no third image since 2026-09-10: the robot has the real Fetch's two cameras; openpi's
   right_wrist_0_rgb slot is filled with zeros and masked off in `MikasaInputs`)
  state        obs["agent"]["qpos"], 15 floats, nothing else (no obs["extra"]: task truth lives
               there under obs_mode="state", and the base/TCP poses are functions of qpos)
  actions      13 floats, all in [-1, 1]: see FORMAT below / the card written next to the data
  task         the env's language instruction (INSTRUCTIONS of the scene module)

The two camera streams are encoded to MP4 (`dtype: "video"`, LeRobot's `videos/chunk-*/`),
which is also what MIKASA-Robo's own converter does by default; `--frames` stores PNGs
instead, the counterpart of their `--no-videos`. Measured on the five Retrieval episodes of
2026-09-10 (2897 frames): 29 MB as MP4 against 235 MB as PNG, and `LeRobotDataset` returns
the same decoded `(3, H, W)` float tensor either way.

The format card `meta/mikasa_format.json` (and `README.md`) goes next to the dataset: neither
the h5 nor LeRobot's own metadata records the control mode, and two datasets in different modes
are indistinguishable 13-float vectors otherwise.
"""
import argparse, csv, glob, json, os, shutil, subprocess, sys
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
    "MikasaDepthRecall-v1": (
        "Take the props out of the cabinet onto the counter, keep the one from the "
        "back, and put the others back where they stood.",
        "Clear the row of props onto the counter slots, keep the one that stood at the "
        "back, and return every other prop to the shelf spot it came from.",
        "Empty the shelf row onto the counter, hold on to the prop from the deepest "
        "spot, and put the rest back exactly where each of them was.",
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
#: The robot's two cameras, in the order their frames become `image` and `wrist_image`. The
#: sizes are NOT here on purpose — they are read from each recording, see `camera_shapes_of`.
CAMERAS = ("fetch_head", "fetch_hand")

#: The action vector, channel by channel. `-1` and `+1` map to `lo` and `hi`; `kind` says how
#: the env reads the number. This IS `ds_fetch`'s `pd_joint_delta_pos` controller layout.
FORMAT = {
    "control_mode": CONTROL_MODE,
    "control_freq_hz": None,   # filled from the recording; a step is 1/control_freq seconds
    "sim_freq_hz": None,       # filled from the recording; sim_freq/control_freq physx substeps a step
    # both filled from the recordings: which robot executed the actions that produced these
    # frames. `ds_fetch` is ours; `fetch_cam224` is ManiSkill's own Fetch with our camera size,
    # and a dataset rendered on it must not claim our robot's URDF (see ROBOTS below).
    "robot_uid": None,
    "urdf": None,
    "sim_backend": None,       # filled below: what the recording names, or the rule that decided it
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
    "finger_note": "columns 13 and 14 are r_ and l_gripper_finger_joint in URDF declaration order, the same "
                   "order stock Fetch has — but ds_fetch's gripper_link frame is rotated, so OUR column 13 is "
                   "physically the finger on the side where stock's column 14 sits (measured 2026-09-12 on the "
                   "loaded articulation: r_gripper_finger_link centre of mass at y -0.0304 against stock's "
                   "+0.0304; both pads are the stock pads, each with the shape right for the place it occupies). "
                   "They are a mimic pair driven to one target, so the swap changes no value in this dataset; it "
                   "matters only to code that picks a side by name",
    # Sizes are filled from the recordings, like control_freq_hz — see `camera_shapes_of`;
    # the gripper note is counted from the data in the card block, not asserted here.
    "images": {"image": "fetch_head (on the head, turns with head_pan/head_tilt)", "wrist_image": "fetch_hand"},
    "normalization_note": "the data are the env's own normalized actions; any per-channel statistic normalization "
                          "for training must be undone before env.step; action channels listed under "
                          "inactive_channels have zero variance in this dataset and near_inactive_channels a std "
                          "under 1e-3, and near_constant_state_columns are the state columns with a std under "
                          "1e-3 — do not divide any of them by its std (treat as constant, or clamp the std)",
}


def env_info(h5_path: str) -> dict:
    """The recording's json, or an empty dict when there is none.

    A pool can hold a file with no json beside it: a replay that failed used to leave one
    behind (2026-09-12, three of 994 on stock Fetch). That is not a recording, and the caller
    skips it on the empty `episodes` rather than dying on a missing file.
    """
    try:
        with open(h5_path.replace(".h5", ".json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def check_recording(meta: dict, path: str):
    kw = meta["env_info"]["env_kwargs"]
    if kw.get("control_mode") != CONTROL_MODE:
        raise SystemExit(f"{path}: control_mode={kw.get('control_mode')!r}, need {CONTROL_MODE!r} — "
                         "record natively, do not convert")
    if kw.get("obs_mode") != OBS_MODE:
        raise SystemExit(f"{path}: obs_mode={kw.get('obs_mode')!r}, need {OBS_MODE!r}")


def freqs_of(h5_path: str) -> tuple:
    """`(control_freq, sim_freq)` of the recording, read from its `.json`.

    The env kwargs carry a `sim_config` only for what the sweep overrode
    (`--control-freq`); everything else is ManiSkill's `SimConfig` default, 20 Hz control
    and 100 Hz physics. Both numbers belong in the card: the first IS the dataset's frame
    rate (one recorded action is one control step), the second says how many physics
    substeps that action was integrated over, which is what makes a 10 Hz step of ours a
    different animal from a 10 Hz step of a suite that never left 20 Hz control.

    Example:
        >>> freqs_of("seed_1100/20260910_101112.h5")   # doctest: +SKIP
        (10, 100)
    """
    with open(h5_path.replace(".h5", ".json")) as fh:
        meta = json.load(fh)
    sim = meta["env_info"]["env_kwargs"].get("sim_config", {}) or {}
    return int(sim.get("control_freq", 20)), int(sim.get("sim_freq", 100))


def control_freq_of(h5_path: str) -> int:
    """The control frequency alone — the dataset's frame rate."""
    return freqs_of(h5_path)[0]


ROBOTS = {
    "mikasa_ds_fetch": ("ds_fetch (mikasa_ds_fetch in jezvgg/ManiSkill)",
                 "utils.mikasa/motionplanning/fetch/fetch.urdf (the file ds_fetch.py:19 loads; "
                 "roll joints +-3.141 as upstream, base spin +-50 rad)"),
    "fetch_cam224": ("fetch_cam224 (ManiSkill's own `fetch`, unchanged, rendering at 224; "
                     "utils.mikasa/agents/fetch_cam224.py)",
                     "ManiSkill's own fetch.urdf — stock joint limits, the roll joints "
                     "continuous; nothing of ours is in the robot these frames came from"),
}


def robot_card_fields(uid: str) -> tuple:
    """What the card says about the robot the recordings were executed on."""
    if uid in ROBOTS:
        return ROBOTS[uid]
    return (uid, f"the URDF {uid} loads (this converter knows no more about it)")


def rebuild_sources(files: list, raw: str, ds, *, fps: int) -> tuple:
    """The episode table of a dataset whose card was never written, read off the recordings.

    `--card-only` normally carries `sources` forward from the card it is rewriting. A dataset
    merged from parallel shards has no card to carry, so the table is rebuilt here from the pool
    the frames came from — and the rebuild is only accepted when the pool and the dataset agree
    episode by episode: same count, same length, in the same order. That agreement is worth more
    than the table it produces. A shard merged while its converter was still running yields a
    dataset that is perfectly self-consistent and quietly short, and no field inside it can say
    so; the pool it claims to hold can, and does, right here.

    The same rows the conversion loop writes, in the same order: a lost seed and a failed episode
    are not in the dataset, so they are not in the table either.
    """
    lengths = [json.loads(line)["length"]
               for line in open(os.path.join(ds.root, "meta", "episodes.jsonl"))]
    sources, env_ids = [], set()
    frame = 0
    for f in files:
        meta = env_info(f)
        if not meta.get("episodes"):
            continue  # a lost seed: never converted, so never in the dataset
        env_id = meta["env_info"]["env_id"]
        with h5py.File(f, "r") as h:
            t = h["traj_0"]
            ok = bool(np.array(t["success"])[-1])
            n = int(t["actions"].shape[0])
            idx = int(np.array(t["env_states"]["instruction_idx"])[0]) \
                if "instruction_idx" in t["env_states"] else 0
        if not ok:
            continue  # the converter skips a failure unless --keep-failures, and so does this
        i = len(sources)
        if i >= len(lengths):
            raise SystemExit(
                f"{out_name(ds)} holds {len(lengths)} episodes but the pool under {raw} has more "
                f"({os.path.relpath(f, raw)} is number {i + 1}) — the dataset is short of its "
                "pool; a shard was merged before its converter finished")
        if n != lengths[i]:
            raise SystemExit(
                f"episode {i} of {out_name(ds)} is {lengths[i]} frames but "
                f"{os.path.relpath(f, raw)} recorded {n} — the dataset and the pool are not the "
                "same episodes in the same order")
        env_ids.add(env_id)
        sources.append({"file": os.path.relpath(f, raw), "seed": meta["episodes"][0].get("episode_seed"),
                        "steps": n, "success": ok, "episode_index": i, "env_id": env_id,
                        "duration_s": round(n / float(fps), 3), "instruction_idx": idx,
                        "from_index": frame, "to_index": frame + n - 1})
        frame += n
    if len(sources) != len(lengths):
        raise SystemExit(f"{out_name(ds)} holds {len(lengths)} episodes, the pool under {raw} "
                         f"holds {len(sources)} — the dataset is short of its pool")
    return sources, env_ids


def out_name(ds) -> str:
    """The dataset's directory, for a message a reader can act on."""
    return str(getattr(ds, "root", "the dataset"))


def robot_of(h5_path: str) -> str:
    """The robot the recording was made on, for LeRobot's `robot_type`.

    It used to be the string "mikasa_ds_fetch", which was true while every recording came from our
    robot. Since 2026-09-12 the frames come from `fetch_cam224` — stock Fetch with our camera
    size — while the actions still come from the planner on ds_fetch, and a hand-written
    `robot_type` would then name a robot that never rendered a pixel of the dataset. The card
    (`meta/mikasa_format.json`) is where the two-robot story is told in full; this field just
    stops disagreeing with it.
    """
    return str(env_info(h5_path).get("env_info", {}).get("env_kwargs", {}).get(
        "robot_uids") or "mikasa_ds_fetch")


def camera_shapes_of(h5_path: str) -> dict:
    """`{camera: (height, width, channels)}` read from the recording's own frames.

    These used to be written into the feature spec by hand, 256 for the head and 128 for the
    wrist. That silently encodes one day's camera settings into the converter: when the owner
    moved both cameras to 224 on 2026-09-12, a hand-written spec would have promised LeRobot a
    shape the frames do not have. Reading it from the recording costs one open of one file and
    cannot disagree with the data.
    """
    with h5py.File(h5_path, "r") as h:
        sensors = h["traj_0"]["obs"]["sensor_data"]
        return {k: tuple(int(n) for n in sensors[k]["rgb"].shape[1:]) for k in CAMERAS}


def camera_fovs_of(h5_path: str, shapes: dict) -> dict:
    """`{camera: vertical fov in radians}`, derived from the recorded intrinsics.

    `obs["sensor_param"][cam]["intrinsic_cv"]` is stored every step, so the lens the episode was
    actually filmed with is in the recording and need not be trusted to a constant here:
    fov_y = 2 atan(height / 2 f_y). Checked against the settings of 2026-09-12 on a Retrieval
    recording — the head came back 1.5000 and the wrist 2.0000, with f_y constant over the
    episode. Returns None for a camera whose parameters the recording does not carry (an
    `obs_mode` without `sensor_param`), rather than guessing.
    """
    out = {}
    with h5py.File(h5_path, "r") as h:
        params = h["traj_0"]["obs"].get("sensor_param")
        for k in CAMERAS:
            if params is None or k not in params or "intrinsic_cv" not in params[k]:
                out[k] = None
                continue
            f_y = float(np.asarray(params[k]["intrinsic_cv"])[0, 1, 1])
            out[k] = round(2 * float(np.arctan(shapes[k][0] / (2 * f_y))), 4)
    return out


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
    ap.add_argument("--frames", action="store_true",
                    help="store the camera streams as PNG frames instead of encoding them to "
                         "MP4. Default is MP4 (`dtype: \"video\"`), which is what MIKASA-Robo's "
                         "own converter defaults to and what keeps a 200-episode task in "
                         "hundreds of MB instead of ~6 GB; the frames are then lossy, which is "
                         "the reason the escape exists.")
    ap.add_argument("--keep-failures", action="store_true", help="also keep episodes whose success is False")
    ap.add_argument("--append", action="store_true",
                    help="add these recordings to the dataset already at --repo instead of "
                         "replacing it. Episode numbering continues from what is there and "
                         "the card's running action statistics are carried forward, so a "
                         "pool can be converted in batches — which it has to be: the rgb "
                         "h5 of a batch weighs about as much as the dataset it becomes "
                         "(measured 2026-09-10), so a pool of any size only fits if each "
                         "batch's h5 is deleted before the next is rendered. Refuses if the "
                         "repo does not exist yet, and refuses to mix storage formats")
    ap.add_argument("--stock-execution", default=None,
                    help="a measured claim to carry in the card: what an unmodified ManiSkill "
                         "`fetch` does when THIS dataset's action streams are fed to env.step "
                         "step by step, with the same seeds and no subsampling. Write the numbers, "
                         "not an opinion — successes out of N against the task's own predicate, "
                         "the qpos and TCP divergence quantiles, the date and the tree that "
                         "recorded. A rewrite with --card-only keeps it unless a new one is given")
    ap.add_argument("--card-only", action="store_true",
                    help="rewrite meta/mikasa_format.json of the dataset at --repo from its own frames "
                         "and its old card, converting nothing. For when a card fix lands after a long "
                         "conversion (2026-09-11: the gripper note and the URDF path): the statistics "
                         "are recomputed from the dataset, the codec, crf, rates and sources are kept "
                         "from the old card, and source_commit stays the commit that converted — "
                         "card_commit says which tree wrote the card")
    ap.add_argument("--video", action="store_true",
                    help="accepted so older command lines (`collect_pool.py --video`) still "
                         "parse, and changes nothing: MP4 is the default since 2026-09-10 "
                         "(the owner's chain, see --frames). Measured 2026-09-10, whole "
                         "dataset on disk, one episode at 10 Hz: DepthRecall 164 MB as PNG "
                         "against 11.6 as video, SeasonDish 53 against 4.2, CabinetSearch 248 "
                         "against 20.0 - 76-107 KB a frame against 5.7-8.7, and the difference "
                         "between a pool that fits on this machine and one that does not. The "
                         "video is lossy where PNG is not — LeRobot's own defaults are SVT-AV1 "
                         "g=2 crf=30, measured on SeasonDish at PSNR 34.9 dB, mean error 2.8 of "
                         "255 — and openpi's lerobot 0.1.0 decodes it on the fly")
    ap.add_argument("--vcodec", default="libsvtav1", choices=["libsvtav1", "h264", "hevc"],
                    help="codec of the MP4 streams — the three lerobot 0.1.0 accepts. libsvtav1 (AV1) "
                         "is LeRobot's own default; h264 is the one every decoder and player has. At "
                         "matched quality (see --crf) the two took the same disk on Retrieval, and h264 "
                         "encoded about 4x faster. Written into the card, and --append refuses to mix "
                         "codecs in one dataset")
    ap.add_argument("--crf", type=int, default=None,
                    help="the encoder's constant rate factor: lower keeps the frames closer and takes "
                         "more disk. The scales differ — x264 runs 0-51, SVT-AV1 0-63 — so one number is "
                         "not one quality, and lerobot 0.1.0's default of 30 is about 3.4 dB coarser in "
                         "h264 than in AV1. The default is each codec's match for lerobot's own quality: "
                         "30 for libsvtav1 (lerobot's), 23 for h264, 30 for hevc (lerobot's, unmeasured). "
                         "Measured 2026-09-11 on three Retrieval episodes, g=2, PSNR head / wrist: AV1 "
                         "crf 30 36.7 / 35.1 dB at 1.96 MB an episode; h264 crf 23 36.5 / 35.1 at 2.03 "
                         "MB; h264 crf 30 33.2 / 32.3 at 1.0 MB. Written into the card; --append refuses "
                         "to mix two crf values in one dataset")
    args = ap.parse_args()
    if args.crf is None:
        args.crf = {"libsvtav1": 30, "h264": 23, "hevc": 30}[args.vcodec]
    if not 0 <= args.crf <= (63 if args.vcodec == "libsvtav1" else 51):
        raise SystemExit(f"--crf {args.crf} is outside {args.vcodec}'s range")
    if args.frames and args.video:
        raise SystemExit("--frames and --video contradict each other: MP4 is the default, "
                         "--frames stores PNG")
    files = sorted(glob.glob(os.path.join(args.raw, args.glob)))
    if not files and not args.card_only:
        raise SystemExit(f"no recordings under {args.raw}/{args.glob}")
    if args.fps is None and files:
        args.fps = control_freq_of(files[0])
        print(f"fps {args.fps} (the control frequency the episodes were recorded at)")
    shapes = camera_shapes_of(files[0])
    fovs = camera_fovs_of(files[0], shapes)
    print("cameras " + ", ".join(
        f"{k} {h}x{w}" + ("" if fovs[k] is None else f" fov {fovs[k]}")
        for k, (h, w, _c) in shapes.items()) + " (read from the recordings, not assumed)")
    pix = "image" if args.frames else "video"
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    out = HF_LEROBOT_HOME / args.repo
    if pix == "video":
        # lerobot 0.1.0 encodes each episode with `encode_video_frames(img_dir, video_path,
        # fps, overwrite=True)` (lerobot_dataset.py:983): no codec or crf argument, so it
        # always gets its own defaults, libsvtav1 at crf 30. The call resolves the
        # module-level name, which is the one to wrap.
        import functools
        import lerobot.common.datasets.lerobot_dataset as _lds
        _lds.encode_video_frames = functools.partial(_lds.encode_video_frames, vcodec=args.vcodec, crf=args.crf)
        print(f"video codec {args.vcodec}, crf {args.crf}", flush=True)
    n_ep = n_frames = 0
    env_ids, sources = set(), []
    col_sum = np.zeros(ACTION_DIM); col_sq = np.zeros(ACTION_DIM)
    st_sum = np.zeros(STATE_DIM); st_sq = np.zeros(STATE_DIM); grip_mid = 0
    prior = None; raw_files = []
    if args.card_only:
        # Everything the card says about the data is recomputed from the dataset itself; everything
        # it says about how the dataset was made is kept, because this run is not making it.
        if not out.exists():
            raise SystemExit(f"--card-only but {out} does not exist")
        card_path = out / "meta" / "mikasa_format.json"
        # A dataset can arrive here without a card at all: `merge_lerobot_shards.py` joins the
        # slices of a parallel conversion into LeRobot's own metadata and deliberately writes no
        # card, because a card is a claim about the recordings and the merge never reads them.
        # So the episode table is REBUILT from --raw when there is nothing to carry forward, and
        # every row is checked against the dataset's own episode lengths below: a dataset that is
        # short of its pool — a shard merged while its converter was still running — cannot pass
        # that check, which makes this step the completeness gate as well as the card writer.
        prior = json.load(open(card_path)) if card_path.exists() else None
        ds = LeRobotDataset(args.repo)
        acts = np.stack([np.asarray(x, dtype=np.float64) for x in ds.hf_dataset["actions"]])
        st = np.stack([np.asarray(x, dtype=np.float64) for x in ds.hf_dataset["state"]])
        col_sum = acts.sum(0); col_sq = (acts ** 2).sum(0)
        st_sum = st.sum(0); st_sq = (st ** 2).sum(0)
        grip_mid = int((np.abs(acts[:, 7]) < 1.0 - 1e-6).sum())
        n_ep = ds.num_episodes; n_frames = ds.num_frames
        if prior:
            env_ids = set(prior["env_ids"]); sources = list(prior["sources"])
            args.fps = int(prior.get("control_freq_hz", ds.fps))
            pix = prior.get("image_storage", pix)
            args.vcodec = prior.get("video_codec") or args.vcodec
            # a card written before --crf existed was encoded at lerobot's own default, 30,
            # whatever the codec: this run's default must not decide what an old dataset holds
            args.crf = int(prior["video_crf"]) if prior.get("video_crf") is not None else 30
        else:
            if not files:
                raise SystemExit(f"{out} has no card and no recordings were found under "
                                 f"{args.raw}/{args.glob} to rebuild one from")
            args.fps = int(ds.fps)
            pix = "video" if ds.features["image"]["dtype"] == "video" else "image"
            if pix == "video":
                said = ds.features["image"].get("info", {}).get("video.codec")
                if said and said != args.vcodec:
                    raise SystemExit(f"{out} holds {said} video, this run says --vcodec "
                                     f"{args.vcodec} — the card must not name another codec")
            sources, env_ids = rebuild_sources(files, args.raw, ds, fps=args.fps)
        raw_files, files = files, []
        print(f"card only: {n_ep} episodes, {n_frames} frames already in {out}", flush=True)
    elif args.append:
        if not out.exists():
            raise SystemExit(f"--append but {out} does not exist — convert the first batch without it")
        # Carry the running action statistics forward. They are sums, not averages, so a
        # batch cannot just be averaged in afterwards: mean and variance over the WHOLE
        # pool need sum(a) and sum(a^2) over the whole pool, which is what the card keeps.
        prior = json.load(open(out / "meta" / "mikasa_format.json"))
        # A card written before `image_storage` existed says it in `images_encoding`
        # instead (MP4 unless --frames); one older than both was PNG.
        prior_pix = prior.get("image_storage") or (
            "video" if str(prior.get("images_encoding", "")).startswith("MP4") else "image")
        if prior_pix != pix:
            raise SystemExit(f"{out} is stored as {prior_pix!r}, this run is {pix!r} — "
                             "one dataset cannot hold both")
        # A card written before `video_codec` existed was encoded with lerobot's default.
        prior_codec = prior.get("video_codec") or ("libsvtav1" if prior_pix == "video" else None)
        if pix == "video" and prior_codec != args.vcodec:
            raise SystemExit(f"{out} is encoded as {prior_codec!r}, this run is {args.vcodec!r} — "
                             "one dataset cannot hold both codecs")
        # and one written before `video_crf` existed, at lerobot's default crf. A card --card-only
        # rewrote from such a card carries the same 30, so the key can also be there as null.
        prior_crf = prior.get("video_crf") if prior.get("video_crf") is not None else 30
        if pix == "video" and prior_crf != args.crf:
            raise SystemExit(f"{out} is encoded at crf {prior_crf}, this run at {args.crf} — "
                             "one dataset cannot hold two crf values")
        if int(prior.get("control_freq_hz", args.fps)) != int(args.fps):
            raise SystemExit(f"{out} was recorded at {prior['control_freq_hz']} Hz, this batch at {args.fps}")
        n_ep = int(prior["episodes"]); n_frames = int(prior["frames"])
        env_ids = set(prior["env_ids"]); sources = list(prior["sources"])
        col_sum = np.asarray(prior["action_channel_sum"], dtype=np.float64)
        col_sq = np.asarray(prior["action_channel_sq_sum"], dtype=np.float64)
        ds = LeRobotDataset(args.repo)
        if "state_column_sum" in prior:
            st_sum = np.asarray(prior["state_column_sum"], dtype=np.float64)
            st_sq = np.asarray(prior["state_column_sq_sum"], dtype=np.float64)
            grip_mid = int(prior["gripper_intermediate_steps"])
        else:
            # a card older than the state statistics: take them from the frames already there
            st = np.stack([np.asarray(x, dtype=np.float64) for x in ds.hf_dataset["state"]])
            a7 = np.array([float(x[7]) for x in ds.hf_dataset["actions"]])
            st_sum = st.sum(0); st_sq = (st ** 2).sum(0)
            grip_mid = int((np.abs(a7) < 1.0 - 1e-6).sum())
        # A camera change is invisible on this path otherwise: the features come from the
        # dataset already on disk, so frames of another size would be appended under the old
        # shape. Compare against what is there and refuse, naming both sizes.
        for key, cam in (("image", "fetch_head"), ("wrist_image", "fetch_hand")):
            was = tuple(int(n) for n in ds.features[key]["shape"])
            if was != shapes[cam]:
                raise SystemExit(f"{out} holds {key} at {was}, this batch records {cam} at "
                                 f"{shapes[cam]} — a camera change means a new dataset, "
                                 "re-render the earlier episodes instead of appending")
        ds.start_image_writer(num_processes=4, num_threads=8)
        print(f"appending to {out}: {n_ep} episodes, {n_frames} frames already there")
    else:
        if out.exists():
            shutil.rmtree(out)
        ds = LeRobotDataset.create(
            repo_id=args.repo, robot_type=robot_of(files[0]), fps=args.fps,
            features={
                "image": {"dtype": pix, "shape": shapes["fetch_head"], "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": pix, "shape": shapes["fetch_hand"], "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
            },
            image_writer_threads=8, image_writer_processes=4,
        )
    added = 0
    for f in files:
        meta = env_info(f)
        if not meta.get("episodes"):
            # A lost seed: the sweep drops a failure's steps but leaves an almost empty h5 and a
            # json whose "episodes" is []. It is not a recording (check_recordings skips it too).
            print(f"skip {os.path.relpath(f, args.raw)}: no episode in the json (a lost seed)", flush=True)
            continue
        check_recording(meta, f)
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
        imgs = {k: np.array(t["obs"]["sensor_data"][k]["rgb"])[:n] for k in CAMERAS}
        # One pool must not mix camera settings: the feature spec was fixed from the first
        # recording, so a later one of another size would be written under a shape it does not
        # have. Say which file and which camera rather than letting LeRobot fail on a tensor.
        for k in CAMERAS:
            if imgs[k].shape[1:] != shapes[k]:
                raise SystemExit(f"{f}: {k} is {imgs[k].shape[1:]} but this dataset was opened "
                                 f"for {shapes[k]} (from {files[0]}) — one pool, one camera size")
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
        ds.save_episode(); n_ep += 1; n_frames += n; added += 1
        col_sum += acts.sum(0); col_sq += (acts ** 2).sum(0)
        q64 = qpos.astype(np.float64)
        st_sum += q64.sum(0); st_sq += (q64 ** 2).sum(0)
        grip_mid += int((np.abs(acts[:, 7]) < 1.0 - 1e-6).sum())
        sources.append({"file": os.path.relpath(f, args.raw), "seed": ep.get("episode_seed"),
                        "steps": int(n), "success": ok, "episode_index": n_ep - 1, "env_id": env_id,
                        "duration_s": round(n / float(args.fps), 3), "instruction_idx": idx,
                        # where this episode sits in the dataset's own frame numbering
                        "from_index": n_frames - n, "to_index": n_frames - 1})
        print(f"episode {n_ep}: {os.path.basename(os.path.dirname(f))} {n} frames, task='{task[:40]}...'", flush=True)
        # `added`, not `n_ep`: under --append n_ep starts at the pool's total, and a cap
        # compared against it would end the run before converting anything.
        if args.max_episodes and added >= args.max_episodes:
            break
    # the card
    mean = col_sum / max(n_frames, 1); var = col_sq / max(n_frames, 1) - mean ** 2
    inactive = [int(i) for i in range(ACTION_DIM) if var[i] < 1e-12]
    # near-inactive: a std under 1e-3 (head pan in Retrieval: 4e-4 — the drives park the
    # head at 0 from a rest tilt of a few thousandths); dividing by such a std at training
    # time amplifies noise, so the card names them too
    near_inactive = [int(i) for i in range(ACTION_DIM) if 1e-12 <= var[i] < 1e-6]
    # the same test on the state: head_pan in Retrieval has a std of 1.6e-5 rad (the planner parks
    # it at 0), and openpi's quantile normalisation would stretch that noise over the whole range
    st_mean = st_sum / max(n_frames, 1); st_var = st_sq / max(n_frames, 1) - st_mean ** 2
    near_constant_state = [int(i) for i in range(STATE_DIM) if st_var[i] < 1e-6]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
    except Exception:
        commit = None
    if args.card_only and prior:
        control_freq, sim_freq = int(prior["control_freq_hz"]), int(prior["sim_freq_hz"])
    else:
        # the recordings themselves on a first card, whether it is written by a conversion or by
        # a --card-only run over a merged dataset
        control_freq, sim_freq = freqs_of((files or raw_files)[0])
    # Provenance comes from the recordings whenever they are still under --raw — so a --card-only
    # rewrite fills in what an older card never held — and from that old card otherwise.
    # The supervisor's rule asks for physx_cpu and no recording says which backend ran:
    # `evaluate_planner` passes sim_backend only when asked, and ManiSkill resolves "auto" to
    # physx_cpu whenever num_envs is 1 (sapien_env.py:233-237). The card states the rule that
    # decided it rather than leaving the reader to guess.
    probe = raw_files if args.card_only else files
    sim_backend = None
    if probe:
        kw0 = env_info(probe[0])["env_info"]["env_kwargs"]
        sim_backend = kw0.get("sim_backend") or (
            "physx_cpu (ManiSkill's auto rule for num_envs=1; the recordings do not name a backend)"
            if int(kw0.get("num_envs", 1)) == 1 else
            "unknown: num_envs > 1 and the recordings name no backend")
    if sim_backend is None and prior:
        sim_backend = prior.get("sim_backend")
    # The h5 carries no provenance either: RecordEpisode writes commit_info null, so which tree
    # and commit produced the pool lives only in the pool's own pool.env, which does not travel
    # with the dataset. Copy that line in.
    pool_env = os.path.join(args.raw, "pool.env")
    here = open(pool_env).read().strip().replace("\n", "; ") if os.path.exists(pool_env) else None
    was = prior.get("recording_pool_env") if prior else None
    # A dataset grows across pools under --append, and each pool has its own provenance: the
    # tree that recorded episode 900 is not the one that recorded episode 1. Keep every line.
    recording_env = here if not was else (was if not here or here in was else f"{was}  ||  {here}")
    # The robot the recordings name. A --card-only rewrite with the pool gone keeps what the old
    # card said rather than guessing, and the fields are already prose there.
    if probe:
        said_robot, said_urdf = robot_card_fields(robot_of(probe[0]))
    elif prior:
        said_robot, said_urdf = prior.get("robot_uid"), prior.get("urdf")
    else:
        said_robot, said_urdf = robot_card_fields("mikasa_ds_fetch")
    card = dict(FORMAT)
    card.update({
        "robot_uid": said_robot,
        "urdf": said_urdf,
        # The camera sizes a reader needs in order to know whether this dataset survives a
        # camera change: `(height, width)` exactly as the frames were recorded.
        "images": {key: f"{FORMAT['images'][key]} {shapes[cam][0]}x{shapes[cam][1]}"
                   for key, cam in (("image", "fetch_head"), ("wrist_image", "fetch_hand"))},
        "camera_fov_rad": {cam: fovs.get(cam) for cam in CAMERAS},
        "control_freq_hz": control_freq,
        "sim_freq_hz": sim_freq,
        "sim_backend": sim_backend,
        "recording_pool_env": recording_env,
        # A claim about this dataset that only an execution run can make: not "our robot is
        # like stock Fetch" in general, but "these action streams reach the task's success on
        # unmodified stock, and by this margin". Kept across a --card-only rewrite.
        "stock_fetch_execution": args.stock_execution or (
            prior.get("stock_fetch_execution") if prior else None),
        "physx_substeps_per_action": sim_freq // control_freq,
        "images_encoding": ("PNG frames, lossless (LeRobot dtype 'image', --frames)" if args.frames
                            else "MP4, lossy (LeRobot dtype 'video'; the images ARE the observations, "
                                 "not a render-mode video)"),
        # what `--append` checks a later batch against: one dataset holds one storage
        "image_storage": pix,
        # and one codec: lerobot's own metadata probes the file, the card says what was asked for
        "video_codec": args.vcodec if pix == "video" else None,
        "video_crf": args.crf if pix == "video" else None,
        "env_ids": sorted(env_ids), "episodes": n_ep, "frames": n_frames,
        "inactive_channels": inactive,
        "near_inactive_channels": near_inactive,
        "near_constant_state_columns": near_constant_state,
        "state_column_std": [round(float(v), 7) for v in np.sqrt(np.maximum(st_var, 0))],
        # what index 7 actually holds. The planners' release ramp (change_gripper_state(ramp=...),
        # 12 steps in Retrieval so the pads leaving an 8 g cup do not topple it) writes values
        # between -1 and +1, which the supervisor's note of 2026-09-08 still lists as absent.
        "gripper_intermediate_steps": grip_mid,
        "gripper_note": (f"index 7 is an absolute opening target: {grip_mid} of {n_frames} steps "
                         f"({100 * grip_mid / max(n_frames, 1):.1f}%) lie strictly between -1 and +1 — "
                         "the planners' linear release ramp — and the rest are -1 (closed) or +1 (open)"
                         if grip_mid else "index 7 takes only -1 (closed) or +1 (open) in this dataset"),
        "action_channel_mean": [round(float(v), 6) for v in mean],
        "action_channel_std": [round(float(v), 6) for v in np.sqrt(np.maximum(var, 0))],
        # the raw running sums, so `--append` can continue the statistics over a pool
        # converted in batches rather than re-reading every h5 that has been deleted
        "action_channel_sum": [float(v) for v in col_sum],
        "action_channel_sq_sum": [float(v) for v in col_sq],
        "state_column_sum": [float(v) for v in st_sum],
        "state_column_sq_sum": [float(v) for v in st_sq],
        # the tree that converted stays; card_commit is the tree that wrote this card
        "source_commit": prior["source_commit"] if (args.card_only and prior) else commit,
        "card_commit": commit, "sources": sources,
    })
    meta_dir = out / "meta"; meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "mikasa_format.json", "w") as fh:
        json.dump(card, fh, indent=2)
    # The per-episode table as a file of its own. The same rows live in the card under
    # `sources`, but behind a 200-entry JSON field: which seed an episode came from, how long
    # it ran and whether it succeeded is what a reader asks first, and what a training run
    # needs to hold out a seed. Failures do not appear at all unless --keep-failures was given:
    # the sweep never writes them, `replay_to_rgb` never renders them and the loop above skips
    # them, so `success` is True on every row of an ordinary pool.
    EP_COLS = ["episode_index", "seed", "env_id", "steps", "duration_s", "success",
               "from_index", "to_index", "instruction_idx", "file"]
    with open(meta_dir / "mikasa_episodes.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(EP_COLS)
        for i, s in enumerate(sources):
            row = dict(s)
            row.setdefault("episode_index", i)  # a card written before this table had neither
            row.setdefault("duration_s", round(int(row.get("steps", 0)) / float(args.fps), 3))
            w.writerow([row.get(c) for c in EP_COLS])
    with open(out / "README.md", "w") as fh:
        fh.write(f"# {args.repo}\n\nMIKASA recordings of {', '.join(sorted(env_ids))}: {n_ep} episodes, {n_frames} frames, "
                 f"{args.fps} Hz.\n\n`actions` (13) are the env's own `{CONTROL_MODE}` actions, every channel in [-1, 1], fed "
                 f"straight to `env.step` — see `meta/mikasa_format.json` for the channel table, the state layout and the "
                 f"inactive channels {inactive} (zero variance here; do not normalize them by their std).\n"
                 f"`state` (15) is `obs['agent']['qpos']` and nothing else.\n\n"
                 f"`meta/mikasa_episodes.csv` lists every episode: seed, steps, duration, success, "
                 f"its frame range in this dataset, and the recording it came from.\n")
    print(f"done: +{added} this run, {n_ep} episodes, {n_frames} frames -> {out}; inactive channels {inactive}; "
          f"card {meta_dir / 'mikasa_format.json'}; episodes {meta_dir / 'mikasa_episodes.csv'}")


if __name__ == "__main__":
    main()
