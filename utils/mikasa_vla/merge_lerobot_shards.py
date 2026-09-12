"""Join LeRobot datasets converted in parallel into the one dataset they were meant to be.

    python tools/vla/merge_lerobot_shards.py --out mikasa/retrieval_stock224_h264_crf23 \
        --shards mikasa/retr_shard0 mikasa/retr_shard1 ... [--force]

Why this exists. `h5_to_lerobot.py` packs one episode at a time: read the h5, write the frames,
call ffmpeg, write the parquet. That loop is sequential by construction, and on 2026-09-12 it
ran at about 6.7 episodes a minute — two and a half hours for a thousand — while the container's
whole CPU allowance (13.6 cores of the machine's 128, `cpu.cfs_quota_us`) sat unused, because one
converter can only keep about one and a half cores busy. Cutting the pool into seven slices and
running seven converters fills the allowance and finishes in about half an hour. What the
slices then need is this: a join that produces exactly the dataset the single sequential run
would have produced, so that nothing downstream — the card, the training config, the gate — has
to know the dataset was made in pieces.

"Exactly" is the point, so the join is not a file copy:

* `episode_index` is per-dataset and every shard starts at 0, so it is renumbered;
* `index` is the frame's position in the WHOLE dataset, so it is recomputed as a running count
  rather than shifted (a shard whose episode lengths differ would otherwise leave holes);
* `task_index` points into each shard's own `tasks.jsonl`; the task STRINGS are joined and the
  column is remapped through the join, so two shards that happened to number the same
  instruction differently still end up pointing at one row;
* `meta/episodes.jsonl` and `meta/episodes_stats.jsonl` are rewritten in the new numbering, and
  `meta/info.json` gets the totals of the result, not of any shard.

The videos are hard-linked when the filesystem allows it and copied when it does not: the bytes
of an mp4 are the same bytes under either name, and a thousand episodes of 224×224 h264 is a few
gigabytes that need not exist twice. A hard link also means deleting the shards afterwards
leaves the merged dataset whole.

What this tool does NOT write is our own card — `meta/mikasa_format.json`, `meta/mikasa_episodes.csv`
and `README.md`. Those are claims about the RECORDINGS, and this tool never opens one; a shard's
card describes a seventh of the data, so copying it in would be worse than leaving it out.
`h5_to_lerobot.py --card-only` writes them from the union of the pools, recomputing the
statistics from the merged dataset itself and rebuilding the episode table from the recordings:

    python tools/vla/h5_to_lerobot.py --raw <union of the pools> --repo <out> --card-only \
        --vcodec h264 --crf 23 --stock-execution <the note the frames deserve>

That step is also the completeness check. It pairs the dataset's episodes with the pool's
recordings one by one and refuses if the counts or the lengths disagree — which is what a merge
of a shard that was still converting produces, and what nothing inside such a dataset can reveal
(see `still_being_written`, and `--expect` for the shard whose converter was killed outright).

Run from `mikasa/`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# pyarrow is imported where it is used, not here. This file travels to the fork beside the other
# recording tools, and there it must import on a plain ManiSkill environment — pyarrow lives in
# openpi's venv, the one that also has lerobot. The same split `h5_to_lerobot.py` lives with.


def lerobot_home() -> Path:
    """Where lerobot keeps its datasets — the same answer the library gives."""
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    return Path(HF_LEROBOT_HOME)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def same_features(a: dict, b: dict) -> bool:
    """Two shards hold the same thing per frame — shapes, dtypes, codec and all."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def scalar_stats(values: np.ndarray) -> dict:
    """The per-episode statistics LeRobot keeps for a scalar column, in its own shape.

    `meta/episodes_stats.jsonl` holds min/max/mean/std/count for EVERY column, the bookkeeping
    ones included — and `index` there is the episode's span in the whole dataset (episode 1 of a
    shard reads `{"min": [351], "max": [705], ...}`), while `episode_index` is the episode's own
    number. Renumbering the columns and carrying those rows over unchanged would leave a dataset
    whose data and statistics disagree about where every episode sits. They are cheap to
    recompute exactly, so they are recomputed rather than carried. The std is the population one,
    which is what LeRobot writes (std of 0..350 is 101.3245…, not the sample 101.4690…).
    """
    a = np.asarray(values)
    # int in, int out: LeRobot writes `"min": [1067]` for these columns and the untouched
    # `frame_index` row sits right beside them, so a float here is the one place the merged
    # metadata reads differently from a sequential conversion.
    cast = int if np.issubdtype(a.dtype, np.integer) else float
    v = a.astype(np.float64)
    return {"min": [cast(a.min())], "max": [cast(a.max())], "mean": [float(v.mean())],
            "std": [float(v.std())], "count": [int(v.size)]}


def still_being_written(s: Path, n_listed: int, n_video_keys: int) -> str | None:
    """Why this shard looks like a converter is still inside it — or None if it looks finished.

    This is the failure the tool exists to avoid and the one nothing else can see. LeRobot appends
    a shard's `meta/episodes.jsonl` line, its stats line and its `info.json` total only AFTER the
    parquet and both mp4s are written, so a shard caught between two episodes is a smaller dataset
    whose every internal invariant holds. A merge then joins it, prints a total and says nothing.
    The traces below are the ones that do exist while a converter is working: the `images/`
    staging directory, which LeRobot removes at the end of each `save_episode`, and files on disk
    that the metadata has not caught up with. A shard whose converter was KILLED between episodes
    leaves none of them, which is what `--expect` is for.
    """
    if (s / "images").is_dir():
        return ("meta images/ is present — LeRobot deletes it at the end of every save_episode, "
                "so a converter is still writing here")
    pq_n = len(list((s / "data").rglob("*.parquet"))) if (s / "data").is_dir() else 0
    mp4_n = len(list((s / "videos").rglob("*.mp4"))) if (s / "videos").is_dir() else 0
    if pq_n != n_listed or mp4_n != n_listed * n_video_keys:
        return (f"{n_listed} episodes in meta/episodes.jsonl but {pq_n} parquet and {mp4_n} mp4 "
                "on disk — the shard is mid-episode")
    return None


def link_or_copy(src: Path, dst: Path) -> str:
    """Hard-link `src` to `dst`, copying when the filesystem refuses; says which it did."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def merge(out_repo: str, shard_repos: list[str], *, force: bool = False,
          root: Path | None = None, expect: int | None = None, partial: bool = False) -> dict:
    """Join the shards into `out_repo`; a summary dict, and the dataset on disk."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = root or lerobot_home()
    out = root / out_repo
    shards = [root / r for r in shard_repos]
    # `--out` naming one of its own shards would delete it below, and the merged dataset's videos
    # are hard links INTO the shards, so an operator who has already removed the recordings would
    # have no second copy of those frames anywhere.
    out_r = out.resolve()
    for s, name in zip(shards, shard_repos):
        s_r = s.resolve()
        if out_r == s_r or out_r in s_r.parents or s_r in out_r.parents:
            raise SystemExit(f"--out {out_repo} is {name} (or contains it) — the merge would "
                             "delete a shard before reading it")
    for s in shards:
        for needed in ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
            if not (s / "meta" / needed).exists():
                raise SystemExit(f"{s} is not a finished LeRobot dataset (no meta/{needed}) — "
                                 "a converter that has not saved its first episode looks like this")
    if out.exists() and not force:
        raise SystemExit(f"{out} exists — pass --force to replace it")

    infos = [json.load(open(s / "meta" / "info.json")) for s in shards]
    first = infos[0]
    video_keys = [k for k, f in first["features"].items() if f["dtype"] == "video"]
    for s, info in zip(shards[1:], infos[1:]):
        for key in ("codebase_version", "fps", "robot_type", "chunks_size",
                    "data_path", "video_path"):
            if info.get(key) != first.get(key):
                raise SystemExit(f"{s} has {key}={info.get(key)!r}, the first shard has "
                                 f"{first.get(key)!r} — these are not slices of one dataset")
        if not same_features(info["features"], first["features"]):
            raise SystemExit(f"{s} holds different features than the first shard")
    # LeRobot's own metadata records the codec but not the rate factor, so two shards encoded at
    # different --crf pass every check above and join into a dataset of two picture qualities.
    # Our card is where the crf is written down; compare it wherever the shards carry one.
    crfs = {}
    for s in shards:
        card = s / "meta" / "mikasa_format.json"
        if card.exists():
            crfs[str(s)] = json.load(open(card)).get("video_crf")
    if len(set(crfs.values())) > 1:
        raise SystemExit(f"the shards were encoded at different crf ({crfs}) — one dataset "
                         "cannot hold two picture qualities")
    # A shard still being converted is a smaller, perfectly consistent dataset; refuse it here,
    # before anything is written, so a wrong run leaves nothing behind.
    for s, info in zip(shards, infos):
        why = still_being_written(s, len(read_jsonl(s / "meta" / "episodes.jsonl")), len(video_keys))
        if why and not partial:
            raise SystemExit(f"{s} is still being written: {why} — wait for its converter to "
                             "exit, or pass --partial if a short dataset is what you want")

    # The task strings are the identity; the indices are each shard's private numbering.
    tasks: list[str] = []
    task_index: dict[str, int] = {}
    for s in shards:
        for row in read_jsonl(s / "meta" / "tasks.jsonl"):
            if row["task"] not in task_index:
                task_index[row["task"]] = len(tasks)
                tasks.append(row["task"])

    chunks_size = int(first["chunks_size"])
    episodes_out: list[dict] = []
    stats_out: list[dict] = []
    per_shard: list[tuple] = []
    ep = frame = 0
    linked = copied = 0
    # Built beside the destination and moved into place at the end. Every check below can end the
    # run from inside the loop, and `meta/` is written last: building in place would leave a
    # directory with data and no metadata — unloadable — where a good dataset used to be.
    staging = out.with_name(f"{out.name}.incoming-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    for s, info in zip(shards, infos):
        ep_at_start, frame_at_start = ep, frame
        shard_tasks = {r["task_index"]: r["task"] for r in read_jsonl(s / "meta" / "tasks.jsonl")}
        eps = sorted(read_jsonl(s / "meta" / "episodes.jsonl"), key=lambda r: r["episode_index"])
        stats = {r["episode_index"]: r["stats"]
                 for r in read_jsonl(s / "meta" / "episodes_stats.jsonl")}
        for e in eps:
            old = int(e["episode_index"])
            src = s / info["data_path"].format(episode_chunk=old // chunks_size, episode_index=old)
            table = pq.read_table(src)
            n = table.num_rows
            if n != int(e["length"]):
                raise SystemExit(f"{src} holds {n} rows, its episodes.jsonl says {e['length']}")
            remap = np.array([task_index[shard_tasks[int(t)]]
                              for t in table.column("task_index").to_pylist()], dtype=np.int64)
            table = table.set_column(table.schema.get_field_index("episode_index"),
                                     "episode_index", pa.array(np.full(n, ep, dtype=np.int64)))
            table = table.set_column(table.schema.get_field_index("index"),
                                     "index", pa.array(np.arange(frame, frame + n, dtype=np.int64)))
            table = table.set_column(table.schema.get_field_index("task_index"),
                                     "task_index", pa.array(remap))
            table = table.replace_schema_metadata(pq.read_schema(src).metadata)
            dst = staging / first["data_path"].format(episode_chunk=ep // chunks_size,
                                                      episode_index=ep)
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dst)

            for key in video_keys:
                vsrc = s / info["video_path"].format(
                    episode_chunk=old // chunks_size, video_key=key, episode_index=old)
                vdst = staging / first["video_path"].format(
                    episode_chunk=ep // chunks_size, video_key=key, episode_index=ep)
                if not vsrc.exists():
                    raise SystemExit(f"{vsrc} is missing — the shard is incomplete")
                how = link_or_copy(vsrc, vdst)
                linked += how == "link"
                copied += how == "copy"

            episodes_out.append({"episode_index": ep, "tasks": e["tasks"], "length": n})
            if old not in stats:
                raise SystemExit(f"{s} has no episodes_stats row for episode {old}")
            st = dict(stats[old])
            st["index"] = scalar_stats(np.arange(frame, frame + n))
            st["episode_index"] = scalar_stats(np.full(n, ep))
            st["task_index"] = scalar_stats(remap)
            stats_out.append({"episode_index": ep, "stats": st})
            ep += 1
            frame += n
        per_shard.append((str(s), ep - ep_at_start, frame - frame_at_start))

    if expect is not None and ep != expect:
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit(f"merged {ep} episodes but --expect {expect}: "
                         + ", ".join(f"{Path(p).name}={n}" for p, n, _f in per_shard)
                         + " — a shard is short of what its converter was given")

    meta = staging / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    write_jsonl(meta / "episodes.jsonl", episodes_out)
    write_jsonl(meta / "episodes_stats.jsonl", stats_out)
    write_jsonl(meta / "tasks.jsonl",
                [{"task_index": i, "task": t} for i, t in enumerate(tasks)])
    info = dict(first)
    info["total_episodes"] = ep
    info["total_frames"] = frame
    info["total_tasks"] = len(tasks)
    info["total_videos"] = ep * len(video_keys)
    info["total_chunks"] = max(1, (ep + chunks_size - 1) // chunks_size)
    info["splits"] = {"train": f"0:{ep}"}
    with open(meta / "info.json", "w") as fh:
        json.dump(info, fh, indent=4)
    # Whole, so it can take the name.
    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)
    return dict(out=str(out), episodes=ep, frames=frame, tasks=len(tasks),
                videos=ep * len(video_keys), linked=linked, copied=copied,
                shards=[str(s) for s in shards], per_shard=per_shard)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="repo id of the merged dataset")
    ap.add_argument("--shards", required=True, nargs="+",
                    help="repo ids of the shards, IN ORDER — the merged dataset's episodes "
                         "follow this order, which is what the card's episode table assumes")
    ap.add_argument("--force", action="store_true", help="replace --out if it is already there")
    ap.add_argument("--expect", type=int, default=None,
                    help="how many episodes the whole pool holds; the merge refuses a total that "
                         "is not this, which is the only way to catch a shard whose converter was "
                         "killed between episodes (it leaves a shard that looks finished)")
    ap.add_argument("--partial", action="store_true",
                    help="merge shards that are still being written (a short dataset on purpose)")
    args = ap.parse_args()
    r = merge(args.out, args.shards, force=args.force, expect=args.expect, partial=args.partial)
    print(f"merged {len(r['shards'])} shards into {r['out']}: {r['episodes']} episodes, "
          f"{r['frames']} frames, {r['videos']} videos ({r['linked']} linked, {r['copied']} copied), "
          f"{r['tasks']} task(s)")
    for path, n_eps, n_frames in r["per_shard"]:
        print(f"  {Path(path).name}: {n_eps} episodes, {n_frames} frames")
    print("now write the card: h5_to_lerobot.py --card-only --raw <union pool> --repo "
          f"{args.out} --vcodec <codec> --crf <crf>  (it rebuilds the episode table from the "
          "pool and refuses a dataset that is short of it)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    sys.exit(main())
