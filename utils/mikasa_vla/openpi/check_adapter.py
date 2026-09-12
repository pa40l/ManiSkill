"""Push one real frame of a built LeRobot dataset through the openpi adapter.

    /workspace/openpi/.venv/bin/python tools/vla/openpi/check_adapter.py mikasa/<name>
    ... --policy /workspace/openpi/src/openpi/policies    # check the INSTALLED copy instead

`mikasa_policy.py` in this directory is the adapter we intend openpi to run; the copy under
`/workspace/openpi/src/openpi/policies/` is what it runs today. They drift, and the drift is
invisible until a training run starts: the transforms take a dict, so a missing key or a state
of the wrong width is a `KeyError` or a silent reshape at the first batch, not an import error.

This check closes that gap the only way that proves anything — with a real frame of a real
dataset rather than `make_mikasa_example()`, which is written by the same hand as the adapter
and agrees with it by construction. Found on 2026-09-12: the installed copy still declared
`MIKASA_STATE_DIM = 29` and demanded `observation/image2`, a third camera this robot has not
had since 2026-09-10 — `KeyError: 'observation/image2'` on frame 0 of every dataset we own.

Exit status is 0 when the adapter accepts the frame, 1 when it does not.
"""
import argparse
import os
import sys

import numpy as np

#: The repack map from `config.py.patch` — what the training config hands the adapter. Kept
#: here so that a patch and an adapter which disagree about a key fail HERE, loudly, instead
#: of at the first batch of a training run.
REPACK = {
    "observation/image": "image",
    "observation/wrist_image": "wrist_image",
    "observation/state": "state",
    "actions": "actions",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="a LeRobot dataset under $HF_HOME, e.g. mikasa/cam224_smoke")
    ap.add_argument("--policy", default=HERE,
                    help="directory holding mikasa_policy.py (default: this one, the repo's)")
    ap.add_argument("--openpi", default="/workspace/openpi/src")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

    sys.path.insert(0, args.openpi)
    import importlib.util

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.models import model as _model

    path = os.path.join(args.policy, "mikasa_policy.py")
    spec = importlib.util.spec_from_file_location("mikasa_policy_under_test", path)
    pol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pol)
    print(f"policy   {path}")
    print(f"         MIKASA_STATE_DIM={pol.MIKASA_STATE_DIM} MIKASA_ACTION_DIM={pol.MIKASA_ACTION_DIM}")

    ds = LeRobotDataset(args.repo_id)
    print(f"dataset  {args.repo_id}: {ds.num_episodes} episodes, {ds.num_frames} frames, {ds.fps} Hz")
    fails = []

    missing = sorted(set(REPACK.values()) - set(ds.features))
    if missing:
        print(f"  FAIL: the repack wants {missing}, which this dataset does not have")
        return 1

    item = ds[args.frame]
    data = {dst: (item[src].numpy() if hasattr(item[src], "numpy") else np.asarray(item[src]))
            for dst, src in REPACK.items()}
    data["prompt"] = "take the cup out of the open cabinet and put it on the counter"
    st, act = data["observation/state"], data["actions"]
    print(f"frame {args.frame}: state {tuple(st.shape)}, actions {tuple(act.shape)}, "
          f"image {tuple(np.asarray(data['observation/image']).shape)}, "
          f"wrist {tuple(np.asarray(data['observation/wrist_image']).shape)}")

    if st.shape[-1] != pol.MIKASA_STATE_DIM:
        fails.append(f"the dataset's state is {st.shape[-1]} wide, the adapter declares "
                     f"{pol.MIKASA_STATE_DIM}")
    if act.shape[-1] != pol.MIKASA_ACTION_DIM:
        fails.append(f"the dataset's actions are {act.shape[-1]} wide, the adapter declares "
                     f"{pol.MIKASA_ACTION_DIM}")
    if np.abs(act).max() > 1.0 + 1e-6:
        fails.append(f"an action leaves [-1,1]: max |a| = {np.abs(act).max():.4f}")

    try:
        out = pol.MikasaInputs(model_type=_model.ModelType.PI0)(data)
    except KeyError as e:
        # The whole point of using a real frame: the adapter asks for a key the dataset has
        # no reason to carry, and only a real dataset can refuse it.
        print(f"  FAIL: the adapter asks for {e} — the dataset carries "
              f"{sorted(ds.features)}")
        return 1

    imgs, mask = out["image"], out["image_mask"]
    print("adapter output:")
    print(f"  state              {tuple(out['state'].shape)} {out['state'].dtype}")
    for k in imgs:
        a = np.asarray(imgs[k])
        print(f"  {k:18s} {tuple(a.shape)} {a.dtype} mask={bool(mask[k])}"
              f"{'  (all zeros)' if not a.any() else ''}")
        if bool(mask[k]) and not a.any():
            fails.append(f"{k} is masked in but entirely zero")
        if not bool(mask[k]) and a.any():
            fails.append(f"{k} is masked out but carries pixels")
    if out["state"].dtype != np.float32:
        fails.append(f"state reaches the model as {out['state'].dtype}, openpi expects float32")
    if len(imgs) != 3:
        fails.append(f"openpi expects three image slots, the adapter filled {len(imgs)}")
    if "actions" not in out:
        fails.append("the adapter dropped `actions`, so nothing would train")

    print("FAIL:" if fails else "PASS: the adapter accepts a real frame of this dataset")
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
