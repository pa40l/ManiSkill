"""Run an openpi policy (served by `scripts/serve_policy.py`) in a MIKASA env, pd_joint_delta_pos at 20 Hz.

    # in the openpi venv, on the box with the checkpoint:
    #   uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_mikasa_retrieval --policy.dir=<ckpt>
    # here (mani_skill venv + openpi-client installed into it, or PYTHONPATH to packages/openpi-client/src):
    python eval_policy.py --env MikasaCabinetRetrieval-v0 --seeds 100-119 --video-dir /workspace/vla/step4/eval

The observation dict sent per step is exactly what the dataset holds (see h5_to_lerobot.py):
image (the head camera fetch_head), wrist_image (fetch_hand), state (15 = qpos), prompt. The policy returns a
chunk of 13-dim actions; `replan_steps` of them are executed, then a new chunk is requested.
"""
import argparse, collections, os, sys, time
sys.path.insert(0, os.environ.get("MIKASA_SRC", "/workspace/wt-int/mikasa"))
sys.path.insert(0, "/workspace/openpi/packages/openpi-client/src")
import numpy as np, torch, gymnasium as gym, imageio
import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)
# `openpi_client` lives in the policy server's venv, not in the one that records data —
# imported where it is used so this file stays importable in the recording environment
# (2026-09-10: the published branch's verify imports every module it ships).

def parse_seeds(s):
    a, _, b = s.partition("-"); return list(range(int(a), int(b) + 1)) if b else [int(a)]

def main():
    from openpi_client import websocket_client_policy as wcp

    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MikasaCabinetRetrieval-v0"); ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--replan-steps", type=int, default=5); ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--video-dir", default=""); ap.add_argument("--prompt", default="")
    args = ap.parse_args()
    env = gym.make(args.env, num_envs=1, robot_uids="mikasa_ds_fetch", control_mode="pd_joint_delta_pos", obs_mode="rgb",
                   scene_idx=0, sim_backend="cpu", render_mode="rgb_array" if args.video_dir else None)
    client = wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    print("server metadata:", client.get_server_metadata(), flush=True)
    results = []
    for seed in parse_seeds(args.seeds):
        obs, info = env.reset(seed=seed)
        u = env.unwrapped
        prompt = args.prompt or u.get_language_instruction()[0]
        plan = collections.deque(); frames = []; ok = False; t0 = time.time()
        for step in range(args.max_steps):
            if not plan:
                sd = obs["sensor_data"]
                element = {
                    "observation/image": sd["fetch_head"]["rgb"][0].cpu().numpy(),
                    "observation/wrist_image": sd["fetch_hand"]["rgb"][0].cpu().numpy(),
                    "observation/state": obs["agent"]["qpos"][0].cpu().numpy().astype(np.float32),
                    "prompt": prompt,
                }
                chunk = np.asarray(client.infer(element)["actions"])
                plan.extend(chunk[: args.replan_steps])
            action = np.asarray(plan.popleft(), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if args.video_dir:
                fr = env.render(); frames.append(np.asarray(fr[0].cpu().numpy() if hasattr(fr, "cpu") else fr).astype(np.uint8))
            if bool(info.get("success", torch.tensor([False]))[0]): ok = True; break
            if bool(info.get("fail", torch.tensor([False]))[0]) or bool(terminated[0]) or bool(truncated[0]): break
        results.append(ok); print(f"seed {seed}: {'SUCCESS' if ok else 'FAIL'} steps={step+1} {time.time()-t0:.0f}s prompt='{prompt}'", flush=True)
        if args.video_dir and frames:
            os.makedirs(args.video_dir, exist_ok=True)
            imageio.mimwrite(f"{args.video_dir}/seed{seed}_{'ok' if ok else 'fail'}.mp4", frames, fps=20, macro_block_size=1)
    print(f"=== {sum(results)}/{len(results)} ===")

if __name__ == "__main__":
    main()
