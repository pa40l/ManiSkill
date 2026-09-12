"""Run an openpi policy (served by `scripts/serve_policy.py`) in a MIKASA env, pd_joint_delta_pos at the rate the dataset was recorded at (`--control-freq`, default 10).

    # in the openpi venv, on the box with the checkpoint:
    #   uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_mikasa_retrieval --policy.dir=<ckpt>
    # here (mani_skill venv + openpi-client installed into it, or PYTHONPATH to packages/openpi-client/src):
    python eval_policy.py --env MikasaCabinetRetrieval-v0 --seeds 100-119 --video-dir /workspace/vla/step4/eval

The observation dict sent per step is exactly what the dataset holds (see h5_to_lerobot.py):
image (the head camera fetch_head), wrist_image (fetch_hand), state (15 = qpos), prompt. The policy returns a
chunk of 13-dim actions; `replan_steps` of them are executed, then a new chunk is requested.
"""
import argparse, collections, os, pathlib, sys, time
# The tree this file belongs to, not a path that happened to be current when it was written:
# `tools/vla/eval_policy.py` sits two directories under `mikasa/`. MIKASA_SRC still overrides,
# for an odd layout or a deliberately older tree.
sys.path.insert(0, os.environ.get("MIKASA_SRC", str(pathlib.Path(__file__).resolve().parents[2])))
sys.path.insert(0, "/workspace/openpi/packages/openpi-client/src")
import numpy as np, torch, gymnasium as gym, imageio
import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)
# `openpi_client` lives in the policy server's venv, not in the one that records data —
# imported where it is used so this file stays importable in the recording environment
# (2026-09-10: the published branch's verify imports every module it ships).

def parse_seeds(s):
    a, _, b = s.partition("-"); return list(range(int(a), int(b) + 1)) if b else [int(a)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MikasaCabinetRetrieval-v0"); ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--replan-steps", type=int, default=5); ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--video-dir", default=""); ap.add_argument("--prompt", default="")
    ap.add_argument("--control-freq", type=int, default=10,
                    help="the env's control frequency, which MUST be the rate the dataset was "
                         "recorded at (10 in everything recorded so far; the card's "
                         "control_freq_hz says). ManiSkill's own default is 20, and a policy "
                         "trained on 10 Hz data driving a 20 Hz env covers half the distance per "
                         "step that it learned to — without any error to notice")
    ap.add_argument("--policy", default="server", choices=("server", "dummy"),
                    help="'server' talks to a served checkpoint over the websocket; 'dummy' runs "
                         "tools/vla/dummy_policy.py in-process, which checks the observation "
                         "contract and answers with a chunk of zeros or small jitter. The dummy "
                         "needs no checkpoint and no openpi venv: it is how one tells a broken "
                         "harness from a bad policy before spending a training run")
    ap.add_argument("--horizon", type=int, default=10, help="dummy only: actions per chunk")
    ap.add_argument("--dummy-motion", default="still", choices=("still", "jitter"),
                    help="dummy only: hold still, or jitter the arm inside +-0.05 of its range")
    ap.add_argument("--expect-image", type=int, default=0,
                    help="dummy only: the camera size the dataset card names (224); 0 accepts "
                         "whatever the env produces and only checks the two cameras agree")
    args = ap.parse_args()
    env = gym.make(args.env, num_envs=1, robot_uids="mikasa_ds_fetch", control_mode="pd_joint_delta_pos", obs_mode="rgb",
                   scene_idx=0, sim_backend="cpu", sim_config=dict(control_freq=args.control_freq),
                   render_mode="rgb_array" if args.video_dir else None)
    print(f"env at {args.control_freq} Hz control, {1 / args.control_freq:.3f} s a step", flush=True)
    if args.policy == "dummy":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dummy_policy import DummyPolicy
        client = DummyPolicy(horizon=args.horizon, motion=args.dummy_motion,
                             expect_image=args.expect_image or None)
        print("dummy policy in-process: no checkpoint, the observation contract is what is checked",
              flush=True)
    else:
        from openpi_client import websocket_client_policy as wcp
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
            imageio.mimwrite(f"{args.video_dir}/seed{seed}_{'ok' if ok else 'fail'}.mp4", frames,
                             fps=args.control_freq, macro_block_size=1)
    print(f"=== {sum(results)}/{len(results)} ===")
    if args.policy == "dummy":
        # The point of the dummy run: the loop carried every observation and every action
        # without a contract error. Episodes failing is expected — it solves nothing.
        print(client.report(), flush=True)

if __name__ == "__main__":
    main()
