"""A stand-in policy for checking the evaluation loop without a trained checkpoint.

    python tools/vla/eval_policy.py --policy dummy --seeds 1100-1101 --max-steps 40

The point is not the actions — it is the contract. `eval_policy.py` builds an observation
dict out of the live env and feeds the policy's answer straight to `env.step`, and between
those two lines sit every assumption a training run makes as well: which keys the policy is
handed, what shape and dtype each one carries, how wide the action vector comes back and in
whose units. A trained checkpoint hides a mistake there behind bad behaviour — the robot
simply does the wrong thing and one blames the model. This stand-in names it instead.

Checked on every call, and raising with the offending key and what was seen:
  * the keys are exactly the four the dataset holds: `observation/image`,
    `observation/wrist_image`, `observation/state`, `prompt`;
  * both images are uint8 `(H, W, 3)`, square, and the same size as each other — and as
    `expect_image` when given, which is the dataset card's number;
  * the state is float32 of width `expect_state` (15: `obs["agent"]["qpos"]`, nothing else);
  * the prompt is a non-empty string.

Returned: a chunk of `horizon` actions of the env's own width, either all zeros (the robot
holds still) or small arm jitter. Neither solves anything, on purpose: what is exercised is
the loop — the env's own action clipping, the replan bookkeeping, the video writer — not the
policy. A run that ends with every episode failing and no contract error is a PASS for this
tool: it means the plumbing carries whatever a real checkpoint will send.
"""
from __future__ import annotations

import numpy as np

#: What the dataset holds, and therefore what a policy is handed. `h5_to_lerobot.py` writes
#: exactly these, and `MikasaInputs` in the openpi adapter reads exactly these.
KEYS = ("observation/image", "observation/wrist_image", "observation/state", "prompt")


class ContractError(AssertionError):
    """The observation the client built is not the one the dataset describes."""


class DummyPolicy:
    def __init__(self, *, action_dim: int = 13, horizon: int = 10, motion: str = "still",
                 expect_image: int | None = None, expect_state: int = 15, seed: int = 0):
        self.action_dim, self.horizon, self.motion = action_dim, horizon, motion
        self.expect_image, self.expect_state = expect_image, expect_state
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.seen: dict[str, tuple] = {}

    # `infer` and nothing else: the same call the websocket client answers, so
    # `eval_policy.py` does not know which of the two it is talking to.
    def infer(self, element: dict) -> dict:
        self._check(element)
        self.calls += 1
        chunk = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        if self.motion == "jitter":
            # small, inside the arm's own +-1 normalised range; the body, gripper and base
            # stay at zero so nothing drives into the scene while the plumbing is tested
            chunk[:, :7] = self.rng.uniform(-0.05, 0.05, size=(self.horizon, 7)).astype(np.float32)
        return {"actions": chunk}

    def _check(self, element: dict) -> None:
        missing = [k for k in KEYS if k not in element]
        extra = [k for k in element if k not in KEYS]
        if missing or extra:
            raise ContractError(f"keys: missing {missing}, unexpected {extra}; the dataset holds {list(KEYS)}")

        shapes = {}
        for key in ("observation/image", "observation/wrist_image"):
            img = np.asarray(element[key])
            if img.dtype != np.uint8:
                raise ContractError(f"{key}: dtype {img.dtype}, the dataset holds uint8")
            if img.ndim != 3 or img.shape[2] != 3:
                raise ContractError(f"{key}: shape {img.shape}, want (H, W, 3)")
            if img.shape[0] != img.shape[1]:
                raise ContractError(f"{key}: {img.shape[0]}x{img.shape[1]} is not square")
            if self.expect_image and img.shape[0] != self.expect_image:
                raise ContractError(f"{key}: {img.shape[0]} px, the dataset card says {self.expect_image}")
            shapes[key] = img.shape
        if shapes["observation/image"][:2] != shapes["observation/wrist_image"][:2]:
            raise ContractError(f"the two cameras differ in size: {shapes}")

        state = np.asarray(element["observation/state"])
        if state.dtype != np.float32:
            raise ContractError(f"observation/state: dtype {state.dtype}, the dataset holds float32")
        if state.shape != (self.expect_state,):
            raise ContractError(f"observation/state: shape {state.shape}, want ({self.expect_state},)")
        if not np.all(np.isfinite(state)):
            raise ContractError("observation/state holds a non-finite value")

        prompt = element["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ContractError(f"prompt: {prompt!r}, want a non-empty string")

        self.seen = {"image": shapes["observation/image"], "wrist_image": shapes["observation/wrist_image"],
                     "state": state.shape, "prompt_chars": len(prompt)}

    def report(self) -> str:
        return (f"dummy policy: {self.calls} inference calls, contract held; "
                f"image {self.seen.get('image')}, wrist {self.seen.get('wrist_image')}, "
                f"state {self.seen.get('state')}, prompt {self.seen.get('prompt_chars')} chars")
