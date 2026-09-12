"""Transforms for the MIKASA bench (ds_fetch on RoboCasa kitchen 102), openpi side.

Keys the environment client sends (and the dataset repacks to):
  observation/image        the head camera (fetch_head), 224x224 uint8 -> base_0_rgb
  observation/wrist_image  gripper camera (fetch_hand), 224x224 uint8  -> left_wrist_0_rgb
  (224 is openpi's own input size, so nothing is resized on the way in; both cameras moved
   to 224 at fov 2 on 2026-09-12. The transforms below do not depend on the size -- read it
   off the dataset's feature spec, never from this comment.)
  observation/state        15 floats: obs['agent']['qpos'] as is (the supervisor's rule, 2026-09-08: qpos and images only)
  prompt                   the language instruction
Actions: 13 floats, exactly `env.step()` in control_mode="pd_joint_delta_pos"; the rate is the
dataset's (10 Hz in everything recorded so far, read from the recording, not assumed) —
arm 7 joint deltas in [-1,1] (<-> +-0.1 rad), gripper 1 absolute in [-1,1] (<-> -0.01..0.05 m),
head_pan/head_tilt/torso_lift 3 deltas in [-1,1] (<-> +-0.1 rad, rad, m), base 2 in [-1,1]
(<-> +-1 m/s, +-3.14 rad/s).
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

MIKASA_ACTION_DIM = 13
MIKASA_STATE_DIM = 15


def make_mikasa_example() -> dict:
    return {
        "observation/state": np.random.rand(MIKASA_STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Take the cup out of the open cabinet and put it on the counter.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MikasaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["observation/image"])
        wrist = _parse_image(data["observation/wrist_image"])
        # The robot has the real Fetch's two cameras (2026-09-10): the head and the
        # wrist. openpi's third slot is zeros, masked off — the pi0.5 convention for a
        # camera the robot does not have.
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist,
                      "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class MikasaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :MIKASA_ACTION_DIM])}
