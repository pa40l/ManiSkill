"""The unmodified Fetch, seen through our cameras.

Why this exists. The observations of a dataset should come from the robot a reader can get:
ManiSkill's own `fetch`, with its own joint limits (the roll joints are `continuous` here, not
clamped as in `ds_fetch`), its own inertials and its own solver behaviour. What should NOT
come from it is its camera spec — 128 x 128 is half the resolution openpi feeds the model, and
re-rendering a dataset at 128 throws away detail that costs nothing to keep.

So this is stock Fetch with two numbers changed. Both of its cameras already sit where ours do
(measured 2026-09-12: the head and the wrist camera poses agree to 0.00 cm at the same qpos)
and both already use fov 2 rad, the value the owner chose on 2026-09-12. Only width and height
move, from 128 to 224.

It is a RENDERING robot, not a planning one: mplib will not build a model from `continuous`
joints, which is the whole reason `ds_fetch` clamps them. The route is therefore
`evaluate_planner --obs-mode state` on `ds_fetch` to get the actions, then
`tools/vla/replay_to_rgb.py --robot fetch_cam224` to execute those actions here and keep the
frames. The replay runs real physics, so an episode that this robot cannot finish is dropped
rather than faked — measured on 200 Retrieval episodes, 199 reach the task's success predicate
on the unmodified robot.
"""
from dataclasses import replace

from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.fetch import Fetch

#: What the dataset holds, and what openpi's pi0.5 takes without resizing.
CAMERA_PX = 224


@register_agent()
class FetchOurCameras(Fetch):
    """`fetch`, unchanged, except that both of its cameras render at `CAMERA_PX`."""

    uid = "fetch_cam224"

    @property
    def _sensor_configs(self):
        # `CameraConfig` is a dataclass: copy each of the robot's own cameras and move the two
        # numbers. Everything else — the mount link, the pose, the fov, the near and far planes
        # — stays whatever the upstream robot declares, so this file does not go stale when it
        # changes.
        return [replace(cam, width=CAMERA_PX, height=CAMERA_PX) for cam in super()._sensor_configs]
