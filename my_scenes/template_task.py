"""A task skeleton that follows every ManiSkill convention. Copy this, not the others.

The two inherited tasks in this package violate most of these rules; the review in
docs/review-inherited-code.md lists which and why it matters. Everything below is
annotated with the rule it exists to satisfy, so the comments are the point of the
file — delete them once a real task has grown past them.

The task itself is deliberately trivial: put the cup somewhere on the counter. It
exists to be structurally correct, not interesting.

Shape borrowed from mani_skill/envs/tasks/tabletop/pick_cube.py for the contract,
and from mshab/envs/{planner,subtask}.py for the idea of naming every threshold in
a config dataclass rather than burying constants in evaluate().
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Actor, Pose

from utils.mikasa.scenes.robocasa_utils import parking_pose


@dataclass
class TemplateTaskConfig:
    """Every threshold the task uses, named and in one place.

    Magic numbers inside evaluate() are how you end up unable to say which
    criterion a failed run missed. MS-HAB puts these in SubtaskConfig; same idea.
    """

    horizon: int = 200
    "Episode length. Must match max_episode_steps in the decorator below."

    place_radius: float = 0.15
    "How close to the target the cup must be, in the counter plane."

    max_height_above_counter: float = 0.06
    "Cup base above the counter top. Catches 'still held in the air'."

    settle_lin_speed: float = 0.05
    settle_ang_speed: float = 0.2
    "Cup must have come to rest. Without this a single frame passing through scores."


# max_episode_steps is not optional: without it there is no TimeLimitWrapper and
# `truncated` out of BaseEnv.step is hardcoded False, so an RL run has no episode
# boundary at all. Upstream sets it on 69 of 69 registered tasks.
@register_env(
    "MikasaTemplate-v0",
    max_episode_steps=TemplateTaskConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class TemplateTask(BaseEnv):
    """Put the cup down on the counter."""

    # List every robot the task is actually driven with. `none` is deliberately
    # absent: with robot_uids="none" BaseEnv sets self.agent = None, and anything
    # touching self.agent below would raise on the first reset.
    SUPPORTED_ROBOTS = ["mikasa_ds_fetch", "fetch"]

    # SUPPORTED_REWARD_MODES[0] is the default mode. Listing "dense" without
    # implementing compute_dense_reward makes the first step raise; implementing it
    # without listing it means it never runs. Upstream ships a check for exactly
    # this mismatch in docs/generate_task_docs.py.
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    cfg = TemplateTaskConfig()

    cup: Actor

    def __init__(self, *args, robot_uids="mikasa_ds_fetch", scene_idx: int | None = None, **kwargs):
        # Pinning the kitchen is optional but reproducible. Left as None,
        # RoboCasaSceneBuilder draws randint(0, 120) per env.
        self.scene_idx = scene_idx
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- sensors --

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 1.6], target=[-0.1, 0, 1.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # 512, not 2048. RecordEpisode keeps every frame in host memory until
        # reset, so the render size multiplies straight into RAM: 2048 costs
        # 12.6 MB per simulation step.
        pose = sapien_utils.look_at(eye=[1.2, -1.2, 2.0], target=[0, 0, 1.0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    # ------------------------------------------------------------------ load --

    def _load_agent(self, options: dict):
        # Spawning the robot clear of the kitchen keeps gpu_init() from having to
        # resolve an interpenetration on frame one. Upstream's RoboCasa task
        # spawns at z=5 for this reason.
        super()._load_agent(options, parking_pose(self))

    def _load_scene(self, options: dict):
        """Runs once per reconfigure. Load geometry and set initial poses; nothing else.

        In particular do NOT call set_pose here. `_reconfigure` runs
        `scene._setup()` afterwards, which does `actor.set_pose(actor.initial_pose)`
        for every non-static actor and discards whatever you set.
        """
        super()._load_scene(options)

        self.scene_builder = RoboCasaSceneBuilder(self)
        if self.scene_idx is None:
            self.scene_builder.build()
        else:
            self.scene_builder.build([self.scene_idx] * self.num_envs)

        # Ask the scene builder for fixtures rather than reimplementing a lookup.
        # get_fixture resolves an exact name, a substring, or a FixtureType, and
        # can search relative to a reference fixture.
        self.counters = [
            self.scene_builder.get_fixture(
                self.scene_builder.scene_data[i]["fixtures"], "counter_main_main_group"
            )
            for i in range(self.num_envs)
        ]

        cup_path = os.path.join(
            ASSET_DIR,
            "scene_datasets/robocasa_dataset/assets/objects/objaverse/cup/cup_2/model.xml",
        )
        loader = self.scene.create_mjcf_loader()
        loader.visual_groups = [1]
        builder = loader.parse(cup_path, package_dir=os.path.dirname(cup_path))[
            "actor_builders"
        ][0]

        # Per-env spawn: each sub-scene draws its own layout, so counter positions
        # differ between envs even when the layout index is pinned. Deriving one
        # position from scene_data[0] and applying it everywhere puts objects
        # inside walls in envs 1..N-1 — silently, with no crash.
        spawn = self._counter_top(np.arange(self.num_envs)) + np.array([0.0, 0.0, 0.30])
        builder.initial_pose = sapien.Pose(p=spawn[0])
        self.cup = builder.build_dynamic(name="cup")

        # Cache the per-env target as a tensor once. Recomputing numpy in evaluate()
        # costs a host-to-device copy every step, and float64 numpy silently
        # promotes the whole comparison to float64 on CUDA.
        self.target_pos = torch.tensor(
            self._counter_top(np.arange(self.num_envs)),
            dtype=torch.float32,
            device=self.device,
        )

    def _counter_top(self, env_idx) -> np.ndarray:
        """Centre of the counter top, per env, shape (len(env_idx), 3)."""
        out = []
        for i in np.atleast_1d(env_idx):
            counter = self.counters[int(i)]
            pos = np.asarray(counter.pos, dtype=np.float64).copy()
            size = np.asarray(counter.size, dtype=np.float64)
            pos[2] += size[2] / 2
            out.append(pos)
        return np.stack(out).astype(np.float32)

    # ------------------------------------------------------------ initialize --

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Runs on every reset, for the envs in env_idx only.

        Three rules, all visible below:
          - wrap in `with torch.device(self.device)` so bare torch calls land right;
          - size everything by `b = len(env_idx)`, never self.num_envs;
          - call scene_builder.initialize(env_idx), which is what restores the
            robot's qpos and base pose. Without it the arm starts episode 2 frozen
            wherever episode 1 left it.

        Note you do not index [env_idx] when *setting* sim state — set_pose and
        friends are masked internally. You do index it when reading a cached
        per-env buffer, or when writing your own task state.
        """
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # Objects are not reset for you. _clear_sim_state zeroes velocities;
            # poses are the task's responsibility, and forgetting this means
            # episode 2 starts wherever episode 1 stopped.
            spawn = self.target_pos[env_idx].clone()
            spawn[:, 2] += 0.30
            spawn[:, :2] += (torch.rand((b, 2)) - 0.5) * 0.1
            self.cup.set_pose(Pose.create_from_pq(p=spawn))

            # Own task state must be masked by hand — this is not sim state.
            self.has_been_lifted[env_idx] = False

    def _after_reconfigure(self, options: dict):
        # Task-owned buffers are allocated here, once, at full width.
        self.has_been_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return super()._after_reconfigure(options)

    # -------------------------------------------------------------- evaluate --

    def evaluate(self) -> dict:
        """Runs every step. Everything returned must be batched.

        Return the intermediates, not just `success`. They are merged into `info`,
        which is threaded into _get_obs_extra and the reward functions — and when a
        run fails, they are the only way to say which criterion missed.
        """
        cup_pos = self.cup.pose.p

        xy_distance = torch.linalg.norm(cup_pos[:, :2] - self.target_pos[:, :2], dim=1)
        height_above = cup_pos[:, 2] - self.target_pos[:, 2]
        is_grasped = self.agent.is_grasping(self.cup)

        on_counter = xy_distance <= self.cfg.place_radius
        low_enough = height_above <= self.cfg.max_height_above_counter
        settled = self.cup.is_static(
            lin_thresh=self.cfg.settle_lin_speed, ang_thresh=self.cfg.settle_ang_speed
        )

        self.has_been_lifted = self.has_been_lifted | is_grasped

        return {
            "success": on_counter & low_enough & settled & ~is_grasped & self.has_been_lifted,
            "on_counter": on_counter,
            "low_enough": low_enough,
            "settled": settled,
            "is_grasped": is_grasped,
            "has_been_lifted": self.has_been_lifted,
            "xy_distance": xy_distance,
            "height_above": height_above,
        }

    # ------------------------------------------------------------------- obs --

    def _get_obs_extra(self, info: dict) -> dict:
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            is_grasped=info["is_grasped"],
        )
        # Ground truth only in state mode, or an image-based policy gets to cheat.
        if self.obs_mode_struct.use_state:
            obs.update(
                cup_pose=self.cup.pose.raw_pose,
                cup_to_target=self.target_pos.to(self.device) - self.cup.pose.p,
            )
        return obs

    # ----------------------------------------------------------------- state --

    def get_state_dict(self) -> dict:
        """Task memory has to be checkpointed with the sim state.

        The default returns only scene.get_sim_state(), so a latched flag would be
        dropped by env.set_state(env.get_state()) — which breaks trajectory replay
        and any branch-and-restore. For a memory benchmark this is the state that
        matters most.
        """
        state = super().get_state_dict()
        state["has_been_lifted"] = self.has_been_lifted.clone()
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        # BaseEnv.set_state_dict takes env_idx; envs/template.py's example drops it.
        # Forward it — partial state restore is a real use.
        self.has_been_lifted = state["has_been_lifted"].clone().to(self.device)
        super().set_state_dict(state, env_idx)
