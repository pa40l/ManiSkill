"""MikasaStackRecall-v0: unstack the cubes, take the bottom one, restack THE SAME.

The design blank is docs/task-designs/G-stackrecall-v0.md; the story is the
user's (2026-09-01): things in cupboards sit stacked on each other — to get
the one you need you first remove the ones in the way, and then you put them
back **in the order they were**. The memory is ORDER (K106's cabinet, W13's
shelf): a stack of three colored cubes stands on the wall cabinet's shelf, the
TARGET is the bottom one, and success is the target standing on the counter's
place target while the two movable cubes stand restacked on the shelf in their
original relative order.

Why this is a memory task and not an observation task: the cue (the stack, its
order readable straight off the cube heights) is destroyed by the agent's own
required actions (the SameDrawer pattern), and the one external cheat that
survives — encoding the order in the counter layout while unstacking — is
erased by the SCRAMBLE: the moment the target cube stands placed, the env
teleports both movable cubes to seeded counter slots whose assignment is drawn
independently of the answer. From that frame on, the original order exists
nowhere in the world but the agent's memory. The no-memory floor is 1/2 x
motor SR (two movable cubes, two orders).

Guessing is priced by an unwinnability latch: the FIRST movable-on-movable
rest inside the cabinet box is final — the wrong pair latches `wrong_pair` and
the episode cannot be won (a long-horizon task without the latch lets a
memoryless agent try both orders).

Everything geometric is measured territory: shelf z=1.4200, side-grasp band
x[2.35,2.65] y[-0.28,-0.10] (W13), door held open at 1.75 (K105/K106 drift
margins), place band y=-0.525 (water_plants). The cube-specific legs (side
grasp at stack heights, releasing a cube ON a cube through the opening) are
measured by W17 before the oracle relies on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Pose

from utils.mikasa.scenes.robocasa_utils import require_get_fixture, restore_task_tensor, parking_pose

#: Keys `get_state_dict` adds beyond the simulator state. Every one is task
#: MEMORY (the design blank's point 8): dropping any turns replayed successes
#: into failures.
TASK_STATE_KEYS = (
    "original_order", "target_placed", "scrambled", "wrong_pair",
    "succeeded", "was_lifted", "place_target", "scramble_pos",
)

INSTRUCTIONS = (
    "Unstack the cubes, put the bottom one on the counter, and restack the "
    "others on the shelf in the same order.",
    "Take the bottom cube out of the cabinet stack and put the rest back "
    "exactly as they were.",
    "Fetch the lowest cube of the stack; return the other cubes to the shelf "
    "in their original order.",
)

#: Fixed identity palette: cube i ALWAYS wears color i (you cannot recolor an
#: actor per episode — the example task's measured note); the episode ticket
#: permutes which LEVEL each cube occupies, never which color it is.
CUBE_COLORS = (
    (1.0, 0.1, 0.1, 1.0),   # red
    (0.1, 0.8, 0.1, 1.0),   # green
    (0.15, 0.3, 1.0, 1.0),  # blue
)


@dataclass
class StackRecallConfig:
    """Every threshold named; magic numbers in evaluate() are how failures go mute."""

    horizon: int = 6000
    """PROVISIONAL: five carries plus dock drives against K108's ~2100 for one
    carry with the door work. K22 against the final oracle before any
    published sweep."""

    # -- the cabinet (measured, cabinet_retrieval territory) -------------------
    cabinet_name: str = "cab_main_main_group"
    counter_name: str = "counter_main_main_group"
    door_hinge: str = "rightdoorhinge"
    door_open_rad: float = 1.75
    """The door starts (and stays) open — this task adds ONLY memory and
    stacking on top of the measured motor substrate. 1.75 is the cabinet
    family's K105 angle, and it is load-bearing here twice over: the east
    cargo trip passed under it across twenty solo runs (at 2.2 the panel
    swings east into the z~1.5 carry corridor and the cargo rotate grinds
    `gripper<->hingerightdoor`, solo 22), and the panel's x[2.74, 2.82]
    shadow is out of every corridor this task still uses (the east staging
    plinth that fought it is gone — the scramble slots live on the counter).
    A closed-door variant is a later knob."""

    # -- the cubes and their stack ---------------------------------------------
    n_cubes: int = 3
    "v0 is fixed at 3 (one target + two movable = a one-bit answer)."
    cube_half: float = 0.0225
    """4.5 cm cube: inside the gripper's ~6.4 cm opening (W13 pads), big enough
    that three stacked stay under the cabinet top (1.42 + 0.135 << 2.31)."""
    shelf_top_z: float = 1.4200
    "The cabinet's interior floor (W13, mesh-bottom at settle)."
    spawn_clearance: float = 0.005
    "The bottom cube's underside starts this far above the shelf; settle drops it."
    stack_gap: float = 0.002
    "Vertical daylight between stacked cubes at spawn — no initial penetration."
    stack_depth: float = -0.19
    stack_jitter_depth: float = 0.04
    """Stack y: [-0.23, -0.15], the middle of W13's graspable depth band —
    narrower than the cup's because a stack needs its whole footprint on the
    shelf and the probe's grasp heights were measured mid-band."""
    stack_margin_x: float = 0.13
    stack_jitter_x: float = 0.04
    """Stack x: the jitter runs around the exposed half's CENTRE (2.50 on
    kitchen 102) -> x in [2.44, 2.56]; _load_scene asserts the derived band
    sits inside W13's measured graspable [2.35, 2.65] at build time (the
    margin shapes the centre only through the fixture arithmetic)."""
    west_dx: float = 0.16
    """The WEST staging plinth sits at stack_x - west_dx (clear of the open
    door's x-shadow at 2.74+ and of the sink wall at 2.25). The scramble
    teleports the two movables INTO A STACK on its top, in a seeded-flip
    order that is independent of the answer — the layout after the scramble
    carries no order information, and every post-scramble grasp lands at the
    measured-solid shelf heights (1.49-1.53, the W17 A/B forms) instead of
    the counter fetches that flaked a per-binary coin (sweeps 10-14,
    K109)."""
    temp_dx: float = 0.16
    temp_across: float = -0.55
    temp_col_half_h: float = 0.19
    temp_col_half_xy: float = 0.035
    """The temp peg — the third Hanoi spot for the case where the remembered
    BOTTOM cube lies under the other in the scramble stack — is a slim
    KINEMATIC COLUMN standing ON THE COUNTER south of the cabinet face
    (top at counter + 2*temp_col_half_h = 1.30). Sweep 15/16 measured every
    shelf-side temp position invading a grasp corridor (the pegs and the
    west approach share the shelf's one usable strip); the counter column's
    corridors live entirely in free air — south of the cabinet face, under
    its bottom rail, far west of the door shadow and the stove."""
    stack_xy_tol: float = 0.018
    """How far (xy) a cube's centre may sit off the cube below it and still
    count as stacked — under cube_half so the upper cube genuinely rests on
    the lower, not on its edge."""
    stack_dz_tol: float = 0.012
    "Tolerance on the stacked centre-to-centre height (2 x cube_half)."
    home_xy_tol: float = 0.035
    """How far (xy) the restacked pair's BOTTOM cube may sit off the home
    plinth's centre. Wider than stack_xy_tol on purpose: the cube-on-cube
    check guards against edge-balancing, but the home check only asks
    "standing ON the plinth", and a DRIVEN seat accumulates execution error
    the probe's parked seats never had (solo 29: the full chain restacked
    pair_correct=True and died on the old 0.018 here). At 0.035 the cube's
    footprint still overlaps the plinth top by over two thirds."""

    # -- the place (target cube) and the scramble ------------------------------
    place_across: float = -0.525
    "World y of the target's place band (open sky between overhang and edge)."
    place_radius: float = 0.15
    max_height_above_counter: float = 0.06
    min_height_above_counter: float = -0.02
    """Lower bound on the target's mesh-bottom-vs-counter-top: without it a
    cube that tumbles off the edge and rests on the FLOOR still passes the
    radius check and fires the irreversible scramble."""
    settle_lin_speed: float = 0.05
    settle_ang_speed: float = 0.2

    # -- the robot start -------------------------------------------------------
    start_xy: tuple = (2.55, -1.90)
    start_yaw_deg: float = 90.0
    start_jitter_xy: float = 0.08
    start_jitter_yaw: float = 0.10

    def validate(self) -> None:
        assert self.horizon > 0
        assert self.n_cubes == 3, (
            f"n_cubes={self.n_cubes}: v0 is designed and floored for exactly 3 "
            "(one target, two movable, 2 orders). More cubes is the family's "
            "difficulty knob and needs its own floor, spawn heights and probe."
        )
        assert 0.9 <= self.door_open_rad <= 2.9, (
            f"door_open_rad={self.door_open_rad}: under 0.9 the opening does "
            "not admit the arm (W12); 3.0 is the hinge stop. This task has no "
            "closed-door mode — that is the cabinet_retrieval family's knob."
        )
        assert 0 < self.cube_half <= 0.028, (
            f"cube_half={self.cube_half}: the gripper pads open ~6.4 cm (W13) "
            "and a cube over 5.6 cm cannot be side-grasped."
        )
        top_of_stack = self.shelf_top_z + self.n_cubes * 2 * self.cube_half + 0.02
        assert top_of_stack < 2.31, "the stack pokes through the cabinet top"
        assert self.spawn_clearance > 0 and self.stack_gap >= 0
        assert -0.28 <= self.stack_depth - self.stack_jitter_depth and \
               self.stack_depth + self.stack_jitter_depth <= -0.10, (
            "the stack depth band leaves W13's measured graspable band"
        )
        assert self.stack_margin_x + self.stack_jitter_x < 0.25
        assert 0 < self.stack_xy_tol < self.cube_half, (
            "stack_xy_tol at or over cube_half calls an edge-balanced cube stacked"
        )
        assert 0 < self.stack_dz_tol < self.cube_half
        assert self.stack_xy_tol <= self.home_xy_tol < 2 * self.cube_half
        assert self.place_radius > 0 and self.max_height_above_counter > 0
        assert self.min_height_above_counter < 0
        assert -0.65 < self.place_across < -0.448

        assert self.west_dx > 4 * self.cube_half + 0.03, (
            "the west plinth too close to the home column"
        )
        assert self.stack_jitter_x + self.west_dx + self.cube_half <= 0.245, (
            "the west plinth can leave the exposed half"
        )
        assert -0.65 + self.temp_col_half_xy < self.temp_across < -0.448, (
            "the temp column must stand in the open-sky counter band"
        )
        assert self.temp_across < -0.45 - self.cube_half, (
            "the temp column must stay south of the cabinet face"
        )
        assert abs(self.temp_across - self.place_across) >= 0.0, (
            "informational: the place disc check below is the real guard"
        )
        assert self.temp_dx - self.temp_col_half_xy - self.cube_half > 0.02, (
            "the temp column crowds the place column"
        )


@register_env(
    "MikasaStackRecall-v0",
    max_episode_steps=StackRecallConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class StackRecallTask(BaseEnv):
    """Unstack, retrieve the bottom cube, restack the rest in the original order."""

    SUPPORTED_ROBOTS = ["mikasa_ds_fetch", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    cfg = StackRecallConfig()

    def __init__(self, *args, robot_uids="mikasa_ds_fetch", scene_idx: int | None = 0, **kwargs):
        # Kitchen 102 pinned: every number in the config was measured there.
        self.cfg.validate()
        self.scene_idx = scene_idx
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- sensors --

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        # The cabinet_retrieval framing: open door, cabinet interior, counter,
        # robot — the stack lives exactly where the cup did.
        pose = sapien_utils.look_at(eye=[2.25, -2.75, 2.05], target=[2.35, -0.40, 1.30])
        return CameraConfig("render_camera", pose, 512, 512, 1.2, 0.01, 100)

    # ------------------------------------------------------------------ load --

    def _load_agent(self, options: dict):
        super()._load_agent(options, parking_pose(self))

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        self.scene_builder = RoboCasaSceneBuilder(self)
        if self.scene_idx is None:
            self.scene_builder.build()
        else:
            self.scene_builder.build([self.scene_idx] * self.num_envs)
        self._fix_ds_fetch_collision_bits()

        self.counters = [
            require_get_fixture(
                self.scene_builder, self.scene_builder.scene_data[i]["fixtures"],
                self.cfg.counter_name, scene_idx=self.scene_idx,
            )
            for i in range(self.num_envs)
        ]
        self.cabinets = [
            require_get_fixture(
                self.scene_builder, self.scene_builder.scene_data[i]["fixtures"],
                self.cfg.cabinet_name, scene_idx=self.scene_idx,
            )
            for i in range(self.num_envs)
        ]

        # The door articulation, resolved once (the cabinet_retrieval pattern —
        # this file holds the second kitchen-articulation qpos write on reset).
        matches = [
            (name, art) for name, art in self.scene.articulations.items()
            if self.cfg.cabinet_name in name
        ]
        assert len(matches) == self.num_envs, (
            f"expected one {self.cfg.cabinet_name!r} articulation per env, "
            f"found {[n for n, _ in matches]!r}"
        )
        self._door = []
        for _name, art in sorted(matches):
            names = [j.name for j in art.get_active_joints()]
            assert self.cfg.door_hinge in names, (_name, names)
            self._door.append((art, names.index(self.cfg.door_hinge)))

        # The cubes. Identity (color) is build-time and permanent; the episode
        # permutes LEVELS. Parked apart at load; _initialize_episode stacks them.
        self.cubes = [
            actors.build_cube(
                self.scene,
                half_size=self.cfg.cube_half,
                color=list(CUBE_COLORS[i]),
                name=f"cube_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[2.5, -0.19, 1.60 + 0.10 * i]),
            )
            for i in range(self.cfg.n_cubes)
        ]

        # The plinths: kinematic, grey, one footprint each. Poses are set per
        # episode (the stack jitters); kinematic bodies take set_pose freely.
        self.plinths = [
            actors.build_cube(
                self.scene,
                half_size=self.cfg.cube_half,
                color=[0.45, 0.45, 0.45, 1.0],
                name="plinth_0",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[2.5, -0.19, 1.30]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.cfg.cube_half,
                color=[0.45, 0.45, 0.45, 1.0],
                name="plinth_1",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[2.34, -0.19, 1.30]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[self.cfg.temp_col_half_xy,
                            self.cfg.temp_col_half_xy,
                            self.cfg.temp_col_half_h],
                color=[0.55, 0.55, 0.55, 1.0],
                name="plinth_2",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[2.33, -0.55, 1.11]),
            ),
        ]

        half_lo, half_hi = self._exposed_half()
        centre_x = (half_lo + half_hi) / 2.0
        # The derived spawn band must sit inside W13's MEASURED graspable band
        # — a runtime check because the centre comes off the fixture, and a
        # moved fixture would silently spawn ungraspable stacks.
        assert 2.35 <= centre_x - self.cfg.stack_jitter_x \
            and centre_x + self.cfg.stack_jitter_x <= 2.65, (
                f"stack x band [{centre_x - self.cfg.stack_jitter_x:.2f}, "
                f"{centre_x + self.cfg.stack_jitter_x:.2f}] leaves W13's "
                "measured graspable band [2.35, 2.65]"
            )
        counter_tops = np.array(
            [float(np.asarray(c.pos)[2] + np.asarray(c.size)[2] / 2.0)
             for c in self.counters], dtype=np.float32,
        )
        self._counter_top_z = torch.as_tensor(counter_tops, device=self.device)
        self._spawn_centre_x = torch.full(
            (self.num_envs,), float(centre_x), device=self.device
        )
        self.place_target = torch.zeros((self.num_envs, 3), device=self.device)

    def _exposed_half(self) -> tuple[float, float]:
        cab = self.cabinets[0]
        pos = np.asarray(cab.pos, dtype=np.float64)
        size = np.asarray(cab.size, dtype=np.float64)
        lo = float(pos[0])
        hi = float(pos[0] + size[0] / 2.0)
        return lo + self.cfg.stack_margin_x, hi - self.cfg.stack_margin_x

    def _fix_ds_fetch_collision_bits(self):
        if self.robot_uids == "fetch" or self.agent is None:
            return
        for link in self.agent.robot.links:
            for body in link._bodies:
                for shape in body.get_collision_shapes():
                    groups = shape.get_collision_groups()
                    for bit in range(25, 30):
                        groups[2] |= 1 << bit
                    shape.set_collision_groups(groups)

    # ------------------------------------------------------------ initialize --

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            self._restore_robot(env_idx)

            for i in env_idx.tolist():
                art, j = self._door[int(i)]
                q = art.get_qpos()
                q = q.clone() if torch.is_tensor(q) else torch.as_tensor(q).clone()
                q.reshape(-1)[j] = self.cfg.door_open_rad
                art.set_qpos(q)

            # The episode ticket: level permutation (the ANSWER), stack jitter,
            # scramble slot assignment — all from the batched episode RNG.
            perms, jitters, slot_flips = [], [], []
            for i in env_idx.tolist():
                rng = self._batched_episode_rng[i]
                perms.append(rng.permutation(self.cfg.n_cubes))
                jitters.append(rng.uniform(-1.0, 1.0, size=3))
                slot_flips.append(rng.randint(0, 2))
            perm_t = torch.as_tensor(np.stack(perms), dtype=torch.long)
            jit_t = torch.as_tensor(np.stack(jitters), dtype=torch.float32)
            flip_t = torch.as_tensor(np.asarray(slot_flips), dtype=torch.long)
            self.original_order[env_idx] = perm_t

            sx = self._spawn_centre_x[env_idx] + jit_t[:, 0] * self.cfg.stack_jitter_x
            sy = self.cfg.stack_depth + jit_t[:, 1] * self.cfg.stack_jitter_depth

            # The plinths first: middle under the stack, staging at +-dx.
            # Kinematic — their pose IS their truth, no settle needed.
            plinth_z = self.cfg.shelf_top_z + self.cfg.cube_half + 0.001
            pp = torch.zeros((b, 3))
            pp[:, 0] = sx
            pp[:, 1] = sy
            pp[:, 2] = plinth_z
            self.plinths[0].set_pose(Pose.create_from_pq(p=pp))
            pw = torch.zeros((b, 3))
            pw[:, 0] = sx - self.cfg.west_dx
            pw[:, 1] = sy
            pw[:, 2] = plinth_z
            self.plinths[1].set_pose(Pose.create_from_pq(p=pw))
            pt = torch.zeros((b, 3))
            pt[:, 0] = sx - self.cfg.temp_dx
            pt[:, 1] = self.cfg.temp_across
            pt[:, 2] = self._counter_top_z[env_idx] + self.cfg.temp_col_half_h
            self.plinths[2].set_pose(Pose.create_from_pq(p=pt))
            plinth_top = plinth_z + self.cfg.cube_half

            # Stack the cubes by LEVEL on the middle plinth:
            # original_order[:, k] is the cube index at level k (0 = bottom =
            # target, standing at the measured B-form height 1.4875). One
            # batched (b, 3) set_pose per cube — a full-width raw-pose write
            # would break GPU partial resets.
            h = 2.0 * self.cfg.cube_half
            for ci, cube in enumerate(self.cubes):
                level = (perm_t == ci).float().argmax(dim=1).float()
                p = torch.zeros((b, 3))
                p[:, 0] = sx
                p[:, 1] = sy
                p[:, 2] = plinth_top + self.cfg.spawn_clearance \
                    + self.cfg.cube_half + level * (h + self.cfg.stack_gap)
                cube.set_pose(Pose.create_from_pq(p=p))

            # The target's place point: under the stack column on the
            # counter. The place moved back WEST from the far-east stove span
            # (K109): the east half of this kitchen is a minefield of 3-7 cm
            # margins (stove, microwave box, door shadow) that flipped a
            # per-binary coin on every cargo leg; the memory pause is carried
            # by the ~500-step manipulation between the scramble and the
            # first restack commitment (weaker than the road-borne 600+, and
            # said so in the design blank — still far beyond any baseline
            # frame stack).
            tgt = torch.zeros((b, 3))
            tgt[:, 0] = sx
            tgt[:, 1] = self.cfg.place_across
            tgt[:, 2] = self._counter_top_z[env_idx]
            self.place_target[env_idx] = tgt

            # Scramble slots, assigned to the movable cubes in a seeded order
            # INDEPENDENT of the answer (flip 0/1) — the post-scramble layout
            # must not encode which cube sat lower.
            self.scramble_pos[env_idx] = 0.0
            for row, i in enumerate(env_idx.tolist()):
                m1, m2 = int(perm_t[row, 1]), int(perm_t[row, 2])
                order = (m1, m2) if int(flip_t[row]) == 0 else (m2, m1)
                west_top = plinth_z + self.cfg.cube_half
                for si, ci in enumerate(order):
                    self.scramble_pos[i, ci, 0] = float(sx[row]) - self.cfg.west_dx
                    self.scramble_pos[i, ci, 1] = float(sy[row])
                    self.scramble_pos[i, ci, 2] = west_top \
                        + self.cfg.cube_half + 0.003 \
                        + si * (2 * self.cfg.cube_half + 0.002)

            base = torch.zeros((b, 3))
            base[:, 0] = self.cfg.start_xy[0] + jit_t[:, 0] * self.cfg.start_jitter_xy
            base[:, 1] = self.cfg.start_xy[1] + jit_t[:, 1] * self.cfg.start_jitter_xy
            base[:, 2] = math.radians(self.cfg.start_yaw_deg) \
                + jit_t[:, 2] * self.cfg.start_jitter_yaw
            qpos = self.agent.robot.get_qpos()
            qpos[env_idx, 0] = base[:, 0]
            qpos[env_idx, 1] = base[:, 1]
            qpos[env_idx, 2] = base[:, 2]
            self.agent.robot.set_qpos(qpos[env_idx])

            self.target_placed[env_idx] = False
            self.scrambled[env_idx] = False
            self.wrong_pair[env_idx] = False
            self.succeeded[env_idx] = False
            self.was_lifted[env_idx] = False

    def _restore_robot(self, env_idx: torch.Tensor):
        keyframe = self.agent.keyframes["rest"]
        qpos = torch.as_tensor(np.asarray(keyframe.qpos, dtype=np.float32))
        self.agent.robot.set_qpos(qpos.unsqueeze(0).repeat(len(env_idx), 1))
        self.agent.robot.set_root_pose(sapien.Pose())

    def _after_reconfigure(self, options: dict):
        n = self.cfg.n_cubes
        self.original_order = torch.zeros(
            (self.num_envs, n), dtype=torch.long, device=self.device)
        self.target_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.scrambled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wrong_pair = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.was_lifted = torch.zeros(
            (self.num_envs, n), dtype=torch.bool, device=self.device)
        self.scramble_pos = torch.zeros(
            (self.num_envs, n, 3), device=self.device)
        return super()._after_reconfigure(options)

    # -------------------------------------------------------------- evaluate --

    def _cube_positions(self) -> torch.Tensor:
        "(num_envs, n_cubes, 3) world positions."
        return torch.stack([c.pose.p for c in self.cubes], dim=1)

    def _gather_cube(self, tensor_nc: torch.Tensor, level: int) -> torch.Tensor:
        """Pick per-env the value of the cube living at `level` of the ORIGINAL
        stack. tensor_nc is (num_envs, n_cubes, ...)."""
        idx = self.original_order[:, level]
        return tensor_nc[torch.arange(self.num_envs, device=idx.device), idx]

    def _stacked_on(self, top: torch.Tensor, bottom: torch.Tensor,
                    xy_tol: float | None = None) -> torch.Tensor:
        """Is `top` resting ON `bottom`? Centre alignment within `xy_tol`
        (stack_xy_tol by default), centre-to-centre height within
        stack_dz_tol of one cube."""
        tol = self.cfg.stack_xy_tol if xy_tol is None else float(xy_tol)
        xy = torch.linalg.norm(top[:, :2] - bottom[:, :2], dim=1)
        dz = top[:, 2] - bottom[:, 2]
        return (xy <= tol) \
            & ((dz - 2 * self.cfg.cube_half).abs() <= self.cfg.stack_dz_tol)

    def _in_cabinet(self, p: torch.Tensor) -> torch.Tensor:
        # The full cabinet box (W12/W13 measured: x [1.75, 2.75], z top 2.31),
        # not just a height floor — a pair stacked on the cabinet ROOF or on a
        # far wall shelf must not count as restacked.
        return (p[:, 2] > self.cfg.shelf_top_z - 0.05) & (p[:, 2] < 2.31) \
            & (p[:, 1] > -0.45) & (p[:, 0] > 1.75) & (p[:, 0] < 2.75)

    def _apply_scramble(self, mask: torch.Tensor):
        """Teleport the movable cubes of `mask` envs to their seeded slots and
        zero their velocities (the set_door lesson: a teleported body keeps its
        old velocity and drifts). Called from evaluate() — the repo's blessed
        slot for phase side effects (example_memory_task's measured note). A
        cube still held in the gripper is yanked out by design: the scramble
        fires when the TARGET stands placed, and holding a movable cube at
        that moment is the agent's own ordering choice, priced, documented."""
        if not bool(mask.any()):
            return
        is_movable = torch.zeros(
            (self.num_envs, self.cfg.n_cubes), dtype=torch.bool, device=mask.device)
        for k in (1, 2):
            idx = self.original_order[:, k]
            is_movable[torch.arange(self.num_envs, device=idx.device), idx] = True
        for ci, cube in enumerate(self.cubes):
            m = mask & is_movable[:, ci]
            if not bool(m.any()):
                continue
            pose = cube.pose.raw_pose.clone()
            pose[m, :3] = self.scramble_pos[m, ci]
            pose[m, 3:] = torch.tensor(
                [1.0, 0, 0, 0], device=pose.device)
            cube.pose = pose
            lv = cube.linear_velocity.clone()
            av = cube.angular_velocity.clone()
            lv[m] = 0.0
            av[m] = 0.0
            cube.set_linear_velocity(lv)
            cube.set_angular_velocity(av)
        # Canonicalize the registers the agent could have written the answer
        # into (the design blank's «внешняя шпаргалка», beyond the layout):
        # the TARGET cube snaps to the exact place point with an identity
        # quat (it just settled there — the snap is millimetres) and the door
        # returns to its reset angle. The agent's own body pose remains an
        # unerasable register — an embodied task cannot wipe the robot —
        # accepted and documented in the design blank.
        for ti, cube in enumerate(self.cubes):
            tm = mask & (self.original_order[:, 0] == ti)
            if not bool(tm.any()):
                continue
            pose = cube.pose.raw_pose.clone()
            pose[tm, 0] = self.place_target[tm, 0]
            pose[tm, 1] = self.place_target[tm, 1]
            pose[tm, 2] = self.place_target[tm, 2] + self.cfg.cube_half
            pose[tm, 3:] = torch.tensor([1.0, 0, 0, 0], device=pose.device)
            cube.pose = pose
            lv = cube.linear_velocity.clone()
            av = cube.angular_velocity.clone()
            lv[tm] = 0.0
            av[tm] = 0.0
            cube.set_linear_velocity(lv)
            cube.set_angular_velocity(av)
        for i in torch.nonzero(mask).reshape(-1).tolist():
            art, j = self._door[int(i)]
            q = art.get_qpos()
            q = q.clone() if torch.is_tensor(q) else torch.as_tensor(q).clone()
            q.reshape(-1)[j] = self.cfg.door_open_rad
            art.set_qpos(q)
        if self.gpu_sim_enabled:
            # Without the explicit apply/fetch the teleports are dead writes
            # on the GPU tier and the anti-cheat silently does not happen.
            # UNVERIFIED on GPU (this repo's standing caveat) — the ant.py
            # pattern, flagged for the first GPU smoke run.
            self.scene.px.gpu_apply_rigid_dynamic_data()
            self.scene.px.gpu_fetch_rigid_dynamic_data()

    def evaluate(self) -> dict:
        pos = self._cube_positions()                        # (N, n, 3)
        grasped = torch.stack(
            [self.agent.is_grasping(c) for c in self.cubes], dim=1)  # (N, n)
        self.was_lifted = self.was_lifted | grasped

        # --- the target cube (level 0), the cabinet_retrieval predicates -----
        tpos = self._gather_cube(pos, 0)
        tgrasp = self._gather_cube(grasped, 0)
        tlift = self._gather_cube(self.was_lifted, 0)
        xy_distance = torch.linalg.norm(tpos[:, :2] - self.place_target[:, :2], dim=1)
        height_above = (tpos[:, 2] - self.cfg.cube_half) - self.place_target[:, 2]
        on_counter = xy_distance <= self.cfg.place_radius
        # Both bounds: the lower one keeps a cube that tumbled off the edge
        # onto the FLOOR from firing the irreversible scramble.
        low_enough = (height_above <= self.cfg.max_height_above_counter) \
            & (height_above >= self.cfg.min_height_above_counter)
        settled = torch.stack([
            c.is_static(lin_thresh=self.cfg.settle_lin_speed,
                        ang_thresh=self.cfg.settle_ang_speed)
            for c in self.cubes], dim=1)                    # (N, n)
        tsettled = self._gather_cube(settled, 0)
        target_ok_now = on_counter & low_enough & tsettled & ~tgrasp & tlift

        # --- the scramble: fires ONCE, the frame the target first stands -----
        newly_placed = target_ok_now & ~self.target_placed
        self.target_placed = self.target_placed | target_ok_now
        fire = newly_placed & ~self.scrambled
        self._apply_scramble(fire)
        self.scrambled = self.scrambled | fire
        if bool(fire.any()):
            # Re-read what the teleport just changed.
            pos = self._cube_positions()

        # --- the restack (only meaningful after the scramble) ----------------
        m1 = self._gather_cube(pos, 1)                      # must end BOTTOM
        m2 = self._gather_cube(pos, 2)                      # must end TOP
        m1_set = self._gather_cube(settled, 1)
        m2_set = self._gather_cube(settled, 2)
        m1_grasp = self._gather_cube(grasped, 1)
        m2_grasp = self._gather_cube(grasped, 2)
        pair_correct = self._stacked_on(m2, m1)
        pair_wrong = self._stacked_on(m1, m2)
        in_cab = self._in_cabinet(m1) & self._in_cabinet(m2)
        home = self.plinths[0].pose.p
        on_home = self._stacked_on(m1, home, xy_tol=self.cfg.home_xy_tol)
        home_xy_dist = torch.linalg.norm(m1[:, :2] - home[:, :2], dim=1)
        # The WRONG pair only counts AT HOME: the scramble itself parks the
        # movables stacked on the WEST plinth, and on half the flips that
        # stack is literally m1-on-m2 — a latch that fired there punished
        # the env's own erasure (sweep 17: wrong_pair latched at scramble
        # time on every m1-on-top flip). The answer is only ever committed
        # on the home plinth.
        wrong_at_home = self._stacked_on(m2, home, xy_tol=self.cfg.home_xy_tol)

        # The unwinnability latch: the first movable-on-movable rest inside the
        # cabinet is final. Symmetric with correct_stack: both demand RELEASE
        # (a wrong-order cube merely held over the other must not latch — the
        # commitment is the letting go).
        self.wrong_pair = self.wrong_pair | (
            self.scrambled & pair_wrong & wrong_at_home & in_cab
            & m1_set & m2_set & ~m1_grasp & ~m2_grasp
        )

        correct_stack = pair_correct & on_home & in_cab & m1_set & m2_set \
            & ~m1_grasp & ~m2_grasp
        raw = self.scrambled & correct_stack & target_ok_now & ~self.wrong_pair
        self.succeeded = self.succeeded | raw

        return {
            "success": self.succeeded,
            "target_ok_now": target_ok_now,
            "target_placed": self.target_placed,
            "scrambled": self.scrambled,
            # Gated by in_cab, the same way the latches consume them: an
            # agent test-stacking the pair OUTSIDE the cabinet (where the
            # wrong_pair latch cannot fire) must read nothing here — info is
            # harness-only by repo convention, but a free answer-probe would
            # still be a hole worth not having.
            "pair_correct": pair_correct & in_cab & on_home,
            "pair_wrong": pair_wrong & in_cab & wrong_at_home,
            "wrong_pair": self.wrong_pair,
            "in_cabinet_pair": in_cab,
            "on_home_plinth": on_home,
            "home_xy_dist": home_xy_dist,
            "xy_distance": xy_distance,
            "height_above": height_above,
            "grasped_any": grasped.any(dim=1),
        }

    # ------------------------------------------------------------------- obs --

    def _get_obs_extra(self, info: dict) -> dict:
        # The fence the memory rests on: original_order, scramble_pos and every
        # latch are NEVER emitted. Cube poses are honest observations — before
        # the scramble they ARE the cue, after it they carry no order.
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            base_pose=self.agent.robot.pose.raw_pose,
            grasped_any=info["grasped_any"],
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                # Flattened (N, n*7): the obs flattener refuses 3-D leaves.
                cube_poses=torch.stack(
                    [c.pose.raw_pose for c in self.cubes], dim=1
                ).reshape(self.num_envs, -1),
                place_target=self.place_target.clone(),
            )
        return obs

    # ----------------------------------------------------------------- state --

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        for key in TASK_STATE_KEYS:
            state[key] = getattr(self, key).clone()
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        for key in TASK_STATE_KEYS:
            setattr(self, key, restore_task_tensor(
                getattr(self, key), state[key], self.device
            ))
        super().set_state_dict(state, env_idx)
