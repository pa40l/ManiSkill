"""MikasaDepthRecall-v0: a depth row of tall props — take the FARTHEST one,
put the ones you removed back in THEIR OWN slots.

The design blank is docs/task-designs/H-depthrecall-v0.md; the story is the
user's (2026-09-01): things stand one BEHIND another, the front one blocks
the view, and you need the deepest. The sibling of MikasaStackRecall-v0
(blank G): the same order memory (one bit at n=3), the same latch
discipline — a different geometry, chosen leg by leg
from K110/W18's measurements:

- the row is TALL PROPS (~4.5 x 4.5 x 12 cm) on the BARE shelf: the side
  grasp lands at the prop's mid-height ~1.48 (the measured band) with the
  support being the prop's OWN body — no plinth edge ever enters a corridor,
  and an emptied slot is bare shelf, the one corridor floor measured
  reliable (7.8 cm under the TCP);
- physics itself enforces front-first disassembly (W18 B1: the deep grasp
  behind an occupied front slot is refused);
- the OCCLUSION is real on the robot's own cameras: a tall front prop hides
  equal-height props behind it (the raised-stadium and plinth-row
  alternatives are measured dead — K110);
- there is NO teleport erasure (the user's 2026-09-01 call: «убери
  телепорт, пусть обратно сам ставит») — the robot stages the removed
  props itself and returns them itself. The tradeoff is accepted with
  open eyes: the agent's own staging layout can encode the answer (a
  LIFO/queue convention solves the restore without internal memory), so
  the blind floor is an oracle-relative control, not a strategy-proof
  bound. What the task still enforces: the row must actually be taken
  apart and put back — success requires every movable to have LEFT its
  slot (the `left_slot` latch; standing in the OTHER row slot also
  counts as having left) and the target to have been LIFTED onto the
  place disc, so neither an extraction around an untouched row nor a
  teleport-style state write can be credited. An UNGRASPED shove of the
  movables (plowed aside, nudged back) IS credited on purpose: the
  restore-by-identity is the memory demand, the grasp is not part of it
  (review-settled, 2026-09-01).

The unwinnability latch fires when BOTH movable props stand released and
settled in the two row slots in the WRONG assignment — the committed wrong
answer; single placements en route latch nothing.
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

#: Keys `get_state_dict` adds beyond the simulator state — every one is task
#: MEMORY (blank point 8).
TASK_STATE_KEYS = (
    "original_slots", "target_placed", "left_slot", "wrong_assign",
    "succeeded", "was_lifted", "place_target",
)

INSTRUCTIONS = (
    "Take out the props in front, fetch the one at the back onto the "
    "counter, and put the others back in their own spots.",
    "The farthest prop is the one you need; return the ones you moved to "
    "where they stood.",
    "Clear the row to reach the deepest prop, place it on the counter, and "
    "restore the row as it was.",
)

#: Fixed identity palette (a prop cannot be recolored per episode); the
#: episode ticket permutes which SLOT each prop occupies.
PROP_COLORS = (
    (1.0, 0.1, 0.1, 1.0),   # red
    (0.1, 0.8, 0.1, 1.0),   # green
    (0.15, 0.3, 1.0, 1.0),  # blue
)


@dataclass
class DepthRecallConfig:
    """Every threshold named; magic numbers in evaluate() are how failures go mute."""

    horizon: int = 4000
    """K22 against the TELEPORT-FREE oracle (2026-09-01, K112): calibration
    seeds 11-20 gave successes of 1994-2626 steps; 2626 x 1.5 = 3939,
    rounded up to 100 -> 4000 (the eval set's own max, 2601, sits inside).
    The teleport-era 3500 was calibrated on the shorter scramble route and
    retired with it."""

    # -- the cabinet (measured, K105-K110 territory) ---------------------------
    cabinet_name: str = "cab_main_main_group"
    counter_name: str = "counter_main_main_group"
    door_hinge: str = "rightdoorhinge"
    door_open_rad: float = 1.75
    "The cabinet family's measured angle; the door starts (and stays) open."

    # -- the props and the row -------------------------------------------------
    n_props: int = 3
    "v0 is fixed at 3 (one target + two movable = a one-bit answer)."
    prop_half_xy: float = 0.0225
    prop_half_h: float = 0.06
    """4.5 x 4.5 x 12 cm: the side grasp lands at the prop's mid (~1.48, the
    measured band) with the prop's own body as the support below the pads —
    the K110 design decision that removed every plinth from the row."""
    shelf_top_z: float = 1.4200
    spawn_clearance: float = 0.004
    row_slots_y: tuple = (-0.20, -0.145, -0.09)
    """Front -> deep, all three MEASURED graspable at prop height (W18 A:
    -0.20/-0.145/-0.09 held; -0.06 flaked and is not used). The DEEP slot
    holds the target."""
    row_margin_x: float = 0.13
    row_jitter_x: float = 0.04
    "Row x: jitter around the exposed half's centre (2.50) -> x in [2.46, 2.54]."
    slot_xy_tol: float = 0.035
    "How far (xy) a prop may stand off its slot and still count as IN it."
    upright_tol: float = 0.015
    """A standing prop's centre z sits at shelf + clearance-ish + prop_half_h;
    within this of that height = upright (a tipped 12 cm prop's centre drops
    by ~3.5 cm — far outside)."""

    # -- the place (target prop) and the removal fence -------------------------
    place_across: float = -0.525
    place_radius: float = 0.15
    max_height_above_counter: float = 0.06
    min_height_above_counter: float = -0.02
    settle_lin_speed: float = 0.05
    settle_ang_speed: float = 0.2
    leave_tol: float = 0.06
    """A movable counts as having LEFT its slot once its xy runs this far
    from the slot point (comfortably beyond slot_xy_tol 0.035, so a nudged
    prop does not count as removed). Success requires BOTH movables to
    have left and returned — without the teleport erasure this is what
    keeps «достань дальний, не тронув ряд» from voiding the memory test."""

    # -- the robot start -------------------------------------------------------
    start_y: float = -1.90
    """The start x TRACKS the row's x: the frame reader measured the fixed
    2.55 start 6 cm off the row axis feeding the rear props' color slivers
    past the front prop — aligned, the stereo baseline is the only off-axis
    left (a 1-2 px residual per eye, recorded in the journal)."""
    start_yaw_deg: float = 90.0
    start_jitter_x: float = 0.02
    start_jitter_y: float = 0.08
    start_jitter_yaw: float = 0.10

    def validate(self) -> None:
        assert self.horizon > 0
        assert self.n_props == 3, (
            "v0 is designed and floored for exactly 3 (one target, two "
            "movable); more props is the family's difficulty knob."
        )
        assert 0.9 <= self.door_open_rad <= 2.9
        assert 0 < self.prop_half_xy <= 0.028, (
            "the gripper pads open ~6.4 cm; wider props cannot be side-grasped"
        )
        assert 0.04 <= self.prop_half_h <= 0.10, (
            "the prop must be tall enough to occlude and to put its mid-grasp "
            "in the measured band, short enough to stay under the cabinet top"
        )
        assert len(self.row_slots_y) == 3
        ys = list(self.row_slots_y)
        assert ys == sorted(ys), "row slots must run front (south) -> deep"
        for i in range(len(ys) - 1):
            assert ys[i + 1] - ys[i] > 2 * self.prop_half_xy + 0.005, (
                "adjacent row slots closer than a prop footprint"
            )
        assert -0.28 <= ys[0] and ys[-1] <= -0.035, (
            "row slots leave the measured graspable depth band (W18)"
        )
        top_of_prop = self.shelf_top_z + 0.01 + 2 * self.prop_half_h
        assert top_of_prop < 2.31, "the props poke through the cabinet top"
        assert 0 < self.slot_xy_tol < 2 * self.prop_half_xy
        assert 0 < self.upright_tol < self.prop_half_h / 2
        assert self.place_radius > 0 and self.max_height_above_counter > 0
        assert self.min_height_above_counter < 0
        assert -0.65 < self.place_across < -0.448
        assert self.slot_xy_tol < self.leave_tol <= 0.10, (
            "leave_tol must clear the slot tolerance yet stay local"
        )


@register_env(
    "MikasaDepthRecall-v0",
    max_episode_steps=DepthRecallConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class DepthRecallTask(BaseEnv):
    """Clear the row, fetch the deepest prop, restore the others to their slots."""

    SUPPORTED_ROBOTS = ["mikasa_ds_fetch", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    cfg = DepthRecallConfig()

    def __init__(self, *args, robot_uids="mikasa_ds_fetch", scene_idx: int | None = 0, **kwargs):
        self.cfg.validate()
        self.scene_idx = scene_idx
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- sensors --

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
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

        matches = [
            (name, art) for name, art in self.scene.articulations.items()
            if self.cfg.cabinet_name in name
        ]
        assert len(matches) == self.num_envs, (
            f"expected one {self.cfg.cabinet_name!r} articulation per env, "
            f"found {[n for n, _ in matches]!r}"
        )
        self._door = []
        # numeric sort: names carry an unpadded `scene-{i}` index, and the
        # lexicographic order breaks past 10 envs (review finding)
        import re as _re

        def _env_ix(name):
            m = _re.search(r"scene-(\d+)", name)
            return int(m.group(1)) if m else 0

        for _name, art in sorted(matches, key=lambda kv: _env_ix(kv[0])):
            names = [j.name for j in art.get_active_joints()]
            assert self.cfg.door_hinge in names, (_name, names)
            self._door.append((art, names.index(self.cfg.door_hinge)))

        # The props: tall dynamic boxes, identity = build-time color.
        self.props = [
            actors.build_box(
                self.scene,
                half_sizes=[self.cfg.prop_half_xy, self.cfg.prop_half_xy,
                            self.cfg.prop_half_h],
                color=list(PROP_COLORS[i]),
                name=f"prop_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[2.5, -0.19, 1.60 + 0.15 * i]),
            )
            for i in range(self.cfg.n_props)
        ]

        half_lo, half_hi = self._exposed_half()
        centre_x = (half_lo + half_hi) / 2.0
        assert 2.35 <= centre_x - self.cfg.row_jitter_x \
            and centre_x + self.cfg.row_jitter_x <= 2.65, (
                "the row's x band leaves W13's measured graspable band"
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
        return (float(pos[0]) + self.cfg.row_margin_x,
                float(pos[0] + size[0] / 2.0) - self.cfg.row_margin_x)

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

            perms, jitters = [], []
            for i in env_idx.tolist():
                rng = self._batched_episode_rng[i]
                perms.append(rng.permutation(self.cfg.n_props))
                jitters.append(rng.uniform(-1.0, 1.0, size=4))
            perm_t = torch.as_tensor(np.stack(perms), dtype=torch.long)
            jit_t = torch.as_tensor(np.stack(jitters), dtype=torch.float32)
            # original_slots[:, k] = prop index at row slot k (0=front,
            # 1=middle, 2=deep — the deep one is the TARGET).
            self.original_slots[env_idx] = perm_t

            sx = self._spawn_centre_x[env_idx] + jit_t[:, 0] * self.cfg.row_jitter_x
            prop_z = self.cfg.shelf_top_z + self.cfg.spawn_clearance \
                + self.cfg.prop_half_h

            for pi, prop in enumerate(self.props):
                slot = (perm_t == pi).float().argmax(dim=1).long()
                p = torch.zeros((b, 3))
                p[:, 0] = sx
                p[:, 1] = torch.as_tensor(
                    [self.cfg.row_slots_y[int(s)] for s in slot],
                    dtype=torch.float32)
                p[:, 2] = prop_z
                prop.set_pose(Pose.create_from_pq(p=p))

            # The target's place point: the counter below the row column.
            tgt = torch.zeros((b, 3))
            tgt[:, 0] = sx
            tgt[:, 1] = self.cfg.place_across
            tgt[:, 2] = self._counter_top_z[env_idx]
            self.place_target[env_idx] = tgt

            base = torch.zeros((b, 3))
            base[:, 0] = sx + jit_t[:, 3] * self.cfg.start_jitter_x
            base[:, 1] = self.cfg.start_y + jit_t[:, 1] * self.cfg.start_jitter_y
            base[:, 2] = math.radians(self.cfg.start_yaw_deg) \
                + jit_t[:, 2] * self.cfg.start_jitter_yaw
            qpos = self.agent.robot.get_qpos()
            qpos[env_idx, 0] = base[:, 0]
            qpos[env_idx, 1] = base[:, 1]
            qpos[env_idx, 2] = base[:, 2]
            self.agent.robot.set_qpos(qpos[env_idx])

            self.target_placed[env_idx] = False
            self.left_slot[env_idx] = False
            self.wrong_assign[env_idx] = False
            self.succeeded[env_idx] = False
            self.was_lifted[env_idx] = False

    def _restore_robot(self, env_idx: torch.Tensor):
        keyframe = self.agent.keyframes["rest"]
        qpos = torch.as_tensor(np.asarray(keyframe.qpos, dtype=np.float32))
        self.agent.robot.set_qpos(qpos.unsqueeze(0).repeat(len(env_idx), 1))
        self.agent.robot.set_root_pose(sapien.Pose())

    def _after_reconfigure(self, options: dict):
        n = self.cfg.n_props
        self.original_slots = torch.zeros(
            (self.num_envs, n), dtype=torch.long, device=self.device)
        self.target_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wrong_assign = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.was_lifted = torch.zeros(
            (self.num_envs, n), dtype=torch.bool, device=self.device)
        self.left_slot = torch.zeros(
            (self.num_envs, n), dtype=torch.bool, device=self.device)
        return super()._after_reconfigure(options)

    # -------------------------------------------------------------- evaluate --

    def _prop_positions(self) -> torch.Tensor:
        return torch.stack([p.pose.p for p in self.props], dim=1)

    def _gather_prop(self, tensor_nc: torch.Tensor, slot: int) -> torch.Tensor:
        idx = self.original_slots[:, slot]
        return tensor_nc[torch.arange(self.num_envs, device=idx.device), idx]

    def _standing_at(self, p: torch.Tensor, x: torch.Tensor,
                     y: torch.Tensor) -> torch.Tensor:
        """Is a prop standing upright AT (x, y) on the shelf? xy within
        slot_xy_tol AND y within HALF the slot pitch (the radial disc alone
        overlaps the neighbor at 0.035 > 5.5/2 cm — the review caught a prop
        at y~-0.175 counting as standing at BOTH slots); centre z within
        upright_tol of the standing height (a tipped 12 cm prop's centre
        drops ~3.5 cm — far outside)."""
        stand_z = self.cfg.shelf_top_z + self.cfg.spawn_clearance \
            + self.cfg.prop_half_h
        half_pitch = (self.cfg.row_slots_y[1] - self.cfg.row_slots_y[0]) / 2.0
        xy = torch.sqrt((p[:, 0] - x) ** 2 + (p[:, 1] - y) ** 2)
        return (xy <= self.cfg.slot_xy_tol) \
            & ((p[:, 1] - y).abs() < half_pitch) \
            & ((p[:, 2] - stand_z).abs() <= self.cfg.upright_tol)

    def evaluate(self) -> dict:
        pos = self._prop_positions()
        grasped = torch.stack(
            [self.agent.is_grasping(p) for p in self.props], dim=1)
        self.was_lifted = self.was_lifted | grasped

        # --- the target prop (the DEEP slot), K109 predicates ----------------
        tpos = self._gather_prop(pos, 2)
        tgrasp = self._gather_prop(grasped, 2)
        tlift = self._gather_prop(self.was_lifted, 2)
        xy_distance = torch.linalg.norm(tpos[:, :2] - self.place_target[:, :2], dim=1)
        height_above = (tpos[:, 2] - self.cfg.prop_half_h) - self.place_target[:, 2]
        on_counter = xy_distance <= self.cfg.place_radius
        low_enough = (height_above <= self.cfg.max_height_above_counter) \
            & (height_above >= self.cfg.min_height_above_counter)
        settled = torch.stack([
            p.is_static(lin_thresh=self.cfg.settle_lin_speed,
                        ang_thresh=self.cfg.settle_ang_speed)
            for p in self.props], dim=1)
        tsettled = self._gather_prop(settled, 2)
        target_ok_now = on_counter & low_enough & tsettled & ~tgrasp & tlift

        self.target_placed = self.target_placed | target_ok_now

        # --- the left-slot latches: a prop counts as REMOVED once its xy
        # runs leave_tol from its original slot point. With the teleport
        # erasure gone, success demands both movables to have left and
        # returned — the fence against extracting the target around an
        # untouched row (which would void the memory test).
        sx = self.place_target[:, 0]
        arange = torch.arange(self.num_envs, device=pos.device)
        for k in range(self.cfg.n_props):
            idx = self.original_slots[:, k]
            pk = pos[arange, idx]
            d_slot = torch.sqrt((pk[:, 0] - sx) ** 2
                                + (pk[:, 1] - self.cfg.row_slots_y[k]) ** 2)
            self.left_slot[arange, idx] = self.left_slot[arange, idx] \
                | (d_slot > self.cfg.leave_tol)

        # --- the restore: identity per row slot ------------------------------
        front_y = torch.full_like(sx, self.cfg.row_slots_y[0])
        mid_y = torch.full_like(sx, self.cfg.row_slots_y[1])
        f_prop = self._gather_prop(pos, 0)
        m_prop = self._gather_prop(pos, 1)
        f_set = self._gather_prop(settled, 0)
        m_set = self._gather_prop(settled, 1)
        f_grasp = self._gather_prop(grasped, 0)
        m_grasp = self._gather_prop(grasped, 1)

        front_right = self._standing_at(f_prop, sx, front_y)
        mid_right = self._standing_at(m_prop, sx, mid_y)
        # The committed WRONG answer: the front prop standing at the MIDDLE
        # slot AND the middle prop at the FRONT slot, both released+settled.
        front_swapped = self._standing_at(f_prop, sx, mid_y)
        mid_swapped = self._standing_at(m_prop, sx, front_y)
        # standing in the NEIGHBOR row slot is proof of having left one's
        # own — leave_tol (0.06) alone has a dead zone at the 5.5 cm slot
        # pitch (the review's committed-swap escape); and the swap itself
        # needs no left gate: at t=0 both props stand RIGHT, so a swapped
        # pair can only ever be the committed wrong answer
        arange2 = torch.arange(self.num_envs, device=pos.device)
        self.left_slot[arange2, self.original_slots[:, 0]] |= front_swapped
        self.left_slot[arange2, self.original_slots[:, 1]] |= mid_swapped
        f_left = self._gather_prop(self.left_slot, 0)
        m_left = self._gather_prop(self.left_slot, 1)
        self.wrong_assign = self.wrong_assign | (
            front_swapped & mid_swapped
            & f_set & m_set & ~f_grasp & ~m_grasp
        )

        restore_ok = front_right & mid_right & f_set & m_set \
            & ~f_grasp & ~m_grasp
        # success gates: the INSTANTANEOUS target_ok_now (the target must
        # still stand placed at the end — target_placed is a trace latch,
        # not the gate), and the movables' left_slot (they must have been
        # OUT). Only the target's was_lifted row is consumed: a policy that
        # plows the movables aside ungrasped and nudges them back has still
        # restored them BY IDENTITY — that is the memory demand, and the
        # grasp is not part of it (review-settled, 2026-09-01).
        raw = f_left & m_left & restore_ok & target_ok_now \
            & ~self.wrong_assign
        self.succeeded = self.succeeded | raw

        return {
            "success": self.succeeded,
            "target_ok_now": target_ok_now,
            "target_placed": self.target_placed,
            "movables_left": f_left & m_left,
            "front_right": front_right,
            "mid_right": mid_right,
            "wrong_assign": self.wrong_assign,
            "xy_distance": xy_distance,
            "height_above": height_above,
            "grasped_any": grasped.any(dim=1),
        }

    # ------------------------------------------------------------------- obs --

    def _get_obs_extra(self, info: dict) -> dict:
        # The memory fence: original_slots and the latches are NEVER
        # emitted; prop poses are honest observations.
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            base_pose=self.agent.robot.pose.raw_pose,
            grasped_any=info["grasped_any"],
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                prop_poses=torch.stack(
                    [p.pose.raw_pose for p in self.props], dim=1
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
