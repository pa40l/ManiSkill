"""B02 Station Checklist — remember which stations were already done, finish the rest.

Implements `~/Downloads/b02_station_checklist_spec.md` on the kitchen
`MyRoboCasa_TakeItBack-v1` uses. The construct is `action_history`: the agent has to
hold a set in mind across a blank window and update it with its own actions.

An episode has three parts:

  encoding   t < cue_steps      robot frozen in a viewing pose, head scans, beacons
                                lit on every station that is already completed
  the event  t == cue_steps     the beacons on `ell` of those stations go dark
  action     t >= cue_steps     the policy drives; exactly `k - m` service gestures
                                are allowed, and they must land on the stations that
                                were never completed

After the event those stations are visually indistinguishable from the ones that
were quietly finished before the episode began. Nothing on the scene records what
happened — no texture change, no displaced object, no sound. The answer exists only
in the simulator and in whatever the policy remembered.

**Why the memoryless ceiling is one half, and where it comes from.** With k=6, m=3,
ell=3 the first commitment is a choice among six stations of which three are
targets, so a policy that cannot remember scores 3/6. That is the number the spec
calls `no_memory_floor`, and it is the floor for the *first decision*. The floor for
*episode success* is different and much lower: naming the right three out of six in
any order is 1/C(6,3) = 1/20 = 5%. Both belong in any report; quoting the 0.5 next
to an episode success rate makes a result look three times better than it is.

**Deliberate departure from the spec.** §4 wants the stations to be identical meshes
distinguishable only by pose. Here they are the kitchen's own fixtures — a sink, a
stove, a microwave — because a one-wall kitchen cannot hold six free-standing
stations 0.6 m apart. The compensation is that the *beacons* are identical, and the
beacon is what carries the information: once they are dark the scene is the same
under every one of the C(6,3) draws, which is the ambiguity property §8.2 asks for.
Recorded here rather than left for a reviewer to find.

Conventions: docs/writing-tasks.md. Reference implementation: season_dish.py.
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
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Pose

from utils.mikasa.scenes.emissive import BeaconGrid
from utils.mikasa.scenes.robocasa_utils import (
    parking_pose,
    STATION_FIXTURE_CLASSES,
    base_fixture_for,
    dedup_docks,
    dock_pose_for,
    fixture_frame,
    restore_task_tensor,
    select_spread_stations,
    station_candidates,
)

# Instruction pool. No numbers anywhere: "finish the remaining three" would hand the
# answer's cardinality to a language-conditioned policy (spec trap 5).
INSTRUCTIONS = (
    "Activate every station that has not been activated yet. No repeats.",
    "Service the stations that are still outstanding, a gesture each.",
    "Finish the stations nobody has done. Do not repeat a station.",
)


@dataclass
class StationChecklistConfig:
    """Every threshold in one place, so a failed run can say which one it missed."""

    horizon: int = 1400
    """Episode length, from the measured demo (K22, T4 2026-08-18; was 1500, a guess).

    Calibration seeds 11 (sighted 733 steps, blind 933) and 12 (sighted 953) in
    the container on kitchen 102 with the shipped oracle: manipulation after the
    cue = max(733, 953, 933) − 120 = 833; × 1.5 = 1249.5 → up to a multiple of
    50 = 1250; horizon = cue 120 + 1250 = 1370 → up to a multiple of 100 =
    **1400**. Eval seeds 0–9 were not used for the derivation. The registered
    `max_episode_steps` reads this field.
    """

    # --- phases -------------------------------------------------------------
    cue_steps: int = 120
    "Encoding length. Constant — never derived from `ell` or from the draw (trap 4)."
    gesture_steps: int = 30
    settle_steps: int = 10

    # --- the puzzle ---------------------------------------------------------
    k_stations: int = 6
    m_completed: int = 3
    ell_blinded: int = 3
    "How many of the completed beacons go dark. The load knob. 0 < ell <= m."

    # --- station selection --------------------------------------------------
    station_classes: tuple = STATION_FIXTURE_CLASSES
    station_min_top_z: float = 0.30
    station_max_top_z: float = 1.60
    dock_dedup_tol: float = 0.05
    "Two candidate docks closer than this in xy are one dock: one station survives (K28)."
    dock_min_separation: float = 0.45
    """Closest two chosen docks may be, in xy (radius mode only; see `dock_separation_ok`).

    0.45, not 0.5: on kitchen 102 the closest distinct docks (dishwasher vs the
    `stack_3` cabinet) are exactly 0.50 m apart by the layout YAML, and the
    float32 arithmetic can land a hair under it (D5; T4 probe, journal
    2026-08-18). The old 0.90 was written for a kitchen that does not exist —
    the first build refused with `two dock poses are 0.50 m apart`.
    """

    # --- the service zone ---------------------------------------------------
    zone_mode: str = "radius"
    """`radius` or `nearest`.

    `radius`: in the zone of station s when within `dock_radius` of its dock pose.
    `nearest`: the zone of s is its Voronoi cell among the dock poses, capped at
    `zone_max_range`. Disjoint by construction, which matters because
    `compute_robot_base_placement_pose` gives every station on one counter the same
    heading and separates them only along it — the radii can overlap. Which mode is
    right cannot be settled without the kitchen geometry, so both ship from day one.
    D5 (2026-08-18): `radius` is the default because the burner oracle's measured
    parking error is 0.008 m at most (K26); `nearest` is the fallback if it ever
    exceeds 0.10 m.
    """
    dock_radius: float = 0.20
    "Was 0.35; tightened so two zones 0.45 m apart cannot overlap (0.45 > 2 × 0.20)."
    zone_max_range: float = 1.20
    dock_yaw_tol_deg: float = 35.0
    base_static_speed: float = 0.08

    # --- the gesture --------------------------------------------------------
    gesture_xy_radius: float = 0.45
    gesture_z_low: float = -0.35
    gesture_z_high: float = 0.55
    gesture_amplitude_rad: float = 0.25

    # --- beacons ------------------------------------------------------------
    beacon_radius: float = 0.045
    beacon_height: float = 0.12
    "Beacon centre this far above the higher of the fixture's top and its base counter's top."
    beacon_inset: float = 0.10
    "The SERVICE point sits this far inside the fixture's near edge (`act_pt`; unchanged by K32)."
    beacon_outset: float = 0.02
    """The BEACON sits this far outside the fixture's near face (K32, 2026-08-18).

    It used to share `act_pt`'s xy (0.10 m inside the face) at `top + 0.12`,
    which buries it in the counter body for every fixture under a countertop:
    the frame probe on 102 (seed 3, `state_diff --at 119,121`, reader) saw the
    stove/cabinet beacons over the counter and none at all over the drawer stack
    (top 0.47) or the low cabinet (top 0.68). Moving the beacon does not move
    `act_pt` — see `station_points`.
    """

    # --- viewing pose -------------------------------------------------------
    viewpoint_standoff: float = 1.4
    viewpoint_jitter_xy: float = 0.08
    viewpoint_jitter_yaw: float = 0.10

    # --- head script --------------------------------------------------------
    head_scan_amplitude: float = 1.2
    head_scan_sweeps: float = 2.0
    head_tilt_scan: float = 0.25
    head_tilt_drive: float = 0.05

    instructions: tuple = INSTRUCTIONS

    def validate(self) -> None:
        assert 0 < self.ell_blinded <= self.m_completed < self.k_stations, self
        assert self.zone_mode in ("radius", "nearest"), self.zone_mode
        if self.zone_mode == "radius":
            assert self.dock_min_separation > 2 * self.dock_radius, (
                "service zones can overlap; raise dock_min_separation or use zone_mode='nearest'"
            )
        budget = self.k_stations - self.m_completed
        floor = self.cue_steps + budget * (self.gesture_steps + self.settle_steps)
        assert floor < self.horizon, (
            f"the gestures alone need {floor} steps of a {self.horizon}-step episode"
        )


def dock_separation_ok(sep_min: float, cfg: StationChecklistConfig) -> bool:
    """Is the closest pair of chosen docks far enough apart for this zone mode?

    In `radius` mode two docks closer than `dock_min_separation` would give
    overlapping service zones and an ambiguous commitment; in `nearest` mode the
    zones are Voronoi cells and disjoint whatever the spacing, so no distance is
    too small. A pure function so `_load_scene`'s refusal and the offline tests
    share one rule.

    Example:
        >>> cfg = StationChecklistConfig()
        >>> dock_separation_ok(0.50, cfg), dock_separation_ok(0.30, cfg)
        (True, False)
        >>> dock_separation_ok(0.30, StationChecklistConfig(zone_mode="nearest"))
        True
    """
    if cfg.zone_mode == "nearest":
        return True
    return float(sep_min) >= cfg.dock_min_separation


def station_points(
    top: np.ndarray,
    across: np.ndarray,
    size: np.ndarray,
    cfg: StationChecklistConfig,
    base_top_z: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`(act_pt, beacon_pt)` for one station, from its fixture frame. Two points, on purpose.

    `act_pt` — the service point the gesture predicate measures `d_tcp`/`dz`
    against — is the fixture's top centre pulled `beacon_inset` inside its near
    edge (a Fetch 0.8 m back cannot reach the middle of a microwave). It is
    exactly the pre-K32 formula and must stay so: moving it outward would make
    `d_tcp ≤ gesture_xy_radius` easier and weaken the predicate (Global
    Constraint).

    `beacon_pt` — what the policy sees — is a *separate* point (K32): on the
    fixture's near face, `beacon_outset` outside it, at `beacon_height` above the
    higher of the fixture's own top and its base counter's top (`base_top_z`,
    from `robocasa_utils.base_fixture_for`), so a drawer's beacon stands over the
    countertop edge instead of inside the cabinet.

    Args:
        top, across, size: from `fixture_frame(fixture)`; `across` is the unit
            vector from the near edge toward the fixture's centre.
        cfg: the task config (`beacon_inset`, `beacon_outset`, `beacon_height`).
        base_top_z: top z of the base fixture (counter); None = the fixture's own.

    Returns:
        `(act_pt, beacon_pt)`, two float32 arrays of shape (3,).

    Example:
        >>> cfg = StationChecklistConfig()
        >>> top = np.array([0.5, -0.3, 0.47]); across = np.array([0.0, 1.0, 0.0]); size = np.array([0.5, 0.6, 0.21])
        >>> act, beac = station_points(top, across, size, cfg, base_top_z=0.92)
        >>> np.round(act.astype(float), 3).tolist(), np.round(beac.astype(float), 3).tolist()
        ([0.5, -0.5, 0.47], [0.5, -0.62, 1.04])
    """
    top = np.asarray(top, dtype=np.float64)
    across = np.asarray(across, dtype=np.float64)
    half = float(size[1]) / 2.0
    act = top - across * (half - cfg.beacon_inset)
    beac_xy = top - across * (half + cfg.beacon_outset)
    ref_z = float(top[2]) if base_top_z is None else max(float(top[2]), float(base_top_z))
    beac = np.array([beac_xy[0], beac_xy[1], ref_z + cfg.beacon_height])
    return act.astype(np.float32), beac.astype(np.float32)


def head_script(step: int, cfg: StationChecklistConfig) -> tuple[float, float]:
    """`(head_pan, head_tilt)` for a step. A pure function, and that is the point.

    Spec §5 requires the head trajectory to depend only on phase, time and layout —
    never on which stations are completed, on `ell`, or on the permutation of the
    answer. Written as a module-level function of `(step, cfg)`, that requirement is
    a property of the signature: there is nothing else in scope to leak. A method on
    the env would have `self`, and proving the absence of a read is much harder than
    not having the option.

    Layout is dropped from the dependency list on purpose. The kitchen is pinned, so
    it contributes a constant, and taking it would put a scene-dependent term into a
    function whose whole value is that it has none.
    """
    if step < cfg.cue_steps:
        phase = 2.0 * math.pi * cfg.head_scan_sweeps * step / max(cfg.cue_steps, 1)
        return cfg.head_scan_amplitude * math.sin(phase), cfg.head_tilt_scan
    return 0.0, cfg.head_tilt_drive


def gesture_profile(cfg: StationChecklistConfig, n_arm: int) -> np.ndarray:
    """`(gesture_steps, n_arm)` joint offsets added to the pose held at commit time.

    A damped sweep of the last two arm joints. The shape does not matter; what
    matters is that it is the same every time, so the gesture costs exactly
    `gesture_steps` steps regardless of the policy or of `ell` — the motor
    invariance §3 and §8.1 depend on.
    """
    t = np.arange(cfg.gesture_steps, dtype=np.float32)
    envelope = np.sin(np.pi * t / max(cfg.gesture_steps - 1, 1))
    wave = np.sin(2.0 * np.pi * 2.0 * t / max(cfg.gesture_steps, 1))
    out = np.zeros((cfg.gesture_steps, n_arm), dtype=np.float32)
    if n_arm >= 2:
        out[:, -1] = cfg.gesture_amplitude_rad * envelope * wave
        out[:, -2] = 0.5 * cfg.gesture_amplitude_rad * envelope * wave
    return out


@register_env(
    "MikasaStationChecklist-v0",
    max_episode_steps=StationChecklistConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class StationChecklistTask(BaseEnv):
    """Service the stations that were never completed, remembering which those are."""

    SUPPORTED_ROBOTS = ["mikasa_ds_fetch", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    cfg = StationChecklistConfig()

    def __init__(
        self,
        *args,
        robot_uids="mikasa_ds_fetch",
        scene_idx: int | None = 0,
        ell: int | None = None,
        **kwargs,
    ):
        # scene_idx defaults to 0, the kitchen MyRoboCasa_TakeItBack-v1 builds:
        # RandomState(102).randint(0, 120) == 0, i.e. layout ONE_WALL_SMALL, style
        # INDUSTRIAL. Set before super().__init__, which runs a full reconfigure.
        self.scene_idx = scene_idx
        if ell is not None:
            self.cfg = StationChecklistConfig(**{**self.cfg.__dict__, "ell_blinded": ell})
        self.cfg.validate()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- sensors --

    @property
    def _default_sensor_configs(self):
        # None of the task's own. ds_fetch already carries the stereo pair mounted on
        # head_camera_link and the wrist camera on gripper_link, which is what the
        # spec's observation list asks for. A room camera would be a privileged
        # viewpoint, so it lives only in the human render below.
        return []

    @property
    def _default_human_render_camera_configs(self):
        """The clip's viewpoint — the privileged room view this task denies the
        policy (`_default_sensor_configs` above returns []). Decoration only.

        Re-aimed 2026-08-19. The old eye (2.6 m back, 2.2 m up, looking at the
        centre of the dock line, which sits on the floor) spent the lower half of
        every frame on empty floor and pushed the counter — where all six beacons
        are — into a strip along the top edge. Standing 3.0 m back and 2.8 m up and
        looking 0.75 m above the dock line puts the beacon row across the middle of
        the frame, which is what a viewer has to read: which stations are still
        lit and which the robot has already served. Higher than this and the camera
        clears the wall, and the top corner of the frame turns into the black void
        outside the kitchen — measured on the recorded episode.

        512 is the *recording* default; `RecordEpisode` keeps a whole episode of
        frames in host RAM and this task has the longest episodes in the repo
        (931 steps = 0.73 GB at 512², 11.7 GB at 2048²). Clips get their size from
        `utils.mikasa.replay --render-size` instead.
        """
        eye = self._room_centre + np.array([0.0, -3.0, 2.8], dtype=np.float32)
        target = self._room_centre + np.array([0.0, 0.0, 0.75], dtype=np.float32)
        pose = sapien_utils.look_at(eye=eye, target=target)
        return CameraConfig("render_camera", pose, 512, 512, 1.05, 0.01, 100)

    # ------------------------------------------------------------------- load --

    def _load_agent(self, options: dict):
        super()._load_agent(options, parking_pose(self))

    def _load_scene(self, options: dict):
        """Kitchen, stations, beacons. No `set_pose` anywhere — the lint walks this.

        `scene._setup()` runs straight after and re-applies `initial_pose` to every
        non-static actor, so a pose set here is discarded. The beacons are static and
        never move at all.
        """
        super()._load_scene(options)

        # Same kitchen as TakeItBack, reached honestly. That task calls
        # _set_episode_rng(102) inside _load_scene, which overwrites the whole
        # episode seed (sapien_env.py:1010-1012) and is replayed on :918 — so a
        # user-supplied seed is thrown away and nothing else in the episode can
        # randomise. Pinning the build index does only what it says.
        self.scene_builder = RoboCasaSceneBuilder(self)
        idx = 0 if self.scene_idx is None else self.scene_idx
        self.scene_builder.build([idx] * self.num_envs)

        self._fix_ds_fetch_collision_bits()

        k = self.cfg.k_stations
        names_per_env, dock_pos, dock_yaw, act_pt, beacon_pos = [], [], [], [], []
        station_across, station_top_z = [], []

        for i in range(self.num_envs):
            fixtures = self.scene_builder.scene_data[i]["fixtures"]
            cand = station_candidates(
                fixtures,
                classes=self.cfg.station_classes,
                min_top_z=self.cfg.station_min_top_z,
                max_top_z=self.cfg.station_max_top_z,
            )
            # One station per dock (K28): the drawers of one cabinet stack and the
            # sink over its cabinet share a dock to the millimetre on 102, and two
            # stations with one dock are one zone in every mode. Deduplicated
            # once, here, before the count — so `< k` counts distinct docks.
            raw_dock = np.stack([dock_pose_for(self.scene_builder, fixtures, n)[0] for n in cand]) \
                if cand else np.zeros((0, 3), dtype=np.float32)
            n_raw = len(cand)
            cand, cand_dock, merged = dedup_docks(cand, raw_dock, tol=self.cfg.dock_dedup_tol)
            for survivor, dropped in merged.items():
                print(
                    f"[station_checklist] env {i}: {survivor} keeps the dock shared with "
                    f"{dropped} (within {self.cfg.dock_dedup_tol} m); the others are not stations"
                )
            if len(cand) < k:
                raise KeyError(
                    f"kitchen {idx} (env {i}) offers {len(cand)} station-shaped "
                    f"fixtures with distinct docks ({n_raw} before de-duplication), "
                    f"need {k}. Found: {cand}. All fixtures: {sorted(fixtures)}. "
                    f"Widen station_classes, or pick a bigger layout with scene_idx."
                )
            chosen = select_spread_stations(cand, cand_dock, k)

            poses = [dock_pose_for(self.scene_builder, fixtures, n) for n in chosen]
            pos_i = np.stack([p for p, _ in poses])
            yaw_i = np.array([y for _, y in poses], dtype=np.float32)

            sep = np.linalg.norm(pos_i[:, None, :2] - pos_i[None, :, :2], axis=-1)
            sep = sep + np.eye(k, dtype=np.float32) * 1e6
            if not dock_separation_ok(float(sep.min()), self.cfg):
                raise RuntimeError(
                    f"two dock poses are {sep.min():.2f} m apart, below "
                    f"dock_min_separation={self.cfg.dock_min_separation}. Service "
                    f"zones would overlap and a commitment would be ambiguous. "
                    f"Either set zone_mode='nearest' or choose a larger kitchen.\n"
                    f"stations: {chosen}"
                )

            # Service point on the near edge of the fixture, not its centre: a Fetch
            # standing 0.8 m back cannot reach the middle of a microwave. The
            # beacon is a separate point (K32): outside the face, over the base
            # counter's top, so it is not buried in the cabinet body.
            act_i, beac_i, across_i, top_i = [], [], [], []
            for n in chosen:
                top, _, across, size = fixture_frame(fixtures[n])
                base_top_z = float(fixture_frame(base_fixture_for(fixtures, n))[0][2])
                act, beac = station_points(top, across, size, self.cfg, base_top_z=base_top_z)
                act_i.append(act)
                beac_i.append(beac)
                across_i.append(across)
                top_i.append(float(top[2]))
            names_per_env.append(chosen)
            dock_pos.append(pos_i)
            dock_yaw.append(yaw_i)
            act_pt.append(np.stack(act_i))
            beacon_pos.append(np.stack(beac_i))
            station_across.append(np.stack(across_i))
            station_top_z.append(np.asarray(top_i, dtype=np.float32))

        if any(n != names_per_env[0] for n in names_per_env):
            raise RuntimeError(
                "parallel environments selected different stations, so station index "
                f"s means different things per env: {names_per_env}"
            )
        self.station_names = names_per_env[0]

        self._dock_pos_np = np.stack(dock_pos).astype(np.float32)
        self._dock_yaw_np = np.stack(dock_yaw).astype(np.float32)
        self._act_pt_np = np.stack(act_pt).astype(np.float32)
        self._beacon_pos_np = np.stack(beacon_pos).astype(np.float32)
        # For the oracle's free-space activation pose (station_checklist_planner):
        # the fixture's `across` (unit, near edge -> centre) and its own top z.
        self._station_across_np = np.stack(station_across).astype(np.float32)
        self._station_top_z_np = np.stack(station_top_z).astype(np.float32)
        self._room_centre = self._dock_pos_np[0].mean(axis=0)

        self.beacons = BeaconGrid(
            self,
            positions=self._beacon_pos_np,
            radius=self.cfg.beacon_radius,
        )

        dev = self.device
        self._dock_pos = torch.tensor(self._dock_pos_np, dtype=torch.float32, device=dev)
        self._dock_yaw = torch.tensor(self._dock_yaw_np, dtype=torch.float32, device=dev)
        self._act_pt = torch.tensor(self._act_pt_np, dtype=torch.float32, device=dev)

    def _fix_ds_fetch_collision_bits(self):
        """Restore the wheel/base exemption scene_builder.py:490 skips for our uid."""
        if self.robot_uids == "fetch" or self.agent is None:
            return
        for link in self.agent.robot.links:
            for body in link._bodies:
                for shape in body.get_collision_shapes():
                    groups = shape.get_collision_groups()
                    for bit in range(25, 30):
                        groups[2] |= 1 << bit
                    shape.set_collision_groups(groups)

    # ------------------------------------------------------------- initialize --

    def _after_reconfigure(self, options: dict):
        n, k = self.num_envs, self.cfg.k_stations
        dev = self.device
        b = self.cfg.k_stations - self.cfg.m_completed

        z = lambda *shape, dtype=torch.bool: torch.zeros(shape, dtype=dtype, device=dev)  # noqa: E731
        self.completed_mask = z(n, k)
        self.blinded_mask = z(n, k)
        self.serviced_mask = z(n, k)
        self.in_zone_prev = z(n, k)
        self.visit_committed = z(n, k)
        self.gesture_active = z(n)
        self.first_decision_made = z(n)
        self.first_decision_ok = z(n)

        self.budget_left = torch.full((n,), b, dtype=torch.int32, device=dev)
        self.gesture_timer = z(n, dtype=torch.int32)
        self.gesture_station = torch.full((n,), -1, dtype=torch.int32, device=dev)
        self.double_service_count = z(n, dtype=torch.int32)
        self.revisit_count = z(n, dtype=torch.int32)
        self.decision_correct_n = z(n, dtype=torch.int32)
        self.decision_total_n = z(n, dtype=torch.int32)
        self.instruction_idx = z(n, dtype=torch.int32)

        self.visit_station = torch.full((n, b), -1, dtype=torch.int32, device=dev)
        self.visit_step = torch.full((n, b), -1, dtype=torch.int32, device=dev)
        self.visit_correct = z(n, b)

        self._last_eval_step = torch.full((n,), -1, dtype=torch.int32, device=dev)
        # -1 means "the beacons on screen do not match blinded_mask". evaluate()
        # resyncs when it sees this, which is what makes a restored checkpoint show
        # the right beacons — and what makes the paired memory-free control possible
        # without replaying actions (spec §6, trap 2).
        self._beacon_applied = torch.full((n,), -1, dtype=torch.int32, device=dev)

        # Action layout, read from the controller rather than hardcoded as 8 and 9.
        # Two index sets, kept apart on purpose (K31): `_head_pan_i/_head_tilt_i`
        # address the ACTION vector (`_override_action` writes the head script
        # there), `_head_qpos_idx`/`_arm_qpos_idx` address `get_qpos()`. Under
        # pd_joint_pos the action is arm|gripper|body|base while qpos follows the
        # URDF (root x/y/yaw, torso, head, arm, fingers), so the same integer names
        # different joints in the two vectors.
        ctrl = self.agent.controller
        a_start, a_end = ctrl.action_mapping["arm"]
        b_start, _ = ctrl.action_mapping["body"]
        body_joints = list(ctrl.controllers["body"].config.joint_names)
        self._arm_slice = (a_start, a_end)
        self._head_pan_i = b_start + body_joints.index("head_pan_joint")
        self._head_tilt_i = b_start + body_joints.index("head_tilt_joint")
        self._n_arm = a_end - a_start
        self._gesture_profile = torch.tensor(
            gesture_profile(self.cfg, self._n_arm), dtype=torch.float32, device=dev
        )
        joints_map = self.agent.robot.active_joints_map
        self._arm_qpos_idx = [
            joints_map[j].active_index[0].item() for j in ctrl.controllers["arm"].config.joint_names
        ]
        self._head_qpos_idx = [
            joints_map[j].active_index[0].item() for j in ("head_pan_joint", "head_tilt_joint")
        ]
        # `self.agent.action_space`, not `self.action_space`: BaseEnv assigns the
        # latter only after its first reset (sapien_env.py:338-346), i.e. after this
        # hook has run — the first build died here with AttributeError (K27).
        self._freeze_action = z(n, int(self.agent.action_space.shape[-1]), dtype=torch.float32)
        self._gesture_anchor = z(n, self._n_arm, dtype=torch.float32)
        return super()._after_reconfigure(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            k = self.cfg.k_stations

            self.scene_builder.initialize(env_idx)
            self._restore_robot(env_idx)

            rng = self._batched_episode_rng[env_idx]
            # ONE permutation gives both which stations are completed and which of
            # those go dark. Drawing them separately, or rejection-sampling, would
            # make the RNG consumption depend on `ell` — a timing leak at the level
            # of the random stream rather than of the clock (spec trap 4).
            perm = np.stack([np.asarray(p) for p in rng.permutation(k)])
            completed = perm[:, : self.cfg.m_completed]
            blinded = completed[:, : self.cfg.ell_blinded]
            instr = np.asarray(rng.randint(0, len(INSTRUCTIONS))).reshape(b)
            jitter = np.asarray(rng.uniform(-1.0, 1.0)).reshape(b)

            comp_t = torch.zeros((b, k), dtype=torch.bool)
            blind_t = torch.zeros((b, k), dtype=torch.bool)
            rows = torch.arange(b).unsqueeze(-1)
            comp_t[rows, torch.as_tensor(completed)] = True
            blind_t[rows, torch.as_tensor(blinded)] = True

            # Certification hook: an override replaces the draw but does NOT skip it,
            # so the random stream advances identically either way.
            ov = (options or {}).get("station_checklist") or {}
            if "completed" in ov:
                comp_t = torch.as_tensor(ov["completed"], dtype=torch.bool).reshape(b, k)
            if "blinded" in ov:
                blind_t = torch.as_tensor(ov["blinded"], dtype=torch.bool).reshape(b, k)

            self.completed_mask[env_idx] = comp_t
            self.blinded_mask[env_idx] = blind_t
            self.instruction_idx[env_idx] = torch.as_tensor(instr, dtype=torch.int32)

            # Viewing pose: back off from the centroid of the dock poses, jittered.
            centre = self._dock_pos[env_idx].mean(dim=1)
            offset = torch.tensor([0.0, -self.cfg.viewpoint_standoff, 0.0])
            p = centre + offset
            p[:, :2] += torch.as_tensor(jitter, dtype=torch.float32).unsqueeze(-1) * (
                self.cfg.viewpoint_jitter_xy
            )
            p[:, 2] = self.agent.robot.pose.p[env_idx][:, 2]
            yaw = torch.full((b,), math.pi / 2) + torch.as_tensor(
                jitter, dtype=torch.float32
            ) * self.cfg.viewpoint_jitter_yaw
            q = torch.stack(
                [torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)],
                dim=1,
            )
            self.agent.robot.set_pose(Pose.create_from_pq(p=p, q=q))

            self.serviced_mask[env_idx] = False
            self.in_zone_prev[env_idx] = False
            self.visit_committed[env_idx] = False
            self.gesture_active[env_idx] = False
            self.gesture_timer[env_idx] = 0
            self.gesture_station[env_idx] = -1
            self.budget_left[env_idx] = k - self.cfg.m_completed
            self.double_service_count[env_idx] = 0
            self.revisit_count[env_idx] = 0
            self.decision_correct_n[env_idx] = 0
            self.decision_total_n[env_idx] = 0
            self.first_decision_made[env_idx] = False
            self.first_decision_ok[env_idx] = False
            self.visit_station[env_idx] = -1
            self.visit_step[env_idx] = -1
            self.visit_correct[env_idx] = False
            self._last_eval_step[env_idx] = -1
            self._beacon_applied[env_idx] = -1

            # Snapshot the hold action once, from the restored pose. Recomputing it
            # every step would let the arm drift a little further each time.
            self._freeze_action[env_idx] = self._hold_action()[env_idx]

    def _restore_robot(self, env_idx: torch.Tensor):
        """scene_builder.initialize only restores the robot when uid == "fetch".

        scene_builder.py:563-577 compares a literal string, so `ds_fetch` gets
        nothing and episode 2 starts with the arm wherever episode 1 left it.
        """
        if self.robot_uids == "fetch" or self.agent is None:
            return
        self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
        self.agent.robot.set_pose(self.scene_builder.robot_poses[env_idx])

    def _hold_action(self) -> torch.Tensor:
        """The action that means "stay put" under pd_joint_pos.

        Not zeros: the arm and body controllers take absolute targets with
        normalize_action=False, so a zero action commands qpos 0 and drops the torso
        through the floor.
        """
        qpos = self.agent.robot.get_qpos()
        parts = []
        for name, ctrl in self.agent.controller.controllers.items():
            dim = int(np.prod(ctrl.single_action_space.shape))
            if "base" in name:
                parts.append(torch.zeros((qpos.shape[0], dim), device=qpos.device))
            else:
                idx = list(ctrl.active_joint_indices)[:dim]
                parts.append(qpos[:, idx])
        return torch.cat(parts, dim=1)

    # ----------------------------------------------------------- action gate --

    def _step_action(self, action):
        """Override the head, freeze the encoding phase, and drive the gesture.

        This is `_step_action` rather than `_before_control_step` for a concrete
        reason: the body controller runs with `interpolate=True` (ds_fetch.py:200),
        and an interpolating PD controller does not set its drive targets in
        `set_action` — it sets them in `before_simulation_step`, once per sub-step
        (pd_joint_pos.py:90-99), which happens *after* `_before_control_step`.
        Anything written there would be overwritten before the first sub-step.

        Note `elapsed_steps` here is the value *before* the increment
        (sapien_env.py:1050-1051), so `t < cue_steps` covers actions 0..cue_steps-1
        and `evaluate()` sees `t >= cue_steps` for the observation that follows.
        """
        return super()._step_action(self._override_action(action))

    def _override_action(self, action):
        if action is None:
            return self._freeze_action.clone()
        if isinstance(action, dict):
            return action  # a control-mode switch; nothing to patch
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if act.ndim == 1:
            act = act.unsqueeze(0).repeat(self.num_envs, 1)
        act = act.clone()

        t = self.elapsed_steps
        encoding = t < self.cfg.cue_steps
        # The spec says the robot stands in a fixed viewing pose during encoding.
        # Replacing the action outright makes that true rather than hoped for, and
        # it is what keeps the encoding window exactly cue_steps long for every
        # policy and every value of ell.
        act[encoding] = self._freeze_action[encoding]

        # While the environment plays a gesture it owns the arm, so the duration is
        # identical every time — the motor invariance the load axis rests on.
        busy = self.gesture_active & ~encoding
        if bool(busy.any()):
            step = self.gesture_timer.clamp(0, self.cfg.gesture_steps - 1).long()
            a0, a1 = self._arm_slice
            act[busy, a0:a1] = (
                self._gesture_anchor[busy] + self._gesture_profile[step[busy]]
            )
            for name, ctrl in self.agent.controller.controllers.items():
                if "base" in name:
                    s, e = self.agent.controller.action_mapping[name]
                    act[busy, s:e] = 0.0

        pan, tilt = head_script(int(t[0].item()) if t.numel() else 0, self.cfg)
        act[:, self._head_pan_i] = pan
        act[:, self._head_tilt_i] = tilt
        return act

    # --------------------------------------------------------------- evaluate --

    def evaluate(self) -> dict:
        t = self.elapsed_steps.to(torch.int32)
        advance = t != self._last_eval_step
        is_encoding = self.elapsed_steps < self.cfg.cue_steps
        is_action = ~is_encoding

        # --- beacons, level-triggered ---------------------------------------
        # Written from the current phase rather than fired on the transition, so an
        # out-of-band evaluate() (myrobocasa_planner.py:284, diagnose_task.py:58)
        # can neither skip the event nor trigger it twice.
        want_lit = self.completed_mask & ~(self.blinded_mask & is_action.unsqueeze(-1))
        phase_id = is_action.to(torch.int32)
        stale = self._beacon_applied != phase_id
        if bool(stale.any()):
            self.beacons.set_lit(want_lit.cpu().numpy())
            self._beacon_applied = phase_id.clone()

        # --- where the base is ----------------------------------------------
        # agent.base_link, not agent.robot: Fetch drives through root_x/root_y/
        # root_z_rotation joints (fetch.urdf:21,28,35), so the articulation's root
        # pose stays at the spawn point for the whole episode.
        base_pose = self.agent.base_link.pose
        base_p = base_pose.p
        base_yaw = torch.atan2(
            2.0 * (base_pose.q[:, 0] * base_pose.q[:, 3] + base_pose.q[:, 1] * base_pose.q[:, 2]),
            1.0 - 2.0 * (base_pose.q[:, 2] ** 2 + base_pose.q[:, 3] ** 2),
        )
        d_xy = torch.linalg.norm(base_p[:, None, :2] - self._dock_pos[:, :, :2], dim=-1)
        dyaw = torch.atan2(
            torch.sin(base_yaw.unsqueeze(-1) - self._dock_yaw),
            torch.cos(base_yaw.unsqueeze(-1) - self._dock_yaw),
        ).abs()
        facing = dyaw <= math.radians(self.cfg.dock_yaw_tol_deg)

        if self.cfg.zone_mode == "radius":
            in_zone = (d_xy <= self.cfg.dock_radius) & facing
        else:
            nearest = d_xy.argmin(dim=-1, keepdim=True)
            in_zone = torch.zeros_like(d_xy, dtype=torch.bool)
            in_zone.scatter_(1, nearest, True)
            in_zone &= (d_xy <= self.cfg.zone_max_range) & facing

        tcp = self.agent.tcp.pose.p
        d_tcp = torch.linalg.norm(tcp[:, None, :2] - self._act_pt[:, :, :2], dim=-1)
        dz = tcp[:, None, 2] - self._act_pt[:, :, 2]
        at_station = (
            (d_tcp <= self.cfg.gesture_xy_radius)
            & (dz >= self.cfg.gesture_z_low)
            & (dz <= self.cfg.gesture_z_high)
        )
        base_static = (
            torch.linalg.norm(self.agent.robot.get_qvel()[:, :3], dim=-1)
            < self.cfg.base_static_speed
        )

        # One commitment per zone visit: `visit_committed[s]` is set by `_commit`
        # and cleared only when the base LEAVES s's zone (below). Without the term
        # the same station re-committed the step its gesture ended while the robot
        # still stood in the zone — every station serviced twice, the budget gone,
        # exact cover unreachable (read 2026-08-18; the fix is the task's stated
        # intent, "exactly k − m gestures", not a weaker predicate).
        ready = in_zone & at_station & ~self.visit_committed
        can_commit = (
            advance
            & is_action
            & (self.budget_left > 0)
            & ~self.gesture_active
            & base_static
            & ready.any(dim=-1)  # never argmax an all-False row: it returns 0
        )

        if bool(can_commit.any()):
            self._commit(can_commit, ready, t)

        if bool(advance.any()):
            self._tick_gesture(advance)
            left = ~in_zone & self.in_zone_prev & advance.unsqueeze(-1)
            # Order matters (K30, regression R3): count the departures that
            # happened without a commitment FIRST, then clear the latch for every
            # zone that was left. The other way round, a serviced-and-left station
            # reads as "left without a gesture" and revisit_count lies.
            self.revisit_count = self.revisit_count + (
                left & ~self.visit_committed
            ).sum(dim=-1).to(torch.int32)
            self.visit_committed = self.visit_committed & ~left
            self.in_zone_prev = torch.where(
                advance.unsqueeze(-1), in_zone, self.in_zone_prev
            )
            self._last_eval_step = torch.where(advance, t, self._last_eval_step)

        uncompleted = ~self.completed_mask
        omission = (uncompleted & ~self.serviced_mask).sum(dim=-1).to(torch.int32)
        episode_over = (self.budget_left <= 0) & ~self.gesture_active
        exact_cover = (self.serviced_mask == uncompleted).all(dim=-1)
        success = episode_over & exact_cover
        fail = episode_over & ~exact_cover

        return {
            "success": success,
            "fail": fail,
            "is_encoding": is_encoding,
            "budget_left": self.budget_left,
            "serviced_count": self.serviced_mask.sum(dim=-1).to(torch.int32),
            "double_service_count": self.double_service_count,
            "omission_count": omission,
            "revisit_count": self.revisit_count,
            "decision_correct_n": self.decision_correct_n,
            "decision_total_n": self.decision_total_n,
            "first_decision_made": self.first_decision_made,
            "first_decision_ok": self.first_decision_ok,
            "in_zone_any": in_zone.any(dim=-1),
            "gesture_active": self.gesture_active,
            "beacon_lit": want_lit,
            # Ground truth for the oracle and the trajectory file. info is not an
            # observation: only what _get_obs_extra copies out reaches a policy.
            "completed_mask": self.completed_mask,
            "blinded_mask": self.blinded_mask,
            "serviced_mask": self.serviced_mask,
            "uncompleted_mask": uncompleted,
            "visit_station": self.visit_station,
            "visit_step": self.visit_step,
            "visit_correct": self.visit_correct,
        }

    def _commit(self, can_commit: torch.Tensor, ready: torch.Tensor, t: torch.Tensor):
        """Latch a decision: this is the event the primary endpoint is measured on."""
        s = ready.float().argmax(dim=-1)
        rows = torch.arange(self.num_envs, device=self.device)
        sel = can_commit

        already = self.serviced_mask[rows, s]
        done_before = self.completed_mask[rows, s]
        correct = (~done_before) & (~already)

        b = self.cfg.k_stations - self.cfg.m_completed
        slot = (b - self.budget_left).clamp(0, b - 1).long()

        self.serviced_mask[rows[sel], s[sel]] = True
        self.visit_committed[rows[sel], s[sel]] = True
        self.double_service_count = self.double_service_count + (sel & already).to(torch.int32)
        self.decision_total_n = self.decision_total_n + sel.to(torch.int32)
        self.decision_correct_n = self.decision_correct_n + (sel & correct).to(torch.int32)

        first = sel & ~self.first_decision_made
        self.first_decision_ok = torch.where(first, correct, self.first_decision_ok)
        self.first_decision_made = self.first_decision_made | sel

        self.visit_station[rows[sel], slot[sel]] = s[sel].to(torch.int32)
        self.visit_step[rows[sel], slot[sel]] = t[sel]
        self.visit_correct[rows[sel], slot[sel]] = correct[sel]

        self.budget_left = self.budget_left - sel.to(torch.int32)
        self.gesture_active = self.gesture_active | sel
        self.gesture_timer = torch.where(sel, torch.zeros_like(self.gesture_timer), self.gesture_timer)
        self.gesture_station = torch.where(sel, s.to(torch.int32), self.gesture_station)
        anchor = self.agent.robot.get_qpos()[:, self._arm_qpos_idx]
        self._gesture_anchor = torch.where(sel.unsqueeze(-1), anchor, self._gesture_anchor)

    def _tick_gesture(self, advance: torch.Tensor):
        running = self.gesture_active & advance
        self.gesture_timer = self.gesture_timer + running.to(torch.int32)
        done = running & (self.gesture_timer >= self.cfg.gesture_steps + self.cfg.settle_steps)
        self.gesture_active = self.gesture_active & ~done
        self.gesture_station = torch.where(
            done, torch.full_like(self.gesture_station, -1), self.gesture_station
        )

    # -------------------------------------------------------------------- obs --

    def _get_obs_extra(self, info: dict) -> dict:
        """What the policy is allowed to see. The omissions are the design.

        `beacon_lit` is given honestly — it is the state-mode counterpart of the
        pixels, and with ell == m it is identically zero after the event, so it
        carries no answer. Without it a state-mode run could not tell "the policy
        forgot" from "the policy was never told".

        What is never given: `budget_left`, `serviced`, `completed`, `blinded`,
        `visit_*`. The construct under test is `action_history`; handing the agent
        its own history of actions would switch off the thing being measured. This
        is stricter than spec §5, which only forbids `completed` and `serviced`, and
        the extra strictness is deliberate.

        Key order is fixed and no key is named for a target — in state mode the dict
        is flattened in insertion order, so a role-ordered key would put the answer
        at a constant index.
        """
        t = self.elapsed_steps
        encoding = (t < self.cfg.cue_steps).to(torch.float32)
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            base_pose=self.agent.base_link.pose.raw_pose,
            base_qvel=self.agent.robot.get_qvel()[:, :3],
            head_qpos=self.agent.robot.get_qpos()[:, self._head_qpos_idx],
            phase=torch.stack([encoding, 1.0 - encoding], dim=1),
            beacon_lit=info["beacon_lit"].to(torch.float32),
            instruction_id=torch.nn.functional.one_hot(
                self.instruction_idx.long(), num_classes=len(INSTRUCTIONS)
            ).to(torch.float32),
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                station_pos=self._act_pt.reshape(self.num_envs, -1),
                dock_pos=self._dock_pos.reshape(self.num_envs, -1),
            )
        return obs

    def get_language_instruction(self, **kwargs):
        return [INSTRUCTIONS[int(i)] for i in self.instruction_idx.tolist()]

    # ------------------------------------------------------------------ state --

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        for name in (
            "completed_mask",
            "blinded_mask",
            "serviced_mask",
            "in_zone_prev",
            "visit_committed",
            "gesture_active",
            "gesture_timer",
            "gesture_station",
            "budget_left",
            "double_service_count",
            "revisit_count",
            "decision_correct_n",
            "decision_total_n",
            "first_decision_made",
            "first_decision_ok",
            "instruction_idx",
            "visit_station",
            "visit_step",
            "visit_correct",
        ):
            state[name] = getattr(self, name).clone()
        state["last_eval_step"] = self._last_eval_step.clone()
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        """Tolerates a state dict with no task keys, because one is routine.

        BaseEnv.set_state(flat) rebuilds a dict holding only "actors" and
        "articulations" (sapien_env.py:1309-1327), so every task key is gone by the
        time it arrives here.
        """
        super().set_state_dict(state, env_idx)
        for name in (
            "completed_mask",
            "blinded_mask",
            "serviced_mask",
            "in_zone_prev",
            "visit_committed",
            "gesture_active",
            "gesture_timer",
            "gesture_station",
            "budget_left",
            "double_service_count",
            "revisit_count",
            "decision_correct_n",
            "decision_total_n",
            "first_decision_made",
            "first_decision_ok",
            "instruction_idx",
            "visit_station",
            "visit_step",
            "visit_correct",
        ):
            if name in state:
                setattr(self, name, restore_task_tensor(getattr(self, name), state[name], self.device))
        if "last_eval_step" in state:
            self._last_eval_step = restore_task_tensor(
                self._last_eval_step, state["last_eval_step"], self.device
            )
        # Beacon colours are render state and are not serialised. Invalidating the
        # cache makes the next evaluate() repaint them from the restored masks —
        # which is also the whole mechanism behind the paired memory-free control:
        # restore, overwrite blinded_mask, render. No action replay (spec trap 2).
        self._beacon_applied[:] = -1
