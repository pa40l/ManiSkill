"""The burner that went out — a spatial memory task on the RoboCasa stove, driven by Fetch.

The chore: one of four burners on the stove briefly "burns" (an orange marker sits
on it), then goes out. The robot, holding position at the far counter with a cup in
front of it, must drive to the stove and put the cup on **that** burner. After the
marker is gone the four burners are identical — the marker is a separate actor, not
a stove texture, so nothing on the scene records which one was lit.

Design source: `.claude/skills/designing-a-memory-task/references/burner-design-example.md`
(the filled-in 8-point form this file implements). Conventions: docs/writing-tasks.md
and AGENTS.md "Writing a memory task"; the worked references are season_dish.py and
example_memory_task.py.

Memoryless floors, stated so a success rate means something
-----------------------------------------------------------
- **first placement:** four burners, one correct → a policy that cannot remember
  puts the cup on the right burner 1/4 of the time;
- **episode success:** the same 1/4, because the first settled placement decides
  the episode — see the latch below. Success = 0.25 x motor success rate, where
  the motor rate is what the oracle's seed sweep reports. The `--blind` arm of
  `planners/burner_planner.py` runs the same script with a random burner and must
  land at that number; anything above it means the cue leaks into the act phase.

One caveat the frames showed (2026-08-16, CPU, `reading-sim-frames`): the kitchen-102
stove `basic_sleek_induc` has **five** burner sites (`rear_center` is the fifth,
`burner_rear_center_place_site` in its MJCF), and after the marker goes out a camera
sees five identical zones, not four. That closes the census question ("burner count of
`basic_sleek_induc`") — but it does **not** lower the floor to 1/5. A cup settled on the
uncued `rear_center` matches no entry of `burner_pos` (built from `BURNER_LOCATIONS[:n]`
in `_load_scene`), so `score_placement` returns neither `placed` nor the
`wrong_burner_settled` latch: the episode is not lost and the policy may place again.
A policy picking uniformly among the five zones it sees therefore places correctly 1/5
of the time on its **first** placement, but converges on (1/5)/(1 - 1/5) = 1/4 per
episode. So 1/5 is a first-placement statistic, not the floor; quote 1/4 as the
memoryless ceiling. The `--blind` arm draws from `range(n_burners)`
(`planners/burner_planner.py`, `choose_target`) and cannot produce 1/5 at all.

Why 1/4 is true and not merely claimed: `wrong_burner_settled` latches the moment
the cup rests, ungrasped, on a wrong burner in the act phase, and success is
unreachable afterwards. Without it the episode horizon (`BurnerConfig.horizon`,
derived from a measured successful demo) lets a memoryless policy try all four
burners in turn (docs/memory-benchmark-primer.md, principle 5).

Phases (per env, drawn from the episode RNG, never constants — principle 4)
------------------------------------------------------------------------
    cue    elapsed <  cue_steps                        marker on burner cue_id
    delay  cue_steps <= elapsed < cue_steps + delay    marker at z = HIDDEN_Z
    act    elapsed >= cue_steps + delay                verdicts on

Driving is allowed during the delay — the road to the stove naturally eats the
delay, which is what makes a frame buffer useless here (principle 4). Verdicts
(success and the latch) are gated on the act phase.

Where the marker moves, and why not in evaluate()
--------------------------------------------------
The marker pose is written from the step hook `_before_control_step`, not from
`evaluate()`. `evaluate()` here is *pure* apart from the latch, so `diagnose_task`
and `state_diff` may call it any number of times without moving anything.
`_before_control_step` exists in the base class and runs inside `_step_action`
before the physics sub-steps: sapien_env.py:1022 (3.0.0b14) / :1122 (3.0.1), with the
sub-steps at :1023-1028 / :1123-1128 and `_gpu_fetch_all` at :1031 / :1131.
`elapsed_steps` is incremented only after `_step_action` returns (:957 / :1049),
so the hook sees the *previous* step and applies the schedule for `elapsed + 1` —
the step the coming observation will carry. Semantics: the marker is on the burner
in the observation after step k if and only if k < cue_steps; it is also applied at
t = 0 from `_initialize_episode`. The write is level-triggered from
`elapsed_steps` and idempotent (a masked batched pose write to the burner or to
`[0, 0, HIDDEN_Z]`), guarded by the `_marker_applied_step` cache.

**GPU apply not verified.** Under GPU sim `Actor.pose = ...` writes into
`cuda_rigid_body_data` (structs/actor.py:365) and nothing in `_step_action` calls
`_gpu_apply_all` after `_before_control_step` — only `_gpu_fetch_all` at :1031,
which would overwrite the buffer with the sim's pose. `_apply_marker` therefore
calls `px.gpu_apply_rigid_dynamic_data()` itself when `gpu_sim_enabled`. That
line has never run: this task has only been built on CPU (`sim_backend="cpu"`,
`num_envs=1`) on a Mac. Never run on GPU; never run with mplib.

Burner positions
----------------
The ManiSkill port of `Stove` keeps `get_reset_regions`, but it reads
`self.worldbody`, which the port never sets, so it raises AttributeError in both
b14 and 3.0.1. The MJCF tree is still there as `fixture.loader.xml`, and the sites
`burner_<loc>_place_site` are in it, in the fixture's body frame at scale 1. World
position = `fixture.pos + R_z(fixture.rot) @ (site_pos * fixture._scale)` — the same
arithmetic `get_pos_after_rel_offset` uses, plus the scale the port applies to
`_bounds_sites` in `MujocoObject.set_scale`. `burner_sites_world` does exactly that
and falls back to a 2 x 2 grid over the stove's exterior bounding box when a
kitchen's stove has no such sites — the fallback logs a warning and the task
records which one it used in `self.burner_positions_from_sites`. Measured in
kitchen 0 (`scene_idx=0`, layout one_wall_small, style industrial), interpreter
`course/.venv-cpu` (mani_skill 3.0.1, CPU): sites found; burners at
(2.95, -0.47), (3.32, -0.47), (2.90, -0.22), (3.36, -0.22), cooktop z = 0.92 (the
site z is 0.908, the cup rests with its origin at 0.974); cup home (1.92, -0.485)
on `counter_main`; robot base start (1.92, -1.40) facing +y. The cached stove dock
(`_stove_dock_np`, (3.134, -1.46 + `stove_dock_toward`, pi/2) in kitchen 0) faces
the middle of the stove at RoboCasa's 0.8 m standoff minus the knob; the oracle
slides it along the stove's face to the *target burner's column* before driving
(`burner_planner.dock_for_target`), so the reach the hover needs is the
across-face distance only — 0.99 m to the front row, 1.24 m to the rear, minus
`stove_dock_toward`.

Is the cue observable from the start pose? (frames read 2026-08-16, CPU, seed 3)
--------------------------------------------------------------------------------
Design check 1 ("the stove is visible from the start") holds only per camera.
`render_camera`: marker 12 x 12 px, four burners in view. `right_base_camera_link`
(256 px, on the torso): marker 10 x 10 px, unoccluded, but the right edge of the
cooktop is clipped by the frame, so not all burners are in view at once.
`base_camera` (the task's own 128 px wide-angle): whole stove in frame, but the
cooktop is 25 x 11 px and the marker 5 px. `left_base_camera_link`: the robot's own
arm and torso fill the right half of the frame; the stove is not in view and the
marker shows only as a ~6 x 8 px sliver between two links. Consequence: a policy on
the stereo pair sees the cue with one eye and cannot see all four burners; one on
`base_camera` sees everything at a few pixels. **Open design item, not fixed here**:
turn the start yaw ~30 deg toward the stove, or narrow/aim `base_camera` at the
stove (FOV pi/2 at 1.6 m is what makes it 25 px), or both — then re-read the
frames. Recorded in docs/lab-journal.md.
"""

from __future__ import annotations

import logging
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
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig

from utils.mikasa.scenes.robocasa_utils import (
    parking_pose,
    counter_frame,
    dock_pose_for,
    load_objaverse_actor,
    require_get_fixture,
    restore_task_tensor,
)

logger = logging.getLogger(__name__)

# Where the marker goes while it must not be seen. MIKASA-Robo's constant
# (memory_envs/chain_rule_transfer.py:47) and idiom (remember_color.py:203).
HIDDEN_Z = 1000.0

# The four burners, in a fixed order. Index i of every per-burner tensor means this
# location; the oracle iterates nothing spatial. rear_center exists on this stove
# model too and is deliberately unused: four is the design's chance floor.
BURNER_LOCATIONS = ("front_left", "front_right", "rear_left", "rear_right")

# No digits, no number words: "one of the four burners" would hand a
# language-conditioned policy the size of the answer space.
INSTRUCTION = "Put the cup on the burner that was lit."

# What get_state_dict adds on top of the sim state. The task's entire memory.
TASK_STATE_KEYS = (
    "cue_id",
    "cue_steps",
    "delay_steps",
    "wrong_burner_settled",
    "last_eval_step",
)


@dataclass
class BurnerConfig:
    """Every threshold the task uses, named once, so a failed run can say which one it missed."""

    horizon: int = 800
    """Episode length. Must equal max_episode_steps in the decorator.

    From the measured demo (K22, container, CPU, emulated, seeds 3 and 1,
    2026-08-18): elapsed_at_landed − cue_steps = 375 (seed 3) and 392 (seed 1);
    manip_steps = 392 × 1.5 rounded up to a multiple of 50 = 600; horizon =
    cue_hi (40) + delay_hi (120) + manip_steps, rounded up to a multiple of
    100 = 800. The demo was the oracle's staging without the carry stage (the
    carry is unplannable under mplib's attached-body frame error — see
    docs/lab-journal.md T3); the ×1.5 margin is what absorbs the tuck once a
    solver decision lands.

    Two honest caveats (final review, 2026-08-18). (1) Seeds 3 and 1 ARE eval
    seeds (the eval set is 0–9): the plan named them for T3 before K22 froze
    the eval set, so the horizon and the dock standoff below were fitted on two
    of the ten seeds the published 6/10 is measured on. (2) Re-derived on the
    out-of-eval seeds 11 and 12 with the SHIPPED oracle (carry stage included,
    post-K53): seed 11 lands at 657 with cue 31 → manip 626 → ×1.5 = 939 → 950
    → horizon 40 + 120 + 950 = 1110 → 1200; seed 12 refuses the rear-row hover
    (approximate RRT) and never lands, so it gives no number. The K22 formula on
    the shipped oracle therefore says 1200, not 800. 800 was NOT changed: on the
    eval set every one of the 20 post-K53 episodes (both arms) finishes under
    it — the longest verdict is at step 776 and `truncated 0` on both sweeps —
    and changing it now would invalidate the published pair. Re-deriving the
    horizon on out-of-eval seeds is filed in TODOS.md / the journal's Open list.
    """

    n_burners: int = 4
    "Size of the answer space. The memoryless first-placement floor is 1 / n_burners."

    cue_steps_range: tuple[int, int] = (15, 40)
    "How long the marker sits on the burner, drawn per env per episode (inclusive)."

    delay_steps_range: tuple[int, int] = (20, 120)
    "Blank window between the marker vanishing and verdicts switching on (inclusive)."

    manip_steps: int = 600
    """Budget for drive + place after the cue, from the measured demo (see `horizon`).

    max(elapsed_at_landed − cue_steps) over the calibration seeds 3 and 1, times
    1.5, rounded up to a multiple of 50. validate() keeps it inside the horizon.
    Seeds 3 and 1 overlap the frozen eval set 0–9 (see `horizon`); the
    out-of-eval re-derivation on seed 11 with the shipped oracle gives 950.
    """

    on_burner_radius: float = 0.08
    "Cup centre within this xy distance of a burner centre counts as on that burner."

    place_z_band: tuple[float, float] = (-0.02, 0.12)
    "Cup origin height relative to the burner top; the upper bound catches 'still held in the air'."

    settle_lin_vel: float = 0.05
    settle_ang_vel: float = 0.2
    "The cup must have come to rest before any verdict — success or the latch."

    cup_along: float = 0.42
    "Cup home along `counter_main`, from its centre. Positive is toward the stove in kitchen 0."
    cup_across: float = -0.16
    "Toward the robot side of the counter. Negative is 'in front' in kitchen 0."
    cup_clearance: float = 0.10
    """Spawn height of the cup *origin* above the work surface; physics settles the rest.

    Not 0.02: cup_2's origin is at its centre, ~5.4 cm above its base (measured: it
    rests at dz = 0.054 over the cooktop), so 2 cm buries it in the counter and
    PhysX ejects it — in the first smoke run the cup was 5 m away by step 37.
    """
    cup_jitter: float = 0.05
    "Uniform xy jitter of the cup home, +- this, per episode."

    marker_radius: float = 0.045
    marker_height: float = 0.07
    "Marker centre above the burner top while lit."
    marker_color: tuple[float, float, float, float] = (1.0, 0.45, 0.05, 1.0)
    hidden_z: float = HIDDEN_Z

    counter_query: str = "counter_main_main_group"
    "Exact fixture name of the work counter the cup spawns on."
    stove_classes: tuple[str, ...] = ("Stove", "Stovetop")
    "Fixture classes that count as the stove; the first by name is used (no RNG)."

    stove_dock_toward: float = 0.25
    """Metres the stove dock moves toward the stove from RoboCasa's 0.8 m standoff (D3).

    0.25 is the first rung of the measured ladder (container, CPU, seeds 3 and 1 —
    two of the ten eval seeds, see `horizon`; 2026-08-18), taken because the
    ladder's condition fired: from the 0.8 m dock
    the seed-3 rear-row hover is 1.23 m out and is refused (`converged 11.1 cm /
    8.4 deg`, then `IK Failed`) with the torso free, and the FK envelope probe
    caps the arm's forward reach at hover height at 1.065 m (torso down) /
    1.100 m (torso up) — the torso lever cannot close the gap, the dock must.
    At 0.25 both calibration seeds land on the cued burner and no base<->stove
    collision appears. validate() caps the knob at 0.4 so at least 0.4 m of
    standoff — roughly the Fetch base's footprint — always remains.
    """

    def validate(self) -> None:
        assert self.n_burners >= 2, self.n_burners
        assert self.n_burners <= len(BURNER_LOCATIONS), (
            f"n_burners={self.n_burners} but only {len(BURNER_LOCATIONS)} named locations"
        )
        for lo, hi in (self.cue_steps_range, self.delay_steps_range):
            assert 0 < lo <= hi, (lo, hi)
        assert self.on_burner_radius > 0, self.on_burner_radius
        assert self.marker_radius > 0, self.marker_radius
        assert self.place_z_band[0] < self.place_z_band[1], self.place_z_band
        assert self.settle_lin_vel > 0 and self.settle_ang_vel > 0
        assert self.cup_jitter >= 0
        assert 0.0 <= self.stove_dock_toward <= 0.4, self.stove_dock_toward
        worst = self.cue_steps_range[1] + self.delay_steps_range[1] + self.manip_steps
        assert worst < self.horizon, (
            f"cue + delay + manipulation can need {worst} steps of a {self.horizon}-step episode"
        )


def memoryless_floor(cfg: BurnerConfig) -> float:
    """First-placement chance level. Episode success is this times the motor rate."""
    return 1.0 / cfg.n_burners


# ------------------------------------------------------------ pure helpers --
# Module-level and free of `self`, so tests can run them without a simulator and
# so evaluate() stays a thin geometry layer over arithmetic that can be checked
# on paper.


def phase_masks(
    elapsed: torch.Tensor, cue_steps: torch.Tensor, delay_steps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cue_phase, act_phase)` boolean masks. The delay phase is neither."""
    elapsed = elapsed.to(torch.long)
    cue_phase = elapsed < cue_steps
    act_phase = elapsed >= cue_steps + delay_steps
    return cue_phase, act_phase


def score_placement(
    on_burner: torch.Tensor,
    settled: torch.Tensor,
    grasped: torch.Tensor,
    act_phase: torch.Tensor,
    cue_id: torch.Tensor,
    wrong_before: torch.Tensor,
) -> dict:
    """The verdict, as arithmetic over booleans.

    on_burner   (b, n) bool   cup within the placement zone of burner i
    settled     (b,)   bool   cup at rest
    grasped     (b,)   bool   fingers still on the cup
    act_phase   (b,)   bool   verdicts are on
    cue_id      (b,)   long   the answer
    wrong_before (b,)  bool   the latch as it stood before this step

    A placement only counts when the cup is at rest, released, in the act phase and
    on some burner. The latch is a priced mistake (primer, principle 5): once a
    placement lands on a wrong burner the episode is unwinnable, so a memoryless
    policy cannot search the four burners in turn.
    """
    b, n = on_burner.shape
    rows = torch.arange(b, device=on_burner.device)
    on_any = on_burner.any(dim=-1)
    on_cue = on_burner[rows, cue_id]
    placed = settled & ~grasped & act_phase & on_any
    wrong_now = wrong_before | (placed & ~on_cue)
    # -1 when on no burner; argmax of an all-False row would say 0.
    on_burner_id = torch.where(
        on_any, on_burner.to(torch.int64).argmax(dim=-1), torch.full((b,), -1, dtype=torch.int64, device=on_burner.device)
    )
    return {
        "success": placed & on_cue & ~wrong_now,
        "placed": placed,
        "on_any": on_any,
        "on_cue": on_cue,
        "on_burner_id": on_burner_id,
        "wrong_burner_settled": wrong_now,
    }


def burner_sites_world(fixture, locations=BURNER_LOCATIONS) -> np.ndarray | None:
    """World positions `(len(locations), 3)` of the stove's burner sites, or None.

    Reads `burner_<loc>_place_site` (fallback `burner_on_<loc>`) out of the fixture's
    parsed MJCF (`fixture.loader.xml`) — `Stove.get_reset_regions` would do this but
    dereferences `self.worldbody`, which the ManiSkill port never sets. Site
    positions are in the fixture body frame at scale 1; the port scales the fixture
    by `_scale` (MujocoObject.set_scale) and places it at `pos` with yaw `rot`.
    Returns None when any location is missing, so the caller can fall back.
    """
    xml = getattr(getattr(fixture, "loader", None), "xml", None)
    if xml is None:
        return None
    prefix = getattr(fixture, "naming_prefix", "") or ""
    scale = np.asarray(getattr(fixture, "_scale", 1.0), dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    pos = np.asarray(fixture.pos, dtype=np.float64).copy()
    yaw = float(getattr(fixture, "rot", 0.0) or 0.0)
    rot = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    by_name = {}
    for elem in xml.iter("site"):
        name = elem.get("name")
        if name:
            by_name[name] = elem
    out = []
    for loc in locations:
        # Explicit None checks, not `a or b`: a leaf ET.Element is falsy (no
        # children), so `or` would skip every place_site it found.
        elem = by_name.get(f"{prefix}burner_{loc}_place_site")
        if elem is None:
            elem = by_name.get(f"{prefix}burner_on_{loc}")
        if elem is None:
            return None
        local = np.fromstring(elem.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
        out.append(pos + rot @ (local * scale))
    return np.stack(out).astype(np.float32)


def stove_top_and_grid(fixture, n: int) -> tuple[float, np.ndarray]:
    """`(top_z, grid (n, 3))` from the stove's exterior bounding box.

    `fixture_frame` cannot be used for a stove: its `pos` is not the centre of its
    bounding box (kitchen 0: pos.z = 0.64, box z in [0.04, 0.92]), so
    `pos.z + size.z / 2` lands 16 cm above the cooktop. The exterior sites are the
    honest surface. The grid is the fallback when `burner_sites_world` finds no
    sites: n points on a 2-column lattice inset from the box edges.
    """
    ext = np.stack([np.asarray(p, dtype=np.float64) for p in fixture.get_ext_sites(relative=False)])
    lo = ext.min(axis=0)
    hi = ext.max(axis=0)
    top_z = float(hi[2])
    cols = 2
    rows = int(math.ceil(n / cols))
    xs = np.linspace(lo[0], hi[0], cols + 2)[1:-1]
    ys = np.linspace(lo[1], hi[1], rows + 2)[1:-1]
    pts = [np.array([x, y, top_z]) for y in ys for x in xs][:n]
    return top_z, np.stack(pts).astype(np.float32)


def find_stove(fixtures: dict, classes=("Stove", "Stovetop")) -> str:
    """Name of the kitchen's stove: the first fixture of a stove class, by name.

    Not `get_fixture(fixtures, "stove")`: a substring that matches several fixtures
    is resolved with `self.env._episode_rng.choice` (scene_builder.py:656), which
    both makes the pick random and shifts the shared episode stream — a fixture
    lookup must not consume the RNG the ticket is drawn from.
    """
    names = sorted(n for n, f in fixtures.items() if type(f).__name__ in classes)
    if not names:
        raise KeyError(
            f"no fixture of class {classes} in this kitchen. Not every RoboCasa layout "
            f"has a stove; pin one that does with scene_idx. Available: {sorted(fixtures)}"
        )
    return names[0]


def use_privileged_state(env) -> bool:
    """`obs_mode_struct.use_state`, the repo's gate for ground truth in observations."""
    return bool(env.obs_mode_struct.use_state)


# ------------------------------------------------------------------- task --


@register_env(
    "MikasaBurner-v0",
    max_episode_steps=BurnerConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class BurnerTask(BaseEnv):
    """Watch which burner is lit, drive to the stove, put the cup on that burner."""

    # ds_fetch is what the planner in this repo drives. "none" is absent on purpose:
    # it sets self.agent = None and everything below dereferences self.agent.
    SUPPORTED_ROBOTS = ["mikasa_ds_fetch", "fetch"]

    # "dense" is listed because compute_dense_reward is overridden below — the
    # convention check wants the two to agree. In v1 the dense reward IS the sparse
    # one; a shaped reward is future work and this says so rather than pretending.
    SUPPORTED_REWARD_MODES = ["sparse", "none", "dense", "normalized_dense"]

    cfg = BurnerConfig()

    cup: Actor
    marker: Actor

    def __init__(self, *args, robot_uids="mikasa_ds_fetch", scene_idx: int | None = 0, **kwargs):
        # scene_idx defaults to 0: the pinned kitchen of the design form (one wall,
        # industrial), where the stove is in view from the start zone. None asks
        # RoboCasa for a random kitchen per env — and not every kitchen has a
        # stove, in which case _load_scene raises with the fixture list. Set
        # before super().__init__, which runs a full reconfigure.
        self.scene_idx = scene_idx
        self.cfg.validate()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------ sim config --

    @property
    def _default_sim_config(self):
        # spacing=8 is what upstream's RoboCasa kitchen task uses: parallel
        # sub-scenes must not overlap and a kitchen is wider than the 5 m default.
        return SimConfig(spacing=8, control_freq=20)

    # ---------------------------------------------------------------- sensors --

    def _view_point(self, distance: float, height: float, slide: float = 0.0) -> np.ndarray:
        """A camera position `distance` out from the work line, on the robot's side.

        The side is read from the sign of `cfg.cup_across`, as season_dish does:
        `across` is a unit vector in the counter's yaw frame and "+across" is not
        reliably the room side. Both cameras look at the midpoint between the cup
        home and the stove so the recording shows the whole route; `slide` moves
        the eye along the counter, toward the stove, so the robot standing in front
        of the cup does not hide it (it did, in the first rendered frame).
        """
        side = math.copysign(1.0, self.cfg.cup_across) if self.cfg.cup_across else -1.0
        mid = 0.5 * (self._cup_home_np[0] + self._stove_centre_np[0])
        return (
            mid
            + self._across_np[0] * (side * distance)
            + self._along_np[0] * slide
            + np.array([0.0, 0.0, height], dtype=np.float32)
        )

    @property
    def _default_sensor_configs(self):
        # Read during _reconfigure, after _load_scene, before any episode — so it
        # may use geometry cached in _load_scene and nothing drawn per episode.
        target = 0.5 * (self._cup_home_np[0] + self._stove_centre_np[0])
        pose = sapien_utils.look_at(eye=self._view_point(1.6, 1.4), target=target)
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        """The clip's viewpoint. Decoration: no predicate reads a camera, and the
        sweeps that produced this task's numbers run `--obs-mode state` and render
        nothing at all.

        Re-aimed 2026-08-19 by rendering the same recorded episode (traj/burner-*)
        from a dozen eyes at four moments each. The old eye — 2.4 m out, 2.1 m up,
        slid 1.1 m — framed the whole room and left the cooktop small and half
        behind the robot's own body at the moment of placement, which is the one
        moment the clip exists to show: *which burner*. This one steps in and up
        (1.8 m out, 2.1 m up, slid 1.7 m toward the stove) and lifts the look-at
        point 0.35 m off the counter, so the cooktop fills the middle of the frame,
        the cue marker on it is legible at step 0, and the cup is still in frame at
        the far end. Slid further than 1.7 m the robot is clipped by the left edge
        during the cue phase — measured, not guessed.

        512 stays the default, and it is a *recording* default: `RecordEpisode`
        keeps every frame of an episode in host RAM until the flush, so at this
        repo's longest episode (931 steps) 512² is 0.73 GB and 2048² would be
        11.7 GB. Rendering one clip at a time is not that constrained — replay
        takes the size from `utils.mikasa.replay --render-size`.
        """
        target = 0.5 * (self._cup_home_np[0] + self._stove_centre_np[0]) + np.array(
            [0.0, 0.0, 0.35], dtype=np.float32
        )
        pose = sapien_utils.look_at(eye=self._view_point(1.8, 2.1, slide=1.7), target=target)
        return CameraConfig("render_camera", pose, 512, 512, 1.05, 0.01, 100)

    # ------------------------------------------------------------------- load --

    def _load_agent(self, options: dict):
        # Clear of the kitchen so gpu_init() is not resolving an interpenetration
        # on frame one; the scene builder overwrites this pose during build().
        super()._load_agent(options, parking_pose(self))

    def _load_scene(self, options: dict):
        """Kitchen, cup, marker, cached geometry. No `set_pose` here — the lint walks
        this method, and `scene._setup()` re-applies `initial_pose` right after."""
        super()._load_scene(options)

        self.scene_builder = RoboCasaSceneBuilder(self)
        if self.scene_idx is None:
            self.scene_builder.build()
        else:
            self.scene_builder.build([self.scene_idx] * self.num_envs)

        self._fix_ds_fetch_collision_bits()

        n = self.cfg.n_burners
        cup_homes, robot_starts, alongs, acrosses = [], [], [], []
        stove_centres, stove_docks, burner_pos = [], [], []
        used_fallback = False
        for i in range(self.num_envs):
            fixtures = self.scene_builder.scene_data[i]["fixtures"]
            counter = require_get_fixture(
                self.scene_builder, fixtures, self.cfg.counter_query, scene_idx=self.scene_idx
            )
            stove_name = find_stove(fixtures, self.cfg.stove_classes)
            stove = fixtures[stove_name]

            top, along, across = counter_frame(counter)
            home = (
                top
                + along * self.cfg.cup_along
                + across * self.cfg.cup_across
                + np.array([0.0, 0.0, self.cfg.cup_clearance], dtype=np.float32)
            )
            # Robot: the counter's own dock pose (front-facing standoff, yaw toward
            # the counter), slid along the counter to stand in front of the cup.
            dock_p, dock_yaw = dock_pose_for(self.scene_builder, fixtures, self.cfg.counter_query)
            start_p = dock_p + along * self.cfg.cup_along
            start = np.array(
                [start_p[0], start_p[1], 0.0, math.cos(dock_yaw / 2), 0.0, 0.0, math.sin(dock_yaw / 2)],
                dtype=np.float32,
            )

            top_z, grid = stove_top_and_grid(stove, n)
            sites = burner_sites_world(stove, BURNER_LOCATIONS[:n])
            if sites is None:
                used_fallback = True
                pos_i = grid
            else:
                # xy from the sites, z from the cooktop surface: the site z sits
                # ~1 cm below the exterior top in kitchen 0 and the cup rests on top.
                pos_i = sites.copy()
                pos_i[:, 2] = top_z
            centre = np.array(
                [pos_i[:, 0].mean(), pos_i[:, 1].mean(), top_z], dtype=np.float32
            )
            dock_s = dock_pose_for(
                self.scene_builder, fixtures, stove_name,
                offset=(0.0, self.cfg.stove_dock_toward),
            )

            cup_homes.append(home)
            robot_starts.append(start)
            alongs.append(along)
            acrosses.append(across)
            stove_centres.append(centre)
            stove_docks.append(np.array([*dock_s[0][:2], dock_s[1]], dtype=np.float32))
            burner_pos.append(pos_i)

        if used_fallback:
            logger.warning(
                "MikasaBurner-v0: stove has no burner_*_place_site in its MJCF; using a "
                "2x2 grid over its bounding box. Check the positions against the "
                "rendered cooktop before trusting a run."
            )
        self.burner_positions_from_sites = not used_fallback

        self._cup_home_np = np.stack(cup_homes).astype(np.float32)
        self._robot_start_np = np.stack(robot_starts).astype(np.float32)
        self._along_np = np.stack(alongs).astype(np.float32)
        self._across_np = np.stack(acrosses).astype(np.float32)
        self._stove_centre_np = np.stack(stove_centres).astype(np.float32)
        self._stove_dock_np = np.stack(stove_docks).astype(np.float32)
        self._burner_pos_np = np.stack(burner_pos).astype(np.float32)

        # Same category and instance as the inherited cup task (cup_2), at the
        # scale RoboCasa declares for it — robocasa_utils.load_objaverse_actor.
        self.cup = load_objaverse_actor(
            self, "cup", "cup", sapien.Pose(p=self._cup_home_np[0]), index=0
        )

        # One kinematic marker per env. Kinematic (the pose setter asserts a
        # non-static body under GPU sim, structs/actor.py:346) and without collision
        # (a collision shape would let the cup rest on the marker, and hide_visual
        # asserts none, :184). Parked at hidden_z; _apply_marker places it.
        self.marker = actors.build_sphere(
            self.scene,
            radius=self.cfg.marker_radius,
            color=list(self.cfg.marker_color),
            name="burner_marker",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.cfg.hidden_z]),
        )

        dev = self.device
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
        self.burner_pos = t(self._burner_pos_np)  # (num_envs, n, 3)
        self._cup_home = t(self._cup_home_np)
        self._robot_start = t(self._robot_start_np)

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
        # Task buffers, allocated once at full width. get_state_dict runs before
        # the first _initialize_episode (sapien_env.py:332 and RecordEpisode), so
        # they cannot be lazy.
        n, dev = self.num_envs, self.device
        self.cue_id = torch.zeros(n, dtype=torch.long, device=dev)
        self.cue_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.delay_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.wrong_burner_settled = torch.zeros(n, dtype=torch.bool, device=dev)
        self._last_eval_step = torch.full((n,), -1, dtype=torch.int32, device=dev)
        # -1: "the marker on screen may not match the schedule". _apply_marker
        # rewrites when the step it applied differs from the one asked for.
        self._marker_applied_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        return super()._after_reconfigure(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Restores fixtures; restores the robot only for uid == "fetch"
            # (scene_builder.py:566), hence _restore_robot right after.
            self.scene_builder.initialize(env_idx)
            self._restore_robot(env_idx)

            # The ticket — answer, schedule, jitter — from the *episode* RNG. torch's
            # generator is only seeded when a seed was passed (sapien_env.py:948-953),
            # so a torch-drawn answer is not reproducible from a seed.
            rng = self._batched_episode_rng[env_idx]
            cue_id = np.asarray(rng.randint(0, self.cfg.n_burners)).reshape(b)
            lo, hi = self.cfg.cue_steps_range
            cue_steps = np.asarray(rng.randint(lo, hi + 1)).reshape(b)
            lo, hi = self.cfg.delay_steps_range
            delay_steps = np.asarray(rng.randint(lo, hi + 1)).reshape(b)
            jitter = np.stack(
                [np.asarray(rng.uniform(-1.0, 1.0)).reshape(b) for _ in range(2)], axis=1
            )

            self.cue_id[env_idx] = torch.as_tensor(cue_id, dtype=torch.long)
            self.cue_steps[env_idx] = torch.as_tensor(cue_steps, dtype=torch.long)
            self.delay_steps[env_idx] = torch.as_tensor(delay_steps, dtype=torch.long)

            spawn = self._cup_home[env_idx].clone()
            spawn[:, :2] += torch.as_tensor(jitter, dtype=torch.float32) * self.cfg.cup_jitter
            self.cup.set_pose(Pose.create_from_pq(p=spawn))

            # Task state is not sim state — masked by hand.
            self.wrong_burner_settled[env_idx] = False
            self._last_eval_step[env_idx] = -1
            self._marker_applied_step[env_idx] = -1

            # t = 0: reset() zeroes elapsed_steps for env_idx before calling us
            # (sapien_env.py:849), so this shows the marker on burner cue_id.
            self._apply_marker(self.elapsed_steps)

    def _restore_robot(self, env_idx: torch.Tensor):
        """The rest keyframe and the start pose in front of the cup, for every robot.

        `scene_builder.initialize` restores the robot only when uid == "fetch"
        (scene_builder.py:563-577 compares a literal), and even then to *its* pose —
        `robot_poses[env_idx]`, a dock at a fixture it picked by `rng.choice`
        (scene_builder.py:528-551) — not to the task's `_robot_start`, which is the
        counter dock slid in front of the cup. season_dish.py can skip the "fetch" uid
        because it restores to `robot_poses` too; this task cannot. Measured on
        kitchen 0, seed 3, CPU: with the uid test in place a `fetch` run started at
        (3.868, -1.40) instead of the designed (1.92, -1.40) — 2.15 m from the cup,
        past the stove, and stage 2 of the oracle cannot reach. So no uid test: this
        runs after `scene_builder.initialize` and simply overwrites it, which is safe
        and idempotent. (The uid test in `_fix_ds_fetch_collision_bits` is a different
        matter and must stay — upstream really does apply that exemption for "fetch".)
        """
        if self.agent is None:
            return
        self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
        self.agent.robot.set_pose(Pose.create(self._robot_start[env_idx]))

    # ----------------------------------------------------------------- marker --

    def _apply_marker(self, step: torch.Tensor):
        """Put the marker where the schedule says it is at `step`, for every env.

        Level-triggered and idempotent: computed from `step` against the per-env
        schedule, written as one masked batched pose, skipped when the cache says
        this step was already applied. Called from `_before_control_step` with
        `elapsed + 1` and from `_initialize_episode` with `elapsed` (== 0).

        **Partial resets are handled but unverified on GPU.** `Actor.pose = ...`
        writes `cuda_rigid_body_data[..., scene._reset_mask[scene_idxs], :7] = value`
        (structs/actor.py:365-367), and `reset(options=dict(env_idx=[i]))` narrows
        `_reset_mask` to those envs for the whole of `_initialize_episode`
        (sapien_env.py:923-926, restored to all-ones at :953). A full-width `(b, 7)`
        write against a narrowed mask raises a shape mismatch, so the write and the
        cache update below are both masked. On the full-width path — every
        `_before_control_step` call, and CPU sim, where `num_envs == 1` — the mask is
        all-ones, `p[mask]` is `p` and the `torch.where` is `step.clone()`, so the
        behaviour is exactly what it was. Only GPU sim with `num_envs > 1` and
        `auto_reset` can take the narrowed path, and this task has never run on GPU.
        """
        step = step.to(torch.long)
        if bool((step == self._marker_applied_step).all()):
            return
        show = step < self.cue_steps
        rows = torch.arange(self.num_envs, device=self.device)
        lit = self.burner_pos[rows, self.cue_id].clone()
        lit[:, 2] += self.cfg.marker_height
        hidden = torch.tensor([0.0, 0.0, self.cfg.hidden_z], device=self.device)
        p = torch.where(show.unsqueeze(-1), lit, hidden.expand_as(lit))
        mask = self.scene._reset_mask
        self.marker.set_pose(Pose.create_from_pq(p=p[mask]))
        if self.gpu_sim_enabled:
            # UNVERIFIED — see the module docstring. Without it the write sits in
            # cuda_rigid_body_data and _gpu_fetch_all (:1031) overwrites it.
            self.scene.px.gpu_apply_rigid_dynamic_data()
        # Only the rows actually written. A blanket `step.clone()` would record a
        # repaint for envs the mask blocked, and _restore_task_state sets the whole
        # cache to -1 precisely so the next call repaints them.
        self._marker_applied_step = torch.where(mask, step, self._marker_applied_step)

    def _before_control_step(self):
        # elapsed_steps is still the previous step here (incremented at :957 after
        # _step_action returns); the observation this step produces carries
        # elapsed + 1, so that is the step the marker is scheduled for.
        self._apply_marker(self.elapsed_steps + 1)
        return super()._before_control_step()

    # --------------------------------------------------------------- evaluate --

    def evaluate(self) -> dict:
        """Pure except for the latch, and the latch is idempotent within a step.

        Runs every step, once inside reset() at t = 0, and out of band from
        `diagnose_task` / `state_diff`. Nothing here moves an actor. The latch only
        advances when `elapsed_steps` has moved since the last call, so a second
        call in the same step returns the same dict.
        """
        elapsed = self.elapsed_steps.to(torch.long)
        cue_phase, act_phase = phase_masks(elapsed, self.cue_steps, self.delay_steps)

        cup_p = self.cup.pose.p  # (n_envs, 3)
        d_xy = torch.linalg.norm(cup_p[:, None, :2] - self.burner_pos[:, :, :2], dim=-1)
        dz = cup_p[:, None, 2] - self.burner_pos[:, :, 2]
        lo, hi = self.cfg.place_z_band
        on_burner = (d_xy <= self.cfg.on_burner_radius) & (dz >= lo) & (dz <= hi)

        settled = self.cup.is_static(
            lin_thresh=self.cfg.settle_lin_vel, ang_thresh=self.cfg.settle_ang_vel
        )
        is_grasped = self.agent.is_grasping(self.cup)

        step = elapsed.to(torch.int32)
        advance = step != self._last_eval_step
        verdict = score_placement(
            on_burner, settled, is_grasped, act_phase, self.cue_id, self.wrong_burner_settled
        )
        self.wrong_burner_settled = torch.where(
            advance, verdict["wrong_burner_settled"], self.wrong_burner_settled
        )
        self._last_eval_step = torch.where(advance, step, self._last_eval_step)

        rows = torch.arange(self.num_envs, device=self.device)
        cup_to_cue_dist = d_xy[rows, self.cue_id]

        return {
            "success": verdict["success"],
            "fail": self.wrong_burner_settled,
            "cue_phase": cue_phase,
            "act_phase": act_phase,
            "elapsed": elapsed,
            "cue_id": self.cue_id,
            "on_burner_id": verdict["on_burner_id"],
            "placed": verdict["placed"],
            "is_grasped": is_grasped,
            "settled": settled,
            "wrong_burner_settled": self.wrong_burner_settled,
            "cup_to_cue_dist": cup_to_cue_dist,
        }

    # -------------------------------------------------------------------- obs --

    def _get_obs_extra(self, info: dict) -> dict:
        """What the policy sees. Key order fixed; no key is named for the target.

        Always: the phase one-hot (time, which a policy can count anyway),
        `marker_visible` (the state-space counterpart of the pixels — zeros after
        the cue, so it carries no answer), the TCP and the cup. Ground truth only
        behind `use_state`: the burner positions, and `cue_id` masked to -1 outside
        the cue phase — the cue is observable only during its phase in every mode
        (AGENTS.md, "Writing a memory task"). The observation space is frozen from
        the t = 0 dict (sapien_env.py:330), so nothing here is conditional on time.
        """
        cue = info["cue_phase"]
        act = info["act_phase"]
        obs = dict(
            phase=torch.stack([cue, ~cue & ~act, act], dim=1).to(torch.float32),
            marker_visible=cue.to(torch.float32).unsqueeze(-1),
            tcp_pose=self.agent.tcp_pose.raw_pose,
            cup_pose=self.cup.pose.raw_pose,
        )
        if use_privileged_state(self):
            obs.update(
                burner_positions=self.burner_pos.reshape(self.num_envs, -1),
                cue_id=torch.where(cue, self.cue_id, torch.full_like(self.cue_id, -1)),
            )
        return obs

    def get_language_instruction(self, **kwargs):
        return [INSTRUCTION] * self.num_envs

    # ----------------------------------------------------------------- reward --

    def compute_dense_reward(self, obs, action, info: dict):
        # v1: the sparse signal. Named "dense" only so the mode is honest about
        # existing; shaping (distance to the remembered burner) is future work.
        return info["success"].to(torch.float32)

    def compute_normalized_dense_reward(self, obs, action, info: dict):
        return self.compute_dense_reward(obs, action, info) / 1.0

    # ------------------------------------------------------------------ state --

    def _task_state(self) -> dict:
        """The task's memory, keyed exactly as TASK_STATE_KEYS."""
        return {
            "cue_id": self.cue_id.clone(),
            "cue_steps": self.cue_steps.clone(),
            "delay_steps": self.delay_steps.clone(),
            "wrong_burner_settled": self.wrong_burner_settled.clone(),
            "last_eval_step": self._last_eval_step.clone(),
        }

    def _restore_task_state(self, state: dict) -> None:
        """Tolerates a dict with no task keys: BaseEnv.set_state(flat) rebuilds one
        holding only "actors" and "articulations" (sapien_env.py:1309-1327)."""
        attrs = {
            "cue_id": "cue_id",
            "cue_steps": "cue_steps",
            "delay_steps": "delay_steps",
            "wrong_burner_settled": "wrong_burner_settled",
            "last_eval_step": "_last_eval_step",
        }
        for key, attr in attrs.items():
            if key in state:
                setattr(self, attr, restore_task_tensor(getattr(self, attr), state[key], self.device))
        # The marker's own pose round-trips with the sim state, but the schedule
        # it was painted from may not match the restored one (the memory-free
        # control overwrites cue_id). Invalidate: the next _before_control_step
        # repaints from the restored buffers.
        self._marker_applied_step[:] = -1

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        state.update(self._task_state())
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        # super() first: sim state lands, then the task's flags on a consistent scene.
        super().set_state_dict(state, env_idx)
        self._restore_task_state(state)
