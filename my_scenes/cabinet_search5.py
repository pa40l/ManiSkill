"""MikasaCabinetSearch5-v0: the same search over FIVE compartments, one of which
marks itself.

The variant the owner asked for on 2026-09-02: `cab_1`, the west wall unit, added
to the four the v0 task ships, with its own closing rule — "just drive the door to
the wall" (`close_policy="wall_park"`). Everything else is the v0 territory
untouched: the same scene machinery (`cabinet_search_base.py`), the same latches,
the same thresholds, the same home spot, the same instructions. The variant is one
config object.

Why cab_1 needs a rule of its own (W24, docs/lab-journal.md): the push side of an
opened left leaf lies WEST of its hinge, i.e. inside the room's west wall at x=0,
so the family's fist push has nowhere to stand. The wall itself is the stop, and it
is a real one once the door is untangled from the robot's collision-group exemption
(`CabinetSearchConfig.untangle_stems`, `_untangle_fixture_door`).

WHAT IT COSTS, stated here because a variant that quietly moved the floor would be
worse than no variant: a leaf parked at the wall is a mark that SURVIVES the round.
cab_1 visited does not look like cab_1 untouched, so a memoryless agent never needs
to re-try it — while the other four are still erased behind the agent and still
indistinguishable. The floor is therefore NOT the plain 1/N formula but
`self_marking_search_floor(5)` = 272083/500000 = **0.5442** of the motor rate,
against the plain `memoryless_search_floor(5)` = 0.5021 and the four-compartment
task's 0.5547. The fifth door buys less than a fifth door should; the gap is what
the wall park costs, and it is the reason `MikasaCabinetSearch-v0` still ships four.
"""
from dataclasses import dataclass

from mani_skill.utils.registration import register_env

from utils.mikasa.scenes.cabinet_search_base import (
    CAB_1_STEM,
    COMPARTMENTS_WITH_CAB_1,
    CabinetSearchConfig,
    CabinetSearchTaskBase,
)


@dataclass
class CabinetSearch5Config(CabinetSearchConfig):
    """The five-compartment variant's numbers. Three fields move — the compartment
    set, the one collision mask it unties for them, and the horizon; every
    threshold, dock and band is the measured v0 territory, inherited untouched."""

    compartments: tuple = COMPARTMENTS_WITH_CAB_1
    """cab_1 FIRST, then the four v0 compartments — west to east, spawn centres
    0.50/1.00/1.50/2.00/2.50. The index therefore means something different here
    than in the v0 task (index 0 is cab_1); see `COMPARTMENTS_WITH_CAB_1`."""

    untangle_stems: frozenset = frozenset({CAB_1_STEM})
    """cab_1's door, and NOTHING else — the one collision mask this variant
    touches. v0 unties nothing (`CabinetSearchConfig.untangle_stems` is empty),
    and the other four compartments here are the v0 fixtures under the v0 masks,
    so every number measured on them carries over unchanged. Without this the
    fifth compartment's bar is a ghost and `validate()` refuses to build the
    task at all (W24; `UNTANGLE_BITS`, `_untangle_fixture_door`)."""

    horizon: int = 13800
    """PROVISIONAL, arithmetic over the SAME measured round costs the v0 horizon
    is derived from — nothing has driven a five-round episode, and this number
    expires on the first N=5 sweep.

    K113 (2026-09-02, container CPU, calibration seeds 11-22, before the look
    leg) measured, at N=3: a one-round find 593-676, a two-round find 2393-2641,
    the three-round worst 4284 — so an EMPTY round costs ~1825 — and the look leg
    that landed afterwards costs ~250 a round. The v0 horizon takes the measured
    three-round worst and adds one more empty round plus a look leg per round:

        4284 + 1825 + 4 x 250 = 7110;  x 1.5 (K22) = 10665 -> 10700   (N=4)

    N=5's worst path is FIVE rounds, so it adds TWO empty rounds instead of one:

        4284 + 2 x 1825 + 5 x 250 = 9184;  x 1.5 = 13776 -> 13800     (N=5)

    That charges cab_1's round the price of a full push round, which is a cushion
    rather than an estimate: its round SKIPS the closing stage entirely (the
    ladder drive plus CLOSE_MAX_STEPS = 500) and pays instead one extra segment
    of arc pull, 1.75 -> 2.0 rad, about 21 steps at the K107 rate of ~144 steps
    per 1.75 rad. Must equal max_episode_steps in the decorator."""


@register_env(
    "MikasaCabinetSearch5-v0",
    max_episode_steps=CabinetSearch5Config.horizon,
    asset_download_ids=["RoboCasa"],
)
class CabinetSearch5Task(CabinetSearchTaskBase):
    """Find the hidden cube across five compartments, one of which cannot be shut.

    Everything — scene, spawn, evaluate(), obs, state keys — is inherited from
    `CabinetSearchTaskBase`; the variant is one config object. The latch step
    reads `close_policy` off the compartments, so the four `push` compartments
    behave exactly as they do in v0 and only cab_1's round ends differently. The
    oracle grows one branch, gated on the same field
    (`cabinet_search_planner.solve`), so every other compartment's round is
    byte-identical to v0's.
    """

    cfg = CabinetSearch5Config()
