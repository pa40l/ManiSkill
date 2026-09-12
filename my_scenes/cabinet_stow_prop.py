"""MikasaCabinetStowProp-v0: the stow chore carrying DEPTHRECALL'S PROP, not the cup.

Owner, 2026-09-10: «пробуй заменить кружку на этот предмет. Если получается с таким
же SR, то делай задачу полностью». This is that swap, registered rather than run as
a throwaway probe, because the answer it gives is what the DepthRecall rewrite rests
on and a number nobody can re-run is not a number.

Why it cannot be inferred from `MikasaCabinetStow-v0`'s 100/100: the two objects
differ in every property a gripper cares about.

| | cup (v0's object) | prop (this one) |
|---|---|---|
| width across | 7.44 cm | 4.5 cm |
| mass | ~8 g | ~243 g (a 4.5x4.5x12 cm box at default density) |
| wall | round, tapering | flat faces, square section |
| pads' seat below the top | 3.75 cm | 4.0 cm |

The narrower body leaves more clearance inside the cabinet mouth, which should help;
the mass is thirty times the cup's, which changes what a graze does rather than
whether one happens. Neither prediction is worth trusting, hence the sweep.

Everything else is `MikasaCabinetStow-v0`: same direction, same two measured points,
same oracle (`cabinet_stow_planner`, re-exported), same height floor.
"""
from dataclasses import dataclass

from mani_skill.utils.registration import register_env

from utils.mikasa.scenes.cabinet_retrieval_base import CabinetRetrievalTaskBase, CabinetStowConfig


@dataclass
class CabinetStowPropConfig(CabinetStowConfig):
    """The stow variant's numbers with the row's prop in place of the cup."""

    object_kind: str = "box"
    "A plain coloured box, built exactly as `depth_recall.py` builds its row."

    object_name: str = "prop"
    """The ACTOR's name, and therefore the needle `oracle_common.touchable` matches
    in the planning world. Not cosmetic: the oracle asks the task for it
    (`cabinet_stow_planner._needle`), and a box still called "cup" would work by
    accident today and break the first time two objects are in the scene."""

    prop_half_xy: float = 0.0225
    prop_half_h: float = 0.06
    "4.5 x 4.5 x 12 cm — `DepthRecallConfig`'s prop, unchanged."

    horizon: int = 1000
    """MEASURED 2026-09-10 by the K22 rule against this variant's own oracle, on
    calibration seeds 11-22 and never the eval seeds 0-9: 12/12 in 574-609 steps, no
    cue phase, so 609 x 1.5 = 913.5, rounded up to a multiple of 50 and then of 100.

    Longer than the cup variant's 900 because the prop's episodes are ~15 steps
    longer throughout (the corrective screw at the entry runs on every episode with a
    payload this heavy, where the cup usually skips it). The 100-seed population
    (1300-1399) then ran 569-617 steps with 0 truncations."""


@register_env(
    "MikasaCabinetStowProp-v0",
    max_episode_steps=CabinetStowPropConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class CabinetStowPropTask(CabinetRetrievalTaskBase):
    """Take the prop off the counter and stand it on the open cabinet's shelf."""

    cfg = CabinetStowPropConfig()
