"""MikasaCabinetRetrievalClosed-v0: the door starts CLOSED; the robot opens it.

Hook 1 of the design blank (docs/task-designs/F-cabinetretrieval-v0.md), built
once K103/K104 measured the arc pull opening the door to the full 1.6 rad. The
task machinery is `cabinet_retrieval_base.py`, shared with v0; the variant is one
config object — door closed, its own horizon — and the oracle grows the opening
stage (`cabinet_retrieval_planner.open_the_door`, gated on the config).
"""
from dataclasses import dataclass

from mani_skill.utils.registration import register_env

from utils.mikasa.scenes.cabinet_retrieval_base import CabinetRetrievalConfig, CabinetRetrievalTaskBase


@dataclass
class CabinetRetrievalClosedConfig(CabinetRetrievalConfig):
    """The closed-door variant's numbers: hook 1 of the design blank, built once
    K104 measured the arc pull opening the door to the full 1.6 rad. Only two
    fields move; everything else — spawn band, dock, place target, thresholds —
    is the measured v0 territory, untouched."""

    horizon: int = 3900
    """K22 against the closing oracle WITH the fast pull (PULL_V_HANDLE=0.20,
    K107/K108): calibration successes (seeds 12/18/21 — never the eval seeds
    0-9) ran 2544/2016/2050 steps; 2544 x 1.5 = 3816 -> 3900. Supersedes the
    4400 measured at the v=0.05 pull (calibration 2354/2919/2536, K106), the
    2700 of the open-and-retrieve oracle (K105), and the provisional 3600."""

    door_open_rad: float = 0.0
    "CLOSED. The oracle opens the door itself — that is the variant."

    require_door_closed: bool = True
    """The full cycle: open the door, take the cup, stand it on the counter,
    close the door behind you. A tightening on top of the K105-measured task
    (the 5/10 published for this variant predates the requirement and gets
    re-measured under it); v0 keeps the field False and its verdict
    byte-identical."""


@register_env(
    "MikasaCabinetRetrievalClosed-v0",
    max_episode_steps=CabinetRetrievalClosedConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class CabinetRetrievalClosedTask(CabinetRetrievalTaskBase):
    """The same retrieval, but the door starts closed and the robot opens it.

    Everything — scene, spawn, evaluate(), obs, state keys — is inherited; the
    variant is one config object. The oracle grows the opening stage (handle
    grasp + `pull_hinge_arc`) gated on `cfg.door_open_rad < 0.9`, so the v0
    oracle path is byte-identical when the door starts open.
    """

    cfg = CabinetRetrievalClosedConfig()
