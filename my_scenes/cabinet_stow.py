"""MikasaCabinetStow-v0: the cup starts on the COUNTER and must end on the shelf.

The reverse of `MikasaCabinetRetrieval-v0`, and built for a reason rather than for
symmetry: `MikasaDepthRecall-v0`'s rewrite has to put props BACK into the cabinet,
and no task in this package had ever measured that stroke. Owner, 2026-09-09:
«сперва сделаем для упрощения тестов задачу обратную retrieval» — this is that
task, the substrate the memory variant's restore leg is debugged on.

Motor-only, like v0: there is nothing to remember. Its whole content is the
put-away stroke — a loaded reach through the cabinet mouth, which is the half of
the family that the 200/200 straight flow never exercised (there the hand enters
the cabinet EMPTY and leaves it loaded; here it is the other way round).

The task machinery is `cabinet_retrieval_base.py`, shared with v0 and the
closed-door variant. The variant is one config object: `direction="in"` swaps
which of the two measured points is the spawn and which the target, and the
height floor stops a cup left on the counter under the cabinet from being
credited as stowed. Nothing geometric is new — the shelf point is v0's spawn band
and the counter point is v0's place target, unchanged.
"""
from mani_skill.utils.registration import register_env

from utils.mikasa.scenes.cabinet_retrieval_base import CabinetStowConfig, CabinetRetrievalTaskBase


@register_env(
    "MikasaCabinetStow-v0",
    max_episode_steps=CabinetStowConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class CabinetStowTask(CabinetRetrievalTaskBase):
    """Take the cup off the counter and stand it on the open cabinet's shelf.

    Everything — scene, door, evaluate(), obs, state keys — is inherited; the
    variant is one config object. The oracle is `cabinet_stow_planner.solve`,
    which is the straight flow run backwards, not a re-parameterisation of it:
    the torso rises with the cup in hand, the loaded drive is what the ready
    posture's pre-check has to clear, and the retreat out of the cabinet comes
    before any torso motion.
    """

    cfg = CabinetStowConfig()
