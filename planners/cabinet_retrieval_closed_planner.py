"""Oracle for MikasaCabinetRetrievalClosed-v0 — the cabinet_retrieval algorithm.

One planner file per task is the convention, and the closed-door variant's
algorithm IS `cabinet_retrieval_planner.solve`: it reads `cfg.door_open_rad` and
runs the opening stage (`open_the_door` — the K104 arc pull) exactly when the
door starts below the passable angle. This module is the variant's address for
`evaluate_planner -p cabinet_retrieval_closed_planner`; the code lives next door
on purpose (a planner import registers no envs, unlike a scene import, so the
scene files' no-cross-import rule does not apply here).
"""
from planners.cabinet_retrieval_planner import solve  # noqa: F401
