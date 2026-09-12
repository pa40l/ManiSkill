"""Oracle for MikasaCabinetSearch5-v0 — the cabinet_search algorithm.

One planner file per task is the convention, and the five-compartment variant's
algorithm IS `cabinet_search_planner.solve`: it walks `cfg.compartments`, looks
each round's leaf up in `DOOR_SPECS`, and branches on the compartment's own
`close_policy` — so cab_1's round ends at the wall and the other four end shut,
with no line that names this env. This module is the variant's address for
`evaluate_planner -p cabinet_search5_planner`; the code lives next door on
purpose (a planner import registers no envs, unlike a scene import, so the scene
files' no-cross-import rule does not apply here — the
`cabinet_retrieval_closed_planner` precedent).
"""
from planners.cabinet_search_planner import solve  # noqa: F401
