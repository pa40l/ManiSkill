"""Oracle for MikasaCabinetStowProp-v0 — the same algorithm as the cup variant.

One planner file per task is the convention, and this variant's algorithm IS
`cabinet_stow_planner.solve`: nothing in the flow names the object, and what could
(the `touchable` needle) is read from `task.cfg.object_name`. This module is the
variant's address for `evaluate_planner -p cabinet_stow_prop_planner`; the code
lives next door on purpose, the way `cabinet_retrieval_closed_planner` does.
"""
from planners.cabinet_stow_planner import solve  # noqa: F401
