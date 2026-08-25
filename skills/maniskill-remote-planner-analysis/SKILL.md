---
name: maniskill-remote-planner-analysis
description: Runs the ManiSkill planner benchmark remotely on gangway (ssh) with parallel workers and pulls the logs back. Read maniskill-planner-analysis, but use this skill's script for the remote computation.
---

Read **maniskill-planner-analysis**, but run the benchmark with this script instead of its local one:

```bash
scripts/remote_analyze.sh {planner} {num_runs} {workers}
```
