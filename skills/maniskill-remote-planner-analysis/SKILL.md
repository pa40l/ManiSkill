---
name: maniskill-remote-planner-analysis
description: Runs the ManiSkill planner benchmark remotely on gangway (ssh) with parallel workers and pulls the logs back. Read maniskill-planner-analysis, but use this skill's script for the remote computation.
---

Read **maniskill-planner-analysis**, but run the benchmark with this script instead of its local one.

**Where the script is** — do not hunt for it:
- The repository has a root-level alias (a tracked symlink): `scripts/remote_analyze.sh`. Run it from the repo root / worktree root.
- If the alias is missing, resolve it from this skill's `location` in the available-skills list: `<skill-location>/scripts/remote_analyze.sh`.

```bash
scripts/remote_analyze.sh {planner} {num_runs} {workers}
```

Rendering is by far the biggest CPU cost; for large batches (10+ workers) run video-less:

```bash
NO_VIDEO=1 scripts/remote_analyze.sh {planner} {num_runs} {workers}
```

## Re-record failed seeds with video

After the batch, re-run the **failed** seeds **locally with video** in the background (only the few failed seeds, so the render cost is fine) and keep their mp4s for the final analysis step:

```bash
uv run python -m planners.{planner} --seed {failed_seed} &
```

Wait for them at the end; the video-correlation step of the analysis (per `maniskill-logs-analysis`) uses these mp4s to visually confirm the anomalies found in the event/CSV logs.
