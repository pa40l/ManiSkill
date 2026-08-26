---
name: maniskill-remote-planner-analysis
description: Runs the ManiSkill planner benchmark remotely on gangway (ssh) with parallel workers and pulls the logs back. Read maniskill-planner-analysis, but use this skill's script for the remote computation.
---

Read **maniskill-planner-analysis**, but run the benchmark with this script instead of its local one.

**Where the script is** — do not hunt for it. Resolve it once (works from any directory inside this repo/worktree):

```bash
SCRIPT=$(find "$(git rev-parse --show-toplevel)/skills" -name remote_analyze.sh | head -1)
```

Then run the benchmark with it:

```bash
bash "$SCRIPT" {planner} {num_runs} {workers}
```

Rendering is by far the biggest CPU cost; for large batches (10+ workers) run video-less:

```bash
NO_VIDEO=1 bash "$SCRIPT" {planner} {num_runs} {workers}
```

## Re-record failed seeds with video

After the batch, re-run the **failed** seeds **locally with video** in the background (only the few failed seeds, so the render cost is fine) and keep their mp4s for the final analysis step:

```bash
uv run python -m planners.{planner} --seed {failed_seed} &
```

Wait for them at the end; the video-correlation step of the analysis (per `maniskill-logs-analysis`) uses these mp4s to visually confirm the anomalies found in the event/CSV logs.
