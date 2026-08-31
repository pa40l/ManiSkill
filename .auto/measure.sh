#!/usr/bin/env bash
set -euo pipefail
rm -rf logs/*
NO_VIDEO=1 bash skills/maniskill-remote-planner-analysis/scripts/remote_analyze.sh myrobocasa_takeitback_planner 100 10 >/tmp/takeitback-autoresearch.out 2>&1
uv run python - <<'PY'
import datetime
import glob
import json
import statistics

success = 0
failures = {}
planner_times = []
pregrasp_times = []
for path in glob.glob('logs/*/*_events.jsonl'):
    events = [json.loads(line) for line in open(path) if line.strip()]
    result = next((event for event in reversed(events) if event.get('event') == 'result'), None)
    if result and result.get('success') is True:
        success += 1
    elif result:
        message = result.get('message', 'unknown')
        failures[message] = failures.get(message, 0) + 1
    if not result or not events:
        continue
    start = datetime.datetime.fromisoformat(events[0]['time'])
    end = datetime.datetime.fromisoformat(result['time'])
    planner_times.append((end - start).total_seconds())
    phases = {
        event['message']: event
        for event in events
        if event.get('event') == 'phase'
    }
    grasp = phases.get('Stage 3: grasp')
    if grasp:
        grasp_time = datetime.datetime.fromisoformat(grasp['time'])
        pregrasp_times.append((grasp_time - start).total_seconds())

print(f'METRIC planner_median_s={statistics.median(planner_times):.6f}')
print(f'METRIC pregrasp_median_s={statistics.median(pregrasp_times):.6f}')
print(f'METRIC success_count={success}')
print(f'METRIC observed_runs={len(glob.glob("logs/*/*_events.jsonl"))}')
for category, count in sorted(failures.items()):
    print(f'METRIC failure_{category.replace(" ", "_")}={count}')
PY
