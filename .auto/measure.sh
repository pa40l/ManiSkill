#!/usr/bin/env bash
set -euo pipefail
rm -rf logs/*
NO_VIDEO=1 bash skills/maniskill-remote-planner-analysis/scripts/remote_analyze.sh myrobocasa_takeitback_planner 50 20 >/tmp/takeitback-autoresearch.out 2>&1
uv run python - <<'PY'
import glob, json
success = 0
failures = {}
for path in glob.glob('logs/*/*_events.jsonl'):
    events = [json.loads(line) for line in open(path) if line.strip()]
    result = next((event for event in reversed(events) if event.get('event') == 'result'), None)
    if result and result.get('success') is True:
        success += 1
    elif result:
        message = result.get('message', 'unknown')
        failures[message] = failures.get(message, 0) + 1
print(f'METRIC success_count={success}')
print(f'METRIC observed_runs={len(glob.glob("logs/*/*_events.jsonl"))}')
for category, count in sorted(failures.items()):
    print(f'METRIC failure_{category.replace(" ", "_")}={count}')
PY
