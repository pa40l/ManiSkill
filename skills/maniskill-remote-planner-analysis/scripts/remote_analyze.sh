#!/usr/bin/env bash
# Run a ManiSkill planner across N seeds on a remote host (gangway) with W
# parallel workers, pull the run logs back, and list the failed runs.
#
# The run happens in an isolated git worktree on the remote so it never
# disturbs the remote's main checkout; everyone else's work (and the venv at
# $REMOTE_REPO/.venv) is left untouched.
#
# Usage: remote_analyze.sh <planner_module> <seed_count> <workers> [remote_host] [remote_repo]
#   planner_module  Python module under planners/, e.g. myrobocasa_fridge_veggies_planner
#   seed_count      number of seeds to run (1..N)
#   workers         how many seeds to run in parallel (1..N)
#   remote_host     ssh alias (default: gangway)
#   remote_repo     remote clone that holds the working .venv (default: ~/ManiSkill)
#
# Env: NO_VIDEO=1 appends --no-video to every remote run (skips render, ~5-10x
# faster). Use it for any batch with workers >= 10.
set -uo pipefail

PLANNER_NAME=${1:?usage: remote_analyze.sh <planner> <seed_count> <workers> [remote_host] [remote_repo]}
SEED_COUNT=${2:?usage: remote_analyze.sh <planner> <seed_count> <workers> [remote_host] [remote_repo]}
WORKERS=${3:?usage: remote_analyze.sh <planner> <seed_count> <workers> [remote_host] [remote_repo]}
REMOTE_HOST=${4:-gangway}
# Quoted default keeps the literal ~ so the REMOTE shell expands it; unquoted it
# would expand to the LOCAL home and the remote git/venv paths would not exist.
REMOTE_REPO=${5:-'~/ManiSkill'}

# Be forgiving if a caller passes <seed_count> <planner> <workers>.
if [[ ${1} =~ ^[0-9]+$ ]]; then
  SEED_COUNT=$1
  PLANNER_NAME=$2
fi

JOB="gw-$(date +%Y%m%d_%H%M%S)"
RUNDIR="~/gw-runs/${JOB}"
VENV_PY="${REMOTE_REPO}/.venv/bin/python"
# NO_VIDEO=1 appends --no-video to every remote planner run (no render, much faster).
PLANNER_ARGS=""
if [[ ${NO_VIDEO:-0} == 1 ]]; then
  PLANNER_ARGS="--no-video"
  echo "==> video recording disabled (--no-video)"
fi

echo "==> sync working tree -> ${REMOTE_HOST}:${RUNDIR} (workers=${WORKERS})"
THRESHOLD_START=$(date +%s)

# Isolated remote run dir: a git worktree of the remote clone (shares its .git).
ssh "${REMOTE_HOST}" "git -C ${REMOTE_REPO} worktree add --detach ${RUNDIR} HEAD >/dev/null 2>&1 || (rm -rf ${RUNDIR} && mkdir -p ${RUNDIR})" ||
  {
    echo "!! could not create remote worktree"
    exit 1
  }

# Push the whole local working tree (incl. uncommitted edits) into the run dir.
rsync -rlt --no-perms --no-owner --no-group \
  --exclude '.venv' --exclude 'logs' --exclude '.git' --exclude '.pi' --exclude '__pycache__' \
  ./ "${REMOTE_HOST}:${RUNDIR}/" >/dev/null 2>&1 ||
  {
    echo "!! rsync to remote failed"
    exit 1
  }

# Run seeds in parallel with ONE ssh connection: the parallelism is spawned
# INSIDE the remote host (xargs -P on gangway). N concurrent ssh tunnels through
# the bastion jump host get dropped with "Connection closed by UNKNOWN port
# 65535" (roughly half the workers fail), while a single tunnel survives
# arbitrary worker counts. Idle-marker options keep the one tunnel alive
# through long runs.
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=12 "$REMOTE_HOST" \
  "cd ${RUNDIR} && seq 1 ${SEED_COUNT} | xargs -P ${WORKERS} -I{} env PYTHONPATH=${RUNDIR} MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 ${VENV_PY} -m planners.${PLANNER_NAME} --seed {} ${PLANNER_ARGS} >/dev/null 2>&1"
# xargs exits 123 if any seed crashed; that is not fatal (failures are judged
# by the events below), so ignore the exit status.

# Alternative (only if one-ssh-per-worker is ever needed again): ssh ControlMaster
# multiplexing in ~/.ssh/config for Host gangway, so parallel sessions share one
# bastion tunnel instead of each opening its own.

echo "==> pull logs back -> ./logs/"
rsync -rlt --no-perms --no-owner --no-group "${REMOTE_HOST}:${RUNDIR}/logs/" ./logs/ >/dev/null 2>&1

echo "==> failed runs (result event success=false) created since start:"
# rsync -t preserves the remote mtimes, so the same newermt filter as the
# local analyze.sh selects exactly the runs this invocation produced.
find ./logs -maxdepth 2 -type f -name '*_events.jsonl' -newermt "@${THRESHOLD_START}" 2>/dev/null |
  while IFS= read -r f; do
    if grep -q '"success": *false' "$f"; then
      echo "$(dirname "$f")"
    fi
  done | sort -u

# Clean up the remote run dir (logs already pulled).
ssh "${REMOTE_HOST}" "git -C ${REMOTE_REPO} worktree remove --force ${RUNDIR} >/dev/null 2>&1; rm -rf ${RUNDIR}" >/dev/null 2>&1
echo "==> done"
