#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
cd "$REPO_ROOT"
SCRIPT_PATH="$(readlink -f "$0")"
EXP_ID="012_voxtell_category_specialists_replay50_cont100"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
RUN1_GROUP_NAME="${RUN1_GROUP_NAME:-exp012_category_specialists_20260801T215219Z}"
RUN1_GROUP_DIR="$EXP_DIR/runs/$RUN1_GROUP_NAME"
QUEUE_DIR="$EXP_DIR/queue"
STATE_JSON="$QUEUE_DIR/r02_queue_state.json"
STATE_MD="$QUEUE_DIR/r02_queue_state.md"
QUEUE_LOG="$QUEUE_DIR/r02_queue.log"
MIN_FREE_GPU_GIB="${MIN_FREE_GPU_GIB:-32}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DETACH="${DETACH:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%s\n' waiting_run1 waiting_gpus launching running
  printf 'run1=%s min_free_gpu_gib=%s profile=r02_2d_diffuse_replay_ratio\n' \
    "$RUN1_GROUP_NAME" "$MIN_FREE_GPU_GIB"
  exit 0
fi

[[ "${QUEUE_RUN2:-0}" == "1" ]] || {
  echo "QUEUE_RUN2=1 is required to arm Exp012 Run 2" >&2
  exit 2
}

mkdir -p "$QUEUE_DIR"

write_state() {
  local state="$1" detail="$2" run_group="${3:-}"
  python - "$STATE_JSON" "$STATE_MD" "$state" "$detail" "$run_group" \
    "$RUN1_GROUP_NAME" "$MIN_FREE_GPU_GIB" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

json_path, md_path = map(Path, sys.argv[1:3])
state, detail, run_group, run1, minimum = sys.argv[3:8]
payload = {
    "schema_version": 1,
    "experiment": "012_voxtell_category_specialists_replay50_cont100",
    "profile": "r02_2d_diffuse_replay_ratio",
    "state": state,
    "detail": detail,
    "run1_group": run1,
    "run2_group": run_group or None,
    "minimum_free_gpu_gib": float(minimum),
    "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
markdown = "\n".join([
    "# Exp012 Run-2 Queue",
    "",
    f"- State: `{state}`",
    f"- Detail: {detail}",
    f"- Run 1: `{run1}`",
    f"- Run 2: `{run_group or 'not allocated'}`",
    f"- Minimum free memory per GPU: `{minimum} GiB`",
    f"- Updated: `{payload['updated_at_utc']}`",
    "",
])
for path, text in (
    (json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n"),
    (md_path, markdown),
):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)
PY
}

run1_failed() {
  [[ -f "$RUN1_GROUP_DIR/.orchestrator_failed" ]] && return 0
  local report="$RUN1_GROUP_DIR/reports/exp012_progress.json"
  [[ -f "$report" ]] || return 1
  python - "$report" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "failed" else 1)
PY
}

run1_container_active() {
  local names
  names="$(docker ps --format '{{.Names}}')"
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    if [[ "$name" == rex012_* && "$name" != rex012_r02_* ]]; then
      return 0
    fi
  done <<<"$names"
  return 1
}

gpus_ready() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
    python -c 'import sys
minimum = float(sys.argv[1]) * 1024
values = [float(line.strip()) for line in sys.stdin if line.strip()]
print("gpu_free_mib=" + ",".join(f"{value:.0f}" for value in values))
raise SystemExit(0 if len(values) == 4 and all(value >= minimum for value in values) else 1)' \
    "$MIN_FREE_GPU_GIB"
}

run_queue() {
  exec 9>"$QUEUE_DIR/.r02_queue.lock"
  flock -n 9 || { echo "Exp012 Run-2 queue already owns the lock" >&2; return 3; }
  while [[ ! -f "$RUN1_GROUP_DIR/.selection_ready" ]]; do
    if run1_failed; then
      write_state failed "Run 1 failed; Run 2 was not launched"
      return 1
    fi
    write_state waiting_run1 "Waiting for the exact Run-1 selection_ready marker"
    sleep "$POLL_SECONDS"
  done

  while true; do
    if run1_container_active; then
      write_state waiting_run1 "Run 1 is selection_ready but a Run-1 container remains"
      sleep "$POLL_SECONDS"
      continue
    fi
    if ! gpu_detail="$(gpus_ready 2>&1)"; then
      write_state waiting_gpus "All four GPUs must have at least ${MIN_FREE_GPU_GIB} GiB free; ${gpu_detail}"
      sleep "$POLL_SECONDS"
      continue
    fi
    break
  done

  local timestamp run_group group_dir orchestrator_log orchestrator_pid status
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  run_group="exp012_category_specialists_r02_replay_ratio_$timestamp"
  group_dir="$EXP_DIR/runs/$run_group"
  orchestrator_log="$EXP_DIR/logs/${run_group}_orchestrator.log"
  mkdir -p "$group_dir" "$EXP_DIR/logs"
  write_state launching "Run-1 and GPU gates passed; starting resumable Run 2" "$run_group"
  START_TRAINING=1 DETACH=0 RUN1_GROUP_DIR="$RUN1_GROUP_DIR" \
    RUN_GROUP="$run_group" GROUP_DIR="$group_dir" ORCHESTRATOR_LOG="$orchestrator_log" \
    bash "$REPO_ROOT/scripts/rexgroundingct/run_012_r02_host_orchestrator.sh" \
    >"$orchestrator_log" 2>&1 &
  orchestrator_pid="$!"
  printf '%s\n' "$orchestrator_pid" >"$QUEUE_DIR/r02_orchestrator.pid"
  write_state running "Run 2 orchestrator is active (PID $orchestrator_pid)" "$run_group"
  status=0
  wait "$orchestrator_pid" || status="$?"
  if ((status != 0)); then
    write_state failed "Run 2 orchestrator exited with status $status" "$run_group"
    return "$status"
  fi
  [[ -f "$group_dir/.selection_ready" ]] || {
    write_state failed "Run 2 exited without selection_ready" "$run_group"
    return 1
  }
  write_state running "Run 2 reached selection_ready; queue watcher finished" "$run_group"
}

if [[ "$DETACH" == "1" && "${QUEUE_CHILD:-0}" != "1" ]]; then
  session="rex012_r02_queue"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Queue tmux session already exists: $session" >&2
    exit 3
  fi
  printf -v child_command \
    'QUEUE_CHILD=1 QUEUE_RUN2=1 DETACH=0 REPO_ROOT=%q EXP_DIR=%q RUN1_GROUP_NAME=%q MIN_FREE_GPU_GIB=%q POLL_SECONDS=%q bash %q >>%q 2>&1' \
    "$REPO_ROOT" "$EXP_DIR" "$RUN1_GROUP_NAME" "$MIN_FREE_GPU_GIB" "$POLL_SECONDS" \
    "$SCRIPT_PATH" "$QUEUE_LOG"
  tmux new-session -d -s "$session" "$child_command"
  pid="$(tmux display-message -p -t "$session" '#{pane_pid}')"
  printf '%s\n' "$pid" >"$QUEUE_DIR/r02_queue.pid"
  printf '%s\n' "$session" >"$QUEUE_DIR/r02_queue.tmux_session"
  write_state waiting_run1 "Detached queue watcher started"
  echo "Started Exp012 Run-2 queue watcher"
  echo "tmux session: $session"
  echo "state: $STATE_MD"
  echo "log: $QUEUE_LOG"
  exit 0
fi

run_queue
