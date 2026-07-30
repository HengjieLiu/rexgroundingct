#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
SCRIPT_PATH="$(readlink -f "$0")"
IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
EXP_ID="011_voxtell_v123_e4d4_ct_normalization_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
RUNS_DIR="$EXP_DIR/runs"
DETACH="${DETACH:-1}"
START_E5D4="${START_E5D4:-0}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ "$START_E5D4" != "1" ]]; then
  echo "START_E5D4=1 is required to launch the exp011 e5d4 rerun" >&2
  exit 2
fi

resolve_e4_group() {
  if [[ -n "${E4_GROUP_DIR:-}" ]]; then
    readlink -f "$E4_GROUP_DIR"
  elif [[ -e "$RUNS_DIR/latest_e4d4" ]]; then
    readlink -f "$RUNS_DIR/latest_e4d4"
  elif [[ -e "$RUNS_DIR/latest" ]]; then
    local candidate
    candidate="$(readlink -f "$RUNS_DIR/latest")"
    [[ "$(basename "$candidate")" == exp011_ct_norm_e4d4_* ]] || {
      echo "Cannot infer completed e4d4 group from $candidate" >&2
      return 1
    }
    echo "$candidate"
  else
    echo "No completed exp011 e4d4 group is available" >&2
    return 1
  fi
}

E4_GROUP_DIR="$(resolve_e4_group)"

if [[ -z "${RUN_GROUP:-}" ]]; then
  active_file="$EXP_DIR/config/active_e5d4_run_group.txt"
  if [[ -f "$active_file" ]]; then
    candidate="$(head -n 1 "$active_file")"
    if [[ -n "$candidate" && ! -f "$RUNS_DIR/$candidate/.experiment_complete" ]]; then
      RUN_GROUP="$candidate"
    fi
  fi
fi
RUN_GROUP="${RUN_GROUP:-exp011_ct_norm_e5d4_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$RUNS_DIR/$RUN_GROUP}"
ORCHESTRATOR_LOG="${ORCHESTRATOR_LOG:-$EXP_DIR/logs/${RUN_GROUP}_orchestrator.log}"

mkdir -p "$EXP_DIR/logs" "$GROUP_DIR"

if [[ "$DETACH" == "1" && "${ORCHESTRATOR_CHILD:-0}" != "1" ]]; then
  session="rex011_e5d4_${RUN_GROUP##*_}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Orchestrator tmux session already exists: $session" >&2
    exit 3
  fi
  printf -v child_command \
    'ORCHESTRATOR_CHILD=1 DETACH=0 START_E5D4=1 REPO_ROOT=%q IMAGE=%q EXP_DIR=%q E4_GROUP_DIR=%q RUN_GROUP=%q GROUP_DIR=%q ORCHESTRATOR_LOG=%q bash %q >%q 2>&1' \
    "$REPO_ROOT" "$IMAGE" "$EXP_DIR" "$E4_GROUP_DIR" "$RUN_GROUP" \
    "$GROUP_DIR" "$ORCHESTRATOR_LOG" "$SCRIPT_PATH" "$ORCHESTRATOR_LOG"
  tmux new-session -d -s "$session" "$child_command"
  pid="$(tmux display-message -p -t "$session" '#{pane_pid}')"
  printf '%s\n' "$pid" > "$GROUP_DIR/orchestrator.pid"
  printf '%s\n' "$session" > "$GROUP_DIR/orchestrator.tmux_session"
  echo "Started exp011 e5d4 host orchestrator"
  echo "PID: $pid"
  echo "tmux session: $session"
  echo "Run group: $RUN_GROUP"
  echo "Log: $ORCHESTRATOR_LOG"
  exit 0
fi

exec 9>"$GROUP_DIR/.orchestrator.lock"
if ! flock -n 9; then
  echo "Another orchestrator owns $GROUP_DIR/.orchestrator.lock" >&2
  exit 3
fi

on_exit() {
  local status="$?"
  rm -f "$GROUP_DIR/.orchestrator_running"
  if ((status != 0)); then
    touch "$GROUP_DIR/.orchestrator_failed"
  fi
}
trap on_exit EXIT
rm -f "$GROUP_DIR/.orchestrator_failed"
touch "$GROUP_DIR/.orchestrator_running"

stage_uses_gpu() {
  case "$1" in
    smoke|train|eval) return 0 ;;
    *) return 1 ;;
  esac
}

run_stage() {
  local stage="$1" epoch="${2:-}" probe="${3:-}"
  local suffix="$stage"
  [[ -n "$epoch" ]] && suffix="${suffix}_e$(printf '%03d' "$epoch")"
  [[ -n "$probe" ]] && suffix="${suffix}_${probe}"
  local container="rex011_e5d4_${suffix}_${RUN_GROUP##*_}"
  local stage_log="$GROUP_DIR/${suffix}.log"

  if docker ps -a --format '{{.Names}}' | grep -Fxq "$container"; then
    echo "Waiting for existing stage container: $container"
    docker wait "$container" || true
  fi

  local docker_args=(
    docker run
    --rm
    --name "$container"
    --ipc=host
    --shm-size=32g
    --user "$(id -u):$(id -g)"
    -v "$REPO_ROOT:/workspace:ro"
    -v /data/hengjie:/data/hengjie:ro
    -v /mnt/shengdata1:/mnt/shengdata1
    -e HOME=/tmp
    -e PYTHONDONTWRITEBYTECODE=1
    -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home
    -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub
    -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet
    -e STAGE="$stage"
    -e EXP_DIR="$EXP_DIR"
    -e GROUP_DIR="$GROUP_DIR"
    -e E4_GROUP_DIR="$E4_GROUP_DIR"
    -e TARGET_EPOCH="$epoch"
    -e EVAL_PROBE="$probe"
  )
  if stage_uses_gpu "$stage"; then
    docker_args+=(--gpus '"device=0,1,2"')
  fi
  echo "[$(date -u +%FT%TZ)] starting stage=$stage epoch=${epoch:-n/a} probe=${probe:-n/a}"
  "${docker_args[@]}" "$IMAGE" \
    bash /workspace/scripts/rexgroundingct/run_011_e5d4_stage.sh \
    2>&1 | tee "$stage_log"
  echo "[$(date -u +%FT%TZ)] completed stage=$stage epoch=${epoch:-n/a} probe=${probe:-n/a}"
}

run_stage prepare
run_stage sample-test
run_stage smoke

for epoch in 5 20 40 60 80 100; do
  run_stage train "$epoch"
  run_stage eval "$epoch" val20
  run_stage report "$epoch" val20
done

run_stage eval 100 val200
run_stage report 100 val200

[[ -f "$GROUP_DIR/.experiment_complete" ]] || {
  echo "Final completion marker was not written" >&2
  exit 1
}
touch "$GROUP_DIR/.orchestrator_complete"
echo "Exp011 e5d4 rerun complete: $GROUP_DIR"
