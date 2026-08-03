#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
cd "$REPO_ROOT"
SCRIPT_PATH="$(readlink -f "$0")"
IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
EXP_ID="013_voxtell_public_category_only_specialists"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
RUNS_DIR="$EXP_DIR/runs"
START_TRAINING="${START_TRAINING:-0}"
DETACH="${DETACH:-1}"
DRY_RUN="${DRY_RUN:-0}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ "$START_TRAINING" != "1" && "$DRY_RUN" != "1" ]]; then
  echo "START_TRAINING=1 is required to launch Exp013" >&2
  exit 2
fi

if [[ -z "${RUN_GROUP:-}" ]]; then
  active_file="$EXP_DIR/config/active_run_group.txt"
  if [[ -f "$active_file" ]]; then
    candidate="$(head -n 1 "$active_file")"
    if [[ -n "$candidate" && ! -f "$RUNS_DIR/$candidate/.selection_ready" ]]; then
      RUN_GROUP="$candidate"
    fi
  fi
fi
RUN_GROUP="${RUN_GROUP:-exp013_category_only_voxtell_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$RUNS_DIR/$RUN_GROUP}"
ORCHESTRATOR_LOG="${ORCHESTRATOR_LOG:-$EXP_DIR/logs/${RUN_GROUP}_orchestrator.log}"
SOURCE_CHECKPOINT="/data/hengjie/datasets/rexgroundingct/.hf_home/hub/models--mrokuss--VoxTell/snapshots/0809ac94c5ed594198bf4e85d63245cf222464d7/voxtell_v1.1/fold_0/checkpoint_final.pth"
CURRENT_STAGE="launcher"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "prepare"
  echo "sample-test"
  echo "smoke"
  echo "eval epoch=0 scope=target"
  echo "report epoch=0 scope=target"
  for spec in "20 target" "40 target" "60 target" "80 target" "100 val200"; do
    read -r epoch scope <<<"$spec"
    echo "train epoch=$epoch"
    echo "eval epoch=$epoch scope=$scope"
    echo "report epoch=$epoch scope=$scope"
  done
  echo "stop state=selection_ready"
  exit 0
fi

mkdir -p "$EXP_DIR/logs" "$GROUP_DIR"

if [[ "$DETACH" == "1" && "${ORCHESTRATOR_CHILD:-0}" != "1" ]]; then
  session="rex013_${RUN_GROUP##*_}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Orchestrator tmux session already exists: $session" >&2
    exit 3
  fi
  printf -v child_command \
    'ORCHESTRATOR_CHILD=1 DETACH=0 START_TRAINING=1 REPO_ROOT=%q IMAGE=%q EXP_DIR=%q RUN_GROUP=%q GROUP_DIR=%q ORCHESTRATOR_LOG=%q bash %q >%q 2>&1' \
    "$REPO_ROOT" "$IMAGE" "$EXP_DIR" "$RUN_GROUP" "$GROUP_DIR" \
    "$ORCHESTRATOR_LOG" "$SCRIPT_PATH" "$ORCHESTRATOR_LOG"
  tmux new-session -d -s "$session" "$child_command"
  pid="$(tmux display-message -p -t "$session" '#{pane_pid}')"
  printf '%s\n' "$pid" >"$GROUP_DIR/orchestrator.pid"
  printf '%s\n' "$session" >"$GROUP_DIR/orchestrator.tmux_session"
  echo "Started Exp013 host orchestrator"
  echo "PID: $pid"
  echo "tmux session: $session"
  echo "Run group: $RUN_GROUP"
  echo "Live report: $GROUP_DIR/reports/exp013_progress.md"
  echo "Log: $ORCHESTRATOR_LOG"
  exit 0
fi

exec 9>"$GROUP_DIR/.orchestrator.lock"
flock -n 9 || { echo "Another orchestrator owns $GROUP_DIR" >&2; exit 3; }

on_exit() {
  local status="$?"
  rm -f "$GROUP_DIR/.orchestrator_running"
  if ((status != 0)); then
    touch "$GROUP_DIR/.orchestrator_failed"
    if [[ -f "$GROUP_DIR/run_group_manifest.json" ]]; then
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO_ROOT/scripts/rexgroundingct" python \
        "$REPO_ROOT/scripts/rexgroundingct/summarize_013_results.py" \
        --group-dir "$GROUP_DIR" \
        --source-checkpoint "$SOURCE_CHECKPOINT" \
        --subset-manifest "$REPO_ROOT/configs/evaluation/rexgroundingct_exp013_category_only_evaluation_subsets.manifest.json" \
        --val200-json "$REPO_ROOT/configs/evaluation/rexgroundingct_val200_seed20260723.json" \
        --output-json "$GROUP_DIR/reports/exp013_progress.json" \
        --output-md "$GROUP_DIR/reports/exp013_progress.md" \
        --current-stage "$CURRENT_STAGE" \
        --failure "orchestrator failed during $CURRENT_STAGE" || true
    fi
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
  local stage="$1" epoch="${2:-}" scope="${3:-}"
  local suffix="$stage"
  [[ -n "$epoch" ]] && suffix="${suffix}_e$(printf '%03d' "$epoch")"
  [[ -n "$scope" ]] && suffix="${suffix}_${scope}"
  CURRENT_STAGE="$suffix"
  local container="rex013_${suffix}_${RUN_GROUP##*_}"
  local stage_log="$GROUP_DIR/${suffix}.log"
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
    -e PYTHONDONTWRITEBYTECODE=1
    -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home
    -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub
    -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet
    -e STAGE="$stage"
    -e EXP_DIR="$EXP_DIR"
    -e GROUP_DIR="$GROUP_DIR"
    -e TARGET_EPOCH="$epoch"
    -e EVAL_SCOPE="$scope"
  )
  if stage_uses_gpu "$stage"; then
    docker_args+=(--gpus '"device=0,1,2,3"')
  fi
  echo "[$(date -u +%FT%TZ)] starting stage=$stage epoch=${epoch:-n/a} scope=${scope:-n/a}"
  "${docker_args[@]}" "$IMAGE" \
    bash /workspace/scripts/rexgroundingct/run_013_category_only_stage.sh \
    2>&1 | tee "$stage_log"
  echo "[$(date -u +%FT%TZ)] completed stage=$stage epoch=${epoch:-n/a} scope=${scope:-n/a}"
}

run_stage prepare
run_stage sample-test
run_stage smoke
run_stage eval 0 target
run_stage report 0 target

for spec in "20 target" "40 target" "60 target" "80 target" "100 val200"; do
  read -r epoch scope <<<"$spec"
  run_stage train "$epoch"
  run_stage eval "$epoch" "$scope"
  run_stage report "$epoch" "$scope"
done

[[ -f "$GROUP_DIR/.selection_ready" ]] || {
  echo "Exp013 did not reach selection_ready" >&2
  exit 1
}
touch "$GROUP_DIR/.orchestrator_complete"
echo "Exp013 paused at selection_ready: $GROUP_DIR"
