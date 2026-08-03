#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
cd "$REPO_ROOT"
SCRIPT_PATH="$(readlink -f "$0")"
IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
EXP_ID="012_voxtell_category_specialists_replay50_cont100"
PROFILE_ID="r02_2d_diffuse_replay_ratio"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
RUNS_DIR="$EXP_DIR/runs"
START_TRAINING="${START_TRAINING:-0}"
DETACH="${DETACH:-1}"
DRY_RUN="${DRY_RUN:-0}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN1_GROUP_DIR="${RUN1_GROUP_DIR:-$RUNS_DIR/exp012_category_specialists_20260801T215219Z}"

if [[ "$START_TRAINING" != "1" && "$DRY_RUN" != "1" ]]; then
  echo "START_TRAINING=1 is required to launch Exp012 Run 2" >&2
  exit 2
fi

if [[ -z "${RUN_GROUP:-}" ]]; then
  active_file="$EXP_DIR/config/active_r02_run_group.txt"
  if [[ -f "$active_file" ]]; then
    candidate="$(head -n 1 "$active_file")"
    if [[ -n "$candidate" && ! -f "$RUNS_DIR/$candidate/.selection_ready" ]]; then
      RUN_GROUP="$candidate"
    fi
  fi
fi
RUN_GROUP="${RUN_GROUP:-exp012_category_specialists_r02_replay_ratio_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$RUNS_DIR/$RUN_GROUP}"
ORCHESTRATOR_LOG="${ORCHESTRATOR_LOG:-$EXP_DIR/logs/${RUN_GROUP}_orchestrator.log}"
SOURCE_CHECKPOINT="/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/baseline_cont100/model_epoch100/fold_0/checkpoint_final.pth"
CURRENT_STAGE="launcher"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "prepare"
  echo "sample-test"
  echo "smoke"
  echo "eval epoch=0 scope=union"
  echo "report epoch=0 scope=union"
  for spec in "5 target" "20 union" "40 union" "60 union" "80 union" "100 val200"; do
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
  session="rex012_r02_${RUN_GROUP##*_}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Orchestrator tmux session already exists: $session" >&2
    exit 3
  fi
  printf -v child_command \
    'ORCHESTRATOR_CHILD=1 DETACH=0 START_TRAINING=1 REPO_ROOT=%q IMAGE=%q EXP_DIR=%q RUN1_GROUP_DIR=%q RUN_GROUP=%q GROUP_DIR=%q ORCHESTRATOR_LOG=%q bash %q >%q 2>&1' \
    "$REPO_ROOT" "$IMAGE" "$EXP_DIR" "$RUN1_GROUP_DIR" "$RUN_GROUP" "$GROUP_DIR" \
    "$ORCHESTRATOR_LOG" "$SCRIPT_PATH" "$ORCHESTRATOR_LOG"
  tmux new-session -d -s "$session" "$child_command"
  pid="$(tmux display-message -p -t "$session" '#{pane_pid}')"
  printf '%s\n' "$pid" >"$GROUP_DIR/orchestrator.pid"
  printf '%s\n' "$session" >"$GROUP_DIR/orchestrator.tmux_session"
  echo "Started Exp012 Run-2 host orchestrator"
  echo "PID: $pid"
  echo "tmux session: $session"
  echo "Run group: $RUN_GROUP"
  echo "Live report: $GROUP_DIR/reports/exp012_r02_progress.md"
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
        "$REPO_ROOT/scripts/rexgroundingct/summarize_012_r02_results.py" \
        --group-dir "$GROUP_DIR" \
        --source-checkpoint "$SOURCE_CHECKPOINT" \
        --subset-manifest "$REPO_ROOT/configs/evaluation/rexgroundingct_exp012_r02_evaluation_subsets.manifest.json" \
        --val80-json "$REPO_ROOT/side_experiments/sideexp001_validation_probe_design/outputs/rexgroundingct_val80_sideexp001_validation_probe_design.json" \
        --val200-json "$REPO_ROOT/configs/evaluation/rexgroundingct_val200_seed20260723.json" \
        --run1-reference-json "$GROUP_DIR/config/run1_reference_report.json" \
        --output-json "$GROUP_DIR/reports/exp012_r02_progress.json" \
        --output-md "$GROUP_DIR/reports/exp012_r02_progress.md" \
        --current-stage "$CURRENT_STAGE" \
        --failure "orchestrator failed during $CURRENT_STAGE" || true
    fi
  fi
  PYTHONDONTWRITEBYTECODE=1 python \
    "$REPO_ROOT/scripts/rexgroundingct/update_012_run_index.py" \
    --experiment-dir "$EXP_DIR" || true
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
  local container="rex012_r02_${suffix}_${RUN_GROUP##*_}"
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
    -e RUN1_GROUP_DIR="$RUN1_GROUP_DIR"
    -e TARGET_EPOCH="$epoch"
    -e EVAL_SCOPE="$scope"
  )
  if stage_uses_gpu "$stage"; then
    docker_args+=(--gpus '"device=0,1,2,3"')
  fi
  echo "[$(date -u +%FT%TZ)] starting stage=$stage epoch=${epoch:-n/a} scope=${scope:-n/a}"
  "${docker_args[@]}" "$IMAGE" \
    bash /workspace/scripts/rexgroundingct/run_012_r02_stage.sh \
    2>&1 | tee "$stage_log"
  echo "[$(date -u +%FT%TZ)] completed stage=$stage epoch=${epoch:-n/a} scope=${scope:-n/a}"
}

run_stage prepare
run_stage sample-test
run_stage smoke
run_stage eval 0 union
run_stage report 0 union

for spec in "5 target" "20 union" "40 union" "60 union" "80 union" "100 val200"; do
  read -r epoch scope <<<"$spec"
  run_stage train "$epoch"
  run_stage eval "$epoch" "$scope"
  run_stage report "$epoch" "$scope"
done

[[ -f "$GROUP_DIR/.selection_ready" ]] || {
  echo "Exp012 Run 2 did not reach selection_ready" >&2
  exit 1
}
touch "$GROUP_DIR/.orchestrator_complete"
echo "Exp012 Run 2 paused at selection_ready: $GROUP_DIR"
