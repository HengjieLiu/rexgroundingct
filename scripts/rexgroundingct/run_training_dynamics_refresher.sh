#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
EXP_ROOT="${EXP_ROOT:-/mnt/shengdata1/hengjie/experiments/rexgroundingct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXP_ROOT/comparisons/training_dynamics}"
EXP011_DIR="$EXP_ROOT/011_voxtell_v123_e4d4_ct_normalization_ablation"
E5_GROUP_SELECTOR="${E5_GROUP_SELECTOR:-$EXP011_DIR/runs/latest_e5d4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DETACH="${DETACH:-1}"
DRY_RUN="${DRY_RUN:-0}"
SESSION_NAME="${SESSION_NAME:-rex_training_dynamics_exp011}"
SCRIPT_PATH="$REPO_ROOT/scripts/rexgroundingct/run_training_dynamics_refresher.sh"
PLOTTER="$REPO_ROOT/scripts/rexgroundingct/plot_training_dynamics.py"

[[ -x "$(command -v python)" ]] || {
  echo "python is required" >&2
  exit 1
}
[[ -f "$PLOTTER" ]] || {
  echo "Missing plotter: $PLOTTER" >&2
  exit 1
}
[[ -e "$E5_GROUP_SELECTOR" ]] || {
  echo "Missing Exp011 e5d4 run selector: $E5_GROUP_SELECTOR" >&2
  exit 1
}
E5_GROUP="$(readlink -f "$E5_GROUP_SELECTOR")"

plot_command=(
  python "$PLOTTER"
  --experiments 008 009 011
  --exp-root "$EXP_ROOT"
  --output-root "$OUTPUT_ROOT"
)

marker_state() {
  {
    find "$E5_GROUP/milestones" \
      -mindepth 2 -maxdepth 2 -type f -name report.complete \
      -printf '%P\n' 2>/dev/null | sort
    if [[ -f "$E5_GROUP/.experiment_complete" ]]; then
      echo ".experiment_complete"
    fi
  } | sha256sum | awk '{print $1}'
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "mode=dry-run"
  echo "e5_group=$E5_GROUP"
  echo "output_root=$OUTPUT_ROOT"
  echo "poll_seconds=$POLL_SECONDS"
  echo "cuda_visible_devices=<empty>"
  find "$E5_GROUP/milestones" \
    -mindepth 2 -maxdepth 2 -type f -name report.complete \
    -printf 'completed_marker=%P\n' 2>/dev/null | sort
  CUDA_VISIBLE_DEVICES="" \
    MPLCONFIGDIR="/tmp/rexgroundingct-training-dynamics-${UID}" \
    "${plot_command[@]}" --dry-run
  exit 0
fi

if [[ "$DETACH" == "1" && "${REFRESHER_CHILD:-0}" != "1" ]]; then
  command -v tmux >/dev/null || {
    echo "tmux is required for detached mode" >&2
    exit 1
  }
  mkdir -p "$OUTPUT_ROOT"
  LOG_PATH="$OUTPUT_ROOT/refresher.log"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Training-dynamics refresher already running in tmux: $SESSION_NAME"
    echo "Log: $LOG_PATH"
    exit 0
  fi
  tmux new-session -d -s "$SESSION_NAME" \
    "cd '$REPO_ROOT' && REFRESHER_CHILD=1 DETACH=0 POLL_SECONDS='$POLL_SECONDS' EXP_ROOT='$EXP_ROOT' OUTPUT_ROOT='$OUTPUT_ROOT' E5_GROUP_SELECTOR='$E5_GROUP' bash '$SCRIPT_PATH' >> '$LOG_PATH' 2>&1"
  echo "Started training-dynamics refresher in tmux: $SESSION_NAME"
  echo "Log: $LOG_PATH"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
LOCK_PATH="$OUTPUT_ROOT/.refresher.lock"
STATE_PATH="$OUTPUT_ROOT/.last_exp011_marker_state"
STATUS_PATH="$OUTPUT_ROOT/refresher_status.json"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "Another training-dynamics refresher owns $LOCK_PATH"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="/tmp/rexgroundingct-training-dynamics-${UID}"
mkdir -p "$MPLCONFIGDIR"

write_status() {
  local state="$1"
  local token="$2"
  local temporary="$STATUS_PATH.tmp.$$"
  printf '{\n  "state": "%s",\n  "marker_state": "%s",\n  "run_group": "%s",\n  "updated_at_utc": "%s"\n}\n' \
    "$state" "$token" "$(basename "$E5_GROUP")" "$(date -u +%FT%TZ)" \
    > "$temporary"
  mv "$temporary" "$STATUS_PATH"
}

echo "[$(date -u +%FT%TZ)] refresher started for $E5_GROUP"
while true; do
  token="$(marker_state)"
  previous=""
  [[ -f "$STATE_PATH" ]] && previous="$(<"$STATE_PATH")"
  if [[ "$token" != "$previous" ]]; then
    echo "[$(date -u +%FT%TZ)] new completed report marker state: $token"
    write_status "rendering" "$token"
    if "${plot_command[@]}"; then
      temporary="$STATE_PATH.tmp.$$"
      printf '%s\n' "$token" > "$temporary"
      mv "$temporary" "$STATE_PATH"
      write_status "current" "$token"
      echo "[$(date -u +%FT%TZ)] plots refreshed"
    else
      write_status "retry_pending" "$token"
      echo "[$(date -u +%FT%TZ)] plot refresh failed; retrying" >&2
    fi
  fi
  if [[ -f "$E5_GROUP/.experiment_complete" && -f "$STATE_PATH" ]]; then
    final_token="$(marker_state)"
    if [[ "$(<"$STATE_PATH")" == "$final_token" ]]; then
      write_status "complete" "$final_token"
      echo "[$(date -u +%FT%TZ)] Exp011 complete; refresher exiting"
      exit 0
    fi
  fi
  sleep "$POLL_SECONDS"
done
