#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

EXP_ID="008_voxtell_dual_branch_proposal_refinement_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/exp008_dual_branch_20260726T182647Z}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
EXPECTED_VAL200_SHA="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
WINDOW_BATCH="${WINDOW_BATCH:-1}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-43200}"

VARIANTS=(
  v1_sharedfusion_softguide
  v1_dualfusion_softguide
  v2_dualfusion_precision
  v3_dualfusion_softguide_joint
)
GPUS=(0 1 2 3)

[[ -d "$GROUP_DIR" ]] || { echo "Missing run group: $GROUP_DIR" >&2; exit 1; }
[[ -f "$CACHE_ROOT/.complete" ]] || { echo "Incomplete cache: $CACHE_ROOT" >&2; exit 1; }
[[ -f "$EMBEDDINGS" ]] || { echo "Missing embeddings: $EMBEDDINGS" >&2; exit 1; }
actual_val200_sha="$(sha256sum "$VAL200_JSON" | cut -d ' ' -f 1)"
[[ "$actual_val200_sha" == "$EXPECTED_VAL200_SHA" ]] || {
  echo "val200 SHA mismatch: $actual_val200_sha" >&2
  exit 1
}

eval_complete() {
  local eval_dir="$1"
  python - "$eval_dir" "$VAL200_JSON" <<'PY'
import json
import sys
from pathlib import Path

root, dataset_path = map(Path, sys.argv[1:3])
summary_path = root / "reports" / "val_quick_global_eval_summary.json"
eval_path = root / "eval" / "val_quick_global_eval.json"
predictions = root / "predictions"
if not (
    summary_path.is_file()
    and eval_path.is_file()
    and predictions.is_dir()
    and dataset_path.is_file()
):
    raise SystemExit(1)
entries = json.loads(dataset_path.read_text())["test"]
summary = json.loads(summary_path.read_text())
if int(summary.get("total_cases", -1)) != len(entries):
    raise SystemExit(1)
if int(summary.get("total_findings", -1)) != sum(
    len(entry["findings"]) for entry in entries
):
    raise SystemExit(1)
if any(not (predictions / entry["name"]).is_file() for entry in entries):
    raise SystemExit(1)
PY
}

lock_age_seconds() {
  local lock_dir="$1"
  python - "$lock_dir" <<'PY'
import sys
import time
from pathlib import Path

print(max(0.0, time.time() - Path(sys.argv[1]).stat().st_mtime))
PY
}

acquire_lock() {
  local eval_dir="$1"
  local lock_dir="$eval_dir/.epoch080_val200_runner.lock"
  mkdir -p "$eval_dir"
  if mkdir "$lock_dir" 2>/dev/null; then
    printf '{"pid":%s,"created":"%s"}\n' "$$" "$(date -u +%FT%TZ)" \
      >"$lock_dir/owner.json"
    echo "$lock_dir"
    return 0
  fi
  if eval_complete "$eval_dir"; then
    return 1
  fi
  local age
  age="$(lock_age_seconds "$lock_dir")"
  if python - "$age" "$LOCK_STALE_SECONDS" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
  then
    rm -rf "$lock_dir"
    mkdir "$lock_dir"
    printf '{"pid":%s,"created":"%s","stale_lock_replaced":true}\n' \
      "$$" "$(date -u +%FT%TZ)" >"$lock_dir/owner.json"
    echo "$lock_dir"
    return 0
  fi
  echo "Active eval lock: $lock_dir" >&2
  return 2
}

run_eval() {
  local variant="$1" gpu="$2"
  local arm_dir="$GROUP_DIR/$variant"
  local model_dir="$arm_dir/model_epoch080"
  local eval_dir="$arm_dir/eval_epoch080_val200"
  local lock_dir=""

  if eval_complete "$eval_dir"; then
    echo "Already complete: $variant epoch80 val200"
    return 0
  fi
  [[ -f "$model_dir/fold_0/checkpoint_final.pth" ]] || {
    echo "Missing epoch-80 model: $model_dir" >&2
    return 1
  }
  [[ -f "$model_dir/model_spec.json" ]] || {
    echo "Missing model spec: $model_dir/model_spec.json" >&2
    return 1
  }
  lock_dir="$(acquire_lock "$eval_dir")" || {
    eval_complete "$eval_dir" && return 0
    return 1
  }
  mkdir -p "$eval_dir/logs" "$eval_dir/status" "$eval_dir/predictions"
  echo "Starting $variant epoch80 val200 on GPU $gpu"

  if ! EVAL_LOCK_DISABLE=1 CUDA_VISIBLE_DEVICES="$gpu" \
    python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
      --split val \
      --dataset-json "$VAL200_JSON" \
      --gpu 0 \
      --model-dir "$model_dir" \
      --embeddings "$EMBEDDINGS" \
      --output-dir "$eval_dir/predictions" \
      --output-type mask \
      --threshold 0.5 \
      --preprocessed-cache-dir "$CACHE_ROOT" \
      --sliding-window-batch-size "$WINDOW_BATCH" \
      --status-json "$eval_dir/status/inference.json" \
      >"$eval_dir/logs/inference.log" 2>&1; then
    rm -rf "$lock_dir"
    echo "Inference failed: $variant" >&2
    return 1
  fi

  if ! EVAL_LOCK_DISABLE=1 GLOBAL_ONLY=1 \
    EVAL_DATASET_JSON_OVERRIDE="$VAL200_JSON" \
    EVAL_LABEL="Experiment 008 $variant epoch 80 fixed val200 threshold 0.5" \
    NUM_WORKERS=8 \
    bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh \
      "$eval_dir" val >"$eval_dir/logs/final_eval.log" 2>&1; then
    rm -rf "$lock_dir"
    echo "Evaluator failed: $variant" >&2
    return 1
  fi

  eval_complete "$eval_dir"
  touch "$eval_dir/.complete"
  rm -rf "$lock_dir"
  echo "Completed $variant epoch80 val200"
}

pids=()
for index in "${!VARIANTS[@]}"; do
  run_eval "${VARIANTS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
[[ "$status" == "0" ]] || exit "$status"

python /workspace/scripts/rexgroundingct/summarize_008_epoch080_val200.py \
  --group-dir "$GROUP_DIR" \
  --output-json "$EXP_DIR/reports/dual_branch_epoch080_val200_summary.json" \
  --output-md "$EXP_DIR/reports/dual_branch_epoch080_val200_report.md"
touch "$GROUP_DIR/.epoch080_val200_complete"
echo "All Exp008 epoch-80 val200 evaluations are complete."
