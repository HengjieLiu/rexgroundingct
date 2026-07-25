#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

EXP_ID="006_voxtell_cached_native_v123_lr_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
GROUP_DIR="${GROUP_DIR:?GROUP_DIR is required}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:?SOURCE_MODEL_DIR is required}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed20260723.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"
STABILITY_SECONDS="${STABILITY_SECONDS:-10}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-43200}"
EVAL_SKIP_USED_MIB="${EVAL_SKIP_USED_MIB:-47000}"

arms=(v123_cached_e7_d6 v123_cached_e7_d4 v123_cached_e6_d4 v123_cached_e5_d4)
val20_epochs=(5 20 40 60 80 100)

arm_gpu() {
  case "$1" in
    v123_cached_e7_d6) echo 0 ;;
    v123_cached_e7_d4) echo 1 ;;
    v123_cached_e6_d4) echo 2 ;;
    v123_cached_e5_d4) echo 3 ;;
    *) return 1 ;;
  esac
}

epoch_update() {
  echo "$(($1 * 100))"
}

eval_complete() {
  local eval_dir="$1"
  local dataset_json="$2"
  python - "$eval_dir" "$dataset_json" <<'PY'
import json
import sys
from pathlib import Path

root, dataset = map(Path, sys.argv[1:3])
summary = root / "reports" / "val_quick_global_eval_summary.json"
eval_json = root / "eval" / "val_quick_global_eval.json"
predictions = root / "predictions"
if not (summary.is_file() and eval_json.is_file() and predictions.is_dir() and dataset.is_file()):
    raise SystemExit(1)
entries = json.loads(dataset.read_text())["test"]
data = json.loads(summary.read_text())
if int(data.get("total_cases", -1)) != len(entries):
    raise SystemExit(1)
if int(data.get("total_findings", -1)) != sum(len(e["findings"]) for e in entries):
    raise SystemExit(1)
if any(not (predictions / e["name"]).is_file() for e in entries):
    raise SystemExit(1)
PY
}

checkpoint_stable() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  local a b
  a="$(stat -c '%s:%Y' "$path")"
  sleep "$STABILITY_SECONDS"
  [[ -s "$path" ]] || return 1
  b="$(stat -c '%s:%Y' "$path")"
  [[ "$a" == "$b" ]]
}

gpu_too_busy() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 || true)"
  [[ -n "$used" ]] || return 1
  if [[ ! "$used" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Skipping GPU memory gate because nvidia-smi returned non-numeric output for GPU $gpu: $used" >&2
    return 1
  fi
  python - "$used" "$EVAL_SKIP_USED_MIB" <<'PY'
import sys
used = float(sys.argv[1])
limit = float(sys.argv[2])
raise SystemExit(0 if used >= limit else 1)
PY
}

lock_age_seconds() {
  local lock="$1"
  python - "$lock" <<'PY'
import sys
import time
from pathlib import Path
path = Path(sys.argv[1])
print(max(0.0, time.time() - path.stat().st_mtime))
PY
}

acquire_lock() {
  local eval_dir="$1"
  local lock="$eval_dir/.eval.lock"
  mkdir -p "$eval_dir"
  while true; do
    if (set -o noclobber; printf '{"pid":%s,"created":"%s","phase":"coordinator"}\n' "$$" "$(date -u +%FT%TZ)" > "$lock") 2>/dev/null; then
      echo "$lock"
      return 0
    fi
    local age
    age="$(lock_age_seconds "$lock" 2>/dev/null || echo 0)"
    if python - "$age" "$LOCK_STALE_SECONDS" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
    then
      echo "Removing stale eval lock: $lock"
      rm -f "$lock"
      continue
    fi
    return 1
  done
}

materialize_model() {
  local checkpoint="$1"
  local model_dir="$2"
  mkdir -p "$model_dir/fold_0"
  cp "$SOURCE_MODEL_DIR/plans.json" "$model_dir/plans.json"
  local target="$model_dir/fold_0/checkpoint_final.pth"
  if [[ ! -f "$target" || "$(stat -c %s "$target")" != "$(stat -c %s "$checkpoint")" ]]; then
    local tmp="$target.tmp.$$"
    rm -f "$tmp"
    ln "$checkpoint" "$tmp" 2>/dev/null || cp "$checkpoint" "$tmp"
    mv -f "$tmp" "$target"
  fi
}

run_eval_once() {
  local arm="$1" epoch="$2" probe="$3"
  local gpu
  gpu="$(arm_gpu "$arm")"
  local dataset="$VAL20_JSON"
  [[ "$probe" == "val200" ]] && dataset="$VAL200_JSON"
  local run_dir="$GROUP_DIR/$arm"
  local eval_dir="$run_dir/eval_epoch$(printf '%03d' "$epoch")_$probe"
  local checkpoint="$run_dir/checkpoints/checkpoint_update_$(printf '%06d' "$(epoch_update "$epoch")").pth"

  if eval_complete "$eval_dir" "$dataset"; then
    echo "Already complete: $arm epoch=$epoch $probe"
    return 0
  fi
  checkpoint_stable "$checkpoint" || return 2
  if gpu_too_busy "$gpu"; then
    echo "GPU $gpu above eval memory threshold; retry later: $arm epoch=$epoch $probe"
    return 3
  fi

  local lock
  lock="$(acquire_lock "$eval_dir")" || return 4
  local model_dir="$run_dir/model_epoch$(printf '%03d' "$epoch")"
  materialize_model "$checkpoint" "$model_dir"
  mkdir -p "$eval_dir/logs" "$eval_dir/predictions"
  local log="$eval_dir/logs/coordinator_$(date -u +%Y%m%dT%H%M%SZ).log"
  if (
    set -euo pipefail
    echo "start arm=$arm epoch=$epoch probe=$probe gpu=$gpu"
    EVAL_LOCK_HELD=1 CUDA_VISIBLE_DEVICES="$gpu" python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
      --split val \
      --dataset-json "$dataset" \
      --gpu 0 \
      --model-dir "$model_dir" \
      --embeddings "$EMBEDDINGS" \
      --output-dir "$eval_dir/predictions" \
      --output-type mask \
      --threshold 0.5 \
      --preprocessed-cache-dir "$CACHE_ROOT"
    EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset" \
      EVAL_LABEL="Experiment 006 $arm epoch $epoch fixed $probe threshold 0.5" NUM_WORKERS=8 \
      bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$eval_dir" val
    echo "done arm=$arm epoch=$epoch probe=$probe"
  ) >"$log" 2>&1; then
    rm -f "$lock"
    return 0
  fi
  rm -f "$lock"
  echo "Evaluation failed; will retry: arm=$arm epoch=$epoch probe=$probe log=$log" >&2
  return 5
}

run_until_complete() {
  local arm="$1" epoch="$2" probe="$3"
  local dataset="$VAL20_JSON"
  [[ "$probe" == "val200" ]] && dataset="$VAL200_JSON"
  local eval_dir="$GROUP_DIR/$arm/eval_epoch$(printf '%03d' "$epoch")_$probe"
  while true; do
    if eval_complete "$eval_dir" "$dataset"; then
      return 0
    fi
    if [[ -f "$GROUP_DIR/$arm/.train_failed" ]]; then
      echo "Training failed before eval completed: $arm epoch=$epoch $probe" >&2
      return 1
    fi
    if run_eval_once "$arm" "$epoch" "$probe"; then
      return 0
    fi
    sleep "$RETRY_SECONDS"
  done
}

arm_worker() {
  local arm="$1"
  for epoch in "${val20_epochs[@]}"; do
    run_until_complete "$arm" "$epoch" val20
  done
  until [[ -f "$GROUP_DIR/$arm/.train_complete" || -f "$GROUP_DIR/$arm/.train_failed" ]]; do
    sleep "$POLL_SECONDS"
  done
  run_until_complete "$arm" 100 val200
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for arm in "${arms[@]}"; do
    for epoch in "${val20_epochs[@]}"; do
      run_dir="$GROUP_DIR/$arm"
      eval_dir="$run_dir/eval_epoch$(printf '%03d' "$epoch")_val20"
      checkpoint="$run_dir/checkpoints/checkpoint_update_$(printf '%06d' "$(epoch_update "$epoch")").pth"
      if eval_complete "$eval_dir" "$VAL20_JSON"; then status=complete; elif [[ -f "$checkpoint" ]]; then status=checkpoint_present; else status=waiting; fi
      echo "$arm epoch=$epoch val20 status=$status checkpoint=$checkpoint"
    done
    eval_dir="$GROUP_DIR/$arm/eval_epoch100_val200"
    if eval_complete "$eval_dir" "$VAL200_JSON"; then status=complete; else status=waiting; fi
    echo "$arm epoch=100 val200 status=$status"
  done
  exit 0
fi

echo "Experiment 006 eval coordinator started for $GROUP_DIR"
pids=()
for arm in "${arms[@]}"; do
  arm_worker "$arm" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

EXP003_SUMMARY="/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v123_opt_poscrop_emptyloss/full_100ep/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
EXP004_SUMMARY="/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft/runs/latest/native192_cont100/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
reference_args=()
[[ -f "$EXP003_SUMMARY" ]] && reference_args+=(--reference "exp003_v123_epoch100=$EXP003_SUMMARY")
[[ -f "$EXP004_SUMMARY" ]] && reference_args+=(--reference "exp004_native_cont100=$EXP004_SUMMARY")

python /workspace/scripts/rexgroundingct/summarize_006_results.py \
  --exp-dir "$EXP_DIR" \
  --group-dir "$GROUP_DIR" \
  --output-json "$EXP_DIR/reports/lr_ablation_summary.json" \
  --output-md "$EXP_DIR/reports/lr_ablation_report.md" \
  "${reference_args[@]}"

if [[ "$status" == "0" ]]; then
  touch "$GROUP_DIR/.experiment_complete"
fi
exit "$status"
