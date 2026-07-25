#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

EXP_ID="006_voxtell_cached_native_v123_lr_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
GROUP_DIR="${GROUP_DIR:?GROUP_DIR is required}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$EXP_DIR/config/public_voxtell_v1_1_model}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed20260723.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
STABILITY_SECONDS="${STABILITY_SECONDS:-10}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-43200}"
FORCE_STEAL_LOCKS="${FORCE_STEAL_LOCKS:-0}"
RECOVERY_GPUS="${RECOVERY_GPUS:-0 1 2 3}"
VAL200_ARMS="${VAL200_ARMS:-v123_cached_e7_d4 v123_cached_e6_d4 v123_cached_e5_d4 v123_cached_e7_d6}"
VAL20_ARMS="${VAL20_ARMS:-v123_cached_e7_d6 v123_cached_e7_d4 v123_cached_e6_d4 v123_cached_e5_d4}"
VAL20_EPOCHS="${VAL20_EPOCHS:-40 60 80 100}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

dataset_for_probe() {
  case "$1" in
    val20) echo "$VAL20_JSON" ;;
    val200) echo "$VAL200_JSON" ;;
    *) echo "Unknown probe: $1" >&2; return 1 ;;
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
summary_data = json.loads(summary.read_text())
if int(summary_data.get("total_cases", -1)) != len(entries):
    raise SystemExit(1)
if int(summary_data.get("total_findings", -1)) != sum(len(entry["findings"]) for entry in entries):
    raise SystemExit(1)
if any(not (predictions / entry["name"]).is_file() for entry in entries):
    raise SystemExit(1)
PY
}

prediction_count() {
  local eval_dir="$1"
  if [[ -d "$eval_dir/predictions" ]]; then
    find "$eval_dir/predictions" -maxdepth 1 -type f -name '*.nii.gz' | wc -l
  else
    echo 0
  fi
}

expected_case_count() {
  local dataset_json="$1"
  python - "$dataset_json" <<'PY'
import json
import sys
from pathlib import Path
print(len(json.loads(Path(sys.argv[1]).read_text())["test"]))
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
  local forced_steal_done=0
  mkdir -p "$eval_dir"
  while true; do
    if (set -o noclobber; printf '{"pid":%s,"created":"%s","phase":"exp006_recovery_eval"}\n' "$$" "$(date -u +%FT%TZ)" > "$lock") 2>/dev/null; then
      echo "$lock"
      return 0
    fi
    if [[ "$FORCE_STEAL_LOCKS" == "1" && "$forced_steal_done" == "0" ]]; then
      echo "Stealing orphaned eval lock: $lock" >&2
      rm -f "$lock"
      forced_steal_done=1
      continue
    fi
    local age
    age="$(lock_age_seconds "$lock" 2>/dev/null || echo 0)"
    if python - "$age" "$LOCK_STALE_SECONDS" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
    then
      echo "Removing stale eval lock: $lock" >&2
      rm -f "$lock"
      continue
    fi
    echo "Eval lock is active; skip for now: $lock" >&2
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

summarize_partial_results() {
  local exp003_summary="/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v123_opt_poscrop_emptyloss/full_100ep/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
  local exp004_summary="/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft/runs/latest/native192_cont100/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
  local reference_args=()
  [[ -f "$exp003_summary" ]] && reference_args+=(--reference "exp003_v123_epoch100=$exp003_summary")
  [[ -f "$exp004_summary" ]] && reference_args+=(--reference "exp004_native_cont100=$exp004_summary")
  python /workspace/scripts/rexgroundingct/summarize_006_results.py \
    --exp-dir "$EXP_DIR" \
    --group-dir "$GROUP_DIR" \
    --output-json "$EXP_DIR/reports/lr_ablation_summary.json" \
    --output-md "$EXP_DIR/reports/lr_ablation_report.md" \
    "${reference_args[@]}"
}

run_inference_shard() {
  local arm="$1" epoch="$2" probe="$3" gpu="$4" shard_count="$5" shard_index="$6"
  local dataset eval_dir model_dir status_json shard_log
  dataset="$(dataset_for_probe "$probe")"
  eval_dir="$GROUP_DIR/$arm/eval_epoch$(printf '%03d' "$epoch")_$probe"
  model_dir="$GROUP_DIR/$arm/model_epoch$(printf '%03d' "$epoch")"
  mkdir -p "$eval_dir/logs" "$eval_dir/status" "$eval_dir/predictions"
  status_json="$eval_dir/status/recovery_${TIMESTAMP}_shard${shard_index}.json"
  shard_log="$eval_dir/logs/recovery_${TIMESTAMP}_shard${shard_index}_gpu${gpu}.log"
  echo "Starting shard arm=$arm epoch=$epoch probe=$probe shard=$shard_index/$shard_count gpu=$gpu log=$shard_log"
  EVAL_LOCK_HELD=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
      --split val \
      --dataset-json "$dataset" \
      --gpu 0 \
      --model-dir "$model_dir" \
      --embeddings "$EMBEDDINGS" \
      --output-dir "$eval_dir/predictions" \
      --output-type mask \
      --threshold 0.5 \
      --preprocessed-cache-dir "$CACHE_ROOT" \
      --num-shards "$shard_count" \
      --shard-index "$shard_index" \
      --status-json "$status_json" \
      >"$shard_log" 2>&1
}

run_eval_sharded() {
  local arm="$1" epoch="$2" probe="$3"
  shift 3
  local gpus=("$@")
  local dataset eval_dir run_dir checkpoint model_dir expected existing lock eval_log
  dataset="$(dataset_for_probe "$probe")"
  run_dir="$GROUP_DIR/$arm"
  eval_dir="$run_dir/eval_epoch$(printf '%03d' "$epoch")_$probe"
  checkpoint="$run_dir/checkpoints/checkpoint_update_$(printf '%06d' "$(epoch_update "$epoch")").pth"
  model_dir="$run_dir/model_epoch$(printf '%03d' "$epoch")"
  expected="$(expected_case_count "$dataset")"
  existing="$(prediction_count "$eval_dir")"

  if eval_complete "$eval_dir" "$dataset"; then
    echo "Already complete: $arm epoch=$epoch $probe"
    return 0
  fi
  checkpoint_stable "$checkpoint" || { echo "Checkpoint not stable: $checkpoint" >&2; return 1; }
  lock="$(acquire_lock "$eval_dir")" || return 0
  materialize_model "$checkpoint" "$model_dir"
  mkdir -p "$eval_dir/logs" "$eval_dir/predictions"
  eval_log="$eval_dir/logs/recovery_${TIMESTAMP}_eval.log"
  echo "Recovery eval start: arm=$arm epoch=$epoch probe=$probe gpus=${gpus[*]} existing_predictions=$existing/$expected"
  {
    echo "start arm=$arm epoch=$epoch probe=$probe gpus=${gpus[*]} existing_predictions=$existing/$expected"
    pids=()
    for shard in "${!gpus[@]}"; do
      run_inference_shard "$arm" "$epoch" "$probe" "${gpus[$shard]}" "${#gpus[@]}" "$shard" &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    if [[ "$status" != "0" ]]; then
      echo "one or more inference shards failed"
      exit 1
    fi
    echo "prediction_count=$(prediction_count "$eval_dir") expected=$expected"
    EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset" \
      EVAL_LABEL="Experiment 006 $arm epoch $epoch fixed $probe threshold 0.5" NUM_WORKERS=8 \
      bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$eval_dir" val
    echo "done arm=$arm epoch=$epoch probe=$probe"
  } >"$eval_log" 2>&1 || {
    rm -f "$lock"
    echo "Recovery eval failed: arm=$arm epoch=$epoch probe=$probe log=$eval_log" >&2
    return 1
  }
  rm -f "$lock"
  summarize_partial_results
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Exp006 recovery eval dry run for $GROUP_DIR"
  for arm in $VAL200_ARMS; do
    dataset="$(dataset_for_probe val200)"
    eval_dir="$GROUP_DIR/$arm/eval_epoch100_val200"
    if eval_complete "$eval_dir" "$dataset"; then status=complete; else status="pending predictions=$(prediction_count "$eval_dir")/$(expected_case_count "$dataset")"; fi
    echo "$arm epoch=100 val200 $status"
  done
  for epoch in $VAL20_EPOCHS; do
    for arm in $VAL20_ARMS; do
      dataset="$(dataset_for_probe val20)"
      eval_dir="$GROUP_DIR/$arm/eval_epoch$(printf '%03d' "$epoch")_val20"
      if eval_complete "$eval_dir" "$dataset"; then status=complete; else status="pending predictions=$(prediction_count "$eval_dir")/$(expected_case_count "$dataset")"; fi
      echo "$arm epoch=$epoch val20 $status"
    done
  done
  exit 0
fi

echo "Experiment 006 recovery eval started for $GROUP_DIR"
echo "Priority: epoch-100 val200 first, then missing val20 intervals."
echo "GPUs per eval: $RECOVERY_GPUS"
for arm in $VAL200_ARMS; do
  run_eval_sharded "$arm" 100 val200 $RECOVERY_GPUS
done
for epoch in $VAL20_EPOCHS; do
  for arm in $VAL20_ARMS; do
    run_eval_sharded "$arm" "$epoch" val20 $RECOVERY_GPUS
  done
done
touch "$GROUP_DIR/.recovery_eval_complete"
echo "Experiment 006 recovery eval complete."
