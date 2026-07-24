#!/usr/bin/env bash
set -euo pipefail

EXP_ID="004_voxtell_v123_native_vs_2mm_global_context_ft"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
GROUP_DIR="${GROUP_DIR:?GROUP_DIR is required}"
CACHE_ROOT="${CACHE_ROOT:-$EXP_DIR/cache/crop_zscore_2mm_v1}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:?SOURCE_MODEL_DIR is required}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed20260723.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABILITY_SECONDS="${STABILITY_SECONDS:-5}"

arms=(native192_cont100 iso2mm_global192_cont100)

arm_gpu() {
  case "$1" in
    native192_cont100) echo 0 ;;
    iso2mm_global192_cont100) echo 1 ;;
    *) return 1 ;;
  esac
}

eval_complete() {
  local eval_dir="$1"
  local dataset_json="$2"
  local require_probabilities="${3:-0}"
  python - "$eval_dir" "$dataset_json" "$require_probabilities" <<'PY'
import json, sys
from pathlib import Path
root, dataset = map(Path, sys.argv[1:3])
require_probabilities = sys.argv[3] == "1"
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
if require_probabilities:
    probabilities = root / "probabilities"
    if any(not (probabilities / e["name"]).is_file() for e in entries):
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

materialize_model() {
  local checkpoint="$1"
  local model_dir="$2"
  mkdir -p "$model_dir/fold_0"
  cp "$SOURCE_MODEL_DIR/plans.json" "$model_dir/plans.json"
  local target="$model_dir/fold_0/checkpoint_final.pth"
  if [[ ! -f "$target" || "$(stat -c %s "$target")" != "$(stat -c %s "$checkpoint")" ]]; then
    local tmp="$target.tmp.$$"
    rm -f "$tmp"
    ln "$checkpoint" "$tmp"
    mv -f "$tmp" "$target"
  fi
}

acquire_lock() {
  local eval_dir="$1"
  local lock="$eval_dir/.eval.lock"
  mkdir -p "$eval_dir"
  if (set -o noclobber; printf '{"pid":%s,"created":"%s"}\n' "$$" "$(date -u +%FT%TZ)" > "$lock") 2>/dev/null; then
    echo "$lock"
    return 0
  fi
  return 1
}

run_inference_shard() {
  local arm="$1" gpu="$2" model_dir="$3" dataset="$4" eval_dir="$5" shards="$6" shard="$7" save_probs="$8"
  local common=(
    --dataset-json "$dataset"
    --gpu 0
    --model-dir "$model_dir"
    --embeddings "$EMBEDDINGS"
    --output-dir "$eval_dir/predictions"
    --num-shards "$shards"
    --shard-index "$shard"
    --threshold 0.5
  )
  local probability_args=()
  if [[ "$save_probs" == "1" ]]; then
    probability_args=(--probability-output-dir "$eval_dir/probabilities")
  fi
  if [[ "$arm" == "native192_cont100" ]]; then
    EVAL_LOCK_HELD=1 CUDA_VISIBLE_DEVICES="$gpu" python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
      --split val --output-type mask "${common[@]}" "${probability_args[@]}"
  else
    CUDA_VISIBLE_DEVICES="$gpu" python /workspace/scripts/rexgroundingct/run_voxtell_2mm_val_inference.py \
      --cache-root "$CACHE_ROOT" "${common[@]}" "${probability_args[@]}"
  fi
}

run_eval() {
  local arm="$1" epoch="$2" probe="$3"
  shift 3
  local gpus=("$@")
  local dataset="$VAL20_JSON"
  [[ "$probe" == "val200" ]] && dataset="$VAL200_JSON"
  local run_dir="$GROUP_DIR/$arm"
  local eval_dir="$run_dir/eval_epoch$(printf '%03d' "$epoch")_$probe"
  local require_probabilities=0
  [[ "$probe" == "val200" ]] && require_probabilities=1
  if eval_complete "$eval_dir" "$dataset" "$require_probabilities"; then
    echo "Already complete: $eval_dir"
    return 0
  fi
  local lock
  lock="$(acquire_lock "$eval_dir")" || return 0
  local checkpoint="$run_dir/checkpoints/checkpoint_update_$(printf '%06d' "$((epoch * 100))").pth"
  local model_dir="$run_dir/model_epoch$(printf '%03d' "$epoch")"
  materialize_model "$checkpoint" "$model_dir"
  mkdir -p "$eval_dir/logs" "$eval_dir/predictions"
  local log="$eval_dir/logs/coordinator_$(date -u +%Y%m%dT%H%M%SZ).log"
  local save_probs=0
  [[ "$probe" == "val200" ]] && save_probs=1
  if (
    set -euo pipefail
    echo "start arm=$arm epoch=$epoch probe=$probe gpus=${gpus[*]}"
    pids=()
    for shard in "${!gpus[@]}"; do
      run_inference_shard "$arm" "${gpus[$shard]}" "$model_dir" "$dataset" "$eval_dir" "${#gpus[@]}" "$shard" "$save_probs" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset" \
      EVAL_LABEL="Experiment 004 $arm epoch $epoch fixed $probe threshold 0.5" NUM_WORKERS=8 \
      bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$eval_dir" val
    echo "done arm=$arm epoch=$epoch probe=$probe"
  ) >"$log" 2>&1; then
    rm -f "$lock"
  else
    rm -f "$lock"
    echo "Evaluation failed: arm=$arm epoch=$epoch probe=$probe log=$log" >&2
    return 1
  fi
}

val20_worker() {
  local arm="$1" gpu
  gpu="$(arm_gpu "$arm")"
  for epoch in 20 40 60 80 100; do
    local checkpoint="$GROUP_DIR/$arm/checkpoints/checkpoint_update_$(printf '%06d' "$((epoch * 100))").pth"
    until checkpoint_stable "$checkpoint"; do sleep "$POLL_SECONDS"; done
    run_eval "$arm" "$epoch" val20 "$gpu"
  done
}

echo "Experiment 004 eval coordinator started for $GROUP_DIR"
val20_worker native192_cont100 &
native_worker=$!
val20_worker iso2mm_global192_cont100 &
iso_worker=$!
wait "$native_worker"
wait "$iso_worker"

until [[ -f "$GROUP_DIR/native192_cont100/.train_complete" && -f "$GROUP_DIR/iso2mm_global192_cont100/.train_complete" ]]; do
  sleep "$POLL_SECONDS"
done

run_eval native192_cont100 100 val200 0 2 &
native_final=$!
run_eval iso2mm_global192_cont100 100 val200 1 3 &
iso_final=$!
wait "$native_final"
wait "$iso_final"

BASELINE_EVAL="${BASELINE_EVAL:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v123_opt_poscrop_emptyloss/full_100ep/eval_epoch100_val200/eval/val_quick_global_eval.json}"
python /workspace/scripts/rexgroundingct/summarize_004_results.py \
  --dataset-json "$VAL200_JSON" \
  --input "exp003_v123_epoch100=$BASELINE_EVAL" \
  --input "native192_cont100=$GROUP_DIR/native192_cont100/eval_epoch100_val200/eval/val_quick_global_eval.json" \
  --input "iso2mm_global192_cont100=$GROUP_DIR/iso2mm_global192_cont100/eval_epoch100_val200/eval/val_quick_global_eval.json" \
  --output-json "$EXP_DIR/reports/paired_comparison.json" \
  --output-md "$EXP_DIR/reports/paired_comparison_report.md"
touch "$GROUP_DIR/.experiment_complete"
echo "Experiment 004 evaluation complete."
