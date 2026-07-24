#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:?EXP_DIR required}"
GROUP_DIR="${GROUP_DIR:?GROUP_DIR required}"
GLOBAL_CACHE="${GLOBAL_CACHE:?GLOBAL_CACHE required}"
STAGE2_MODEL="${STAGE2_MODEL:?STAGE2_MODEL required}"
EMBEDDINGS="${EMBEDDINGS:?EMBEDDINGS required}"
VAL20_JSON="${VAL20_JSON:?VAL20_JSON required}"
VAL200_JSON="${VAL200_JSON:?VAL200_JSON required}"
EXP004_GROUP="${EXP004_GROUP:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft/runs/latest}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-15000}"

variant_gpu() {
  case "$1" in
    strict_inclusion) echo 0 ;;
    inclusion_margin16mm) echo 1 ;;
    *) return 1 ;;
  esac
}

exp004_arm() {
  case "$1" in
    strict_inclusion) echo native192_cont100 ;;
    inclusion_margin16mm) echo iso2mm_global192_cont100 ;;
    *) return 1 ;;
  esac
}

gpu_memory_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

exp004_eval_pending() {
  local variant="$1" arm epoch checkpoint summary
  arm="$(exp004_arm "$variant")"
  for epoch in 20 40 60 80 100; do
    checkpoint="$EXP004_GROUP/$arm/checkpoints/checkpoint_update_$(printf '%06d' "$((epoch * 100))").pth"
    summary="$EXP004_GROUP/$arm/eval_epoch$(printf '%03d' "$epoch")_val20/reports/val_quick_global_eval_summary.json"
    if [[ -f "$checkpoint" && ! -f "$summary" ]]; then
      return 0
    fi
  done
  return 1
}

wait_gpu_window() {
  local variant="$1" gpu="$2"
  while true; do
    local used
    used="$(gpu_memory_mib "$gpu")"
    if (( used < GPU_MEMORY_LIMIT_MIB )) && ! exp004_eval_pending "$variant"; then
      return 0
    fi
    echo "defer cascade variant=$variant gpu=$gpu memory=${used}MiB exp004_pending=$(exp004_eval_pending "$variant" && echo yes || echo no)"
    sleep "$POLL_SECONDS"
  done
}

checkpoint_stable() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  local first second
  first="$(stat -c '%s:%Y' "$path")"
  sleep 5
  second="$(stat -c '%s:%Y' "$path")"
  [[ "$first" == "$second" ]]
}

eval_complete() {
  local root="$1" dataset="$2"
  python - "$root" "$dataset" <<'PY'
import json,sys
from pathlib import Path
root,dataset=map(Path,sys.argv[1:])
summary=root/"reports"/"val_quick_global_eval_summary.json"
pred=root/"predictions"
if not summary.is_file(): raise SystemExit(1)
entries=json.loads(dataset.read_text())["test"]
data=json.loads(summary.read_text())
if data.get("total_cases") != len(entries): raise SystemExit(1)
if any(not (pred/e["name"]).is_file() for e in entries): raise SystemExit(1)
PY
}

run_cascade() {
  local variant="$1" step="$2" probe="$3"
  local gpu dataset run_root checkpoint
  gpu="$(variant_gpu "$variant")"
  dataset="$VAL20_JSON"
  [[ "$probe" == val200 ]] && dataset="$VAL200_JSON"
  run_root="$GROUP_DIR/$variant/cascade_step$(printf '%05d' "$step")_$probe"
  checkpoint="$GROUP_DIR/$variant/checkpoints/checkpoint_step_$(printf '%05d' "$step").pth"
  if eval_complete "$run_root" "$dataset"; then return 0; fi
  wait_gpu_window "$variant" "$gpu"
  mkdir -p "$run_root/logs"
  local probability_args=()
  [[ "$probe" == val200 ]] && probability_args=(--probability-output-dir "$run_root/probabilities")
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="$gpu" python /workspace/scripts/rexgroundingct/run_global_to_local_cascade.py \
      --dataset-json "$dataset" \
      --global-cache "$GLOBAL_CACHE" \
      --proposal-checkpoint "$checkpoint" \
      --stage2-model-dir "$STAGE2_MODEL" \
      --embeddings "$EMBEDDINGS" \
      --output-dir "$run_root/predictions" \
      --candidate-json "$run_root/candidates.json" \
      --gpu 0 --top-k 3 --nms-radius 8 --threshold 0.5 \
      "${probability_args[@]}"
    GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset" \
      EVAL_LABEL="Experiment 005 $variant proposal step $step $probe global-to-local cascade" \
      NUM_WORKERS=8 bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$run_root" val
  ) >"$run_root/logs/cascade.log" 2>&1
}

worker() {
  local variant="$1"
  for step in 1000 2500 5000; do
    local checkpoint="$GROUP_DIR/$variant/checkpoints/checkpoint_step_$(printf '%05d' "$step").pth"
    until checkpoint_stable "$checkpoint"; do
      if [[ -f "$GROUP_DIR/$variant/.train_complete" ]]; then
        echo "variant=$variant stopped before cascade checkpoint step=$step"
        return 0
      fi
      sleep "$POLL_SECONDS"
    done
    run_cascade "$variant" "$step" val20
  done
  until [[ -f "$GROUP_DIR/$variant/.train_complete" ]]; do sleep "$POLL_SECONDS"; done
  local gpu
  gpu="$(variant_gpu "$variant")"
  wait_gpu_window "$variant" "$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python /workspace/scripts/rexgroundingct/evaluate_global_proposal.py \
    --dataset-json "$VAL200_JSON" \
    --cache-root "$GLOBAL_CACHE" \
    --checkpoint "$GROUP_DIR/$variant/checkpoints/checkpoint_step_05000.pth" \
    --embeddings "$EMBEDDINGS" \
    --output-json "$GROUP_DIR/$variant/reports/val200_step_05000.json" \
    --output-md "$GROUP_DIR/$variant/reports/val200_step_05000.md" \
    --gpu 0
  run_cascade "$variant" 5000 val200
}

worker strict_inclusion &
pid0=$!
worker inclusion_margin16mm &
pid1=$!
wait "$pid0"
wait "$pid1"
python /workspace/scripts/rexgroundingct/summarize_005_results.py \
  --group-dir "$GROUP_DIR" \
  --output-json "$EXP_DIR/reports/final_comparison.json" \
  --output-md "$EXP_DIR/reports/final_comparison.md"
touch "$GROUP_DIR/.evaluation_complete"
