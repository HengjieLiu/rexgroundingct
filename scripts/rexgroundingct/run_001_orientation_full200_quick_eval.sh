#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval}"
RUN_DIR="${RUN_DIR:-$EXP_DIR/diagnostics/orientation_full200_quick_eval}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SPLIT="${SPLIT:-val}"
NUM_WORKERS="${NUM_WORKERS:-16}"

mkdir -p "$RUN_DIR/config" "$RUN_DIR/eval" "$RUN_DIR/logs" "$RUN_DIR/predictions" "$RUN_DIR/reports"

python - <<'PY'
import torch
print(
    {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    },
    flush=True,
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for VoxTell inference")
PY

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment 001_voxtell_v1_1_miccai200_val_eval \
  --exp-dir "$EXP_DIR"

python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits "$SPLIT" \
  --json "$RUN_DIR/config/${SPLIT}_ct_readiness.json"

python /workspace/scripts/rexgroundingct/prepare_eval_dataset_json.py \
  --split "$SPLIT" \
  --output-json "$RUN_DIR/eval/${SPLIT}200_as_test_dataset.json"

pids=()
for gpu in $(seq 0 $((NUM_SHARDS - 1))); do
  status_json="$RUN_DIR/logs/${SPLIT}_shard_${gpu}_status.json"
  log_file="$RUN_DIR/logs/${SPLIT}_shard_${gpu}.log"
  echo "Launching corrected-orientation shard $gpu/$NUM_SHARDS on GPU $gpu"
  python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split "$SPLIT" \
    --output-dir "$RUN_DIR/predictions" \
    --gpu "$gpu" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$gpu" \
    --status-json "$status_json" \
    > "$log_file" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one corrected-orientation inference shard failed. Check $RUN_DIR/logs."
  exit 1
fi

python /data/hengjie/datasets/rexgroundingct/rexrank_eval.py \
  --gt_dir /data/hengjie/datasets/rexgroundingct/segmentations \
  --pred_dir "$RUN_DIR/predictions" \
  --output_json "$RUN_DIR/eval/${SPLIT}_quick_global_eval.json" \
  --dataset_json "$RUN_DIR/eval/${SPLIT}200_as_test_dataset.json" \
  --num_workers "$NUM_WORKERS" \
  --global_only

python /workspace/scripts/rexgroundingct/summarize_quick_global_eval.py \
  --eval-json "$RUN_DIR/eval/${SPLIT}_quick_global_eval.json" \
  --output-md "$RUN_DIR/reports/${SPLIT}_quick_global_eval_report.md" \
  --output-json "$RUN_DIR/reports/${SPLIT}_quick_global_eval_summary.json" \
  --label "Experiment 001 corrected orientation MICCAI val quick eval"

echo "Corrected full-val quick report: $RUN_DIR/reports/${SPLIT}_quick_global_eval_report.md"
