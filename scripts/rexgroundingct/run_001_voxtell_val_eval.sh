#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SPLIT="${SPLIT:-val}"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/predictions" "$EXP_DIR/eval" "$EXP_DIR/reports" "$EXP_DIR/config"
python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment 001_voxtell_v1_1_miccai200_val_eval \
  --exp-dir "$EXP_DIR"

python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits "$SPLIT" \
  --json "$EXP_DIR/config/${SPLIT}_ct_readiness.json"

pids=()
for gpu in $(seq 0 $((NUM_SHARDS - 1))); do
  status_json="$EXP_DIR/logs/${SPLIT}_shard_${gpu}_status.json"
  log_file="$EXP_DIR/logs/${SPLIT}_shard_${gpu}.log"
  echo "Launching shard $gpu/$NUM_SHARDS on GPU $gpu"
  python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split "$SPLIT" \
    --output-dir "$EXP_DIR/predictions" \
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
  echo "At least one VoxTell inference shard failed. Check $EXP_DIR/logs."
  exit 1
fi

GLOBAL_ONLY=1 bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$EXP_DIR" "$SPLIT"
