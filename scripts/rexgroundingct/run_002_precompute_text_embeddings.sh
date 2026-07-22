#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val}"
GPU="${GPU:-0}"
OUTPUT="${OUTPUT:-$EXP_DIR/config/rex_text_embeddings.npz}"
LOG_FILE="$EXP_DIR/logs/precompute_text_embeddings_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/config"
python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment 002_voxtell_text_ft_miccai_train_val \
  --exp-dir "$EXP_DIR"

python /workspace/scripts/rexgroundingct/precompute_text_embeddings.py \
  --splits train val \
  --gpu "$GPU" \
  --output "$OUTPUT" \
  "$@" \
  2>&1 | tee "$LOG_FILE"

echo "Embedding bank: $OUTPUT"
echo "Precompute log: $LOG_FILE"
