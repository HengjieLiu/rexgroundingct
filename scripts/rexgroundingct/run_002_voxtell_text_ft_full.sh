#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val}"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
LOG_FILE="$EXP_DIR/logs/text_ft_full_$(date -u +%Y%m%dT%H%M%SZ).log"
CONFIG_SNAPSHOT="$EXP_DIR/config/002_voxtell_text_ft_miccai_train_val.json"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/checkpoints" "$EXP_DIR/reports" "$EXP_DIR/eval" "$EXP_DIR/config"
python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment 002_voxtell_text_ft_miccai_train_val \
  --exp-dir "$EXP_DIR"

python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val \
  --json "$EXP_DIR/config/train_val_ct_readiness.json"

EMBEDDING_ARGS=()
if [[ -f "$EMBEDDINGS" ]]; then
  EMBEDDING_ARGS=(--embeddings "$EMBEDDINGS")
else
  echo "No precomputed embedding bank found at $EMBEDDINGS; prompts will be embedded on demand."
fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
  --config "$CONFIG_SNAPSHOT" \
  --exp-dir "$EXP_DIR" \
  --distributed \
  --allow-experimental-train \
  --max-iterations "$MAX_ITERATIONS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  "${EMBEDDING_ARGS[@]}" \
  "$@" \
  2>&1 | tee "$LOG_FILE"

echo "Full fine-tuning log: $LOG_FILE"
