#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val}"
GPU="${GPU:-0}"
SEED="${SEED:-20260723}"
LR_SCHEDULE="${LR_SCHEDULE:-fixed}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$EXP_DIR/runs/smoke_single_gpu_$TIMESTAMP}"
LOG_FILE="$RUN_DIR/logs/text_ft_smoke.log"
CONFIG_SNAPSHOT="$EXP_DIR/config/002_voxtell_text_ft_miccai_train_val.json"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/checkpoints" "$EXP_DIR/reports" "$EXP_DIR/eval" "$EXP_DIR/config" "$RUN_DIR/logs"
python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment 002_voxtell_text_ft_miccai_train_val \
  --exp-dir "$EXP_DIR" \
  --overwrite

python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val \
  --json "$EXP_DIR/config/train_val_ct_readiness.json"

EMBEDDING_ARGS=()
if [[ -f "$EMBEDDINGS" ]]; then
  EMBEDDING_ARGS=(--embeddings "$EMBEDDINGS")
else
  echo "No precomputed embedding bank found at $EMBEDDINGS; run run_002_precompute_text_embeddings.sh first."
  exit 2
fi

python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
  --mode train \
  --config "$CONFIG_SNAPSHOT" \
  --exp-dir "$EXP_DIR" \
  --run-dir "$RUN_DIR" \
  --gpu "$GPU" \
  --seed "$SEED" \
  --allow-experimental-train \
  --epochs "${EPOCHS:-1}" \
  --steps-per-epoch "${STEPS_PER_EPOCH:-5}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --lr-schedule "$LR_SCHEDULE" \
  --checkpoint-every-updates "${CHECKPOINT_EVERY:-5}" \
  "${EMBEDDING_ARGS[@]}" \
  "$@" \
  2>&1 | tee "$LOG_FILE"

echo "Smoke fine-tuning log: $LOG_FILE"
