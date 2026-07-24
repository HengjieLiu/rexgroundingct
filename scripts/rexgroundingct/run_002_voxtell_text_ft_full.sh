#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val}"
GPU="${GPU:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
SEED="${SEED:-20260723}"
EPOCHS="${EPOCHS:-50}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR_SCHEDULE="${LR_SCHEDULE:-fixed}"
USE_SAMPLE_SCHEDULE="${USE_SAMPLE_SCHEDULE:-1}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$EXP_DIR/runs/full_${NUM_GPUS}gpu_$TIMESTAMP}"
LOG_FILE="$RUN_DIR/logs/text_ft_full.log"
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

SCHEDULE_ARGS=()
if [[ "$USE_SAMPLE_SCHEDULE" == "1" ]]; then
  GLOBAL_BATCH="$((BATCH_SIZE * NUM_GPUS))"
  TRAIN_SCHEDULE_EVENTS="${TRAIN_SCHEDULE_EVENTS:-$((EPOCHS * STEPS_PER_EPOCH * GLOBAL_BATCH * GRAD_ACCUM))}"
  TRAIN_SCHEDULE="${TRAIN_SCHEDULE:-$EXP_DIR/config/train_schedule_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${GLOBAL_BATCH}.jsonl}"
  TRAIN_SCHEDULE_MANIFEST="${TRAIN_SCHEDULE_MANIFEST:-$EXP_DIR/config/train_schedule_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${GLOBAL_BATCH}.manifest.json}"
  if [[ ! -f "$TRAIN_SCHEDULE" ]]; then
    python /workspace/scripts/rexgroundingct/prepare_training_schedule.py \
      --seed "$SEED" \
      --events "$TRAIN_SCHEDULE_EVENTS" \
      --output-jsonl "$TRAIN_SCHEDULE" \
      --manifest-json "$TRAIN_SCHEDULE_MANIFEST"
  fi
  SCHEDULE_ARGS=(--sample-schedule "$TRAIN_SCHEDULE")
fi

TRAIN_ARGS=(
  /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py
  --mode train
  --config "$CONFIG_SNAPSHOT"
  --exp-dir "$EXP_DIR"
  --run-dir "$RUN_DIR"
  --allow-experimental-train
  --seed "$SEED"
  --epochs "$EPOCHS"
  --steps-per-epoch "$STEPS_PER_EPOCH"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --lr-schedule "$LR_SCHEDULE"
  --checkpoint-every-updates "$CHECKPOINT_EVERY"
  "${EMBEDDING_ARGS[@]}"
  "${SCHEDULE_ARGS[@]}"
  "$@"
)

if [[ "$NUM_GPUS" == "1" ]]; then
  python "${TRAIN_ARGS[@]}" --gpu "$GPU" 2>&1 | tee "$LOG_FILE"
else
  torchrun --standalone --nproc_per_node="$NUM_GPUS" "${TRAIN_ARGS[@]}" --distributed 2>&1 | tee "$LOG_FILE"
fi

echo "Full fine-tuning log: $LOG_FILE"
