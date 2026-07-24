#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val}"
GPU="${GPU:-0}"
SEED="${SEED:-20260723}"
EPOCHS="${EPOCHS:-5}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR_SCHEDULE="${LR_SCHEDULE:-fixed}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
PROBE_STEPS="${PROBE_STEPS:-2}"
RUN_SAMPLE_TEST="${RUN_SAMPLE_TEST:-1}"
RUN_VAL="${RUN_VAL:-1}"
VAL_PROBE_SIZE="${VAL_PROBE_SIZE:-${VAL_LIMIT:-20}}"
VAL_PROBE_JSON="${VAL_PROBE_JSON:-/workspace/configs/evaluation/rexgroundingct_val${VAL_PROBE_SIZE}_seed${SEED}.json}"
VAL_PROBE_MANIFEST="${VAL_PROBE_MANIFEST:-/workspace/configs/evaluation/rexgroundingct_val${VAL_PROBE_SIZE}_seed${SEED}.manifest.json}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$EXP_DIR/runs/single_gpu_zscore192_$TIMESTAMP}"
CONFIG_SNAPSHOT="$EXP_DIR/config/002_voxtell_text_ft_miccai_train_val.json"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/checkpoints" "$EXP_DIR/reports" "$EXP_DIR/eval" "$EXP_DIR/config" "$RUN_DIR/logs"

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment 002_voxtell_text_ft_miccai_train_val \
  --exp-dir "$EXP_DIR" \
  --overwrite

python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val \
  --json "$EXP_DIR/config/train_val_ct_readiness.json"

if [[ ! -f "$EMBEDDINGS" ]]; then
  echo "No complete embedding bank found at $EMBEDDINGS; starting precompute."
  GPU="$GPU" OUTPUT="$EMBEDDINGS" \
    bash /workspace/scripts/rexgroundingct/run_002_precompute_text_embeddings.sh
fi

if [[ ! -f "$VAL_PROBE_JSON" ]]; then
  python /workspace/scripts/rexgroundingct/prepare_fixed_probe_subset.py \
    --split val \
    --seed "$SEED" \
    --size "$VAL_PROBE_SIZE" \
    --output-json "$VAL_PROBE_JSON" \
    --manifest-json "$VAL_PROBE_MANIFEST"
fi

if [[ "$RUN_SAMPLE_TEST" == "1" ]]; then
  python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    --mode sample-test \
    --config "$CONFIG_SNAPSHOT" \
    --exp-dir "$EXP_DIR" \
    --run-dir "$RUN_DIR" \
    --gpu "$GPU" \
    --seed "$SEED" \
    --sample-test-steps 8 \
    2>&1 | tee "$RUN_DIR/logs/sample_test.log"
fi

python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
  --mode batch-probe \
  --config "$CONFIG_SNAPSHOT" \
  --exp-dir "$EXP_DIR" \
  --run-dir "$RUN_DIR" \
  --gpu "$GPU" \
  --embeddings "$EMBEDDINGS" \
  --seed "$SEED" \
  --min-batch-size "$MIN_BATCH_SIZE" \
  --max-batch-size "$MAX_BATCH_SIZE" \
  --probe-steps "$PROBE_STEPS" \
  --grad-accum "$GRAD_ACCUM" \
  --allow-experimental-train \
  2>&1 | tee "$RUN_DIR/logs/batch_probe.log"

PROBE_JSON="$RUN_DIR/reports/batch_probe.json"
if [[ -z "${BATCH_SIZE:-}" ]]; then
  BATCH_SELECTION="${BATCH_SELECTION:-recommended}"
  BATCH_SIZE="$(python - "$PROBE_JSON" "$BATCH_SELECTION" <<'PY'
import json
import sys
path = sys.argv[1]
selection = sys.argv[2]
data = json.load(open(path))
if selection == "largest":
    batch_size = data.get("largest_successful_batch_size")
else:
    batch_size = data.get("recommended_batch_size") or data.get("largest_successful_batch_size")
if batch_size is None:
    raise SystemExit(f"No successful batch size in {path}")
print(batch_size)
PY
)"
fi
echo "Training with batch size: $BATCH_SIZE"

TRAIN_SCHEDULE_EVENTS="${TRAIN_SCHEDULE_EVENTS:-$((EPOCHS * STEPS_PER_EPOCH * BATCH_SIZE * GRAD_ACCUM))}"
TRAIN_SCHEDULE="${TRAIN_SCHEDULE:-$EXP_DIR/config/train_schedule_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${BATCH_SIZE}.jsonl}"
TRAIN_SCHEDULE_MANIFEST="${TRAIN_SCHEDULE_MANIFEST:-$EXP_DIR/config/train_schedule_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${BATCH_SIZE}.manifest.json}"
if [[ ! -f "$TRAIN_SCHEDULE" ]]; then
  python /workspace/scripts/rexgroundingct/prepare_training_schedule.py \
    --seed "$SEED" \
    --events "$TRAIN_SCHEDULE_EVENTS" \
    --output-jsonl "$TRAIN_SCHEDULE" \
    --manifest-json "$TRAIN_SCHEDULE_MANIFEST"
fi

python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
  --mode train \
  --config "$CONFIG_SNAPSHOT" \
  --exp-dir "$EXP_DIR" \
  --run-dir "$RUN_DIR" \
  --gpu "$GPU" \
  --embeddings "$EMBEDDINGS" \
  --seed "$SEED" \
  --sample-schedule "$TRAIN_SCHEDULE" \
  --epochs "$EPOCHS" \
  --steps-per-epoch "$STEPS_PER_EPOCH" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --lr-schedule "$LR_SCHEDULE" \
  --checkpoint-every-updates "$STEPS_PER_EPOCH" \
  --allow-experimental-train \
  2>&1 | tee "$RUN_DIR/logs/train.log"

if [[ "$RUN_VAL" == "1" ]]; then
  python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split val \
    --dataset-json "$VAL_PROBE_JSON" \
    --gpu "$GPU" \
    --model-dir "$RUN_DIR/model" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$RUN_DIR/predictions" \
    --overwrite \
    2>&1 | tee "$RUN_DIR/logs/val${VAL_PROBE_SIZE}_seed${SEED}_inference.log"

  GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$VAL_PROBE_JSON" \
    EVAL_LABEL="Experiment 002 single-GPU z-score 192 baseline fixed val${VAL_PROBE_SIZE} seed ${SEED} quick/global eval" \
    bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$RUN_DIR" val \
    2>&1 | tee "$RUN_DIR/logs/val${VAL_PROBE_SIZE}_seed${SEED}_quick_eval.log"
fi

echo "Experiment 002 run dir: $RUN_DIR"
echo "Batch probe report: $RUN_DIR/reports/batch_probe.md"
echo "Training report: $RUN_DIR/reports/training_report.md"
