#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
SEED="${SEED:-20260723}"
EPOCHS="${EPOCHS:-5}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR_SCHEDULE="${LR_SCHEDULE:-fixed}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
RUN_PARALLEL="${RUN_PARALLEL:-1}"
RUN_VAL="${RUN_VAL:-0}"
VAL_PROBE_SIZE="${VAL_PROBE_SIZE:-${VAL_LIMIT:-20}}"
VAL_PROBE_JSON="${VAL_PROBE_JSON:-/workspace/configs/evaluation/rexgroundingct_val${VAL_PROBE_SIZE}_seed${SEED}.json}"
VAL_PROBE_MANIFEST="${VAL_PROBE_MANIFEST:-/workspace/configs/evaluation/rexgroundingct_val${VAL_PROBE_SIZE}_seed${SEED}.manifest.json}"
ARM_SLEEP_SECONDS="${ARM_SLEEP_SECONDS:-0}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/five_epoch_batch_ddp_comparison_$TIMESTAMP}"
CONFIG_SNAPSHOT="$EXP_DIR/config/002_voxtell_text_ft_miccai_train_val.json"
TRAIN_SCHEDULE_EVENTS="${TRAIN_SCHEDULE_EVENTS:-$((EPOCHS * STEPS_PER_EPOCH * 2 * GRAD_ACCUM))}"
TRAIN_SCHEDULE="${TRAIN_SCHEDULE:-$EXP_DIR/config/train_schedule_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb2.jsonl}"
TRAIN_SCHEDULE_MANIFEST="${TRAIN_SCHEDULE_MANIFEST:-$EXP_DIR/config/train_schedule_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb2.manifest.json}"

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$GROUP_DIR"

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment 002_voxtell_text_ft_miccai_train_val \
  --exp-dir "$EXP_DIR" \
  --overwrite

python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val \
  --json "$EXP_DIR/config/train_val_ct_readiness.json"

if [[ ! -f "$EMBEDDINGS" ]]; then
  echo "No embedding bank found at $EMBEDDINGS; starting precompute on GPU0."
  GPU=0 OUTPUT="$EMBEDDINGS" \
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

if [[ ! -f "$TRAIN_SCHEDULE" ]]; then
  python /workspace/scripts/rexgroundingct/prepare_training_schedule.py \
    --seed "$SEED" \
    --events "$TRAIN_SCHEDULE_EVENTS" \
    --output-jsonl "$TRAIN_SCHEDULE" \
    --manifest-json "$TRAIN_SCHEDULE_MANIFEST"
fi

run_optional_val() {
  local name="$1"
  local run_dir="$2"
  local visible_gpu="$3"

  if [[ "$RUN_VAL" != "1" ]]; then
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$visible_gpu" python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split val \
    --dataset-json "$VAL_PROBE_JSON" \
    --gpu 0 \
    --model-dir "$run_dir/model" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$run_dir/predictions" \
    --overwrite \
    2>&1 | tee "$run_dir/logs/val${VAL_PROBE_SIZE}_seed${SEED}_inference.log"

  GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$VAL_PROBE_JSON" \
    EVAL_LABEL="Experiment 002 ${name} fixed val${VAL_PROBE_SIZE} seed ${SEED} quick/global eval" \
    bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$run_dir" val \
    2>&1 | tee "$run_dir/logs/val${VAL_PROBE_SIZE}_seed${SEED}_quick_eval.log"
}

run_single_gpu_arm() {
  local name="$1"
  local visible_gpu="$2"
  local batch_size="$3"
  local run_dir="$GROUP_DIR/$name"
  mkdir -p "$run_dir/logs"

  echo "Starting $name: visible_gpu=$visible_gpu batch_size=$batch_size run_dir=$run_dir"
  CUDA_VISIBLE_DEVICES="$visible_gpu" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    --mode train \
    --config "$CONFIG_SNAPSHOT" \
    --exp-dir "$EXP_DIR" \
    --run-dir "$run_dir" \
    --gpu 0 \
    --embeddings "$EMBEDDINGS" \
    --seed "$SEED" \
    --sample-schedule "$TRAIN_SCHEDULE" \
    --epochs "$EPOCHS" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --batch-size "$batch_size" \
    --grad-accum "$GRAD_ACCUM" \
    --lr-schedule "$LR_SCHEDULE" \
    --checkpoint-every-updates "$CHECKPOINT_EVERY" \
    --allow-experimental-train \
    2>&1 | tee "$run_dir/logs/train.log"

  run_optional_val "$name" "$run_dir" "$visible_gpu"
  echo "Completed $name"
}

run_ddp_arm() {
  local name="ddp2_b1_gpu23"
  local run_dir="$GROUP_DIR/$name"
  mkdir -p "$run_dir/logs"

  echo "Starting $name: visible_gpus=2,3 per_gpu_batch=1 global_batch=2 run_dir=$run_dir"
  CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    --mode train \
    --config "$CONFIG_SNAPSHOT" \
    --exp-dir "$EXP_DIR" \
    --run-dir "$run_dir" \
    --distributed \
    --embeddings "$EMBEDDINGS" \
    --seed "$SEED" \
    --sample-schedule "$TRAIN_SCHEDULE" \
    --epochs "$EPOCHS" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --batch-size 1 \
    --grad-accum "$GRAD_ACCUM" \
    --lr-schedule "$LR_SCHEDULE" \
    --checkpoint-every-updates "$CHECKPOINT_EVERY" \
    --allow-experimental-train \
    2>&1 | tee "$run_dir/logs/train.log"

  run_optional_val "$name" "$run_dir" 2
  echo "Completed $name"
}

if [[ "$RUN_PARALLEL" == "1" ]]; then
  pids=()
  names=()

  run_single_gpu_arm b1_gpu0 0 1 &
  pids+=("$!")
  names+=("b1_gpu0")
  if [[ "$ARM_SLEEP_SECONDS" != "0" ]]; then
    sleep "$ARM_SLEEP_SECONDS"
  fi

  run_single_gpu_arm b2_gpu1 1 2 &
  pids+=("$!")
  names+=("b2_gpu1")
  if [[ "$ARM_SLEEP_SECONDS" != "0" ]]; then
    sleep "$ARM_SLEEP_SECONDS"
  fi

  run_ddp_arm &
  pids+=("$!")
  names+=("ddp2_b1_gpu23")

  status=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "Arm failed: ${names[$index]}" >&2
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    exit "$status"
  fi
else
  run_single_gpu_arm b1_gpu0 0 1
  if [[ "$ARM_SLEEP_SECONDS" != "0" ]]; then
    sleep "$ARM_SLEEP_SECONDS"
  fi
  run_single_gpu_arm b2_gpu1 1 2
  if [[ "$ARM_SLEEP_SECONDS" != "0" ]]; then
    sleep "$ARM_SLEEP_SECONDS"
  fi
  run_ddp_arm
fi

echo "Five-epoch comparison group: $GROUP_DIR"
find "$GROUP_DIR" -maxdepth 3 -type f \( -name training_report.md -o -name training_metrics.json \) -print | sort
