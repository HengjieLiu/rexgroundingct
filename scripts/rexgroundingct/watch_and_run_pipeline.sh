#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
EXP001_DIR="${EXP001_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval}"
EXP002_DIR="${EXP002_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val}"
INTERVAL_VAL="${INTERVAL_VAL:-600}"
INTERVAL_TRAIN_VAL="${INTERVAL_TRAIN_VAL:-600}"
NUM_GPUS="${NUM_GPUS:-4}"
RUN_FULL_FT="${RUN_FULL_FT:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"

docker_run_common=(
  docker run --rm
  --gpus all
  --ipc=host
  --shm-size=32g
  --user "$(id -u):$(id -g)"
  -v "$REPO_ROOT:/workspace"
  -v /data/hengjie:/data/hengjie
  -v /mnt/shengdata1:/mnt/shengdata1
  -e HOME=/tmp
  -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home
  -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub
  -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet
  "$IMAGE"
)

echo "Stage 001 gate: polling MICCAI validation CT readiness every ${INTERVAL_VAL}s."
python "$REPO_ROOT/scripts/rexgroundingct/poll_ct_subset.py" \
  --splits val \
  --watch \
  --interval "$INTERVAL_VAL" \
  --json "$EXP001_DIR/config/val_ct_readiness.json"

echo "Validation CT data is ready. Starting experiment 001 checkpoint evaluation."
"${docker_run_common[@]}" \
  env EXP_DIR="$EXP001_DIR" NUM_SHARDS="$NUM_GPUS" \
  bash /workspace/scripts/rexgroundingct/run_001_voxtell_val_eval.sh

echo "Experiment 001 completed successfully. Stage 002 gate: polling train+val CT readiness every ${INTERVAL_TRAIN_VAL}s."
python "$REPO_ROOT/scripts/rexgroundingct/poll_ct_subset.py" \
  --splits train val \
  --watch \
  --interval "$INTERVAL_TRAIN_VAL" \
  --json "$EXP002_DIR/config/train_val_ct_readiness.json"

echo "Train+val CT data is ready. Starting experiment 002 fine-tuning pipeline."
"${docker_run_common[@]}" \
  env EXP_DIR="$EXP002_DIR" NUM_GPUS="$NUM_GPUS" RUN_FULL_FT="$RUN_FULL_FT" \
    MAX_ITERATIONS="$MAX_ITERATIONS" CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  bash /workspace/scripts/rexgroundingct/run_002_voxtell_text_ft_pipeline.sh
