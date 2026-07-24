#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
CONTAINER_NAME="${CONTAINER_NAME:-rex003_rescue_ablation_$(date -u +%Y%m%dT%H%M%SZ)}"
DETACH="${DETACH:-0}"

docker_args=(
  docker run
  --rm
  --name "$CONTAINER_NAME"
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
)

if [[ "$DETACH" == "1" ]]; then
  docker_args=(docker run -d --rm "${docker_args[@]:3}")
fi

forward_env=(
  EXP_DIR
  EXP002_DIR
  SEED
  STEPS_PER_EPOCH
  BATCH_SIZE
  GRAD_ACCUM
  SHORT_EPOCHS
  FULL_EPOCHS
  RUN_PARALLEL
  RUN_SAMPLE_TEST
  SAMPLE_TEST_STEPS
  RUN_TINY_TRAIN_SMOKE
  RUN_SHORT
  RUN_FULL
  RUN_VAL
  SCHEDULE_NUM_WORKERS
  FORCE_REGENERATE_SCHEDULES
  ARM_SLEEP_SECONDS
  RUN_GROUP
  GROUP_DIR
  VARIANTS
  ENCODER_LR
  DECODER_LR
  WARMUP_UPDATES
  POLY_POWER
  MOMENTUM
  WEIGHT_DECAY
  CLIP_GRAD_NORM
)

for key in "${forward_env[@]}"; do
  if [[ -n "${!key:-}" ]]; then
    docker_args+=(-e "$key=${!key}")
  fi
done

"${docker_args[@]}" "$IMAGE" \
  bash /workspace/scripts/rexgroundingct/run_003_voxtell_rescue_ablation.sh

if [[ "$DETACH" == "1" ]]; then
  echo "Started detached container: $CONTAINER_NAME"
  echo "Logs: docker logs -f $CONTAINER_NAME"
fi
