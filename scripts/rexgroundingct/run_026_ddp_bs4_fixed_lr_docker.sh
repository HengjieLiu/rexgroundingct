#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONTAINER_NAME="${CONTAINER_NAME:-rex026_fixed_lr_ddp_bs4_$TIMESTAMP}"
DETACH="${DETACH:-1}"

docker_args=(
  docker run --rm --name "$CONTAINER_NAME" --gpus all --ipc=host --shm-size=32g
  --user "$(id -u):$(id -g)"
  -v "$REPO_ROOT:/workspace"
  -v /data/hengjie:/data/hengjie
  -v /mnt/shengdata1:/mnt/shengdata1
  -e HOME=/tmp
  -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home
  -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub
  -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet
)

for key in EXP_DIR EXP007_DIR CACHE_ROOT SEED STEPS_PER_EPOCH EPOCHS WORLD_SIZE \
  BATCH_SIZE GRAD_ACCUM RUN_GROUP GROUP_DIR EMBEDDINGS VAL200_JSON SOURCE_SCHEDULE \
  VAL200_EPOCHS CHECKPOINT_UPDATES RUN_SAMPLE_TEST RUN_DDP_SMOKE RUN_INFERENCE_SMOKE \
  RUN_STATIC_ONLY RUN_SMOKE_ONLY RUN_FULL LOCK_STALE_SECONDS STABILITY_SECONDS MASTER_PORT \
  GPU_LIST; do
  if [[ -n "${!key:-}" ]]; then
    docker_args+=( -e "$key=${!key}" )
  fi
done

if [[ "$DETACH" == "1" ]]; then
  docker_args=(docker run -d --rm "${docker_args[@]:3}")
fi

"${docker_args[@]}" "$IMAGE" bash /workspace/scripts/rexgroundingct/run_026_ddp_bs4_fixed_lr.sh

if [[ "$DETACH" == "1" ]]; then
  echo "Started detached container: $CONTAINER_NAME"
  echo "Logs: docker logs -f $CONTAINER_NAME"
fi
