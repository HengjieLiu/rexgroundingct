#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONTAINER_NAME="${CONTAINER_NAME:-rex011_ct_norm_$TIMESTAMP}"
DETACH="${DETACH:-1}"
START_TRAINING="${START_TRAINING:-0}"

docker_args=(
  docker run
  --rm
  --name "$CONTAINER_NAME"
  --ipc=host
  --shm-size=32g
  --user "$(id -u):$(id -g)"
  -v "$REPO_ROOT:/workspace:ro"
  -v /data/hengjie:/data/hengjie:ro
  -v /mnt/shengdata1:/mnt/shengdata1
  -e HOME=/tmp
  -e PYTHONDONTWRITEBYTECODE=1
  -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home
  -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub
  -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet
  -e START_TRAINING="$START_TRAINING"
)

if [[ "$START_TRAINING" == "1" ]]; then
  docker_args+=(--gpus '"device=0,1,2"')
fi

for key in EXP_DIR EXP003_DIR CACHE_BASE SEED STEPS_PER_EPOCH EPOCHS BATCH_SIZE \
  GRAD_ACCUM RUN_GROUP GROUP_DIR EMBEDDINGS VAL20_JSON VAL200_JSON \
  SOURCE_SCHEDULE CHECKPOINT_UPDATES RUN_SAMPLE_TEST ARM_SLEEP_SECONDS \
  SEGMENT_EPOCHS MIN_FREE_GPU_GIB POLL_SECONDS STABILITY_SECONDS LOCK_STALE_SECONDS \
  VOXTELL_MODEL; do
  if [[ -n "${!key:-}" ]]; then
    docker_args+=(-e "$key=${!key}")
  fi
done

if [[ "$DETACH" == "1" ]]; then
  docker_args=(docker run -d --rm "${docker_args[@]:3}")
fi

"${docker_args[@]}" "$IMAGE" \
  bash /workspace/scripts/rexgroundingct/run_011_ct_normalization_ablation.sh

if [[ "$DETACH" == "1" ]]; then
  echo "Started detached container: $CONTAINER_NAME"
  echo "Logs: docker logs -f $CONTAINER_NAME"
fi
