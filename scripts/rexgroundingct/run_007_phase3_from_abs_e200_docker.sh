#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUP="${RUN_GROUP:-exp007_phase3_from_abs_e200_$TIMESTAMP}"
CONTAINER_NAME="${CONTAINER_NAME:-rex007_phase3_from_abs_e200_$TIMESTAMP}"
DETACH="${DETACH:-1}"

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched}"
SOURCE_RUN_GROUP="${CONTINUATION_SOURCE_RUN_GROUP:-exp007_cont100_from_ddp100_20260730T062051Z}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$EXP_DIR/runs/$SOURCE_RUN_GROUP/ddp_bs4/checkpoints/checkpoint_update_010000.pth}"

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

for key in EXP_DIR RUN_GROUP CONTAINER_NAME CONTINUATION_SOURCE_RUN_GROUP INIT_CHECKPOINT \
  INIT_CHECKPOINT_EXPECTED_UPDATE EXPECTED_INIT_CHECKPOINT_SHA ABSOLUTE_EPOCH_OFFSET \
  EXTENDED_SCHEDULE_EVENTS SCHEDULE_START_EVENT PRIOR_EXTENDED_SCHEDULE_EVENTS \
  CHECKPOINT_UPDATES SEGMENT_EPOCHS RUN_SAMPLE_TEST RUN_DDP_SMOKE RUN_INFERENCE_SMOKE \
  RUN_FULL RUN_SMOKE_ONLY MASTER_PORT GPU_LIST CACHE_ROOT SEED STEPS_PER_EPOCH EPOCHS \
  WORLD_SIZE BATCH_SIZE GRAD_ACCUM SCHEDULE_NUM_WORKERS LOCK_STALE_SECONDS STABILITY_SECONDS; do
  if [[ -n "${!key:-}" ]]; then
    docker_args+=( -e "$key=${!key}" )
  fi
done

if [[ "$DETACH" == "1" ]]; then
  docker_args=(docker run -d --rm "${docker_args[@]:3}")
fi

"${docker_args[@]}" "$IMAGE" bash /workspace/scripts/rexgroundingct/run_007_phase3_from_abs_e200.sh

if [[ "$DETACH" == "1" ]]; then
  echo "Started detached container: $CONTAINER_NAME"
  echo "Logs: docker logs -f $CONTAINER_NAME"
else
  echo "Completed container: $CONTAINER_NAME"
fi
