#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONTAINER_NAME="${CONTAINER_NAME:-rex005_global_local_$TIMESTAMP}"
DETACH="${DETACH:-1}"
MAX_HOURS="${MAX_HOURS:-24}"

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
for key in EXP_DIR EXP004_DIR EXP003_DIR SOURCE_2MM_CACHE SOURCE_SCHEDULE GLOBAL_CACHE \
  STAGE2_MODEL EMBEDDINGS VAL20_JSON VAL200_JSON RUN_GROUP GROUP_DIR CACHE_WORKERS \
  WAIT_SECONDS POLL_SECONDS GPU_MEMORY_LIMIT_MIB; do
  if [[ -n "${!key:-}" ]]; then docker_args+=(-e "$key=${!key}"); fi
done
if [[ "$DETACH" == 1 ]]; then
  docker_args=(docker run -d --rm "${docker_args[@]:3}")
fi
"${docker_args[@]}" "$IMAGE" \
  timeout --signal=TERM --kill-after=300s "${MAX_HOURS}h" \
  bash /workspace/scripts/rexgroundingct/run_005_global_proposal_cascade.sh
if [[ "$DETACH" == 1 ]]; then
  echo "Started detached container: $CONTAINER_NAME (hard limit ${MAX_HOURS}h)"
  echo "Logs: docker logs -f $CONTAINER_NAME"
fi
