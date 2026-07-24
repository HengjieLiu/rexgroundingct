#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation}"
RUN_GROUP="${RUN_GROUP:-exp003_full_20260723T075256Z}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONTAINER_NAME="${CONTAINER_NAME:-rex003_v123_e100_multiscale_val20_${TIMESTAMP}}"
DETACH="${DETACH:-1}"
RUN_LOG="${RUN_LOG:-$EXP_DIR/logs/v123_epoch100_multiscale_val20_${RUN_GROUP}_${TIMESTAMP}.log}"

mkdir -p "$EXP_DIR/logs"

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
  -e EXP_DIR="$EXP_DIR"
  -e RUN_GROUP="$RUN_GROUP"
  -e RUN_LOG="$RUN_LOG"
)

forward_env=(
  GROUP_DIR
  VARIANT
  SEED
  MODEL_DIR
  EMBEDDINGS
  VAL20_JSON
  OUTPUT_ROOT
  SCALES
  THRESHOLDS
  RUN_MULTISCALE
  RUN_NATIVE_PROB
  RUN_EVAL
  OVERWRITE_PROBABILITIES
  OVERWRITE_THRESHOLDS
  NUM_WORKERS
  NATIVE_GPU
)

for key in "${forward_env[@]}"; do
  if [[ -n "${!key:-}" ]]; then
    docker_args+=(-e "$key=${!key}")
  fi
done

if [[ "$DETACH" == "1" ]]; then
  docker_args=(docker run -d --rm "${docker_args[@]:3}")
fi

"${docker_args[@]}" "$IMAGE" \
  bash -lc 'bash /workspace/scripts/rexgroundingct/run_003_v123_epoch100_multiscale_val20.sh 2>&1 | tee "$RUN_LOG"'

if [[ "$DETACH" == "1" ]]; then
  echo "Started detached container: $CONTAINER_NAME"
  echo "Logs: docker logs -f $CONTAINER_NAME"
  echo "Runtime log: $RUN_LOG"
fi
