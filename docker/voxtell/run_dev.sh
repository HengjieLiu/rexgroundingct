#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --user "$(id -u):$(id -g)" \
  -v "$REPO_ROOT:/workspace" \
  -v /data/hengjie:/data/hengjie \
  -v /mnt/shengdata1:/mnt/shengdata1 \
  -e HOME=/tmp \
  -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home \
  -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub \
  -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet \
  "$IMAGE" "$@"
