#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval}"
INTERVAL="${INTERVAL:-600}"

echo "Polling MICCAI validation CT readiness every ${INTERVAL}s."
python "$REPO_ROOT/scripts/rexgroundingct/poll_ct_subset.py" \
  --splits val \
  --watch \
  --interval "$INTERVAL" \
  --json "$EXP_DIR/config/val_ct_readiness.json"

echo "Validation CT data is ready. Starting experiment 001 in Docker."
docker run --rm \
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
  -e EXP_DIR="$EXP_DIR" \
  "$IMAGE" \
  bash /workspace/scripts/rexgroundingct/run_001_voxtell_val_eval.sh
