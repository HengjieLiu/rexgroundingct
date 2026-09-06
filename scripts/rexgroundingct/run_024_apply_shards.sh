#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR="${1:?source directory}"
OUTPUT_DIR="${2:?output directory}"
POLICY="${3:?unchanged|whole_lung_exact|whole_lung|fine}"
SPLIT="${4:-val}"
IMAGE="${IMAGE:-rexgroundingct-totalsegmentator:2.16.0-cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit}"
NUM_SHARDS="${NUM_SHARDS:-8}"
CASES=200
if [[ "$SPLIT" == "test" ]]; then CASES=300; fi
mkdir -p "$OUTPUT_DIR" "$RUNTIME_ROOT/logs"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  start=$((CASES * shard / NUM_SHARDS)); end=$((CASES * (shard + 1) / NUM_SHARDS))
  docker run --rm --name "exp024_apply_${SPLIT}_${POLICY}_${shard}" --ipc=host --shm-size=8g \
    --user "$(id -u):$(id -g)" -v "$REPO_ROOT:/workspace" -v /data/hengjie:/data/hengjie -v /mnt/shengdata1:/mnt/shengdata1 \
    -e HOME=/tmp "$IMAGE" bash -lc "PYTHONPATH=/workspace/scripts/rexgroundingct python /workspace/scripts/rexgroundingct/run_024_test_inference_anatomy.py apply --split '$SPLIT' --source '$SOURCE_DIR' --output '$OUTPUT_DIR' --policy '$POLICY' --start '$start' --end '$end'" \
    >"$RUNTIME_ROOT/logs/apply_${SPLIT}_${POLICY}_${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
