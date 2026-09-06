#!/usr/bin/env bash
set -euo pipefail

# Run one policy/model over one split using one container per GPU.  The policy
# (set directory) is completed before this launcher returns; sets themselves
# are therefore sequential while each set uses the available GPUs safely.
MODEL_DIR="${1:?model dir}"
CACHE_DIR="${2:?cache dir}"
OUTPUT_DIR="${3:?output dir}"
SPLIT="${4:-test}"
DATASET="${5:-/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json}"
IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit}"
NUM_SHARDS="${NUM_SHARDS:-4}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "$OUTPUT_DIR")")}" # keep per-candidate logs/status files distinct
LOCK_ENV=()
if [[ "$SPLIT" == "test" ]]; then
  # Test shards write independent native outputs and have no evaluator summary;
  # the validation-only evaluator lock would otherwise serialize them.
  LOCK_ENV=(-e EVAL_LOCK_DISABLE=1)
fi
mkdir -p "$OUTPUT_DIR" "$RUNTIME_ROOT/logs"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  name="exp024_${SPLIT}_$(basename "$OUTPUT_DIR")_shard${shard}"
  docker run --rm --name "$name" --gpus "device=${shard}" --ipc=host --shm-size=32g \
    --user "$(id -u):$(id -g)" \
    -v "$REPO_ROOT:/workspace" -v /data/hengjie:/data/hengjie -v /mnt/shengdata1:/mnt/shengdata1 \
    -e HOME=/tmp "${LOCK_ENV[@]}" -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home \
    -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub \
    -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet \
    "$IMAGE" bash -lc "PYTHONPATH=/workspace/scripts/rexgroundingct python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py --dataset-json '$DATASET' --split '$SPLIT' --model-dir '$MODEL_DIR' --ct-root /mnt/shengdata1/hengjie/datasets/rexgroundingct/ct --seg-dir /data/hengjie/datasets/rexgroundingct/segmentations --preprocessed-cache-dir '$CACHE_DIR' --output-dir '$OUTPUT_DIR' --output-type mask --threshold 0.5 --num-shards '$NUM_SHARDS' --shard-index '$shard' --status-json '$RUNTIME_ROOT/logs/$RUN_TAG.shard${shard}.status.json'" \
    >"$RUNTIME_ROOT/logs/$RUN_TAG.shard${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
