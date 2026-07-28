#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
IMAGE="${IMAGE:-rexgroundingct-totalsegmentator:2.16.0-cu126}"
EXP_ID="010_voxtell_public_anatomy_prior_fusion"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/totalsegmentator_total_fast_3mm_v2_16_0}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/mnt/shengdata1/hengjie/models/totalsegmentator/2.16.0}"
PLAN="${PLAN:-$EXP_DIR/config/val200_totalsegmentator_case_plan.jsonl}"
VAL_JSON="${VAL_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
NUM_SHARDS="${NUM_SHARDS:-4}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
MODE="${1:-status}"

common_mounts=(
  --user "$HOST_UID:$HOST_GID"
  -e HOME=/tmp
  -v "$REPO_ROOT:/workspace:ro"
  -v /mnt/shengdata1/hengjie:/mnt/shengdata1/hengjie
  -v /data/hengjie:/data/hengjie:ro
)

prepare_plan() {
  docker run --rm \
    "${common_mounts[@]}" \
    "$IMAGE" \
    python /workspace/scripts/rexgroundingct/prepare_010_totalsegmentator_val_audit.py \
      --val-json "$VAL_JSON" \
      --cache-root "$CACHE_ROOT" \
      --exp-dir "$EXP_DIR"
}

download_weights() {
  docker run --rm \
    "${common_mounts[@]}" \
    -e TOTALSEG_HOME_DIR="$WEIGHTS_ROOT" \
    "$IMAGE" \
    bash -lc '
      mkdir -p "$TOTALSEG_HOME_DIR"
      totalseg_download_weights -t total_fast
      python - "$TOTALSEG_HOME_DIR" <<"PY"
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = []
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    files.append({
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
payload = {
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "totalsegmentator_version": "2.16.0",
    "download_task": "total_fast",
    "root": str(root),
    "files": files,
    "total_bytes": sum(item["bytes"] for item in files),
}
manifest = root / "weights_manifest_total_fast.json"
temporary = root / f".{manifest.name}.{os.getpid()}.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, manifest)
complete = root / ".weights_complete_total_fast"
complete.write_text(payload["created_at_utc"] + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY'
}

run_smoke() {
  [[ -f "$PLAN" ]] || prepare_plan
  docker run --rm \
    --gpus device=0 \
    --ipc=host \
    --shm-size=16g \
    "${common_mounts[@]}" \
    -e TOTALSEG_HOME_DIR="$EXP_DIR/runtime/totalseg_home_smoke" \
    -e TOTALSEG_WEIGHTS_PATH="$WEIGHTS_ROOT/nnunet/results" \
    "$IMAGE" \
    python /workspace/scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
      --plan "$PLAN" \
      --cache-root "$CACHE_ROOT" \
      --task total \
      --fast \
      --max-cases 1 \
      --overwrite \
      --require-device gpu \
      --wait-free-mib 12000 \
      --poll-seconds "$POLL_SECONDS"
}

launch_workers() {
  [[ -f "$PLAN" ]] || prepare_plan
  for gpu in $(seq 0 $((NUM_SHARDS - 1))); do
    name="rex010_totalseg_val200_gpu${gpu}"
    if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
      echo "$name already exists; leaving it untouched"
      continue
    fi
    docker run -d \
      --name "$name" \
      --gpus "device=$gpu" \
      --ipc=host \
      --shm-size=16g \
      "${common_mounts[@]}" \
      -e TOTALSEG_HOME_DIR="$EXP_DIR/runtime/totalseg_home_gpu${gpu}" \
      -e TOTALSEG_WEIGHTS_PATH="$WEIGHTS_ROOT/nnunet/results" \
      "$IMAGE" \
      python /workspace/scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
        --plan "$PLAN" \
        --cache-root "$CACHE_ROOT" \
        --task total \
        --fast \
        --shard-index "$gpu" \
        --num-shards "$NUM_SHARDS" \
        --require-device gpu \
        --wait-free-mib "$MIN_FREE_MIB" \
        --poll-seconds "$POLL_SECONDS" \
        --free-stability-checks 2 \
        --continue-on-error
  done
}

launch_finalizer() {
  local name="rex010_totalseg_val200_finalizer"
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    echo "$name already exists; leaving it untouched"
    return
  fi
  docker run -d \
    --name "$name" \
    "${common_mounts[@]}" \
    "$IMAGE" \
    python /workspace/scripts/rexgroundingct/summarize_010_totalsegmentator_val_audit.py \
      --plan "$PLAN" \
      --cache-root "$CACHE_ROOT" \
      --exp-dir "$EXP_DIR" \
      --watch \
      --interval 300
}

run_summary() {
  docker run --rm \
    "${common_mounts[@]}" \
    "$IMAGE" \
    python /workspace/scripts/rexgroundingct/summarize_010_totalsegmentator_val_audit.py \
      --plan "$PLAN" \
      --cache-root "$CACHE_ROOT" \
      --exp-dir "$EXP_DIR"
}

show_status() {
  local complete=0 failed=0
  if [[ -d "$CACHE_ROOT/cases" ]]; then
    complete="$(find "$CACHE_ROOT/cases" -mindepth 2 -maxdepth 2 -name .complete -type f | wc -l)"
    failed="$(find "$CACHE_ROOT/cases" -mindepth 2 -maxdepth 2 -name failure.json -type f | wc -l)"
  fi
  echo "cache_root=$CACHE_ROOT"
  echo "complete_cases=$complete/200 failed_cases=$failed"
  docker ps -a \
    --filter name=rex010_totalseg \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
}

case "$MODE" in
  prepare)
    prepare_plan
    ;;
  weights)
    download_weights
    ;;
  smoke)
    run_smoke
    ;;
  full)
    launch_workers
    launch_finalizer
    ;;
  summarize)
    run_summary
    ;;
  status)
    show_status
    ;;
  all)
    prepare_plan
    download_weights
    run_smoke
    launch_workers
    launch_finalizer
    ;;
  *)
    echo "Usage: $0 {prepare|weights|smoke|full|summarize|status|all}" >&2
    exit 2
    ;;
esac
