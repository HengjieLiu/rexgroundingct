#!/usr/bin/env bash
# Explicit, bounded diagnostic. No old-arm resume, GPU poller or full-run launch.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
command="${1:-prepare}"
if (($#)); then shift; fi
image_id="$(docker image inspect --format '{{.Id}}' rexgroundingct-voxtell:cu126)"
gpu_args=()
allow_args=()
if [[ "$command" == "run" ]]; then
  if [[ "${START_GPU_WORK:-0}" != "1" ]]; then
    echo 'Set START_GPU_WORK=1 for the authorized diagnostic run.' >&2
    exit 2
  fi
  gpu_args=(--gpus "device=${DIAGNOSTIC_GPU:-0}")
  allow_args=(--allow-gpu)
fi
exec docker run --rm "${gpu_args[@]}" --ipc=host --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -v "$repo_root:$repo_root:ro" -v /data/hengjie:/data/hengjie:ro \
  -v /mnt/shengdata1:/mnt/shengdata1 \
  -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/matplotlib \
  -e "DIAGNOSTIC_IMAGE_ID=$image_id" -e "START_GPU_WORK=${START_GPU_WORK:-0}" \
  -e "PYTHONPATH=$repo_root/scripts/rexgroundingct:$repo_root/external/VoxTell" \
  "$image_id" python "$repo_root/scripts/rexgroundingct/fit_027_deletion.py" \
  "$command" "${allow_args[@]}" "$@"
