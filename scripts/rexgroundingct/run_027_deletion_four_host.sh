#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
command="${1:-orchestrate}"
if (($#)); then shift; fi
if [[ "$command" == "orchestrate" && "${START_GPU_WORK:-0}" != "1" ]]; then
  exec python "$repo_root/scripts/rexgroundingct/run_027_deletion_four.py" orchestrate --dry-run "$@"
fi
image_id="$(docker image inspect --format '{{.Id}}' rexgroundingct-voxtell:cu126)"
gpu_args=()
allow_args=()
detach_args=(--rm)
if [[ "$command" == "orchestrate" ]]; then
  gpu_args=(--gpus all)
  allow_args=(--allow-gpu)
fi
if [[ "${DETACH:-0}" == "1" ]]; then
  detach_args=(--detach --name "${DELETION_CONTAINER_NAME:-rex027_deletion_four_20ep}")
fi
exec docker run "${detach_args[@]}" "${gpu_args[@]}" --ipc=host --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -v "$repo_root:$repo_root:ro" -v /data/hengjie:/data/hengjie:ro -v /mnt/shengdata1:/mnt/shengdata1 \
  -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/deletion027_mpl \
  -e "DELETION027_IMAGE_ID=$image_id" -e "START_GPU_WORK=${START_GPU_WORK:-0}" \
  -e "PYTHONPATH=$repo_root/scripts/rexgroundingct:$repo_root/external/VoxTell" \
  "$image_id" python "$repo_root/scripts/rexgroundingct/run_027_deletion_four.py" \
  "$command" "${allow_args[@]}" "$@"
