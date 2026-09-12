#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
command="${1:-dry-run}"
if (($#)); then shift; fi
dry_run=0
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" ]]; then dry_run=1; fi
done
if [[ "$command" == "dry-run" || "$dry_run" == "1" ]]; then
  exec python "$repo_root/scripts/rexgroundingct/run_027_multicategory_cache.py" "$command" --dry-run "$@"
fi
gpu_args=()
allow_args=()
if [[ "$command" == "orchestrate" || "$command" == "export" ]]; then
  if [[ "${START_GPU_WORK:-0}" != "1" ]]; then
    echo 'GPU export requires START_GPU_WORK=1 and --share-gpus' >&2
    exit 2
  fi
  gpu_args=(--gpus all)
  allow_args=(--allow-gpu)
fi
if [[ "$command" == "worker" ]]; then
  echo 'Use orchestrate/export or import-validation; workers belong to their coordinator.' >&2
  exit 2
fi
image_id="$(docker image inspect --format '{{.Id}}' rexgroundingct-voxtell:cu126)"
detach_args=(--rm)
if [[ "${DETACH:-0}" == "1" ]]; then
  detach_args=(--detach --name "${MULTICATEGORY_CONTAINER_NAME:-rex027_cache_2bcde_v1}")
fi
exec docker run "${detach_args[@]}" "${gpu_args[@]}" --ipc=host --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -v "$repo_root:$repo_root:ro" -v /data/hengjie:/data/hengjie:ro -v /mnt/shengdata1:/mnt/shengdata1 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home \
  -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e "MULTICATEGORY_IMAGE_ID=$image_id" -e "START_GPU_WORK=${START_GPU_WORK:-0}" \
  -e "PYTHONPATH=$repo_root/scripts/rexgroundingct:$repo_root/external/VoxTell" \
  "$image_id" python "$repo_root/scripts/rexgroundingct/run_027_multicategory_cache.py" \
  "$command" "${allow_args[@]}" "$@"
