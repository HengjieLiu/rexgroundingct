#!/usr/bin/env bash
# Explicit launcher only. It never waits for GPUs or starts jobs automatically.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
image="${IMAGE:-rexgroundingct-voxtell:cu126}"
command="${1:-benchmark}"
if (($#)); then shift; fi
case "$command" in
  prepare|report)
    exec python "$repo_root/scripts/rexgroundingct/run_027_residual_refinement.py" "$command" "$@" ;;
  benchmark|orchestrate) ;;
  *) echo "Use prepare, report, benchmark or orchestrate" >&2; exit 2 ;;
esac
if [[ "${START_GPU_WORK:-0}" != "1" ]]; then
  exec python "$repo_root/scripts/rexgroundingct/run_027_residual_refinement.py" "$command" --dry-run "$@"
fi
resolved_image="$(docker image inspect --format '{{.Id}}' "$image")"
exec docker run --rm --gpus all --ipc=host --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -v "$repo_root:$repo_root" -v /data/hengjie:/data/hengjie \
  -v /mnt/shengdata1:/mnt/shengdata1 \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e "EXP027_CONTAINER_IMAGE_ID=$resolved_image" \
  -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home \
  -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub \
  -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet \
  -e "PYTHONPATH=$repo_root/scripts/rexgroundingct:$repo_root/external/VoxTell" \
  "$resolved_image" python "$repo_root/scripts/rexgroundingct/run_027_residual_refinement.py" \
  "$command" --allow-gpu "$@"
