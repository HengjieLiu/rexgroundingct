#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
IMAGE="${IMAGE:-rexgroundingct-totalsegmentator:2.16.0-cu126}"
EXP_ID="010_voxtell_public_anatomy_prior_fusion"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/totalsegmentator_thorax_v1}"
PHASE_A_CACHE_ROOT="${PHASE_A_CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/totalsegmentator_total_fast_3mm_v2_16_0}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/mnt/shengdata1/hengjie/models/totalsegmentator/2.16.0}"
PLAN="${PLAN:-$EXP_DIR/config/val200_totalsegmentator_case_plan.jsonl}"
VAL_JSON="${VAL_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
EXPECTED_VAL_SHA256="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FREE_STABILITY_CHECKS="${FREE_STABILITY_CHECKS:-2}"
NUM_SHARDS="${NUM_SHARDS:-4}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
MODE="${1:-status}"

TOTAL_ROI_SUBSET=(
  lung_upper_lobe_left
  lung_lower_lobe_left
  lung_upper_lobe_right
  lung_middle_lobe_right
  lung_lower_lobe_right
  trachea
  esophagus
  heart
  aorta
  pulmonary_vein
  brachiocephalic_trunk
  subclavian_artery_right
  subclavian_artery_left
  common_carotid_artery_right
  common_carotid_artery_left
  brachiocephalic_vein_left
  brachiocephalic_vein_right
  atrial_appendage_left
  superior_vena_cava
  inferior_vena_cava
  vertebrae_T12
  vertebrae_T11
  vertebrae_T10
  vertebrae_T9
  vertebrae_T8
  vertebrae_T7
  vertebrae_T6
  vertebrae_T5
  vertebrae_T4
  vertebrae_T3
  vertebrae_T2
  vertebrae_T1
  vertebrae_C7
  spinal_cord
  rib_left_1
  rib_left_2
  rib_left_3
  rib_left_4
  rib_left_5
  rib_left_6
  rib_left_7
  rib_left_8
  rib_left_9
  rib_left_10
  rib_left_11
  rib_left_12
  rib_right_1
  rib_right_2
  rib_right_3
  rib_right_4
  rib_right_5
  rib_right_6
  rib_right_7
  rib_right_8
  rib_right_9
  rib_right_10
  rib_right_11
  rib_right_12
  sternum
  costal_cartilages
  clavicula_left
  clavicula_right
  scapula_left
  scapula_right
  liver
  spleen
  stomach
  adrenal_gland_right
  adrenal_gland_left
)

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
      --expected-val-sha256 "$EXPECTED_VAL_SHA256" \
      --cache-root "$PHASE_A_CACHE_ROOT" \
      --exp-dir "$EXP_DIR"
}

validate_roi_subset() {
  [[ -f "$PLAN" ]] || prepare_plan
  docker run --rm \
    "${common_mounts[@]}" \
    "$IMAGE" \
    python - "$EXPECTED_VAL_SHA256" "$VAL_JSON" "${TOTAL_ROI_SUBSET[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

expected_hash = sys.argv[1]
val_json = Path(sys.argv[2])
roi_subset = sys.argv[3:]
observed_hash = hashlib.sha256(val_json.read_bytes()).hexdigest()
if observed_hash != expected_hash:
    raise SystemExit(
        f"val JSON hash mismatch: expected={expected_hash} observed={observed_hash}"
    )
classes = json.loads(
    subprocess.check_output(
        ["totalseg_info", "--classes", "-ta", "total", "--json"],
        text=True,
    )
)
known = set(classes.values())
missing = [name for name in roi_subset if name not in known]
if missing:
    raise SystemExit(f"Unknown total ROI names: {missing}")
print(
    json.dumps(
        {
            "val_json": str(val_json),
            "val_json_sha256": observed_hash,
            "total_roi_count": len(roi_subset),
            "total_roi_subset": roi_subset,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
}

download_weights() {
  docker run --rm \
    "${common_mounts[@]}" \
    -e TOTALSEG_HOME_DIR="$WEIGHTS_ROOT" \
    "$IMAGE" \
    bash -lc '
      set -euo pipefail
      mkdir -p "$TOTALSEG_HOME_DIR"
      for task in total lung_vessels body; do
        echo "Downloading/checking TotalSegmentator weights for ${task}"
        totalseg_download_weights -t "$task"
      done
      echo "Downloading/checking TotalSegmentator weights for trunk_cavities task id 343"
      python - <<"PY"
from totalsegmentator.libs import download_pretrained_weights

download_pretrained_weights(343)
PY
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
    "download_tasks": ["total", "lung_vessels", "trunk_cavities:343", "body"],
    "root": str(root),
    "files": files,
    "total_bytes": sum(item["bytes"] for item in files),
}
manifest = root / "weights_manifest_thorax_v1.json"
temporary = root / f".{manifest.name}.{os.getpid()}.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, manifest)
complete = root / ".weights_complete_thorax_v1"
complete.write_text(payload["created_at_utc"] + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY'
}

worker_common_args() {
  local shard_index="$1"
  printf '%q ' \
    --plan "$PLAN" \
    --cache-root "$CACHE_ROOT" \
    --shard-index "$shard_index" \
    --num-shards "$NUM_SHARDS" \
    --require-device gpu \
    --wait-free-mib "$MIN_FREE_MIB" \
    --poll-seconds "$POLL_SECONDS" \
    --free-stability-checks "$FREE_STABILITY_CHECKS" \
    --wait-before-each-case \
    --continue-on-error
}

run_smoke() {
  [[ -f "$PLAN" ]] || prepare_plan
  validate_roi_subset
  docker run --rm \
    --gpus device=0 \
    --ipc=host \
    --shm-size=16g \
    "${common_mounts[@]}" \
    -e TOTALSEG_HOME_DIR="$EXP_DIR/runtime/totalseg_home_thorax_smoke" \
    -e TOTALSEG_WEIGHTS_PATH="$WEIGHTS_ROOT/nnunet/results" \
    "$IMAGE" \
    bash -lc "
      set -euo pipefail
      python /workspace/scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
        $(worker_common_args 0) --max-cases 1 --overwrite \
        --task total --output-name total_roi_labels.nii.gz \
        --roi-subset ${TOTAL_ROI_SUBSET[*]}
      python /workspace/scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
        $(worker_common_args 0) --max-cases 1 --overwrite \
        --task lung_vessels --output-name lung_vessels_labels.nii.gz
      python /workspace/scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
        $(worker_common_args 0) --max-cases 1 --overwrite \
        --task trunk_cavities --output-name trunk_cavities_labels.nii.gz
      python /workspace/scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
        $(worker_common_args 0) --max-cases 1 --overwrite \
        --task body --output-name body_labels.nii.gz
    "
}

launch_workers() {
  [[ -f "$PLAN" ]] || prepare_plan
  validate_roi_subset
  for gpu in $(seq 0 $((NUM_SHARDS - 1))); do
    name="rex010_totalseg_thorax_val200_gpu${gpu}"
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
      -e TOTALSEG_HOME_DIR="$EXP_DIR/runtime/totalseg_home_thorax_gpu${gpu}" \
      -e TOTALSEG_WEIGHTS_PATH="$WEIGHTS_ROOT/nnunet/results" \
      "$IMAGE" \
      bash -lc "
        set -euo pipefail
        python /workspace/scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
          $(worker_common_args "$gpu") \
          --task total --output-name total_roi_labels.nii.gz \
          --roi-subset ${TOTAL_ROI_SUBSET[*]}
        python /workspace/scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
          $(worker_common_args "$gpu") \
          --task lung_vessels --output-name lung_vessels_labels.nii.gz
        python /workspace/scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
          $(worker_common_args "$gpu") \
          --task trunk_cavities --output-name trunk_cavities_labels.nii.gz
        python /workspace/scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
          $(worker_common_args "$gpu") \
          --task body --output-name body_labels.nii.gz
      "
  done
}

launch_finalizer() {
  local name="rex010_totalseg_thorax_val200_finalizer"
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    echo "$name already exists; leaving it untouched"
    return
  fi
  docker run -d \
    --name "$name" \
    "${common_mounts[@]}" \
    "$IMAGE" \
    python /workspace/scripts/rexgroundingct/summarize_010_totalsegmentator_thorax_cache.py \
      --plan "$PLAN" \
      --cache-root "$CACHE_ROOT" \
      --phase-a-cache-root "$PHASE_A_CACHE_ROOT" \
      --exp-dir "$EXP_DIR" \
      --watch \
      --interval 300
}

run_summary() {
  docker run --rm \
    "${common_mounts[@]}" \
    "$IMAGE" \
    python /workspace/scripts/rexgroundingct/summarize_010_totalsegmentator_thorax_cache.py \
      --plan "$PLAN" \
      --cache-root "$CACHE_ROOT" \
      --phase-a-cache-root "$PHASE_A_CACHE_ROOT" \
      --exp-dir "$EXP_DIR"
}

show_status() {
  echo "cache_root=$CACHE_ROOT"
  for prefix in total_roi lung_vessels trunk_cavities body; do
    local complete=0 failed=0 locked=0
    if [[ -d "$CACHE_ROOT/cases" ]]; then
      complete="$(find "$CACHE_ROOT/cases" -mindepth 2 -maxdepth 2 -name ".complete_${prefix}" -type f | wc -l)"
      failed="$(find "$CACHE_ROOT/cases" -mindepth 2 -maxdepth 2 -name "${prefix}_failure.json" -type f | wc -l)"
      locked="$(find "$CACHE_ROOT/cases" -mindepth 2 -maxdepth 2 -name ".case_${prefix}.lock" -type d | wc -l)"
    fi
    echo "${prefix}: complete=${complete}/200 failed=${failed} locked=${locked}"
  done
  docker ps -a \
    --filter name=rex010_totalseg_thorax \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
}

case "$MODE" in
  prepare)
    prepare_plan
    ;;
  validate)
    validate_roi_subset
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
    validate_roi_subset
    download_weights
    run_smoke
    launch_workers
    launch_finalizer
    ;;
  *)
    echo "Usage: $0 {prepare|validate|weights|smoke|full|summarize|status|all}" >&2
    exit 2
    ;;
esac
