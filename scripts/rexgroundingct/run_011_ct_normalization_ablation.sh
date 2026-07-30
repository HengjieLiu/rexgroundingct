#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

EXP_ID="011_voxtell_v123_e4d4_ct_normalization_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
EXP003_DIR="${EXP003_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation}"
CACHE_BASE="${CACHE_BASE:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell}"
NATIVE_CACHE="$CACHE_BASE/crop_zscore_native_v1"
CLIPPED_ZSCORE_CACHE="$CACHE_BASE/crop_clip1024_zscore_native_v1"
CLIPPED_LINEAR_CACHE="$CACHE_BASE/crop_clip1024_linear_native_v1"
SEED="${SEED:-20260723}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUP="${RUN_GROUP:-exp011_ct_norm_e4d4_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
SOURCE_SCHEDULE="${SOURCE_SCHEDULE:-$EXP003_DIR/config/train_schedule_v123_opt_poscrop_emptyloss_seed${SEED}_100ep_100steps_gb1.jsonl}"
SCHEDULE="$EXP_DIR/config/train_schedule_v123_e4d4_ct_norm_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${BATCH_SIZE}.jsonl"
SCHEDULE_MANIFEST="${SCHEDULE%.jsonl}.manifest.json"
SOURCE_MODEL_DIR="$EXP_DIR/config/public_voxtell_v1_1_model"
HU_AUDIT_JSON="$EXP_DIR/analysis/hu_normalization_val200_audit.json"
EXP006_E5_D4_RUN="${EXP006_E5_D4_RUN:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/runs/latest/v123_cached_e5_d4}"
EXPECTED_SCHEDULE_SHA="f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776"
EXPECTED_NATIVE_CACHE_SHA="fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3"
EXPECTED_VAL20_SHA="31557d624c47ee8c06799bf89cceb862471b34cfd47065001d6092e32d0dab5d"
EXPECTED_VAL200_SHA="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-500,2000,4000,6000,8000,10000}"
RUN_SAMPLE_TEST="${RUN_SAMPLE_TEST:-1}"
START_TRAINING="${START_TRAINING:-0}"
ARM_SLEEP_SECONDS="${ARM_SLEEP_SECONDS:-10}"
SEGMENT_EPOCHS="${SEGMENT_EPOCHS:-5 20 40 60 80 100}"
MIN_FREE_GPU_GIB="${MIN_FREE_GPU_GIB:-32}"

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$EXP_DIR/reports" "$EXP_DIR/analysis"

sha256_path() {
  sha256sum "$1" | cut -d ' ' -f 1
}

require_sha() {
  local path="$1" expected="$2" label="$3"
  [[ -f "$path" ]] || { echo "Missing $label: $path" >&2; exit 1; }
  local actual
  actual="$(sha256_path "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA mismatch: expected=$expected actual=$actual path=$path" >&2
    exit 1
  fi
  echo "$label sha256=$actual"
}

config_cache_sha() {
  local preprocess_id="$1"
  python - "$CONFIG_SNAPSHOT" "$preprocess_id" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
preprocess_id = sys.argv[2]
if preprocess_id == "crop_zscore_native_v1":
    value = config["preprocessing"]["reference_cache"]["manifest_sha256"]
else:
    value = config["preprocessing"]["generated_caches"][preprocess_id]["manifest_sha256"]
if not value:
    raise SystemExit(
        f"Canonical config has no finalized manifest SHA for {preprocess_id}; "
        "finish cache generation and update the config before preparation."
    )
print(value)
PY
}

validate_cache() {
  local cache_root="$1" preprocess_id="$2" expected_sha="$3"
  [[ -f "$cache_root/.complete" ]] || {
    echo "Missing cache completion marker: $cache_root/.complete" >&2
    exit 1
  }
  require_sha "$cache_root/manifest.json" "$expected_sha" "$preprocess_id manifest"
  python - "$cache_root/manifest.json" "$preprocess_id" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
preprocess_id = sys.argv[2]
expected = {
    "preprocess_id": preprocess_id,
    "cases": 3192,
    "targets": 8068,
    "empty_targets": 0,
}
observed = {key: manifest.get(key) for key in expected}
if observed != expected:
    raise SystemExit(f"Cache audit mismatch: expected={expected} observed={observed}")
cross = manifest.get("cross_cache_audit")
if preprocess_id != "crop_zscore_native_v1":
    required_cross = {
        "geometry_or_target_mismatches": 0,
        "materialized_hu_header_failures": 0,
        "normalization_failures": 0,
    }
    observed_cross = {key: cross.get(key) for key in required_cross} if cross else {}
    if observed_cross != required_cross:
        raise SystemExit(
            f"Cross-cache audit mismatch: expected={required_cross} observed={observed_cross}"
        )
print(f"Cache accepted: {observed}")
PY
}

arm_cache() {
  case "$1" in
    v123_e4d4_zscore) echo "$NATIVE_CACHE" ;;
    v123_e4d4_clip1024_zscore) echo "$CLIPPED_ZSCORE_CACHE" ;;
    v123_e4d4_clip1024_linear) echo "$CLIPPED_LINEAR_CACHE" ;;
    *) return 1 ;;
  esac
}

arm_gpu() {
  case "$1" in
    v123_e4d4_zscore) echo 0 ;;
    v123_e4d4_clip1024_zscore) echo 1 ;;
    v123_e4d4_clip1024_linear) echo 2 ;;
    *) return 1 ;;
  esac
}

common_train_args() {
  local arm="$1"
  printf '%s\n' \
    --mode train \
    --experiment-id "$EXP_ID" \
    --config "$CONFIG_SNAPSHOT" \
    --exp-dir "$EXP_DIR" \
    --gpu 0 \
    --model-dir "$BASE_MODEL_DIR" \
    --embeddings "$EMBEDDINGS" \
    --seed "$SEED" \
    --patch-size 192 192 192 \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --encoder-lr 1e-4 \
    --decoder-lr 1e-4 \
    --lr-schedule poly \
    --poly-power 0.9 \
    --warmup-updates 100 \
    --momentum 0.99 \
    --weight-decay 3e-5 \
    --clip-grad-norm 12 \
    --require-positive-crop \
    --loss-mode empty_bce_only \
    --empty-target-loss-weight 0.5 \
    --voxel-bce-weighting \
    --bce-foreground-weight 1.0 \
    --bce-boundary-weight 1.5 \
    --bce-background-weight 0.5 \
    --bce-boundary-radius 10 \
    --deep-supervision-weights 1,0.5,0.25,0.125,0.0625 \
    --preprocessed-cache-dir "$(arm_cache "$arm")" \
    --sample-schedule "$SCHEDULE" \
    --allow-experimental-train
}

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment "$EXP_ID" --exp-dir "$EXP_DIR" --overwrite
python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val --json "$EXP_DIR/config/train_val_ct_readiness.json"

native_expected="$(config_cache_sha crop_zscore_native_v1)"
[[ "$native_expected" == "$EXPECTED_NATIVE_CACHE_SHA" ]] || {
  echo "Canonical native cache SHA differs from the locked exp006 reference" >&2
  exit 1
}
validate_cache "$NATIVE_CACHE" crop_zscore_native_v1 "$native_expected"
validate_cache \
  "$CLIPPED_ZSCORE_CACHE" \
  crop_clip1024_zscore_native_v1 \
  "$(config_cache_sha crop_clip1024_zscore_native_v1)"
validate_cache \
  "$CLIPPED_LINEAR_CACHE" \
  crop_clip1024_linear_native_v1 \
  "$(config_cache_sha crop_clip1024_linear_native_v1)"

require_sha "$SOURCE_SCHEDULE" "$EXPECTED_SCHEDULE_SHA" "exp003 v123 schedule"
require_sha "$VAL20_JSON" "$EXPECTED_VAL20_SHA" "val20 JSON"
require_sha "$VAL200_JSON" "$EXPECTED_VAL200_SHA" "val200 JSON"

python - "$HU_AUDIT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing HU audit: {path}")
audit = json.loads(path.read_text())
expected = {"cases": 200, "findings": 381, "gt_voxels": 32527800}
observed = {key: audit.get(key) for key in expected}
if observed != expected:
    raise SystemExit(f"HU audit mismatch: expected={expected} observed={observed}")
print(f"HU audit accepted: {observed}")
PY

cp "$SOURCE_SCHEDULE" "$SCHEDULE"
python - "$SCHEDULE" "$SCHEDULE_MANIFEST" "$SOURCE_SCHEDULE" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

schedule, manifest, source = map(Path, sys.argv[1:4])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

events = sum(1 for _ in schedule.open())
if events != 10000:
    raise SystemExit(f"Schedule must contain 10000 events, got {events}")
payload = {
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "purpose": "exp011 shared v123 e4/d4 CT-normalization schedule",
    "copied_from": str(source),
    "copied_from_sha256": sha256(source),
    "output_jsonl": str(schedule),
    "output_jsonl_sha256": sha256(schedule),
    "events": events,
    "seed": 20260723,
    "consumption": "all three arms consume events 0-9999 with batch1/no-DDP",
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [[ ! -f "$EMBEDDINGS" ]]; then
  cp "$EXP003_DIR/config/rex_text_embeddings.npz" "$EMBEDDINGS"
fi

BASE_MODEL_DIR="$(python -c 'from train_text_conditioned_voxtell import resolve_model_dir; print(resolve_model_dir(None))')"
[[ -f "$BASE_MODEL_DIR/plans.json" ]] || {
  echo "Missing VoxTell plans: $BASE_MODEL_DIR/plans.json" >&2
  exit 1
}
[[ -f "$BASE_MODEL_DIR/fold_0/checkpoint_final.pth" ]] || {
  echo "Missing VoxTell checkpoint: $BASE_MODEL_DIR/fold_0/checkpoint_final.pth" >&2
  exit 1
}
mkdir -p "$SOURCE_MODEL_DIR"
cp "$BASE_MODEL_DIR/plans.json" "$SOURCE_MODEL_DIR/plans.json"
python - "$BASE_MODEL_DIR" "$SOURCE_MODEL_DIR" "$EXP_DIR/config/public_voxtell_v1_1_provenance.json" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

base, source, output = map(Path, sys.argv[1:4])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

checkpoint = base / "fold_0" / "checkpoint_final.pth"
payload = {
    "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source": "public VoxTell v1.1 resolved by VoxTell download helper",
    "base_model_dir": str(base),
    "plans_path": str(base / "plans.json"),
    "plans_sha256": sha256(base / "plans.json"),
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "materialized_plans_dir": str(source),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

arms=(
  v123_e4d4_zscore
  v123_e4d4_clip1024_zscore
  v123_e4d4_clip1024_linear
)

if [[ "$RUN_SAMPLE_TEST" == "1" ]]; then
  for arm in "${arms[@]}"; do
    sample_dir="$EXP_DIR/preparation/sample_tests/$arm"
    mkdir -p "$sample_dir/logs"
    CUDA_VISIBLE_DEVICES="" python \
      /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
      --mode sample-test \
      --experiment-id "$EXP_ID" \
      --config "$CONFIG_SNAPSHOT" \
      --exp-dir "$EXP_DIR" \
      --run-dir "$sample_dir" \
      --gpu 0 \
      --seed "$SEED" \
      --patch-size 192 192 192 \
      --sample-schedule "$SCHEDULE" \
      --preprocessed-cache-dir "$(arm_cache "$arm")" \
      --require-positive-crop \
      --loss-mode empty_bce_only \
      --sample-test-steps 8 \
      2>&1 | tee "$sample_dir/logs/sample_test.log"
  done
  python - "$EXP_DIR/preparation" "${arms[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = sys.argv[2:]
reports = {
    arm: json.loads((root / "sample_tests" / arm / "reports" / "sample_test.json").read_text())
    for arm in arms
}
baseline_arm = arms[0]
baseline = reports[baseline_arm]
fields = ("samples", "patch_size", "sample_schedule", "sample_stats", "examples")
mismatches = []
for arm in arms[1:]:
    for field in fields:
        if reports[arm].get(field) != baseline.get(field):
            mismatches.append(
                {"baseline": baseline_arm, "candidate": arm, "field": field}
            )
comparison = {
    "status": "passed" if not mismatches else "failed",
    "arms": arms,
    "fields_compared": list(fields),
    "schedule_sha256": baseline["sample_schedule"]["sha256"],
    "events_checked": baseline["samples"],
    "mismatches": mismatches,
}
output = root / "sample_test_comparison.json"
output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
print(json.dumps(comparison, indent=2, sort_keys=True))
if mismatches:
    raise SystemExit("Sampler outputs differ across normalization arms")
PY
fi

commands_path="$EXP_DIR/config/prepared_training_commands.txt"
{
  echo "# Authorized synchronous milestone plan."
  echo "# Every arm stops at each target; val20 is a global barrier before resume."
  previous_update=""
  for epoch in $SEGMENT_EPOCHS; do
    echo
    echo "# Concurrent training segment to epoch $epoch."
    for arm in "${arms[@]}"; do
      gpu="$(arm_gpu "$arm")"
      run_dir="$GROUP_DIR/$arm"
      mapfile -t train_args < <(common_train_args "$arm")
      target_update="$((epoch * STEPS_PER_EPOCH))"
      printf 'CUDA_VISIBLE_DEVICES=%q python %q ' \
        "$gpu" \
        /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py
      printf '%q ' "${train_args[@]}"
      printf '%q ' \
        --run-dir "$run_dir" \
        --epochs "$EPOCHS" \
        --steps-per-epoch "$STEPS_PER_EPOCH" \
        --target-global-update "$target_update" \
        --checkpoint-every-updates 0 \
        --checkpoint-updates "$CHECKPOINT_UPDATES" \
        --latest-checkpoint-every-updates 500 \
        --no-materialize-final-model
      if [[ -n "$previous_update" ]]; then
        printf '%q ' \
          --resume-checkpoint \
          "$run_dir/checkpoints/checkpoint_update_$(printf '%06d' "$previous_update").pth"
      fi
      printf '\n'
    done
    echo "# Wait for all three commands above, then run concurrent fixed val20."
    for arm in "${arms[@]}"; do
      gpu="$(arm_gpu "$arm")"
      printf 'EVAL_ARM=%q EVAL_GPU=%q EVAL_EPOCH=%q EVAL_PROBE=val20 bash %q\n' \
        "$arm" "$gpu" "$epoch" \
        /workspace/scripts/rexgroundingct/run_011_eval_coordinator.sh
    done
    echo "# Wait for all three val20 summaries before the next training segment."
    previous_update="$target_update"
  done
  echo
  echo "# After epoch-100 val20, run concurrent fixed val200."
  for arm in "${arms[@]}"; do
    gpu="$(arm_gpu "$arm")"
    printf 'EVAL_ARM=%q EVAL_GPU=%q EVAL_EPOCH=100 EVAL_PROBE=val200 bash %q\n' \
      "$arm" "$gpu" /workspace/scripts/rexgroundingct/run_011_eval_coordinator.sh
  done
  echo
  echo "# Guarded launcher: three training arms, synchronous val20 barriers, resume."
  echo "START_TRAINING=1 bash /workspace/scripts/rexgroundingct/run_011_ct_normalization_ablation.sh"
} > "$commands_path"

if [[ "$START_TRAINING" != "1" ]]; then
  touch "$EXP_DIR/preparation/.prepared_not_started"
  echo "Experiment 011 preparation complete."
  echo "Training was not started. Explicit START_TRAINING=1 is required."
  exit 0
fi

python - "$EXP_DIR/config/prelaunch_gpu_check.json" "$MIN_FREE_GPU_GIB" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

output = Path(sys.argv[1])
minimum_free_gib = float(sys.argv[2])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the authorized training container")
device_count = torch.cuda.device_count()
if device_count != 3:
    raise SystemExit(
        f"Exp011 requires exactly three visible GPUs (host devices 0,1,2), got {device_count}"
    )
devices = []
for index in range(device_count):
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    free_gib = free_bytes / float(1024**3)
    devices.append(
        {
            "container_index": index,
            "name": torch.cuda.get_device_name(index),
            "free_gib": free_gib,
            "total_gib": total_bytes / float(1024**3),
        }
    )
    if free_gib < minimum_free_gib:
        raise SystemExit(
            f"GPU {index} has {free_gib:.2f} GiB free; "
            f"exp011 requires at least {minimum_free_gib:.2f} GiB"
        )
payload = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    "required_visible_devices": 3,
    "minimum_free_gib": minimum_free_gib,
    "devices": devices,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

mkdir -p "$GROUP_DIR"
echo "$RUN_GROUP" > "$EXP_DIR/config/latest_run_group.txt"
ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest"

checkpoint_has_update() {
  local checkpoint="$1" expected="$2"
  [[ -s "$checkpoint" ]] || return 1
  python - "$checkpoint" "$expected" <<'PY'
import sys
from pathlib import Path
import torch

checkpoint = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
if "rng_state" not in checkpoint:
    raise SystemExit("Checkpoint has no RNG state required for exact segmented resume")
if checkpoint.get("rng_state_scope") != "single_process":
    raise SystemExit(
        f"Checkpoint RNG scope is not single_process: {checkpoint.get('rng_state_scope')}"
    )
raise SystemExit(
    0 if int(checkpoint.get("global_update", -1)) == int(sys.argv[2]) else 1
)
PY
}

run_train_segment() {
  local arm="$1" gpu="$2" epoch="$3" previous_update="$4"
  local run_dir="$GROUP_DIR/$arm"
  local target_update="$((epoch * STEPS_PER_EPOCH))"
  local checkpoint="$run_dir/checkpoints/checkpoint_update_$(printf '%06d' "$target_update").pth"
  local segment_label
  segment_label="$(printf '%03d' "$epoch")"
  mkdir -p "$run_dir/logs" "$run_dir/reports"
  rm -f "$run_dir/.train_failed"
  if checkpoint_has_update "$checkpoint" "$target_update"; then
    echo "Training segment already complete: arm=$arm epoch=$epoch"
    return 0
  fi

  local resume_args=()
  if [[ -n "$previous_update" ]]; then
    local previous_checkpoint
    previous_checkpoint="$run_dir/checkpoints/checkpoint_update_$(printf '%06d' "$previous_update").pth"
    checkpoint_has_update "$previous_checkpoint" "$previous_update" || {
      echo "Missing valid resume checkpoint: $previous_checkpoint" >&2
      return 1
    }
    resume_args=(--resume-checkpoint "$previous_checkpoint")
  fi

  mapfile -t train_args < <(common_train_args "$arm")
  CUDA_VISIBLE_DEVICES="$gpu" python \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${train_args[@]}" \
    --run-dir "$run_dir" \
    --epochs "$EPOCHS" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --target-global-update "$target_update" \
    --checkpoint-every-updates 0 \
    --checkpoint-updates "$CHECKPOINT_UPDATES" \
    --latest-checkpoint-every-updates 500 \
    --no-materialize-final-model \
    "${resume_args[@]}" \
    >"$run_dir/logs/train_segment_to_epoch${segment_label}.log" 2>&1
  checkpoint_has_update "$checkpoint" "$target_update"
  cp "$run_dir/reports/training_metrics.json" \
    "$run_dir/reports/training_metrics_segment_epoch${segment_label}.json"
  cp "$run_dir/reports/training_report.md" \
    "$run_dir/reports/training_report_segment_epoch${segment_label}.md"
  cp "$run_dir/run_manifest.json" \
    "$run_dir/config/run_manifest_segment_epoch${segment_label}.json"
}

run_target_eval() {
  local arm="$1" gpu="$2" epoch="$3" probe="$4"
  rm -f "$GROUP_DIR/$arm/.eval_failed"
  EXP_DIR="$EXP_DIR" GROUP_DIR="$GROUP_DIR" EMBEDDINGS="$EMBEDDINGS" \
    SOURCE_MODEL_DIR="$SOURCE_MODEL_DIR" VAL20_JSON="$VAL20_JSON" \
    VAL200_JSON="$VAL200_JSON" CACHE_BASE="$CACHE_BASE" \
    EVAL_ARM="$arm" EVAL_GPU="$gpu" EVAL_EPOCH="$epoch" EVAL_PROBE="$probe" \
    bash /workspace/scripts/rexgroundingct/run_011_eval_coordinator.sh
}

write_intermediate_summary() {
  local trajectory_args=()
  if [[ -d "$EXP006_E5_D4_RUN" ]]; then
    trajectory_args+=(
      --trajectory-reference
      "Exp006 native z-score e5d4=$EXP006_E5_D4_RUN"
    )
  fi
  python /workspace/scripts/rexgroundingct/summarize_011_results.py \
    --exp-dir "$EXP_DIR" \
    --group-dir "$GROUP_DIR" \
    --output-json "$EXP_DIR/reports/ct_normalization_ablation_summary.json" \
    --output-md "$EXP_DIR/reports/ct_normalization_ablation_report.md" \
    "${trajectory_args[@]}" || true
}

status=0
previous_update=""
for epoch in $SEGMENT_EPOCHS; do
  echo "Starting synchronous training segment to epoch $epoch"
  segment_pids=()
  segment_arms=()
  for arm in "${arms[@]}"; do
    gpu="$(arm_gpu "$arm")"
    (
      run_train_segment "$arm" "$gpu" "$epoch" "$previous_update"
    ) &
    segment_pids+=("$!")
    segment_arms+=("$arm")
    sleep "$ARM_SLEEP_SECONDS"
  done
  for index in "${!segment_pids[@]}"; do
    if ! wait "${segment_pids[$index]}"; then
      arm="${segment_arms[$index]}"
      touch "$GROUP_DIR/$arm/.train_failed"
      echo "Training segment failed: arm=$arm epoch=$epoch" >&2
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    break
  fi

  echo "All arms paused at epoch $epoch; starting synchronous val20 barrier"
  eval_pids=()
  eval_arms=()
  for arm in "${arms[@]}"; do
    gpu="$(arm_gpu "$arm")"
    (
      run_target_eval "$arm" "$gpu" "$epoch" val20
    ) >"$GROUP_DIR/$arm/logs/eval_epoch$(printf '%03d' "$epoch")_val20_launcher.log" 2>&1 &
    eval_pids+=("$!")
    eval_arms+=("$arm")
  done
  for index in "${!eval_pids[@]}"; do
    if ! wait "${eval_pids[$index]}"; then
      arm="${eval_arms[$index]}"
      touch "$GROUP_DIR/$arm/.eval_failed"
      echo "Val20 failed: arm=$arm epoch=$epoch" >&2
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    break
  fi
  echo "Val20 barrier complete at epoch $epoch; training may resume"
  write_intermediate_summary
  previous_update="$((epoch * STEPS_PER_EPOCH))"
done

if [[ "$status" == "0" ]]; then
  for arm in "${arms[@]}"; do
    touch "$GROUP_DIR/$arm/.train_complete"
  done
  echo "Epoch 100 val20 complete; starting final val200 on GPUs 0-2"
  final_pids=()
  final_arms=()
  for arm in "${arms[@]}"; do
    gpu="$(arm_gpu "$arm")"
    (
      run_target_eval "$arm" "$gpu" 100 val200
    ) >"$GROUP_DIR/$arm/logs/eval_epoch100_val200_launcher.log" 2>&1 &
    final_pids+=("$!")
    final_arms+=("$arm")
  done
  for index in "${!final_pids[@]}"; do
    if ! wait "${final_pids[$index]}"; then
      arm="${final_arms[$index]}"
      touch "$GROUP_DIR/$arm/.eval_failed"
      echo "Val200 failed: arm=$arm" >&2
      status=1
    else
      touch "$GROUP_DIR/${final_arms[$index]}/.arm_success"
    fi
  done
fi

EXP006_E5_D4_SUMMARY="$EXP006_E5_D4_RUN/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
reference_args=()
if [[ -f "$EXP006_E5_D4_SUMMARY" ]]; then
  reference_args+=(--reference "exp006_v123_cached_e5_d4=$EXP006_E5_D4_SUMMARY")
fi
trajectory_args=()
if [[ -d "$EXP006_E5_D4_RUN" ]]; then
  trajectory_args+=(
    --trajectory-reference
    "Exp006 native z-score e5d4=$EXP006_E5_D4_RUN"
  )
fi

python /workspace/scripts/rexgroundingct/summarize_011_results.py \
  --exp-dir "$EXP_DIR" \
  --group-dir "$GROUP_DIR" \
  --output-json "$EXP_DIR/reports/ct_normalization_ablation_summary.json" \
  --output-md "$EXP_DIR/reports/ct_normalization_ablation_report.md" \
  "${trajectory_args[@]}" \
  "${reference_args[@]}" || true

if [[ "$status" == "0" ]]; then
  touch "$GROUP_DIR/.experiment_complete"
else
  touch "$GROUP_DIR/.experiment_failed"
fi
exit "$status"
