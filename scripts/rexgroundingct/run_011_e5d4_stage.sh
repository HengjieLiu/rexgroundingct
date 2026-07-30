#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

STAGE="${STAGE:?STAGE is required}"
EXP_ID="011_voxtell_v123_e4d4_ct_normalization_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
GROUP_DIR="${GROUP_DIR:?GROUP_DIR is required}"
E4_GROUP_DIR="${E4_GROUP_DIR:?E4_GROUP_DIR is required}"
EXP003_DIR="${EXP003_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation}"
CACHE_BASE="${CACHE_BASE:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell}"
NATIVE_CACHE="$CACHE_BASE/crop_zscore_native_v1"
CLIPPED_ZSCORE_CACHE="$CACHE_BASE/crop_clip1024_zscore_native_v1"
CLIPPED_LINEAR_CACHE="$CACHE_BASE/crop_clip1024_linear_native_v1"
SEED="${SEED:-20260723}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EPOCHS="${EPOCHS:-100}"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
SOURCE_MODEL_DIR="$EXP_DIR/config/public_voxtell_v1_1_model"
BASE_MODEL_DIR="${VOXTELL_MODEL:-/data/hengjie/datasets/rexgroundingct/.hf_home/hub/models--mrokuss--VoxTell/snapshots/0809ac94c5ed594198bf4e85d63245cf222464d7/voxtell_v1.1}"
SOURCE_SCHEDULE="$EXP003_DIR/config/train_schedule_v123_opt_poscrop_emptyloss_seed${SEED}_100ep_100steps_gb1.jsonl"
SCHEDULE="$EXP_DIR/config/train_schedule_v123_ct_norm_seed${SEED}_100ep_100steps_gb1.jsonl"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-500,2000,4000,6000,8000,10000}"
EXPECTED_SCHEDULE_SHA="f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776"
EXPECTED_VAL20_SHA="31557d624c47ee8c06799bf89cceb862471b34cfd47065001d6092e32d0dab5d"
EXPECTED_VAL200_SHA="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
EXPECTED_NATIVE_CACHE_SHA="fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3"
EXPECTED_CLIPPED_ZSCORE_SHA="f834ec24e16c0787f02f2e0589a10fafe16a2962dfffdb3666ea9a83eb425c35"
EXPECTED_LINEAR_SHA="0ba44e6792b32a91159a1aa9d35465c732ed6c6bf7122f63368634c59e298229"
MIN_FREE_GPU_GIB="${MIN_FREE_GPU_GIB:-32}"

arms=(
  v123_e5d4_zscore
  v123_e5d4_clip1024_zscore
  v123_e5d4_clip1024_linear
)

arm_cache() {
  case "$1" in
    v123_e5d4_zscore) echo "$NATIVE_CACHE" ;;
    v123_e5d4_clip1024_zscore) echo "$CLIPPED_ZSCORE_CACHE" ;;
    v123_e5d4_clip1024_linear) echo "$CLIPPED_LINEAR_CACHE" ;;
    *) return 1 ;;
  esac
}

arm_gpu() {
  case "$1" in
    v123_e5d4_zscore) echo 0 ;;
    v123_e5d4_clip1024_zscore) echo 1 ;;
    v123_e5d4_clip1024_linear) echo 2 ;;
    *) return 1 ;;
  esac
}

sha256_path() {
  sha256sum "$1" | cut -d ' ' -f 1
}

require_sha() {
  local path="$1" expected="$2" label="$3"
  [[ -f "$path" ]] || {
    echo "Missing $label: $path" >&2
    return 1
  }
  local actual
  actual="$(sha256_path "$path")"
  [[ "$actual" == "$expected" ]] || {
    echo "$label SHA mismatch: expected=$expected actual=$actual path=$path" >&2
    return 1
  }
  echo "$label sha256=$actual"
}

validate_cache() {
  local cache_root="$1" preprocess_id="$2" expected_sha="$3"
  [[ -f "$cache_root/.complete" ]] || {
    echo "Missing cache completion marker: $cache_root/.complete" >&2
    return 1
  }
  require_sha "$cache_root/manifest.json" "$expected_sha" "$preprocess_id manifest"
  python - "$cache_root/manifest.json" "$preprocess_id" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "preprocess_id": sys.argv[2],
    "cases": 3192,
    "targets": 8068,
    "empty_targets": 0,
}
observed = {key: manifest.get(key) for key in expected}
if observed != expected:
    raise SystemExit(f"Cache audit mismatch: expected={expected} observed={observed}")
cross = manifest.get("cross_cache_audit") or {}
if sys.argv[2] != "crop_zscore_native_v1":
    required = {
        "geometry_or_target_mismatches": 0,
        "materialized_hu_header_failures": 0,
        "normalization_failures": 0,
    }
    observed_cross = {key: cross.get(key) for key in required}
    if observed_cross != required:
        raise SystemExit(
            f"Cross-cache audit mismatch: expected={required} observed={observed_cross}"
        )
print(f"Cache accepted: {observed}")
PY
}

validate_shared_inputs() {
  validate_cache "$NATIVE_CACHE" crop_zscore_native_v1 "$EXPECTED_NATIVE_CACHE_SHA"
  validate_cache \
    "$CLIPPED_ZSCORE_CACHE" \
    crop_clip1024_zscore_native_v1 \
    "$EXPECTED_CLIPPED_ZSCORE_SHA"
  validate_cache \
    "$CLIPPED_LINEAR_CACHE" \
    crop_clip1024_linear_native_v1 \
    "$EXPECTED_LINEAR_SHA"
  require_sha "$SOURCE_SCHEDULE" "$EXPECTED_SCHEDULE_SHA" "exp003 v123 schedule"
  require_sha "$VAL20_JSON" "$EXPECTED_VAL20_SHA" "val20 JSON"
  require_sha "$VAL200_JSON" "$EXPECTED_VAL200_SHA" "val200 JSON"
  [[ -f "$EMBEDDINGS" ]] || {
    echo "Missing prompt embeddings: $EMBEDDINGS" >&2
    return 1
  }
  [[ -f "$BASE_MODEL_DIR/plans.json" ]] || {
    echo "Missing public VoxTell plans: $BASE_MODEL_DIR/plans.json" >&2
    return 1
  }
  [[ -f "$BASE_MODEL_DIR/fold_0/checkpoint_final.pth" ]] || {
    echo "Missing public VoxTell checkpoint: $BASE_MODEL_DIR/fold_0/checkpoint_final.pth" >&2
    return 1
  }
}

verify_gpu_policy() {
  python - "$MIN_FREE_GPU_GIB" <<'PY'
import os
import sys
import torch

minimum = float(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 3:
    raise SystemExit(
        f"Expected exactly host GPUs 0,1,2 to be visible; "
        f"found {torch.cuda.device_count()} devices"
    )
visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
if "3" in [value.strip() for value in visible.split(",")]:
    raise SystemExit(f"GPU 3 must not be visible: NVIDIA_VISIBLE_DEVICES={visible}")
for index in range(3):
    free, total = torch.cuda.mem_get_info(index)
    free_gib = free / 1024**3
    print(
        f"gpu={index} name={torch.cuda.get_device_name(index)} "
        f"free_gib={free_gib:.2f} total_gib={total / 1024**3:.2f}"
    )
    if free_gib < minimum:
        raise SystemExit(
            f"GPU {index} has {free_gib:.2f} GiB free; required={minimum:.2f}"
        )
PY
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
    --batch-size 1 \
    --grad-accum 1 \
    --encoder-lr 1e-5 \
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

checkpoint_update() {
  python - "$1" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if "rng_state" not in checkpoint:
    raise SystemExit(f"Checkpoint has no RNG state: {path}")
if checkpoint.get("rng_state_scope") != "single_process":
    raise SystemExit(
        f"Checkpoint RNG scope is not single_process: "
        f"{checkpoint.get('rng_state_scope')}"
    )
print(int(checkpoint.get("global_update", -1)))
PY
}

checkpoint_has_update() {
  local checkpoint="$1" expected="$2"
  [[ -s "$checkpoint" ]] || return 1
  [[ "$(checkpoint_update "$checkpoint")" == "$expected" ]]
}

previous_epoch() {
  case "$1" in
    5) echo 0 ;;
    20) echo 5 ;;
    40) echo 20 ;;
    60) echo 40 ;;
    80) echo 60 ;;
    100) echo 80 ;;
    *) return 1 ;;
  esac
}

find_resume_checkpoint() {
  local run_dir="$1" minimum_update="$2" target_update="$3"
  local best_path="" best_update="$minimum_update"
  local candidate update
  for candidate in \
    "$run_dir"/checkpoints/checkpoint_update_*.pth \
    "$run_dir"/checkpoints/checkpoint_latest.pth; do
    [[ -s "$candidate" ]] || continue
    update="$(checkpoint_update "$candidate" 2>/dev/null || echo -1)"
    if ((update >= best_update && update < target_update)); then
      best_path="$candidate"
      best_update="$update"
    fi
  done
  if [[ -n "$best_path" ]]; then
    printf '%s|%s\n' "$best_path" "$best_update"
  fi
}

verify_eval_output() {
  local eval_dir="$1" dataset_json="$2" expected_cases="$3" expected_findings="$4"
  python - "$eval_dir" "$dataset_json" "$expected_cases" "$expected_findings" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset = Path(sys.argv[2])
expected_cases = int(sys.argv[3])
expected_findings = int(sys.argv[4])
summary_path = root / "reports" / "val_quick_global_eval_summary.json"
eval_path = root / "eval" / "val_quick_global_eval.json"
predictions = root / "predictions"
if not (summary_path.is_file() and eval_path.is_file() and predictions.is_dir()):
    raise SystemExit(f"Incomplete evaluation directory: {root}")
entries = json.loads(dataset.read_text())["test"]
summary = json.loads(summary_path.read_text())
if len(entries) != expected_cases:
    raise SystemExit(f"Dataset case mismatch: {len(entries)} != {expected_cases}")
if int(summary.get("total_cases", -1)) != expected_cases:
    raise SystemExit(f"Summary case mismatch: {summary.get('total_cases')}")
if int(summary.get("total_findings", -1)) != expected_findings:
    raise SystemExit(f"Summary finding mismatch: {summary.get('total_findings')}")
missing = [entry["name"] for entry in entries if not (predictions / entry["name"]).is_file()]
if missing:
    raise SystemExit(f"Missing {len(missing)} predictions under {root}")
print(
    f"Evaluation accepted: cases={expected_cases} findings={expected_findings} "
    f"dice={summary.get('mean_global_dice_per_finding')} "
    f"hit_rate={summary.get('hit_rate')}"
)
PY
}

run_prepare() {
  mkdir -p "$EXP_DIR/config" "$EXP_DIR/reports" "$EXP_DIR/logs" "$GROUP_DIR"
  validate_shared_inputs
  python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
    --experiment "$EXP_ID" --exp-dir "$EXP_DIR" --overwrite
  if [[ ! -f "$SCHEDULE" ]]; then
    cp "$SOURCE_SCHEDULE" "$SCHEDULE"
  fi
  require_sha "$SCHEDULE" "$EXPECTED_SCHEDULE_SHA" "exp011 shared schedule"
  if [[ ! -f "$SOURCE_MODEL_DIR/plans.json" ]]; then
    mkdir -p "$SOURCE_MODEL_DIR"
    cp "$BASE_MODEL_DIR/plans.json" "$SOURCE_MODEL_DIR/plans.json"
  fi
  ln -sfn "$E4_GROUP_DIR" "$EXP_DIR/runs/latest_e4d4"
  ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest_e5d4"
  ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest"
  basename "$GROUP_DIR" > "$EXP_DIR/config/active_e5d4_run_group.txt"
  python - "$GROUP_DIR/run_group_manifest.json" "$E4_GROUP_DIR" "$GROUP_DIR" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

output, e4_group, e5_group = map(Path, sys.argv[1:4])
payload = {
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "experiment": "011_voxtell_v123_e4d4_ct_normalization_ablation",
    "profile": "e5d4",
    "e4d4_group": str(e4_group),
    "e5d4_group": str(e5_group),
    "arms": [
        "v123_e5d4_zscore",
        "v123_e5d4_clip1024_zscore",
        "v123_e5d4_clip1024_linear",
    ],
    "host_gpus": [0, 1, 2],
    "gpu3_exposed": False,
    "encoder_lr": 1e-5,
    "decoder_lr": 1e-4,
    "schedule_sha256": (
        "f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776"
    ),
    "barrier_epochs": [5, 20, 40, 60, 80, 100],
}
tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, output)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  touch "$GROUP_DIR/.prepared"
}

run_sample_tests() {
  validate_shared_inputs
  require_sha "$SCHEDULE" "$EXPECTED_SCHEDULE_SHA" "exp011 shared schedule"
  local root="$GROUP_DIR/preparation/sample_tests"
  if [[ -f "$root/.complete" ]]; then
    echo "Sample tests already complete"
    return 0
  fi
  local arm sample_dir
  for arm in "${arms[@]}"; do
    sample_dir="$root/$arm"
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
      >"$sample_dir/logs/sample_test.log" 2>&1
  done
  python - "$root" "${arms[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = sys.argv[2:]
reports = {
    arm: json.loads((root / arm / "reports" / "sample_test.json").read_text())
    for arm in arms
}
baseline = reports[arms[0]]
fields = ("samples", "patch_size", "sample_schedule", "sample_stats", "examples")
mismatches = []
for arm in arms[1:]:
    for field in fields:
        if reports[arm].get(field) != baseline.get(field):
            mismatches.append({"arm": arm, "field": field})
if mismatches:
    raise SystemExit(f"Sampler outputs differ across arms: {mismatches}")
print(
    f"Sampler comparison passed: arms={len(arms)} "
    f"events={baseline['samples']} schedule={baseline['sample_schedule']['sha256']}"
)
PY
  touch "$root/.complete"
}

run_train_invocation() {
  local arm="$1" gpu="$2" run_dir="$3" total_steps="$4" target_update="$5"
  local checkpoint_updates="$6"
  shift 6
  local extra_args=("$@")
  local train_args=()
  mapfile -t train_args < <(common_train_args "$arm")
  mkdir -p "$run_dir/logs"
  CUDA_VISIBLE_DEVICES="$gpu" python \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${train_args[@]}" \
    --run-dir "$run_dir" \
    --epochs 1 \
    --steps-per-epoch "$total_steps" \
    --target-global-update "$target_update" \
    --checkpoint-every-updates 0 \
    --checkpoint-updates "$checkpoint_updates" \
    --latest-checkpoint-every-updates 0 \
    --no-materialize-final-model \
    "${extra_args[@]}"
}

run_smokes() {
  verify_gpu_policy
  validate_shared_inputs
  require_sha "$SCHEDULE" "$EXPECTED_SCHEDULE_SHA" "exp011 shared schedule"
  local root="$GROUP_DIR/preparation/smokes"
  if [[ -f "$root/.complete" ]]; then
    echo "Optimizer and resume smokes already complete"
    return 0
  fi
  mkdir -p "$root"
  if ! checkpoint_has_update \
    "$root/native_continuous2/checkpoints/checkpoint_update_000002.pth" 2; then
    run_train_invocation \
      v123_e5d4_zscore 0 "$root/native_continuous2" 2 2 "2" \
      >"$root/native_continuous2.log" 2>&1
  fi
  if ! checkpoint_has_update \
    "$root/native_segmented2/checkpoints/checkpoint_update_000001.pth" 1; then
    run_train_invocation \
      v123_e5d4_zscore 0 "$root/native_segmented2" 2 1 "1,2" \
      >"$root/native_segmented_part1.log" 2>&1
  fi
  if ! checkpoint_has_update \
    "$root/native_segmented2/checkpoints/checkpoint_update_000002.pth" 2; then
    run_train_invocation \
      v123_e5d4_zscore 0 "$root/native_segmented2" 2 2 "1,2" \
      --resume-checkpoint "$root/native_segmented2/checkpoints/checkpoint_update_000001.pth" \
      >"$root/native_segmented_part2.log" 2>&1
  fi
  if ! checkpoint_has_update \
    "$root/linear_one_update/checkpoints/checkpoint_update_000001.pth" 1; then
    run_train_invocation \
      v123_e5d4_clip1024_linear 2 "$root/linear_one_update" 1 1 "1" \
      >"$root/linear_one_update.log" 2>&1
  fi
  python - \
    "$root/native_continuous2/checkpoints/checkpoint_update_000002.pth" \
    "$root/native_segmented2/checkpoints/checkpoint_update_000002.pth" \
    "$root/resume_equivalence.json" <<'PY'
import json
import os
import sys
from pathlib import Path
import torch

continuous_path, segmented_path, output = map(Path, sys.argv[1:4])
continuous = torch.load(continuous_path, map_location="cpu", weights_only=False)
segmented = torch.load(segmented_path, map_location="cpu", weights_only=False)
max_abs = 0.0
mismatched = []
for key, value in continuous["network_weights"].items():
    candidate = segmented["network_weights"][key]
    if torch.is_tensor(value):
        difference = float((value - candidate).abs().max()) if value.numel() else 0.0
        max_abs = max(max_abs, difference)
        if not torch.equal(value, candidate):
            mismatched.append(key)
payload = {
    "continuous_checkpoint": str(continuous_path),
    "segmented_checkpoint": str(segmented_path),
    "global_update_continuous": int(continuous["global_update"]),
    "global_update_segmented": int(segmented["global_update"]),
    "network_tensor_mismatch_count": len(mismatched),
    "network_tensor_mismatch_examples": mismatched[:20],
    "network_max_abs_difference": max_abs,
    "network_max_abs_tolerance": 1e-7,
    "grad_scaler_equal": continuous["grad_scaler"] == segmented["grad_scaler"],
}
tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, output)
print(json.dumps(payload, indent=2, sort_keys=True))
if max_abs > payload["network_max_abs_tolerance"] or not payload["grad_scaler_equal"]:
    raise SystemExit("Continuous/resume equivalence smoke failed")
PY
  touch "$root/.complete"
}

run_train_segment() {
  local target_epoch="${TARGET_EPOCH:?TARGET_EPOCH is required for train}"
  local prior_epoch target_update prior_update
  prior_epoch="$(previous_epoch "$target_epoch")"
  target_update="$((target_epoch * STEPS_PER_EPOCH))"
  prior_update="$((prior_epoch * STEPS_PER_EPOCH))"
  if ((prior_epoch > 0)); then
    [[ -f "$GROUP_DIR/milestones/epoch$(printf '%03d' "$prior_epoch")/report.complete" ]] || {
      echo "Previous report barrier is incomplete for epoch $prior_epoch" >&2
      return 1
    }
  fi
  verify_gpu_policy
  local milestone="$GROUP_DIR/milestones/epoch$(printf '%03d' "$target_epoch")"
  mkdir -p "$milestone"
  if [[ -f "$milestone/train.complete" ]]; then
    echo "Training barrier already complete: epoch=$target_epoch"
    return 0
  fi

  local pids=() launched_arms=() arm gpu
  for arm in "${arms[@]}"; do
    gpu="$(arm_gpu "$arm")"
    (
      local run_dir="$GROUP_DIR/$arm"
      local target_checkpoint
      target_checkpoint="$run_dir/checkpoints/checkpoint_update_$(printf '%06d' "$target_update").pth"
      mkdir -p "$run_dir/logs" "$run_dir/reports" "$run_dir/config"
      if checkpoint_has_update "$target_checkpoint" "$target_update"; then
        echo "Training target already complete: arm=$arm epoch=$target_epoch"
        exit 0
      fi
      local resume_spec="" resume_path="" resume_update="$prior_update"
      resume_spec="$(find_resume_checkpoint "$run_dir" "$prior_update" "$target_update")"
      if [[ -n "$resume_spec" ]]; then
        resume_path="${resume_spec%|*}"
        resume_update="${resume_spec##*|}"
      elif ((prior_update > 0)); then
        echo "No valid resume checkpoint for arm=$arm prior_update=$prior_update" >&2
        exit 1
      fi
      if ((resume_update > prior_update)) && [[ -f "$run_dir/reports/training_metrics.json" ]]; then
        cp "$run_dir/reports/training_metrics.json" \
          "$run_dir/reports/training_metrics_recovery_epoch$(printf '%03d' "$target_epoch")_to_update$(printf '%06d' "$resume_update").json"
      fi
      local train_args=() resume_args=()
      mapfile -t train_args < <(common_train_args "$arm")
      if [[ -n "$resume_path" ]]; then
        resume_args=(--resume-checkpoint "$resume_path")
      fi
      rm -f "$run_dir/.train_failed"
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
        >"$run_dir/logs/train_segment_to_epoch$(printf '%03d' "$target_epoch").log" 2>&1
      checkpoint_has_update "$target_checkpoint" "$target_update"
      cp "$run_dir/reports/training_metrics.json" \
        "$run_dir/reports/training_metrics_segment_epoch$(printf '%03d' "$target_epoch").json"
      cp "$run_dir/reports/training_report.md" \
        "$run_dir/reports/training_report_segment_epoch$(printf '%03d' "$target_epoch").md"
      cp "$run_dir/run_manifest.json" \
        "$run_dir/config/run_manifest_segment_epoch$(printf '%03d' "$target_epoch").json"
    ) &
    pids+=("$!")
    launched_arms+=("$arm")
    sleep 5
  done
  local status=0 index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      touch "$GROUP_DIR/${launched_arms[$index]}/.train_failed"
      status=1
    fi
  done
  ((status == 0)) || return 1
  for arm in "${arms[@]}"; do
    checkpoint_has_update \
      "$GROUP_DIR/$arm/checkpoints/checkpoint_update_$(printf '%06d' "$target_update").pth" \
      "$target_update"
  done
  touch "$milestone/train.complete"
}

run_eval_barrier() {
  local target_epoch="${TARGET_EPOCH:?TARGET_EPOCH is required for eval}"
  local probe="${EVAL_PROBE:?EVAL_PROBE is required for eval}"
  local milestone="$GROUP_DIR/milestones/epoch$(printf '%03d' "$target_epoch")"
  [[ -f "$milestone/train.complete" ]] || {
    echo "Training barrier is incomplete: epoch=$target_epoch" >&2
    return 1
  }
  if [[ "$probe" == "val200" ]]; then
    [[ -f "$milestone/val20.complete" && -f "$milestone/report.complete" ]] || {
      echo "Epoch-100 val20/report barrier must precede val200" >&2
      return 1
    }
  fi
  verify_gpu_policy
  local marker="$milestone/$probe.complete"
  if [[ -f "$marker" ]]; then
    echo "Evaluation barrier already complete: epoch=$target_epoch probe=$probe"
    return 0
  fi
  local pids=() eval_arms=() arm gpu
  for arm in "${arms[@]}"; do
    gpu="$(arm_gpu "$arm")"
    (
      EVAL_ARM="$arm" EVAL_GPU="$gpu" EVAL_EPOCH="$target_epoch" EVAL_PROBE="$probe" \
        EXP_DIR="$EXP_DIR" GROUP_DIR="$GROUP_DIR" \
        SOURCE_MODEL_DIR="$SOURCE_MODEL_DIR" EMBEDDINGS="$EMBEDDINGS" \
        VAL20_JSON="$VAL20_JSON" VAL200_JSON="$VAL200_JSON" CACHE_BASE="$CACHE_BASE" \
        bash /workspace/scripts/rexgroundingct/run_011_eval_coordinator.sh
    ) >"$GROUP_DIR/$arm/logs/eval_epoch$(printf '%03d' "$target_epoch")_${probe}_launcher.log" 2>&1 &
    pids+=("$!")
    eval_arms+=("$arm")
  done
  local status=0 index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      touch "$GROUP_DIR/${eval_arms[$index]}/.eval_failed"
      status=1
    fi
  done
  ((status == 0)) || return 1
  local dataset="$VAL20_JSON" expected_cases=20 expected_findings=31
  if [[ "$probe" == "val200" ]]; then
    dataset="$VAL200_JSON"
    expected_cases=200
    expected_findings=381
  fi
  for arm in "${arms[@]}"; do
    verify_eval_output \
      "$GROUP_DIR/$arm/eval_epoch$(printf '%03d' "$target_epoch")_$probe" \
      "$dataset" "$expected_cases" "$expected_findings"
  done
  touch "$marker"
}

run_report() {
  local target_epoch="${TARGET_EPOCH:?TARGET_EPOCH is required for report}"
  local probe="${EVAL_PROBE:?EVAL_PROBE is required for report}"
  local milestone="$GROUP_DIR/milestones/epoch$(printf '%03d' "$target_epoch")"
  [[ -f "$milestone/$probe.complete" ]] || {
    echo "Cannot report before $probe completes at epoch $target_epoch" >&2
    return 1
  }
  python /workspace/scripts/rexgroundingct/summarize_011_results.py \
    --exp-dir "$EXP_DIR" \
    --e4-group-dir "$E4_GROUP_DIR" \
    --e5-group-dir "$GROUP_DIR" \
    --output-json "$EXP_DIR/reports/ct_normalization_ablation_summary.json" \
    --output-md "$EXP_DIR/reports/ct_normalization_ablation_report.md"
  grep -Fq "$(basename "$GROUP_DIR")" \
    "$EXP_DIR/reports/ct_normalization_ablation_report.md"
  if [[ "$probe" == "val20" ]]; then
    touch "$milestone/report.complete"
  else
    touch "$milestone/val200_report.complete"
    for arm in "${arms[@]}"; do
      touch "$GROUP_DIR/$arm/.arm_success"
    done
    touch "$GROUP_DIR/.experiment_complete"
    rm -f "$EXP_DIR/config/active_e5d4_run_group.txt"
  fi
}

case "$STAGE" in
  prepare) run_prepare ;;
  sample-test) run_sample_tests ;;
  smoke) run_smokes ;;
  train) run_train_segment ;;
  eval) run_eval_barrier ;;
  report) run_report ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
