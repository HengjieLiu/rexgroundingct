#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

STAGE="${STAGE:?STAGE is required}"
EXP_ID="012_voxtell_category_specialists_replay50_cont100"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
GROUP_DIR="${GROUP_DIR:?GROUP_DIR is required}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/baseline_cont100/model_epoch100}"
SOURCE_CHECKPOINT="$SOURCE_MODEL_DIR/fold_0/checkpoint_final.pth"
SOURCE_REPLAY_SCHEDULE="${SOURCE_REPLAY_SCHEDULE:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/config/train_schedule_v123_ddp_bs4_seed20260723_100ep_100steps_gb4.jsonl}"
SOURCE_EMBEDDINGS="${SOURCE_EMBEDDINGS:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/config/rex_text_embeddings.npz}"
CACHE_DIR="${CACHE_DIR:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1}"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
SCHEDULE_DIR="$EXP_DIR/config/schedules"
EMBEDDINGS="$EXP_DIR/config/rex_text_embeddings.npz"
SUBSET_MANIFEST="/workspace/configs/evaluation/rexgroundingct_exp012_category_evaluation_subsets.manifest.json"
VAL80_JSON="/workspace/side_experiments/sideexp001_validation_probe_design/outputs/rexgroundingct_val80_sideexp001_validation_probe_design.json"
VAL200_JSON="/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json"
SEED="${SEED:-20260801}"
EPOCHS=100
STEPS_PER_EPOCH=100
CHECKPOINT_UPDATES="500,2000,4000,6000,8000,10000"
MIN_FREE_GPU_GIB="${MIN_FREE_GPU_GIB:-32}"
TARGET_EPOCH="${TARGET_EPOCH:-}"
EVAL_SCOPE="${EVAL_SCOPE:-}"

EXPECTED_SOURCE_CHECKPOINT_SHA="50e631f174ad4c16222df1d8114c30171df6e974cac5e934fa230448d91479bf"
EXPECTED_SOURCE_REPLAY_SHA="acdfed8dd8d9908e9ea33d6790d47f8f7e3cbeed4bad326274d115d38e96a0aa"
EXPECTED_EMBEDDINGS_SHA="a7351992fa57e019acb485f6b1f37045e7efcd879118cbac972b1b7ed94d5575"
EXPECTED_CACHE_SHA="fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3"
EXPECTED_VAL200_SHA="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
EXPECTED_VAL80_SHA="61048d78be0d72ff28619ebfff3efac8501dc0a6154deaa38881e26bca054faf"

arms=(
  category_1alldiffuse_replay50
  category_2a_replay50
  category_2b_replay50
  category_2c_replay50
)

arm_gpu() {
  case "$1" in
    category_1alldiffuse_replay50) echo 0 ;;
    category_2a_replay50) echo 1 ;;
    category_2b_replay50) echo 2 ;;
    category_2c_replay50) echo 3 ;;
    *) return 1 ;;
  esac
}

arm_schedule() {
  echo "$SCHEDULE_DIR/$1.jsonl"
}

sha256_path() {
  sha256sum "$1" | cut -d ' ' -f 1
}

require_sha() {
  local path="$1" expected="$2" label="$3"
  [[ -f "$path" ]] || { echo "Missing $label: $path" >&2; return 1; }
  local actual
  actual="$(sha256_path "$path")"
  [[ "$actual" == "$expected" ]] || {
    echo "$label SHA mismatch: expected=$expected actual=$actual path=$path" >&2
    return 1
  }
  echo "$label sha256=$actual"
}

validate_shared_inputs() {
  [[ -f "$CACHE_DIR/.complete" ]] || { echo "Missing cache marker: $CACHE_DIR/.complete" >&2; return 1; }
  require_sha "$CACHE_DIR/manifest.json" "$EXPECTED_CACHE_SHA" "native cache manifest"
  require_sha "$SOURCE_CHECKPOINT" "$EXPECTED_SOURCE_CHECKPOINT_SHA" "Exp009 source checkpoint"
  require_sha "$SOURCE_REPLAY_SCHEDULE" "$EXPECTED_SOURCE_REPLAY_SHA" "source replay schedule"
  require_sha "$SOURCE_EMBEDDINGS" "$EXPECTED_EMBEDDINGS_SHA" "source embeddings"
  require_sha "$VAL200_JSON" "$EXPECTED_VAL200_SHA" "val200 JSON"
  require_sha "$VAL80_JSON" "$EXPECTED_VAL80_SHA" "accepted val80 JSON"
  [[ -f "$SOURCE_MODEL_DIR/plans.json" ]] || { echo "Missing source plans" >&2; return 1; }
  [[ -f "$SUBSET_MANIFEST" ]] || { echo "Missing Exp012 subset manifest" >&2; return 1; }
  python - "$SUBSET_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
if not manifest.get("zero_train_validation_case_overlap"):
    raise SystemExit("Subset manifest does not certify zero train/validation overlap")
for arm, record in manifest["arms"].items():
    for scope in ("target", "union"):
        path = Path(record[scope]["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record[scope]["sha256"]:
            raise SystemExit(f"{arm}/{scope} hash mismatch: {path}")
print("Exp012 evaluation subset hashes accepted")
PY
}

verify_gpu_policy() {
  python - "$MIN_FREE_GPU_GIB" <<'PY'
import sys
import torch

minimum = float(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit(f"Expected exactly four CUDA devices, found {torch.cuda.device_count()}")
for index in range(4):
    free, total = torch.cuda.mem_get_info(index)
    free_gib = free / 1024**3
    print(
        f"gpu={index} name={torch.cuda.get_device_name(index)} "
        f"free_gib={free_gib:.2f} total_gib={total / 1024**3:.2f}"
    )
    if free_gib < minimum:
        raise SystemExit(f"GPU {index} has {free_gib:.2f} GiB free; required={minimum:.2f}")
PY
}

checkpoint_update() {
  python - "$1" <<'PY'
import sys
from pathlib import Path
import torch

checkpoint = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
if checkpoint.get("rng_state_scope") != "single_process":
    raise SystemExit(1)
print(int(checkpoint.get("global_update", -1)))
PY
}

checkpoint_has_update() {
  local checkpoint="$1" expected="$2"
  [[ -s "$checkpoint" ]] || return 1
  [[ "$(checkpoint_update "$checkpoint" 2>/dev/null)" == "$expected" ]]
}

find_resume_checkpoint() {
  local run_dir="$1" minimum_update="$2" target_update="$3"
  local best_path="" best_update="$minimum_update" candidate update
  for candidate in "$run_dir"/checkpoints/checkpoint_update_*.pth "$run_dir"/checkpoints/checkpoint_latest.pth; do
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
  return 0
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

common_train_args() {
  local arm="$1"
  printf '%s\n' \
    --mode train \
    --experiment-id "$EXP_ID" \
    --config "$CONFIG_SNAPSHOT" \
    --exp-dir "$EXP_DIR" \
    --gpu 0 \
    --model-dir "$SOURCE_MODEL_DIR" \
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
    --preprocessed-cache-dir "$CACHE_DIR" \
    --sample-schedule "$(arm_schedule "$arm")" \
    --allow-experimental-train
}

refresh_report() {
  local current_stage="$1" snapshot_dir="${2:-}" failure="${3:-}"
  local args=(
    --group-dir "$GROUP_DIR"
    --source-checkpoint "$SOURCE_CHECKPOINT"
    --subset-manifest "$SUBSET_MANIFEST"
    --val80-json "$VAL80_JSON"
    --val200-json "$VAL200_JSON"
    --output-json "$GROUP_DIR/reports/exp012_progress.json"
    --output-md "$GROUP_DIR/reports/exp012_progress.md"
    --current-stage "$current_stage"
  )
  [[ -n "$snapshot_dir" ]] && args+=(--snapshot-dir "$snapshot_dir")
  [[ -n "$failure" ]] && args+=(--failure "$failure")
  python /workspace/scripts/rexgroundingct/summarize_012_results.py "${args[@]}"
}

verify_schedules() {
  python - "$SCHEDULE_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
combined = json.loads((root / "exp012_schedules.manifest.json").read_text())
expected_hashes = {
    "category_1alldiffuse_replay50": "6fe0f689d27745f1691d10951b223ebb14badaf28a7584a80dce3ccca6e3987b",
    "category_2a_replay50": "e713e3bd0d5df2d662e0f2f74e1465a2cc0ab076e9cb76d7e09812a34227fc70",
    "category_2b_replay50": "fc207bd0b4174daa69e57d051cf4bab46116053bc09527d127dc1169c2e2c8f6",
    "category_2c_replay50": "5b5be3bf9970b60e6c794794e98db6ef084a382c2520dc3d0117d85342070532",
}
expected_replay = "c1dd7e9e992bb5a38ee87556cfc5dda6746eb40ec368cae36dbc9e305e20509e"
fingerprints = set()
for arm, manifest in combined["arms"].items():
    schedule = Path(manifest["output_jsonl"])
    digest = hashlib.sha256(schedule.read_bytes()).hexdigest()
    if digest != manifest["output_jsonl_sha256"]:
        raise SystemExit(f"Schedule hash mismatch: {arm}")
    if digest != expected_hashes[arm]:
        raise SystemExit(f"Schedule differs from canonical Exp012 hash: {arm}")
    audit = manifest["audit"]
    if audit["events"] != 10000 or audit["sampling_source_counts"] != {"replay": 5000, "targeted": 5000}:
        raise SystemExit(f"Schedule count mismatch: {arm}")
    if any(row != {"replay": 50, "targeted": 50} for row in audit["per_epoch"]):
        raise SystemExit(f"Per-epoch schedule ratio mismatch: {arm}")
    if audit["target_anchor_failures"] != 0:
        raise SystemExit(f"Target anchor failures: {arm}")
    fingerprints.add(audit["shared_replay_fingerprint"])
if len(fingerprints) != 1:
    raise SystemExit("Replay fingerprints differ across arms")
if fingerprints != {expected_replay}:
    raise SystemExit("Replay fingerprint differs from canonical Exp012 fingerprint")
print(f"Exp012 schedules accepted; replay_fingerprint={next(iter(fingerprints))}")
PY
}

run_prepare() {
  mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$EXP_DIR/reports" "$EXP_DIR/runs" "$GROUP_DIR/reports"
  validate_shared_inputs
  python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
    --experiment "$EXP_ID" --exp-dir "$EXP_DIR"
  if [[ ! -f "$EMBEDDINGS" ]]; then
    cp "$SOURCE_EMBEDDINGS" "$EMBEDDINGS"
  fi
  require_sha "$EMBEDDINGS" "$EXPECTED_EMBEDDINGS_SHA" "runtime embeddings"
  if [[ ! -f "$SCHEDULE_DIR/exp012_schedules.manifest.json" ]]; then
    python /workspace/scripts/rexgroundingct/prepare_012_category_specialists.py \
      --mode schedules \
      --seed "$SEED" \
      --preprocessed-cache-dir "$CACHE_DIR" \
      --replay-schedule "$SOURCE_REPLAY_SCHEDULE" \
      --schedule-output-dir "$SCHEDULE_DIR"
  fi
  verify_schedules
  ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest"
  ln -sfn "$GROUP_DIR/reports/exp012_progress.md" "$EXP_DIR/reports/latest_progress.md"
  ln -sfn "$GROUP_DIR/reports/exp012_progress.json" "$EXP_DIR/reports/latest_progress.json"
  basename "$GROUP_DIR" >"$EXP_DIR/config/active_run_group.txt"
  python - "$GROUP_DIR/run_group_manifest.json" "$GROUP_DIR" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

output, group_dir = map(Path, sys.argv[1:3])
existing = json.loads(output.read_text()) if output.is_file() else {}
payload = {
    "created_at_utc": existing.get(
        "created_at_utc", dt.datetime.now(dt.timezone.utc).isoformat()
    ),
    "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "experiment": "012_voxtell_category_specialists_replay50_cont100",
    "run_group": group_dir.name,
    "arms": [
        "category_1alldiffuse_replay50",
        "category_2a_replay50",
        "category_2b_replay50",
        "category_2c_replay50",
    ],
    "host_gpus": [0, 1, 2, 3],
    "milestones": [0, 5, 20, 40, 60, 80, 100],
    "terminal_state": "selection_ready",
}
tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, output)
PY
  refresh_report prepare
  touch "$GROUP_DIR/.prepared"
}

run_sample_tests() {
  validate_shared_inputs
  verify_schedules
  local root="$GROUP_DIR/preparation/sample_tests"
  [[ -f "$root/.complete" ]] && { echo "Sample tests already complete"; return 0; }
  local arm
  for arm in "${arms[@]}"; do
    mkdir -p "$root/$arm/logs"
    CUDA_VISIBLE_DEVICES="" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
      --mode sample-test \
      --experiment-id "$EXP_ID" \
      --config "$CONFIG_SNAPSHOT" \
      --exp-dir "$EXP_DIR" \
      --run-dir "$root/$arm" \
      --gpu 0 \
      --seed "$SEED" \
      --patch-size 192 192 192 \
      --sample-schedule "$(arm_schedule "$arm")" \
      --preprocessed-cache-dir "$CACHE_DIR" \
      --require-positive-crop \
      --loss-mode empty_bce_only \
      --sample-test-steps 8 \
      >"$root/$arm/logs/sample_test.log" 2>&1
    [[ -f "$root/$arm/reports/sample_test.json" ]]
  done
  touch "$root/.complete"
}

run_train_invocation() {
  local arm="$1" gpu="$2" run_dir="$3" horizon="$4" target="$5" checkpoints="$6"
  shift 6
  local extra=("$@") args=()
  mapfile -t args < <(common_train_args "$arm")
  mkdir -p "$run_dir/logs"
  CUDA_VISIBLE_DEVICES="$gpu" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${args[@]}" \
    --run-dir "$run_dir" \
    --epochs 1 \
    --steps-per-epoch "$horizon" \
    --target-global-update "$target" \
    --checkpoint-every-updates 0 \
    --checkpoint-updates "$checkpoints" \
    --latest-checkpoint-every-updates 0 \
    --no-materialize-final-model \
    "${extra[@]}"
}

run_smokes() {
  verify_gpu_policy
  validate_shared_inputs
  verify_schedules
  local root="$GROUP_DIR/preparation/smokes"
  [[ -f "$root/.complete" ]] && { echo "GPU smokes already complete"; return 0; }
  mkdir -p "$root"
  local pids=() launched=() arm gpu status=0 index
  for arm in "${arms[@]}"; do
    gpu="$(arm_gpu "$arm")"
    mkdir -p "$GROUP_DIR/$arm/logs"
    (
      local smoke_dir="$root/${arm}_one_update"
      if ! checkpoint_has_update "$smoke_dir/checkpoints/checkpoint_update_000001.pth" 1; then
        run_train_invocation "$arm" "$gpu" "$smoke_dir" 1 1 1 \
          --init-checkpoint "$SOURCE_CHECKPOINT" \
          >"$root/${arm}_one_update.log" 2>&1
      fi
      checkpoint_has_update "$smoke_dir/checkpoints/checkpoint_update_000001.pth" 1
    ) &
    pids+=("$!")
    launched+=("$arm")
  done
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || { echo "One-update smoke failed: ${launched[$index]}" >&2; status=1; }
  done
  ((status == 0)) || { refresh_report smoke "" "one-update smoke failed"; return 1; }

  local control="$root/resume_equivalence"
  if [[ ! -f "$control/.complete" ]]; then
    mkdir -p "$control"
    run_train_invocation category_1alldiffuse_replay50 0 "$control/continuous" 2 2 2 \
      --init-checkpoint "$SOURCE_CHECKPOINT" >"$control/continuous.log" 2>&1
    run_train_invocation category_1alldiffuse_replay50 0 "$control/segmented" 2 1 1,2 \
      --init-checkpoint "$SOURCE_CHECKPOINT" >"$control/segmented_1.log" 2>&1
    run_train_invocation category_1alldiffuse_replay50 0 "$control/segmented" 2 2 1,2 \
      --resume-checkpoint "$control/segmented/checkpoints/checkpoint_update_000001.pth" \
      >"$control/segmented_2.log" 2>&1
    python - \
      "$control/continuous/checkpoints/checkpoint_update_000002.pth" \
      "$control/segmented/checkpoints/checkpoint_update_000002.pth" \
      "$control/result.json" <<'PY'
import json
import sys
from pathlib import Path
import torch

continuous_path, segmented_path, output = map(Path, sys.argv[1:4])
continuous = torch.load(continuous_path, map_location="cpu", weights_only=False)
segmented = torch.load(segmented_path, map_location="cpu", weights_only=False)
max_abs = 0.0
for key, value in continuous["network_weights"].items():
    candidate = segmented["network_weights"][key]
    if torch.is_tensor(value) and value.numel():
        max_abs = max(max_abs, float((value - candidate).abs().max()))
payload = {
    "network_max_abs_difference": max_abs,
    "grad_scaler_equal": continuous["grad_scaler"] == segmented["grad_scaler"],
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if max_abs > 1e-7 or not payload["grad_scaler_equal"]:
    raise SystemExit("Continuous/resume equivalence failed")
PY
    touch "$control/.complete"
  fi
  touch "$root/.complete"
}

run_train_segment() {
  local epoch="${TARGET_EPOCH:?TARGET_EPOCH is required for train}" prior_epoch target_update prior_update
  prior_epoch="$(previous_epoch "$epoch")"
  target_update="$((epoch * STEPS_PER_EPOCH))"
  prior_update="$((prior_epoch * STEPS_PER_EPOCH))"
  [[ -f "$GROUP_DIR/milestones/epoch$(printf '%03d' "$prior_epoch")/report.complete" ]] || {
    echo "Prior report barrier is incomplete: epoch=$prior_epoch" >&2
    return 1
  }
  verify_gpu_policy
  local milestone="$GROUP_DIR/milestones/epoch$(printf '%03d' "$epoch")"
  mkdir -p "$milestone"
  [[ -f "$milestone/train.complete" ]] && { echo "Training barrier already complete: epoch=$epoch"; return 0; }

  local pids=() launched=() arm gpu status=0 index
  for arm in "${arms[@]}"; do
    gpu="$(arm_gpu "$arm")"
    (
      local run_dir="$GROUP_DIR/$arm"
      local target_checkpoint="$run_dir/checkpoints/checkpoint_update_$(printf '%06d' "$target_update").pth"
      mkdir -p "$run_dir/logs" "$run_dir/reports" "$run_dir/config"
      if checkpoint_has_update "$target_checkpoint" "$target_update"; then
        exit 0
      fi
      local resume_spec="" resume_path="" resume_update="$prior_update" resume_args=()
      resume_spec="$(find_resume_checkpoint "$run_dir" "$prior_update" "$target_update")"
      if [[ -n "$resume_spec" ]]; then
        resume_path="${resume_spec%|*}"
        resume_update="${resume_spec##*|}"
        resume_args=(--resume-checkpoint "$resume_path")
      elif ((prior_update == 0)); then
        resume_args=(--init-checkpoint "$SOURCE_CHECKPOINT")
      else
        echo "No valid resume checkpoint for arm=$arm prior=$prior_update" >&2
        exit 1
      fi
      local args=()
      mapfile -t args < <(common_train_args "$arm")
      CUDA_VISIBLE_DEVICES="$gpu" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
        "${args[@]}" \
        --run-dir "$run_dir" \
        --epochs "$EPOCHS" \
        --steps-per-epoch "$STEPS_PER_EPOCH" \
        --target-global-update "$target_update" \
        --checkpoint-every-updates 0 \
        --checkpoint-updates "$CHECKPOINT_UPDATES" \
        --latest-checkpoint-every-updates 500 \
        --no-materialize-final-model \
        "${resume_args[@]}" \
        >"$run_dir/logs/train_segment_to_epoch$(printf '%03d' "$epoch").log" 2>&1
      checkpoint_has_update "$target_checkpoint" "$target_update"
      cp "$run_dir/reports/training_metrics.json" "$run_dir/reports/training_metrics_segment_epoch$(printf '%03d' "$epoch").json"
      cp "$run_dir/reports/training_report.md" "$run_dir/reports/training_report_segment_epoch$(printf '%03d' "$epoch").md"
    ) &
    pids+=("$!")
    launched+=("$arm")
    sleep 5
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      touch "$GROUP_DIR/${launched[$index]}/.train_failed"
      status=1
    fi
  done
  if ((status != 0)); then
    refresh_report "train_epoch$epoch" "" "training barrier failed at epoch $epoch"
    return 1
  fi
  for arm in "${arms[@]}"; do
    checkpoint_has_update "$GROUP_DIR/$arm/checkpoints/checkpoint_update_$(printf '%06d' "$target_update").pth" "$target_update"
  done
  touch "$milestone/train.complete"
}

run_eval_barrier() {
  local epoch="${TARGET_EPOCH:?TARGET_EPOCH is required for eval}" scope="${EVAL_SCOPE:?EVAL_SCOPE is required for eval}"
  local milestone="$GROUP_DIR/milestones/epoch$(printf '%03d' "$epoch")"
  if ((epoch > 0)); then
    [[ -f "$milestone/train.complete" ]] || { echo "Training barrier incomplete: epoch=$epoch" >&2; return 1; }
  else
    [[ -f "$GROUP_DIR/preparation/smokes/.complete" ]] || { echo "Smoke gate incomplete" >&2; return 1; }
  fi
  verify_gpu_policy
  local marker="$milestone/$scope.complete"
  mkdir -p "$milestone"
  [[ -f "$marker" ]] && { echo "Evaluation barrier already complete: epoch=$epoch scope=$scope"; return 0; }
  local pids=() launched=() arm gpu status=0 index
  for arm in "${arms[@]}"; do
    gpu="$(arm_gpu "$arm")"
    mkdir -p "$GROUP_DIR/$arm/logs"
    (
      EVAL_ARM="$arm" EVAL_GPU="$gpu" EVAL_EPOCH="$epoch" EVAL_SCOPE="$scope" \
        GROUP_DIR="$GROUP_DIR" SOURCE_MODEL_DIR="$SOURCE_MODEL_DIR" \
        SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT" EMBEDDINGS="$EMBEDDINGS" \
        CACHE_DIR="$CACHE_DIR" VAL200_JSON="$VAL200_JSON" \
        bash /workspace/scripts/rexgroundingct/run_012_eval_coordinator.sh
    ) >"$GROUP_DIR/$arm/logs/eval_epoch$(printf '%03d' "$epoch")_${scope}_launcher.log" 2>&1 &
    pids+=("$!")
    launched+=("$arm")
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      touch "$GROUP_DIR/${launched[$index]}/.eval_failed"
      status=1
    fi
  done
  if ((status != 0)); then
    refresh_report "eval_epoch${epoch}_${scope}" "" "evaluation barrier failed at epoch $epoch scope $scope"
    return 1
  fi
  touch "$marker"
}

run_report() {
  local epoch="${TARGET_EPOCH:?TARGET_EPOCH is required for report}" scope="${EVAL_SCOPE:?EVAL_SCOPE is required for report}"
  local milestone="$GROUP_DIR/milestones/epoch$(printf '%03d' "$epoch")"
  [[ -f "$milestone/$scope.complete" ]] || { echo "Evaluation marker missing: epoch=$epoch scope=$scope" >&2; return 1; }
  local verify=()
  [[ "$epoch" == "0" ]] && verify=(--verify-epoch0)
  python /workspace/scripts/rexgroundingct/summarize_012_results.py \
    --group-dir "$GROUP_DIR" \
    --source-checkpoint "$SOURCE_CHECKPOINT" \
    --subset-manifest "$SUBSET_MANIFEST" \
    --val80-json "$VAL80_JSON" \
    --val200-json "$VAL200_JSON" \
    --output-json "$GROUP_DIR/reports/exp012_progress.json" \
    --output-md "$GROUP_DIR/reports/exp012_progress.md" \
    --snapshot-dir "$milestone" \
    --current-stage "reported_epoch${epoch}_${scope}" \
    "${verify[@]}"
  touch "$milestone/report.complete"
  if [[ "$epoch" == "100" ]]; then
    python - "$GROUP_DIR/reports/exp012_progress.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if summary.get("status") != "selection_ready":
    raise SystemExit(f"Expected selection_ready, got {summary.get('status')}")
for arm, data in summary["arms"].items():
    if data.get("selection") is None:
        raise SystemExit(f"Missing selection for {arm}")
PY
    for arm in "${arms[@]}"; do
      touch "$GROUP_DIR/$arm/.arm_success"
    done
    touch "$GROUP_DIR/.selection_ready"
    rm -f "$EXP_DIR/config/active_run_group.txt"
  fi
}

case "$STAGE" in
  prepare) run_prepare ;;
  sample-test) run_sample_tests ;;
  smoke) run_smokes ;;
  train) run_train_segment ;;
  eval) run_eval_barrier ;;
  report) run_report ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
