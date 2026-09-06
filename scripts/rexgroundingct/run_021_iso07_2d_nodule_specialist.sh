#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

EXP_ID="021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100"
EXP017_DIR="${EXP017_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched}"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1}"
METADATA="${METADATA:-/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json}"
CT_ROOT="${CT_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct}"
SEG_DIR="${SEG_DIR:-/data/hengjie/datasets/rexgroundingct/segmentations}"
SEED="${SEED:-20260723}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EPOCHS="${EPOCHS:-100}"
WORLD_SIZE="${WORLD_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SCHEDULE_EVENTS="${SCHEDULE_EVENTS:-40000}"
POSITIVE_EVENTS_PER_BLOCK="${POSITIVE_EVENTS_PER_BLOCK:-300}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUP="${RUN_GROUP:-exp021_iso07_2d_nodule_specialist_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
RUN_DIR="$GROUP_DIR/ddp_bs4"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
SCHEDULE="${SCHEDULE:-$EXP_DIR/config/train_schedule_2d_only_iso07_ddp_bs4_seed${SEED}_100ep_${STEPS_PER_EPOCH}steps.jsonl}"
SCHEDULE_MANIFEST="${SCHEDULE_MANIFEST:-$EXP_DIR/config/train_schedule_2d_only_iso07_ddp_bs4_seed${SEED}_100ep_${STEPS_PER_EPOCH}steps.manifest.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$EXP017_DIR/runs/exp017_iso07_hu_ddp_bs4_20260815T092119Z/ddp_bs4/checkpoints/checkpoint_update_010000.pth}"
SOURCE_EVAL_JSON="${SOURCE_EVAL_JSON:-$EXP017_DIR/runs/exp017_iso07_hu_ddp_bs4_20260815T092119Z/ddp_bs4/eval_epoch100_val200/eval/val_quick_global_eval.json}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$EXP017_DIR/config/public_voxtell_v1_1_model}"
EMBEDDINGS="${EMBEDDINGS:-$EXP017_DIR/config/rex_text_embeddings.npz}"
EXPECTED_INIT_CHECKPOINT_SHA="${EXPECTED_INIT_CHECKPOINT_SHA:-f11d238ffeaf9d96fe6ba7f81f6dd134953b18b0eaaae3f5404176aa0cec6cc4}"
EXPECTED_CACHE_MANIFEST_SHA="${EXPECTED_CACHE_MANIFEST_SHA:-59b53ed9fcd93b8e9872bf89770a14ffd6aeed31b206af33ca751b85d4bd2640}"
EXPECTED_VAL200_SHA="${EXPECTED_VAL200_SHA:-7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897}"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-2500,5000,7500,10000}"
SEGMENT_EPOCHS="${SEGMENT_EPOCHS:-25 50 75 100}"
RUN_SAMPLE_TEST="${RUN_SAMPLE_TEST:-1}"
RUN_DDP_SMOKE="${RUN_DDP_SMOKE:-1}"
RUN_INFERENCE_SMOKE="${RUN_INFERENCE_SMOKE:-1}"
RUN_EVAL_SMOKE="${RUN_EVAL_SMOKE:-1}"
RUN_STATIC_ONLY="${RUN_STATIC_ONLY:-0}"
RUN_SMOKE_ONLY="${RUN_SMOKE_ONLY:-1}"
RUN_FULL="${RUN_FULL:-0}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-43200}"
STABILITY_SECONDS="${STABILITY_SECONDS:-10}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
REPORT_JSON="$EXP_DIR/reports/nodule_specialist_summary.json"
REPORT_MD="$EXP_DIR/reports/nodule_specialist_report.md"

read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -ne "$WORLD_SIZE" ]]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but WORLD_SIZE=$WORLD_SIZE: $GPU_LIST" >&2
  exit 1
fi
DDP_CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$EXP_DIR/reports" "$GROUP_DIR" "$RUN_DIR/logs"
echo "$RUN_GROUP" > "$EXP_DIR/config/latest_run_group.txt"
ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest"

sha256_path() { sha256sum "$1" | cut -d ' ' -f 1; }

require_sha() {
  local path="$1" expected="$2" label="$3"
  [[ -f "$path" ]] || { echo "Missing $label: $path" >&2; exit 1; }
  local actual="$(sha256_path "$path")"
  [[ "$actual" == "$expected" ]] || { echo "$label SHA mismatch: expected=$expected actual=$actual" >&2; exit 1; }
  echo "$label sha256=$actual"
}

line_count_matches() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || return 1
  python - "$path" "$expected" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
expected = int(sys.argv[2])
with path.open("rb") as handle:
    observed = sum(1 for _ in handle)
raise SystemExit(0 if observed == expected else 1)
PY
}

verify_cache() {
  [[ -f "$CACHE_ROOT/.complete" ]] || { echo "Missing cache completion marker: $CACHE_ROOT/.complete" >&2; exit 1; }
  require_sha "$CACHE_ROOT/manifest.json" "$EXPECTED_CACHE_MANIFEST_SHA" "iso07 cache manifest"
  python - "$CACHE_ROOT/manifest.json" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "cases": 3492, "targets": 8068, "empty_targets": 0,
    "foreground_fallback_targets": 0, "geometry_or_target_mismatches": 0,
    "materialized_hu_header_failures": 0, "normalization_failures": 0,
    "image_padding_value": -1.0, "target_padding_value": 0,
    "preprocess_id": "crop_clip1024_linear_iso07_v1",
}
observed = {
    "cases": manifest.get("cases"), "targets": manifest.get("targets"),
    "empty_targets": manifest.get("empty_targets"),
    "foreground_fallback_targets": manifest.get("foreground_fallback_targets"),
    "geometry_or_target_mismatches": manifest.get("cross_cache_audit", {}).get("geometry_or_target_mismatches"),
    "materialized_hu_header_failures": manifest.get("cross_cache_audit", {}).get("materialized_hu_header_failures"),
    "normalization_failures": manifest.get("cross_cache_audit", {}).get("normalization_failures"),
    "image_padding_value": manifest.get("image_padding_value"),
    "target_padding_value": manifest.get("target_padding_value"),
    "preprocess_id": manifest.get("preprocess_id"),
}
if observed != expected:
    raise SystemExit(f"Iso07 cache audit mismatch: expected={expected} observed={observed}")
if [float(value) for value in manifest.get("target_spacing_zyx_mm", [])] != [0.7, 0.7, 0.7]:
    raise SystemExit(f"Iso07 cache spacing mismatch: {manifest.get('target_spacing_zyx_mm')}")
print(f"Iso07 cache audit accepted: {observed}")
PY
}

verify_source_checkpoint() {
  require_sha "$SOURCE_CHECKPOINT" "$EXPECTED_INIT_CHECKPOINT_SHA" "Exp017 e100 source checkpoint"
  [[ -f "$SOURCE_EVAL_JSON" ]] || { echo "Missing initial reference evaluation: $SOURCE_EVAL_JSON" >&2; exit 1; }
  python - "$SOURCE_CHECKPOINT" <<'PY'
import sys
from pathlib import Path
import torch
payload = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
if int(payload.get("global_update", -1)) != 10000:
    raise SystemExit(f"Expected source global_update=10000, got {payload.get('global_update')}")
print(f"Accepted source checkpoint global_update={payload['global_update']} epoch={payload.get('epoch')}")
PY
  python - "$SOURCE_CHECKPOINT" "$GROUP_DIR/config/init_checkpoint_provenance.json" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
import torch
source, output = map(Path, sys.argv[1:3])
payload = torch.load(source, map_location="cpu", weights_only=False)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source_checkpoint": str(source),
    "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source_global_update": int(payload.get("global_update", -1)),
    "source_epoch": int(payload.get("epoch", -1)),
    "load_policy": "network weights only via --init-checkpoint; optimizer, scaler, scheduler, RNG state, and update counter reset",
}, indent=2, sort_keys=True) + "\n")
PY
}

prepare_schedule() {
  if line_count_matches "$SCHEDULE" "$SCHEDULE_EVENTS"; then
    echo "Schedule ready: $SCHEDULE"
    return 0
  fi
  rm -f "$SCHEDULE" "$SCHEDULE_MANIFEST"
  python /workspace/scripts/rexgroundingct/prepare_021_nodule_schedule.py \
    --metadata "$METADATA" --preprocessed-cache-dir "$CACHE_ROOT" \
    --output-jsonl "$SCHEDULE" --manifest-json "$SCHEDULE_MANIFEST" \
    --seed "$SEED" --epochs "$EPOCHS" --steps-per-epoch "$STEPS_PER_EPOCH" \
    --world-size "$WORLD_SIZE" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
    --positive-events-per-block "$POSITIVE_EVENTS_PER_BLOCK" \
    2>&1 | tee "$EXP_DIR/logs/prepare_schedule_$TIMESTAMP.log"
}

verify_schedule() {
  line_count_matches "$SCHEDULE" "$SCHEDULE_EVENTS" || { echo "Schedule line count mismatch: $SCHEDULE" >&2; exit 1; }
  python - "$SCHEDULE_MANIFEST" "$SCHEDULE" "$SCHEDULE_EVENTS" "$WORLD_SIZE" <<'PY'
import json
import sys
from pathlib import Path
manifest_path, schedule_path = map(Path, sys.argv[1:3])
expected_events = int(sys.argv[3])
world_size = int(sys.argv[4])
manifest = json.loads(manifest_path.read_text())
if int(manifest.get("events", -1)) != expected_events:
    raise SystemExit(f"Manifest events mismatch: {manifest.get('events')}")
if int(manifest.get("world_size", -1)) != world_size:
    raise SystemExit(f"Manifest world size mismatch: {manifest.get('world_size')}")
print(f"Schedule accepted: events={manifest['events']} positive={manifest.get('positive_events')} negative={manifest.get('negative_events')} sha256={manifest.get('output_jsonl_sha256')}")
PY
}

prepare_embeddings_and_model() {
  if [[ ! -f "$EMBEDDINGS" ]]; then
    for candidate in "$EXP017_DIR/config/rex_text_embeddings.npz" "/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/config/rex_text_embeddings.npz"; do
      if [[ -f "$candidate" ]]; then EMBEDDINGS="$candidate"; break; fi
    done
  fi
  if [[ ! -f "$EMBEDDINGS" ]]; then
    EMBEDDINGS="$EXP_DIR/config/rex_text_embeddings.npz"
    python /workspace/scripts/rexgroundingct/precompute_text_embeddings.py --splits train val --gpu 0 --output "$EMBEDDINGS" \
      2>&1 | tee "$EXP_DIR/logs/precompute_text_embeddings_$TIMESTAMP.log"
  fi
  local base_model_dir="${BASE_MODEL_DIR:-}"
  if [[ -z "$base_model_dir" ]]; then
    base_model_dir="$(python -c 'from train_text_conditioned_voxtell import resolve_model_dir; print(resolve_model_dir(None))')"
  fi
  if [[ ! -f "$SOURCE_MODEL_DIR/plans.json" ]]; then
    mkdir -p "$SOURCE_MODEL_DIR"
    cp "$base_model_dir/plans.json" "$SOURCE_MODEL_DIR/plans.json"
  fi
  [[ -f "$base_model_dir/plans.json" ]] || { echo "Missing public model plans: $base_model_dir" >&2; exit 1; }
  [[ -f "$base_model_dir/fold_0/checkpoint_final.pth" ]] || { echo "Missing public model checkpoint: $base_model_dir" >&2; exit 1; }
  BASE_MODEL_DIR="$base_model_dir"
}

common_train_args() {
  printf '%s\n' \
    --mode train --experiment-id "$EXP_ID" --config "$CONFIG_SNAPSHOT" --exp-dir "$EXP_DIR" \
    --model-dir "$BASE_MODEL_DIR" --embeddings "$EMBEDDINGS" --metadata "$METADATA" \
    --ct-root "$CT_ROOT" --seg-dir "$SEG_DIR" --seed "$SEED" --patch-size 192 192 192 \
    --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" --encoder-lr 1e-5 --decoder-lr 1e-4 \
    --lr-schedule poly --poly-power 0.9 --warmup-updates 100 --momentum 0.99 --weight-decay 3e-5 \
    --clip-grad-norm 12 --loss-mode empty_bce_only --empty-target-loss-weight 0.5 \
    --voxel-bce-weighting --bce-foreground-weight 1.0 --bce-boundary-weight 1.5 \
    --bce-background-weight 0.5 --bce-boundary-radius 10 \
    --deep-supervision-weights 1,0.5,0.25,0.125,0.0625 --preprocessed-cache-dir "$CACHE_ROOT" \
    --sample-schedule "$SCHEDULE" --distributed --allow-experimental-train
}

run_sample_test() {
  local sample_dir="$GROUP_DIR/smoke/sample_test"
  mkdir -p "$sample_dir/logs"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    --mode sample-test --experiment-id "$EXP_ID" --config "$CONFIG_SNAPSHOT" --exp-dir "$EXP_DIR" \
    --run-dir "$sample_dir" --gpu 0 --metadata "$METADATA" --ct-root "$CT_ROOT" --seg-dir "$SEG_DIR" \
    --seed "$SEED" --patch-size 192 192 192 --sample-schedule "$SCHEDULE" \
    --preprocessed-cache-dir "$CACHE_ROOT" --sample-test-steps 8 \
    2>&1 | tee "$sample_dir/logs/sample_test.log"
}

run_ddp_smoke_segment() {
  local smoke_dir="$1" target="$2" checkpoint_updates="$3" resume_checkpoint="${4:-}"
  local log="$smoke_dir/logs/train_to_update_${target}.log"
  mkdir -p "$smoke_dir/logs"
  mapfile -t args < <(common_train_args)
  local resume_args=() init_args=()
  if [[ -n "$resume_checkpoint" ]]; then resume_args=(--resume-checkpoint "$resume_checkpoint"); else init_args=(--init-checkpoint "$SOURCE_CHECKPOINT"); fi
  {
    CUDA_VISIBLE_DEVICES="$DDP_CUDA_VISIBLE_DEVICES" python -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" \
      /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py "${args[@]}" \
      --run-dir "$smoke_dir" --epochs 1 --steps-per-epoch 4 --target-global-update "$target" \
      --checkpoint-every-updates 0 --checkpoint-updates "$checkpoint_updates" --latest-checkpoint-every-updates 0 \
      --no-materialize-final-model "${init_args[@]}" "${resume_args[@]}"
  } >"$log" 2>&1
}

checkpoint_has_update() {
  local checkpoint="$1" expected="$2"
  [[ -s "$checkpoint" ]] || return 1
  python - "$checkpoint" "$expected" <<'PY'
import sys
from pathlib import Path
import torch
payload = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
raise SystemExit(0 if int(payload.get("global_update", -1)) == int(sys.argv[2]) else 1)
PY
}

checkpoint_stable() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  local first="$(stat -c '%s:%Y' "$path")"
  sleep "$STABILITY_SECONDS"
  local second="$(stat -c '%s:%Y' "$path")"
  [[ "$first" == "$second" ]]
}

materialize_model() {
  local checkpoint="$1" model_dir="$2"
  mkdir -p "$model_dir/fold_0"
  cp "$SOURCE_MODEL_DIR/plans.json" "$model_dir/plans.json"
  local target="$model_dir/fold_0/checkpoint_final.pth"
  if [[ ! -f "$target" || "$(stat -c %s "$target")" != "$(stat -c %s "$checkpoint")" ]]; then
    local temporary="$target.tmp.$$"
    rm -f "$temporary"
    ln "$checkpoint" "$temporary" 2>/dev/null || cp "$checkpoint" "$temporary"
    mv -f "$temporary" "$target"
  fi
}

write_subset_json() {
  local source="$1" target="$2" count="$3"
  python - "$source" "$target" "$count" <<'PY'
import json
import sys
from pathlib import Path
source, target = map(Path, sys.argv[1:3])
data = json.loads(source.read_text())
data["test"] = data["test"][:int(sys.argv[3])]
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

run_cached_inference_shard() {
  local dataset_json="$1" model_dir="$2" output_dir="$3" status_dir="$4" log_dir="$5" gpu="$6" shard_count="$7" shard_index="$8"
  mkdir -p "$output_dir" "$status_dir" "$log_dir"
  EVAL_LOCK_DISABLE=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split val --dataset-json "$dataset_json" --gpu 0 --model-dir "$model_dir" --embeddings "$EMBEDDINGS" \
    --output-dir "$output_dir" --output-type mask --threshold 0.5 --preprocessed-cache-dir "$CACHE_ROOT" \
    --num-shards "$shard_count" --shard-index "$shard_index" --status-json "$status_dir/shard${shard_index}_gpu${gpu}.json" \
    >"$log_dir/inference_shard${shard_index}_gpu${gpu}.log" 2>&1
}

verify_prediction_shapes() {
  local dataset_json="$1" prediction_dir="$2"
  python - "$dataset_json" "$prediction_dir" <<'PY'
import json
import sys
from pathlib import Path
import nibabel as nib
dataset = json.loads(Path(sys.argv[1]).read_text())
prediction_dir = Path(sys.argv[2])
seg_dir = Path("/data/hengjie/datasets/rexgroundingct/segmentations")
for entry in dataset["test"]:
    pred = prediction_dir / entry["name"]
    gt = seg_dir / entry["name"]
    if not pred.is_file(): raise SystemExit(f"Missing prediction: {pred}")
    if nib.load(str(pred)).shape != nib.load(str(gt)).shape:
        raise SystemExit(f"{entry['name']}: prediction/GT shape mismatch")
print(f"Prediction shape check passed for {len(dataset['test'])} case(s)")
PY
}

eval_complete() {
  local eval_dir="$1" dataset_json="$2"
  python - "$eval_dir" "$dataset_json" <<'PY'
import json
import sys
from pathlib import Path
root, dataset_path = map(Path, sys.argv[1:3])
dataset = json.loads(dataset_path.read_text())["test"]
summary_path = root / "reports" / "val_quick_global_eval_summary.json"
eval_path = root / "eval" / "val_quick_global_eval.json"
predictions = root / "predictions"
if not summary_path.is_file() or not eval_path.is_file() or not predictions.is_dir(): raise SystemExit(1)
summary = json.loads(summary_path.read_text())
if int(summary.get("total_cases", -1)) != len(dataset): raise SystemExit(1)
if int(summary.get("total_findings", -1)) != sum(len(entry["findings"]) for entry in dataset): raise SystemExit(1)
if any(not (predictions / entry["name"]).is_file() for entry in dataset): raise SystemExit(1)
PY
}

run_inference_smoke() {
  local smoke_dir="$GROUP_DIR/smoke/one_case_inference"
  local model_dir="$GROUP_DIR/smoke/model_epoch000"
  local subset="$GROUP_DIR/smoke/one_case_val200.json"
  mkdir -p "$smoke_dir"
  materialize_model "$GROUP_DIR/smoke/ddp_resume_smoke/checkpoints/checkpoint_update_000004.pth" "$model_dir"
  write_subset_json "$VAL200_JSON" "$subset" 1
  run_cached_inference_shard "$subset" "$model_dir" "$smoke_dir/predictions" "$smoke_dir/status" "$smoke_dir/logs" "${GPUS[0]}" 1 0
  verify_prediction_shapes "$subset" "$smoke_dir/predictions"
}

run_eval_smoke() {
  local smoke_dir="$GROUP_DIR/smoke/four_case_eval"
  local model_dir="$GROUP_DIR/smoke/model_epoch000"
  local subset="$GROUP_DIR/smoke/four_case_val200.json"
  write_subset_json "$VAL200_JSON" "$subset" 4
  mkdir -p "$smoke_dir/predictions" "$smoke_dir/status" "$smoke_dir/logs"
  local pids=()
  for shard in 0 1 2 3; do
    run_cached_inference_shard "$subset" "$model_dir" "$smoke_dir/predictions" "$smoke_dir/status" "$smoke_dir/logs" "${GPUS[$shard]}" 4 "$shard" &
    pids+=("$!")
  done
  local status=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [[ "$status" == "0" ]] || { echo "Evaluation smoke inference failed" >&2; return 1; }
  verify_prediction_shapes "$subset" "$smoke_dir/predictions"
  EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$subset" \
    EVAL_LABEL="Experiment 021 four-case evaluation smoke" NUM_WORKERS=4 \
    bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$smoke_dir" val
  eval_complete "$smoke_dir" "$subset"
}

lock_age_seconds() {
  python - "$1" <<'PY'
import sys
import time
from pathlib import Path
print(max(0.0, time.time() - Path(sys.argv[1]).stat().st_mtime))
PY
}

acquire_lock() {
  local eval_dir="$1" lock="$eval_dir/.eval.lock"
  mkdir -p "$eval_dir"
  while true; do
    if (set -o noclobber; printf '{"pid":%s,"created":"%s","phase":"exp021_val200"}\n' "$$" "$(date -u +%FT%TZ)" > "$lock") 2>/dev/null; then
      echo "$lock"
      return 0
    fi
    local age
    age="$(lock_age_seconds "$lock" 2>/dev/null || echo 0)"
    if python - "$age" "$LOCK_STALE_SECONDS" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
    then
      echo "Removing stale eval lock: $lock"
      rm -f "$lock"
    else
      echo "Evaluation already locked, waiting: $lock"
      sleep 60
    fi
  done
}

run_val200_eval() {
  local epoch="$1" update="$((epoch * STEPS_PER_EPOCH))"
  local checkpoint="$RUN_DIR/checkpoints/checkpoint_update_$(printf '%06d' "$update").pth"
  local model_dir="$RUN_DIR/model_epoch$(printf '%03d' "$epoch")"
  local eval_dir="$RUN_DIR/eval_epoch$(printf '%03d' "$epoch")_val200"
  if eval_complete "$eval_dir" "$VAL200_JSON"; then
    echo "Already complete: epoch=$epoch val200"
    summarize_results
    return 0
  fi
  checkpoint_stable "$checkpoint" || { echo "Checkpoint not stable: $checkpoint" >&2; return 1; }
  local lock
  lock="$(acquire_lock "$eval_dir")"
  materialize_model "$checkpoint" "$model_dir"
  mkdir -p "$eval_dir/logs" "$eval_dir/status" "$eval_dir/predictions"
  local eval_log="$eval_dir/logs/eval_$(date -u +%Y%m%dT%H%M%SZ).log"
  {
    echo "start epoch=$epoch update=$update full_val200 gpus=$GPU_LIST"
    local pids=()
    for shard in 0 1 2 3; do
      run_cached_inference_shard "$VAL200_JSON" "$model_dir" "$eval_dir/predictions" "$eval_dir/status" "$eval_dir/logs" "${GPUS[$shard]}" 4 "$shard" &
      pids+=("$!")
    done
    local status=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    [[ "$status" == "0" ]] || { echo "One or more inference shards failed"; exit 1; }
    verify_prediction_shapes "$VAL200_JSON" "$eval_dir/predictions"
    EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$VAL200_JSON" \
      EVAL_LABEL="Experiment 021 iso07 nodule specialist epoch $epoch full val200 threshold 0.5" \
      NUM_WORKERS=8 bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$eval_dir" val
    eval_complete "$eval_dir" "$VAL200_JSON"
    echo "done epoch=$epoch full_val200"
  } >"$eval_log" 2>&1 || {
    rm -f "$lock"
    echo "Val200 evaluation failed: epoch=$epoch log=$eval_log" >&2
    return 1
  }
  rm -f "$lock"
  summarize_results
}

run_full_segment() {
  local epoch="$1" target_update="$((epoch * STEPS_PER_EPOCH))"
  local checkpoint="$RUN_DIR/checkpoints/checkpoint_update_$(printf '%06d' "$target_update").pth"
  local log="$RUN_DIR/logs/train_segment_to_epoch$(printf '%03d' "$epoch").log"
  mapfile -t args < <(common_train_args)
  if checkpoint_has_update "$checkpoint" "$target_update"; then
    echo "Training segment already complete: epoch=$epoch update=$target_update"
    return 0
  fi
  local resume_args=() init_args=()
  if [[ "$target_update" -gt 2500 ]]; then
    local previous_update="$((target_update - 2500))"
    local previous_checkpoint="$RUN_DIR/checkpoints/checkpoint_update_$(printf '%06d' "$previous_update").pth"
    checkpoint_has_update "$previous_checkpoint" "$previous_update" || { echo "Missing previous checkpoint: $previous_checkpoint" >&2; return 1; }
    resume_args=(--resume-checkpoint "$previous_checkpoint")
  else
    init_args=(--init-checkpoint "$SOURCE_CHECKPOINT")
  fi
  echo "Starting DDP segment to epoch=$epoch update=$target_update log=$log"
  {
    CUDA_VISIBLE_DEVICES="$DDP_CUDA_VISIBLE_DEVICES" python -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" \
      /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py "${args[@]}" \
      --run-dir "$RUN_DIR" --epochs "$EPOCHS" --steps-per-epoch "$STEPS_PER_EPOCH" \
      --target-global-update "$target_update" --checkpoint-every-updates 0 \
      --checkpoint-updates "$CHECKPOINT_UPDATES" --latest-checkpoint-every-updates 500 \
      --no-materialize-final-model "${init_args[@]}" "${resume_args[@]}"
  } >"$log" 2>&1
  checkpoint_has_update "$checkpoint" "$target_update" || { echo "Missing expected checkpoint: $checkpoint" >&2; return 1; }
  cp "$RUN_DIR/reports/training_metrics.json" "$RUN_DIR/reports/training_metrics_segment_epoch$(printf '%03d' "$epoch").json"
}

summarize_results() {
  python /workspace/scripts/rexgroundingct/summarize_021_nodule_results.py \
    --group-dir "$GROUP_DIR" --val200-json "$VAL200_JSON" --source-evaluation-json "$SOURCE_EVAL_JSON" \
    --steps-per-epoch "$STEPS_PER_EPOCH" --output-json "$REPORT_JSON" --output-md "$REPORT_MD" \
    >"$EXP_DIR/logs/summarize_$TIMESTAMP.log" 2>&1
}

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py --experiment "$EXP_ID" --exp-dir "$EXP_DIR" --overwrite
verify_cache
require_sha "$VAL200_JSON" "$EXPECTED_VAL200_SHA" "val200 JSON"
verify_source_checkpoint
prepare_schedule
verify_schedule
prepare_embeddings_and_model

if [[ "$RUN_STATIC_ONLY" == "1" ]]; then
  echo "Static checks complete; RUN_STATIC_ONLY=1"
  exit 0
fi
if [[ "$RUN_SAMPLE_TEST" == "1" ]]; then run_sample_test; fi
if [[ "$RUN_DDP_SMOKE" == "1" ]]; then
  smoke_dir="$GROUP_DIR/smoke/ddp_resume_smoke"
  run_ddp_smoke_segment "$smoke_dir" 2 "2,4"
  run_ddp_smoke_segment "$smoke_dir" 4 "2,4" "$smoke_dir/checkpoints/checkpoint_update_000002.pth"
  checkpoint_has_update "$smoke_dir/checkpoints/checkpoint_update_000004.pth" 4
fi
if [[ "$RUN_INFERENCE_SMOKE" == "1" ]]; then run_inference_smoke; fi
if [[ "$RUN_EVAL_SMOKE" == "1" ]]; then run_eval_smoke; fi
if [[ "$RUN_SMOKE_ONLY" == "1" ]]; then
  echo "Smoke checks complete; RUN_SMOKE_ONLY=1"
  exit 0
fi
if [[ "$RUN_FULL" != "1" ]]; then
  echo "RUN_FULL=$RUN_FULL; full training not started"
  exit 0
fi

status=0
for epoch in $SEGMENT_EPOCHS; do
  if run_full_segment "$epoch" && run_val200_eval "$epoch"; then
    :
  else
    status=1
    break
  fi
done

if [[ "$status" == "0" ]]; then
  final_checkpoint="$RUN_DIR/checkpoints/checkpoint_update_010000.pth"
  final_eval_dir="$RUN_DIR/eval_epoch100_val200"
  if ! checkpoint_has_update "$final_checkpoint" 10000 || ! eval_complete "$final_eval_dir" "$VAL200_JSON"; then
    echo "Exp021 cannot be marked complete: final checkpoint or full val200 evaluation is incomplete" >&2
    status=1
  fi
fi
summarize_results
if [[ "$status" == "0" ]]; then
  touch "$RUN_DIR/.train_complete"
  touch "$GROUP_DIR/.experiment_complete"
else
  touch "$RUN_DIR/.train_failed"
fi
exit "$status"
