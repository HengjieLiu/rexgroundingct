#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

EXP_ID="017_voxtell_iso07_hu_ddp_bs4_update_matched"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
EXP003_DIR="${EXP003_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation}"
EXP006_DIR="${EXP006_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation}"
EXP007_DIR="${EXP007_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched}"
EXP011_DIR="${EXP011_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/011_voxtell_v123_e4d4_ct_normalization_ablation}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1}"
SEED="${SEED:-20260723}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EPOCHS="${EPOCHS:-100}"
WORLD_SIZE="${WORLD_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SCHEDULE_EVENTS="${SCHEDULE_EVENTS:-40000}"
SCHEDULE_NUM_WORKERS="${SCHEDULE_NUM_WORKERS:-4}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONTINUATION_MODE="${CONTINUATION_MODE:-0}"
DEFAULT_RUN_GROUP="exp017_iso07_hu_ddp_bs4_$TIMESTAMP"
if [[ "$CONTINUATION_MODE" == "1" ]]; then
  DEFAULT_RUN_GROUP="exp017_cont100_from_iso07_epoch100_$TIMESTAMP"
fi
RUN_GROUP="${RUN_GROUP:-$DEFAULT_RUN_GROUP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
RUN_DIR="$GROUP_DIR/ddp_bs4"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
ORIGINAL_SCHEDULE="${ORIGINAL_SCHEDULE:-$EXP_DIR/config/train_schedule_v123_iso07_hu_ddp_bs4_seed${SEED}_100ep_${STEPS_PER_EPOCH}steps_gb4.jsonl}"
ORIGINAL_SCHEDULE_MANIFEST="${ORIGINAL_SCHEDULE_MANIFEST:-$EXP_DIR/config/train_schedule_v123_iso07_hu_ddp_bs4_seed${SEED}_100ep_${STEPS_PER_EPOCH}steps_gb4.manifest.json}"
EXTENDED_SCHEDULE_EVENTS="${EXTENDED_SCHEDULE_EVENTS:-80000}"
EXTENDED_SCHEDULE="${EXTENDED_SCHEDULE:-$EXP_DIR/config/train_schedule_v123_iso07_hu_ddp_bs4_seed${SEED}_200ep_${STEPS_PER_EPOCH}steps_gb4.jsonl}"
EXTENDED_SCHEDULE_MANIFEST="${EXTENDED_SCHEDULE_MANIFEST:-$EXP_DIR/config/train_schedule_v123_iso07_hu_ddp_bs4_seed${SEED}_200ep_${STEPS_PER_EPOCH}steps_gb4.manifest.json}"
SCHEDULE_START_EVENT="${SCHEDULE_START_EVENT:-40000}"
SOURCE_MODEL_DIR="$EXP_DIR/config/public_voxtell_v1_1_model"
CONTINUATION_SOURCE_RUN_GROUP="${CONTINUATION_SOURCE_RUN_GROUP:-exp017_iso07_hu_ddp_bs4_20260815T092119Z}"
CONTINUATION_SOURCE_RUN_DIR="${CONTINUATION_SOURCE_RUN_DIR:-$EXP_DIR/runs/$CONTINUATION_SOURCE_RUN_GROUP/ddp_bs4}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$CONTINUATION_SOURCE_RUN_DIR/checkpoints/checkpoint_update_010000.pth}"
INIT_CHECKPOINT_EXPECTED_UPDATE="${INIT_CHECKPOINT_EXPECTED_UPDATE:-10000}"
EXPECTED_INIT_CHECKPOINT_SHA="${EXPECTED_INIT_CHECKPOINT_SHA:-f11d238ffeaf9d96fe6ba7f81f6dd134953b18b0eaaae3f5404176aa0cec6cc4}"
if [[ "$CONTINUATION_MODE" == "1" ]]; then
  ABSOLUTE_EPOCH_OFFSET="${ABSOLUTE_EPOCH_OFFSET:-100}"
  SCHEDULE="${SCHEDULE:-$EXP_DIR/config/train_schedule_v123_iso07_hu_ddp_bs4_seed${SEED}_cont100_from_iso07_epoch100_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb4.jsonl}"
  SCHEDULE_MANIFEST="${SCHEDULE_MANIFEST:-$EXP_DIR/config/train_schedule_v123_iso07_hu_ddp_bs4_seed${SEED}_cont100_from_iso07_epoch100_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb4.manifest.json}"
else
  ABSOLUTE_EPOCH_OFFSET="${ABSOLUTE_EPOCH_OFFSET:-0}"
  SCHEDULE="${SCHEDULE:-$ORIGINAL_SCHEDULE}"
  SCHEDULE_MANIFEST="${SCHEDULE_MANIFEST:-$ORIGINAL_SCHEDULE_MANIFEST}"
fi
EXPECTED_CACHE_MANIFEST_SHA="59b53ed9fcd93b8e9872bf89770a14ffd6aeed31b206af33ca751b85d4bd2640"
EXPECTED_VAL20_SHA="31557d624c47ee8c06799bf89cceb862471b34cfd47065001d6092e32d0dab5d"
EXPECTED_VAL200_SHA="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
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
MASTER_PORT="${MASTER_PORT:-29617}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
REPORT_JSON="$EXP_DIR/reports/iso07_hu_ddp_bs4_summary.json"
REPORT_MD="$EXP_DIR/reports/iso07_hu_ddp_bs4_report.md"
if [[ "$CONTINUATION_MODE" == "1" ]]; then
  REPORT_JSON="$EXP_DIR/reports/iso07_hu_ddp_bs4_continue100_from_epoch100_summary.json"
  REPORT_MD="$EXP_DIR/reports/iso07_hu_ddp_bs4_continue100_from_epoch100_report.md"
fi
EXP007_BASELINE_RUN_DIR="${EXP007_BASELINE_RUN_DIR:-$EXP007_DIR/runs/exp007_full_20260725T231624Z/ddp_bs4}"
EXP007_CONTINUATION_RUN_DIR="${EXP007_CONTINUATION_RUN_DIR:-$EXP007_DIR/runs/exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4}"
EXP011_LINEAR_HU_SUMMARY="${EXP011_LINEAR_HU_SUMMARY:-$EXP011_DIR/runs/exp011_ct_norm_e5d4_20260728T225333Z/v123_e5d4_clip1024_linear/eval_epoch100_val200/reports/val_quick_global_eval_summary.json}"

read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -ne "$WORLD_SIZE" ]]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but WORLD_SIZE=$WORLD_SIZE: $GPU_LIST" >&2
  exit 1
fi
DDP_CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$EXP_DIR/reports" "$GROUP_DIR" "$RUN_DIR/logs"
echo "$RUN_GROUP" > "$EXP_DIR/config/latest_run_group.txt"
ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest"

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
    "cases": 3492,
    "targets": 8068,
    "empty_targets": 0,
    "foreground_fallback_targets": 0,
    "geometry_or_target_mismatches": 0,
    "materialized_hu_header_failures": 0,
    "normalization_failures": 0,
    "image_padding_value": -1.0,
    "target_padding_value": 0,
    "preprocess_id": "crop_clip1024_linear_iso07_v1",
}
observed = {
    "cases": manifest.get("cases"),
    "targets": manifest.get("targets"),
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
spacing = [float(value) for value in manifest.get("target_spacing_zyx_mm", [])]
if spacing != [0.7, 0.7, 0.7]:
    raise SystemExit(f"Iso07 cache spacing mismatch: {spacing}")
print(f"Iso07 cache audit accepted: {observed}")
PY
}

generate_schedule_path_if_needed() {
  local schedule="$1" manifest="$2" events="$3" label="$4"
  if line_count_matches "$schedule" "$events"; then
    echo "Schedule ready: $schedule"
    return 0
  fi
  echo "Generating iso07 DDP schedule: label=$label events=$events path=$schedule"
  rm -f "$schedule" "$manifest"
  python /workspace/scripts/rexgroundingct/prepare_training_schedule.py \
    --output-jsonl "$schedule" \
    --manifest-json "$manifest" \
    --seed "$SEED" \
    --events "$events" \
    --patch-size 192 192 192 \
    --foreground-oversample-prob 0.85 \
    --require-positive-crop \
    --preprocessed-cache-dir "$CACHE_ROOT" \
    --num-workers "$SCHEDULE_NUM_WORKERS" \
    2>&1 | tee "$EXP_DIR/logs/prepare_schedule_${label}_$TIMESTAMP.log"
}

verify_schedule() {
  python - "$SCHEDULE" "$SCHEDULE_MANIFEST" "$SCHEDULE_EVENTS" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

schedule = Path(sys.argv[1])
manifest = Path(sys.argv[2])
expected_events = int(sys.argv[3])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with schedule.open("rb") as handle:
    event_count = sum(1 for _ in handle)
if event_count != expected_events:
    raise SystemExit(f"Schedule event count mismatch: expected={expected_events} observed={event_count}")

record = json.loads(manifest.read_text()) if manifest.is_file() else {}
record.update(
    {
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": "017_voxtell_iso07_hu_ddp_bs4_update_matched",
        "preprocess_id": "crop_clip1024_linear_iso07_v1",
        "ddp_world_size": 4,
        "effective_global_batch_size": 4,
        "event_consumption_rule": "event_index = update * 4 + rank",
        "output_jsonl": str(schedule),
        "output_jsonl_sha256": sha256(schedule),
        "events": event_count,
    }
)
manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

verify_schedule_prefix() {
  python - "$ORIGINAL_SCHEDULE" "$EXTENDED_SCHEDULE" "$SCHEDULE_EVENTS" <<'PY'
import sys
from pathlib import Path

original = Path(sys.argv[1])
extended = Path(sys.argv[2])
expected = int(sys.argv[3])

with original.open("rb") as lhs, extended.open("rb") as rhs:
    for index in range(expected):
        original_line = lhs.readline()
        extended_line = rhs.readline()
        if not original_line:
            raise SystemExit(f"Original schedule ended before line {index}")
        if original_line != extended_line:
            raise SystemExit(f"Extended schedule prefix differs at line {index}")
print(f"Extended schedule prefix matches original for {expected} events")
PY
}

prepare_continuation_schedule() {
  generate_schedule_path_if_needed \
    "$ORIGINAL_SCHEDULE" "$ORIGINAL_SCHEDULE_MANIFEST" "$SCHEDULE_EVENTS" \
    "iso07_ddp40k"
  generate_schedule_path_if_needed \
    "$EXTENDED_SCHEDULE" "$EXTENDED_SCHEDULE_MANIFEST" "$EXTENDED_SCHEDULE_EVENTS" \
    "iso07_ddp80k"
  verify_schedule_prefix
  python /workspace/scripts/rexgroundingct/slice_training_schedule.py \
    --source-jsonl "$EXTENDED_SCHEDULE" \
    --start-event "$SCHEDULE_START_EVENT" \
    --events "$SCHEDULE_EVENTS" \
    --output-jsonl "$SCHEDULE" \
    --manifest-json "$SCHEDULE_MANIFEST"
  python - \
    "$SCHEDULE_MANIFEST" \
    "$ORIGINAL_SCHEDULE" \
    "$ORIGINAL_SCHEDULE_MANIFEST" \
    "$EXTENDED_SCHEDULE" \
    "$EXTENDED_SCHEDULE_MANIFEST" \
    "$SCHEDULE_START_EVENT" \
    "$WORLD_SIZE" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

from common import sha256_file, write_json

manifest = Path(sys.argv[1])
original = Path(sys.argv[2])
original_manifest = Path(sys.argv[3])
extended = Path(sys.argv[4])
extended_manifest = Path(sys.argv[5])
start_event = int(sys.argv[6])
world_size = int(sys.argv[7])

record = json.loads(manifest.read_text())
record.update(
    {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": "017_voxtell_iso07_hu_ddp_bs4_update_matched",
        "phase": "continue100_from_iso07_epoch100",
        "purpose": "exp017 iso07 continuation events 40000..79999 rewritten to zero-based line order",
        "preprocess_id": "crop_clip1024_linear_iso07_v1",
        "source_original_schedule": str(original),
        "source_original_schedule_sha256": sha256_file(original),
        "source_original_manifest": str(original_manifest),
        "source_original_manifest_sha256": (
            sha256_file(original_manifest) if original_manifest.is_file() else None
        ),
        "source_extended_schedule": str(extended),
        "source_extended_schedule_sha256": sha256_file(extended),
        "source_extended_manifest": str(extended_manifest),
        "source_extended_manifest_sha256": (
            sha256_file(extended_manifest) if extended_manifest.is_file() else None
        ),
        "source_event_start_inclusive": start_event,
        "source_event_end_exclusive": start_event + int(record["events"]),
        "event_index_policy": "source event_index rewritten to 0..39999; original index stored as source_event_index",
        "ddp_world_size": world_size,
        "effective_global_batch_size": world_size,
        "event_consumption_rule": "event_index = update * 4 + rank",
        "positive_crop_policy": "events come from --require-positive-crop schedule generation; sliced events preserve selected positive masks",
    }
)
write_json(manifest, record)
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

prepare_embeddings_and_model() {
  if [[ ! -f "$EMBEDDINGS" ]]; then
    for candidate in "$EXP006_DIR/config/rex_text_embeddings.npz" "$EXP003_DIR/config/rex_text_embeddings.npz"; do
      if [[ -f "$candidate" ]]; then
        cp "$candidate" "$EMBEDDINGS"
        break
      fi
    done
  fi
  if [[ ! -f "$EMBEDDINGS" ]]; then
    python /workspace/scripts/rexgroundingct/precompute_text_embeddings.py \
      --splits train val --gpu 0 --output "$EMBEDDINGS" \
      2>&1 | tee "$EXP_DIR/logs/precompute_text_embeddings_$TIMESTAMP.log"
  fi

  BASE_MODEL_DIR="$(python -c 'from train_text_conditioned_voxtell import resolve_model_dir; print(resolve_model_dir(None))')"
  [[ -f "$BASE_MODEL_DIR/plans.json" ]] || { echo "Missing VoxTell plans: $BASE_MODEL_DIR/plans.json" >&2; exit 1; }
  [[ -f "$BASE_MODEL_DIR/fold_0/checkpoint_final.pth" ]] || { echo "Missing VoxTell checkpoint: $BASE_MODEL_DIR/fold_0/checkpoint_final.pth" >&2; exit 1; }
  mkdir -p "$SOURCE_MODEL_DIR"
  cp "$BASE_MODEL_DIR/plans.json" "$SOURCE_MODEL_DIR/plans.json"
  python - "$BASE_MODEL_DIR" "$SOURCE_MODEL_DIR" "$EXP_DIR/config/public_voxtell_v1_1_provenance.json" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
source = Path(sys.argv[2])
output = Path(sys.argv[3])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

checkpoint = base / "fold_0" / "checkpoint_final.pth"
payload = {
    "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source_model_dir": str(base),
    "snapshot_model_dir": str(source),
    "plans_sha256": sha256(base / "plans.json"),
    "checkpoint_sha256": sha256(checkpoint),
    "load_policy": "public VoxTell v1.1 weights only",
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

verify_continuation_source_checkpoint() {
  [[ "$CONTINUATION_MODE" == "1" ]] || return 0
  require_sha "$INIT_CHECKPOINT" "$EXPECTED_INIT_CHECKPOINT_SHA" "exp017 continuation init checkpoint"
  mkdir -p "$GROUP_DIR/config"
  python - "$INIT_CHECKPOINT" "$INIT_CHECKPOINT_EXPECTED_UPDATE" "$GROUP_DIR/config/init_checkpoint_provenance.json" "$CONTINUATION_SOURCE_RUN_DIR" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1])
expected_update = int(sys.argv[2])
output = Path(sys.argv[3])
source_run_dir = Path(sys.argv[4])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
observed_update = int(payload.get("global_update", -1))
if observed_update != expected_update:
    raise SystemExit(
        f"Continuation init checkpoint update mismatch: "
        f"expected={expected_update} observed={observed_update}"
    )
record = {
    "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source_run_dir": str(source_run_dir),
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "checkpoint_bytes": checkpoint.stat().st_size,
    "checkpoint_global_update": observed_update,
    "checkpoint_epoch": int(payload.get("epoch", -1)),
    "load_policy": "network weights only via --init-checkpoint; optimizer, scaler, scheduler, RNG-controlled run state, and update counter reset",
    "optimizer_state_present_but_not_loaded": "optimizer" in payload,
    "grad_scaler_state_present_but_not_loaded": "grad_scaler" in payload,
    "source_optimizer_config": payload.get("optimizer_config"),
    "source_loss_config": payload.get("loss_config"),
    "source_preprocessing_config": payload.get("preprocessing_config"),
    "source_model_spec": payload.get("model_spec"),
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

common_train_args() {
  printf '%s\n' \
    --mode train \
    --experiment-id "$EXP_ID" \
    --config "$CONFIG_SNAPSHOT" \
    --exp-dir "$EXP_DIR" \
    --model-dir "$BASE_MODEL_DIR" \
    --embeddings "$EMBEDDINGS" \
    --seed "$SEED" \
    --patch-size 192 192 192 \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
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
    --preprocessed-cache-dir "$CACHE_ROOT" \
    --sample-schedule "$SCHEDULE" \
    --distributed \
    --allow-experimental-train
}

run_sample_test() {
  local sample_dir="$GROUP_DIR/smoke/sample_test"
  mkdir -p "$sample_dir/logs"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    --mode sample-test \
    --experiment-id "$EXP_ID" \
    --config "$CONFIG_SNAPSHOT" \
    --exp-dir "$EXP_DIR" \
    --run-dir "$sample_dir" \
    --gpu 0 \
    --seed "$SEED" \
    --patch-size 192 192 192 \
    --sample-schedule "$SCHEDULE" \
    --preprocessed-cache-dir "$CACHE_ROOT" \
    --require-positive-crop \
    --loss-mode empty_bce_only \
    --sample-test-steps 8 \
    2>&1 | tee "$sample_dir/logs/sample_test.log"
}

run_ddp_smoke_segment() {
  local smoke_dir="$1" target="$2" checkpoint_updates="$3"
  local resume_checkpoint="${4:-}"
  local log="$smoke_dir/logs/train_to_update_${target}.log"
  mkdir -p "$smoke_dir/logs"
  mapfile -t args < <(common_train_args)
  local resume_args=()
  local init_args=()
  if [[ -n "$resume_checkpoint" ]]; then
    resume_args=(--resume-checkpoint "$resume_checkpoint")
  elif [[ "$CONTINUATION_MODE" == "1" ]]; then
    init_args=(--init-checkpoint "$INIT_CHECKPOINT")
  fi
  {
    printf 'init_args=%s\n' "${init_args[*]:-none}"
    printf 'resume_args=%s\n' "${resume_args[*]:-none}"
    CUDA_VISIBLE_DEVICES="$DDP_CUDA_VISIBLE_DEVICES" python -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$WORLD_SIZE" \
      /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
      "${args[@]}" \
      --run-dir "$smoke_dir" \
      --epochs 1 \
      --steps-per-epoch 4 \
      --target-global-update "$target" \
      --checkpoint-every-updates 0 \
      --checkpoint-updates "$checkpoint_updates" \
      --latest-checkpoint-every-updates 0 \
      --no-materialize-final-model \
      "${init_args[@]}" \
      "${resume_args[@]}"
  } >"$log" 2>&1
}

checkpoint_has_update() {
  local checkpoint="$1" expected="$2"
  [[ -s "$checkpoint" ]] || return 1
  python - "$checkpoint" "$expected" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
expected = int(sys.argv[2])
checkpoint = torch.load(path, map_location="cpu")
raise SystemExit(0 if int(checkpoint.get("global_update", -1)) == expected else 1)
PY
}

checkpoint_stable() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  local a b
  a="$(stat -c '%s:%Y' "$path")"
  sleep "$STABILITY_SECONDS"
  [[ -s "$path" ]] || return 1
  b="$(stat -c '%s:%Y' "$path")"
  [[ "$a" == "$b" ]]
}

materialize_model() {
  local checkpoint="$1" model_dir="$2"
  mkdir -p "$model_dir/fold_0"
  cp "$SOURCE_MODEL_DIR/plans.json" "$model_dir/plans.json"
  local target="$model_dir/fold_0/checkpoint_final.pth"
  if [[ ! -f "$target" || "$(stat -c %s "$target")" != "$(stat -c %s "$checkpoint")" ]]; then
    local tmp="$target.tmp.$$"
    rm -f "$tmp"
    ln "$checkpoint" "$tmp" 2>/dev/null || cp "$checkpoint" "$tmp"
    mv -f "$tmp" "$target"
  fi
}

write_subset_json() {
  local source="$1" target="$2" count="$3"
  python - "$source" "$target" "$count" <<'PY'
import json
import sys
from pathlib import Path
source = Path(sys.argv[1])
target = Path(sys.argv[2])
count = int(sys.argv[3])
data = json.loads(source.read_text())
data["test"] = data["test"][:count]
target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
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
    name = entry["name"]
    pred_path = prediction_dir / name
    gt_path = seg_dir / name
    if not pred_path.is_file():
        raise SystemExit(f"Missing prediction: {pred_path}")
    pred_shape = nib.load(str(pred_path)).shape
    gt_shape = nib.load(str(gt_path)).shape
    if pred_shape != gt_shape:
        raise SystemExit(f"{name}: prediction shape {pred_shape} != label shape {gt_shape}")
print(f"Prediction shape check passed for {len(dataset['test'])} case(s)")
PY
}

run_cached_inference_shard() {
  local dataset_json="$1" model_dir="$2" output_dir="$3" status_dir="$4" log_dir="$5" gpu="$6" shard_count="$7" shard_index="$8"
  mkdir -p "$output_dir" "$status_dir" "$log_dir"
  EVAL_LOCK_DISABLE=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
      --split val \
      --dataset-json "$dataset_json" \
      --gpu 0 \
      --model-dir "$model_dir" \
      --embeddings "$EMBEDDINGS" \
      --output-dir "$output_dir" \
      --output-type mask \
      --threshold 0.5 \
      --preprocessed-cache-dir "$CACHE_ROOT" \
      --num-shards "$shard_count" \
      --shard-index "$shard_index" \
      --status-json "$status_dir/shard${shard_index}_gpu${gpu}.json" \
      >"$log_dir/inference_shard${shard_index}_gpu${gpu}.log" 2>&1
}

run_inference_smoke() {
  local smoke_dir="$GROUP_DIR/smoke/ddp_resume_smoke"
  local smoke_model="$GROUP_DIR/smoke/model_epoch000"
  local smoke_eval="$GROUP_DIR/smoke/one_case_inference"
  local subset="$GROUP_DIR/smoke/one_case_val20.json"
  mkdir -p "$GROUP_DIR/smoke" "$smoke_eval"
  materialize_model "$smoke_dir/checkpoints/checkpoint_update_000004.pth" "$smoke_model"
  write_subset_json "$VAL20_JSON" "$subset" 1
  run_cached_inference_shard "$subset" "$smoke_model" "$smoke_eval/predictions" "$smoke_eval/status" "$smoke_eval/logs" "${GPUS[0]}" 1 0
  verify_prediction_shapes "$subset" "$smoke_eval/predictions"
}

run_eval_smoke() {
  local smoke_model="$GROUP_DIR/smoke/model_epoch000"
  local smoke_eval="$GROUP_DIR/smoke/four_case_sharded_eval"
  local subset="$GROUP_DIR/smoke/four_case_val20.json"
  write_subset_json "$VAL20_JSON" "$subset" 4
  mkdir -p "$smoke_eval/predictions" "$smoke_eval/status" "$smoke_eval/logs"
  pids=()
  for shard in 0 1 2 3; do
    run_cached_inference_shard "$subset" "$smoke_model" "$smoke_eval/predictions" "$smoke_eval/status" "$smoke_eval/logs" "${GPUS[$shard]}" 4 "$shard" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  [[ "$status" == "0" ]] || { echo "One or more eval-smoke shards failed" >&2; return 1; }
  verify_prediction_shapes "$subset" "$smoke_eval/predictions"
}

eval_complete() {
  local eval_dir="$1" dataset_json="$2"
  python - "$eval_dir" "$dataset_json" <<'PY'
import json
import sys
from pathlib import Path

root, dataset = map(Path, sys.argv[1:3])
summary = root / "reports" / "val_quick_global_eval_summary.json"
eval_json = root / "eval" / "val_quick_global_eval.json"
predictions = root / "predictions"
if not (summary.is_file() and eval_json.is_file() and predictions.is_dir() and dataset.is_file()):
    raise SystemExit(1)
entries = json.loads(dataset.read_text())["test"]
summary_data = json.loads(summary.read_text())
if int(summary_data.get("total_cases", -1)) != len(entries):
    raise SystemExit(1)
if int(summary_data.get("total_findings", -1)) != sum(len(e["findings"]) for e in entries):
    raise SystemExit(1)
if any(not (predictions / e["name"]).is_file() for e in entries):
    raise SystemExit(1)
PY
}

lock_age_seconds() {
  local lock="$1"
  python - "$lock" <<'PY'
import sys
import time
from pathlib import Path
path = Path(sys.argv[1])
print(max(0.0, time.time() - path.stat().st_mtime))
PY
}

acquire_lock() {
  local eval_dir="$1"
  local lock="$eval_dir/.eval.lock"
  mkdir -p "$eval_dir"
  while true; do
    if (set -o noclobber; printf '{"pid":%s,"created":"%s","phase":"exp017_segment_eval"}\n' "$$" "$(date -u +%FT%TZ)" > "$lock") 2>/dev/null; then
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
      continue
    fi
    echo "Eval already locked, waiting: $lock"
    sleep 60
  done
}

prediction_count() {
  local eval_dir="$1"
  if [[ ! -d "$eval_dir/predictions" ]]; then
    echo 0
    return 0
  fi
  find "$eval_dir/predictions" -maxdepth 1 -name '*.nii.gz' | wc -l
}

expected_case_count() {
  local dataset="$1"
  python - "$dataset" <<'PY'
import json
import sys
from pathlib import Path
print(len(json.loads(Path(sys.argv[1]).read_text()).get("test", [])))
PY
}

summarize_results() {
  local exp006_summary="$EXP006_DIR/runs/latest/v123_cached_e5_d4/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
  local exp006_training="$EXP006_DIR/runs/latest/v123_cached_e5_d4/reports/training_metrics.json"
  local phase1_epoch50_summary="$CONTINUATION_SOURCE_RUN_DIR/eval_epoch050_val200/reports/val_quick_global_eval_summary.json"
  local phase1_epoch100_summary="$CONTINUATION_SOURCE_RUN_DIR/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
  local phase1_training="$CONTINUATION_SOURCE_RUN_DIR/reports/training_metrics.json"
  local reference_args=()
  local reference_training_args=()
  if [[ "$CONTINUATION_MODE" == "1" ]]; then
    [[ -f "$phase1_epoch100_summary" ]] && reference_args+=(--reference "exp017_phase1_epoch100=$phase1_epoch100_summary")
    [[ -f "$phase1_epoch50_summary" ]] && reference_args+=(--reference "exp017_phase1_best_epoch50=$phase1_epoch50_summary")
    for epoch in 25 50 75 100; do
      local exp007_cont_summary="$EXP007_CONTINUATION_RUN_DIR/eval_epoch$(printf '%03d' "$epoch")_val200/reports/val_quick_global_eval_summary.json"
      [[ -f "$exp007_cont_summary" ]] && reference_args+=(--reference "exp007_phase2_relative_epoch${epoch}=$exp007_cont_summary")
    done
    local exp007_epoch100_summary="$EXP007_BASELINE_RUN_DIR/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
    [[ -f "$exp007_epoch100_summary" ]] && reference_args+=(--reference "exp007_phase1_epoch100=$exp007_epoch100_summary")
    [[ -f "$phase1_training" ]] && reference_training_args+=(--reference-training "exp017_phase1_iso07_hu_ddp_bs4=$phase1_training")
    [[ -f "$EXP007_CONTINUATION_RUN_DIR/reports/training_metrics.json" ]] && reference_training_args+=(--reference-training "exp007_phase2_native_zscore_ddp_bs4=$EXP007_CONTINUATION_RUN_DIR/reports/training_metrics.json")
    [[ -f "$EXP007_BASELINE_RUN_DIR/reports/training_metrics.json" ]] && reference_training_args+=(--reference-training "exp007_phase1_native_zscore_ddp_bs4=$EXP007_BASELINE_RUN_DIR/reports/training_metrics.json")
    python /workspace/scripts/rexgroundingct/summarize_007_continuation_results.py \
      --experiment-id "$EXP_ID" \
      --report-title "Experiment 017 Iso07 HU DDP Batch4 Continuation Report" \
      --training-label "exp017_iso07_hu_ddp_bs4_cont100_e5_d4" \
      --run-type "continue100_from_iso07_epoch100" \
      --exp-dir "$EXP_DIR" \
      --group-dir "$GROUP_DIR" \
      --source-run-dir "$CONTINUATION_SOURCE_RUN_DIR" \
      --source-checkpoint "$INIT_CHECKPOINT" \
      --absolute-epoch-offset "$ABSOLUTE_EPOCH_OFFSET" \
      --schedule-manifest "$SCHEDULE_MANIFEST" \
      --output-json "$REPORT_JSON" \
      --output-md "$REPORT_MD" \
      "${reference_args[@]}" \
      "${reference_training_args[@]}" || true
  else
    for epoch in 25 50 75 100; do
      local exp007_epoch_summary="$EXP007_BASELINE_RUN_DIR/eval_epoch$(printf '%03d' "$epoch")_val200/reports/val_quick_global_eval_summary.json"
      [[ -f "$exp007_epoch_summary" ]] && reference_args+=(--reference "exp007_epoch${epoch}_native_zscore=$exp007_epoch_summary")
    done
    [[ -f "$EXP011_LINEAR_HU_SUMMARY" ]] && reference_args+=(--reference "exp011_native_linear_hu_e5d4_epoch100=$EXP011_LINEAR_HU_SUMMARY")
    [[ -f "$exp006_summary" ]] && reference_args+=(--reference "exp006_native_zscore_e5d4_epoch100=$exp006_summary")
    [[ -f "$EXP007_BASELINE_RUN_DIR/reports/training_metrics.json" ]] && reference_training_args+=(--reference-training "exp007_ddp_bs4_native_zscore=$EXP007_BASELINE_RUN_DIR/reports/training_metrics.json")
    [[ -f "$exp006_training" ]] && reference_training_args+=(--reference-training "exp006_bs1_e5_d4=$exp006_training")
    python /workspace/scripts/rexgroundingct/summarize_007_results.py \
      --experiment-id "$EXP_ID" \
      --report-title "Experiment 017 Iso07 HU DDP Batch4 Report" \
      --training-label "exp017_iso07_hu_ddp_bs4_e5_d4" \
      --training-time-note "Training time excludes validation pauses between exp017 segments." \
      --checkpoint-role "25=same optimizer epoch as exp007 epoch25" \
      --checkpoint-role "50=same optimizer epoch as exp007 epoch50" \
      --checkpoint-role "75=same optimizer epoch as exp007 epoch75" \
      --checkpoint-role "100=headline comparison to exp007 epoch100" \
      --exp-dir "$EXP_DIR" \
      --group-dir "$GROUP_DIR" \
      --output-json "$REPORT_JSON" \
      --output-md "$REPORT_MD" \
      "${reference_args[@]}" \
      "${reference_training_args[@]}" || true
  fi
}

run_inference_shard() {
  local epoch="$1" gpu="$2" shard_count="$3" shard_index="$4"
  local eval_dir model_dir status_json shard_log
  eval_dir="$RUN_DIR/eval_epoch$(printf '%03d' "$epoch")_val200"
  model_dir="$RUN_DIR/model_epoch$(printf '%03d' "$epoch")"
  mkdir -p "$eval_dir/logs" "$eval_dir/status" "$eval_dir/predictions"
  status_json="$eval_dir/status/shard${shard_index}_gpu${gpu}.json"
  shard_log="$eval_dir/logs/inference_shard${shard_index}_gpu${gpu}.log"
  echo "Starting val200 shard epoch=$epoch shard=$shard_index/$shard_count gpu=$gpu log=$shard_log"
  EVAL_LOCK_HELD=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
      --split val \
      --dataset-json "$VAL200_JSON" \
      --gpu 0 \
      --model-dir "$model_dir" \
      --embeddings "$EMBEDDINGS" \
      --output-dir "$eval_dir/predictions" \
      --output-type mask \
      --threshold 0.5 \
      --preprocessed-cache-dir "$CACHE_ROOT" \
      --num-shards "$shard_count" \
      --shard-index "$shard_index" \
      --status-json "$status_json" \
      >"$shard_log" 2>&1
}

run_val200_eval() {
  local epoch="$1"
  local update="$((epoch * STEPS_PER_EPOCH))"
  local checkpoint="$RUN_DIR/checkpoints/checkpoint_update_$(printf '%06d' "$update").pth"
  local model_dir="$RUN_DIR/model_epoch$(printf '%03d' "$epoch")"
  local eval_dir="$RUN_DIR/eval_epoch$(printf '%03d' "$epoch")_val200"
  local expected existing lock eval_log
  expected="$(expected_case_count "$VAL200_JSON")"
  existing="$(prediction_count "$eval_dir")"

  if eval_complete "$eval_dir" "$VAL200_JSON"; then
    echo "Already complete: epoch=$epoch val200"
    summarize_results
    return 0
  fi
  checkpoint_stable "$checkpoint" || { echo "Checkpoint not stable: $checkpoint" >&2; return 1; }
  lock="$(acquire_lock "$eval_dir")"
  materialize_model "$checkpoint" "$model_dir"
  mkdir -p "$eval_dir/logs" "$eval_dir/predictions"
  eval_log="$eval_dir/logs/eval_$(date -u +%Y%m%dT%H%M%SZ).log"
  {
    echo "start epoch=$epoch val200 gpus=$GPU_LIST existing_predictions=$existing/$expected"
    pids=()
    for shard in 0 1 2 3; do
      run_inference_shard "$epoch" "${GPUS[$shard]}" 4 "$shard" &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    if [[ "$status" != "0" ]]; then
      echo "one or more inference shards failed"
      exit 1
    fi
    verify_prediction_shapes "$VAL200_JSON" "$eval_dir/predictions"
    echo "prediction_count=$(prediction_count "$eval_dir") expected=$expected"
    EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$VAL200_JSON" \
      EVAL_LABEL="Experiment 017 iso07 HU DDP bs4 relative epoch $epoch absolute epoch $((ABSOLUTE_EPOCH_OFFSET + epoch)) fixed val200 threshold 0.5" NUM_WORKERS=8 \
      bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$eval_dir" val
    echo "done epoch=$epoch val200"
  } >"$eval_log" 2>&1 || {
    rm -f "$lock"
    echo "Val200 eval failed: epoch=$epoch log=$eval_log" >&2
    return 1
  }
  rm -f "$lock"
  summarize_results
}

run_full_segment() {
  local epoch="$1"
  local target_update="$((epoch * STEPS_PER_EPOCH))"
  local checkpoint="$RUN_DIR/checkpoints/checkpoint_update_$(printf '%06d' "$target_update").pth"
  local resume_args=()
  local init_args=()
  local log="$RUN_DIR/logs/train_segment_to_epoch$(printf '%03d' "$epoch").log"
  mapfile -t args < <(common_train_args)
  if checkpoint_has_update "$checkpoint" "$target_update"; then
    echo "Training segment already complete: epoch=$epoch update=$target_update"
    return 0
  fi
  if [[ "$target_update" -gt 2500 ]]; then
    local prev_update="$((target_update - 2500))"
    local prev_checkpoint="$RUN_DIR/checkpoints/checkpoint_update_$(printf '%06d' "$prev_update").pth"
    checkpoint_has_update "$prev_checkpoint" "$prev_update" || {
      echo "Missing previous checkpoint for resume: $prev_checkpoint" >&2
      return 1
    }
    resume_args=(--resume-checkpoint "$prev_checkpoint")
  elif [[ "$CONTINUATION_MODE" == "1" ]]; then
    init_args=(--init-checkpoint "$INIT_CHECKPOINT")
  fi
  echo "Starting DDP segment to epoch=$epoch update=$target_update log=$log"
  {
    printf 'init_args=%s\n' "${init_args[*]:-none}"
    printf 'resume_args=%s\n' "${resume_args[*]:-none}"
    CUDA_VISIBLE_DEVICES="$DDP_CUDA_VISIBLE_DEVICES" python -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$WORLD_SIZE" \
      /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
      "${args[@]}" \
      --run-dir "$RUN_DIR" \
      --epochs "$EPOCHS" \
      --steps-per-epoch "$STEPS_PER_EPOCH" \
      --target-global-update "$target_update" \
      --checkpoint-every-updates 0 \
      --checkpoint-updates "$CHECKPOINT_UPDATES" \
      --latest-checkpoint-every-updates 500 \
      --no-materialize-final-model \
      "${init_args[@]}" \
      "${resume_args[@]}"
  } >"$log" 2>&1
  checkpoint_has_update "$checkpoint" "$target_update"
  cp "$RUN_DIR/reports/training_metrics.json" \
    "$RUN_DIR/reports/training_metrics_segment_epoch$(printf '%03d' "$epoch").json"
}

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment "$EXP_ID" --exp-dir "$EXP_DIR" --overwrite
python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val --json "$EXP_DIR/config/train_val_ct_readiness.json"
verify_cache
require_sha "$VAL20_JSON" "$EXPECTED_VAL20_SHA" "val20 JSON"
require_sha "$VAL200_JSON" "$EXPECTED_VAL200_SHA" "val200 JSON"
if [[ "$CONTINUATION_MODE" == "1" ]]; then
  prepare_continuation_schedule
else
  generate_schedule_path_if_needed "$SCHEDULE" "$SCHEDULE_MANIFEST" "$SCHEDULE_EVENTS" "iso07_ddp40k"
fi
verify_schedule
prepare_embeddings_and_model
verify_continuation_source_checkpoint

if [[ "$RUN_STATIC_ONLY" == "1" ]]; then
  echo "Static checks complete; RUN_STATIC_ONLY=1"
  exit 0
fi

if [[ "$RUN_SAMPLE_TEST" == "1" ]]; then
  run_sample_test
fi

if [[ "$RUN_DDP_SMOKE" == "1" ]]; then
  smoke_dir="$GROUP_DIR/smoke/ddp_resume_smoke"
  run_ddp_smoke_segment "$smoke_dir" 2 "2,4"
  run_ddp_smoke_segment "$smoke_dir" 4 "2,4" "$smoke_dir/checkpoints/checkpoint_update_000002.pth"
  checkpoint_has_update "$smoke_dir/checkpoints/checkpoint_update_000004.pth" 4
fi

if [[ "$RUN_INFERENCE_SMOKE" == "1" ]]; then
  run_inference_smoke
fi

if [[ "$RUN_EVAL_SMOKE" == "1" ]]; then
  run_eval_smoke
fi

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
  if run_full_segment "$epoch"; then
    if ! run_val200_eval "$epoch"; then
      status=1
      break
    fi
  else
    status=1
    break
  fi
done

summarize_results
if [[ "$status" == "0" ]]; then
  touch "$RUN_DIR/.train_complete"
  touch "$GROUP_DIR/.experiment_complete"
else
  touch "$RUN_DIR/.train_failed"
fi
echo "Experiment 017 group dir: $GROUP_DIR"
exit "$status"
