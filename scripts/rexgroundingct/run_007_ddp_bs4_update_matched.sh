#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

EXP_ID="007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
EXP003_DIR="${EXP003_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation}"
EXP006_DIR="${EXP006_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1}"
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
DEFAULT_RUN_GROUP="exp007_ddp_bs4_update_matched_$TIMESTAMP"
if [[ "$CONTINUATION_MODE" == "1" ]]; then
  DEFAULT_RUN_GROUP="exp007_cont100_from_ddp100_$TIMESTAMP"
fi
RUN_GROUP="${RUN_GROUP:-$DEFAULT_RUN_GROUP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
RUN_DIR="$GROUP_DIR/ddp_bs4"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
SOURCE_PREFIX_SCHEDULE="${SOURCE_PREFIX_SCHEDULE:-$EXP003_DIR/config/train_schedule_v123_opt_poscrop_emptyloss_seed${SEED}_100ep_100steps_gb1.jsonl}"
ORIGINAL_DDP_SCHEDULE="${ORIGINAL_DDP_SCHEDULE:-$EXP_DIR/config/train_schedule_v123_ddp_bs4_seed${SEED}_100ep_${STEPS_PER_EPOCH}steps_gb4.jsonl}"
ORIGINAL_DDP_SCHEDULE_MANIFEST="${ORIGINAL_DDP_SCHEDULE_MANIFEST:-$EXP_DIR/config/train_schedule_v123_ddp_bs4_seed${SEED}_100ep_${STEPS_PER_EPOCH}steps_gb4.manifest.json}"
EXTENDED_SCHEDULE_EVENTS="${EXTENDED_SCHEDULE_EVENTS:-80000}"
EXTENDED_SCHEDULE="${EXTENDED_SCHEDULE:-$EXP_DIR/config/train_schedule_v123_ddp_bs4_seed${SEED}_200ep_${STEPS_PER_EPOCH}steps_gb4.jsonl}"
EXTENDED_SCHEDULE_MANIFEST="${EXTENDED_SCHEDULE_MANIFEST:-$EXP_DIR/config/train_schedule_v123_ddp_bs4_seed${SEED}_200ep_${STEPS_PER_EPOCH}steps_gb4.manifest.json}"
SCHEDULE_START_EVENT="${SCHEDULE_START_EVENT:-40000}"
CONTINUATION_SOURCE_RUN_GROUP="${CONTINUATION_SOURCE_RUN_GROUP:-exp007_full_20260725T231624Z}"
CONTINUATION_SOURCE_RUN_DIR="${CONTINUATION_SOURCE_RUN_DIR:-$EXP_DIR/runs/$CONTINUATION_SOURCE_RUN_GROUP/ddp_bs4}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$CONTINUATION_SOURCE_RUN_DIR/checkpoints/checkpoint_update_010000.pth}"
INIT_CHECKPOINT_EXPECTED_UPDATE="${INIT_CHECKPOINT_EXPECTED_UPDATE:-10000}"
if [[ "$CONTINUATION_MODE" == "1" ]]; then
  ABSOLUTE_EPOCH_OFFSET="${ABSOLUTE_EPOCH_OFFSET:-100}"
else
  ABSOLUTE_EPOCH_OFFSET="${ABSOLUTE_EPOCH_OFFSET:-0}"
fi
REPORT_JSON="$EXP_DIR/reports/ddp_bs4_update_matched_summary.json"
REPORT_MD="$EXP_DIR/reports/ddp_bs4_update_matched_report.md"
SCHEDULE="$ORIGINAL_DDP_SCHEDULE"
SCHEDULE_MANIFEST="$ORIGINAL_DDP_SCHEDULE_MANIFEST"
if [[ "$CONTINUATION_MODE" == "1" ]]; then
  SCHEDULE="$EXP_DIR/config/train_schedule_v123_ddp_bs4_seed${SEED}_cont100_from_ddp100_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb4.jsonl"
  SCHEDULE_MANIFEST="$EXP_DIR/config/train_schedule_v123_ddp_bs4_seed${SEED}_cont100_from_ddp100_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb4.manifest.json"
  REPORT_JSON="$EXP_DIR/reports/ddp_bs4_continue100_from_epoch100_summary.json"
  REPORT_MD="$EXP_DIR/reports/ddp_bs4_continue100_from_epoch100_report.md"
fi
SOURCE_MODEL_DIR="$EXP_DIR/config/public_voxtell_v1_1_model"
EXPECTED_PREFIX_SHA="f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776"
EXPECTED_CACHE_MANIFEST_SHA="fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3"
EXPECTED_VAL20_SHA="31557d624c47ee8c06799bf89cceb862471b34cfd47065001d6092e32d0dab5d"
EXPECTED_VAL200_SHA="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-2500,5000,7500,10000}"
SEGMENT_EPOCHS="${SEGMENT_EPOCHS:-25 50 75 100}"
RUN_SAMPLE_TEST="${RUN_SAMPLE_TEST:-1}"
RUN_DDP_SMOKE="${RUN_DDP_SMOKE:-1}"
RUN_INFERENCE_SMOKE="${RUN_INFERENCE_SMOKE:-1}"
RUN_STATIC_ONLY="${RUN_STATIC_ONLY:-0}"
RUN_SMOKE_ONLY="${RUN_SMOKE_ONLY:-0}"
RUN_FULL="${RUN_FULL:-1}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-43200}"
STABILITY_SECONDS="${STABILITY_SECONDS:-10}"
MASTER_PORT="${MASTER_PORT:-29607}"

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

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment "$EXP_ID" --exp-dir "$EXP_DIR" --overwrite
python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val --json "$EXP_DIR/config/train_val_ct_readiness.json"

[[ -f "$CACHE_ROOT/.complete" ]] || { echo "Missing cache completion marker: $CACHE_ROOT/.complete" >&2; exit 1; }
require_sha "$CACHE_ROOT/manifest.json" "$EXPECTED_CACHE_MANIFEST_SHA" "native cache manifest"
python - "$CACHE_ROOT/manifest.json" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
expected = {"cases": 3192, "targets": 8068, "empty_targets": 0}
observed = {key: manifest.get(key) for key in expected}
if observed != expected:
    raise SystemExit(f"Native cache audit mismatch: expected={expected} observed={observed}")
print(f"Native cache audit accepted: {observed}")
PY

require_sha "$SOURCE_PREFIX_SCHEDULE" "$EXPECTED_PREFIX_SHA" "exp003 v123 prefix schedule"
require_sha "$VAL20_JSON" "$EXPECTED_VAL20_SHA" "val20 JSON"
require_sha "$VAL200_JSON" "$EXPECTED_VAL200_SHA" "val200 JSON"

line_count_matches() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || return 1
  python - "$path" "$expected" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    with path.open("rb") as handle:
        observed = sum(1 for _ in handle)
except OSError:
    raise SystemExit(1)
raise SystemExit(0 if observed == expected else 1)
PY
}

generate_schedule_if_needed() {
  local schedule="$1" manifest="$2" events="$3" label="$4"
  if line_count_matches "$schedule" "$events"; then
    echo "Schedule ready: $label $schedule"
    return 0
  fi
  echo "Generating schedule: $label events=$events path=$schedule"
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

verify_original_schedule() {
  python - "$ORIGINAL_DDP_SCHEDULE" "$ORIGINAL_DDP_SCHEDULE_MANIFEST" "$SOURCE_PREFIX_SCHEDULE" "$EXPECTED_PREFIX_SHA" "$SCHEDULE_EVENTS" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

schedule = Path(sys.argv[1])
manifest = Path(sys.argv[2])
source = Path(sys.argv[3])
expected_prefix_sha = sys.argv[4]
expected_events = int(sys.argv[5])

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

with source.open("rb") as src, schedule.open("rb") as sched:
    for index, source_line in enumerate(src):
        schedule_line = sched.readline()
        if schedule_line != source_line:
            raise SystemExit(f"Schedule prefix differs from exp003 at line {index}")
source_sha = sha256(source)
if source_sha != expected_prefix_sha:
    raise SystemExit(f"Source prefix SHA mismatch: {source_sha} != {expected_prefix_sha}")

record = json.loads(manifest.read_text()) if manifest.is_file() else {}
record.update(
    {
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ddp_world_size": 4,
        "effective_global_batch_size": 4,
        "event_consumption_rule": "event_index = update * 4 + rank",
        "exp003_prefix_path": str(source),
        "exp003_prefix_sha256": source_sha,
        "exp003_prefix_lines_verified": 10000,
        "output_jsonl": str(schedule),
        "output_jsonl_sha256": sha256(schedule),
        "events": event_count,
    }
)
manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

prepare_continuation_schedule() {
  generate_schedule_if_needed "$ORIGINAL_DDP_SCHEDULE" "$ORIGINAL_DDP_SCHEDULE_MANIFEST" "$SCHEDULE_EVENTS" "original_ddp40k"
  verify_original_schedule
  generate_schedule_if_needed "$EXTENDED_SCHEDULE" "$EXTENDED_SCHEDULE_MANIFEST" "$EXTENDED_SCHEDULE_EVENTS" "extended_ddp80k"
  python - \
    "$ORIGINAL_DDP_SCHEDULE" \
    "$EXTENDED_SCHEDULE" \
    "$SCHEDULE" \
    "$SCHEDULE_MANIFEST" \
    "$EXTENDED_SCHEDULE_MANIFEST" \
    "$SCHEDULE_EVENTS" \
    "$EXTENDED_SCHEDULE_EVENTS" \
    "$SCHEDULE_START_EVENT" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

original = Path(sys.argv[1])
extended = Path(sys.argv[2])
continuation = Path(sys.argv[3])
manifest = Path(sys.argv[4])
extended_manifest = Path(sys.argv[5])
phase_events = int(sys.argv[6])
extended_events = int(sys.argv[7])
start_event = int(sys.argv[8])
end_event = start_event + phase_events

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)

if line_count(original) != phase_events:
    raise SystemExit(f"Original schedule line count mismatch: {original}")
if line_count(extended) != extended_events:
    raise SystemExit(f"Extended schedule line count mismatch: {extended}")
if end_event > extended_events:
    raise SystemExit(
        f"Continuation range {start_event}:{end_event} exceeds extended events {extended_events}"
    )

with original.open("rb") as lhs, extended.open("rb") as rhs:
    for index, original_line in enumerate(lhs):
        extended_line = rhs.readline()
        if extended_line != original_line:
            raise SystemExit(f"Extended schedule prefix differs from original at line {index}")

stats = Counter()
source_case_names = Counter()
continuation.parent.mkdir(parents=True, exist_ok=True)
with extended.open() as src, continuation.open("w") as out:
    for source_index, line in enumerate(src):
        if source_index < start_event:
            continue
        if source_index >= end_event:
            break
        event = json.loads(line)
        if int(event.get("event_index", -1)) != source_index:
            raise SystemExit(
                f"Extended event index mismatch: line={source_index} event={event.get('event_index')}"
            )
        continuation_index = source_index - start_event
        event["source_event_index"] = source_index
        event["event_index"] = continuation_index
        stats.update(event.get("stats", {}))
        source_case_names[str(event.get("case_name"))] += 1
        out.write(json.dumps(event, sort_keys=True) + "\n")

if line_count(continuation) != phase_events:
    raise SystemExit(f"Continuation schedule line count mismatch: {continuation}")

record = {
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "schedule_schema_version": 2,
    "purpose": "exp007 continuation events 40000..79999 rewritten to zero-based line order",
    "source_original_schedule": str(original),
    "source_original_schedule_sha256": sha256(original),
    "source_extended_schedule": str(extended),
    "source_extended_schedule_sha256": sha256(extended),
    "source_extended_manifest": str(extended_manifest),
    "source_extended_manifest_sha256": (
        sha256(extended_manifest) if extended_manifest.is_file() else None
    ),
    "output_jsonl": str(continuation),
    "output_jsonl_sha256": sha256(continuation),
    "events": phase_events,
    "extended_events": extended_events,
    "source_event_start_inclusive": start_event,
    "source_event_end_exclusive": end_event,
    "event_index_policy": "source event_index rewritten to 0..39999; original index stored as source_event_index",
    "ddp_world_size": 4,
    "effective_global_batch_size": 4,
    "event_consumption_rule": "event_index = update * 4 + rank",
    "unique_case_count": len(source_case_names),
    "stats": dict(stats),
}
manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

if [[ "$CONTINUATION_MODE" == "1" ]]; then
  prepare_continuation_schedule
else
  generate_schedule_if_needed "$SCHEDULE" "$SCHEDULE_MANIFEST" "$SCHEDULE_EVENTS" "ddp40k"
  verify_original_schedule
fi

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
    "source": "public VoxTell v1.1 resolved by voxtell.inference.predictor.download_voxtell_model",
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

if [[ "$CONTINUATION_MODE" == "1" ]]; then
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

if not checkpoint.is_file():
    raise SystemExit(f"Missing continuation init checkpoint: {checkpoint}")
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
    "load_policy": "network weights only via --init-checkpoint; optimizer, scaler, scheduler, and update counter reset",
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
fi

if [[ "$RUN_STATIC_ONLY" == "1" ]]; then
  echo "Static checks complete; RUN_STATIC_ONLY=1"
  exit 0
fi

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
  CUDA_VISIBLE_DEVICES=0 python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
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
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
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
    "${resume_args[@]}" \
    >"$log" 2>&1
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
    if (set -o noclobber; printf '{"pid":%s,"created":"%s","phase":"exp007_segment_eval"}\n' "$$" "$(date -u +%FT%TZ)" > "$lock") 2>/dev/null; then
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
  local exp003_summary="$EXP003_DIR/runs/exp003_full_20260723T075256Z/v123_opt_poscrop_emptyloss/full_100ep/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
  local source_summary="$CONTINUATION_SOURCE_RUN_DIR/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
  local source_training="$CONTINUATION_SOURCE_RUN_DIR/reports/training_metrics.json"
  local reference_args=()
  if [[ "$CONTINUATION_MODE" == "1" ]]; then
    [[ -f "$source_summary" ]] && reference_args+=(--reference "exp007_original_ddp_epoch100=$source_summary")
    [[ -f "$exp006_summary" ]] && reference_args+=(--reference "exp006_v123_cached_e5_d4_epoch100=$exp006_summary")
    [[ -f "$exp003_summary" ]] && reference_args+=(--reference "exp003_v123_epoch100=$exp003_summary")
    [[ -f "$source_training" ]] && reference_args+=(--reference-training "exp007_original_ddp=$source_training")
    [[ -f "$exp006_training" ]] && reference_args+=(--reference-training "exp006_bs1_e5_d4=$exp006_training")
    python /workspace/scripts/rexgroundingct/summarize_007_continuation_results.py \
      --exp-dir "$EXP_DIR" \
      --group-dir "$GROUP_DIR" \
      --source-run-dir "$CONTINUATION_SOURCE_RUN_DIR" \
      --source-checkpoint "$INIT_CHECKPOINT" \
      --absolute-epoch-offset "$ABSOLUTE_EPOCH_OFFSET" \
      --output-json "$REPORT_JSON" \
      --output-md "$REPORT_MD" \
      "${reference_args[@]}" || true
  else
    [[ -f "$exp006_summary" ]] && reference_args+=(--reference "exp006_v123_cached_e5_d4_epoch100=$exp006_summary")
    [[ -f "$exp006_training" ]] && reference_args+=(--reference-training "exp006_bs1_e5_d4=$exp006_training")
    [[ -f "$exp003_summary" ]] && reference_args+=(--reference "exp003_v123_epoch100=$exp003_summary")
    python /workspace/scripts/rexgroundingct/summarize_007_results.py \
      --exp-dir "$EXP_DIR" \
      --group-dir "$GROUP_DIR" \
      --output-json "$REPORT_JSON" \
      --output-md "$REPORT_MD" \
      "${reference_args[@]}" || true
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
  local absolute_epoch="$((ABSOLUTE_EPOCH_OFFSET + epoch))"
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
    echo "start epoch=$epoch val200 gpus=0 1 2 3 existing_predictions=$existing/$expected"
    pids=()
    for shard in 0 1 2 3; do
      run_inference_shard "$epoch" "$shard" 4 "$shard" &
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
    echo "prediction_count=$(prediction_count "$eval_dir") expected=$expected"
    EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$VAL200_JSON" \
      EVAL_LABEL="Experiment 007 DDP bs4 relative epoch $epoch absolute epoch $absolute_epoch fixed val200 threshold 0.5" NUM_WORKERS=8 \
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
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
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
    "${resume_args[@]}" \
    >"$log" 2>&1
  checkpoint_has_update "$checkpoint" "$target_update"
  cp "$RUN_DIR/reports/training_metrics.json" \
    "$RUN_DIR/reports/training_metrics_segment_epoch$(printf '%03d' "$epoch").json"
}

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
  smoke_dir="$GROUP_DIR/smoke/ddp_resume_smoke"
  smoke_model="$GROUP_DIR/smoke/model_epoch000"
  smoke_eval="$GROUP_DIR/smoke/one_case_inference"
  mkdir -p "$GROUP_DIR/smoke" "$smoke_eval"
  materialize_model "$smoke_dir/checkpoints/checkpoint_update_000004.pth" "$smoke_model"
  python - "$VAL20_JSON" "$GROUP_DIR/smoke/one_case_val20.json" <<'PY'
import json
import sys
from pathlib import Path
source = Path(sys.argv[1])
target = Path(sys.argv[2])
data = json.loads(source.read_text())
data["test"] = data["test"][:1]
target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  EVAL_LOCK_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split val \
    --dataset-json "$GROUP_DIR/smoke/one_case_val20.json" \
    --gpu 0 \
    --model-dir "$smoke_model" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$smoke_eval/predictions" \
    --preprocessed-cache-dir "$CACHE_ROOT" \
    --threshold 0.5 \
    --status-json "$smoke_eval/status.json" \
    2>&1 | tee "$smoke_eval/inference.log"
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
echo "Experiment 007 group dir: $GROUP_DIR"
exit "$status"
