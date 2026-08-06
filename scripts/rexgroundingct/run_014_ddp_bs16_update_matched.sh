#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

EXP_ID="014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
EXP003_DIR="${EXP003_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation}"
EXP006_DIR="${EXP006_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation}"
EXP007_DIR="${EXP007_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1}"
SEED="${SEED:-20260723}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EPOCHS="${EPOCHS:-100}"
WORLD_SIZE="${WORLD_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SCHEDULE_EVENTS="${SCHEDULE_EVENTS:-$((EPOCHS * STEPS_PER_EPOCH * WORLD_SIZE * BATCH_SIZE * GRAD_ACCUM))}"
SCHEDULE_NUM_WORKERS="${SCHEDULE_NUM_WORKERS:-4}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUP="${RUN_GROUP:-exp014_ddp_bs16_update_matched_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
RUN_DIR="$GROUP_DIR/ddp_bs16"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
SOURCE_PREFIX_SCHEDULE="${SOURCE_PREFIX_SCHEDULE:-$EXP007_DIR/config/train_schedule_v123_ddp_bs4_seed${SEED}_100ep_${STEPS_PER_EPOCH}steps_gb4.jsonl}"
SCHEDULE="$EXP_DIR/config/train_schedule_v123_ddp_bs16_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb16.jsonl"
SCHEDULE_MANIFEST="$EXP_DIR/config/train_schedule_v123_ddp_bs16_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb16.manifest.json"
SOURCE_MODEL_DIR="$EXP_DIR/config/public_voxtell_v1_1_model"
EXPECTED_CACHE_MANIFEST_SHA="fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3"
EXPECTED_VAL20_SHA="31557d624c47ee8c06799bf89cceb862471b34cfd47065001d6092e32d0dab5d"
EXPECTED_VAL200_SHA="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-2500,5000,7500,10000}"
SEGMENT_EPOCHS="${SEGMENT_EPOCHS:-25 50 75 100}"
RUN_SAMPLE_TEST="${RUN_SAMPLE_TEST:-1}"
RUN_DDP_SMOKE="${RUN_DDP_SMOKE:-1}"
RUN_INFERENCE_SMOKE="${RUN_INFERENCE_SMOKE:-1}"
RUN_STATIC_ONLY="${RUN_STATIC_ONLY:-0}"
RUN_SMOKE_ONLY="${RUN_SMOKE_ONLY:-1}"
RUN_FULL="${RUN_FULL:-0}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-43200}"
STABILITY_SECONDS="${STABILITY_SECONDS:-10}"
MASTER_PORT="${MASTER_PORT:-29614}"
LATEST_CHECKPOINT_EVERY_UPDATES="${LATEST_CHECKPOINT_EVERY_UPDATES:-500}"
GLOBAL_BATCH="$((WORLD_SIZE * BATCH_SIZE * GRAD_ACCUM))"

if [[ "$GLOBAL_BATCH" -ne 16 ]]; then
  echo "Exp014 expects effective global batch 16; observed $GLOBAL_BATCH" >&2
  exit 1
fi

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

generate_schedule_if_needed() {
  if line_count_matches "$SCHEDULE" "$SCHEDULE_EVENTS"; then
    echo "Schedule ready: $SCHEDULE"
    return 0
  fi
  echo "Generating exp014 schedule: events=$SCHEDULE_EVENTS path=$SCHEDULE"
  rm -f "$SCHEDULE" "$SCHEDULE_MANIFEST"
  python /workspace/scripts/rexgroundingct/prepare_training_schedule.py \
    --output-jsonl "$SCHEDULE" \
    --manifest-json "$SCHEDULE_MANIFEST" \
    --seed "$SEED" \
    --events "$SCHEDULE_EVENTS" \
    --patch-size 192 192 192 \
    --foreground-oversample-prob 0.85 \
    --require-positive-crop \
    --preprocessed-cache-dir "$CACHE_ROOT" \
    --num-workers "$SCHEDULE_NUM_WORKERS" \
    2>&1 | tee "$EXP_DIR/logs/prepare_schedule_bs16_$TIMESTAMP.log"
}

verify_schedule() {
  python - "$SCHEDULE" "$SCHEDULE_MANIFEST" "$SOURCE_PREFIX_SCHEDULE" "$SCHEDULE_EVENTS" "$GLOBAL_BATCH" "$GRAD_ACCUM" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

schedule = Path(sys.argv[1])
manifest = Path(sys.argv[2])
source_prefix = Path(sys.argv[3])
expected_events = int(sys.argv[4])
global_batch = int(sys.argv[5])
grad_accum = int(sys.argv[6])

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

prefix_record = None
if source_prefix.is_file():
    with source_prefix.open("rb") as src, schedule.open("rb") as sched:
        prefix_lines = 0
        for index, source_line in enumerate(src):
            schedule_line = sched.readline()
            if schedule_line != source_line:
                raise SystemExit(f"Exp014 schedule prefix differs from exp007 at line {index}")
            prefix_lines += 1
    prefix_record = {
        "source_prefix_path": str(source_prefix),
        "source_prefix_sha256": sha256(source_prefix),
        "source_prefix_lines_verified": prefix_lines,
    }

record = json.loads(manifest.read_text()) if manifest.is_file() else {}
record.update(
    {
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ddp_world_size": 4,
        "per_gpu_batch_size": 1,
        "gradient_accumulation": grad_accum,
        "effective_global_batch_size": global_batch,
        "event_consumption_rule": "event_index = update * 16 + rank * 4 + local_offset",
        "output_jsonl": str(schedule),
        "output_jsonl_sha256": sha256(schedule),
        "events": event_count,
    }
)
if prefix_record is not None:
    record.update(prefix_record)
manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
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

select_resume_checkpoint() {
  local target_update="$1"
  python - "$RUN_DIR/checkpoints" "$target_update" <<'PY'
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
target_update = int(sys.argv[2])
if not root.is_dir():
    raise SystemExit(0)
candidates = list(root.glob("checkpoint_update_*.pth"))
latest = root / "checkpoint_latest.pth"
if latest.is_file():
    candidates.append(latest)
best: tuple[int, Path] | None = None
for path in candidates:
    if not path.is_file() or path.stat().st_size <= 0:
        continue
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        continue
    update = int(payload.get("global_update", -1))
    if update <= 0 or update >= target_update:
        continue
    if best is None or update > best[0]:
        best = (update, path)
if best is not None:
    print(f"{best[0]}\t{best[1]}")
PY
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
  if [[ -n "$resume_checkpoint" ]]; then
    resume_args=(--resume-checkpoint "$resume_checkpoint")
  fi
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$WORLD_SIZE" \
    --master_port="$MASTER_PORT" \
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
    "${resume_args[@]}" \
    >"$log" 2>&1
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

prediction_count() {
  local eval_dir="$1"
  if [[ ! -d "$eval_dir/predictions" ]]; then
    echo 0
    return 0
  fi
  find "$eval_dir/predictions" -maxdepth 1 -name '*.nii.gz' | wc -l
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
    if (set -o noclobber; printf '{"pid":%s,"created":"%s","phase":"exp014_segment_eval"}\n' "$$" "$(date -u +%FT%TZ)" > "$lock") 2>/dev/null; then
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

summarize_results() {
  local reference_args=()
  local exp007_summary="$EXP007_DIR/runs/exp007_full_20260725T231624Z/ddp_bs4/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
  local exp007_training="$EXP007_DIR/runs/exp007_full_20260725T231624Z/ddp_bs4/reports/training_metrics.json"
  local exp006_summary="$EXP006_DIR/runs/latest/v123_cached_e5_d4/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
  local exp006_training="$EXP006_DIR/runs/latest/v123_cached_e5_d4/reports/training_metrics.json"
  [[ -f "$exp007_summary" ]] && reference_args+=(--reference "exp007_ddp_bs4_epoch100=$exp007_summary")
  [[ -f "$exp006_summary" ]] && reference_args+=(--reference "exp006_v123_cached_e5_d4_epoch100=$exp006_summary")
  [[ -f "$exp007_training" ]] && reference_args+=(--reference-training "exp007_ddp_bs4=$exp007_training")
  [[ -f "$exp006_training" ]] && reference_args+=(--reference-training "exp006_bs1_e5_d4=$exp006_training")
  python /workspace/scripts/rexgroundingct/summarize_014_results.py \
    --exp-dir "$EXP_DIR" \
    --group-dir "$GROUP_DIR" \
    --output-json "$EXP_DIR/reports/ddp_bs16_update_matched_summary.json" \
    --output-md "$EXP_DIR/reports/ddp_bs16_update_matched_report.md" \
    "${reference_args[@]}" || true
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
      EVAL_LABEL="Experiment 014 DDP bs16 epoch $epoch fixed val200 threshold 0.5" NUM_WORKERS=8 \
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
  local log="$RUN_DIR/logs/train_segment_to_epoch$(printf '%03d' "$epoch").log"
  mapfile -t args < <(common_train_args)
  if checkpoint_has_update "$checkpoint" "$target_update"; then
    echo "Training segment already complete: epoch=$epoch update=$target_update"
    return 0
  fi
  local selected selected_update selected_path
  selected="$(select_resume_checkpoint "$target_update")"
  if [[ -n "$selected" ]]; then
    selected_update="${selected%%$'\t'*}"
    selected_path="${selected#*$'\t'}"
    checkpoint_stable "$selected_path" || { echo "Selected checkpoint not stable: $selected_path" >&2; return 1; }
    resume_args=(--resume-checkpoint "$selected_path")
    log="$RUN_DIR/logs/train_segment_to_epoch$(printf '%03d' "$epoch")_from_update$(printf '%06d' "$selected_update").log"
    echo "Resuming segment target=$target_update from update=$selected_update checkpoint=$selected_path"
  fi
  echo "Starting DDP segment to epoch=$epoch update=$target_update log=$log"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$WORLD_SIZE" \
    --master_port="$MASTER_PORT" \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${args[@]}" \
    --run-dir "$RUN_DIR" \
    --epochs "$EPOCHS" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --target-global-update "$target_update" \
    --checkpoint-every-updates 0 \
    --checkpoint-updates "$CHECKPOINT_UPDATES" \
    --latest-checkpoint-every-updates "$LATEST_CHECKPOINT_EVERY_UPDATES" \
    --no-materialize-final-model \
    "${resume_args[@]}" \
    >"$log" 2>&1
  checkpoint_has_update "$checkpoint" "$target_update"
  cp "$RUN_DIR/reports/training_metrics.json" \
    "$RUN_DIR/reports/training_metrics_segment_epoch$(printf '%03d' "$epoch").json"
}

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment "$EXP_ID" --exp-dir "$EXP_DIR" --overwrite
python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val --json "$EXP_DIR/config/train_val_ct_readiness.json"

[[ -f "$CACHE_ROOT/.complete" ]] || { echo "Missing cache completion marker: $CACHE_ROOT/.complete" >&2; exit 1; }
require_sha "$CACHE_ROOT/manifest.json" "$EXPECTED_CACHE_MANIFEST_SHA" "native cache manifest"
require_sha "$VAL20_JSON" "$EXPECTED_VAL20_SHA" "val20 JSON"
require_sha "$VAL200_JSON" "$EXPECTED_VAL200_SHA" "val200 JSON"

generate_schedule_if_needed
verify_schedule

if [[ ! -f "$EMBEDDINGS" ]]; then
  for candidate in "$EXP007_DIR/config/rex_text_embeddings.npz" "$EXP006_DIR/config/rex_text_embeddings.npz" "$EXP003_DIR/config/rex_text_embeddings.npz"; do
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
  echo "Experiment 014 group dir: $GROUP_DIR"
  exit 0
fi
if [[ "$RUN_FULL" != "1" ]]; then
  echo "RUN_FULL=$RUN_FULL; full training not started"
  echo "Experiment 014 group dir: $GROUP_DIR"
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
echo "Experiment 014 group dir: $GROUP_DIR"
exit "$status"
