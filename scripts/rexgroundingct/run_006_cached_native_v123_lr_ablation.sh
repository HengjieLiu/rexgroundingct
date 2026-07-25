#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

EXP_ID="006_voxtell_cached_native_v123_lr_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
EXP003_DIR="${EXP003_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1}"
SEED="${SEED:-20260723}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUP="${RUN_GROUP:-exp006_cached_native_lr_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
SOURCE_SCHEDULE="${SOURCE_SCHEDULE:-$EXP003_DIR/config/train_schedule_v123_opt_poscrop_emptyloss_seed${SEED}_100ep_100steps_gb1.jsonl}"
SCHEDULE="$EXP_DIR/config/train_schedule_v123_cached_native_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${BATCH_SIZE}.jsonl"
SCHEDULE_MANIFEST="$EXP_DIR/config/train_schedule_v123_cached_native_seed${SEED}_${EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${BATCH_SIZE}.manifest.json"
SOURCE_MODEL_DIR="$EXP_DIR/config/public_voxtell_v1_1_model"
EXPECTED_SCHEDULE_SHA="f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776"
EXPECTED_CACHE_MANIFEST_SHA="fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3"
EXPECTED_VAL20_SHA="31557d624c47ee8c06799bf89cceb862471b34cfd47065001d6092e32d0dab5d"
EXPECTED_VAL200_SHA="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-500,2000,4000,6000,8000,10000}"
RUN_SAMPLE_TEST="${RUN_SAMPLE_TEST:-1}"
RUN_TRAIN_SMOKE="${RUN_TRAIN_SMOKE:-1}"
RUN_INFERENCE_SMOKE="${RUN_INFERENCE_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"
RUN_STATIC_ONLY="${RUN_STATIC_ONLY:-0}"
RUN_SMOKE_ONLY="${RUN_SMOKE_ONLY:-0}"
ARM_SLEEP_SECONDS="${ARM_SLEEP_SECONDS:-10}"

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$EXP_DIR/reports" "$GROUP_DIR"
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

require_sha "$SOURCE_SCHEDULE" "$EXPECTED_SCHEDULE_SHA" "exp003 v123 schedule"
require_sha "$VAL20_JSON" "$EXPECTED_VAL20_SHA" "val20 JSON"
require_sha "$VAL200_JSON" "$EXPECTED_VAL200_SHA" "val200 JSON"
cp "$SOURCE_SCHEDULE" "$SCHEDULE"
if [[ -f "${SOURCE_SCHEDULE%.jsonl}.manifest.json" ]]; then
  cp "${SOURCE_SCHEDULE%.jsonl}.manifest.json" "$SCHEDULE_MANIFEST.source_exp003.json"
fi
python - "$SCHEDULE" "$SCHEDULE_MANIFEST" "$SOURCE_SCHEDULE" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

schedule = Path(sys.argv[1])
manifest = Path(sys.argv[2])
source = Path(sys.argv[3])

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
    "purpose": "exp006 shared cached-native v123 LR-ablation event schedule",
    "copied_from": str(source),
    "copied_from_sha256": sha256(source),
    "output_jsonl": str(schedule),
    "output_jsonl_sha256": sha256(schedule),
    "events": events,
    "seed": 20260723,
    "consumption": "all four arms consume events 0-9999 with batch1/no-DDP",
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [[ ! -f "$EMBEDDINGS" ]]; then
  if [[ -f "$EXP003_DIR/config/rex_text_embeddings.npz" ]]; then
    cp "$EXP003_DIR/config/rex_text_embeddings.npz" "$EMBEDDINGS"
  else
    python /workspace/scripts/rexgroundingct/precompute_text_embeddings.py \
      --splits train val --gpu 0 --output "$EMBEDDINGS" \
      2>&1 | tee "$EXP_DIR/logs/precompute_text_embeddings_$TIMESTAMP.log"
  fi
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

if [[ "$RUN_STATIC_ONLY" == "1" ]]; then
  echo "Static checks complete; RUN_STATIC_ONLY=1"
  exit 0
fi

arm_gpu() {
  case "$1" in
    v123_cached_e7_d6) echo 0 ;;
    v123_cached_e7_d4) echo 1 ;;
    v123_cached_e6_d4) echo 2 ;;
    v123_cached_e5_d4) echo 3 ;;
    *) return 1 ;;
  esac
}

arm_encoder_lr() {
  case "$1" in
    v123_cached_e7_d6|v123_cached_e7_d4) echo 1e-7 ;;
    v123_cached_e6_d4) echo 1e-6 ;;
    v123_cached_e5_d4) echo 1e-5 ;;
    *) return 1 ;;
  esac
}

arm_decoder_lr() {
  case "$1" in
    v123_cached_e7_d6) echo 1e-6 ;;
    v123_cached_e7_d4|v123_cached_e6_d4|v123_cached_e5_d4) echo 1e-4 ;;
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
    --encoder-lr "$(arm_encoder_lr "$arm")" \
    --decoder-lr "$(arm_decoder_lr "$arm")" \
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
    --allow-experimental-train
}

run_sample_test() {
  local arm="$1" gpu="$2"
  local sample_dir="$GROUP_DIR/$arm/sample_test"
  mkdir -p "$sample_dir/logs"
  CUDA_VISIBLE_DEVICES="$gpu" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
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

run_train_smoke() {
  local arm="$1" gpu="$2"
  local smoke_dir="$GROUP_DIR/$arm/train_smoke_one_update"
  mkdir -p "$smoke_dir/logs"
  mapfile -t args < <(common_train_args "$arm")
  CUDA_VISIBLE_DEVICES="$gpu" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${args[@]}" \
    --run-dir "$smoke_dir" \
    --epochs 1 \
    --steps-per-epoch 1 \
    --checkpoint-every-updates 0 \
    --checkpoint-updates "" \
    --latest-checkpoint-every-updates 0 \
    --no-materialize-final-model \
    2>&1 | tee "$smoke_dir/logs/train.log"
}

arms=(v123_cached_e7_d6 v123_cached_e7_d4 v123_cached_e6_d4 v123_cached_e5_d4)

if [[ "$RUN_SAMPLE_TEST" == "1" ]]; then
  for arm in "${arms[@]}"; do
    run_sample_test "$arm" "$(arm_gpu "$arm")"
  done
fi

if [[ "$RUN_TRAIN_SMOKE" == "1" ]]; then
  run_train_smoke v123_cached_e7_d6 0
  run_train_smoke v123_cached_e5_d4 3
fi

if [[ "$RUN_INFERENCE_SMOKE" == "1" ]]; then
  mkdir -p "$GROUP_DIR/smoke"
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
    --model-dir "$BASE_MODEL_DIR" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$GROUP_DIR/smoke/cached_native_inference/predictions" \
    --preprocessed-cache-dir "$CACHE_ROOT" \
    --threshold 0.5
fi

if [[ "$RUN_SMOKE_ONLY" == "1" ]]; then
  echo "Smoke checks complete; RUN_SMOKE_ONLY=1"
  exit 0
fi
if [[ "$RUN_FULL" != "1" ]]; then
  echo "RUN_FULL=$RUN_FULL; full training not started"
  exit 0
fi

EXP_DIR="$EXP_DIR" GROUP_DIR="$GROUP_DIR" CACHE_ROOT="$CACHE_ROOT" \
  SOURCE_MODEL_DIR="$SOURCE_MODEL_DIR" EMBEDDINGS="$EMBEDDINGS" \
  VAL20_JSON="$VAL20_JSON" VAL200_JSON="$VAL200_JSON" \
  bash /workspace/scripts/rexgroundingct/run_006_eval_coordinator.sh \
  >"$GROUP_DIR/eval_coordinator.log" 2>&1 &
coordinator_pid=$!
printf '%s\n' "$coordinator_pid" > "$GROUP_DIR/eval_coordinator.pid"

pids=()
names=()
for arm in "${arms[@]}"; do
  gpu="$(arm_gpu "$arm")"
  run_dir="$GROUP_DIR/$arm"
  mkdir -p "$run_dir/logs"
  mapfile -t args < <(common_train_args "$arm")
  (
    CUDA_VISIBLE_DEVICES="$gpu" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
      "${args[@]}" \
      --run-dir "$run_dir" \
      --epochs "$EPOCHS" \
      --steps-per-epoch "$STEPS_PER_EPOCH" \
      --checkpoint-every-updates 0 \
      --checkpoint-updates "$CHECKPOINT_UPDATES" \
      --latest-checkpoint-every-updates 500
  ) >"$run_dir/logs/training.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > "$run_dir/training.pid"
  pids+=("$pid")
  names+=("$arm")
  echo "Started $arm on GPU $gpu pid=$pid"
  sleep "$ARM_SLEEP_SECONDS"
done

status=0
for index in "${!pids[@]}"; do
  arm="${names[$index]}"
  if wait "${pids[$index]}"; then
    touch "$GROUP_DIR/$arm/.train_complete"
    echo "Training complete: $arm"
  else
    touch "$GROUP_DIR/$arm/.train_failed"
    echo "Training failed: $arm" >&2
    status=1
  fi
done

if ! wait "$coordinator_pid"; then
  echo "Eval coordinator failed" >&2
  status=1
fi

EXP003_SUMMARY="/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v123_opt_poscrop_emptyloss/full_100ep/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
EXP004_SUMMARY="/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft/runs/latest/native192_cont100/eval_epoch100_val200/reports/val_quick_global_eval_summary.json"
reference_args=()
[[ -f "$EXP003_SUMMARY" ]] && reference_args+=(--reference "exp003_v123_epoch100=$EXP003_SUMMARY")
[[ -f "$EXP004_SUMMARY" ]] && reference_args+=(--reference "exp004_native_cont100=$EXP004_SUMMARY")

python /workspace/scripts/rexgroundingct/summarize_006_results.py \
  --exp-dir "$EXP_DIR" \
  --group-dir "$GROUP_DIR" \
  --output-json "$EXP_DIR/reports/lr_ablation_summary.json" \
  --output-md "$EXP_DIR/reports/lr_ablation_report.md" \
  "${reference_args[@]}" || true

if [[ "$status" == "0" ]]; then
  touch "$GROUP_DIR/.experiment_complete"
fi
echo "Experiment 006 group dir: $GROUP_DIR"
exit "$status"
