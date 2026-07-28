#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

EXP_ID="009_voxtell_s3_attention_coupling_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
EXP006_DIR="${EXP006_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation}"
EXP007_DIR="${EXP007_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1}"
SOURCE_RUN_GROUP="${SOURCE_RUN_GROUP:-exp006_cached_native_lr_20260725T050001Z}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$EXP006_DIR/runs/$SOURCE_RUN_GROUP/v123_cached_e5_d4/model_epoch100}"
SOURCE_CHECKPOINT="$SOURCE_MODEL_DIR/fold_0/checkpoint_final.pth"
SOURCE_EXP006_SCHEDULE="${SOURCE_EXP006_SCHEDULE:-$EXP006_DIR/config/train_schedule_v123_cached_native_seed20260723_100ep_100steps_gb1.jsonl}"
SOURCE_EXTENDED_SCHEDULE="${SOURCE_EXTENDED_SCHEDULE:-$EXP007_DIR/config/train_schedule_v123_ddp_bs4_seed20260723_100ep_100steps_gb4.jsonl}"
SEED="${SEED:-20260723}"
EPOCHS="${EPOCHS:-100}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
SCHEDULE_START="${SCHEDULE_START:-10000}"
SCHEDULE_EVENTS="${SCHEDULE_EVENTS:-10000}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUP="${RUN_GROUP:-exp009_s3_attention_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
SCHEDULE="$EXP_DIR/config/train_schedule_continuation_events_${SCHEDULE_START}_$((SCHEDULE_START + SCHEDULE_EVENTS - 1))_seed${SEED}.jsonl"
SCHEDULE_MANIFEST="$EXP_DIR/config/train_schedule_continuation_events_${SCHEDULE_START}_$((SCHEDULE_START + SCHEDULE_EVENTS - 1))_seed${SEED}.manifest.json"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-500,2000,4000,6000,8000,10000}"
SEGMENT_EPOCHS="${SEGMENT_EPOCHS:-5 20 40 60 80 100}"
RUN_STATIC_ONLY="${RUN_STATIC_ONLY:-0}"
RUN_SMOKE_ONLY="${RUN_SMOKE_ONLY:-0}"
RUN_FULL="${RUN_FULL:-1}"
RUN_MODEL_MATERIALIZATION="${RUN_MODEL_MATERIALIZATION:-1}"
RUN_TRAIN_SMOKES="${RUN_TRAIN_SMOKES:-1}"
RUN_EPOCH0_EVAL="${RUN_EPOCH0_EVAL:-1}"
EPOCH0_DICE_TOLERANCE="${EPOCH0_DICE_TOLERANCE:-0.0001}"
ARM_STAGGER_SECONDS="${ARM_STAGGER_SECONDS:-10}"
S3_COUPLING_RAMP_UPDATES="${S3_COUPLING_RAMP_UPDATES:-2000}"
EXPECTED_EXP006_SCHEDULE_SHA="f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776"
EXPECTED_EXTENDED_SCHEDULE_SHA="acdfed8dd8d9908e9ea33d6790d47f8f7e3cbeed4bad326274d115d38e96a0aa"
EXPECTED_CACHE_MANIFEST_SHA="fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3"
EXPECTED_VAL20_SHA="31557d624c47ee8c06799bf89cceb862471b34cfd47065001d6092e32d0dab5d"
EXPECTED_VAL200_SHA="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"

VARIANTS=(
  baseline_cont100
  s3v1_fixedrho_suppress_half_quarter
  s3v2_balanced_feature_half_quarter
  s3v3_logit_residual_half_quarter
)
S3_VARIANTS=(
  ""
  s3v1_fixedrho_suppress_half_quarter
  s3v2_balanced_feature_half_quarter
  s3v3_logit_residual_half_quarter
)
GPUS=(0 1 2 3)

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$EXP_DIR/reports" "$GROUP_DIR"
printf '%s\n' "$RUN_GROUP" > "$EXP_DIR/config/latest_run_group.txt"
ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv \
    >"$EXP_DIR/logs/gpu_memory_before_launch_$TIMESTAMP.csv" || true
fi

sha256_path() {
  sha256sum "$1" | cut -d ' ' -f 1
}

require_sha() {
  local path="$1" expected="$2" label="$3"
  [[ -f "$path" ]] || { echo "Missing $label: $path" >&2; exit 1; }
  local actual
  actual="$(sha256_path "$path")"
  [[ "$actual" == "$expected" ]] || {
    echo "$label SHA mismatch expected=$expected actual=$actual path=$path" >&2
    exit 1
  }
  echo "$label sha256=$actual"
}

checkpoint_has_update() {
  local checkpoint="$1" expected="$2"
  [[ -s "$checkpoint" ]] || return 1
  python - "$checkpoint" "$expected" <<'PY'
import sys
from pathlib import Path
import torch
checkpoint = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
raise SystemExit(0 if int(checkpoint.get("global_update", -1)) == int(sys.argv[2]) else 1)
PY
}

eval_complete() {
  local eval_dir="$1" dataset_json="$2"
  python - "$eval_dir" "$dataset_json" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
summary_path = root / "reports" / "val_quick_global_eval_summary.json"
predictions = root / "predictions"
if not (summary_path.is_file() and predictions.is_dir() and dataset_path.is_file()):
    raise SystemExit(1)
entries = json.loads(dataset_path.read_text()).get("test", [])
summary = json.loads(summary_path.read_text())
if int(summary.get("total_cases", -1)) != len(entries):
    raise SystemExit(1)
if int(summary.get("total_findings", -1)) != sum(len(e["findings"]) for e in entries):
    raise SystemExit(1)
if any(not (predictions / e["name"]).is_file() for e in entries):
    raise SystemExit(1)
PY
}

materialize_baseline_model() {
  local model_dir="$1"
  mkdir -p "$model_dir/fold_0"
  cp "$SOURCE_MODEL_DIR/plans.json" "$model_dir/plans.json"
  rm -f "$model_dir/model_spec.json"
  local target="$model_dir/fold_0/checkpoint_final.pth"
  if [[ ! -f "$target" || "$(stat -c %s "$target")" != "$(stat -c %s "$SOURCE_CHECKPOINT")" ]]; then
    local tmp="$target.tmp.$$"
    rm -f "$tmp"
    ln "$SOURCE_CHECKPOINT" "$tmp" 2>/dev/null || cp "$SOURCE_CHECKPOINT" "$tmp"
    mv -f "$tmp" "$target"
  fi
}

materialize_checkpoint_model() {
  local arm_dir="$1" checkpoint="$2" epoch="$3" s3_variant="$4"
  local model_dir="$arm_dir/model_epoch$(printf '%03d' "$epoch")"
  mkdir -p "$model_dir/fold_0"
  cp "$SOURCE_MODEL_DIR/plans.json" "$model_dir/plans.json"
  if [[ -n "$s3_variant" ]]; then
    cp "$arm_dir/model_epoch000/model_spec.json" "$model_dir/model_spec.json"
  else
    rm -f "$model_dir/model_spec.json"
  fi
  local target="$model_dir/fold_0/checkpoint_final.pth"
  if [[ ! -f "$target" || "$(stat -c %s "$target")" != "$(stat -c %s "$checkpoint")" ]]; then
    local tmp="$target.tmp.$$"
    rm -f "$tmp"
    ln "$checkpoint" "$tmp" 2>/dev/null || cp "$checkpoint" "$tmp"
    mv -f "$tmp" "$target"
  fi
}

summarize_partial() {
  (
    flock 9
    python /workspace/scripts/rexgroundingct/summarize_009_results.py \
      --group-dir "$GROUP_DIR" \
      --output-json "$EXP_DIR/reports/s3_attention_coupling_summary.json" \
      --output-md "$EXP_DIR/reports/s3_attention_coupling_report.md" \
      >/dev/null
  ) 9>"$EXP_DIR/reports/.summary.lock"
}

arm_failed() {
  local variant="$1"
  [[ -f "$GROUP_DIR/$variant/.arm_failed" ]]
}

any_active_arm() {
  local variant
  for variant in "${VARIANTS[@]}"; do
    if ! arm_failed "$variant"; then
      return 0
    fi
  done
  return 1
}

any_failed_arm() {
  local variant
  for variant in "${VARIANTS[@]}"; do
    if arm_failed "$variant"; then
      return 0
    fi
  done
  return 1
}

mark_arm_failed() {
  local variant="$1" phase="$2" log_path="${3:-}"
  local arm_dir="$GROUP_DIR/$variant"
  local reason="failed"
  mkdir -p "$arm_dir/reports"
  if [[ -n "$log_path" && -f "$log_path" ]]; then
    if grep -Eq "CUDA out of memory|OutOfMemoryError|torch.OutOfMemoryError" "$log_path"; then
      reason="cuda_oom"
    fi
  fi
  touch "$arm_dir/.arm_failed" "$arm_dir/.skip_remaining"
  python - "$arm_dir/failure_status.json" "$variant" "$phase" "$reason" "$log_path" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
output.write_text(json.dumps({
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "variant": sys.argv[2],
    "phase": sys.argv[3],
    "reason": sys.argv[4],
    "log_path": sys.argv[5] or None,
    "policy": "arm stopped after failure; no batch size, scale, or architecture fallback applied",
}, indent=2, sort_keys=True) + "\n")
PY
  echo "Marked failed arm: $variant phase=$phase reason=$reason log=$log_path"
}

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment "$EXP_ID" --exp-dir "$EXP_DIR" --overwrite
python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val --json "$EXP_DIR/config/train_val_ct_readiness.json"

[[ -f "$CACHE_ROOT/.complete" ]] || {
  echo "Missing native cache completion marker: $CACHE_ROOT/.complete" >&2
  exit 1
}
require_sha "$CACHE_ROOT/manifest.json" "$EXPECTED_CACHE_MANIFEST_SHA" "native cache"
require_sha "$SOURCE_EXP006_SCHEDULE" "$EXPECTED_EXP006_SCHEDULE_SHA" "exp006 schedule"
require_sha "$SOURCE_EXTENDED_SCHEDULE" "$EXPECTED_EXTENDED_SCHEDULE_SHA" "extended schedule"
require_sha "$VAL20_JSON" "$EXPECTED_VAL20_SHA" "val20 JSON"
require_sha "$VAL200_JSON" "$EXPECTED_VAL200_SHA" "val200 JSON"
[[ -f "$SOURCE_CHECKPOINT" ]] || { echo "Missing source checkpoint: $SOURCE_CHECKPOINT" >&2; exit 1; }
[[ -f "$SOURCE_MODEL_DIR/plans.json" ]] || { echo "Missing source plans" >&2; exit 1; }

python - "$SOURCE_EXP006_SCHEDULE" "$SOURCE_EXTENDED_SCHEDULE" <<'PY'
import sys
from pathlib import Path
short, extended = map(Path, sys.argv[1:3])
with short.open("rb") as left, extended.open("rb") as right:
    for index, source_line in enumerate(left):
        if source_line != right.readline():
            raise SystemExit(f"Extended schedule differs from exp006 at event {index}")
print("Verified extended schedule events 0-9999 reproduce exp006.")
PY

if [[ ! -f "$SCHEDULE" || ! -f "$SCHEDULE_MANIFEST" ]]; then
  python /workspace/scripts/rexgroundingct/slice_training_schedule.py \
    --source-jsonl "$SOURCE_EXTENDED_SCHEDULE" \
    --start-event "$SCHEDULE_START" \
    --events "$SCHEDULE_EVENTS" \
    --output-jsonl "$SCHEDULE" \
    --manifest-json "$SCHEDULE_MANIFEST" \
    2>&1 | tee "$EXP_DIR/logs/slice_schedule_$TIMESTAMP.log"
fi

if [[ ! -f "$EMBEDDINGS" ]]; then
  cp "$EXP006_DIR/config/rex_text_embeddings.npz" "$EMBEDDINGS"
fi

python - "$SOURCE_CHECKPOINT" "$EXP_DIR/config/source_checkpoint_provenance.json" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
source, output = map(Path, sys.argv[1:3])
digest = hashlib.sha256()
with source.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
output.write_text(json.dumps({
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_checkpoint": str(source),
    "source_checkpoint_sha256": digest.hexdigest(),
}, indent=2, sort_keys=True) + "\n")
PY

PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  /workspace/scripts/rexgroundingct/voxtell_s3_attention.py \
  /workspace/scripts/rexgroundingct/materialize_s3_attention_model.py \
  /workspace/scripts/rexgroundingct/summarize_009_results.py \
  /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
  /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py

if [[ "$RUN_STATIC_ONLY" == "1" ]]; then
  echo "Exp009 static checks complete."
  exit 0
fi

if [[ "$RUN_MODEL_MATERIALIZATION" == "1" ]]; then
  for index in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$index]}"
    s3_variant="${S3_VARIANTS[$index]}"
    arm_dir="$GROUP_DIR/$variant"
    model_dir="$arm_dir/model_epoch000"
    manifest="$arm_dir/config/epoch000_initialization_manifest.json"
    mkdir -p "$arm_dir/config" "$arm_dir/logs" "$arm_dir/reports"
    if [[ -z "$s3_variant" ]]; then
      materialize_baseline_model "$model_dir"
      python - "$SOURCE_CHECKPOINT" "$model_dir/fold_0/checkpoint_final.pth" "$manifest" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
source, target, manifest = map(Path, sys.argv[1:4])
manifest.write_text(json.dumps({
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "variant": "baseline_cont100",
    "source_checkpoint": str(source),
    "output_checkpoint": str(target),
    "copy_audit": "baseline source checkpoint copied",
}, indent=2, sort_keys=True) + "\n")
PY
    elif [[ ! -f "$model_dir/fold_0/checkpoint_final.pth" || ! -f "$manifest" ]]; then
      python /workspace/scripts/rexgroundingct/materialize_s3_attention_model.py \
        --source-model-dir "$SOURCE_MODEL_DIR" \
        --variant "$s3_variant" \
        --coupling-ramp-updates "$S3_COUPLING_RAMP_UPDATES" \
        --output-model-dir "$model_dir" \
        --manifest-json "$manifest" \
        >"$arm_dir/logs/materialize_epoch000.log" 2>&1
    fi
  done
fi

common_train_args() {
  local s3_variant="$1"
  printf '%s\n' \
    --mode train \
    --experiment-id "$EXP_ID" \
    --config "$CONFIG_SNAPSHOT" \
    --exp-dir "$EXP_DIR" \
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
    --s3-coupling-ramp-updates "$S3_COUPLING_RAMP_UPDATES" \
    --s3-attention-loss-weight 0.05 \
    --s3-hard-bg-loss-weight 0.01 \
    --s3-fullres-alignment-loss-weight 0.0 \
    --s3-attention-margin 0.05 \
    --s3-hard-background-fraction 0.01 \
    --s3-fullres-sample-cap 4096 \
    --preprocessed-cache-dir "$CACHE_ROOT" \
    --sample-schedule "$SCHEDULE" \
    --allow-experimental-train
  if [[ -n "$s3_variant" ]]; then
    printf '%s\n' --s3-attention-variant "$s3_variant"
  fi
}

run_one_update_smoke() {
  local variant="$1" s3_variant="$2" gpu="$3"
  local arm_dir="$GROUP_DIR/$variant"
  local smoke_dir="$GROUP_DIR/smoke/${variant}_one_update"
  mkdir -p "$smoke_dir/logs"
  mapfile -t args < <(common_train_args "$s3_variant")
  CUDA_VISIBLE_DEVICES="$gpu" python \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${args[@]}" \
    --run-dir "$smoke_dir" \
    --gpu 0 \
    --init-checkpoint "$arm_dir/model_epoch000/fold_0/checkpoint_final.pth" \
    --epochs 1 \
    --steps-per-epoch 1 \
    --target-global-update 1 \
    --checkpoint-every-updates 0 \
    --checkpoint-updates 1 \
    --latest-checkpoint-every-updates 0 \
    --no-materialize-final-model \
    >"$smoke_dir/logs/training.log" 2>&1
  checkpoint_has_update "$smoke_dir/checkpoints/checkpoint_update_000001.pth" 1
}

run_resume_smoke() {
  local variant="$1" s3_variant="$2" gpu="$3"
  local arm_dir="$GROUP_DIR/$variant"
  local smoke_dir="$GROUP_DIR/smoke/${variant}_resume"
  mkdir -p "$smoke_dir/logs"
  mapfile -t args < <(common_train_args "$s3_variant")
  CUDA_VISIBLE_DEVICES="$gpu" python \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${args[@]}" \
    --run-dir "$smoke_dir" --gpu 0 \
    --init-checkpoint "$arm_dir/model_epoch000/fold_0/checkpoint_final.pth" \
    --epochs 1 --steps-per-epoch 4 --target-global-update 2 \
    --checkpoint-every-updates 0 --checkpoint-updates 2,4 \
    --latest-checkpoint-every-updates 0 --no-materialize-final-model \
    >"$smoke_dir/logs/train_to_2.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" python \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${args[@]}" \
    --run-dir "$smoke_dir" --gpu 0 \
    --resume-checkpoint "$smoke_dir/checkpoints/checkpoint_update_000002.pth" \
    --epochs 1 --steps-per-epoch 4 --target-global-update 4 \
    --checkpoint-every-updates 0 --checkpoint-updates 2,4 \
    --latest-checkpoint-every-updates 0 --no-materialize-final-model \
    >"$smoke_dir/logs/train_to_4.log" 2>&1
  checkpoint_has_update "$smoke_dir/checkpoints/checkpoint_update_000004.pth" 4
}

if [[ "$RUN_TRAIN_SMOKES" == "1" ]]; then
  smoke_pids=()
  smoke_variants=()
  for index in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$index]}"
    s3_variant="${S3_VARIANTS[$index]}"
    gpu="${GPUS[$index]}"
    (
      if ! run_one_update_smoke "$variant" "$s3_variant" "$gpu"; then
        mark_arm_failed "$variant" "one_update_smoke" \
          "$GROUP_DIR/smoke/${variant}_one_update/logs/training.log"
        exit 1
      fi
    ) &
    smoke_pids+=("$!")
    smoke_variants+=("$variant")
  done
  for index in "${!smoke_pids[@]}"; do
    pid="${smoke_pids[$index]}"
    if ! wait "$pid"; then
      echo "Exp009 one-update smoke failed for ${smoke_variants[$index]}; arm is stopped." >&2
    fi
  done
  any_active_arm || {
    summarize_partial
    touch "$GROUP_DIR/.experiment_failed"
    echo "All Exp009 arms failed during one-update smoke." >&2
    exit 1
  }
  for index in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$index]}"
    if arm_failed "$variant"; then
      continue
    fi
    if ! run_resume_smoke "$variant" "${S3_VARIANTS[$index]}" "${GPUS[$index]}"; then
      mark_arm_failed "$variant" "resume_smoke" \
        "$GROUP_DIR/smoke/${variant}_resume/logs/train_to_4.log"
    fi
  done
  any_active_arm || {
    summarize_partial
    touch "$GROUP_DIR/.experiment_failed"
    echo "All Exp009 arms failed during resume smoke." >&2
    exit 1
  }
  summarize_partial
fi

run_eval() {
  local variant="$1" gpu="$2" epoch="$3" dataset_size="$4"
  local arm_dir="$GROUP_DIR/$variant"
  local dataset_json model_dir eval_dir
  if [[ "$dataset_size" == "20" ]]; then
    dataset_json="$VAL20_JSON"
  else
    dataset_json="$VAL200_JSON"
  fi
  model_dir="$arm_dir/model_epoch$(printf '%03d' "$epoch")"
  eval_dir="$arm_dir/eval_epoch$(printf '%03d' "$epoch")_val${dataset_size}"
  if eval_complete "$eval_dir" "$dataset_json"; then
    echo "Already complete: $variant epoch=$epoch val$dataset_size"
    return 0
  fi
  mkdir -p "$eval_dir/logs" "$eval_dir/status" "$eval_dir/predictions"
  EVAL_LOCK_DISABLE=1 CUDA_VISIBLE_DEVICES="$gpu" python \
    /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split val \
    --dataset-json "$dataset_json" \
    --gpu 0 \
    --model-dir "$model_dir" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$eval_dir/predictions" \
    --output-type mask \
    --threshold 0.5 \
    --preprocessed-cache-dir "$CACHE_ROOT" \
    --sliding-window-batch-size 1 \
    --status-json "$eval_dir/status/inference.json" \
    >"$eval_dir/logs/inference.log" 2>&1

  EVAL_LOCK_DISABLE=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset_json" \
    EVAL_LABEL="Experiment 009 $variant epoch $epoch val$dataset_size threshold 0.5" \
    NUM_WORKERS=8 \
    bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$eval_dir" val \
    >"$eval_dir/logs/final_eval.log" 2>&1
  touch "$eval_dir/.complete"
}

if [[ "$RUN_EPOCH0_EVAL" == "1" ]]; then
  epoch0_pids=()
  epoch0_variants=()
  for index in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$index]}"
    if arm_failed "$variant"; then
      continue
    fi
    run_eval "$variant" "${GPUS[$index]}" 0 20 &
    epoch0_pids+=("$!")
    epoch0_variants+=("$variant")
  done
  for index in "${!epoch0_pids[@]}"; do
    pid="${epoch0_pids[$index]}"
    if ! wait "$pid"; then
      mark_arm_failed "${epoch0_variants[$index]}" "epoch0_val20" \
        "$GROUP_DIR/${epoch0_variants[$index]}/eval_epoch000_val20/logs/inference.log"
    fi
  done
  equivalence_variants=()
  for variant in "${epoch0_variants[@]}"; do
    if [[ ! -f "$GROUP_DIR/$variant/.arm_failed" ]]; then
      equivalence_variants+=("$variant")
    fi
  done
  [[ "${#equivalence_variants[@]}" -gt 0 ]] || {
    summarize_partial
    touch "$GROUP_DIR/.experiment_failed"
    echo "No active Exp009 arms completed epoch-0 val20 evaluation." >&2
    exit 1
  }
  python - "$GROUP_DIR" "$EPOCH0_DICE_TOLERANCE" "${equivalence_variants[@]}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
tolerance = float(sys.argv[2])
expected_dice = 0.3907023703162601
expected_hit = 0.7419354838709677
expected_hits = 23
expected_findings = 31
records = {}
for variant in sys.argv[3:]:
    path = root / variant / "eval_epoch000_val20/reports/val_quick_global_eval_summary.json"
    value = json.loads(path.read_text())
    dice = float(value["mean_global_dice_per_finding"])
    hit = float(value["hit_rate"])
    hits = int(value["total_hits"])
    findings = int(value["total_findings"])
    records[variant] = {
        "dice": dice,
        "dice_delta_from_stored_exp006": dice - expected_dice,
        "hit_rate": hit,
        "hits": hits,
        "findings": findings,
    }
    if (
        abs(dice - expected_dice) > tolerance
        or abs(hit - expected_hit) > 1e-12
        or hits != expected_hits
        or findings != expected_findings
    ):
        raise SystemExit(
            f"{variant}: epoch-0 mismatch dice={dice} hit={hit}; "
            f"expected dice={expected_dice} +/- {tolerance}, hit={expected_hit}, "
            f"hits/findings={expected_hits}/{expected_findings}"
        )
output = root / "benchmark/epoch0_metric_equivalence.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "stored_exp006_reference": {
        "dice": expected_dice,
        "hit_rate": expected_hit,
        "hits": expected_hits,
        "findings": expected_findings,
    },
    "dice_tolerance": tolerance,
    "variants": records,
}, indent=2, sort_keys=True) + "\n")
print("All epoch-0 val20 results preserve hit counts and fit the Dice tolerance.")
PY
  summarize_partial
fi

if [[ "$RUN_SMOKE_ONLY" == "1" ]]; then
  echo "Exp009 smoke and epoch-0 gates complete."
  exit 0
fi
if [[ "$RUN_FULL" != "1" ]]; then
  echo "RUN_FULL=$RUN_FULL; full training not launched."
  exit 0
fi

run_train_segment() {
  local variant="$1" s3_variant="$2" gpu="$3" epoch="$4" previous_epoch="$5"
  local arm_dir="$GROUP_DIR/$variant"
  local target_update="$((epoch * STEPS_PER_EPOCH))"
  local checkpoint="$arm_dir/checkpoints/checkpoint_update_$(printf '%06d' "$target_update").pth"
  if checkpoint_has_update "$checkpoint" "$target_update"; then
    echo "Training already complete: $variant epoch=$epoch"
    return 0
  fi
  local resume_args=()
  if [[ "$previous_epoch" == "0" ]]; then
    resume_args=(
      --init-checkpoint "$arm_dir/model_epoch000/fold_0/checkpoint_final.pth"
    )
  else
    local previous_update="$((previous_epoch * STEPS_PER_EPOCH))"
    local previous_checkpoint="$arm_dir/checkpoints/checkpoint_update_$(printf '%06d' "$previous_update").pth"
    checkpoint_has_update "$previous_checkpoint" "$previous_update" || {
      echo "Missing previous checkpoint: $previous_checkpoint" >&2
      return 1
    }
    resume_args=(--resume-checkpoint "$previous_checkpoint")
  fi
  mapfile -t args < <(common_train_args "$s3_variant")
  CUDA_VISIBLE_DEVICES="$gpu" python \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${args[@]}" \
    --run-dir "$arm_dir" \
    --gpu 0 \
    --epochs "$EPOCHS" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --target-global-update "$target_update" \
    --checkpoint-every-updates 0 \
    --checkpoint-updates "$CHECKPOINT_UPDATES" \
    --latest-checkpoint-every-updates 500 \
    --no-materialize-final-model \
    "${resume_args[@]}" \
    >"$arm_dir/logs/train_segment_to_epoch$(printf '%03d' "$epoch").log" 2>&1
  checkpoint_has_update "$checkpoint" "$target_update"
  cp "$arm_dir/reports/training_metrics.json" \
    "$arm_dir/reports/training_metrics_segment_epoch$(printf '%03d' "$epoch").json"
}

run_arm() {
  local variant="$1" s3_variant="$2" gpu="$3"
  local arm_dir="$GROUP_DIR/$variant"
  local previous_epoch=0
  for epoch in $SEGMENT_EPOCHS; do
    run_train_segment "$variant" "$s3_variant" "$gpu" "$epoch" "$previous_epoch"
    local update="$((epoch * STEPS_PER_EPOCH))"
    local checkpoint="$arm_dir/checkpoints/checkpoint_update_$(printf '%06d' "$update").pth"
    materialize_checkpoint_model "$arm_dir" "$checkpoint" "$epoch" "$s3_variant"
    run_eval "$variant" "$gpu" "$epoch" 20
    summarize_partial
    previous_epoch="$epoch"
  done
  run_eval "$variant" "$gpu" 100 200
  touch "$arm_dir/.complete"
  summarize_partial
}

arm_pids=()
arm_variants=()
for index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$index]}"
  s3_variant="${S3_VARIANTS[$index]}"
  gpu="${GPUS[$index]}"
  if arm_failed "$variant"; then
    echo "Skipping failed arm before full run: $variant"
    continue
  fi
  (
    if run_arm "$variant" "$s3_variant" "$gpu"; then
      touch "$GROUP_DIR/$variant/.arm_success"
    else
      mark_arm_failed "$variant" "full_training_or_eval" "$GROUP_DIR/$variant/logs/supervisor.log"
      exit 1
    fi
  ) >"$GROUP_DIR/$variant/logs/supervisor.log" 2>&1 &
  arm_pids+=("$!")
  arm_variants+=("$variant")
  sleep "$ARM_STAGGER_SECONDS"
done

[[ "${#arm_pids[@]}" -gt 0 ]] || {
  summarize_partial
  touch "$GROUP_DIR/.experiment_failed"
  echo "No active Exp009 arms remain for full training." >&2
  exit 1
}

status=0
for index in "${!arm_pids[@]}"; do
  pid="${arm_pids[$index]}"
  if ! wait "$pid"; then
    echo "Exp009 active arm failed during full run: ${arm_variants[$index]}" >&2
    status=1
  fi
done
summarize_partial
if [[ "$status" == "0" ]]; then
  if any_failed_arm; then
    touch "$GROUP_DIR/.experiment_complete_with_failed_arms"
  else
    touch "$GROUP_DIR/.experiment_complete"
  fi
else
  touch "$GROUP_DIR/.experiment_failed"
fi
echo "Experiment 009 group dir: $GROUP_DIR"
exit "$status"
