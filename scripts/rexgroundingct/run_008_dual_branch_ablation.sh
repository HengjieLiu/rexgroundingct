#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

EXP_ID="008_voxtell_dual_branch_proposal_refinement_ablation"
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
RUN_GROUP="${RUN_GROUP:-exp008_dual_branch_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
SCHEDULE="$EXP_DIR/config/train_schedule_continuation_events_${SCHEDULE_START}_$((SCHEDULE_START + SCHEDULE_EVENTS - 1))_seed${SEED}.jsonl"
SCHEDULE_MANIFEST="$EXP_DIR/config/train_schedule_continuation_events_${SCHEDULE_START}_$((SCHEDULE_START + SCHEDULE_EVENTS - 1))_seed${SEED}.manifest.json"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-2000,4000,6000,8000,10000}"
SEGMENT_EPOCHS="${SEGMENT_EPOCHS:-20 40 60 80 100}"
PROPOSAL_THRESHOLDS="${PROPOSAL_THRESHOLDS:-0.1,0.3,0.5}"
IFS=',' read -r -a PROPOSAL_THRESHOLD_VALUES <<< "$PROPOSAL_THRESHOLDS"
RUN_STATIC_ONLY="${RUN_STATIC_ONLY:-0}"
RUN_SMOKE_ONLY="${RUN_SMOKE_ONLY:-0}"
RUN_FULL="${RUN_FULL:-1}"
RUN_MODEL_MATERIALIZATION="${RUN_MODEL_MATERIALIZATION:-1}"
RUN_TRAIN_SMOKES="${RUN_TRAIN_SMOKES:-1}"
RUN_BATCH_BENCHMARK="${RUN_BATCH_BENCHMARK:-1}"
RUN_LOGIT_EQUIVALENCE="${RUN_LOGIT_EQUIVALENCE:-1}"
RUN_EPOCH0_EVAL="${RUN_EPOCH0_EVAL:-1}"
EPOCH0_DICE_TOLERANCE="${EPOCH0_DICE_TOLERANCE:-0.0001}"
ARM_STAGGER_SECONDS="${ARM_STAGGER_SECONDS:-10}"
PAUSE_AFTER_EPOCH="${PAUSE_AFTER_EPOCH:-0}"
RUN_CONTROL_POLL_SECONDS="${RUN_CONTROL_POLL_SECONDS:-5}"
EXPECTED_EXP006_SCHEDULE_SHA="f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776"
EXPECTED_EXTENDED_SCHEDULE_SHA="acdfed8dd8d9908e9ea33d6790d47f8f7e3cbeed4bad326274d115d38e96a0aa"
EXPECTED_CACHE_MANIFEST_SHA="fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3"
EXPECTED_VAL20_SHA="31557d624c47ee8c06799bf89cceb862471b34cfd47065001d6092e32d0dab5d"
EXPECTED_VAL200_SHA="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"

VARIANTS=(
  v1_sharedfusion_softguide
  v1_dualfusion_softguide
  v2_dualfusion_precision
  v3_dualfusion_softguide_joint
)
GPUS=(0 1 2 3)

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$EXP_DIR/reports" "$GROUP_DIR"
printf '%s\n' "$RUN_GROUP" > "$EXP_DIR/config/latest_run_group.txt"
ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest"

PAUSE_MARKER=""
if [[ "$PAUSE_AFTER_EPOCH" -gt 0 ]]; then
  mkdir -p "$GROUP_DIR/control"
  PAUSE_MARKER="$GROUP_DIR/control/pause_after_epoch$(printf '%03d' "$PAUSE_AFTER_EPOCH")"
  touch "$PAUSE_MARKER"
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
  local eval_dir="$1" dataset_json="$2" dataset_size="$3"
  python - "$eval_dir" "$dataset_json" "$dataset_size" "$PROPOSAL_THRESHOLDS" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
dataset_size = int(sys.argv[3])
proposal_thresholds = [
    float(value) for value in sys.argv[4].split(",") if value.strip()
]
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
if dataset_size == 20:
    diagnostics = root / "reports" / "proposal_diagnostics.json"
    if not diagnostics.is_file():
        raise SystemExit(1)
    for threshold in proposal_thresholds:
        label = f"thr{round(threshold * 100):03d}"
        proposal_root = root / "proposal" / label
        proposal_summary_path = (
            proposal_root / "reports" / "val_quick_global_eval_summary.json"
        )
        proposal_predictions = proposal_root / "predictions"
        if not proposal_summary_path.is_file():
            raise SystemExit(1)
        proposal_summary = json.loads(proposal_summary_path.read_text())
        if int(proposal_summary.get("total_cases", -1)) != len(entries):
            raise SystemExit(1)
        if int(proposal_summary.get("total_findings", -1)) != sum(
            len(e["findings"]) for e in entries
        ):
            raise SystemExit(1)
        if any(
            not (proposal_predictions / entry["name"]).is_file()
            for entry in entries
        ):
            raise SystemExit(1)
PY
}

materialize_checkpoint_model() {
  local arm_dir="$1" checkpoint="$2" epoch="$3"
  local model_dir="$arm_dir/model_epoch$(printf '%03d' "$epoch")"
  mkdir -p "$model_dir/fold_0"
  cp "$SOURCE_MODEL_DIR/plans.json" "$model_dir/plans.json"
  cp "$arm_dir/model_epoch000/model_spec.json" "$model_dir/model_spec.json"
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
    python /workspace/scripts/rexgroundingct/summarize_008_results.py \
      --group-dir "$GROUP_DIR" \
      --output-json "$EXP_DIR/reports/dual_branch_ablation_summary.json" \
      --output-md "$EXP_DIR/reports/dual_branch_ablation_report.md" \
      >/dev/null
  ) 9>"$EXP_DIR/reports/.summary.lock"
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

PYTHONDONTWRITEBYTECODE=1 python \
  /workspace/scripts/rexgroundingct/test_voxtell_dual_branch.py
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  /workspace/scripts/rexgroundingct/voxtell_dual_branch.py \
  /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
  /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
  /workspace/scripts/rexgroundingct/materialize_dual_branch_model.py \
  /workspace/scripts/rexgroundingct/check_dual_branch_epoch0_equivalence.py \
  /workspace/scripts/rexgroundingct/benchmark_dual_branch_inference_batch.py \
  /workspace/scripts/rexgroundingct/summarize_proposal_diagnostics.py \
  /workspace/scripts/rexgroundingct/summarize_008_results.py

if [[ "$RUN_STATIC_ONLY" == "1" ]]; then
  echo "Exp008 static checks complete."
  exit 0
fi

if [[ "$RUN_MODEL_MATERIALIZATION" == "1" ]]; then
  for index in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$index]}"
    arm_dir="$GROUP_DIR/$variant"
    model_dir="$arm_dir/model_epoch000"
    manifest="$arm_dir/config/epoch000_initialization_manifest.json"
    mkdir -p "$arm_dir/config" "$arm_dir/logs" "$arm_dir/reports"
    if [[ ! -f "$model_dir/fold_0/checkpoint_final.pth" || ! -f "$manifest" ]]; then
      python /workspace/scripts/rexgroundingct/materialize_dual_branch_model.py \
        --source-model-dir "$SOURCE_MODEL_DIR" \
        --variant "$variant" \
        --output-model-dir "$model_dir" \
        --manifest-json "$manifest" \
        >"$arm_dir/logs/materialize_epoch000.log" 2>&1
    fi
  done
fi

common_train_args() {
  local variant="$1"
  printf '%s\n' \
    --mode train \
    --experiment-id "$EXP_ID" \
    --config "$CONFIG_SNAPSHOT" \
    --exp-dir "$EXP_DIR" \
    --model-dir "$SOURCE_MODEL_DIR" \
    --dual-branch-variant "$variant" \
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
    --proposal-branch-weight 0.5 \
    --proposal-recall-weight 0.5 \
    --proposal-tversky-alpha 0.3 \
    --proposal-tversky-beta 0.7 \
    --precision-tversky-alpha 0.7 \
    --precision-tversky-beta 0.3 \
    --preprocessed-cache-dir "$CACHE_ROOT" \
    --sample-schedule "$SCHEDULE" \
    --allow-experimental-train
}

run_one_update_smoke() {
  local variant="$1" gpu="$2"
  local arm_dir="$GROUP_DIR/$variant"
  local smoke_dir="$GROUP_DIR/smoke/${variant}_one_update"
  mkdir -p "$smoke_dir/logs"
  mapfile -t args < <(common_train_args "$variant")
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
  local variant="v3_dualfusion_softguide_joint"
  local gpu=3
  local arm_dir="$GROUP_DIR/$variant"
  local smoke_dir="$GROUP_DIR/smoke/${variant}_resume"
  mkdir -p "$smoke_dir/logs"
  mapfile -t args < <(common_train_args "$variant")
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
  for index in "${!VARIANTS[@]}"; do
    run_one_update_smoke "${VARIANTS[$index]}" "${GPUS[$index]}" &
    smoke_pids+=("$!")
  done
  smoke_status=0
  for pid in "${smoke_pids[@]}"; do
    if ! wait "$pid"; then
      smoke_status=1
    fi
  done
  [[ "$smoke_status" == "0" ]] || {
    echo "One or more Exp008 one-update smokes failed." >&2
    exit 1
  }
  run_resume_smoke
fi

BENCHMARK_JSON="$GROUP_DIR/benchmark/inference_window_batch.json"
SELECTED_WINDOW_BATCH=1
if [[ "$RUN_BATCH_BENCHMARK" == "1" ]]; then
  mkdir -p "$GROUP_DIR/benchmark"
  CUDA_VISIBLE_DEVICES=3 python \
    /workspace/scripts/rexgroundingct/benchmark_dual_branch_inference_batch.py \
    --model-dir "$GROUP_DIR/v3_dualfusion_softguide_joint/model_epoch000" \
    --dataset-json "$VAL20_JSON" \
    --cache-root "$CACHE_ROOT" \
    --embeddings "$EMBEDDINGS" \
    --case-name train_19753_a_2.nii.gz \
    --case-name train_13481_a_2.nii.gz \
    --case-name train_3029_a_1.nii.gz \
    --batch-sizes 1,2,4 \
    --gpu 0 \
    --max-reserved-gib 42 \
    --minimum-speedup 0.1 \
    --minimum-mask-agreement 0.99999 \
    --output-json "$BENCHMARK_JSON" \
    --output-md "$GROUP_DIR/benchmark/inference_window_batch.md" \
    >"$GROUP_DIR/benchmark/inference_window_batch.log" 2>&1
  SELECTED_WINDOW_BATCH="$(python -c "import json; print(json.load(open('$BENCHMARK_JSON'))['selected_sliding_window_batch_size'])")"
fi
printf '%s\n' "$SELECTED_WINDOW_BATCH" > "$GROUP_DIR/benchmark/selected_window_batch_size.txt"

if [[ "$RUN_LOGIT_EQUIVALENCE" == "1" ]]; then
  mkdir -p "$GROUP_DIR/benchmark"
  equivalence_pids=()
  for index in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$index]}"
    gpu="${GPUS[$index]}"
    output_json="$GROUP_DIR/benchmark/epoch0_logit_equivalence_${variant}.json"
    (
      CUDA_VISIBLE_DEVICES="$gpu" python \
        /workspace/scripts/rexgroundingct/check_dual_branch_epoch0_equivalence.py \
        --source-model-dir "$SOURCE_MODEL_DIR" \
        --dual-model-dir "$GROUP_DIR/$variant/model_epoch000" \
        --dataset-json "$VAL20_JSON" \
        --cache-root "$CACHE_ROOT" \
        --embeddings "$EMBEDDINGS" \
        --case-name train_19753_a_2.nii.gz \
        --gpu 0 \
        --output-json "$output_json" \
        >"$GROUP_DIR/benchmark/epoch0_logit_equivalence_${variant}.log" 2>&1
    ) &
    equivalence_pids+=("$!")
  done
  equivalence_status=0
  for pid in "${equivalence_pids[@]}"; do
    if ! wait "$pid"; then
      equivalence_status=1
    fi
  done
  [[ "$equivalence_status" == "0" ]] || {
    echo "One or more epoch-0 logit equivalence checks failed." >&2
    exit 1
  }
  python - "$GROUP_DIR/benchmark" "${VARIANTS[@]}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
sections = (
    "source_repeatability",
    "source_vs_proposal",
    "source_vs_final",
    "proposal_vs_final",
)
for variant in sys.argv[2:]:
    path = root / f"epoch0_logit_equivalence_{variant}.json"
    report = json.loads(path.read_text())
    for section in sections:
        rows = report[section]
        if len(rows) != 5:
            raise SystemExit(f"{variant} {section}: expected five scales")
        for row in rows:
            if (
                not row["logits_bitwise_equal"]
                or row["max_abs_logit_difference"] != 0.0
                or row["threshold_mask_agreement"] != 1.0
            ):
                raise SystemExit(
                    f"{variant} {section} scale {row['scale']}: "
                    f"epoch-0 logits are not bitwise identical"
                )
print("All four epoch-0 models are bitwise source-equivalent at five scales.")
PY
fi

run_eval() {
  local variant="$1" gpu="$2" epoch="$3" dataset_size="$4" window_batch="$5"
  local arm_dir="$GROUP_DIR/$variant"
  local dataset_json model_dir eval_dir
  if [[ "$dataset_size" == "20" ]]; then
    dataset_json="$VAL20_JSON"
  else
    dataset_json="$VAL200_JSON"
  fi
  model_dir="$arm_dir/model_epoch$(printf '%03d' "$epoch")"
  eval_dir="$arm_dir/eval_epoch$(printf '%03d' "$epoch")_val${dataset_size}"
  if eval_complete "$eval_dir" "$dataset_json" "$dataset_size"; then
    echo "Already complete: $variant epoch=$epoch val$dataset_size"
    return 0
  fi
  mkdir -p "$eval_dir/logs" "$eval_dir/status" "$eval_dir/predictions"
  proposal_args=()
  if [[ "$dataset_size" == "20" ]]; then
    proposal_args=(
      --proposal-output-root "$eval_dir/proposal"
      --proposal-thresholds "$PROPOSAL_THRESHOLDS"
    )
  fi
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
    --sliding-window-batch-size "$window_batch" \
    --status-json "$eval_dir/status/inference.json" \
    "${proposal_args[@]}" \
    >"$eval_dir/logs/inference.log" 2>&1

  EVAL_LOCK_DISABLE=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset_json" \
    EVAL_LABEL="Experiment 008 $variant epoch $epoch val$dataset_size final threshold 0.5" \
    NUM_WORKERS=8 \
    bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$eval_dir" val \
    >"$eval_dir/logs/final_eval.log" 2>&1

  if [[ "$dataset_size" == "20" ]]; then
    for threshold in "${PROPOSAL_THRESHOLD_VALUES[@]}"; do
      label="$(python -c "print(f'thr{round(float(\"$threshold\") * 100):03d}')")"
      proposal_eval="$eval_dir/proposal/$label"
      EVAL_LOCK_DISABLE=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset_json" \
        EVAL_LABEL="Experiment 008 $variant epoch $epoch proposal threshold $threshold" \
        NUM_WORKERS=8 \
        bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh \
        "$proposal_eval" val \
        >"$eval_dir/logs/proposal_${label}_eval.log" 2>&1
    done
    python /workspace/scripts/rexgroundingct/summarize_proposal_diagnostics.py \
      --dataset-json "$dataset_json" \
      --proposal-output-root "$eval_dir/proposal" \
      --thresholds "$PROPOSAL_THRESHOLDS" \
      --output-json "$eval_dir/reports/proposal_diagnostics.json" \
      --output-md "$eval_dir/reports/proposal_diagnostics.md" \
      >"$eval_dir/logs/proposal_diagnostics.log" 2>&1
  fi
  touch "$eval_dir/.complete"
}

if [[ "$RUN_EPOCH0_EVAL" == "1" ]]; then
  epoch0_pids=()
  for index in "${!VARIANTS[@]}"; do
    run_eval "${VARIANTS[$index]}" "${GPUS[$index]}" 0 20 1 &
    epoch0_pids+=("$!")
  done
  epoch0_status=0
  for pid in "${epoch0_pids[@]}"; do
    if ! wait "$pid"; then
      epoch0_status=1
    fi
  done
  [[ "$epoch0_status" == "0" ]] || {
    echo "One or more epoch-0 val20 evaluations failed." >&2
    exit 1
  }
  python - "$GROUP_DIR" "$EPOCH0_DICE_TOLERANCE" "${VARIANTS[@]}" <<'PY'
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
    proposal_path = (
        root
        / variant
        / "eval_epoch000_val20/reports/proposal_diagnostics.json"
    )
    proposal = json.loads(proposal_path.read_text())["thresholds"]["thr050"]
    proposal_dice = float(proposal["mean_global_dice_per_finding"])
    proposal_hit = float(proposal["hit_rate"])
    records[variant] = {
        "dice": dice,
        "dice_delta_from_stored_exp006": dice - expected_dice,
        "hit_rate": hit,
        "hits": hits,
        "findings": findings,
        "proposal_thr050_dice": proposal_dice,
        "proposal_thr050_hit_rate": proposal_hit,
    }
    if (
        abs(dice - expected_dice) > tolerance
        or abs(hit - expected_hit) > 1e-12
        or hits != expected_hits
        or findings != expected_findings
        or abs(proposal_dice - dice) > 1e-12
        or abs(proposal_hit - hit) > 1e-12
    ):
        raise SystemExit(
            f"{variant}: epoch-0 mismatch dice={dice} hit={hit}; "
            f"expected dice={expected_dice} +/- {tolerance}, hit={expected_hit}, "
            f"hits/findings={expected_hits}/{expected_findings}"
        )
output = root / "benchmark/epoch0_metric_equivalence.json"
output.write_text(json.dumps({
    "stored_exp006_reference": {
        "dice": expected_dice,
        "hit_rate": expected_hit,
        "hits": expected_hits,
        "findings": expected_findings,
    },
    "dice_tolerance": tolerance,
    "reason_for_tolerance": (
        "Independent mixed-precision sliding-window runs on separate GPUs show "
        "small aggregate Dice variation; direct same-patch five-scale logits "
        "are bitwise source-equivalent."
    ),
    "variants": records,
}, indent=2, sort_keys=True) + "\n")
print(
    "All four epoch-0 val20 results preserve hit counts and fall within "
    f"the recorded Dice tolerance {tolerance}."
)
PY
  summarize_partial
fi

if [[ "$RUN_SMOKE_ONLY" == "1" ]]; then
  echo "Exp008 smoke and epoch-0 gates complete."
  exit 0
fi
if [[ "$RUN_FULL" != "1" ]]; then
  echo "RUN_FULL=$RUN_FULL; full training not launched."
  exit 0
fi

run_train_segment() {
  local variant="$1" gpu="$2" epoch="$3"
  local arm_dir="$GROUP_DIR/$variant"
  local target_update="$((epoch * STEPS_PER_EPOCH))"
  local checkpoint="$arm_dir/checkpoints/checkpoint_update_$(printf '%06d' "$target_update").pth"
  if checkpoint_has_update "$checkpoint" "$target_update"; then
    echo "Training already complete: $variant epoch=$epoch"
    return 0
  fi
  local resume_args=()
  if [[ "$target_update" == "2000" ]]; then
    resume_args=(
      --init-checkpoint "$arm_dir/model_epoch000/fold_0/checkpoint_final.pth"
    )
  else
    local previous_update="$((target_update - 2000))"
    local previous_checkpoint="$arm_dir/checkpoints/checkpoint_update_$(printf '%06d' "$previous_update").pth"
    checkpoint_has_update "$previous_checkpoint" "$previous_update" || {
      echo "Missing previous checkpoint: $previous_checkpoint" >&2
      return 1
    }
    resume_args=(--resume-checkpoint "$previous_checkpoint")
  fi
  mapfile -t args < <(common_train_args "$variant")
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
  local variant="$1" gpu="$2"
  local arm_dir="$GROUP_DIR/$variant"
  for epoch in $SEGMENT_EPOCHS; do
    run_train_segment "$variant" "$gpu" "$epoch"
    local update="$((epoch * STEPS_PER_EPOCH))"
    local checkpoint="$arm_dir/checkpoints/checkpoint_update_$(printf '%06d' "$update").pth"
    materialize_checkpoint_model "$arm_dir" "$checkpoint" "$epoch"
    run_eval "$variant" "$gpu" "$epoch" 20 "$SELECTED_WINDOW_BATCH"
    summarize_partial
    if [[ "$PAUSE_AFTER_EPOCH" -gt 0 && "$epoch" -eq "$PAUSE_AFTER_EPOCH" ]]; then
      local paused_status="$arm_dir/.paused_after_epoch$(printf '%03d' "$epoch")"
      touch "$paused_status"
      echo "Paused after epoch $epoch; remove $PAUSE_MARKER to resume $variant."
      while [[ -f "$PAUSE_MARKER" ]]; do
        sleep "$RUN_CONTROL_POLL_SECONDS"
      done
      rm -f "$paused_status"
      echo "Pause released after epoch $epoch; resuming $variant."
    fi
  done
  run_eval "$variant" "$gpu" 100 200 "$SELECTED_WINDOW_BATCH"
  touch "$arm_dir/.complete"
  summarize_partial
}

arm_pids=()
for index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$index]}"
  gpu="${GPUS[$index]}"
  (
    if run_arm "$variant" "$gpu"; then
      touch "$GROUP_DIR/$variant/.arm_success"
    else
      touch "$GROUP_DIR/$variant/.arm_failed"
      exit 1
    fi
  ) >"$GROUP_DIR/$variant/logs/supervisor.log" 2>&1 &
  arm_pids+=("$!")
  sleep "$ARM_STAGGER_SECONDS"
done

status=0
for pid in "${arm_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
summarize_partial
if [[ "$status" == "0" ]]; then
  touch "$GROUP_DIR/.experiment_complete"
else
  touch "$GROUP_DIR/.experiment_failed"
fi
echo "Experiment 008 group dir: $GROUP_DIR"
exit "$status"
