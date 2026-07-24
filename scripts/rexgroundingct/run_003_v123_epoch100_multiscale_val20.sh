#!/usr/bin/env bash
set -euo pipefail

EXP_ID="003_voxtell_rex_ft_rescue_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
RUN_GROUP="${RUN_GROUP:-exp003_full_20260723T075256Z}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
VARIANT="${VARIANT:-v123_opt_poscrop_emptyloss}"
SEED="${SEED:-20260723}"
MODEL_DIR="${MODEL_DIR:-$GROUP_DIR/$VARIANT/full_100ep/model_epoch100}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$GROUP_DIR/multiscale/${VARIANT}_epoch100_val20}"
SCALES="${SCALES:-96 128 160 256}"
THRESHOLDS="${THRESHOLDS:-0.10 0.20 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.80 0.90}"
RUN_MULTISCALE="${RUN_MULTISCALE:-1}"
RUN_NATIVE_PROB="${RUN_NATIVE_PROB:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
OVERWRITE_PROBABILITIES="${OVERWRITE_PROBABILITIES:-0}"
OVERWRITE_THRESHOLDS="${OVERWRITE_THRESHOLDS:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"

scale_gpu() {
  case "$1" in
    96) echo 0 ;;
    128) echo 1 ;;
    160) echo 2 ;;
    256) echo 3 ;;
    *) echo "Unknown scale: $1" >&2; return 1 ;;
  esac
}

scale_label() {
  printf 'scale%03d' "$1"
}

threshold_label() {
  python - "$1" <<'PY'
import sys
print(f"thr{int(round(float(sys.argv[1]) * 100)):03d}")
PY
}

threshold_dir_name() {
  printf 'probavg1_%s' "$(threshold_label "$1")"
}

dataset_complete() {
  local output_dir="$1"
  local dataset_json="$2"
  python - "$output_dir" "$dataset_json" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
dataset_json = Path(sys.argv[2])
if not output_dir.is_dir() or not dataset_json.is_file():
    raise SystemExit(1)
entries = json.loads(dataset_json.read_text()).get("test", [])
if not entries:
    raise SystemExit(1)
if any(not (output_dir / entry["name"]).is_file() for entry in entries):
    raise SystemExit(1)
PY
}

run_one_scale() {
  local scale="$1"
  local gpu
  gpu="$(scale_gpu "$scale")"
  local label
  label="$(scale_label "$scale")"
  local scale_root="$OUTPUT_ROOT/$label"
  local prob_dir="$scale_root/probabilities"
  local log="$scale_root/logs/inference.log"
  local status_json="$prob_dir/status.json"
  mkdir -p "$scale_root/logs" "$prob_dir"

  if [[ "$OVERWRITE_PROBABILITIES" != "1" ]] && dataset_complete "$prob_dir" "$VAL20_JSON"; then
    echo "Probability output already complete for $label: $prob_dir"
    return 0
  fi

  overwrite_args=()
  if [[ "$OVERWRITE_PROBABILITIES" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python /workspace/scripts/rexgroundingct/run_voxtell_multiscale_val_inference.py \
    --dataset-json "$VAL20_JSON" \
    --gpu 0 \
    --model-dir "$MODEL_DIR" \
    --embeddings "$EMBEDDINGS" \
    --source-window-size "$scale" \
    --output-dir "$prob_dir" \
    --status-json "$status_json" \
    "${overwrite_args[@]}" \
    2>&1 | tee "$log"
}

run_native_probability() {
  local root="$OUTPUT_ROOT/native192"
  local prob_dir="$root/probabilities"
  local log="$root/logs/inference.log"
  mkdir -p "$root/logs" "$prob_dir"
  if [[ "$OVERWRITE_PROBABILITIES" != "1" ]] && dataset_complete "$prob_dir" "$VAL20_JSON"; then
    echo "Native probability output already complete: $prob_dir"
    return 0
  fi
  overwrite_args=()
  if [[ "$OVERWRITE_PROBABILITIES" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="${NATIVE_GPU:-0}" PYTHONUNBUFFERED=1 python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split val \
    --dataset-json "$VAL20_JSON" \
    --gpu 0 \
    --model-dir "$MODEL_DIR" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$prob_dir" \
    --output-type probability \
    --status-json "$prob_dir/status.json" \
    "${overwrite_args[@]}" \
    2>&1 | tee "$log"
}

threshold_probabilities() {
  local label="$1"
  local root="$OUTPUT_ROOT/$label"
  local prob_dir="$root/probabilities"
  local log="$root/logs/thresholds.log"
  mkdir -p "$root/logs"
  args=(
    --dataset-json "$VAL20_JSON"
    --output-root "$root"
    --thresholds $THRESHOLDS
    --input "$label=$prob_dir"
    --model-dir "$label=$MODEL_DIR"
  )
  if [[ "$OVERWRITE_THRESHOLDS" == "1" ]]; then
    args+=(--overwrite)
  fi
  python /workspace/scripts/rexgroundingct/ensemble_voxtell_probabilities.py "${args[@]}" \
    2>&1 | tee "$log"
}

run_threshold_evals() {
  local label="$1"
  local root="$OUTPUT_ROOT/$label"
  if [[ "$RUN_EVAL" != "1" ]]; then
    echo "Skipping eval for $label because RUN_EVAL=$RUN_EVAL"
    return 0
  fi
  for threshold in $THRESHOLDS; do
    local threshold_dir="$root/$(threshold_dir_name "$threshold")"
    local tlabel
    tlabel="$(threshold_label "$threshold")"
    mkdir -p "$threshold_dir/logs"
    GLOBAL_ONLY=1 \
      EVAL_DATASET_JSON_OVERRIDE="$VAL20_JSON" \
      EVAL_LABEL="Experiment 003 ${VARIANT} epoch 100 ${label} ${tlabel} fixed val20 seed ${SEED} quick/global eval" \
      NUM_WORKERS="$NUM_WORKERS" \
      bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$threshold_dir" val \
      2>&1 | tee "$threshold_dir/logs/quick_eval.log"
  done
}

summarize_results() {
  python - "$OUTPUT_ROOT" "$GROUP_DIR" "$VARIANT" "$THRESHOLDS" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
group_dir = Path(sys.argv[2])
variant = sys.argv[3]
thresholds = [float(value) for value in sys.argv[4].split()]
native_hard_summary = (
    group_dir / variant / "full_100ep" / "eval_epoch100_val20" /
    "reports" / "val_quick_global_eval_summary.json"
)
labels = ["native192"] + [f"scale{scale:03d}" for scale in (96, 128, 160, 256)]

reference = None
if native_hard_summary.is_file():
    data = json.loads(native_hard_summary.read_text())
    reference = {
        "label": "native192 existing hard-mask eval",
        "mean_global_dice_per_finding": data.get("mean_global_dice_per_finding"),
        "hit_rate": data.get("hit_rate"),
        "total_hits": data.get("total_hits"),
        "total_findings": data.get("total_findings"),
        "summary_json": str(native_hard_summary),
    }

rows = []
best_by_label = []
for label in labels:
    for threshold in thresholds:
        threshold_label = f"thr{int(round(threshold * 100)):03d}"
        summary_path = output_root / label / f"probavg1_{threshold_label}" / "reports" / "val_quick_global_eval_summary.json"
        if not summary_path.is_file():
            rows.append({"label": label, "threshold": threshold, "status": "missing_eval"})
            continue
        data = json.loads(summary_path.read_text())
        rows.append({
            "label": label,
            "threshold": threshold,
            "status": "complete",
            "mean_global_dice_per_finding": data.get("mean_global_dice_per_finding"),
            "mean_global_dice_per_case": data.get("mean_global_dice_per_case"),
            "hit_rate": data.get("hit_rate"),
            "total_hits": data.get("total_hits"),
            "total_findings": data.get("total_findings"),
            "total_cases": data.get("total_cases"),
            "summary_json": str(summary_path),
        })
    complete = [row for row in rows if row["label"] == label and row["status"] == "complete"]
    if complete:
        best_by_label.append(max(complete, key=lambda row: row["mean_global_dice_per_finding"]))

complete_all = [row for row in rows if row["status"] == "complete"]
best_overall = max(complete_all, key=lambda row: row["mean_global_dice_per_finding"]) if complete_all else None
payload = {
    "variant": variant,
    "output_root": str(output_root),
    "thresholds": thresholds,
    "native_hard_mask_reference": reference,
    "rows": rows,
    "best_by_label": best_by_label,
    "best_overall_by_mean_global_dice_per_finding": best_overall,
}
json_path = output_root / "multiscale_val20_summary.json"
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

lines = [
    "# Exp003 v123 Epoch 100 Multiscale Val20",
    "",
    f"- Output root: `{output_root}`",
]
if reference is not None:
    lines.append(
        "- Native hard-mask reference: "
        f"Dice `{reference['mean_global_dice_per_finding']:.6f}`, "
        f"hit `{reference['hit_rate']:.6f}` "
        f"({reference['total_hits']}/{reference['total_findings']})"
    )
lines.extend([
    "",
    "## Best Threshold Per Scale",
    "",
    "| Scale | Best Threshold | Dice/Finding | Hit Rate | Hits/Targets |",
    "| --- | ---: | ---: | ---: | ---: |",
])
for row in best_by_label:
    lines.append(
        f"| {row['label']} | {row['threshold']:.2f} | "
        f"{row['mean_global_dice_per_finding']:.6f} | {row['hit_rate']:.6f} | "
        f"{row['total_hits']}/{row['total_findings']} |"
    )
lines.extend([
    "",
    "## Full Threshold Sweep",
    "",
    "| Scale | Threshold | Dice/Finding | Hit Rate | Hits/Targets | Status |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
])
for row in rows:
    if row["status"] != "complete":
        lines.append(f"| {row['label']} | {row['threshold']:.2f} |  |  |  | {row['status']} |")
        continue
    lines.append(
        f"| {row['label']} | {row['threshold']:.2f} | "
        f"{row['mean_global_dice_per_finding']:.6f} | {row['hit_rate']:.6f} | "
        f"{row['total_hits']}/{row['total_findings']} | complete |"
    )
if best_overall is not None:
    lines.extend([
        "",
        f"Best overall Dice/Finding: `{best_overall['label']}` threshold `{best_overall['threshold']:.2f}` "
        f"with Dice `{best_overall['mean_global_dice_per_finding']:.6f}` and hit rate `{best_overall['hit_rate']:.6f}`.",
    ])
md_path = output_root / "multiscale_val20_summary.md"
md_path.write_text("\n".join(lines) + "\n")
print(f"Multiscale summary JSON: {json_path}")
print(f"Multiscale summary MD: {md_path}")
PY
}

echo "Exp003 ${VARIANT} epoch100 multiscale val20"
echo "model_dir=$MODEL_DIR"
echo "output_root=$OUTPUT_ROOT"
echo "scales=$SCALES thresholds=$THRESHOLDS"
mkdir -p "$OUTPUT_ROOT/logs"

if [[ ! -f "$MODEL_DIR/plans.json" || ! -f "$MODEL_DIR/fold_0/checkpoint_final.pth" ]]; then
  echo "Missing model dir files under $MODEL_DIR" >&2
  exit 1
fi

if [[ "$RUN_MULTISCALE" == "1" ]]; then
  pids=()
  names=()
  for scale in $SCALES; do
    run_one_scale "$scale" &
    pids+=("$!")
    names+=("$(scale_label "$scale")")
  done
  status=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "Multiscale inference failed for ${names[$index]}" >&2
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    exit "$status"
  fi
fi

if [[ "$RUN_NATIVE_PROB" == "1" ]]; then
  run_native_probability
fi

labels=(native192)
for scale in $SCALES; do
  labels+=("$(scale_label "$scale")")
done

for label in "${labels[@]}"; do
  threshold_probabilities "$label"
  run_threshold_evals "$label"
done

summarize_results
echo "Multiscale root: $OUTPUT_ROOT"
