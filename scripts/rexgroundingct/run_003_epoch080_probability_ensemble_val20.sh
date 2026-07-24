#!/usr/bin/env bash
set -euo pipefail

EXP_ID="003_voxtell_rex_ft_rescue_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
RUN_GROUP="${RUN_GROUP:-exp003_full_20260723T075256Z}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
SEED="${SEED:-20260723}"
EPOCH="${EPOCH:-80}"
EPOCH_LABEL="$(printf '%03d' "$EPOCH")"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
UPDATE="$(printf '%06d' "$((EPOCH * STEPS_PER_EPOCH))")"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
PROBE_NAME="${PROBE_NAME:-val20}"
DATASET_JSON="${DATASET_JSON:-}"
if [[ -z "$DATASET_JSON" ]]; then
  case "$PROBE_NAME" in
    val20) DATASET_JSON="$VAL20_JSON" ;;
    val200) DATASET_JSON="$VAL200_JSON" ;;
    *) echo "DATASET_JSON must be set for PROBE_NAME=$PROBE_NAME" >&2; exit 1 ;;
  esac
fi
ENSEMBLE_ROOT="${ENSEMBLE_ROOT:-$GROUP_DIR/ensembles/epoch${EPOCH_LABEL}_${PROBE_NAME}}"
THRESHOLDS="${THRESHOLDS:-0.10 0.20 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.80 0.90}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_SMOKE_EVAL="${RUN_SMOKE_EVAL:-1}"
RUN_FULL="${RUN_FULL:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
OVERWRITE_PROBABILITIES="${OVERWRITE_PROBABILITIES:-0}"
OVERWRITE_ENSEMBLE="${OVERWRITE_ENSEMBLE:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"

variants=(v1_opt v12_opt_poscrop v123_opt_poscrop_emptyloss v1234_opt_poscrop_emptyloss_ds)
if [[ -n "${VARIANTS:-}" ]]; then
  read -r -a variants <<< "$VARIANTS"
fi

variant_gpu() {
  case "$1" in
    v1_opt) echo 0 ;;
    v12_opt_poscrop) echo 1 ;;
    v123_opt_poscrop_emptyloss) echo 2 ;;
    v1234_opt_poscrop_emptyloss_ds) echo 3 ;;
    *) echo "Unknown variant: $1" >&2; return 1 ;;
  esac
}

threshold_label() {
  python - "$1" <<'PY'
import sys
print(f"thr{int(round(float(sys.argv[1]) * 100)):03d}")
PY
}

threshold_dir_name() {
  local threshold="$1"
  printf 'probavg%s_%s' "${#variants[@]}" "$(threshold_label "$threshold")"
}

prepare_model_dir() {
  local variant="$1"
  local full_dir="$GROUP_DIR/$variant/full_100ep"
  local model_dir="$full_dir/model_epoch${EPOCH_LABEL}"
  local checkpoint="$full_dir/checkpoints/checkpoint_update_${UPDATE}.pth"

  if [[ -f "$model_dir/plans.json" && -f "$model_dir/fold_0/checkpoint_final.pth" ]]; then
    echo "$model_dir"
    return 0
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing epoch ${EPOCH} checkpoint for $variant: $checkpoint" >&2
    return 1
  fi

  local plans=""
  for candidate in "$full_dir/model/plans.json" "$GROUP_DIR/$variant/short_005ep/model/plans.json"; do
    if [[ -f "$candidate" ]]; then
      plans="$candidate"
      break
    fi
  done
  if [[ -z "$plans" ]]; then
    echo "Missing plans.json for $variant under $full_dir/model or short_005ep/model" >&2
    return 1
  fi

  mkdir -p "$model_dir/fold_0"
  cp "$plans" "$model_dir/plans.json"
  cp "$checkpoint" "$model_dir/fold_0/checkpoint_final.pth"
  echo "$model_dir"
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

create_smoke_json() {
  local output_json="$1"
  mkdir -p "$(dirname "$output_json")"
  python - "$DATASET_JSON" "$output_json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
data = json.loads(source.read_text())
data["test"] = data.get("test", [])[:1]
target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

run_probability_inference() {
  local dataset_json="$1"
  local probability_root="$2"
  local log_root="$3"
  local tag="$4"
  mkdir -p "$probability_root" "$log_root"

  pids=()
  names=()
  for variant in "${variants[@]}"; do
    local gpu
    gpu="$(variant_gpu "$variant")"
    local model_dir
    model_dir="$(prepare_model_dir "$variant")"
    local output_dir="$probability_root/$variant"
    local status_json="$output_dir/status.json"
    local log="$log_root/probability_${tag}_${variant}.log"

    if [[ "$OVERWRITE_PROBABILITIES" != "1" ]] && dataset_complete "$output_dir" "$dataset_json"; then
      echo "Probability output already complete for $tag $variant: $output_dir"
      continue
    fi

    mkdir -p "$output_dir"
    overwrite_args=()
    if [[ "$OVERWRITE_PROBABILITIES" == "1" ]]; then
      overwrite_args=(--overwrite)
    fi

    (
      set -euo pipefail
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] probability start tag=$tag variant=$variant gpu=$gpu model=$model_dir"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
        --split val \
        --dataset-json "$dataset_json" \
        --gpu 0 \
        --model-dir "$model_dir" \
        --embeddings "$EMBEDDINGS" \
        --output-dir "$output_dir" \
        --output-type probability \
        --status-json "$status_json" \
        "${overwrite_args[@]}"
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] probability done tag=$tag variant=$variant"
    ) >"$log" 2>&1 &
    pids+=("$!")
    names+=("$variant")
  done

  local status=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "Probability inference failed for ${names[$index]} tag=$tag" >&2
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    return "$status"
  fi

  for variant in "${variants[@]}"; do
    dataset_complete "$probability_root/$variant" "$dataset_json"
  done
}

run_ensemble_generation() {
  local dataset_json="$1"
  local probability_root="$2"
  local output_root="$3"
  local log_root="$4"
  local tag="$5"
  mkdir -p "$output_root" "$log_root"

  ensemble_args=(
    --dataset-json "$dataset_json"
    --output-root "$output_root"
    --thresholds $THRESHOLDS
  )
  if [[ "$OVERWRITE_ENSEMBLE" == "1" ]]; then
    ensemble_args+=(--overwrite)
  fi
  for variant in "${variants[@]}"; do
    ensemble_args+=(--input "$variant=$probability_root/$variant")
    ensemble_args+=(--model-dir "$variant=$(prepare_model_dir "$variant")")
  done

  python /workspace/scripts/rexgroundingct/ensemble_voxtell_probabilities.py \
    "${ensemble_args[@]}" \
    2>&1 | tee "$log_root/ensemble_generate_${tag}.log"
}

run_threshold_evals() {
  local dataset_json="$1"
  local output_root="$2"
  local log_root="$3"
  local tag="$4"
  mkdir -p "$log_root"
  if [[ "$RUN_EVAL" != "1" ]]; then
    echo "Skipping eval for $tag because RUN_EVAL=$RUN_EVAL"
    return 0
  fi

  for threshold in $THRESHOLDS; do
    local threshold_dir
    threshold_dir="$output_root/$(threshold_dir_name "$threshold")"
    local label
    label="$(threshold_label "$threshold")"
    mkdir -p "$threshold_dir/logs"
    GLOBAL_ONLY=1 \
      EVAL_DATASET_JSON_OVERRIDE="$dataset_json" \
      EVAL_LABEL="Experiment 003 epoch ${EPOCH} four-model probability average ${tag} ${label} fixed ${PROBE_NAME} seed ${SEED} quick/global eval" \
      NUM_WORKERS="$NUM_WORKERS" \
      bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$threshold_dir" val \
      2>&1 | tee "$threshold_dir/logs/quick_eval_${tag}.log"
  done
}

summarize_thresholds() {
  local dataset_json="$1"
  local output_root="$2"
  local tag="$3"
  python - "$dataset_json" "$output_root" "$tag" "$THRESHOLDS" "${#variants[@]}" "$EPOCH" "$PROBE_NAME" "$GROUP_DIR" "${variants[*]}" <<'PY'
import json
import sys
from pathlib import Path

dataset_json = Path(sys.argv[1])
output_root = Path(sys.argv[2])
tag = sys.argv[3]
thresholds = [float(value) for value in sys.argv[4].split()]
input_count = int(sys.argv[5])
epoch = int(sys.argv[6])
probe_name = sys.argv[7]
group_dir = Path(sys.argv[8])
variants = sys.argv[9].split()


def load_single_model_references() -> list[dict]:
    references = []
    epoch_label = f"{epoch:03d}"
    for variant in variants:
        summary_path = (
            group_dir
            / variant
            / "full_100ep"
            / f"eval_epoch{epoch_label}_{probe_name}"
            / "reports"
            / "val_quick_global_eval_summary.json"
        )
        if not summary_path.is_file():
            continue
        data = json.loads(summary_path.read_text())
        dice = data.get("mean_global_dice_per_finding")
        if dice is None:
            continue
        references.append({
            "label": f"{variant} epoch{epoch_label} {probe_name}",
            "summary_json": str(summary_path),
            "mean_global_dice_per_finding": dice,
            "mean_global_dice_per_case": data.get("mean_global_dice_per_case"),
            "hit_rate": data.get("hit_rate"),
            "total_hits": data.get("total_hits"),
            "total_findings": data.get("total_findings"),
            "total_cases": data.get("total_cases"),
        })
    return references


single_model_references = load_single_model_references()
single_best = (
    max(single_model_references, key=lambda item: item["mean_global_dice_per_finding"])
    if single_model_references
    else None
)

rows = []
for threshold in thresholds:
    label = f"thr{int(round(threshold * 100)):03d}"
    summary_path = output_root / f"probavg{input_count}_{label}" / "reports" / "val_quick_global_eval_summary.json"
    if not summary_path.is_file():
        rows.append({"threshold": threshold, "threshold_label": label, "status": "missing_eval"})
        continue
    data = json.loads(summary_path.read_text())
    rows.append({
        "threshold": threshold,
        "threshold_label": label,
        "status": "complete",
        "mean_global_dice_per_finding": data.get("mean_global_dice_per_finding"),
        "mean_global_dice_per_case": data.get("mean_global_dice_per_case"),
        "hit_rate": data.get("hit_rate"),
        "total_hits": data.get("total_hits"),
        "total_findings": data.get("total_findings"),
        "total_cases": data.get("total_cases"),
        "summary_json": str(summary_path),
    })

complete = [row for row in rows if row["status"] == "complete"]
best = max(complete, key=lambda row: row["mean_global_dice_per_finding"]) if complete else None
payload = {
    "tag": tag,
    "dataset_json": str(dataset_json),
    "thresholds": thresholds,
    "rows": rows,
    "best_by_mean_global_dice_per_finding": best,
    "single_model_references": single_model_references,
    "single_model_reference": single_best,
}
json_path = output_root / "threshold_sweep_summary.json"
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

lines = [
    f"# Exp003 Epoch {epoch} Probability Ensemble Threshold Sweep ({tag})",
    "",
    f"- Dataset JSON: `{dataset_json}`",
]
if single_best is not None:
    lines.extend([
        f"- Single-model reference: `{single_best['label']}` Dice `{single_best['mean_global_dice_per_finding']:.6f}`, hit rate `{single_best['hit_rate']:.6f}` ({single_best['total_hits']}/{single_best['total_findings']})",
        "",
    ])
else:
    lines.extend(["- Single-model reference: unavailable", ""])
lines.extend([
    "| Threshold | Dice/Finding | Hit Rate | Hits/Targets | Status |",
    "| --- | ---: | ---: | ---: | --- |",
])
for row in rows:
    if row["status"] != "complete":
        lines.append(f"| {row['threshold']:.2f} |  |  |  | {row['status']} |")
        continue
    lines.append(
        f"| {row['threshold']:.2f} | {row['mean_global_dice_per_finding']:.6f} | "
        f"{row['hit_rate']:.6f} | {row['total_hits']}/{row['total_findings']} | complete |"
    )
if best is not None:
    lines.extend([
        "",
        f"Best Dice/Finding: threshold `{best['threshold']:.2f}` with `{best['mean_global_dice_per_finding']:.6f}` and hit rate `{best['hit_rate']:.6f}`.",
    ])
md_path = output_root / "threshold_sweep_summary.md"
md_path.write_text("\n".join(lines) + "\n")
print(f"Threshold summary JSON: {json_path}")
print(f"Threshold summary MD: {md_path}")
PY
}

echo "Exp003 epoch ${EPOCH} probability ensemble ${PROBE_NAME}"
echo "group=$GROUP_DIR"
echo "ensemble_root=$ENSEMBLE_ROOT"
echo "dataset_json=$DATASET_JSON"
echo "thresholds=$THRESHOLDS"
mkdir -p "$ENSEMBLE_ROOT/logs" "$ENSEMBLE_ROOT/config"

if [[ "$RUN_SMOKE" == "1" ]]; then
  SMOKE_JSON="$ENSEMBLE_ROOT/config/${PROBE_NAME}_first_case_seed${SEED}.json"
  SMOKE_ROOT="$ENSEMBLE_ROOT/smoke_first_case"
  create_smoke_json "$SMOKE_JSON"
  run_probability_inference "$SMOKE_JSON" "$SMOKE_ROOT/probabilities" "$SMOKE_ROOT/logs" "smoke"
  run_ensemble_generation "$SMOKE_JSON" "$SMOKE_ROOT/probabilities" "$SMOKE_ROOT" "$SMOKE_ROOT/logs" "smoke"
  if [[ "$RUN_SMOKE_EVAL" == "1" ]]; then
    run_threshold_evals "$SMOKE_JSON" "$SMOKE_ROOT" "$SMOKE_ROOT/logs" "smoke"
    summarize_thresholds "$SMOKE_JSON" "$SMOKE_ROOT" "smoke"
  fi
fi

if [[ "$RUN_FULL" == "1" ]]; then
  run_probability_inference "$DATASET_JSON" "$ENSEMBLE_ROOT/probabilities" "$ENSEMBLE_ROOT/logs" "$PROBE_NAME"
  run_ensemble_generation "$DATASET_JSON" "$ENSEMBLE_ROOT/probabilities" "$ENSEMBLE_ROOT" "$ENSEMBLE_ROOT/logs" "$PROBE_NAME"
  run_threshold_evals "$DATASET_JSON" "$ENSEMBLE_ROOT" "$ENSEMBLE_ROOT/logs" "$PROBE_NAME"
  summarize_thresholds "$DATASET_JSON" "$ENSEMBLE_ROOT" "$PROBE_NAME"
fi

echo "Ensemble root: $ENSEMBLE_ROOT"
