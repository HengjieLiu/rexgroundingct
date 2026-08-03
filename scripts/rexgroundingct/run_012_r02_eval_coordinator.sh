#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

GROUP_DIR="${GROUP_DIR:?GROUP_DIR is required}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:?SOURCE_MODEL_DIR is required}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
EMBEDDINGS="${EMBEDDINGS:?EMBEDDINGS is required}"
CACHE_DIR="${CACHE_DIR:?CACHE_DIR is required}"
EVAL_ARM="${EVAL_ARM:?EVAL_ARM is required}"
EVAL_GPU="${EVAL_GPU:?EVAL_GPU is required}"
EVAL_EPOCH="${EVAL_EPOCH:?EVAL_EPOCH is required}"
EVAL_SCOPE="${EVAL_SCOPE:?EVAL_SCOPE is required}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"

arm_dataset() {
  case "$EVAL_ARM:$EVAL_SCOPE" in
    category_1alldiffuse_replay25:target|category_1alldiffuse_replay10:target|category_1alldiffuse_replay00:target)
      echo /workspace/configs/evaluation/rexgroundingct_val42_category_1alldiffuse_target_census.json ;;
    category_1alldiffuse_replay25:union|category_1alldiffuse_replay10:union|category_1alldiffuse_replay00:union)
      echo /workspace/configs/evaluation/rexgroundingct_val80_category_1alldiffuse_milestone_union.json ;;
    category_2d_replay50:target)
      echo /workspace/configs/evaluation/rexgroundingct_val119_category_2d_target_census.json ;;
    category_2d_replay50:union)
      echo /workspace/configs/evaluation/rexgroundingct_val160_category_2d_milestone_union.json ;;
    *:val200) echo "$VAL200_JSON" ;;
    *) echo "Unsupported arm/scope: $EVAL_ARM/$EVAL_SCOPE" >&2; return 1 ;;
  esac
}

verify_complete() {
  local eval_dir="$1" dataset="$2"
  python - "$eval_dir" "$dataset" <<'PY'
import json
import sys
from pathlib import Path

root, dataset_path = map(Path, sys.argv[1:3])
dataset = json.loads(dataset_path.read_text())["test"]
summary_path = root / "reports" / "val_quick_global_eval_summary.json"
eval_path = root / "eval" / "val_quick_global_eval.json"
predictions = root / "predictions"
if not summary_path.is_file() or not eval_path.is_file() or not predictions.is_dir():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text())
expected_findings = sum(len(entry["findings"]) for entry in dataset)
if int(summary.get("total_cases", -1)) != len(dataset):
    raise SystemExit(1)
if int(summary.get("total_findings", -1)) != expected_findings:
    raise SystemExit(1)
missing = [entry["name"] for entry in dataset if not (predictions / entry["name"]).is_file()]
if missing:
    raise SystemExit(1)
print(
    f"accepted cases={len(dataset)} findings={expected_findings} "
    f"dice={summary.get('mean_global_dice_per_finding')}"
)
PY
}

checkpoint_for_epoch() {
  if [[ "$EVAL_EPOCH" == "0" ]]; then
    echo "$SOURCE_CHECKPOINT"
  else
    printf '%s/%s/checkpoints/checkpoint_update_%06d.pth\n' \
      "$GROUP_DIR" "$EVAL_ARM" "$((EVAL_EPOCH * 100))"
  fi
}

materialize_model() {
  local checkpoint="$1" model_dir="$2"
  mkdir -p "$model_dir/fold_0"
  cp "$SOURCE_MODEL_DIR/plans.json" "$model_dir/plans.json"
  local target="$model_dir/fold_0/checkpoint_final.pth"
  if [[ ! -s "$target" || "$(stat -c %s "$target")" != "$(stat -c %s "$checkpoint")" ]]; then
    local tmp="$target.tmp.$$"
    rm -f "$tmp"
    ln "$checkpoint" "$tmp" 2>/dev/null || cp "$checkpoint" "$tmp"
    mv -f "$tmp" "$target"
  fi
}

dataset="$(arm_dataset)"
eval_dir="$GROUP_DIR/$EVAL_ARM/eval_epoch$(printf '%03d' "$EVAL_EPOCH")_$EVAL_SCOPE"
if verify_complete "$eval_dir" "$dataset" 2>/dev/null; then
  echo "Evaluation already complete: arm=$EVAL_ARM epoch=$EVAL_EPOCH scope=$EVAL_SCOPE"
  exit 0
fi

checkpoint="$(checkpoint_for_epoch)"
[[ -s "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
mkdir -p "$eval_dir/logs" "$eval_dir/predictions"
exec 9>"$eval_dir/.eval.lock"
flock -n 9 || { echo "Evaluation lock is held: $eval_dir" >&2; exit 3; }

model_dir="$GROUP_DIR/$EVAL_ARM/model_epoch$(printf '%03d' "$EVAL_EPOCH")"
materialize_model "$checkpoint" "$model_dir"
log="$eval_dir/logs/eval_$(date -u +%Y%m%dT%H%M%SZ).log"
(
  set -euo pipefail
  EVAL_LOCK_HELD=1 CUDA_VISIBLE_DEVICES="$EVAL_GPU" python \
    /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split val \
    --dataset-json "$dataset" \
    --gpu 0 \
    --model-dir "$model_dir" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$eval_dir/predictions" \
    --output-type mask \
    --threshold 0.5 \
    --preprocessed-cache-dir "$CACHE_DIR"
  EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset" \
    EVAL_LABEL="Experiment 012 r02 $EVAL_ARM epoch $EVAL_EPOCH $EVAL_SCOPE threshold 0.5" \
    NUM_WORKERS=8 bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$eval_dir" val
) >"$log" 2>&1

verify_complete "$eval_dir" "$dataset"
echo "Evaluation complete: arm=$EVAL_ARM epoch=$EVAL_EPOCH scope=$EVAL_SCOPE"
