#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${1:?usage: run_rexrank_eval.sh EXP_DIR [split] [limit]}"
SPLIT="${2:-val}"
LIMIT="${3:-}"

REX_DATA_ROOT="${REX_DATA_ROOT:-/data/hengjie/datasets/rexgroundingct}"
PRED_DIR="$EXP_DIR/predictions"
EVAL_DIR="$EXP_DIR/eval"
REPORT_DIR="$EXP_DIR/reports"
mkdir -p "$EVAL_DIR" "$REPORT_DIR"

EVAL_DATASET_JSON="$EVAL_DIR/${SPLIT}_as_test_dataset.json"
if [[ "${GLOBAL_ONLY:-0}" == "1" ]]; then
  EVAL_JSON="$EVAL_DIR/${SPLIT}_quick_global_eval.json"
else
  EVAL_JSON="$EVAL_DIR/${SPLIT}_official_eval.json"
fi
REPORT_MD="$REPORT_DIR/${SPLIT}_evaluation_report.md"
SUMMARY_JSON="$REPORT_DIR/${SPLIT}_quick_global_eval_summary.json"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

python /workspace/scripts/rexgroundingct/prepare_eval_dataset_json.py \
  --split "$SPLIT" \
  --output-json "$EVAL_DATASET_JSON" \
  "${LIMIT_ARGS[@]}"

if [[ "${GLOBAL_ONLY:-0}" == "1" ]]; then
  python "$REX_DATA_ROOT/rexrank_eval.py" \
    --gt_dir "$REX_DATA_ROOT/segmentations" \
    --pred_dir "$PRED_DIR" \
    --dataset_json "$EVAL_DATASET_JSON" \
    --output_json "$EVAL_JSON" \
    --num_workers "${NUM_WORKERS:-16}" \
    --global_only

  python /workspace/scripts/rexgroundingct/summarize_quick_global_eval.py \
    --eval-json "$EVAL_JSON" \
    --output-md "$REPORT_MD" \
    --output-json "$SUMMARY_JSON" \
    --label "Experiment 001 corrected orientation MICCAI val quick eval"
else
  python "$REX_DATA_ROOT/rexrank_eval.py" \
    --gt_dir "$REX_DATA_ROOT/segmentations" \
    --pred_dir "$PRED_DIR" \
    --dataset_json "$EVAL_DATASET_JSON" \
    --output_json "$EVAL_JSON" \
    --num_workers "${NUM_WORKERS:-16}"

  python /workspace/scripts/rexgroundingct/summarize_rexrank_eval.py \
    --eval-json "$EVAL_JSON" \
    --output-md "$REPORT_MD"
fi

echo "Evaluation JSON: $EVAL_JSON"
echo "Report: $REPORT_MD"
