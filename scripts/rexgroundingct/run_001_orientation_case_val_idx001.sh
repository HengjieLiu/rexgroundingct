#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval}"
DIAG_DIR="$EXP_DIR/diagnostics/orientation_case_val_idx001"
REX_DATA_ROOT="${REX_DATA_ROOT:-/data/hengjie/datasets/rexgroundingct}"
CASE_INDEX=1
CASE_NAME="train_13591_a_1.nii.gz"

mkdir -p "$DIAG_DIR/after/predictions" "$DIAG_DIR/eval" "$DIAG_DIR/logs" "$DIAG_DIR/reports"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to run CPU inference")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
PY

DIAG_DIR="$DIAG_DIR" CASE_INDEX="$CASE_INDEX" CASE_NAME="$CASE_NAME" python - <<'PY'
import json
import os
from pathlib import Path

metadata = Path("/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json")
diag_dir = Path(os.environ["DIAG_DIR"])
case_index = int(os.environ["CASE_INDEX"])
case_name = os.environ["CASE_NAME"]
data = json.loads(metadata.read_text())
entry = dict(data["val"][case_index])
if entry["name"] != case_name:
    raise SystemExit(f"Expected {case_name}, found {entry['name']} at val index {case_index}")
dataset = {
    "test": [
        {
            "name": entry["name"],
            "seg_path": entry["name"],
            "findings": entry.get("findings", {}),
            "categories": entry.get("categories", {}),
        }
    ]
}
out = diag_dir / "eval" / "val_idx001_as_test_dataset.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n")
print(f"Wrote one-case dataset JSON: {out}")
PY

python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
  --split val \
  --case-index "$CASE_INDEX" \
  --case-name "$CASE_NAME" \
  --output-dir "$DIAG_DIR/after/predictions" \
  --gpu "${GPU:-0}" \
  --overwrite \
  --status-json "$DIAG_DIR/logs/after_status.json" \
  2>&1 | tee "$DIAG_DIR/logs/after_inference.log"

python "$REX_DATA_ROOT/rexrank_eval.py" \
  --gt_dir "$REX_DATA_ROOT/segmentations" \
  --pred_dir "$DIAG_DIR/after/predictions" \
  --dataset_json "$DIAG_DIR/eval/val_idx001_as_test_dataset.json" \
  --output_json "$DIAG_DIR/eval/after_official_eval.json" \
  --num_workers 1

python /workspace/scripts/rexgroundingct/summarize_orientation_case.py \
  --case-name "$CASE_NAME" \
  --baseline-eval-json "$EXP_DIR/eval/val_official_eval.json" \
  --after-eval-json "$DIAG_DIR/eval/after_official_eval.json" \
  --after-status-json "$DIAG_DIR/logs/after_status.json" \
  --gt-path "$REX_DATA_ROOT/segmentations/$CASE_NAME" \
  --baseline-pred-path "$EXP_DIR/predictions/$CASE_NAME" \
  --after-pred-path "$DIAG_DIR/after/predictions/$CASE_NAME" \
  --output-md "$DIAG_DIR/reports/orientation_case_val_idx001_report.md" \
  --output-json "$DIAG_DIR/reports/orientation_case_val_idx001_summary.json" \
  --fail-unless-improved

echo "Single-case orientation report: $DIAG_DIR/reports/orientation_case_val_idx001_report.md"
