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
EVAL_LABEL="${EVAL_LABEL:-ReXGroundingCT ${SPLIT} quick/global eval}"
EVAL_DATASET_JSON_OVERRIDE="${EVAL_DATASET_JSON_OVERRIDE:-}"
FORCE_REX_EVAL="${FORCE_REX_EVAL:-0}"
LOCK_FILE="$EXP_DIR/.eval.lock"
LOCK_POLL_SECONDS="${EVAL_LOCK_POLL_SECONDS:-60}"
LOCK_STALE_SECONDS="${EVAL_LOCK_STALE_SECONDS:-43200}"
EVAL_LOCK_ACQUIRED=0

dataset_json_source() {
  if [[ -n "$EVAL_DATASET_JSON_OVERRIDE" ]]; then
    echo "$EVAL_DATASET_JSON_OVERRIDE"
  else
    echo "$EVAL_DATASET_JSON"
  fi
}

eval_complete() {
  if [[ "$FORCE_REX_EVAL" == "1" ]]; then
    return 1
  fi
  local dataset_json
  dataset_json="$(dataset_json_source)"
  python - "$SUMMARY_JSON" "$EVAL_JSON" "$PRED_DIR" "$dataset_json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
eval_path = Path(sys.argv[2])
pred_dir = Path(sys.argv[3])
dataset_path = Path(sys.argv[4])
if not (summary_path.is_file() and eval_path.is_file() and pred_dir.is_dir() and dataset_path.is_file()):
    raise SystemExit(1)
dataset = json.loads(dataset_path.read_text())
entries = dataset.get("test", [])
expected_cases = len(entries)
expected_findings = sum(len(entry.get("findings", {})) for entry in entries)
summary = json.loads(summary_path.read_text())
if int(summary.get("total_cases", -1)) != expected_cases:
    raise SystemExit(1)
if int(summary.get("total_findings", -1)) != expected_findings:
    raise SystemExit(1)
predictions = list(pred_dir.glob("*.nii.gz"))
if len(predictions) != expected_cases:
    raise SystemExit(1)
if any(not (pred_dir / entry["name"]).is_file() for entry in entries):
    raise SystemExit(1)
PY
}

lock_phase() {
  python - "$LOCK_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    print(json.loads(path.read_text()).get("phase", "unknown"))
except Exception:
    print("unknown")
PY
}

lock_age_seconds() {
  python - "$LOCK_FILE" <<'PY'
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
print(max(0.0, time.time() - path.stat().st_mtime))
PY
}

write_lock() {
  local phase="$1"
  python - "$LOCK_FILE" "$phase" "$0" "$EXP_DIR" <<'PY'
import datetime as dt
import json
import os
import socket
import sys
from pathlib import Path

path = Path(sys.argv[1])
phase = sys.argv[2]
path.write_text(json.dumps({
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "pid": os.getpid(),
    "phase": phase,
    "command": sys.argv[3:],
}, indent=2, sort_keys=True) + "\n")
PY
}

release_eval_lock() {
  if [[ "$EVAL_LOCK_ACQUIRED" == "1" ]]; then
    rm -f "$LOCK_FILE"
  fi
}

acquire_eval_lock() {
  if [[ "${EVAL_LOCK_HELD:-0}" == "1" || "${EVAL_LOCK_DISABLE:-0}" == "1" ]]; then
    return 0
  fi
  while true; do
    if eval_complete; then
      echo "Evaluation already complete: $SUMMARY_JSON"
      exit 0
    fi
    if [[ -f "$LOCK_FILE" ]]; then
      local phase
      phase="$(lock_phase)"
      local age
      age="$(lock_age_seconds)"
      if python - "$age" "$LOCK_STALE_SECONDS" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
      then
        echo "Removing stale eval lock: $LOCK_FILE"
        rm -f "$LOCK_FILE"
        continue
      fi
      if [[ "$phase" == "inference_complete_pending_eval" ]]; then
        write_lock "eval"
        EVAL_LOCK_ACQUIRED=1
        trap release_eval_lock EXIT
        return 0
      fi
      echo "Waiting for eval lock: $LOCK_FILE"
      sleep "$LOCK_POLL_SECONDS"
      continue
    fi
    if ( set -o noclobber; printf 'pending\n' > "$LOCK_FILE" ) 2>/dev/null; then
      write_lock "eval"
      EVAL_LOCK_ACQUIRED=1
      trap release_eval_lock EXIT
      return 0
    fi
  done
}

if eval_complete; then
  echo "Evaluation already complete: $SUMMARY_JSON"
  exit 0
fi

acquire_eval_lock

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

if [[ -n "$EVAL_DATASET_JSON_OVERRIDE" ]]; then
  if [[ -n "$LIMIT" ]]; then
    echo "Ignoring positional limit because EVAL_DATASET_JSON_OVERRIDE is set: $EVAL_DATASET_JSON_OVERRIDE"
  fi
  cp "$EVAL_DATASET_JSON_OVERRIDE" "$EVAL_DATASET_JSON"
else
  python /workspace/scripts/rexgroundingct/prepare_eval_dataset_json.py \
    --split "$SPLIT" \
    --output-json "$EVAL_DATASET_JSON" \
    "${LIMIT_ARGS[@]}"
fi

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
    --label "$EVAL_LABEL"
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
