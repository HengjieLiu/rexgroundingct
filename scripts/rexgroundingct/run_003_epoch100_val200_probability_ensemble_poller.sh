#!/usr/bin/env bash
set -euo pipefail

EXP_ID="003_voxtell_rex_ft_rescue_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
RUN_GROUP="${RUN_GROUP:-exp003_full_20260723T075256Z}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
SEED="${SEED:-20260723}"
EPOCH="${EPOCH:-100}"
EPOCH_LABEL="$(printf '%03d' "$EPOCH")"
PROBE_NAME="${PROBE_NAME:-val200}"
DATASET_JSON="${DATASET_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
ENSEMBLE_ROOT="${ENSEMBLE_ROOT:-$GROUP_DIR/ensembles/epoch${EPOCH_LABEL}_${PROBE_NAME}}"
POLL_SECONDS="${POLL_SECONDS:-300}"
DRY_RUN="${DRY_RUN:-0}"
LOCK_STALE_SECONDS="${ENSEMBLE_LOCK_STALE_SECONDS:-43200}"
LOCK_POLL_SECONDS="${ENSEMBLE_LOCK_POLL_SECONDS:-60}"
THRESHOLDS="${THRESHOLDS:-0.10 0.20 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.80 0.90}"

variants=(v1_opt v12_opt_poscrop v123_opt_poscrop_emptyloss v1234_opt_poscrop_emptyloss_ds)
if [[ -n "${VARIANTS:-}" ]]; then
  read -r -a variants <<< "$VARIANTS"
fi

dataset_counts() {
  python - "$DATASET_JSON" <<'PY'
import json
import sys
from pathlib import Path

entries = json.loads(Path(sys.argv[1]).read_text()).get("test", [])
print(len(entries), sum(len(entry.get("findings", {})) for entry in entries))
PY
}

prediction_count() {
  local pred_dir="$1"
  find "$pred_dir" -maxdepth 1 -type f -name '*.nii.gz' 2>/dev/null | wc -l
}

eval_complete() {
  local eval_dir="$1"
  python - "$eval_dir" "$DATASET_JSON" <<'PY'
import json
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
dataset_json = Path(sys.argv[2])
summary = eval_dir / "reports" / "val_quick_global_eval_summary.json"
eval_json = eval_dir / "eval" / "val_quick_global_eval.json"
pred_dir = eval_dir / "predictions"
if not (summary.is_file() and eval_json.is_file() and pred_dir.is_dir() and dataset_json.is_file()):
    raise SystemExit(1)
entries = json.loads(dataset_json.read_text()).get("test", [])
expected_cases = len(entries)
expected_findings = sum(len(entry.get("findings", {})) for entry in entries)
data = json.loads(summary.read_text())
if int(data.get("total_cases", -1)) != expected_cases:
    raise SystemExit(1)
if int(data.get("total_findings", -1)) != expected_findings:
    raise SystemExit(1)
predictions = list(pred_dir.glob("*.nii.gz"))
if len(predictions) != expected_cases:
    raise SystemExit(1)
if any(not (pred_dir / entry["name"]).is_file() for entry in entries):
    raise SystemExit(1)
PY
}

models_ready() {
  local missing=0
  for variant in "${variants[@]}"; do
    local model_dir="$GROUP_DIR/$variant/full_100ep/model_epoch${EPOCH_LABEL}"
    if [[ ! -f "$model_dir/plans.json" || ! -f "$model_dir/fold_0/checkpoint_final.pth" ]]; then
      echo "missing model_epoch${EPOCH_LABEL}: $variant $model_dir" >&2
      missing=1
    fi
  done
  return "$missing"
}

threshold_sweep_complete() {
  python - "$ENSEMBLE_ROOT" "$DATASET_JSON" "$THRESHOLDS" "${#variants[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset_json = Path(sys.argv[2])
thresholds = [float(value) for value in sys.argv[3].split()]
input_count = int(sys.argv[4])
summary_path = root / "threshold_sweep_summary.json"
if not (root.is_dir() and dataset_json.is_file() and summary_path.is_file()):
    raise SystemExit(1)

entries = json.loads(dataset_json.read_text()).get("test", [])
expected_cases = len(entries)
expected_findings = sum(len(entry.get("findings", {})) for entry in entries)
summary = json.loads(summary_path.read_text())
rows = {
    int(round(float(row.get("threshold", -1)) * 100)): row
    for row in summary.get("rows", [])
}
for threshold in thresholds:
    threshold_id = int(round(threshold * 100))
    label = f"thr{threshold_id:03d}"
    row = rows.get(threshold_id)
    if row is None or row.get("status") != "complete":
        raise SystemExit(1)
    eval_root = root / f"probavg{input_count}_{label}"
    eval_summary = eval_root / "reports" / "val_quick_global_eval_summary.json"
    eval_json = eval_root / "eval" / "val_quick_global_eval.json"
    pred_dir = eval_root / "predictions"
    if not (eval_summary.is_file() and eval_json.is_file() and pred_dir.is_dir()):
        raise SystemExit(1)
    data = json.loads(eval_summary.read_text())
    if int(data.get("total_cases", -1)) != expected_cases:
        raise SystemExit(1)
    if int(data.get("total_findings", -1)) != expected_findings:
        raise SystemExit(1)
    if len(list(pred_dir.glob("*.nii.gz"))) != expected_cases:
        raise SystemExit(1)
    if any(not (pred_dir / entry["name"]).is_file() for entry in entries):
        raise SystemExit(1)
PY
}

lock_age_seconds() {
  local lock_file="$1"
  python - "$lock_file" <<'PY'
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
print(max(0.0, time.time() - path.stat().st_mtime))
PY
}

write_lock() {
  local lock_file="$1"
  local phase="$2"
  python - "$lock_file" "$phase" <<'PY'
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
    "command": sys.argv,
}, indent=2, sort_keys=True) + "\n")
PY
}

acquire_ensemble_lock() {
  local lock_file="$ENSEMBLE_ROOT/.ensemble.lock"
  mkdir -p "$ENSEMBLE_ROOT"
  while true; do
    if threshold_sweep_complete; then
      echo "complete $ENSEMBLE_ROOT" >&2
      return 1
    fi
    if [[ -f "$lock_file" ]]; then
      local age
      age="$(lock_age_seconds "$lock_file")"
      if python - "$age" "$LOCK_STALE_SECONDS" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
      then
        echo "removing stale ensemble lock: $lock_file" >&2
        rm -f "$lock_file"
        continue
      fi
      echo "waiting for ensemble lock: $lock_file" >&2
      sleep "$LOCK_POLL_SECONDS"
      continue
    fi
    if ( set -o noclobber; printf 'pending\n' > "$lock_file" ) 2>/dev/null; then
      write_lock "$lock_file" "epoch100_val200_ensemble"
      echo "$lock_file"
      return 0
    fi
  done
}

release_ensemble_lock() {
  local lock_file="$1"
  rm -f "$lock_file"
}

print_status() {
  read -r expected_cases expected_findings < <(dataset_counts)
  local v1_eval_dir="$GROUP_DIR/v1_opt/full_100ep/eval_epoch${EPOCH_LABEL}_${PROBE_NAME}"
  local v1_predictions
  v1_predictions="$(prediction_count "$v1_eval_dir/predictions")"
  local v1_status="incomplete"
  if eval_complete "$v1_eval_dir"; then
    v1_status="complete"
  fi
  local model_status="missing"
  if models_ready >/dev/null 2>&1; then
    model_status="ready"
  fi
  local ensemble_status="incomplete"
  if threshold_sweep_complete; then
    ensemble_status="complete"
  fi
  echo "status time=$(date -u +%Y-%m-%dT%H:%M:%SZ) v1_val200=$v1_status predictions=${v1_predictions}/${expected_cases} findings=${expected_findings} models=$model_status ensemble=$ensemble_status"
}

run_ensemble_once() {
  local lock_file
  if ! lock_file="$(acquire_ensemble_lock)"; then
    return 0
  fi
  if EPOCH="$EPOCH" \
    PROBE_NAME="$PROBE_NAME" \
    DATASET_JSON="$DATASET_JSON" \
    ENSEMBLE_ROOT="$ENSEMBLE_ROOT" \
    THRESHOLDS="$THRESHOLDS" \
    RUN_SMOKE=0 \
    RUN_FULL=1 \
    RUN_EVAL=1 \
    bash /workspace/scripts/rexgroundingct/run_003_epoch080_probability_ensemble_val20.sh
  then
    release_ensemble_lock "$lock_file"
    return 0
  fi
  release_ensemble_lock "$lock_file"
  return 1
}

echo "Exp003 epoch ${EPOCH} ${PROBE_NAME} probability ensemble poller"
echo "group=$GROUP_DIR"
echo "dataset_json=$DATASET_JSON"
echo "ensemble_root=$ENSEMBLE_ROOT"
echo "poll_seconds=$POLL_SECONDS dry_run=$DRY_RUN"

while true; do
  print_status
  if threshold_sweep_complete; then
    echo "ensemble already complete: $ENSEMBLE_ROOT"
    exit 0
  fi

  v1_eval_dir="$GROUP_DIR/v1_opt/full_100ep/eval_epoch${EPOCH_LABEL}_${PROBE_NAME}"
  if [[ "$DRY_RUN" == "1" ]]; then
    if eval_complete "$v1_eval_dir" && models_ready; then
      echo "dry-run: would launch epoch ${EPOCH} ${PROBE_NAME} probability ensemble"
    else
      echo "dry-run: would wait for v1 ${PROBE_NAME} eval and ready model dirs"
    fi
    exit 0
  fi

  if ! eval_complete "$v1_eval_dir"; then
    sleep "$POLL_SECONDS"
    continue
  fi
  if ! models_ready; then
    sleep "$POLL_SECONDS"
    continue
  fi

  if run_ensemble_once && threshold_sweep_complete; then
    echo "ensemble complete: $ENSEMBLE_ROOT"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
