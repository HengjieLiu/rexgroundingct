#!/usr/bin/env bash
set -euo pipefail

EXP_ID="003_voxtell_rex_ft_rescue_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
RUN_GROUP="${RUN_GROUP:-exp003_full_20260723T075256Z}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
SEED="${SEED:-20260723}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
POLL_SECONDS="${POLL_SECONDS:-600}"
CHECKPOINT_STABILITY_SECONDS="${CHECKPOINT_STABILITY_SECONDS:-30}"
LOCK_POLL_SECONDS="${EVAL_LOCK_POLL_SECONDS:-60}"
LOCK_STALE_SECONDS="${EVAL_LOCK_STALE_SECONDS:-43200}"
DRY_RUN="${DRY_RUN:-0}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"

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

target_dataset() {
  local epoch="$1"
  local probe="$2"
  case "$probe" in
    val20) echo "$VAL20_JSON" ;;
    val200) echo "$VAL200_JSON" ;;
    *) echo "Unknown probe: $probe" >&2; return 1 ;;
  esac
}

expected_cases() {
  local dataset_json="$1"
  python - "$dataset_json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print(len(data.get("test", [])))
PY
}

eval_complete() {
  local eval_dir="$1"
  local dataset_json="$2"
  python - "$eval_dir" "$dataset_json" <<'PY'
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

checkpoint_path() {
  local full_dir="$1"
  local epoch="$2"
  local update=$((epoch * STEPS_PER_EPOCH))
  printf '%s/checkpoints/checkpoint_update_%06d.pth' "$full_dir" "$update"
}

checkpoint_stable() {
  local checkpoint="$1"
  if [[ ! -f "$checkpoint" ]]; then
    return 1
  fi
  local first_size first_mtime second_size second_mtime
  first_size="$(stat -c '%s' "$checkpoint")"
  first_mtime="$(stat -c '%Y' "$checkpoint")"
  if [[ "$first_size" == "0" ]]; then
    return 1
  fi
  sleep "$CHECKPOINT_STABILITY_SECONDS"
  if [[ ! -f "$checkpoint" ]]; then
    return 1
  fi
  second_size="$(stat -c '%s' "$checkpoint")"
  second_mtime="$(stat -c '%Y' "$checkpoint")"
  [[ "$first_size" == "$second_size" && "$first_mtime" == "$second_mtime" ]]
}

lock_phase() {
  local lock_file="$1"
  python - "$lock_file" <<'PY'
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

acquire_eval_lock() {
  local eval_dir="$1"
  local dataset_json="$2"
  local lock_file="$eval_dir/.eval.lock"
  mkdir -p "$eval_dir"
  while true; do
    if eval_complete "$eval_dir" "$dataset_json"; then
      echo "complete $eval_dir"
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
        echo "removing stale lock $lock_file"
        rm -f "$lock_file"
        continue
      fi
      echo "waiting for lock $lock_file phase=$(lock_phase "$lock_file")"
      sleep "$LOCK_POLL_SECONDS"
      continue
    fi
    if ( set -o noclobber; printf 'pending\n' > "$lock_file" ) 2>/dev/null; then
      write_lock "$lock_file" "sidecar"
      echo "$lock_file"
      return 0
    fi
  done
}

release_eval_lock() {
  local lock_file="$1"
  rm -f "$lock_file"
}

materialize_checkpoint_model() {
  local source_plans="$1"
  local checkpoint="$2"
  local model_dir="$3"
  mkdir -p "$model_dir/fold_0"
  cp "$source_plans" "$model_dir/plans.json"
  local target="$model_dir/fold_0/checkpoint_final.pth"
  if [[ ! -f "$target" || "$(stat -c '%s' "$target")" != "$(stat -c '%s' "$checkpoint")" ]]; then
    local tmp="$target.tmp.$$"
    cp "$checkpoint" "$tmp"
    mv "$tmp" "$target"
  fi
}

run_eval_target() {
  local variant="$1"
  local gpu="$2"
  local epoch="$3"
  local probe="$4"
  local full_dir="$GROUP_DIR/$variant/full_100ep"
  local checkpoint
  checkpoint="$(checkpoint_path "$full_dir" "$epoch")"
  local eval_dir="$full_dir/eval_epoch$(printf '%03d' "$epoch")_$probe"
  local dataset_json
  dataset_json="$(target_dataset "$epoch" "$probe")"
  local source_plans="$GROUP_DIR/$variant/short_005ep/model/plans.json"
  local model_dir="$full_dir/model_epoch$(printf '%03d' "$epoch")"

  if eval_complete "$eval_dir" "$dataset_json"; then
    echo "skip complete $variant epoch=$epoch probe=$probe"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ -f "$checkpoint" ]]; then
      echo "dry-run would eval variant=$variant epoch=$epoch probe=$probe gpu=$gpu checkpoint=$checkpoint"
    else
      echo "dry-run would wait variant=$variant epoch=$epoch probe=$probe checkpoint=$checkpoint"
    fi
    return 0
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "wait checkpoint $variant epoch=$epoch checkpoint=$checkpoint"
    return 1
  fi
  if ! checkpoint_stable "$checkpoint"; then
    echo "wait stable checkpoint $variant epoch=$epoch checkpoint=$checkpoint"
    return 1
  fi
  if [[ ! -f "$source_plans" ]]; then
    echo "missing plans $source_plans" >&2
    return 1
  fi

  local lock_file
  if ! lock_file="$(acquire_eval_lock "$eval_dir" "$dataset_json")"; then
    return 0
  fi

  mkdir -p "$eval_dir/logs" "$eval_dir/predictions"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local log="$eval_dir/logs/intermediate_poller_eval_${stamp}.log"
  local label="Experiment 003 ${variant} full epoch ${epoch} fixed ${probe} seed ${SEED} quick/global eval"

  if (
    set -euo pipefail
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start variant=$variant epoch=$epoch probe=$probe gpu=$gpu"
    materialize_checkpoint_model "$source_plans" "$checkpoint" "$model_dir"
    EVAL_LOCK_HELD=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
      --split val \
      --dataset-json "$dataset_json" \
      --gpu 0 \
      --model-dir "$model_dir" \
      --embeddings "$EMBEDDINGS" \
      --output-dir "$eval_dir/predictions" \
      --overwrite
    EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset_json" EVAL_LABEL="$label" NUM_WORKERS=8 \
      bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$eval_dir" val
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done variant=$variant epoch=$epoch probe=$probe"
  ) >"$log" 2>&1; then
    release_eval_lock "$lock_file"
    return 0
  fi

  release_eval_lock "$lock_file"
  echo "eval failed variant=$variant epoch=$epoch probe=$probe log=$log" >&2
  return 1
}

variant_worker() {
  local variant="$1"
  local gpu
  gpu="$(variant_gpu "$variant")"
  echo "worker start variant=$variant gpu=$gpu"
  local targets=("60:val20" "80:val20" "100:val20" "100:val200")
  while true; do
    local all_done=1
    for target in "${targets[@]}"; do
      local epoch="${target%%:*}"
      local probe="${target##*:}"
      local dataset_json
      dataset_json="$(target_dataset "$epoch" "$probe")"
      local eval_dir="$GROUP_DIR/$variant/full_100ep/eval_epoch$(printf '%03d' "$epoch")_$probe"
      if eval_complete "$eval_dir" "$dataset_json"; then
        continue
      fi
      all_done=0
      run_eval_target "$variant" "$gpu" "$epoch" "$probe" || true
      break
    done
    if [[ "$all_done" == "1" || "$DRY_RUN" == "1" ]]; then
      echo "worker done variant=$variant"
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

echo "Exp003 intermediate eval poller"
echo "group=$GROUP_DIR poll_seconds=$POLL_SECONDS dry_run=$DRY_RUN"
for variant in "${variants[@]}"; do
  variant_worker "$variant" &
done

status=0
for pid in $(jobs -p); do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
