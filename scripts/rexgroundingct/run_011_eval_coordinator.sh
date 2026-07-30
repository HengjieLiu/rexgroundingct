#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

EXP_DIR="${EXP_DIR:?EXP_DIR is required}"
GROUP_DIR="${GROUP_DIR:?GROUP_DIR is required}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:?SOURCE_MODEL_DIR is required}"
EMBEDDINGS="${EMBEDDINGS:?EMBEDDINGS is required}"
VAL20_JSON="${VAL20_JSON:?VAL20_JSON is required}"
VAL200_JSON="${VAL200_JSON:?VAL200_JSON is required}"
CACHE_BASE="${CACHE_BASE:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABILITY_SECONDS="${STABILITY_SECONDS:-10}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-43200}"

arms=(
  v123_e4d4_zscore
  v123_e4d4_clip1024_zscore
  v123_e4d4_clip1024_linear
  v123_e5d4_zscore
  v123_e5d4_clip1024_zscore
  v123_e5d4_clip1024_linear
)
val20_epochs=(5 20 40 60 80 100)

arm_cache() {
  case "$1" in
    v123_e4d4_zscore|v123_e5d4_zscore)
      echo "$CACHE_BASE/crop_zscore_native_v1"
      ;;
    v123_e4d4_clip1024_zscore|v123_e5d4_clip1024_zscore)
      echo "$CACHE_BASE/crop_clip1024_zscore_native_v1"
      ;;
    v123_e4d4_clip1024_linear|v123_e5d4_clip1024_linear)
      echo "$CACHE_BASE/crop_clip1024_linear_native_v1"
      ;;
    *) return 1 ;;
  esac
}

arm_final_gpu() {
  case "$1" in
    v123_e4d4_zscore|v123_e5d4_zscore) echo 0 ;;
    v123_e4d4_clip1024_zscore|v123_e5d4_clip1024_zscore) echo 1 ;;
    v123_e4d4_clip1024_linear|v123_e5d4_clip1024_linear) echo 2 ;;
    *) return 1 ;;
  esac
}

epoch_update() {
  echo "$(($1 * 100))"
}

eval_complete() {
  local eval_dir="$1" dataset_json="$2"
  python - "$eval_dir" "$dataset_json" <<'PY'
import json
import sys
from pathlib import Path

root, dataset = map(Path, sys.argv[1:3])
summary = root / "reports" / "val_quick_global_eval_summary.json"
eval_json = root / "eval" / "val_quick_global_eval.json"
predictions = root / "predictions"
if not (summary.is_file() and eval_json.is_file() and predictions.is_dir()):
    raise SystemExit(1)
entries = json.loads(dataset.read_text())["test"]
data = json.loads(summary.read_text())
if int(data.get("total_cases", -1)) != len(entries):
    raise SystemExit(1)
if int(data.get("total_findings", -1)) != sum(len(e["findings"]) for e in entries):
    raise SystemExit(1)
if any(not (predictions / entry["name"]).is_file() for entry in entries):
    raise SystemExit(1)
PY
}

checkpoint_stable() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  local before after
  before="$(stat -c '%s:%Y' "$path")"
  sleep "$STABILITY_SECONDS"
  [[ -s "$path" ]] || return 1
  after="$(stat -c '%s:%Y' "$path")"
  [[ "$before" == "$after" ]]
}

lock_age_seconds() {
  local lock="$1"
  python - "$lock" <<'PY'
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
print(max(0.0, time.time() - path.stat().st_mtime))
PY
}

acquire_lock() {
  local eval_dir="$1"
  local lock="$eval_dir/.eval.lock"
  mkdir -p "$eval_dir"
  while true; do
    if (
      set -o noclobber
      printf '{"pid":%s,"created":"%s","phase":"exp011_coordinator"}\n' \
        "$$" "$(date -u +%FT%TZ)" > "$lock"
    ) 2>/dev/null; then
      echo "$lock"
      return 0
    fi
    local age
    age="$(lock_age_seconds "$lock" 2>/dev/null || echo 0)"
    if python - "$age" "$LOCK_STALE_SECONDS" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
    then
      rm -f "$lock"
      continue
    fi
    return 1
  done
}

materialize_model() {
  local checkpoint="$1" model_dir="$2"
  mkdir -p "$model_dir/fold_0"
  cp "$SOURCE_MODEL_DIR/plans.json" "$model_dir/plans.json"
  local target="$model_dir/fold_0/checkpoint_final.pth"
  if [[ ! -f "$target" || "$(stat -c %s "$target")" != "$(stat -c %s "$checkpoint")" ]]; then
    local tmp="$target.tmp.$$"
    rm -f "$tmp"
    ln "$checkpoint" "$tmp" 2>/dev/null || cp "$checkpoint" "$tmp"
    mv -f "$tmp" "$target"
  fi
}

run_eval() {
  local arm="$1" epoch="$2" probe="$3" gpu="$4"
  local dataset="$VAL20_JSON"
  [[ "$probe" == "val200" ]] && dataset="$VAL200_JSON"
  local run_dir="$GROUP_DIR/$arm"
  local eval_dir="$run_dir/eval_epoch$(printf '%03d' "$epoch")_$probe"
  local checkpoint="$run_dir/checkpoints/checkpoint_update_$(printf '%06d' "$(epoch_update "$epoch")").pth"

  if eval_complete "$eval_dir" "$dataset"; then
    echo "Already complete: $arm epoch=$epoch $probe"
    return 0
  fi
  checkpoint_stable "$checkpoint" || return 2

  local lock
  lock="$(acquire_lock "$eval_dir")" || return 3
  local model_dir="$run_dir/model_epoch$(printf '%03d' "$epoch")"
  materialize_model "$checkpoint" "$model_dir"
  mkdir -p "$eval_dir/logs" "$eval_dir/predictions"
  local log="$eval_dir/logs/coordinator_$(date -u +%Y%m%dT%H%M%SZ).log"
  if (
    set -euo pipefail
    echo "start arm=$arm epoch=$epoch probe=$probe gpu=$gpu"
    EVAL_LOCK_HELD=1 CUDA_VISIBLE_DEVICES="$gpu" python \
      /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
      --split val \
      --dataset-json "$dataset" \
      --gpu 0 \
      --model-dir "$model_dir" \
      --embeddings "$EMBEDDINGS" \
      --output-dir "$eval_dir/predictions" \
      --output-type mask \
      --threshold 0.5 \
      --preprocessed-cache-dir "$(arm_cache "$arm")"
    EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset" \
      EVAL_LABEL="Experiment 011 $arm epoch $epoch fixed $probe threshold 0.5" \
      NUM_WORKERS=8 \
      bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$eval_dir" val
    echo "done arm=$arm epoch=$epoch probe=$probe"
  ) >"$log" 2>&1; then
    rm -f "$lock"
    return 0
  fi
  rm -f "$lock"
  echo "Evaluation failed: arm=$arm epoch=$epoch probe=$probe log=$log" >&2
  return 4
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for epoch in "${val20_epochs[@]}"; do
    for arm in "${arms[@]}"; do
      checkpoint="$GROUP_DIR/$arm/checkpoints/checkpoint_update_$(printf '%06d' "$(epoch_update "$epoch")").pth"
      gpu="$(arm_final_gpu "$arm")"
      echo "$arm epoch=$epoch val20 checkpoint=$checkpoint cache=$(arm_cache "$arm") gpu=$gpu timing=synchronous_milestone_barrier"
    done
  done
  for arm in "${arms[@]}"; do
    echo "$arm epoch=100 val200 gpu=$(arm_final_gpu "$arm") timing=after_epoch100_val20_barrier"
  done
  exit 0
fi

if [[ -n "${EVAL_ARM:-}" ]]; then
  EVAL_EPOCH="${EVAL_EPOCH:?EVAL_EPOCH is required with EVAL_ARM}"
  EVAL_PROBE="${EVAL_PROBE:-val20}"
  EVAL_GPU="${EVAL_GPU:-$(arm_final_gpu "$EVAL_ARM")}"
  case "$EVAL_ARM" in
    v123_e4d4_zscore|v123_e4d4_clip1024_zscore|v123_e4d4_clip1024_linear|\
v123_e5d4_zscore|v123_e5d4_clip1024_zscore|v123_e5d4_clip1024_linear) ;;
    *) echo "Unknown EVAL_ARM=$EVAL_ARM" >&2; exit 2 ;;
  esac
  case "$EVAL_PROBE" in
    val20|val200) ;;
    *) echo "EVAL_PROBE must be val20 or val200" >&2; exit 2 ;;
  esac
  if [[ "$EVAL_GPU" != "$(arm_final_gpu "$EVAL_ARM")" ]]; then
    echo "EVAL_GPU=$EVAL_GPU does not match $EVAL_ARM assignment" >&2
    exit 2
  fi
  while true; do
    if run_eval "$EVAL_ARM" "$EVAL_EPOCH" "$EVAL_PROBE" "$EVAL_GPU"; then
      exit 0
    fi
    sleep "$POLL_SECONDS"
  done
fi

echo "EVAL_ARM is required; exp011 evaluations are launched only by milestone barriers." >&2
exit 2
