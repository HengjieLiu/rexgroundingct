#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

EXP_ID="004_voxtell_v123_native_vs_2mm_global_context_ft"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/exp004_paired_20260724T081000Z}"
ARM_DIR="$GROUP_DIR/iso2mm_global192_cont100"
CACHE_ROOT="${CACHE_ROOT:-$EXP_DIR/cache/crop_zscore_2mm_v1}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
MODEL_DIR="$ARM_DIR/model_epoch100"
CHECKPOINT="$ARM_DIR/checkpoints/checkpoint_update_010000.pth"
EVAL_DIR="$ARM_DIR/eval_epoch100_val200"
LOCK="$EVAL_DIR/.eval.lock"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-3}"

eval_complete() {
  python - "$EVAL_DIR" "$VAL200_JSON" <<'PY'
import json
import sys
from pathlib import Path

root, dataset_path = map(Path, sys.argv[1:])
entries = json.loads(dataset_path.read_text())["test"]
summary_path = root / "reports" / "val_quick_global_eval_summary.json"
eval_path = root / "eval" / "val_quick_global_eval.json"
if not (summary_path.is_file() and eval_path.is_file()):
    raise SystemExit(1)
summary = json.loads(summary_path.read_text())
if int(summary.get("total_cases", -1)) != len(entries):
    raise SystemExit(1)
if int(summary.get("total_findings", -1)) != sum(len(e["findings"]) for e in entries):
    raise SystemExit(1)
for subdir in ("predictions", "probabilities"):
    if any(not (root / subdir / entry["name"]).is_file() for entry in entries):
        raise SystemExit(1)
PY
}

if eval_complete; then
  echo "Already complete: $EVAL_DIR"
  exit 0
fi

for path in "$CHECKPOINT" "$MODEL_DIR/plans.json" "$VAL200_JSON" "$EMBEDDINGS" "$CACHE_ROOT/.complete"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
done

mkdir -p "$MODEL_DIR/fold_0" "$EVAL_DIR/logs" "$EVAL_DIR/predictions" "$EVAL_DIR/probabilities"
if [[ ! -f "$MODEL_DIR/fold_0/checkpoint_final.pth" ]]; then
  ln "$CHECKPOINT" "$MODEL_DIR/fold_0/checkpoint_final.pth"
fi

if ! (set -o noclobber; printf '{"pid":%s,"created":"%s","owner":"iso2mm_val200_early"}\n' \
  "$$" "$(date -u +%FT%TZ)" > "$LOCK") 2>/dev/null; then
  if eval_complete; then
    echo "Already complete: $EVAL_DIR"
    exit 0
  fi
  echo "Evaluation lock already held: $LOCK" >&2
  exit 1
fi
trap 'rm -f "$LOCK"' EXIT

run_shard() {
  local gpu="$1" shard="$2"
  EVAL_LOCK_HELD=1 CUDA_VISIBLE_DEVICES="$gpu" \
    python /workspace/scripts/rexgroundingct/run_voxtell_2mm_val_inference.py \
      --dataset-json "$VAL200_JSON" \
      --cache-root "$CACHE_ROOT" \
      --gpu 0 \
      --model-dir "$MODEL_DIR" \
      --embeddings "$EMBEDDINGS" \
      --output-dir "$EVAL_DIR/predictions" \
      --probability-output-dir "$EVAL_DIR/probabilities" \
      --num-shards 2 \
      --shard-index "$shard" \
      --threshold 0.5
}

echo "Starting early 2 mm val200 on GPUs $GPU_A and $GPU_B"
run_shard "$GPU_A" 0 &
pid_a=$!
run_shard "$GPU_B" 1 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$VAL200_JSON" \
  EVAL_LABEL="Experiment 004 iso2mm_global192_cont100 epoch 100 fixed val200 threshold 0.5" \
  NUM_WORKERS=8 \
  bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$EVAL_DIR" val

eval_complete
touch "$EVAL_DIR/.early_complete"
echo "Early 2 mm val200 complete: $EVAL_DIR"
