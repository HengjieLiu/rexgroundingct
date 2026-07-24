#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

EXP_ID="004_voxtell_v123_native_vs_2mm_global_context_ft"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/exp004_paired_20260724T081000Z}"
ARM_DIR="$GROUP_DIR/native192_cont100"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$EXP_DIR/config/source_v123_epoch100_model}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
BASELINE_EVAL="${BASELINE_EVAL:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v123_opt_poscrop_emptyloss/full_100ep/eval_epoch100_val200/eval/val_quick_global_eval.json}"
ISO_EVAL="$GROUP_DIR/iso2mm_global192_cont100/eval_epoch100_val200/eval/val_quick_global_eval.json"
MODEL_DIR="$ARM_DIR/model_epoch100"
CHECKPOINT="$ARM_DIR/checkpoints/checkpoint_update_010000.pth"
TRAIN_COMPLETE="$ARM_DIR/.train_complete"
EVAL_DIR="$ARM_DIR/eval_epoch100_val200"
LOCK="$EVAL_DIR/.eval.lock"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
POLL_SECONDS="${POLL_SECONDS:-60}"

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

echo "Waiting for native epoch-100 training completion."
until [[ -f "$TRAIN_COMPLETE" && -s "$CHECKPOINT" ]]; do
  sleep "$POLL_SECONDS"
done

first="$(stat -c '%s:%Y' "$CHECKPOINT")"
sleep 5
second="$(stat -c '%s:%Y' "$CHECKPOINT")"
[[ "$first" == "$second" ]] || { echo "Native checkpoint is not stable" >&2; exit 1; }

if eval_complete; then
  echo "Already complete: $EVAL_DIR"
else
  mkdir -p "$MODEL_DIR/fold_0" "$EVAL_DIR/logs" "$EVAL_DIR/predictions" "$EVAL_DIR/probabilities"
  cp "$SOURCE_MODEL_DIR/plans.json" "$MODEL_DIR/plans.json"
  if [[ ! -f "$MODEL_DIR/fold_0/checkpoint_final.pth" ]]; then
    ln "$CHECKPOINT" "$MODEL_DIR/fold_0/checkpoint_final.pth"
  fi
  if ! (set -o noclobber; printf '{"pid":%s,"created":"%s","owner":"native_val200_after_train"}\n' \
    "$$" "$(date -u +%FT%TZ)" > "$LOCK") 2>/dev/null; then
    if eval_complete; then
      echo "Already complete: $EVAL_DIR"
    else
      echo "Evaluation lock already held: $LOCK" >&2
      exit 1
    fi
  else
    trap 'rm -f "$LOCK"' EXIT
    pids=()
    for spec in "$GPU_A:0" "$GPU_B:1"; do
      gpu="${spec%:*}"
      shard="${spec#*:}"
      EVAL_LOCK_HELD=1 CUDA_VISIBLE_DEVICES="$gpu" \
        python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
          --split val \
          --output-type mask \
          --dataset-json "$VAL200_JSON" \
          --gpu 0 \
          --model-dir "$MODEL_DIR" \
          --embeddings "$EMBEDDINGS" \
          --output-dir "$EVAL_DIR/predictions" \
          --probability-output-dir "$EVAL_DIR/probabilities" \
          --num-shards 2 \
          --shard-index "$shard" \
          --threshold 0.5 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$VAL200_JSON" \
      EVAL_LABEL="Experiment 004 native192_cont100 epoch 100 fixed val200 threshold 0.5" \
      NUM_WORKERS=8 \
      bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$EVAL_DIR" val
    eval_complete
    touch "$EVAL_DIR/.sidecar_complete"
    rm -f "$LOCK"
    trap - EXIT
  fi
fi

for path in "$BASELINE_EVAL" "$ISO_EVAL" "$EVAL_DIR/eval/val_quick_global_eval.json"; do
  [[ -f "$path" ]] || { echo "Missing comparison input: $path" >&2; exit 1; }
done
python /workspace/scripts/rexgroundingct/summarize_004_results.py \
  --dataset-json "$VAL200_JSON" \
  --input "exp003_v123_epoch100=$BASELINE_EVAL" \
  --input "native192_cont100=$EVAL_DIR/eval/val_quick_global_eval.json" \
  --input "iso2mm_global192_cont100=$ISO_EVAL" \
  --output-json "$EXP_DIR/reports/paired_comparison.json" \
  --output-md "$EXP_DIR/reports/paired_comparison_report.md"
touch "$GROUP_DIR/.sidecar_final_complete"
echo "Native val200 and paired comparison complete."
