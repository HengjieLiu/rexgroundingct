#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

EXP_ID="005_voxtell_global_proposal_local_cascade"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
EXP004_DIR="${EXP004_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft}"
EXP003_DIR="${EXP003_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation}"
SOURCE_2MM_CACHE="${SOURCE_2MM_CACHE:-$EXP004_DIR/cache/crop_zscore_2mm_v1}"
SOURCE_SCHEDULE="${SOURCE_SCHEDULE:-$EXP004_DIR/config/native192_cont_events10000_19999.jsonl}"
GLOBAL_CACHE="${GLOBAL_CACHE:-$EXP_DIR/cache/full_fov_4mm_192_v1}"
STAGE2_MODEL="${STAGE2_MODEL:-$EXP003_DIR/runs/exp003_full_20260723T075256Z/v123_opt_poscrop_emptyloss/full_100ep/model_epoch100}"
EMBEDDINGS="${EMBEDDINGS:-$EXP003_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed20260723.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUP="${RUN_GROUP:-exp005_global_local_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
CACHE_WORKERS="${CACHE_WORKERS:-4}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$EXP_DIR/reports" "$EXP_DIR/control" "$GROUP_DIR"
ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest"
echo "$RUN_GROUP" > "$EXP_DIR/config/latest_run_group.txt"
python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment "$EXP_ID" --exp-dir "$EXP_DIR" --overwrite
python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val --json "$EXP_DIR/config/train_val_ct_readiness.json"

echo "Waiting for audited exp004 2 mm cache and paired schedule."
until [[ -f "$SOURCE_2MM_CACHE/.complete" && -f "$SOURCE_SCHEDULE" ]]; do
  sleep "$WAIT_SECONDS"
done

if [[ ! -f "$GLOBAL_CACHE/.complete" ]]; then
  python /workspace/scripts/rexgroundingct/prepare_005_global_proposal_cache.py \
    --source-2mm-cache "$SOURCE_2MM_CACHE" \
    --output-cache "$GLOBAL_CACHE" \
    --manifest-json "$GLOBAL_CACHE/manifest.json" \
    --splits train val --num-workers "$CACHE_WORKERS" \
    2>&1 | tee "$EXP_DIR/logs/global_cache_$TIMESTAMP.log"
fi
python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["cases"]==3192,d; assert d["targets"]==8068,d; assert max(d["shape_4mm_max_zyx"])<=192,d; print("Global cache accepted",d["shape_4mm_max_zyx"])' \
  "$GLOBAL_CACHE/manifest.json"

common=(
  --cache-root "$GLOBAL_CACHE"
  --schedule "$SOURCE_SCHEDULE"
  --embeddings "$EMBEDDINGS"
  --val-json "$VAL20_JSON"
  --seed 20260723
  --steps 5000
  --steps-per-epoch 100
  --base-channels 8
  --lr 3e-4
  --weight-decay 1e-4
  --warmup-steps 100
  --background-weight 0.05
  --empty-weight 0.25
  --tversky-beta 0.8
  --hit-loss-weight 0.25
  --eval-every 500
  --checkpoint-every 500
  --top-k 5
  --nms-radius 8
)

mkdir -p "$GROUP_DIR/smoke_strict" "$GROUP_DIR/smoke_margin"
CUDA_VISIBLE_DEVICES=0 python /workspace/scripts/rexgroundingct/train_global_proposal.py \
  "${common[@]}" --run-dir "$GROUP_DIR/smoke_strict" --gpu 0 \
  --target-margin-voxels 0 --smoke-steps 2 \
  >"$GROUP_DIR/smoke_strict.log" 2>&1
CUDA_VISIBLE_DEVICES=1 python /workspace/scripts/rexgroundingct/train_global_proposal.py \
  "${common[@]}" --run-dir "$GROUP_DIR/smoke_margin" --gpu 0 \
  --target-margin-voxels 4 --smoke-steps 2 \
  >"$GROUP_DIR/smoke_margin.log" 2>&1

(
  while true; do
    if [[ -f "$EXP_DIR/control/STOP_ALL" ]]; then
      touch "$EXP_DIR/control/STOP_strict_inclusion"
      touch "$EXP_DIR/control/STOP_inclusion_margin16mm"
      exit 0
    fi
    sleep 30
  done
) &
stop_watcher=$!

mkdir -p "$GROUP_DIR/strict_inclusion" "$GROUP_DIR/inclusion_margin16mm"
(
  CUDA_VISIBLE_DEVICES=0 python /workspace/scripts/rexgroundingct/train_global_proposal.py \
    "${common[@]}" --run-dir "$GROUP_DIR/strict_inclusion" --gpu 0 \
    --target-margin-voxels 0 \
    --stop-file "$EXP_DIR/control/STOP_strict_inclusion"
) >"$GROUP_DIR/strict_inclusion/training.log" 2>&1 &
pid0=$!
(
  CUDA_VISIBLE_DEVICES=1 python /workspace/scripts/rexgroundingct/train_global_proposal.py \
    "${common[@]}" --run-dir "$GROUP_DIR/inclusion_margin16mm" --gpu 0 \
    --target-margin-voxels 4 \
    --stop-file "$EXP_DIR/control/STOP_inclusion_margin16mm"
) >"$GROUP_DIR/inclusion_margin16mm/training.log" 2>&1 &
pid1=$!

EXP_DIR="$EXP_DIR" GROUP_DIR="$GROUP_DIR" GLOBAL_CACHE="$GLOBAL_CACHE" \
  STAGE2_MODEL="$STAGE2_MODEL" EMBEDDINGS="$EMBEDDINGS" \
  VAL20_JSON="$VAL20_JSON" VAL200_JSON="$VAL200_JSON" \
  bash /workspace/scripts/rexgroundingct/run_005_cascade_poller.sh \
  >"$GROUP_DIR/cascade_poller.log" 2>&1 &
poller=$!

status=0
if wait "$pid0"; then touch "$GROUP_DIR/strict_inclusion/.train_complete"; else status=1; fi
if wait "$pid1"; then touch "$GROUP_DIR/inclusion_margin16mm/.train_complete"; else status=1; fi
kill "$stop_watcher" 2>/dev/null || true
wait "$stop_watcher" 2>/dev/null || true
if [[ "$status" != 0 ]]; then
  kill "$poller" 2>/dev/null || true
  wait "$poller" 2>/dev/null || true
  exit "$status"
fi
wait "$poller"
if python - "$GROUP_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for variant in ("strict_inclusion", "inclusion_margin16mm"):
    path = root / variant / "reports" / "training_summary.json"
    if not path.is_file() or json.loads(path.read_text()).get("completed_step") != 5000:
        raise SystemExit(1)
PY
then
  touch "$GROUP_DIR/.experiment_complete"
else
  touch "$GROUP_DIR/.experiment_stopped"
fi
