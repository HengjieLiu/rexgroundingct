#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"

EXP_ID="004_voxtell_v123_native_vs_2mm_global_context_ft"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
EXP003_DIR="${EXP003_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation}"
SOURCE_GROUP="${SOURCE_GROUP:-exp003_full_20260723T075256Z}"
SOURCE_RUN="$EXP003_DIR/runs/$SOURCE_GROUP/v123_opt_poscrop_emptyloss/full_100ep"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$SOURCE_RUN/checkpoints/checkpoint_update_010000.pth}"
SOURCE_PLANS="${SOURCE_PLANS:-$SOURCE_RUN/model_epoch100/plans.json}"
SOURCE_SCHEDULE="${SOURCE_SCHEDULE:-$EXP003_DIR/config/train_schedule_v123_opt_poscrop_emptyloss_seed20260723_100ep_100steps_gb1.jsonl}"
SEED="${SEED:-20260723}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUP="${RUN_GROUP:-exp004_paired_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
CACHE_ROOT="${CACHE_ROOT:-$EXP_DIR/cache/crop_zscore_2mm_v1}"
CACHE_WORKERS="${CACHE_WORKERS:-4}"
SCHEDULE_WORKERS="${SCHEDULE_WORKERS:-16}"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed20260723.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EXTENDED_SCHEDULE="$EXP_DIR/config/v123_seed${SEED}_extended_20000.jsonl"
NATIVE_SCHEDULE="$EXP_DIR/config/native192_cont_events10000_19999.jsonl"
ISO_SCHEDULE="$EXP_DIR/config/iso2mm_global192_cont_events10000_19999.jsonl"
PAIRED_MANIFEST="$EXP_DIR/config/paired_schedule_manifest.json"
SOURCE_MODEL_DIR="$EXP_DIR/config/source_v123_epoch100_model"

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$EXP_DIR/reports" "$GROUP_DIR"
echo "$RUN_GROUP" > "$EXP_DIR/config/latest_run_group.txt"
ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest"

for path in "$SOURCE_CHECKPOINT" "$SOURCE_PLANS" "$SOURCE_SCHEDULE" "$VAL20_JSON" "$VAL200_JSON"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
done

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment "$EXP_ID" --exp-dir "$EXP_DIR" --overwrite
python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val --json "$EXP_DIR/config/train_val_ct_readiness.json"

if [[ ! -f "$EMBEDDINGS" ]]; then
  cp "$EXP003_DIR/config/rex_text_embeddings.npz" "$EMBEDDINGS"
fi

mkdir -p "$SOURCE_MODEL_DIR/fold_0"
cp "$SOURCE_PLANS" "$SOURCE_MODEL_DIR/plans.json"
if [[ ! -f "$SOURCE_MODEL_DIR/fold_0/checkpoint_final.pth" ]]; then
  ln "$SOURCE_CHECKPOINT" "$SOURCE_MODEL_DIR/fold_0/checkpoint_final.pth"
fi
SOURCE_SHA="$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')"
python -c 'import json,sys,datetime,pathlib; pathlib.Path(sys.argv[3]).write_text(json.dumps({"source_checkpoint":sys.argv[1],"source_checkpoint_sha256":sys.argv[2],"load_policy":"network_weights_only_reset_optimizer_momentum_scheduler_amp_scaler_and_update_counter","recorded_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2,sort_keys=True)+"\n")' \
  "$SOURCE_CHECKPOINT" "$SOURCE_SHA" "$EXP_DIR/config/source_checkpoint_provenance.json"
echo "Source checkpoint sha256=$SOURCE_SHA"

if [[ ! -f "$CACHE_ROOT/.complete" ]]; then
  python /workspace/scripts/rexgroundingct/prepare_004_2mm_cache.py \
    --cache-root "$CACHE_ROOT" \
    --manifest-json "$CACHE_ROOT/manifest.json" \
    --splits train val \
    --num-workers "$CACHE_WORKERS" \
    2>&1 | tee "$EXP_DIR/logs/cache_2mm_$TIMESTAMP.log"
fi
python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["cases"]==3192, d; assert d["targets"]==8068, d; assert d["empty_targets"]==0, d; print("Cache audit accepted:",d["cases"],"cases",d["targets"],"nonempty targets")' \
  "$CACHE_ROOT/manifest.json"

if [[ ! -f "$EXTENDED_SCHEDULE" ]]; then
  python /workspace/scripts/rexgroundingct/prepare_training_schedule.py \
    --seed "$SEED" \
    --events 20000 \
    --output-jsonl "$EXTENDED_SCHEDULE" \
    --manifest-json "$EXP_DIR/config/v123_seed${SEED}_extended_20000.manifest.json" \
    --patch-size 192 192 192 \
    --require-positive-crop \
    --num-workers "$SCHEDULE_WORKERS"
fi
if [[ ! -f "$NATIVE_SCHEDULE" || ! -f "$ISO_SCHEDULE" ]]; then
  python /workspace/scripts/rexgroundingct/prepare_004_paired_schedules.py \
    --source-10k "$SOURCE_SCHEDULE" \
    --extended-20k "$EXTENDED_SCHEDULE" \
    --cache-root "$CACHE_ROOT" \
    --native-output "$NATIVE_SCHEDULE" \
    --iso2mm-output "$ISO_SCHEDULE" \
    --manifest-json "$PAIRED_MANIFEST" \
    --seed "$SEED"
fi
python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["prefix_events_verified_exact"]==10000; assert d["continuation_events"]==10000; print("Paired schedules accepted:",d["paired_case_prompt_patch_seed_sha256"])' \
  "$PAIRED_MANIFEST"

common_train=(
  --mode train
  --experiment-id "$EXP_ID"
  --config "$CONFIG_SNAPSHOT"
  --exp-dir "$EXP_DIR"
  --gpu 0
  --model-dir "$SOURCE_MODEL_DIR"
  --embeddings "$EMBEDDINGS"
  --seed "$SEED"
  --patch-size 192 192 192
  --batch-size 1
  --grad-accum 1
  --encoder-lr 1e-7
  --decoder-lr 1e-6
  --lr-schedule poly
  --poly-power 0.9
  --warmup-updates 100
  --momentum 0.99
  --weight-decay 3e-5
  --clip-grad-norm 12
  --require-positive-crop
  --loss-mode empty_bce_only
  --empty-target-loss-weight 0.5
  --voxel-bce-weighting
  --bce-foreground-weight 1.0
  --bce-boundary-weight 1.5
  --bce-background-weight 0.5
  --bce-boundary-radius 10
  --deep-supervision-weights 1,0.5,0.25,0.125,0.0625
  --allow-experimental-train
)

mkdir -p "$GROUP_DIR/smoke"
python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
  --mode sample-test --experiment-id "$EXP_ID" --config "$CONFIG_SNAPSHOT" \
  --exp-dir "$EXP_DIR" --run-dir "$GROUP_DIR/smoke/native_sample" \
  --sample-schedule "$NATIVE_SCHEDULE" --sample-test-steps 8 --require-positive-crop
python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
  --mode sample-test --experiment-id "$EXP_ID" --config "$CONFIG_SNAPSHOT" \
  --exp-dir "$EXP_DIR" --run-dir "$GROUP_DIR/smoke/iso2mm_sample" \
  --sample-schedule "$ISO_SCHEDULE" --sample-test-steps 8 --require-positive-crop \
  --preprocessed-cache-dir "$CACHE_ROOT"

(
  CUDA_VISIBLE_DEVICES=2 python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${common_train[@]}" --run-dir "$GROUP_DIR/smoke/native_train_one_update" \
    --sample-schedule "$NATIVE_SCHEDULE" --epochs 1 --steps-per-epoch 1 \
    --checkpoint-every-updates 0 --latest-checkpoint-every-updates 0 \
    --no-materialize-final-model
) >"$GROUP_DIR/smoke/native_train_one_update.log" 2>&1 &
native_smoke=$!
(
  CUDA_VISIBLE_DEVICES=3 python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${common_train[@]}" --run-dir "$GROUP_DIR/smoke/iso2mm_train_one_update" \
    --sample-schedule "$ISO_SCHEDULE" --preprocessed-cache-dir "$CACHE_ROOT" \
    --epochs 1 --steps-per-epoch 1 \
    --checkpoint-every-updates 0 --latest-checkpoint-every-updates 0 \
    --no-materialize-final-model
) >"$GROUP_DIR/smoke/iso2mm_train_one_update.log" 2>&1 &
iso_smoke=$!
wait "$native_smoke"
wait "$iso_smoke"

python -c 'import json,sys,pathlib; d=json.load(open(sys.argv[1])); d["test"]=d["test"][:1]; pathlib.Path(sys.argv[2]).write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")' \
  "$VAL20_JSON" "$GROUP_DIR/smoke/one_case.json"
CUDA_VISIBLE_DEVICES=2 python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
  --dataset-json "$GROUP_DIR/smoke/one_case.json" --split val --gpu 0 \
  --model-dir "$SOURCE_MODEL_DIR" --embeddings "$EMBEDDINGS" \
  --output-dir "$GROUP_DIR/smoke/native_prediction"
CUDA_VISIBLE_DEVICES=3 python /workspace/scripts/rexgroundingct/run_voxtell_2mm_val_inference.py \
  --dataset-json "$GROUP_DIR/smoke/one_case.json" --cache-root "$CACHE_ROOT" --gpu 0 \
  --model-dir "$SOURCE_MODEL_DIR" --embeddings "$EMBEDDINGS" \
  --output-dir "$GROUP_DIR/smoke/iso2mm_prediction" \
  --probability-output-dir "$GROUP_DIR/smoke/iso2mm_probability"

mkdir -p "$GROUP_DIR/native192_cont100/logs" "$GROUP_DIR/iso2mm_global192_cont100/logs"
START_BARRIER="$GROUP_DIR/.paired_training_start"
date -u +%FT%TZ > "$START_BARRIER"
(
  CUDA_VISIBLE_DEVICES=2 python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${common_train[@]}" \
    --run-dir "$GROUP_DIR/native192_cont100" \
    --sample-schedule "$NATIVE_SCHEDULE" \
    --epochs 100 --steps-per-epoch 100 \
    --checkpoint-every-updates 2000 \
    --latest-checkpoint-every-updates 500
) >"$GROUP_DIR/native192_cont100/logs/training.log" 2>&1 &
native_pid=$!
(
  CUDA_VISIBLE_DEVICES=3 python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    "${common_train[@]}" \
    --run-dir "$GROUP_DIR/iso2mm_global192_cont100" \
    --sample-schedule "$ISO_SCHEDULE" \
    --preprocessed-cache-dir "$CACHE_ROOT" \
    --epochs 100 --steps-per-epoch 100 \
    --checkpoint-every-updates 2000 \
    --latest-checkpoint-every-updates 500
) >"$GROUP_DIR/iso2mm_global192_cont100/logs/training.log" 2>&1 &
iso_pid=$!
printf '%s\n' "$native_pid" > "$GROUP_DIR/native192_cont100/training.pid"
printf '%s\n' "$iso_pid" > "$GROUP_DIR/iso2mm_global192_cont100/training.pid"

EXP_DIR="$EXP_DIR" GROUP_DIR="$GROUP_DIR" CACHE_ROOT="$CACHE_ROOT" \
  SOURCE_MODEL_DIR="$SOURCE_MODEL_DIR" EMBEDDINGS="$EMBEDDINGS" \
  VAL20_JSON="$VAL20_JSON" VAL200_JSON="$VAL200_JSON" \
  bash /workspace/scripts/rexgroundingct/run_004_eval_coordinator.sh \
  >"$GROUP_DIR/eval_coordinator.log" 2>&1 &
coordinator_pid=$!
printf '%s\n' "$coordinator_pid" > "$GROUP_DIR/eval_coordinator.pid"

status=0
if wait "$native_pid"; then
  touch "$GROUP_DIR/native192_cont100/.train_complete"
else
  touch "$GROUP_DIR/native192_cont100/.train_failed"
  status=1
fi
if wait "$iso_pid"; then
  touch "$GROUP_DIR/iso2mm_global192_cont100/.train_complete"
else
  touch "$GROUP_DIR/iso2mm_global192_cont100/.train_failed"
  status=1
fi
if [[ "$status" != "0" ]]; then
  kill "$coordinator_pid" 2>/dev/null || true
  wait "$coordinator_pid" 2>/dev/null || true
  exit "$status"
fi
wait "$coordinator_pid"
echo "Experiment 004 complete: $GROUP_DIR"
