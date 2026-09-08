#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH="/workspace/scripts/rexgroundingct:/workspace/external/VoxTell:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

EXP_ID="026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
EXP007_DIR="${EXP007_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1}"
SEED="${SEED:-20260723}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EPOCHS="${EPOCHS:-100}"
WORLD_SIZE="${WORLD_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUP="${RUN_GROUP:-exp026_fixed_lr_ddp_bs4_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
RUN_DIR="$GROUP_DIR/ddp_bs4"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
SOURCE_SCHEDULE="${SOURCE_SCHEDULE:-$EXP007_DIR/config/train_schedule_v123_ddp_bs4_seed${SEED}_100ep_100steps_gb4.jsonl}"
SCHEDULE="$EXP_DIR/config/train_schedule_v123_ddp_bs4_fixed_lr_seed${SEED}_100ep_${STEPS_PER_EPOCH}steps_gb4.jsonl"
SCHEDULE_MANIFEST="${SCHEDULE%.jsonl}.manifest.json"
SOURCE_MODEL_DIR="$EXP_DIR/config/public_voxtell_v1_1_model"
REPORT_JSON="$EXP_DIR/reports/fixed_lr_summary.json"
REPORT_MD="$EXP_DIR/reports/fixed_lr_report.md"
VAL200_EPOCHS="${VAL200_EPOCHS:-$(seq 5 5 100)}"
CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-$(seq -s, 500 500 10000)}"
RUN_SAMPLE_TEST="${RUN_SAMPLE_TEST:-1}"
RUN_DDP_SMOKE="${RUN_DDP_SMOKE:-1}"
RUN_INFERENCE_SMOKE="${RUN_INFERENCE_SMOKE:-1}"
RUN_STATIC_ONLY="${RUN_STATIC_ONLY:-0}"
RUN_SMOKE_ONLY="${RUN_SMOKE_ONLY:-0}"
RUN_FULL="${RUN_FULL:-1}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-43200}"
STABILITY_SECONDS="${STABILITY_SECONDS:-10}"
MASTER_PORT="${MASTER_PORT:-29626}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"

read -r -a GPUS <<< "$GPU_LIST"
[[ "${#GPUS[@]}" -eq "$WORLD_SIZE" ]] || {
  echo "GPU_LIST has ${#GPUS[@]} entries but WORLD_SIZE=$WORLD_SIZE: $GPU_LIST" >&2
  exit 1
}
DDP_CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$EXP_DIR/reports" "$GROUP_DIR" "$RUN_DIR/logs"
echo "$RUN_GROUP" > "$EXP_DIR/config/latest_run_group.txt"
ln -sfn "$GROUP_DIR" "$EXP_DIR/runs/latest"

sha256_path() { sha256sum "$1" | cut -d ' ' -f 1; }

require_sha() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "$path" ]] || { echo "Missing $label: $path" >&2; exit 1; }
  actual="$(sha256_path "$path")"
  [[ "$actual" == "$expected" ]] || {
    echo "$label SHA mismatch: expected=$expected actual=$actual path=$path" >&2
    exit 1
  }
  echo "$label sha256=$actual"
}

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment "$EXP_ID" --exp-dir "$EXP_DIR" --overwrite
python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val --json "$EXP_DIR/config/train_val_ct_readiness.json"

[[ -f "$CACHE_ROOT/.complete" ]] || { echo "Missing cache completion marker" >&2; exit 1; }
require_sha "$CACHE_ROOT/manifest.json" \
  "fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3" \
  "native cache manifest"
require_sha "$SOURCE_SCHEDULE" \
  "acdfed8dd8d9908e9ea33d6790d47f8f7e3cbeed4bad326274d115d38e96a0aa" \
  "exp007 DDP schedule"
require_sha "$VAL200_JSON" \
  "7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897" \
  "val200 JSON"

if [[ ! -f "$SCHEDULE" || "$(sha256_path "$SCHEDULE")" != "$(sha256_path "$SOURCE_SCHEDULE")" ]]; then
  cp "$SOURCE_SCHEDULE" "$SCHEDULE"
fi
if [[ -f "${SOURCE_SCHEDULE%.jsonl}.manifest.json" ]]; then
  cp "${SOURCE_SCHEDULE%.jsonl}.manifest.json" "$SCHEDULE_MANIFEST"
else
  printf '{"source":"%s","sha256":"%s","events":40000}\n' \
    "$SOURCE_SCHEDULE" "$(sha256_path "$SCHEDULE")" > "$SCHEDULE_MANIFEST"
fi
require_sha "$SCHEDULE" \
  "acdfed8dd8d9908e9ea33d6790d47f8f7e3cbeed4bad326274d115d38e96a0aa" \
  "Exp026 DDP schedule"

if [[ ! -f "$EMBEDDINGS" ]]; then
  for candidate in \
    "$EXP007_DIR/config/rex_text_embeddings.npz" \
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/config/rex_text_embeddings.npz"; do
    if [[ -f "$candidate" ]]; then cp "$candidate" "$EMBEDDINGS"; break; fi
  done
fi
if [[ ! -f "$EMBEDDINGS" ]]; then
  python /workspace/scripts/rexgroundingct/precompute_text_embeddings.py \
    --splits train val --gpu 0 --output "$EMBEDDINGS" \
    2>&1 | tee "$EXP_DIR/logs/precompute_text_embeddings_$TIMESTAMP.log"
fi

BASE_MODEL_DIR="$(python -c 'from train_text_conditioned_voxtell import resolve_model_dir; print(resolve_model_dir(None))')"
[[ -f "$BASE_MODEL_DIR/plans.json" ]] || { echo "Missing VoxTell plans" >&2; exit 1; }
[[ -f "$BASE_MODEL_DIR/fold_0/checkpoint_final.pth" ]] || { echo "Missing public VoxTell checkpoint" >&2; exit 1; }
mkdir -p "$SOURCE_MODEL_DIR"
cp "$BASE_MODEL_DIR/plans.json" "$SOURCE_MODEL_DIR/plans.json"
python -c 'import datetime as d, hashlib, json, pathlib, sys; b=pathlib.Path(sys.argv[1]); o=pathlib.Path(sys.argv[2]); c=b/"fold_0"/"checkpoint_final.pth"; h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(); o.write_text(json.dumps({"recorded_at_utc":d.datetime.now(d.timezone.utc).isoformat(),"source":"public VoxTell v1.1 resolved by download_voxtell_model","base_model_dir":str(b),"plans_path":str(b/"plans.json"),"plans_sha256":h(b/"plans.json"),"checkpoint_path":str(c),"checkpoint_sha256":h(c)},indent=2,sort_keys=True)+"\n")' \
  "$BASE_MODEL_DIR" "$EXP_DIR/config/public_voxtell_v1_1_provenance.json"

[[ "$RUN_STATIC_ONLY" == "1" ]] && { echo "Static checks complete"; exit 0; }

common_train_args() {
  printf '%s\n' --mode train --experiment-id "$EXP_ID" --config "$CONFIG_SNAPSHOT" \
    --exp-dir "$EXP_DIR" --model-dir "$BASE_MODEL_DIR" --embeddings "$EMBEDDINGS" \
    --seed "$SEED" --patch-size 192 192 192 --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
    --encoder-lr 1e-5 --decoder-lr 1e-4 --lr-schedule fixed --warmup-updates 0 \
    --momentum 0.99 --weight-decay 3e-5 --clip-grad-norm 12 --require-positive-crop \
    --loss-mode empty_bce_only --empty-target-loss-weight 0.5 --voxel-bce-weighting \
    --bce-foreground-weight 1.0 --bce-boundary-weight 1.5 --bce-background-weight 0.5 \
    --bce-boundary-radius 10 --deep-supervision-weights 1,0.5,0.25,0.125,0.0625 \
    --preprocessed-cache-dir "$CACHE_ROOT" --sample-schedule "$SCHEDULE" --distributed \
    --allow-experimental-train
}

checkpoint_has_update() {
  local checkpoint="$1" expected="$2"
  [[ -s "$checkpoint" ]] || return 1
  python -c 'import sys,torch; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False); raise SystemExit(0 if int(x.get("global_update",-1))==int(sys.argv[2]) else 1)' "$checkpoint" "$expected"
}

run_sample_test() {
  local d="$GROUP_DIR/smoke/sample_test"
  mkdir -p "$d/logs"
  CUDA_VISIBLE_DEVICES=0 python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
    --mode sample-test --experiment-id "$EXP_ID" --config "$CONFIG_SNAPSHOT" --exp-dir "$EXP_DIR" \
    --run-dir "$d" --gpu 0 --seed "$SEED" --patch-size 192 192 192 --sample-schedule "$SCHEDULE" \
    --preprocessed-cache-dir "$CACHE_ROOT" --require-positive-crop --loss-mode empty_bce_only \
    --sample-test-steps 8 2>&1 | tee "$d/logs/sample_test.log"
}

run_ddp_smoke() {
  local d="$GROUP_DIR/smoke/ddp_resume_smoke"
  mkdir -p "$d/logs"
  mapfile -t args < <(common_train_args)
  CUDA_VISIBLE_DEVICES="$DDP_CUDA_VISIBLE_DEVICES" python -m torch.distributed.run --standalone \
    --master-port="$MASTER_PORT" --nproc_per_node="$WORLD_SIZE" \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py "${args[@]}" \
    --run-dir "$d" --epochs 1 --steps-per-epoch 4 --target-global-update 2 \
    --checkpoint-every-updates 0 --checkpoint-updates 2,4 --latest-checkpoint-every-updates 0 \
    --no-materialize-final-model >"$d/logs/train_to_update_2.log" 2>&1
  checkpoint_has_update "$d/checkpoints/checkpoint_update_000002.pth" 2
  CUDA_VISIBLE_DEVICES="$DDP_CUDA_VISIBLE_DEVICES" python -m torch.distributed.run --standalone \
    --master-port="$((MASTER_PORT + 1))" --nproc_per_node="$WORLD_SIZE" \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py "${args[@]}" \
    --run-dir "$d" --epochs 1 --steps-per-epoch 4 --target-global-update 4 \
    --checkpoint-every-updates 0 --checkpoint-updates 2,4 --latest-checkpoint-every-updates 0 \
    --no-materialize-final-model --resume-checkpoint "$d/checkpoints/checkpoint_update_000002.pth" \
    >"$d/logs/train_to_update_4.log" 2>&1
  checkpoint_has_update "$d/checkpoints/checkpoint_update_000004.pth" 4
}

materialize_model() {
  local checkpoint="$1" model_dir="$2" target tmp
  mkdir -p "$model_dir/fold_0"
  cp "$SOURCE_MODEL_DIR/plans.json" "$model_dir/plans.json"
  target="$model_dir/fold_0/checkpoint_final.pth"
  if [[ ! -f "$target" || "$(stat -c %s "$target")" != "$(stat -c %s "$checkpoint")" ]]; then
    tmp="$target.tmp.$$"; rm -f "$tmp"
    ln "$checkpoint" "$tmp" 2>/dev/null || cp "$checkpoint" "$tmp"
    mv -f "$tmp" "$target"
  fi
}

run_inference_smoke() {
  local d="$GROUP_DIR/smoke/ddp_resume_smoke" model="$GROUP_DIR/smoke/model_epoch000" out="$GROUP_DIR/smoke/one_case_inference"
  mkdir -p "$out"
  materialize_model "$d/checkpoints/checkpoint_update_000004.pth" "$model"
  python -c 'import json,sys; x=json.load(open(sys.argv[1])); x["test"]=x["test"][:1]; json.dump(x,open(sys.argv[2],"w"),indent=2)' \
    "$VAL200_JSON" "$GROUP_DIR/smoke/one_case_val200.json"
  EVAL_LOCK_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split val --dataset-json "$GROUP_DIR/smoke/one_case_val200.json" --gpu 0 --model-dir "$model" \
    --embeddings "$EMBEDDINGS" --output-dir "$out/predictions" --threshold 0.5 \
    --preprocessed-cache-dir "$CACHE_ROOT" --status-json "$out/status.json" \
    2>&1 | tee "$out/inference.log"
}

eval_complete() {
  local d="$1" expected
  [[ -f "$d/reports/val_quick_global_eval_summary.json" && -f "$d/eval/val_quick_global_eval.json" ]] || return 1
  expected="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["test"]))' "$VAL200_JSON")"
  [[ "$(find "$d/predictions" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l)" == "$expected" ]]
}

checkpoint_stable() {
  local path="$1" before after
  [[ -s "$path" ]] || return 1
  before="$(stat -c '%s:%Y' "$path")"
  sleep "$STABILITY_SECONDS"
  after="$(stat -c '%s:%Y' "$path")"
  [[ "$before" == "$after" ]]
}

acquire_lock() {
  local d="$1" lock="$d/.eval.lock" age now
  mkdir -p "$d"
  while true; do
    if (set -o noclobber; printf '{"pid":%s,"created":"%s"}\n' "$$" "$(date -u +%FT%TZ)" > "$lock") 2>/dev/null; then
      echo "$lock"
      return 0
    fi
    now="$(date +%s)"
    age=$((now - $(stat -c %Y "$lock" 2>/dev/null || echo "$now")))
    if (( age > LOCK_STALE_SECONDS )); then
      rm -f "$lock"
    else
      sleep 60
    fi
  done
}

run_inference_shard() {
  local epoch="$1" gpu="$2" shard="$3"
  local d="$RUN_DIR/eval_epoch$(printf '%03d' "$epoch")_val200"
  local model="$RUN_DIR/model_epoch$(printf '%03d' "$epoch")"
  mkdir -p "$d/logs" "$d/status" "$d/predictions"
  EVAL_LOCK_HELD=1 CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
      --split val --dataset-json "$VAL200_JSON" --gpu 0 --model-dir "$model" \
      --embeddings "$EMBEDDINGS" --output-dir "$d/predictions" --output-type mask --threshold 0.5 \
      --preprocessed-cache-dir "$CACHE_ROOT" --num-shards 4 --shard-index "$shard" \
      --status-json "$d/status/shard${shard}_gpu${gpu}.json" \
      >"$d/logs/inference_shard${shard}_gpu${gpu}.log" 2>&1
}

run_val200_eval() {
  local epoch="$1" update="$((epoch * STEPS_PER_EPOCH))" lock
  local checkpoint="$RUN_DIR/checkpoints/checkpoint_update_$(printf '%06d' "$update").pth"
  local model="$RUN_DIR/model_epoch$(printf '%03d' "$epoch")"
  local d="$RUN_DIR/eval_epoch$(printf '%03d' "$epoch")_val200"
  if eval_complete "$d"; then
    python /workspace/scripts/rexgroundingct/summarize_026_results.py \
      --exp-dir "$EXP_DIR" --group-dir "$GROUP_DIR" --output-json "$REPORT_JSON" --output-md "$REPORT_MD" || true
    return 0
  fi
  checkpoint_stable "$checkpoint" || return 1
  lock="$(acquire_lock "$d")"
  materialize_model "$checkpoint" "$model"
  mkdir -p "$d/logs" "$d/predictions"
  {
    local -a pids=()
    for shard in 0 1 2 3; do
      run_inference_shard "$epoch" "${GPUS[$shard]}" "$shard" &
      pids+=("$!")
    done
    local status=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    [[ "$status" == "0" ]] || exit 1
    EVAL_LOCK_HELD=1 GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$VAL200_JSON" \
      EVAL_LABEL="Exp026 fixed-LR DDP batch4 epoch $epoch val200 threshold 0.5" NUM_WORKERS=8 \
      bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$d" val
  } >"$d/logs/eval_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 || {
    rm -f "$lock"
    return 1
  }
  rm -f "$lock"
  python /workspace/scripts/rexgroundingct/summarize_026_results.py \
    --exp-dir "$EXP_DIR" --group-dir "$GROUP_DIR" --output-json "$REPORT_JSON" --output-md "$REPORT_MD" || true
}

run_segment() {
  local epoch="$1" previous_epoch="${2:-}" target_update previous_update checkpoint previous_checkpoint
  target_update="$((epoch * STEPS_PER_EPOCH))"
  checkpoint="$RUN_DIR/checkpoints/checkpoint_update_$(printf '%06d' "$target_update").pth"
  [[ -s "$checkpoint" ]] && checkpoint_has_update "$checkpoint" "$target_update" && return 0
  mapfile -t args < <(common_train_args)
  local -a resume_args=()
  if [[ -n "$previous_epoch" ]]; then
    previous_update="$((previous_epoch * STEPS_PER_EPOCH))"
    previous_checkpoint="$RUN_DIR/checkpoints/checkpoint_update_$(printf '%06d' "$previous_update").pth"
    checkpoint_has_update "$previous_checkpoint" "$previous_update"
    resume_args=(--resume-checkpoint "$previous_checkpoint")
  fi
  CUDA_VISIBLE_DEVICES="$DDP_CUDA_VISIBLE_DEVICES" python -m torch.distributed.run --standalone \
    --master-port="$MASTER_PORT" --nproc_per_node="$WORLD_SIZE" \
    /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py "${args[@]}" \
    --run-dir "$RUN_DIR" --epochs "$EPOCHS" --steps-per-epoch "$STEPS_PER_EPOCH" \
    --target-global-update "$target_update" --checkpoint-every-updates 0 \
    --checkpoint-updates "$CHECKPOINT_UPDATES" --latest-checkpoint-every-updates 500 \
    --no-materialize-final-model "${resume_args[@]}" \
    >"$RUN_DIR/logs/train_segment_to_epoch$(printf '%03d' "$epoch").log" 2>&1
  checkpoint_has_update "$checkpoint" "$target_update"
  cp "$RUN_DIR/reports/training_metrics.json" \
    "$RUN_DIR/reports/training_metrics_segment_epoch$(printf '%03d' "$epoch").json"
}

if [[ "$RUN_SAMPLE_TEST" == "1" ]]; then run_sample_test; fi
if [[ "$RUN_DDP_SMOKE" == "1" ]]; then run_ddp_smoke; fi
if [[ "$RUN_INFERENCE_SMOKE" == "1" ]]; then run_inference_smoke; fi
if [[ "$RUN_SMOKE_ONLY" == "1" ]]; then exit 0; fi
if [[ "$RUN_FULL" != "1" ]]; then exit 0; fi

status=0
previous_epoch=""
for epoch in $VAL200_EPOCHS; do
  if run_segment "$epoch" "$previous_epoch" && run_val200_eval "$epoch"; then
    previous_epoch="$epoch"
  else
    status=1
    break
  fi
done

python /workspace/scripts/rexgroundingct/summarize_026_results.py \
  --exp-dir "$EXP_DIR" --group-dir "$GROUP_DIR" --output-json "$REPORT_JSON" --output-md "$REPORT_MD" || true
if [[ "$status" == "0" ]]; then
  touch "$RUN_DIR/.train_complete" "$GROUP_DIR/.experiment_complete"
else
  touch "$RUN_DIR/.train_failed"
fi
echo "Experiment 026 group dir: $GROUP_DIR"
exit "$status"
