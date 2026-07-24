#!/usr/bin/env bash
set -euo pipefail

EXP_ID="003_voxtell_rex_ft_rescue_ablation"
EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/$EXP_ID}"
EXP002_DIR="${EXP002_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val}"
SEED="${SEED:-20260723}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SHORT_EPOCHS="${SHORT_EPOCHS:-5}"
FULL_EPOCHS="${FULL_EPOCHS:-100}"
FULL_EVENTS="$((FULL_EPOCHS * STEPS_PER_EPOCH * BATCH_SIZE * GRAD_ACCUM))"
SHORT_EVENTS="$((SHORT_EPOCHS * STEPS_PER_EPOCH * BATCH_SIZE * GRAD_ACCUM))"
CHECKPOINT_EVERY_FULL="${CHECKPOINT_EVERY_FULL:-$((20 * STEPS_PER_EPOCH))}"
RUN_PARALLEL="${RUN_PARALLEL:-1}"
RUN_SAMPLE_TEST="${RUN_SAMPLE_TEST:-1}"
SAMPLE_TEST_STEPS="${SAMPLE_TEST_STEPS:-16}"
RUN_TINY_TRAIN_SMOKE="${RUN_TINY_TRAIN_SMOKE:-0}"
RUN_SHORT="${RUN_SHORT:-1}"
RUN_FULL="${RUN_FULL:-1}"
RUN_VAL="${RUN_VAL:-1}"
SCHEDULE_NUM_WORKERS="${SCHEDULE_NUM_WORKERS:-16}"
FORCE_REGENERATE_SCHEDULES="${FORCE_REGENERATE_SCHEDULES:-0}"
ARM_SLEEP_SECONDS="${ARM_SLEEP_SECONDS:-10}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUP="${RUN_GROUP:-exp003_rescue_ablation_$TIMESTAMP}"
GROUP_DIR="${GROUP_DIR:-$EXP_DIR/runs/$RUN_GROUP}"
CONFIG_SNAPSHOT="$EXP_DIR/config/${EXP_ID}.json"
EMBEDDINGS="${EMBEDDINGS:-$EXP_DIR/config/rex_text_embeddings.npz}"
VAL20_JSON="${VAL20_JSON:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.json}"
VAL20_MANIFEST="${VAL20_MANIFEST:-/workspace/configs/evaluation/rexgroundingct_val20_seed${SEED}.manifest.json}"
VAL200_JSON="${VAL200_JSON:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.json}"
VAL200_MANIFEST="${VAL200_MANIFEST:-/workspace/configs/evaluation/rexgroundingct_val200_seed${SEED}.manifest.json}"

ENCODER_LR="${ENCODER_LR:-1e-7}"
DECODER_LR="${DECODER_LR:-1e-6}"
WARMUP_UPDATES="${WARMUP_UPDATES:-100}"
POLY_POWER="${POLY_POWER:-0.9}"
MOMENTUM="${MOMENTUM:-0.99}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-5}"
CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-12}"

mkdir -p "$EXP_DIR/config" "$EXP_DIR/logs" "$GROUP_DIR"

python /workspace/scripts/rexgroundingct/snapshot_experiment_config.py \
  --experiment "$EXP_ID" \
  --exp-dir "$EXP_DIR" \
  --overwrite

python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val \
  --json "$EXP_DIR/config/train_val_ct_readiness.json"

if [[ ! -f "$EMBEDDINGS" ]]; then
  if [[ -f "$EXP002_DIR/config/rex_text_embeddings.npz" ]]; then
    cp "$EXP002_DIR/config/rex_text_embeddings.npz" "$EMBEDDINGS"
  else
    python /workspace/scripts/rexgroundingct/precompute_text_embeddings.py \
      --splits train val \
      --gpu 0 \
      --output "$EMBEDDINGS" \
      2>&1 | tee "$EXP_DIR/logs/precompute_text_embeddings_$(date -u +%Y%m%dT%H%M%SZ).log"
  fi
fi

if [[ ! -f "$VAL20_JSON" ]]; then
  python /workspace/scripts/rexgroundingct/prepare_fixed_probe_subset.py \
    --split val \
    --seed "$SEED" \
    --size 20 \
    --output-json "$VAL20_JSON" \
    --manifest-json "$VAL20_MANIFEST"
fi

if [[ ! -f "$VAL200_JSON" ]]; then
  python /workspace/scripts/rexgroundingct/prepare_fixed_probe_subset.py \
    --split val \
    --seed "$SEED" \
    --size 200 \
    --output-json "$VAL200_JSON" \
    --manifest-json "$VAL200_MANIFEST"
fi

if [[ "$FORCE_REGENERATE_SCHEDULES" == "1" ]]; then
  rm -f \
    "$EXP_DIR"/config/train_schedule_v1_opt_seed"${SEED}"_"${FULL_EPOCHS}"ep_"${STEPS_PER_EPOCH}"steps_gb"${BATCH_SIZE}".jsonl \
    "$EXP_DIR"/config/train_schedule_v1_opt_seed"${SEED}"_"${FULL_EPOCHS}"ep_"${STEPS_PER_EPOCH}"steps_gb"${BATCH_SIZE}".manifest.json \
    "$EXP_DIR"/config/train_schedule_v12_opt_poscrop_seed"${SEED}"_"${FULL_EPOCHS}"ep_"${STEPS_PER_EPOCH}"steps_gb"${BATCH_SIZE}".jsonl \
    "$EXP_DIR"/config/train_schedule_v12_opt_poscrop_seed"${SEED}"_"${FULL_EPOCHS}"ep_"${STEPS_PER_EPOCH}"steps_gb"${BATCH_SIZE}".manifest.json \
    "$EXP_DIR"/config/train_schedule_v123_opt_poscrop_emptyloss_seed"${SEED}"_"${FULL_EPOCHS}"ep_"${STEPS_PER_EPOCH}"steps_gb"${BATCH_SIZE}".jsonl \
    "$EXP_DIR"/config/train_schedule_v123_opt_poscrop_emptyloss_seed"${SEED}"_"${FULL_EPOCHS}"ep_"${STEPS_PER_EPOCH}"steps_gb"${BATCH_SIZE}".manifest.json \
    "$EXP_DIR"/config/train_schedule_v1234_opt_poscrop_emptyloss_ds_seed"${SEED}"_"${FULL_EPOCHS}"ep_"${STEPS_PER_EPOCH}"steps_gb"${BATCH_SIZE}".jsonl \
    "$EXP_DIR"/config/train_schedule_v1234_opt_poscrop_emptyloss_ds_seed"${SEED}"_"${FULL_EPOCHS}"ep_"${STEPS_PER_EPOCH}"steps_gb"${BATCH_SIZE}".manifest.json \
    "$EXP_DIR"/config/train_schedule_poscrop_seed"${SEED}"_"${FULL_EPOCHS}"ep_"${STEPS_PER_EPOCH}"steps_gb"${BATCH_SIZE}".jsonl \
    "$EXP_DIR"/config/train_schedule_poscrop_seed"${SEED}"_"${FULL_EPOCHS}"ep_"${STEPS_PER_EPOCH}"steps_gb"${BATCH_SIZE}".manifest.json
fi

base_train_args=(
  --mode train
  --experiment-id "$EXP_ID"
  --config "$CONFIG_SNAPSHOT"
  --exp-dir "$EXP_DIR"
  --gpu 0
  --embeddings "$EMBEDDINGS"
  --seed "$SEED"
  --patch-size 192 192 192
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --encoder-lr "$ENCODER_LR"
  --decoder-lr "$DECODER_LR"
  --lr-schedule poly
  --poly-power "$POLY_POWER"
  --warmup-updates "$WARMUP_UPDATES"
  --momentum "$MOMENTUM"
  --weight-decay "$WEIGHT_DECAY"
  --clip-grad-norm "$CLIP_GRAD_NORM"
  --allow-experimental-train
)

variant_args() {
  local variant="$1"
  case "$variant" in
    v1_opt)
      printf '%s\n' "--deep-supervision-weights" "1,0.5,0.25,0.125,0.0625"
      ;;
    v12_opt_poscrop)
      printf '%s\n' "--require-positive-crop" "--deep-supervision-weights" "1,0.5,0.25,0.125,0.0625"
      ;;
    v123_opt_poscrop_emptyloss)
      printf '%s\n' \
        "--require-positive-crop" \
        "--loss-mode" "empty_bce_only" \
        "--empty-target-loss-weight" "0.5" \
        "--voxel-bce-weighting" \
        "--bce-foreground-weight" "1.0" \
        "--bce-boundary-weight" "1.5" \
        "--bce-background-weight" "0.5" \
        "--bce-boundary-radius" "10" \
        "--deep-supervision-weights" "1,0.5,0.25,0.125,0.0625"
      ;;
    v1234_opt_poscrop_emptyloss_ds)
      printf '%s\n' \
        "--require-positive-crop" \
        "--loss-mode" "empty_bce_only" \
        "--empty-target-loss-weight" "0.5" \
        "--voxel-bce-weighting" \
        "--bce-foreground-weight" "1.0" \
        "--bce-boundary-weight" "1.5" \
        "--bce-background-weight" "0.5" \
        "--bce-boundary-radius" "10" \
        "--deep-supervision-weights" "1,0.5,0.25,0.125,0"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      return 1
      ;;
  esac
}

variant_gpu() {
  case "$1" in
    v1_opt) echo 0 ;;
    v12_opt_poscrop) echo 1 ;;
    v123_opt_poscrop_emptyloss) echo 2 ;;
    v1234_opt_poscrop_emptyloss_ds) echo 3 ;;
    *) return 1 ;;
  esac
}

schedule_for_variant() {
  local variant="$1"
  local schedule="$EXP_DIR/config/train_schedule_${variant}_seed${SEED}_${FULL_EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${BATCH_SIZE}.jsonl"
  local manifest="$EXP_DIR/config/train_schedule_${variant}_seed${SEED}_${FULL_EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${BATCH_SIZE}.manifest.json"
  if [[ ! -f "$schedule" ]]; then
    case "$variant" in
      v12_opt_poscrop|v123_opt_poscrop_emptyloss|v1234_opt_poscrop_emptyloss_ds)
        local shared_schedule="$EXP_DIR/config/train_schedule_poscrop_seed${SEED}_${FULL_EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${BATCH_SIZE}.jsonl"
        local shared_manifest="$EXP_DIR/config/train_schedule_poscrop_seed${SEED}_${FULL_EPOCHS}ep_${STEPS_PER_EPOCH}steps_gb${BATCH_SIZE}.manifest.json"
        if [[ ! -f "$shared_schedule" ]]; then
          python /workspace/scripts/rexgroundingct/prepare_training_schedule.py \
            --seed "$SEED" \
            --events "$FULL_EVENTS" \
            --output-jsonl "$shared_schedule" \
            --manifest-json "$shared_manifest" \
            --patch-size 192 192 192 \
            --require-positive-crop \
            --num-workers "$SCHEDULE_NUM_WORKERS"
        fi
        cp "$shared_schedule" "$schedule"
        python - "$shared_manifest" "$manifest" "$shared_schedule" "$schedule" <<'PY'
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

source_manifest = Path(sys.argv[1])
target_manifest = Path(sys.argv[2])
source_schedule = Path(sys.argv[3])
target_schedule = Path(sys.argv[4])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

data = json.loads(source_manifest.read_text())
data["copied_from_shared_schedule"] = {
    "schedule": str(source_schedule),
    "schedule_sha256": sha256(source_schedule),
    "manifest": str(source_manifest),
}
data["output_jsonl"] = str(target_schedule)
data["output_jsonl_sha256"] = sha256(target_schedule)
data["variant_manifest_created_at_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
target_manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
        echo "$schedule"
        return 0
        ;;
    esac
    args=(
      --seed "$SEED"
      --events "$FULL_EVENTS"
      --output-jsonl "$schedule"
      --manifest-json "$manifest"
      --patch-size 192 192 192
    )
    python /workspace/scripts/rexgroundingct/prepare_training_schedule.py "${args[@]}"
  fi
  echo "$schedule"
}

materialize_checkpoint_model() {
  local source_model_dir="$1"
  local checkpoint="$2"
  local output_model_dir="$3"
  mkdir -p "$output_model_dir/fold_0"
  cp "$source_model_dir/plans.json" "$output_model_dir/plans.json"
  cp "$checkpoint" "$output_model_dir/fold_0/checkpoint_final.pth"
}

run_eval() {
  local visible_gpu="$1"
  local model_dir="$2"
  local output_dir="$3"
  local dataset_json="$4"
  local label="$5"
  mkdir -p "$output_dir/logs"
  CUDA_VISIBLE_DEVICES="$visible_gpu" python /workspace/scripts/rexgroundingct/run_voxtell_val_inference.py \
    --split val \
    --dataset-json "$dataset_json" \
    --gpu 0 \
    --model-dir "$model_dir" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$output_dir/predictions" \
    --overwrite \
    2>&1 | tee "$output_dir/logs/inference.log"

  GLOBAL_ONLY=1 EVAL_DATASET_JSON_OVERRIDE="$dataset_json" \
    EVAL_LABEL="$label" \
    bash /workspace/scripts/rexgroundingct/run_rexrank_eval.sh "$output_dir" val \
    2>&1 | tee "$output_dir/logs/quick_eval.log"
}

run_variant() {
  local variant="$1"
  local visible_gpu="$2"
  local variant_root="$GROUP_DIR/$variant"
  local schedule
  schedule="$(schedule_for_variant "$variant")"
  mkdir -p "$variant_root"

  mapfile -t extra_args < <(variant_args "$variant")

  if [[ "$RUN_SAMPLE_TEST" == "1" ]]; then
    local sample_dir="$variant_root/sample_test"
    mkdir -p "$sample_dir/logs"
    CUDA_VISIBLE_DEVICES="$visible_gpu" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
      --mode sample-test \
      --experiment-id "$EXP_ID" \
      --config "$CONFIG_SNAPSHOT" \
      --exp-dir "$EXP_DIR" \
      --run-dir "$sample_dir" \
      --gpu 0 \
      --seed "$SEED" \
      --sample-schedule "$schedule" \
      --sample-test-steps "$SAMPLE_TEST_STEPS" \
      "${extra_args[@]}" \
      2>&1 | tee "$sample_dir/logs/sample_test.log"
  fi

  if [[ "$RUN_TINY_TRAIN_SMOKE" == "1" ]]; then
    local smoke_dir="$variant_root/tiny_train_smoke"
    mkdir -p "$smoke_dir/logs"
    CUDA_VISIBLE_DEVICES="$visible_gpu" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
      "${base_train_args[@]}" \
      --run-dir "$smoke_dir" \
      --sample-schedule "$schedule" \
      --epochs 1 \
      --steps-per-epoch 1 \
      --checkpoint-every-updates 1 \
      "${extra_args[@]}" \
      2>&1 | tee "$smoke_dir/logs/train.log"
  fi

  if [[ "$RUN_SHORT" == "1" ]]; then
    local short_dir="$variant_root/short_005ep"
    mkdir -p "$short_dir/logs"
    CUDA_VISIBLE_DEVICES="$visible_gpu" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
      "${base_train_args[@]}" \
      --run-dir "$short_dir" \
      --sample-schedule "$schedule" \
      --epochs "$SHORT_EPOCHS" \
      --steps-per-epoch "$STEPS_PER_EPOCH" \
      --checkpoint-every-updates "$SHORT_EVENTS" \
      "${extra_args[@]}" \
      2>&1 | tee "$short_dir/logs/train.log"

    if [[ "$RUN_VAL" == "1" ]]; then
      run_eval \
        "$visible_gpu" \
        "$short_dir/model" \
        "$short_dir/eval_epoch005_val20" \
        "$VAL20_JSON" \
        "Experiment 003 ${variant} short epoch 5 fixed val20 seed ${SEED} quick/global eval"
    fi
  fi

  if [[ "$RUN_FULL" == "1" ]]; then
    local full_dir="$variant_root/full_100ep"
    mkdir -p "$full_dir/logs"
    CUDA_VISIBLE_DEVICES="$visible_gpu" python /workspace/scripts/rexgroundingct/train_text_conditioned_voxtell.py \
      "${base_train_args[@]}" \
      --run-dir "$full_dir" \
      --sample-schedule "$schedule" \
      --epochs "$FULL_EPOCHS" \
      --steps-per-epoch "$STEPS_PER_EPOCH" \
      --checkpoint-every-updates "$CHECKPOINT_EVERY_FULL" \
      "${extra_args[@]}" \
      2>&1 | tee "$full_dir/logs/train.log"

    if [[ "$RUN_VAL" == "1" ]]; then
      for epoch in 20 40 60 80 100; do
        local update=$((epoch * STEPS_PER_EPOCH))
        local checkpoint="$full_dir/checkpoints/checkpoint_update_$(printf '%06d' "$update").pth"
        local model_dir="$full_dir/model_epoch$(printf '%03d' "$epoch")"
        if [[ ! -f "$checkpoint" ]]; then
          echo "Missing expected checkpoint for ${variant} epoch ${epoch}: $checkpoint" >&2
          return 1
        fi
        materialize_checkpoint_model "$full_dir/model" "$checkpoint" "$model_dir"
        run_eval \
          "$visible_gpu" \
          "$model_dir" \
          "$full_dir/eval_epoch$(printf '%03d' "$epoch")_val20" \
          "$VAL20_JSON" \
          "Experiment 003 ${variant} full epoch ${epoch} fixed val20 seed ${SEED} quick/global eval"
      done

      run_eval \
        "$visible_gpu" \
        "$full_dir/model_epoch100" \
        "$full_dir/eval_epoch100_val200" \
        "$VAL200_JSON" \
        "Experiment 003 ${variant} full epoch 100 fixed val200 seed ${SEED} quick/global eval"
    fi
  fi
}

variants=(v1_opt v12_opt_poscrop v123_opt_poscrop_emptyloss v1234_opt_poscrop_emptyloss_ds)
if [[ -n "${VARIANTS:-}" ]]; then
  read -r -a variants <<< "$VARIANTS"
fi

for variant in "${variants[@]}"; do
  echo "Preparing schedule for $variant"
  schedule_for_variant "$variant" >/dev/null
done

if [[ "$RUN_PARALLEL" == "1" ]]; then
  pids=()
  names=()
  for variant in "${variants[@]}"; do
    gpu="$(variant_gpu "$variant")"
    echo "Starting $variant on GPU $gpu under $GROUP_DIR"
    run_variant "$variant" "$gpu" &
    pids+=("$!")
    names+=("$variant")
    sleep "$ARM_SLEEP_SECONDS"
  done

  status=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "Variant failed: ${names[$index]}" >&2
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    exit "$status"
  fi
else
  for variant in "${variants[@]}"; do
    gpu="$(variant_gpu "$variant")"
    echo "Starting $variant on GPU $gpu under $GROUP_DIR"
    run_variant "$variant" "$gpu"
    sleep "$ARM_SLEEP_SECONDS"
  done
fi

echo "Experiment 003 group dir: $GROUP_DIR"
find "$GROUP_DIR" -maxdepth 5 -type f \( -name training_report.md -o -name training_metrics.json -o -name val_quick_global_eval_summary.json \) -print | sort
