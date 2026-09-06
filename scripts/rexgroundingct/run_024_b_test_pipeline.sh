#!/usr/bin/env bash
set -euo pipefail

ROOT="${RUNTIME_ROOT:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit}"
IMAGE="${IMAGE:-rexgroundingct-totalsegmentator:2.16.0-cu126}"
MODEL_BASE="/mnt/shengdata1/hengjie/experiments/rexgroundingct"
DATASET="/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json"
RAW="$ROOT/test_raw"
mkdir -p "$RAW"
if [[ ! -f "$ROOT/b_validation_complete.json" ]]; then
  echo "B-test gate is closed: b_validation_complete.json is missing" >&2
  exit 2
fi
for set_name in a1 a2 a3; do
  if [[ $(find "$ROOT/outputs/$set_name" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -ne 300 ]]; then
    echo "B-test gate is closed: outputs/$set_name is not complete" >&2
    exit 2
  fi
done

run_one() {
  local candidate="$1" model="$2" cache="$3"
  local out="$RAW/$candidate"
  if [[ "$candidate" == "exp007_cont_e050_abs_e150" && $(find "$ROOT/outputs/a1" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -eq 300 ]]; then
    mkdir -p "$out/predictions"
    cp -al "$ROOT/outputs/a1"/*.nii.gz "$out/predictions/"
    python3 -c 'import json,sys; json.dump({"reused_from": sys.argv[2], "reason": "identical Exp007 e050/abs e150 checkpoint and verified native test outputs", "hardlinks": True}, open(sys.argv[1], "w"), indent=2); open(sys.argv[1], "a").write("\\n")' "$out/reuse_manifest.json" "$ROOT/outputs/a1"
    return
  fi
  if [[ $(find "$out/predictions" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -ne 300 ]]; then
    NUM_SHARDS=4 bash "$PWD/scripts/rexgroundingct/run_024_inference_shards.sh" "$model" "$cache" "$out/predictions" test "$DATASET"
  fi
}

# One unique checkpoint is run once, in the fixed order used by the oracle.
run_one exp017_cont_e025_abs_e125 "$MODEL_BASE/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_cont100_from_iso07_epoch100_20260816T165351Z/ddp_bs4/model_epoch025" "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1"
run_one exp012_r02_category_2d_replay50_e100 "$MODEL_BASE/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_r02_replay_ratio_20260802T122657Z/category_2d_replay50/model_epoch100" "$ROOT/cache/crop_zscore_native_v1"
run_one exp017_cont_e075_abs_e175 "$MODEL_BASE/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_cont100_from_iso07_epoch100_20260816T165351Z/ddp_bs4/model_epoch075" "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1"
run_one exp011_e4d4_clip_linear_e100 "$MODEL_BASE/011_voxtell_v123_e4d4_ct_normalization_ablation/runs/exp011_ct_norm_e4d4_20260728T085103Z/v123_e4d4_clip1024_linear/model_epoch100" "$ROOT/cache/crop_clip1024_linear_native_v1"
run_one exp007_cont_e100_abs_e200 "$MODEL_BASE/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/model_epoch100" "$ROOT/cache/crop_zscore_native_v1"
run_one exp009_s3v2_e100 "$MODEL_BASE/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/s3v2_balanced_feature_half_quarter/model_epoch100" "$ROOT/cache/crop_zscore_native_v1"
run_one exp013_category_2all_focal_target100_e100 "$MODEL_BASE/013_voxtell_public_category_only_specialists/runs/exp013_category_only_voxtell_20260803T061201Z/category_2all_focal_target100/model_epoch100" "$ROOT/cache/crop_zscore_native_v1"
run_one exp007_cont_e050_abs_e150 "$MODEL_BASE/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/model_epoch050" "$ROOT/cache/crop_zscore_native_v1"
run_one exp007_ddp_e025 "$MODEL_BASE/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_full_20260725T231624Z/ddp_bs4/model_epoch025" "$ROOT/cache/crop_zscore_native_v1"

docker run --rm --ipc=host --shm-size=8g --user "$(id -u):$(id -g)" -v "$PWD:/workspace" -v /data/hengjie:/data/hengjie -v /mnt/shengdata1:/mnt/shengdata1 -e HOME=/tmp "$IMAGE" bash -lc "PYTHONPATH=/workspace/scripts/rexgroundingct python /workspace/scripts/rexgroundingct/run_024_test_inference_anatomy.py compose --split test --source '$RAW' --output '$ROOT/outputs/b1'"
NUM_SHARDS=8 bash "$PWD/scripts/rexgroundingct/run_024_apply_shards.sh" "$ROOT/outputs/b1" "$ROOT/outputs/b2" whole_lung test
NUM_SHARDS=8 bash "$PWD/scripts/rexgroundingct/run_024_apply_shards.sh" "$ROOT/outputs/b1" "$ROOT/outputs/b3" fine test
for set_name in b1 b2 b3; do
  docker run --rm --ipc=host --shm-size=8g --user "$(id -u):$(id -g)" \
    -v "$PWD:/workspace" -v /data/hengjie:/data/hengjie -v /mnt/shengdata1:/mnt/shengdata1 \
    -e HOME=/tmp "$IMAGE" bash -lc \
    "PYTHONPATH=/workspace/scripts/rexgroundingct python /workspace/scripts/rexgroundingct/run_024_test_inference_anatomy.py verify --split test --output '$ROOT/outputs/$set_name'"
done
PYTHONPATH="$PWD/scripts/rexgroundingct" python scripts/rexgroundingct/finalize_024_outputs.py
echo "B_TEST_RAW_AND_FINALS_COMPLETE"
