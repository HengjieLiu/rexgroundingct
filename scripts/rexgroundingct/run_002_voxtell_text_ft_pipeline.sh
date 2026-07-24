#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val}"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/checkpoints" "$EXP_DIR/reports" "$EXP_DIR/eval" "$EXP_DIR/config"

python /workspace/scripts/rexgroundingct/poll_ct_subset.py \
  --splits train val \
  --json "$EXP_DIR/config/train_val_ct_readiness.json"

bash /workspace/scripts/rexgroundingct/run_002_precompute_text_embeddings.sh
bash /workspace/scripts/rexgroundingct/run_002_single_gpu_zscore192_baseline.sh
