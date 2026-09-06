#!/usr/bin/env bash
set -euo pipefail

# Exp007 phase 3: fresh 100-epoch continuation from phase-2 absolute epoch 200.
# This entrypoint runs inside the pinned VoxTell container.

export PHASE3_MODE=1
export CONTINUATION_MODE=1
export CONTINUATION_SOURCE_RUN_GROUP="${CONTINUATION_SOURCE_RUN_GROUP:-exp007_cont100_from_ddp100_20260730T062051Z}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/${CONTINUATION_SOURCE_RUN_GROUP}/ddp_bs4/checkpoints/checkpoint_update_010000.pth}"
export INIT_CHECKPOINT_EXPECTED_UPDATE="10000"
export EXPECTED_INIT_CHECKPOINT_SHA="d56474a645f90e827af14e116454362d146db835ef5cf111120fb29ab8b78c0c"
export ABSOLUTE_EPOCH_OFFSET="${ABSOLUTE_EPOCH_OFFSET:-200}"
export EXTENDED_SCHEDULE_EVENTS="${EXTENDED_SCHEDULE_EVENTS:-120000}"
export SCHEDULE_START_EVENT="${SCHEDULE_START_EVENT:-80000}"
export PRIOR_EXTENDED_SCHEDULE_EVENTS="${PRIOR_EXTENDED_SCHEDULE_EVENTS:-80000}"
export CHECKPOINT_UPDATES="${CHECKPOINT_UPDATES:-1000,2000,3000,4000,5000,6000,7000,8000,9000,10000}"
export SEGMENT_EPOCHS="${SEGMENT_EPOCHS:-10 20 30 40 50 60 70 80 90 100}"
export RUN_SAMPLE_TEST="${RUN_SAMPLE_TEST:-0}"
export RUN_DDP_SMOKE="${RUN_DDP_SMOKE:-0}"
export RUN_INFERENCE_SMOKE="${RUN_INFERENCE_SMOKE:-0}"
export RUN_FULL="${RUN_FULL:-1}"
export RUN_SMOKE_ONLY="${RUN_SMOKE_ONLY:-0}"
export MASTER_PORT="${MASTER_PORT:-29618}"
export GPU_LIST="${GPU_LIST:-0 1 2 3}"

exec bash /workspace/scripts/rexgroundingct/run_007_ddp_bs4_update_matched.sh
