#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
REPO_ROOT="${REPO_ROOT:-/home/hengjie/code_sync/rexgroundingct}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONTAINER_NAME="${CONTAINER_NAME:-rex008_dual_branch_$TIMESTAMP}"
DETACH="${DETACH:-1}"

docker_args=(
  docker run
  --rm
  --name "$CONTAINER_NAME"
  --gpus all
  --ipc=host
  --shm-size=32g
  --user "$(id -u):$(id -g)"
  -v "$REPO_ROOT:/workspace"
  -v /data/hengjie:/data/hengjie
  -v /mnt/shengdata1:/mnt/shengdata1
  -e HOME=/tmp
  -e HF_HOME=/data/hengjie/datasets/rexgroundingct/.hf_home
  -e HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub
  -e HF_XET_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/xet
)

for key in EXP_DIR EXP006_DIR EXP007_DIR CACHE_ROOT SOURCE_RUN_GROUP \
  SOURCE_MODEL_DIR SOURCE_EXP006_SCHEDULE SOURCE_EXTENDED_SCHEDULE SEED EPOCHS \
  STEPS_PER_EPOCH SCHEDULE_START SCHEDULE_EVENTS RUN_GROUP GROUP_DIR EMBEDDINGS \
  VAL20_JSON VAL200_JSON CHECKPOINT_UPDATES SEGMENT_EPOCHS PROPOSAL_THRESHOLDS \
  RUN_STATIC_ONLY RUN_SMOKE_ONLY RUN_FULL RUN_MODEL_MATERIALIZATION \
  RUN_TRAIN_SMOKES RUN_BATCH_BENCHMARK RUN_LOGIT_EQUIVALENCE RUN_EPOCH0_EVAL \
  EPOCH0_DICE_TOLERANCE ARM_STAGGER_SECONDS PAUSE_AFTER_EPOCH \
  RUN_CONTROL_POLL_SECONDS; do
  if [[ -n "${!key:-}" ]]; then
    docker_args+=(-e "$key=${!key}")
  fi
done

if [[ "$DETACH" == "1" ]]; then
  docker_args=(docker run -d --rm "${docker_args[@]:3}")
fi

"${docker_args[@]}" "$IMAGE" \
  bash /workspace/scripts/rexgroundingct/run_008_dual_branch_ablation.sh

if [[ "$DETACH" == "1" ]]; then
  echo "Started detached container: $CONTAINER_NAME"
  echo "Logs: docker logs -f $CONTAINER_NAME"
fi
