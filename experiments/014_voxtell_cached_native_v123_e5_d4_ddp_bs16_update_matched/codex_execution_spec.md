---
created: 2026-08-03
updated: 2026-08-03
status: draft
experiment_id: "014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched"
---

# Codex Execution Spec

## Objective

Run a four-GPU DDP effective-batch-16 version of the cached-native v123 e5/d4
recipe to test whether a larger global batch improves over exp007 DDP global
batch 4.

## Prior Evidence

- Exp007 DDP batch4 epoch100 val200: Dice `0.3310`, hit rate `0.7585`
  (`289/381`).
- Exp007 used DDP world size `4`, local batch `1`, accumulation `1`, and was
  stable at about `30.44` GiB peak allocated.
- The experiment 002 batch-size probe found local batch `2` fit but had thin
  memory margin and poor per-patch throughput, so exp014 uses local batch `1`
  with accumulation instead.
- `docs/voxtell/ddp_and_effective_batch_size.md` defines the project comparison
  rules for DDP and effective batch size.

## Scope

In scope:

- Add a canonical exp014 config and launcher.
- Use DDP world size `4`, local batch `1`, gradient accumulation `4`, effective
  global batch `16`.
- Generate a deterministic `160000` event v123 positive-crop schedule.
- Train for `10000` optimizer updates when approved.
- Save immutable checkpoints and run fixed val200 at epochs `25/50/75/100`.
- Resume segmented training from full optimizer/scaler checkpoints, preferring
  the highest valid rolling checkpoint after a mid-segment kill.

Out of scope:

- Launching the full 100-epoch run before user approval.
- Changing preprocessing, normalization, patch size, model architecture, prompt
  policy, loss, optimizer family, or LR values.
- Saving probability maps by default.
- Scaling LR for batch16 in the first controlled run.

## Inputs And Paths

- Canonical config:
  `configs/experiments/014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched.json`
- Metadata:
  `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- CT root:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Segmentation root:
  `/data/hengjie/datasets/rexgroundingct/segmentations`
- Runtime experiment directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched`
- Repo-local summary directory:
  `experiments/014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched`
- Native cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1`
- Fixed val200:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`

## Data And Preprocessing Contract

- Baseline reference: exp007 DDP batch4 e5/d4.
- Changed preprocessing variables: none.
- Unchanged controls: orientation fix, crop-to-nonzero, full cropped-volume
  z-score once, native-resolution `192^3` patch/window sampling.
- Normalization scope: full cropped volume before patch sampling.
- Resampling and interpolation: none.
- Patch/window policy: v123 positive-crop schedule, global batch16 event
  consumption.
- Cache root and ownership: shared derived-data cache under `/mnt/shengdata1`,
  not Git.
- Orientation/export check: one-case cached-native inference smoke from the
  DDP smoke checkpoint.
- Comparability caveat: exp014 epochs are update-defined, not sample-defined:
  `100` optimizer updates per epoch and `16` patch events per update. Relative
  to exp007 batch4, exp014 epoch25 is sample-matched to exp007 epoch100 at
  `40000` patch events but has only `2500` optimizer updates; exp014 epoch100
  is update-matched to exp007 epoch100 at `10000` optimizer updates but sees
  `160000` patch events, i.e. `4x` exp007 batch4 and `16x` a batch1 run with
  the same update count.

## Method

Use public VoxTell v1.1 initialization and reset optimizer state at experiment
start:

```text
world_size=4
per_gpu_batch_size=1
grad_accum=4
effective_global_batch=16
```

Training uses the v123 recipe:

- SGD Nesterov, momentum `0.99`, weight decay `3e-5`;
- encoder LR `1e-5`, decoder LR `1e-4`;
- warmup `100` optimizer updates;
- poly LR decay over `10000` optimizer updates, power `0.9`;
- gradient clipping max norm `12`;
- nonempty target loss: Dice plus weighted BCE;
- empty target loss: weighted BCE only with weight `0.5`;
- BCE weights foreground/boundary/background `1.0/1.5/0.5`;
- boundary radius `10`;
- deep supervision weights `[1, 0.5, 0.25, 0.125, 0.0625]`;
- require every selected positive target to be nonempty in the sampled patch;
- use `DDP.no_sync()` on non-final accumulation microsteps.

The schedule consumption rule is:

```text
event_index = update * 16 + rank * 4 + local_offset
```

Run segments after approval:

```text
segment 1: train update 0    -> 2500,  val200 epoch25
segment 2: train update 2500 -> 5000,  val200 epoch50
segment 3: train update 5000 -> 7500,  val200 epoch75
segment 4: train update 7500 -> 10000, val200 epoch100
```

Smoke launcher:

```bash
DETACH=0 RUN_SMOKE_ONLY=1 RUN_FULL=0 \
  bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/run_014_ddp_bs16_update_matched_docker.sh
```

Full launcher, only after approval:

```bash
RUN_SMOKE_ONLY=0 RUN_FULL=1 \
  bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/run_014_ddp_bs16_update_matched_docker.sh
```

## Evaluation Execution Contract

- Milestone checkpoints: optimizer updates `2500/5000/7500/10000`.
- Evaluation set: fixed val200
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`, SHA256
  `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`.
- Resource policy: synchronous milestone barrier; stop training, run val200 on
  all four GPUs, then resume.
- Barrier scope: one exp014 arm.
- Resume state: model, optimizer, scaler, LR horizon, global update, and
  schedule cursor are restored from checkpoint. Multi-rank RNG is not bitwise
  exact because current DDP checkpoints capture rank0 RNG only; materialized
  schedule preserves data-order resume.
- Locking, idempotency, and retry behavior: eval uses `.eval.lock`, detects
  complete val200 outputs, and skips completed barriers on rerun.

## Smoke Gate

- Static config/readiness/cache/hash checks.
- Schedule generation and sample-test from the exp014 global-batch-16 schedule.
- DDP smoke from update `0 -> 2`, then resume `2 -> 4`.
- One-case cached-native inference smoke from the update-4 checkpoint.
- No full training segment starts unless `RUN_FULL=1`.

## Success Criteria

- Smoke run completes without OOM or nonfinite loss.
- DDP resume smoke checkpoint at update `4` exists and has `global_update=4`.
- One-case inference smoke writes a prediction and status JSON.
- Full run, after approval, completes `10000` optimizer updates and val200 at
  epochs `25/50/75/100`.

## Verification

Before handoff:

```bash
python -m py_compile scripts/rexgroundingct/train_text_conditioned_voxtell.py scripts/rexgroundingct/summarize_014_results.py
bash -n scripts/rexgroundingct/run_014_ddp_bs16_update_matched.sh
bash -n scripts/rexgroundingct/run_014_ddp_bs16_update_matched_docker.sh
python scripts/rexgroundingct/check_experiment_consistency.py --experiments 014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched
```

Smoke execution:

```bash
DETACH=0 RUN_SMOKE_ONLY=1 RUN_FULL=0 \
  bash scripts/rexgroundingct/run_014_ddp_bs16_update_matched_docker.sh
```

## Closeout Plan

After the approved full run completes, sync the experiment index, copy the
runtime report into the repo-local experiment folder, and update
`docs/current_status.md` only if exp014 changes the next training decision.
