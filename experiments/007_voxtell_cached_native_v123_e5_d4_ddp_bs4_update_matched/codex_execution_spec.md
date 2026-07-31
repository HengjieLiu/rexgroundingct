---
created: 2026-07-25
updated: 2026-07-25
status: active
experiment_id: "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched"
---

# Codex Execution Spec

## Objective

Run a four-GPU DDP global-batch-4 version of the best exp006 cached-native v123
recipe (`encoder_lr=1e-5`, `decoder_lr=1e-4`) to determine whether future
VoxTell fine-tuning should prefer batch1 or batch4 training.

## Prior Evidence

- Exp006 best arm: `v123_cached_e5_d4`, epoch100 val200 Dice `0.3241`, hit rate
  `0.7612` (`290/381`).
- Reusable DDP/batch-size note:
  `docs/voxtell/ddp_and_effective_batch_size.md`.
- Cached native preprocessing manifest:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1/manifest.json`
  with `3192` cases, `8068` targets, and `0` empty targets.

## Scope

In scope:

- Generate a deterministic 40,000-event v123 positive-crop schedule with seed
  `20260723`.
- Verify the first 10,000 events reproduce the exp003 v123 schedule hash
  `f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776`.
- Train with DDP world size 4, local batch 1, no gradient accumulation.
- Pause training at epochs `25/50/75/100`, run fixed val200 on all four GPUs,
  then resume from full optimizer/scaler state.
- Compare epoch25 as the sample-matched checkpoint and epoch100 as the
  update-matched checkpoint against exp006 `v123_cached_e5_d4`.

Out of scope:

- Changing preprocessing, normalization, patch size, model architecture, prompt
  policy, loss, or LR values.
- Running val20 sidecar evaluations.
- Saving probability maps by default.

## Inputs And Paths

- Canonical config:
  `configs/experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched.json`
- Metadata: `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- CT root: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Segmentation root: `/data/hengjie/datasets/rexgroundingct/segmentations`
- Runtime experiment directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched`
- Repo-local summary directory:
  `experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched`
- Native cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1`
- Fixed val200:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`

## Data And Preprocessing Contract

- Baseline reference: exp006 `v123_cached_e5_d4`.
- Changed preprocessing variables: none.
- Unchanged controls: orientation fix, crop-to-nonzero, full cropped-volume
  z-score once, and native-resolution `192^3` patch/window sampling.
- Normalization scope: full cropped volume before patch sampling.
- Resampling and interpolation: none.
- Patch/window policy: exp003/006 v123 positive-crop schedule, global batch4
  event consumption.
- Cache root and ownership: shared derived-data cache under `/mnt/shengdata1`,
  not Git.
- Orientation/export check: use existing cached-native inference export path.
- Comparability caveat: epoch25 is sample-matched to exp006 epoch100 but has
  fewer optimizer updates; epoch100 is update-matched but sees four times as
  many scheduled samples.

## Method

Use public VoxTell v1.1 initialization and reset optimizer state at experiment
start. DDP uses one process per GPU:

```text
world_size=4
per_gpu_batch_size=1
grad_accum=1
effective_global_batch=4
```

Training uses the v123 recipe:

- SGD Nesterov, momentum `0.99`, weight decay `3e-5`;
- encoder LR `1e-5`, decoder LR `1e-4`;
- warmup `100` optimizer updates;
- poly LR decay over `10,000` optimizer updates, power `0.9`;
- gradient clipping max norm `12`;
- nonempty target loss: Dice plus weighted BCE;
- empty target loss: weighted BCE only with weight `0.5`;
- BCE weights foreground/boundary/background `1.0/1.5/0.5`;
- boundary radius `10`;
- deep supervision weights `[1, 0.5, 0.25, 0.125, 0.0625]`;
- require every selected positive target to be nonempty in the sampled patch.

The schedule consumption rule is:

```text
event_index = update * 4 + rank
```

Run segments:

```text
segment 1: train update 0    -> 2500,  val200 epoch25
segment 2: train update 2500 -> 5000,  val200 epoch50
segment 3: train update 5000 -> 7500,  val200 epoch75
segment 4: train update 7500 -> 10000, val200 epoch100
```

The launcher is:

```bash
IMAGE=rexgroundingct-voxtell:cu126 \
bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/run_007_ddp_bs4_update_matched_docker.sh
```

## Smoke Gate

Before full launch:

- static shell/Python syntax checks;
- schedule verification: 40,000 events, prefix hash matches exp003 v123;
- sample-test with scheduled events;
- DDP two-update train smoke;
- DDP resume smoke from update 2 to update 4;
- one-case cached-native inference smoke from the materialized smoke checkpoint.

## Success Criteria

- DDP run completes `10,000` optimizer updates.
- Val200 summaries exist at epochs `25/50/75/100`.
- Segment resumes preserve model, optimizer, AMP scaler, global update, and
  schedule cursor.
- No selected positive target is empty.
- Runtime report compares Dice, hit rate, hits/targets, loss, throughput, and
  the exp006 baseline.

## Verification

Run before handoff:

```bash
bash -n scripts/rexgroundingct/run_007_ddp_bs4_update_matched.sh
bash -n scripts/rexgroundingct/run_007_ddp_bs4_update_matched_docker.sh
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/rexgroundingct/train_text_conditioned_voxtell.py scripts/rexgroundingct/summarize_007_results.py
python scripts/rexgroundingct/check_experiment_consistency.py
git diff --check
```

## Closeout Plan

After runtime completion:

- write `/mnt/shengdata1/.../reports/ddp_bs4_update_matched_report.md`;
- write `/mnt/shengdata1/.../reports/ddp_bs4_update_matched_summary.json`;
- sync repo-local metrics/report with `sync_experiment_index.py`;
- update `docs/current_status.md` with the batch-size decision signal.

## Phase 2 Continuation From DDP Epoch100

Request date: 2026-07-29.

Run a second 100-epoch continuation inside exp007 from the completed DDP
epoch100 checkpoint. This remains exp007 because the method variable is still
DDP global batch4 for the same cached-native v123 e5/d4 recipe.

Source checkpoint:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_full_20260725T231624Z/ddp_bs4/checkpoints/checkpoint_update_010000.pth
```

Continuation policy:

- load network weights only with `--init-checkpoint`;
- reset optimizer, AMP scaler, scheduler, and update counter;
- repeat the same LR values, warmup, and poly decay over a fresh 10,000-update
  horizon;
- generate an 80,000-event deterministic DDP schedule, verify the first 40,000
  events match the original exp007 schedule, and train on events 40,000-79,999
  rewritten to a zero-based continuation schedule.

Run segments:

```text
relative epoch 25  -> absolute epoch 125, val200
relative epoch 50  -> absolute epoch 150, val200
relative epoch 75  -> absolute epoch 175, val200
relative epoch 100 -> absolute epoch 200, val200
```

Launch:

```bash
IMAGE=rexgroundingct-voxtell:cu126 \
bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/run_007_ddp_bs4_continue100_from_epoch100_docker.sh
```

Reports:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/reports/ddp_bs4_continue100_from_epoch100_report.md
/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/reports/ddp_bs4_continue100_from_epoch100_summary.json
```
