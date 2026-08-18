---
created: 2026-08-15
updated: 2026-08-16
status: phase2_continuation_ready
experiment_id: "017_voxtell_iso07_hu_ddp_bs4_update_matched"
---

# Codex Execution Spec

## Objective

Run one focused VoxTell finetuning experiment on the completed Exp016
`0.7 mm` isotropic clipped-linear-HU cache using the Exp007 DDP global-batch-4
recipe. The goal is to test whether the selected isotropic HU preprocessing
improves fixed val200 performance without changing optimizer, loss, model
initialization, effective batch size, or evaluation cadence.

Phase 2 continues Exp017 using the exact Exp007 continuation protocol: load the
completed epoch100 network weights only, reset optimizer/scaler/LR/update
state, train another 100 relative epochs, and evaluate fixed val200 at relative
epochs `25/50/75/100`.

## Prior Evidence

- Exp016 cache is complete under
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1`.
- Exp016 manifest SHA256:
  `59b53ed9fcd93b8e9872bf89770a14ffd6aeed31b206af33ca751b85d4bd2640`.
- Exp007 is the primary comparison because it matches DDP world size `4`,
  local batch `1`, gradient accumulation `1`, effective global batch `4`,
  public VoxTell initialization, v123 loss, e5/d4 LR, and val200 checkpoints.
- Exp007 epoch100 val200: Dice `0.3310`, hit rate `0.7585`, `289 / 381`.
- Exp011 native linear HU e5/d4 epoch100 is secondary normalization context:
  Dice `0.3205`, hit rate `0.7533`, `287 / 381`.
- Exp006 native z-score e5/d4 epoch100 is secondary single-GPU context:
  Dice `0.3241`, hit rate `0.7612`, `290 / 381`.
- Exp017 phase 1 completed with best val200 at epoch50: Dice `0.3347`, hit
  rate `0.7769`, `296 / 381`; epoch100 was Dice `0.3328`, hit rate `0.7690`,
  `293 / 381`.
- Exp007 phase 2 is the fair continuation comparator. It used weights-only
  initialization from Exp007 epoch100 with a fresh optimizer/LR horizon. Its
  best val200 checkpoint was relative epoch50 / absolute epoch150: Dice
  `0.3460`, hit rate `0.7769`, `296 / 381`.

## Scope

In scope:

- Add canonical Exp017 config and launchers.
- Generate a deterministic `40,000`-event v123 positive-crop schedule from the
  iso07 cache with seed `20260723`.
- Train with DDP world size `4`, local batch `1`, no gradient accumulation.
- Pause training at epochs `25/50/75/100`, run fixed val200 on all four GPUs,
  then resume from full optimizer/scaler/RNG/checkpoint state.
- Restore validation predictions from iso07 cached crop space to native label
  space before evaluation.
- Summarize final and intermediate val200 metrics against Exp007.
- Run Exp017 phase 2 inside the same experiment ID from the phase-1 epoch100
  checkpoint, not from the epoch50 best checkpoint.

Out of scope:

- Running any `1.0 mm` or z-score iso07 variant.
- Initializing from Exp007 or another fine-tuned checkpoint.
- Starting phase 2 from Exp017 epoch50, because that would not mirror Exp007.
- Changing model architecture, prompt policy, loss, LR, patch size, or
  threshold.
- Saving probability maps by default.
- Launching full finetuning before the smoke gates pass.

## Inputs And Paths

- Canonical config:
  `configs/experiments/017_voxtell_iso07_hu_ddp_bs4_update_matched.json`
- Metadata: `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- CT root: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Segmentation root: `/data/hengjie/datasets/rexgroundingct/segmentations`
- Runtime experiment directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched`
- Repo-local summary directory:
  `experiments/017_voxtell_iso07_hu_ddp_bs4_update_matched`
- Iso07 cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1`
- Fixed val200:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`

## Data And Preprocessing Contract

- Changed preprocessing variable: Exp016 `crop_clip1024_linear_iso07_v1`.
- CT preprocessing: crop native CT to nonzero, clip HU to `[-1024, 1024]`,
  divide by `1024`, trilinear resample to `0.7 mm` isotropic.
- Label preprocessing: same crop, nearest-exact resampling to `0.7 mm`, with
  foreground-center fallback if a mask disappears.
- Image padding: `-1`; target padding: `0`.
- Model tensor shape remains `192^3`; physical patch FOV is about `134.4 mm`
  per axis.
- Interpretation caveat: this is a geometry plus normalization experiment, not
  a pure HU-normalization ablation.

## Method

Use public VoxTell v1.1 initialization and reset optimizer state at experiment
start. DDP uses one process per GPU:

```text
world_size=4
per_gpu_batch_size=1
grad_accum=1
effective_global_batch=4
```

Training keeps the Exp007 v123 e5/d4 recipe:

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

The DDP schedule consumption rule is:

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

Host launcher:

```bash
IMAGE=rexgroundingct-voxtell:cu126 \
RUN_SMOKE_ONLY=1 \
bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/run_017_iso07_hu_ddp_bs4_docker.sh
```

Full launch after smoke gates pass:

```bash
IMAGE=rexgroundingct-voxtell:cu126 \
RUN_SMOKE_ONLY=0 \
RUN_FULL=1 \
bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/run_017_iso07_hu_ddp_bs4_docker.sh
```

## Phase-2 Continuation Protocol

Source:

- run group: `exp017_iso07_hu_ddp_bs4_20260815T092119Z`;
- source run dir:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_iso07_hu_ddp_bs4_20260815T092119Z/ddp_bs4`;
- source checkpoint:
  `checkpoints/checkpoint_update_010000.pth`;
- checkpoint SHA256:
  `f11d238ffeaf9d96fe6ba7f81f6dd134953b18b0eaaae3f5404176aa0cec6cc4`;
- expected checkpoint `global_update`: `10000`.

Continuation semantics:

- first segment passes `--init-checkpoint` and loads network weights only;
- optimizer, AMP scaler, LR schedule, RNG-controlled run state, and global
  update counter are reset;
- later segments resume from continuation checkpoints with `--resume-checkpoint`;
- relative epochs `25/50/75/100` map to absolute epochs `125/150/175/200`.

Schedule:

- generate an 80,000-event deterministic iso07 DDP stream with seed
  `20260723`;
- verify events `0..39999` match the completed phase-1 40,000-event schedule;
- slice events `40000..79999`, rewrite `event_index` to `0..39999`, and keep
  the original index as `source_event_index`.

Continuation smoke:

```bash
IMAGE=rexgroundingct-voxtell:cu126 \
RUN_SMOKE_ONLY=1 \
bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/run_017_iso07_hu_ddp_bs4_continue100_from_epoch100_docker.sh
```

Continuation full launch after smoke gates pass:

```bash
IMAGE=rexgroundingct-voxtell:cu126 \
RUN_SMOKE_ONLY=0 \
RUN_FULL=1 \
bash /home/hengjie/code_sync/rexgroundingct/scripts/rexgroundingct/run_017_iso07_hu_ddp_bs4_continue100_from_epoch100_docker.sh
```

## Smoke Gate

Before full launch:

- static shell/Python syntax checks;
- cache manifest and `.complete` checks;
- schedule generation: `40,000` events from iso07 cache;
- sample-test with scheduled events;
- DDP train smoke update `0 -> 2`;
- DDP resume smoke update `2 -> 4`;
- one-case iso07 cached inference smoke from the materialized smoke checkpoint;
- four-case, four-shard eval smoke using the same restored-cache inference path.

## Success Criteria

- DDP run completes `10,000` optimizer updates.
- Val200 summaries exist at epochs `25/50/75/100`.
- Segment resumes preserve model, optimizer, AMP scaler, global update, schedule
  cursor, and RNG state.
- No selected positive target is empty.
- Iso07 validation predictions export to the released label shape.
- Runtime report compares Dice, hit rate, hits/targets, loss, throughput, GPU
  memory, and Exp007 references.

## Verification

Run before handoff:

```bash
bash -n scripts/rexgroundingct/run_017_iso07_hu_ddp_bs4.sh
bash -n scripts/rexgroundingct/run_017_iso07_hu_ddp_bs4_docker.sh
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/rexgroundingct/run_voxtell_val_inference.py scripts/rexgroundingct/train_text_conditioned_voxtell.py scripts/rexgroundingct/summarize_007_results.py
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/rexgroundingct/summarize_007_continuation_results.py
python scripts/rexgroundingct/check_experiment_consistency.py --experiments 017_voxtell_iso07_hu_ddp_bs4_update_matched
git diff --check
```

## Closeout Plan

After runtime completion:

- write `/mnt/shengdata1/.../reports/iso07_hu_ddp_bs4_report.md`;
- write `/mnt/shengdata1/.../reports/iso07_hu_ddp_bs4_summary.json`;
- sync repo-local metrics/report with `sync_experiment_index.py`;
- update `docs/current_status.md` only if the result changes the next
  finetuning decision.
