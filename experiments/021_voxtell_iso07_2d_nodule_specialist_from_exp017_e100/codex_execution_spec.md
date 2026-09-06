---
created: 2026-09-03
updated: 2026-09-03
status: implementation
experiment_id: "021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100"
---

# Codex Execution Spec: Exp021 Nodule-Only Positive Specialist

## Objective

Determine whether a category-2d specialist improves pulmonary nodule/mass
segmentation when only category `2d` targets receive positive supervision and
all other findings/background are trained as nodule-negative examples.

## Prior Evidence

- Exp017 provides the retained 0.7 mm HU source checkpoint at relative epoch
  100, global update 10,000, SHA-256
  `f11d238ffeaf9d96fe6ba7f81f6dd134953b18b0eaaae3f5404176aa0cec6cc4`.
- Exp017 uses the same completed iso07 cache, DDP world size 4, local batch 1,
  gradient accumulation 1, and effective global batch 4.
- Exp018 defines official category `2d` as pulmonary nodules/masses and fixes
  the validation nodule census at 132 findings in 119 cases.
- The train split has 1,305 nodule-positive cases, 1,743 nodule findings, and
  1,687 cases with no category-2d finding.

## Scope

In scope:

- A fresh Exp021 runtime group initialized from Exp017 epoch-100 weights only.
- The existing `crop_clip1024_linear_iso07_v1` cache and Exp017 training recipe.
- A deterministic schedule with only category-2d positive targets.
- Full fixed-val200 evaluation at epochs 25, 50, 75, and 100.
- Human-readable overall, per-category, and nodule-specific reports.

Out of scope:

- Any change to CT preprocessing, model architecture, optimizer, or loss.
- Positive supervision for categories other than `2d`.
- Per-10-epoch or target-only evaluation passes.
- Reusing the Exp017 optimizer, scheduler, scaler, update counter, or RNG state.

## Inputs And Paths

- Canonical config:
  `configs/experiments/021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100.json`
- Metadata: `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- CT root: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Segmentation root: `/data/hengjie/datasets/rexgroundingct/segmentations`
- Iso07 cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_clip1024_linear_iso07_v1`
- Source checkpoint:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_iso07_hu_ddp_bs4_20260815T092119Z/ddp_bs4/checkpoints/checkpoint_update_010000.pth`
- Runtime root:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100`
- Fixed val200:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`

## Data And Preprocessing Contract

- Keep Exp017 `crop_clip1024_linear_iso07_v1` unchanged.
- Keep 192³ patches, image padding -1, target padding 0, and nearest-exact
  cached target geometry.
- Positive event: one real category-2d target; two non-2d prompts with
  `target_index=None` and zero masks. Materialized patch starts contain the
  selected positive point.
- Negative event: choose a case with no category-2d annotation; use one 2d
  query and two non-2d prompts, all with zero targets and random patch starts.
- This uses the released annotations as the nodule-negative definition; it is
  a weak-negative assumption if an unlisted nodule exists in a case.

## Method

Use 40,000 global events for 100 epochs × 100 optimizer updates × DDP global
batch 4. Each 400-event epoch block contains 300 category-2d positive events
and 100 nodule-negative-case events, shuffled deterministically.

Use the Exp017 e5/d4 v123 recipe:

- DDP world size 4, per-GPU batch 1, gradient accumulation 1;
- SGD Nesterov, momentum 0.99, weight decay 3e-5;
- encoder/decoder learning rates 1e-5 / 1e-4;
- 100 warmup updates and poly decay with power 0.9 over 10,000 updates;
- gradient clipping 12 and the existing weighted BCE/deep-supervision terms.

The first segment loads only source network weights with `--init-checkpoint`.
Subsequent segments resume full Exp021 checkpoints. Segment barriers are:

```text
updates 0 -> 2500, evaluate val200 at epoch 25
updates 2500 -> 5000, evaluate val200 at epoch 50
updates 5000 -> 7500, evaluate val200 at epoch 75
updates 7500 -> 10000, evaluate val200 at epoch 100
```

## Evaluation Execution Contract

- Milestones: epochs 25, 50, 75, and 100 only.
- Dataset: fixed val200, 200 cases and 381 findings, SHA-256
  `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`.
- Resource policy: synchronous barrier; training stops and releases all four
  GPUs before sharded validation starts.
- Every report includes overall metrics, all official category metrics, and
  category-2d nodule metrics (132 findings / 119 cases; hit threshold Dice
  >=0.1). The source Exp017 epoch-100 evaluation is the initial reference.
- Evaluation threshold is 0.5. Predictions are restored through the existing
  iso07 cached-crop path before ReXRank scoring.
- Evaluation is idempotent with a per-milestone lock and complete-case check.

## Smoke Gate

Run Python/shell syntax checks, validate the cache/checkpoint/val200 hashes,
generate and audit the schedule, run scheduled sample-test, run DDP updates
0→2→4 with a checkpoint resume, run one-case cached inference, and run a
four-case four-shard evaluator smoke. Full training is blocked until all pass.

## Success Criteria

- 10,000 optimizer updates complete with immutable checkpoints at 2,500-update
  intervals.
- All four full val200 milestone evaluations complete.
- Nodule metrics are computed from finding-level raw evaluator records rather
  than the overall evaluator aggregate.
- Report category counts sum to the fixed val200 census and show the initial
  Exp017 nodule reference.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/rexgroundingct/prepare_021_nodule_schedule.py scripts/rexgroundingct/summarize_021_nodule_results.py
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/test_prepare_021_nodule_schedule.py
bash -n scripts/rexgroundingct/run_021_iso07_2d_nodule_specialist.sh
```

The runtime smoke and full run execute inside the pinned VoxTell container.

## Closeout Plan

Keep checkpoints, predictions, logs, and raw evaluator JSON on `/mnt/shengdata1`.
After the run, copy only the small human report and metrics/provenance summary
to this repo-local experiment folder and refresh the experiment index.
