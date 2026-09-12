# Exp027: four losses fitted on original training data

Status: implementation and launch authorized by the user. This specification
is recorded before coding. Stop after the accepted schedule at
`pending_user_review`; no automatic continuation, ranking or promotion.

## Scientific contract

Fixed order: `bce_tp2`, `bce_tp1`, `dice_bce_tp2`, `dice_bce_tp1`.
Keep F+2K, F+K, D+0.25(F+2K)/3, D+0.25(F+K)/2 respectively. F/K are separate
FP-deletion/TP-preservation BCE means on valid base-positive voxels. Empty
groups contribute differentiable zero with unchanged normalization. D is the
accepted whole-finding single-patch edit surrogate, with all GT including base
false negatives and predictions outside the patch held at base. Log F, K, D
for every arm. Dice recipes also rescale BCE. No threshold enters training.

Every arm uses original train only: 1,120 findings across 801 CTs / 741 patients.
Exclude only `train_7777_a_2.nii.gz::4`, which has no base-positive tiles. Uniform
sampling therefore draws from 1,119 findings. A=35 findings and B=34 findings
remain patient-separated, excluded from these refiners' gradients, and are
repeatedly inspected development cohorts. Full=A+B, not mixed training exposure.

## Inputs, initialization and schedule

Verify all 864 CTs / 1,189 findings before launching any trainer. Check original
identifiers/prompts, native/original geometry, full-finding counts, cache arrays,
source checkpoint and index hashes. Reuse frozen Exp007 continuation-50 inputs
(checkpoint SHA prefix a499ad1c), preserving clipped [-30,30] base logits and
historical dtype provenance. No VoxTell inference or implicit cache generation.
Missing, altered or invalid required files stop preparation. Use four disjoint
CPU verification shards; share the verified arrays and preserve old receipts.

Preprocessing remains cached-equivalent: native cropped-volume z-score, no
resampling or augmentation, existing CT orientation transforms, valid padding
mask. No change to annotation interpretation or partial-label masking.

Create one immutable 10,000-event schedule using historical `run1_train` with
seed 20260911. Its 2,000-event prefix must have digest
`cfe5c1bc3a450b4a2e4355c97e4ebeac605d9e8dab5a2c5f1e5b56856177e70f`.
Keep exact 50 TP / 25 FP / 25 eligible requested branches per 100 updates,
uniform finding draws and recorded same-finding fallbacks. All four arms replay
identical events. Read-only exploration verified all 1,119 eligible findings
are sampled over 10,000 events (933 in the first 2,000); 501 fallbacks occur.

Copy historical pristine weights (model digest
`ff8d94ec3648a9ef24a0a2d4865c0590782da60a522b7f4b701cf57748cbba73`),
zero head weights, initial removal probability 0.05. Fresh optimizer; common
training RNG seed 20261612 reproduces historical train-only initialization.

## Training, barriers and recovery

10,000 updates per arm; 100 reporting epochs × 100 updates. Epoch is not a
complete dataset pass. Evaluate at 100, 500, 1,000, 2,000, 3,000, 4,000, 5,000,
6,000, 7,000, 8,000, 9,000, 10,000. Keep FP32/no TF32, 192³ tiles, batch 1,
16-channel GN architecture, AdamW LR1e-4/weight decay1e-4/betas(0.9,0.999)/eps1e-8,
gradient clipping1.0, constant LR, no AMP or accumulation.

Save immutable full-state checkpoints every 100 updates and on graceful
interruption: model, optimizer, RNG, history, sampling cursor, loss identity,
initialization, schedule, input/config/code provenance. Resume reconciles tails
without duplicate durable updates, shortened schedules or missing evaluations.
All trainers stop at each evaluation barrier; four evaluators run concurrently;
resume only after all evaluations and applicable reference checks pass.
OOM/non-finite values/integrity failures stop the group and preserve evidence.

Every evaluator processes 69 findings once using native 192³/50%-overlap tiles,
FP32 Gaussian probability blending and original-geometry restoration. Final
foreground is a subset of base; delete only score > threshold. Derive A/B/full
metrics from the same records. Baseline Dice A=.3394159226078028,
B=.334950148355145, full=.3372153961644642. Preserve existing hits, FP/TP removal,
precision/recall and per-finding damage diagnostics; no retention veto gates.
At the first four barriers compare BCE TP2 weights and per-finding threshold
metrics with historical `run1_train`; investigate any discrepancy before
continuing. Compare all arms with A-only results at matched updates <=2,000,
labeling subsequent updates as additional training exposure.

## Reporting, interfaces and compatibility

Canonical config `configs/experiments/027_deletion_loss_ablation_train.json`;
separate external runtime `deletion_loss_ablation_train_100ep`. Separate
train-only adapters reuse existing pure loss/model/geometry/metric utilities;
do not alter code bound to historical runtimes. Provide orchestrate, prepare,
train, evaluate, report, analyze, audit, dry run and explicit --resume.

Append each update immediately; snapshot CSV/JSON each epoch. One independent
CPU board writer polls each second and refreshes curves within about five
seconds, promptly on phase/evaluation/failure events. Publish per-finding
provisional A/B/full metrics and matching baselines, replacing each model's
partial results immediately when its evaluation finishes. Show raw/smoothed
curves, fixed colors/order, source, sampling coverage, update/epoch, memory,
time, findings/tiles, pending values, failures and timestamps.

Keep 3×3 panels: total/F/K; D/A-Dice0.50/B-Dice0.50;
gradient norm/A-Dice0.90/B-Dice0.90. Display total losses as different objectives,
not quantities to rank. Both A and B are held-out development here. Six live
thresholds .5/.8/.9/.95/.99/1 and dense 0:0.005:1 sweeps with baseline controls,
per-finding/aggregate CSV/JSON, combined full tables and separate tradeoff
figures. An independent CPU analyzer consumes saved FP32 scores without extra
inference or plotting on GPU workers. Use atomic outputs and stable README
links. No threshold/checkpoint/model selection.

## Verification and launch

Run CPU PyTorch tests in existing VoxTell Docker with GPUs disabled: complete
train+val gate, patient separation, exclusions, exact schedules/prefix/weights,
loss formulas, empty groups/padding/FNs, geometry/blending/deletion/thresholds,
A+B recomposition, all 12 barriers, recovery beyond update2,000, provisional
publication/exposure/pending/failures/single writer and dry-run GPU protection.
Inspect labeled synthetic dashboard and run workflow/whitespace checks.

Then execute the authorized command:

```bash
START_GPU_WORK=1 DETACH=1 bash scripts/rexgroundingct/run_027_deletion_loss_ablation_train_host.sh orchestrate --gpus 0 1 2 3
```

Recovery uses the same command with `--resume` and a fresh container name when
the previous stopped container is retained. Completion verifies 40,000 total
updates, 48 complete 69-finding evaluations, 48 dense analyses, retained epoch
checkpoints and historical reference reproduction, then `pending_user_review`.
