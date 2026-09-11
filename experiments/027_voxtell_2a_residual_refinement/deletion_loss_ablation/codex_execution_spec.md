# Exp027: four-loss comparison fitted on A

Status: implementation authorized, followed by the four-GPU launch after checks.
The user approved 2,000 updates, finding-aware patch Dice and display thresholds
0.50/0.90. Completion status is `pending_user_review`.

## Scientific contract

Fixed order: `bce_tp2`, `bce_tp1`, `dice_bce_tp2`, `dice_bce_tp1`.
For valid base-positive patch voxels, F is mean softplus(-r) on annotated FP;
K is mean softplus(r) on base TP. Absent groups contribute differentiable zero.
The four losses are F+2K; F+K; D+0.25(F+2K)/3; D+0.25(F+K)/2.
Denominators remain fixed when a group is absent. Comparisons with Dice arms
include BCE rescaling as well as adding Dice; this is a comparison of recipes.

For full-finding base TP T0, base FP F0, total GT G0, and soft TP/FP removals a/b
inside the sampled patch, D=1-[2(T0-a)+1e-6]/[T0+F0+G0-a-b+1e-6].
Predictions outside the patch are held at the base prediction. This is a
single-patch approximation, not exact blended whole-volume Dice. Include all
GT, including base false negatives. No removal threshold enters the loss.
Log F, K and D for every arm, including diagnostic-only D in BCE controls.

All four arms use all 35 A findings, never B gradients. B contains 34 findings;
A/B patient membership and original finding IDs/prompts remain fixed. A is
fitting evidence; B is held-out development evidence after repeated inspection.
No 99%/95% TP-retention acceptance gates. Dice is primary; TP/hit harm remains
visible. No automatic threshold/checkpoint selection or ranking.

## Inputs, preprocessing and initialization

Reuse only the 63 validation CT caches / 69 findings from the frozen Exp007
continuation-50 checkpoint (SHA prefix a499ad1c). Validate arrays, prompt order,
full GT counts, native/original geometry and strict cached baselines.
Preprocessing is cached-equivalent: native cropped-volume z-score, no resampling
or augmentation, valid padding mask, base logits stored under their original
[-30,30] clipping and dtype contract. Never generate logits implicitly.

Replay the prior `deletion_four_arm_20ep/runs/run3_val_a/schedule.json` events
identically across all arms, with digest
ae7c29f411864dde01da288029a44e069f6aed4931f14a72886476b82c679672.
Keep uniform A finding selection and the exact 50 TP / 25 FP / 25 eligible
requested branches per 100 updates with recorded same-finding fallbacks.
Copy the pristine historical model (weight hash
ff8d94ec3648a9ef24a0a2d4865c0590782da60a522b7f4b701cf57748cbba73),
zero head weights and removal probability 0.05. Fresh optimizer; common RNG seed
20263014 reproduces the original A-only worker's starting RNG streams.

## Training and recovery

FP32, no TF32, 192³, batch size 1, unchanged 16-channel GN residual architecture;
AdamW LR1e-4, weight decay1e-4, betas(0.9,0.999), eps1e-8, clip norm1.
Constant LR, no AMP, no accumulation. Exactly 2,000 optimizer updates per arm,
100 updates/epoch. Save full state every epoch and on graceful interruption.
Checkpoints bind arm/loss, config/code, cache proof, initial weights, schedule,
optimizer, RNG and cursor. Resume archives journal tails beyond saved state,
replays those events, and does not duplicate durable optimizer updates.

At updates100/500/1000/2000 all trainers stop; evaluate four models concurrently
on their assigned GPUs and wait for all summaries before continuing. Evaluate
all 69 findings once, Gaussian-blending probabilities in FP32 over native192³
tiles at50% overlap, restoring original geometry. Output is a subset of base
foreground. Stop the group on invalid results/OOM; no automatic method changes.

## Reporting and execution

Append each training update immediately; snapshot JSON/CSV every epoch. One
independent CPU dashboard writer polls each second and refreshes within5 seconds
or after phase/evaluation changes. Publish partial finding scores immediately,
with matching-subset baselines/counts; finalize each arm independently.
The 3×3 figure contains total/F/K losses; D/A-Dice0.5/B-Dice0.5; gradient norm/
A-Dice0.9/B-Dice0.9. Distinguish raw and100-update smoothed training curves.
Show exposure, progress, time/memory, failures and pending values. Total losses
have different definitions. Retain A/B/full Dice, changes, hits gained/lost,
FP/TP tradeoffs, per-finding damage and descriptive A/B improvement gaps.

Six live thresholds:0.5/0.8/0.9/0.95/0.99/1. CPU analysis performs dense0:0.005:1
sweeps from saved FP32 scores without more inference, separately from plotting
and GPU barriers; save per-finding/aggregate CSV/JSON and threshold figures.
Retain existing base-logit controls. All artifacts stay under external
`deletion_loss_ablation_a_20ep`; config is
`configs/experiments/027_deletion_loss_ablation_a.json`.

Authorized launch after CPU tests, dry run and repository checks:

```bash
START_GPU_WORK=1 DETACH=1 bash scripts/rexgroundingct/run_027_deletion_loss_ablation_host.sh orchestrate --gpus 0 1 2 3
```

CLI also provides prepare/train/evaluate/report/analyze/audit and explicit
resume. Preparation and dry run cannot start GPU work. Preserve all previous
experiment runtimes. No automatic continuation after update2000.

## Verification and closeout

Use existing Docker image without GPUs for CPU tests: historical BCE and all
four formulas/gradients; explicit full-finding Dice equivalence; FP/TP signs,
empty groups, base-TP zero, padding and base FNs; schedule/split/initialization;
cache integrity; tiling/orientation/subset/baseline; threshold ties and A+B
recomposition; resumable journals/checkpoints/evaluations; independent partial
dashboard/failures/order/locks and launch opt-in. Inspect a labeled synthetic
dashboard. Run workflow/whitespace checks. At completion verify8,000 updates,
16 complete69-finding evaluations, all dense analyses, and compare the BCE
reference with historical A-only checkpoint hashes/metrics. Investigate any
discrepancy before attributing improvements. Final status pending_user_review.
