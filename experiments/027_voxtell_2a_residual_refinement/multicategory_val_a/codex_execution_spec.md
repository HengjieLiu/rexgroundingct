---
created: 2026-09-11
updated: 2026-09-11
status: authorized
---

# A-only category deletion training and the preceding run's update-2000 stop

## Accepted intent and stop boundary

The user approved four fresh 2b/2c/2d/2e A-only runs, 20 epochs of 100 updates,
the unchanged Dice + auxiliary BCE TP2 recipe, and separate A/B curves in the
four category panels. The final stop correction supersedes the earlier proposal
to interrupt immediately: every current original-training arm must finish exactly
2000 updates, and its update-2000 validation must not start.

At implementation start, hold only the old coordinator, allowing its remaining
trainers to finish their already bounded update-2000 commands. Verify all four
2000-update full-state checkpoints and journals; confirm no update-2000 eval was
started. Then terminate the held coordinator safely, mark the old runtime
stopped_by_user and record its actual stop/checkpoint evidence. Retain completed
evaluations at 100/500/1000. Do not alter old configs, source hashes or artifacts.
New GPU work requires this verified stop receipt and no live old workers.

## Data and preprocessing

Reuse the completed frozen Exp007 a499ad1c cache_2bcde_v1. Verify only required
validation case proofs and tile indexes against its immutable verified inventory:
177 CTs / 252 findings, with category full finding counts 49/60/132/11 and CT
counts 40/53/119/11. No original-training inputs or VoxTell inference are needed.

Fit the existing A findings only: 25/31/61/8, all deletion-eligible. B contains
24/29/71/3 findings and remains excluded from gradients. Retain the three
base-empty 2d B findings in evaluation. Preserve the existing patient split,
original finding IDs, prompts, channels and validation source labels. The old
train_2936 original-training exclusion is irrelevant to this A-only membership;
all original-training findings are unused. Verify no A/B patient overlap.

Preprocessing is cached-equivalent to the base: native cropped-volume z-score,
no resampling or augmentation, existing orientation/restoration transforms,
192-cubed patches, batch 1 and valid-voxel padding exclusion. Base clipping and
storage provenance remain unchanged. Each epoch uniformly samples findings and
shuffles 50 TP/25 FP/25 eligible branches with recorded same-finding fallbacks.
Materialize 2000 immutable events per category, schedule seeds 20260911-20260914.
Events identify source A, never original train. All A findings must be visited.

## Training and evaluation

Reuse the same architecture, pristine weights ff8d94ec... with zero head weights
and initial removal probability 0.05, fresh optimizers and RNG seed 20263014.
Use FP32, no TF32, AdamW LR 1e-4, WD 1e-4, betas 0.9/0.999, epsilon 1e-8,
constant LR, clipping 1.0, no accumulation. Objective: D + 0.25(F + 2K)/3.
F/K remain separate group-mean BCE; D remains the full-finding hypothetical
single-patch edit Dice, including base FN in total GT. Empty groups contribute
differentiable zero, smoothing is 1e-6, and thresholds never enter training.

Assign one category to each GPU in fixed 2b/2c/2d/2e order. Save full-state
checkpoints every 100 updates and on graceful interruption, including optimizer,
RNG, category/loss/source identity, sampling cursor and provenance. Use synchronous
evaluation barriers at 100/500/1000/2000. Evaluate every category's full finding
set once and derive A/B/full from the same records. Preserve full-volume native
inference, base-positive tile gating, 50% overlap, FP32 Gaussian probability
blending and deletion-only output in original geometry.

Verify full baselines 0.357030/0.415254/0.394980/0.410408 and each A/B baseline.
Retain six live thresholds 0.5/0.8/0.9/0.95/0.99/1, both display thresholds
0.50/0.90, frozen-base controls and 201 CPU thresholds spaced by 0.005 from saved
FP32 scores. Retain per-finding Dice, hits, precision/recall, TP loss, FP removal,
harm counts, timings and gradients. Stop on OOM, non-finite or integrity failure;
preserve evidence with no automatic method changes. Resume must not duplicate
updates, alter schedules or repeat completed findings.

## Dashboard and execution interface

Use one independent CPU writer, polling each second and refreshing within about
five seconds, immediate append-only training rows and periodic/checkpoint JSON/CSV
snapshots. Keep an atomic local Markdown dashboard and 3x3 figure: total/F/K on
row 1; D/2b validation/2c validation on row 2; gradient norm/2d validation/2e
validation on row 3. Retain consistent category colors and raw/trailing-100 means.

Each category panel plots A and B separately at 0.50 and 0.90: A dashed/hollow,
B solid/filled; circles 0.50, squares 0.90. Show distinct A/B baseline lines.
There is no full A+B curve; list completed A/B/full metrics and deltas in tables,
with A fitted, B held-out development and full mixed exposure labels.

During evaluation, show pending with completed/expected CT and finding counts
and current tile progress. Count a CT complete only after all its category
findings are done. Publish both A/B curves and all numeric result tables only
after that category's full evaluation is verified complete, independently of
other categories. Retain preceding completed points and never display partial
scores. Display errors, elapsed time, memory and refresh timestamp.

Separate A-only adapters and entrypoint preserve the old code/inventory hashes.
Provide prepare/train/evaluate/report/analyze/audit/dry-run/explicit resume under
run_027_deletion_categories_val_a_host.sh. Runtime is the separate external
deletion_categories_bcde_val_a_20ep folder; explicit START_GPU_WORK=1 is required.
The authorized detached launch assigns GPUs 0 1 2 3 after checks and old-stop gate.

## Verification and completion

CPU tests use the existing VoxTell Docker image with GPUs disabled. Cover A-only
membership and all A findings sampled, B exclusion and patient separation, source
identity, unchanged formulas and gradients, baseline/geometry/deletion identity,
empty B predictions, cache corruption, interruption recovery, four variable-size
evaluation barriers, threshold ties/dense sweeps, separate A/B curves with own
baselines, absent/partial/failed results, CT progress with multiple findings per
CT, independent completion publication, single writer, dry-run and old-stop gate.
Inspect labeled synthetic Markdown/PNG; run workflow and whitespace checks.

After launch verify common initial weights, finite initial updates, FP32 and live
refresh. Completion requires 8000 updates, 16 complete category evaluations and
16 dense analyses. Stop at pending_user_review with no ranking, threshold
selection, checkpoint promotion or automatic continuation.
