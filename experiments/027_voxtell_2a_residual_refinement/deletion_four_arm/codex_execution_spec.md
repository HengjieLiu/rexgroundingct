# Authorized four-arm deletion implementation and execution

Date: 2026-09-10. User authorization: "code and do this" with a similar live
dashboard. The accepted scientific method is in `../deletion_four_arm_plan.md`.

## Scope and paths

Implement, verify and run the 20-epoch stage only: 2,000 updates per arm,
100 updates per epoch, evaluation barriers at 100/500/1000/2000. Stop with
`pending_user_review`. Do not continue to 100 epochs, add FN recovery, resume
the stopped residual runs or overwrite the diagnostic.

Canonical config: `configs/experiments/027_deletion_four_arm.json`.
Runtime: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/027_voxtell_2a_residual_refinement/deletion_four_arm_20ep`.
Reuse existing full-run `prepared.json` and all 864 verified cache cases /
1,189 findings. Original Exp027 config/cache provenance remains immutable.

## Preparation and sampling

CPU preparation validates frozen source hashes, old full-cache coverage,
receipt contracts, and file sizes/mtimes against hash-verified receipts. Changed
files fail the gate; never regenerate logits implicitly. Copy references and
receipt hashes into the new runtime without changing historical provenance.
Four disjoint CPU workers derive exact TP/FP counts for each eligible native
192³ / 50%-overlap tile. Save per-case resumable indexes, then gate all trainers
on complete exact finding coverage and verified inputs. No GPU allocation in
preparation or the reporter.

Materialize immutable 2,000-update schedules in fixed arm order: train;
train+A 50/50; A; A+B uniformly. Uniform eligible findings within source.
Per 100 updates use 50 TP-containing, 25 FP-containing and25 uniform-eligible
requested branches, shuffled deterministically; run 2 has exactly 50 train /
50 A source events per epoch. Branches overlap. Log same-finding fallback,
actual TP/FP/GT-empty composition, and unique findings/windows. Findings without
base-positive voxels are explicitly bypassed for training and retained in eval.
No augmentation; cached native CT normalization/orientation unchanged.

## Model, optimizer and inference

Reuse tested diagnostic architecture and loss: 16 channels, GN/GELU, feature
block dilations 1/2/4/1, zero head weights and removal bias logit(0.05).
Initialize all four from one newly generated pristine CPU state. Loss is
mean_FP BCE(remove,1) + 2 × mean_TP BCE(remove,0), separately normalized over valid
base-positive groups. Empty groups contribute zero. No Dice/L1/addition.
FP32, no TF32, AdamW LR 1e-4 / WD 1e-4 / betas (0.9, 0.999) / eps 1e-8, norm clip 1.0,
batch size 1, no accumulation, constant LR. Never load the diagnostic trained model.

Refine the full native extent. Gaussian-blend removal probabilities in FP32;
empty-base tiles vote zero action with weights retained. Apply threshold only
after blending and restrict deletion to base foreground. Restore masks to
original CT geometry and verify the cache baseline. Main threshold 0.90;
record 0.50/0.80/0.90/0.95/0.99/1.00 from the same pass. Base-logit comparator
threshold0.1910400390625 is fixed from the completed diagnostic; also record
0/0.5/1/2/4 logit thresholds as labeled diagnostics, without auto selection.
Store sparse base-positive scores/indices externally for reproducible review.

## Recovery, barriers and failures

One coordinator owns the group. Each worker owns one run/evaluation lock.
Save full model/optimizer/Python/NumPy/Torch/CUDA RNG state, immutable context,
sampling cursor and committed journal at every epoch and graceful interruption.
Use atomic checkpoint writes and immutable checkpoint names. Resume only with
an explicit resume flag; reconcile extra journal records to the saved cursor
while archiving recovery evidence. Completed evaluations are idempotent and
partial evaluations resume from verified per-finding records.

Trainers cannot pass a milestone until all four full evaluations complete.
OOM, non-finite values or failed integrity checks stop the group, preserve
evidence and never change precision/tile size. Individual harm flags do not
automatically stop or rank a model. Stop at epoch 20 for user review.

## Local live dashboard

One independent CPU writer polls every second and refreshes training within
five seconds, and promptly on phase/finding/evaluation/failure changes. GPU
workers append each update and never wait for plotting. Preserve JSONL and
periodic JSON/CSV snapshots; all dashboard files replace atomically.

Use a 4×3 figure to keep both preservation and FP removal visible: training
loss / GT-containing patch Dice / GT-empty patch FP removal, followed by
A/B/full Dice, TP retention, and FP removal. Raw and 100-update trailing means
are distinct. Four fixed colors/order; cached-base lines; hollow provisional
points with matching partial-subset baseline. No absent metric becomes zero.
Markdown shows phase/update/epoch, source exposure, loss components, elapsed
time, memory, unique findings/windows, finding/tile progress, A/B/full scores,
hit rates and per-finding damage. Flag retention <95%, severe <80%, with counts
and linked complete failure details. Indicate update time and viewer reload.

## Verification and launch

CPU Docker tests (GPUs disabled): exact eligible tile counts, empty branches,
pool membership/mixes, deterministic schedules, cache integrity/coverage gates,
common initialization/FP32, losses, subset-preserving inference and orientation,
checkpoint/RNG replay, journal recovery, milestone barriers, complete A+B
aggregation, partial/final reporting, fixed order and single-writer locking.
Render and inspect labeled synthetic fixture. Run canonical repo/whitespace
checks. Then launch the authorized four-GPU orchestrator, verify live progress
and record its container identity and exact command.

Only small specs/reports/configs remain in Git. Preserve all large arrays,
indexes, schedules, checkpoints, logs and figures under the external runtime.
