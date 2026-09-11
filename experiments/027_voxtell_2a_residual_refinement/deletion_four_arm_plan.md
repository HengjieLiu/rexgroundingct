# Proposed four-arm deletion training plan

Date: 2026-09-10. Status: accepted; user authorized implementation and the
20-epoch four-arm run, with a shared live dashboard. See
[the execution specification](deletion_four_arm/codex_execution_spec.md).

## Evidence and objective

The 200-update diagnostic on eight selected A findings improved full-volume
Dice from0.327060 to0.371671 at a conservative patch-calibrated threshold. It
retained98.813% of pooled TP but removed64% of TP from one small finding. At
threshold0.5 it retained only71.145% of TP and did not improve mean Dice.
Matched-retention analysis supports useful error discrimination beyond simple
base-threshold suppression. This motivates testing the same deletion method
across all four data sources, with better window coverage and explicit tracking
of individual damage. It does not establish performance on unseen findings.

## Four arms and exposure

| Arm | Finding pool | Sampling by source | Available findings |
| --- | --- | --- | ---: |
| 1 | Original train | Train only | 1,120 |
| 2 | Original train + A | 50% train / 50% A | 1,155 |
| 3 | A | A only | 35 |
| 4 | A+B | Uniform findings over A+B | 69 |

Keep the original patient split, seed20260909, prompts and finding identifiers.
Counts describe the available pool. Findings without any base-positive voxel
have no deletion supervision; report their bypass explicitly and retain them
in evaluation. Do not secretly substitute another source when a finding has
no eligible tile.

A is development data because the diagnostic and design decisions used A.
Run1 A findings are held out from refiner gradients, but A is not pristine
method-development evidence. B remains held out from refiner gradients in
arms1–3; arm4 B is in-sample. Do not use arm4 B performance to tune shared
thresholds or hyperparameters for arms1–3. Full A+B summaries for arms2–3 mix
exposures. Arm4 is an empirical fitting-capacity diagnostic, not a guaranteed
upper bound. Preserve fixed run order and user checkpoint/ranking decisions.

## Common method

Use the diagnostic's 16-channel convolutional backbone, four residual feature
blocks and a voxelwise remove/keep output, with fresh common initial weights.
Head weights zero, initial removal probability0.05; no diagnostic checkpoint
initialization. Additions remain disabled. Keep CT+base-logit inputs, native
cropped z-score normalization, no resampling, 192³ windows and the frozen
Exp007 cont e050 / absolute e150 strict cache. Reuse existing arrays; no new
VoxTell exports are needed. Verify cache receipts/hashes before launch.

FP32 for training/evaluation, no TF32, batch1, no gradient accumulation.
AdamW LR1e-4, weight decay1e-4, betas0.9/0.999, eps1e-8, constant LR, clip1.0.
All four arms use the same architecture, objective and optimizer settings.

Retain the demonstrated objective for this data-source comparison:

```text
L = mean_valid_base_FP BCEWithLogits(remove_logit,1)
  + 2 * mean_valid_base_TP BCEWithLogits(remove_logit,0)
```

Each group has its own mean within the sampled finding/patch. Empty groups
contribute zero and padding is excluded. No global Dice, L1 residual term,
extra anatomy masks or partial-label masking. Increasing the TP weight is a
possible later controlled ablation; the pilot alone does not establish its
optimal value or show that a larger coefficient fixes small-finding damage.
Class weights affect the precision/recall tradeoff; they are not a retention
guarantee. [PyTorch BCEWithLogitsLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html).

## Broader, preservation-focused sampling

Choose the source first, then the finding uniformly among eligible findings.
Materialize the entire schedule reproducibly before training. For each finding
use the full established192³ / 50%-overlap inference grid, not the three fixed
diagnostic tiles. Eligible tiles must contain valid base-positive voxels.

Proposed branch probabilities:

- 50%: sample an eligible tile containing base TP.
- 25%: sample an eligible tile containing base FP.
- 25%: sample uniformly over all eligible tiles.

Choose uniformly within the selected branch's eligible tiles. These branches
overlap; log actual TP/FP/GT-empty composition as well as requested branches.
If a requested TP or FP branch is empty, fall back to an eligible tile from
the same finding and record it. GT-empty/base-positive tiles remain trainable.
Never infer GT-based eligibility during model inference.

Keep augmentation off for this first four-arm comparison, as in the diagnostic.
Varied tile positions are the deliberate sampling change. Record unique
findings/windows seen and visit counts: equal update counts yield very different
repetition rates for1,120,35 and69-finding pools. Incomplete training labels
remain a potential confound, not something this objective resolves.

## Inference, thresholds and preservation reporting

Use the same full native extent and Gaussian blending of removal probabilities
in FP32 as the diagnostic. Inactive tiles contribute zero action and retain
their blend weights. Apply deletion only within the frozen base mask, after
blending. No new foreground can be produced. Restore to original geometry and
verify each cache baseline.

Propose threshold0.90 as the common, predeclared reference for all arms and
checkpoints. The pilot threshold0.89595 motivates this candidate; it does not
guarantee99% TP retention for a different model. Also report the fixed grid
0.50/0.80/0.90/0.95/0.99/1.00, without automatically selecting a winner. The
1.00 point is the keep-all reference. Threshold changes remain explicit user
decisions. B must not supply automatic calibration. Retrospective matched-
retention comparisons must be labeled as GT-assisted diagnostic tradeoffs.

Full evaluation processes all69 findings once per checkpoint, then derives
A/B/full results and every threshold's metrics from the same scores. Keep the
simple base-logit suppression comparator; report its threshold and actual TP
retention rather than assuming two settings have matched operating costs.

Report mean Dice and changes, hits, FP removed, TP retained, deletion precision,
per-finding precision/recall and improved/unchanged/worsened counts. For TP
preservation show pooled retention AND per-finding mean, median, lower tail,
minimum, absolute TP losses and all cases below95% retention. Findings with
zero base TP have undefined retention and must be labeled accordingly.
Flag losses above20% prominently, even when Dice improves. These are review
flags, not permission to alter the method or stop an arm automatically.

Separate GT-empty from GT-containing patch curves: clearing a GT-empty patch
can increase macro patch Dice substantially without demonstrating better
foreground preservation. Show full-volume results separately. Inspect the
two harmed diagnostic findings with original prompts/GT before interpreting
the new runs; no inference about missing labels should be made from Dice alone.

## Proposed schedule and review point

Keep100 updates per epoch and one independent arm per GPU. First run20 epochs
(2,000 updates each), with complete69-finding evaluation at epochs1/5/10/20.
Epoch0 uses the verified cache baseline plus initial mask-identity checks.
Use synchronous evaluation barriers and preserve checkpoints every epoch,
including all evaluated states, optimizer/RNG state and immutable schedule
cursors. Keep any future duration expansion separate from an immutable phase1
schedule; a continuation appends deterministic events without replacing its
first2,000 events.

Stop after the epoch20 barrier for user review. A possible continuation then
adds epochs21–100 (10,000 total updates), evaluating at30/40/.../100, from the
saved full states. Do not restart successful arms or silently extend a budget.
The early review is a proposal for the new editor, not a change to the stopped
residual experiment or a claim that2,000 updates sufficiently expose all data.

Use one independent CPU dashboard writer, immediate update journals and
provisional evaluation results as each finding finishes. Maintain fixed arm
order and separate training/final-volume panels, with conservative-threshold
Dice, TP retention, FP removal and per-finding failure details visible together.
Persist raw histories and scores externally. Report OOM/non-finite failures
and stop the group without changing tile size or precision.

## Decisions before coding

The accepted defaults are the50/25/25 sampler, reference threshold
0.90 with the stated reporting grid, and the20-epoch review barrier followed
by a separately decided continuation. The pilot loss weights1/2, architecture
and FP32 optimizer settings remain the common reference. The user's subsequent
instruction authorizes implementing and running stage1 with the live dashboard;
the execution spec records that authorization and its epoch20 stop boundary.
