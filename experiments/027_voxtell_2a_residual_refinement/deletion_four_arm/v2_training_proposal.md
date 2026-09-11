# Proposal: improve FP deletion while protecting vulnerable TP regions

**User priority clarification:** mean Dice is the primary goal, and TP removal is
acceptable when compensated by sufficient FP removal and higher Dice. The 99%
pooled / 95% per-finding targets below were proposals and are superseded as default
acceptance gates. See the [Dice-first tradeoff audit](dice_first_tradeoff_audit.md)
for the mathematical analysis, dense saved-score sweeps and revised loss
direction: evaluate a retained-mask Dice objective with auxiliary deletion BCE,
while keeping TP/hit losses visible as diagnostics. The original loss-ablation
table below remains a historical proposal and needs revision before execution.

Status: **proposal_pending_user_review**. This responds to the request to rethink
the pipeline after the diagnostic and four completed deletion runs. The analysis
uses saved records and scores on CPU. No new training is authorized or launched
by this document; its weights, thresholds, splits and budgets are recommendations.

## What the experiments establish

- The original unconstrained residual method could introduce new foreground and
  did not deliver a consistent benefit. Keep the exact deletion-only subset rule
  for the next experiment; evaluate FN addition separately in the future.
- The eight-case diagnostic demonstrated useful in-sample deletion ranking but
  also substantial damage to an individual small finding. Its isolated-patch
  and full-volume blended scores differed by mean absolute 0.07336; 10.881% of
  decisions differed at 0.5. The larger runs already use the native inference
  tile grid, so merely switching to that grid again is not a new intervention.
- At epoch 20, run 3 at removal threshold 0.5 has A/B/full Dice
  0.363095/0.340140/0.351784, full TP retention 89.9392% and FP removal 26.0370%.
  Forty-four findings retain less than 95% of their TP. At threshold 0.8,
  A/B/full Dice is 0.344856/0.337712/0.341336, full TP retention 99.4637% and
  FP removal 3.0008%; no finding falls below 95% retention.
- A high pooled retention is insufficient: run 4 at threshold 0.9 retains
  99.8700% of TP globally but one A finding retains only 54.44%, losing a hit.
- Simple base-probability thresholding is a substantive control. Its saved
  0.25–0.95 sweep gives full Dice 0.339387 at probability 0.80, versus the
  original base 0.337215. That retrospectively observed grid maximum is not an
  adopted threshold or independent test result. Compare deletion policies at
  comparable preservation as well as against base probability 0.5.
- Equal updates give unequal exposure. In 2,000 updates, run 1 sees 933 distinct
  findings, run 3 all 35 A findings and run 4 all 69 A+B findings. Run 3 has about
  twice as many sampled examples per available finding as run 4. Run 4 is an
  empirical capacity diagnostic with its particular budget, not a mathematical
  upper bound, and its comparison with run 3 does not isolate label quality.

Sources: [all saved thresholds](all_thresholds.md), [diagnostic](../deletion_diagnostic/results.md),
[base-probability sweep](../base_probability_threshold_sweep.md).

## New loss audit: the initial loss decrease has a trivial component

For a patch containing both FP and TP, the existing loss of a constant removal
probability p is `-log(p) - 2 log(1-p)`. Its minimum is 1.909543 at p=1/3.
Initialization at p=0.05 is substantially worse even without any spatial
discrimination. FP-only and TP-only patches change the optimum for a sequence.

For the last 500 recorded patches of each run, define a as the fraction with FP
and b as twice the fraction with TP. The analytic constant optimum is
`p=a/(a+b)`, with loss `-a log(p)-b log(1-p)`.

| Run | Constant p | Constant loss on those patches | Mean recorded online loss |
| --- | --- | --- | --- |
| 1: Train | 0.4111 | 1.63759 | 1.58118 |
| 2: Train+A | 0.3801 | 1.72272 | 1.65442 |
| 3: A | 0.3755 | 1.73790 | 1.56702 |
| 4: A+B | 0.3985 | 1.66356 | 1.65644 |

This analytic reference uses the same recorded patches and their labels. The
online model changes across those updates; it is not a fixed-model held-out
comparison or evidence that the network actually produces a constant map.
It does show why the early aggregate loss decrease alone is weak evidence of
useful deletion ranking. Add fixed probe sets and report class-conditioned losses.

## New operating-point audit: preservation must be measured per finding

Using only saved epoch-20 scores, calculate one global threshold from A:
(1) retain at least 99% of pooled A TP; (2) additionally retain at least 95% of
TP in every A finding with nonzero base TP. Strict score > threshold deletion
and exact discrete counts are used. Apply that same threshold to B.

| Run | A-derived cutoff with both conditions | B Dice | B TP retained | B FP removed | B findings below 95% |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.87543 | 0.33808 | 99.9593% | 0.6403% | 1 |
| 2 | 0.77548 | 0.33635 | 100.0000% | 0.2661% | 0 |
| 3 | 0.76854 | 0.33844 | 99.1548% | 2.2594% | 0 |
| 4 | 0.96600 | 0.33699 | 100.0000% | 0.1959% | 0 |

Exact values and the pooled-only and base-control comparisons are in the
[CPU evidence JSON](../runtime/deletion_four_arm_20ep/reports/v2_loss_audit/evidence.json).
These are retrospective diagnostics, not adopted cutoffs. A was fitted in
runs 2–4, B was fitted in run 4, and B results have already been inspected.
An empirical preservation condition on A does not guarantee it on B, as run 1
demonstrates. The difference between run 3 and run 4 suggests targeting the TP
regions with unusually high deletion scores rather than setting one large
threshold for every model.

## Recommended objective and evaluation procedure

Optimize FP removal while controlling harm to known TP, with mean finding Dice
and hits reported alongside the voxel tradeoff. Provisional development goals:

- At least 99% pooled TP retention on the calibration subset.
- At least 95% retention for each calibration finding with base TP.
- No lost baseline hits on that subset; always report gained and lost hits
  separately, since unchanged hit count can conceal swapped findings.
- Report all failures, raw TP counts, minimum/p10 retention, and instance-level
  retention where original annotations permit it. Findings/instances with no
  base TP remain undefined for retention rather than being assigned 100%.

Derive a single global threshold per model/checkpoint from complete-volume
calibration scores after overlap blending. Show the tradeoff and candidate
threshold to the user; preserve all checkpoints and keep selection with the user.
Do not use a separate GT-derived threshold for every test finding. If preserving
the difficult cases leaves almost no useful deletion, record that limitation.
An identity policy at threshold 1 is always a useful reference.

Evaluate FP-removal curves over the high-retention region (95–100%, with primary
attention to 99–100%). Fit both editor and simple base-threshold controls on the
same calibration patients, and report their actual retention on evaluation
patients. Matched-retention curves computed using evaluation labels are labeled
retrospective discrimination diagnostics, not deployment calibration. Bootstrap
by patient for uncertainty; millions of spatially dependent voxels do not provide
millions of independent calibration observations.

The error-control framing is related to [Neyman–Pearson classification](https://yangfengstat.github.io/projects/neyman_pearson/).
The high-retention ranking objective is related to [partial-AUC optimization](https://proceedings.mlr.press/v162/zhu22g.html).
Those papers support the formulation, not a performance guarantee for this
dataset or the proposed simplified losses.

## First loss ablation: retain BCE and protect the difficult TP scores

Let r be the removal logit. Define:

- `F = mean_FP softplus(-r)` — existing FP deletion loss.
- `T = mean_TP softplus(r)` — existing TP preservation loss.
- `H = mean_top10%_TP softplus(r)` — average preservation loss on the TP voxels
  with highest removal scores within the current finding-patch. Use ceil for the
  selected count, with at least one voxel when TP is present.
- `R = mean_pairs softplus(1 + r_TP - r_FP)` — a sampled pairwise ranking loss,
  pairing vulnerable TP with annotated FP from the same finding-patch. A starting
  implementation can use 2,048 sampled pairs rather than all voxel pairs.

Keep the original BCE terms as anchors. The hard-TP term concentrates additional
gradient on correct voxels the editor is most tempted to delete. The ranking
term asks FP to score above TP; unlike changing a global bias, it encourages
separation. The margin 1, tail fraction 10% and auxiliary weights below are
starting hypotheses, not tuned or established settings.

| Loss variant | Objective | Question |
| --- | --- | --- |
| A | F + 2T | Reproduce the current objective on the new fixed pilot split |
| B | F + 2.5T | Does increasing the total preservation weight alone help? |
| C | F + 2T + 0.5H | Does concentrating that same additional weight on vulnerable TP help? |
| D | F + 2T + 0.5H + 0.1R | Does explicit FP/TP ordering add value beyond C? |

The comparable total TP coefficient in B and C helps separate simple class
reweighting from selective protection. Use identical initialization, schedules,
optimizer, architecture and evaluation procedures across these four variants.
The inference threshold remains outside the training loss in all four.

Absent FP/TP groups contribute differentiable zero to their terms; ranking is
skipped unless both groups are present. All loss voxels are valid, unpadded base
foreground. Avoid aggressive hard-FP mining initially because annotation-negative
voxels may include missing labels. None of these soft losses guarantees a hard
retention percentage; the independent calibration/evaluation checks remain needed.

After this ablation, test averaging the hard-TP term within each original visible
GT instance and then across instances, so a small lesion is not hidden inside a
large finding. This requires recovering original annotation IDs; the cached
binary union alone cannot reconstruct the intended instances. Whole-finding
preservation is evaluated from full volumes, not inferred from one sampled patch.

I would prioritize this ablation over substituting ordinary focal loss or Dice.
The current objective already separately normalizes FP and TP groups. Generic
focal weighting emphasizes hard examples without specifically expressing the
desired per-finding protection. [Tversky loss](https://arxiv.org/abs/1706.05721)
is a reasonable later asymmetric-overlap control, but a pooled overlap ratio can
still conceal damage to small structures. With deletion as the positive action,
an editing false positive means deleting a segmentation TP; loss conventions
must be mapped accordingly.

## Data, sampling and context changes to test separately

1. Review targeted training examples with the CT, original prompt and original
   annotation: high-loss annotated FP, disappearing TP regions, and representative
   successes. Treat original-training-set under-annotation as an unresolved
   hypothesis. A different finding's annotation is not automatically foreground
   for the current prompt. If missing labels are confirmed, define an explicit
   reviewed uncertainty mask or annotation policy before applying strong FP loss;
   distance from GT alone cannot establish reliable background.
2. Retain the current sampler for the first loss-only comparison. Then test
   deterministic shuffled cycles through findings within each source to improve
   coverage, preserving the train/A source quota. Track actual mixed, FP-only
   and TP-only compositions: the nominal 50/25/25 branches overlap. A subsequent
   sampler can target roughly 75% TP-containing and 25% FP-only patches, with a
   capped fraction of the TP-containing quota allocated to small/vulnerable
   regions. Keep broad uniform coverage rather than exclusively replaying errors.
3. Use fixed A-fit and calibration probes, split into mixed, FP-only and TP-only
   cases. Log F, T, H and R separately, plus score distributions and optional
   gradient contributions. A changing sampled-patch mean is not a fixed test.
4. Check overlapping-view score disagreement on the same base-positive voxels.
   If it remains substantial, separately test a small overlap-consistency term
   or a short fine-tuning phase with the deployment blend. Every tile covering
   a base-positive voxel is active, so inactive zero-action tiles cannot explain
   dilution at that voxel. Context/padding/normalization effects need isolation;
   GroupNorm is not yet proven to be the cause.
5. Keep the current architecture for the loss experiment. Its convolutional
   receptive field is about 35 voxels per axis; a 192³ input does not by itself
   provide learned anatomical context across that extent. GroupNorm also uses
   patch-wide statistics, so 35³ is the convolutional field, not an assertion
   of strict locality. If discrimination remains weak on reviewed small cases,
   test a coarse-context branch or frozen prompt-embedding conditioning. Prompts
   specify locations and appearances; the current editor receives that information
   only indirectly through the base logits. Change one architecture factor at
   a time and measure its benefit beyond the loss change.

## Proposed staged execution

1. **Pilot split:** choose one reproducible patient split of A's 31 patients,
   approximately 23 A-fit / 8 A-calibration, keeping scans and findings together
   and balancing finding/instance size. Keep the earlier diagnostic patients in
   A-fit if possible. Report actual counts. A has already informed development,
   so this is an engineering split rather than a new untouched test cohort.
2. **Four loss variants:** use four GPUs to compare A/B/C/D objectives on the same
   A-fit pool, not four different data sources. Start fresh, with FP32/no TF32,
   192³ tiles, AdamW LR 1e-4, weight decay 1e-4, norm clip 1, batch size 1 and
   the same 0.05 initial deletion probability. Run 1,000 updates per variant;
   inspect fixed probes and full-volume A-calibration at updates 100/500/1000.
   Report comparable preservation curves and retain all checkpoints.
3. **Repeat:** after user review, repeat the baseline and a promising modification
   with a second sampling seed and another patient split before attributing a
   small effect to the loss. Favor improved FP deletion at the same retention
   and reduced per-finding harm, not a lower raw training loss alone.
4. **Return to the four data conditions:** after reviewing the loss evidence,
   test the chosen recipe on train; train+A-fit 50/50; A-fit; and A+B as the
   capacity diagnostic. Reserving A-calibration changes the original membership
   of runs 2–3 and must be recorded explicitly. Run 4 remains in-sample on A/B.
   If exact original A membership must be preserved, a separate calibration
   cohort or a declared patient cross-fitting procedure is needed instead.
   Begin with a comparable 2,000-update stage and review whether further training
   is warranted; do not assume a larger number of epochs fixes discrimination.
5. **Report:** preserve A-fit, A-calibration, B and full results, score curves,
   base controls, patient-level uncertainty, individual retention flags and
   exposure counts. B remains gradient-held-out for runs 1–3, but its repeated
   use in development means a future independent confirmation needs new patients.

The first recommended implementation is variant C with a faithful A/B/C/D
comparison. Variant D tests whether ordering helps further. Sampling, instance
normalization and architecture changes follow as separate comparisons so their
contribution is reviewable. Full run/threshold ranking remains with the user.
