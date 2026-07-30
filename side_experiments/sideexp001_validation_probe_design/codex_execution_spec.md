---
created: 2026-07-29
updated: 2026-07-29
status: active
side_experiment_id: sideexp001_validation_probe_design
---

# Execution Spec: Category-Aware Validation Probe Design

## Objective

Determine whether a small, category-aware validation probe can reproduce
full-`val200` category Dice and hit behavior more reliably than the legacy
random `val20`, and select its size from an explicit accuracy-cost frontier.

## Prior Evidence

- The legacy `val20` is the first 20 cases of the same seeded permutation used
  for `val200`; it is not stratified.
- Its 31 findings omit several validation categories and overrepresent others.
- Exp006/007/008/009/011 provide 26 per-case `val200` model/checkpoint results
  suitable for retrospective subset evaluation.

## Scope

In scope:

- CPU-only analysis of existing evaluator JSONs.
- Train/validation/full-test category distribution auditing.
- Secondary auditing of the unknown public 50% leaderboard subset.
- Category census, difficulty-stratified sampling, cross-validation, frozen
  chronological holdout evaluation, and small tracked reports.

Out of scope:

- GPU inference or training.
- Test-mask access or test-performance estimation.
- Changing canonical probes, experiment configs, launchers, or registry state.
- Optimizing a validation probe against released test category ratios.

## Inputs and Provenance

`historical_inputs.json` owns the exact metadata, probe, leaderboard snapshot,
runtime group, arm, checkpoint, and design/holdout definitions. Every consumed
file is hashed into the analysis summary.

Exp006/007/008 are the design cohort. Exp009/011 are loaded only after a final
design-cohort case set is selected and hashed.

## Method

1. Reconstruct every full and category metric from case-level evaluator data.
2. Filter full evaluations to legacy cases and compare with independently run
   `val20` metrics using Dice tolerance `1e-4` and exact hit counts.
3. Audit category counts and ratios for train, validation, full released test,
   legacy probe, candidates, and the public leaderboard subset.
4. Benchmark case budgets `20/30/40/50/60/70/80/100/120` and validation
   category census cutoffs `4/7/11/17`.
5. Within non-census categories, form design-only difficulty strata: three
   strata for at least 40 findings, two for 12-39, and one below 12.
6. Select whole cases. Candidate generation enforces census ownership and
   coverage of every validation category/difficulty stratum.
7. Use leave-one-experiment-group-out selection on Exp006/007/008, build the
   nondominated case-count/error frontier, and select its deterministic knee.
8. Apply the preregistered category RMSE gates. Freeze and hash the selected
   case set before loading Exp009/011.
9. Evaluate the frozen candidate on holdout without retuning.

## Acceptance

- Macro category Dice RMSE <= `0.015`.
- 90th-percentile category Dice RMSE <= `0.030`.
- Macro category hit-rate RMSE <= `0.025`.
- 90th-percentile category hit-rate RMSE <= `0.060`.
- Worst-category Dice RMSE <= `0.040`.
- Worst-category hit-rate RMSE <= `0.100`.
- Every category present in `val200` is represented.
- Reconstruction and deterministic-output checks pass.

If no candidate at or below 120 cases passes, the report must recommend
retaining `val200` for category decisions.

## Closeout

Write the tracked report, tables, summary, and—only when design and holdout
acceptance pass—the evaluator-compatible selected probe and manifest. Do not
update canonical experiment status or validation configs.
