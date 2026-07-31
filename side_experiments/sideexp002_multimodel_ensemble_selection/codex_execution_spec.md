---
created: 2026-07-29
updated: 2026-07-30
status: active
side_experiment_id: sideexp002_multimodel_ensemble_selection
---

# Side Experiment 002 Execution Spec

## Objective

Export reusable final segmentation logits for all fixed-val200 candidates with
mean Dice per finding at least `0.320000`, then choose model count, subset,
weighting, averaging domain, and threshold with deterministic five-fold
case-level cross-validation.

## Inputs and Boundaries

- Fixed validation data:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`
- Candidate source experiments: Exp006, Exp007, Exp008, Exp009, and Exp011.
- Tracked immutable source contract: `candidate_manifest.json`.
- Heavy runtime root:
  `/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp002_multimodel_ensemble_selection`
- No training, model mutation, canonical experiment registration, or test-set
  inference.

## Smoke Gate

Export at least one standard, dual-branch, S3, clipped-z-score, and linear-HU
candidate. Threshold both the in-memory and stored logits at zero and require
exact hit agreement plus Dice within `1e-6`; this isolates serialization loss.
Record the one-case comparison with the historical evaluation as a diagnostic,
not a gate, because it cannot reconstruct a full-run aggregate. For every full
200-case export, retain the original threshold-0.5 Dice and hit comparison as a
warning-only diagnostic. Retry float32 only when float16 storage fails exact
same-pass mask reproduction.

## Full Export Contract

- Save final pre-sigmoid logits in evaluator `(F,X,Y,Z)` layout.
- Clip float16 arrays to `[-30,30]`; record raw ranges and clipping counts.
- Write arrays, case metadata, and candidate manifests atomically.
- Validate dataset, config, evaluation, cache, checkpoint, and geometry hashes.
- Use per-candidate locks, one process per GPU, a 32-GiB free-memory gate, and
  resumable completion checks.
- Permit an explicit no-memory-wait mode for approved co-scheduling. A candidate
  failure is recorded without terminating later candidates in the GPU queue.
- Require an exact export-time threshold-zero voxel comparison for new arrays.
  For older complete arrays, require a zero-free sign-preservation proof while
  revalidating without inference.
- Start the selection handoff only after the deterministic all-20 audit accepts
  complete arrays, source and array hashes, geometry, and storage reproduction.

## Selection Contract

- Five case-level folds, seed `20260729`, balanced by category and findings.
- Default domain: weighted mean of sigmoid probabilities.
- Secondary domain: weighted mean logits followed by sigmoid.
- Search global thresholds coarsely, then refine by `0.01`.
- Fit subset, weights, domain, and threshold on four folds; evaluate once on the
  held-out fold.
- Rank by out-of-fold mean Dice, then hit rate, model count, and uniformity.
- Report the primary recipe, hit-preserving recipe, and smallest near-primary
  recipe. Category-specific policies are post-hoc only and require support of
  at least 20 findings.

## Fixed All-20 Threshold Diagnostic

- Independently probe the uniform probability ensemble containing all 20
  completed exports. Apply sigmoid per model, then average with weight `0.05`.
- Evaluate exactly `0.10`, `0.20`, `0.25` through `0.75` at `0.05` spacing,
  `0.80`, and `0.90`.
- Permit an explicit diagnostic override for failed strict reproduction
  records. The override cannot bypass complete 200-case/381-finding exports,
  exact candidate provenance, recorded array-hash verification, readable array
  structure, dtype, geometry, or finite-value checks.
- Select the single global threshold by overall mean Dice, then hits, then the
  lower threshold. Report every represented category's independent winner,
  including rare categories, and mark the resulting per-category aggregate as
  a hindsight validation oracle.
- Cache only per-case metrics under the external runtime root. Do not
  materialize another probability volume or perform test-set inference.
- Keep this diagnostic separate from the strict finalizer and the
  cross-validated subset/weight/domain search.

## Closeout

Freeze exact candidates, weights, domain, threshold, fold hash, source hashes,
and reproduction command. Cross-check finalists against materialized evaluator
predictions before any later submission decision.
