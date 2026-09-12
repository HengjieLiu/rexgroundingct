---
created: 2026-09-11
updated: 2026-09-11
status: cache_ready_for_training
---

# Frozen Exp007 caches for 2b–2e

This task prepares reusable original-training and validation inputs, without
starting category-specific refinement training. The user authorized sharing
GPUs with the existing 2a experiment, subject to the recorded memory gates.

- [Execution specification](codex_execution_spec.md)
- [Live cache progress](../runtime/cache_2bcde_v1/reports/live_dashboard.md)
- [Baseline tables](../runtime/cache_2bcde_v1/reports/baselines.csv)
- [Verified input inventory](../runtime/cache_2bcde_v1/input_manifest.json)
- [Verification and launch evidence](verification.md)
- [Completion report and category baselines](results.md)

Canonical config: `configs/experiments/027_voxtell_2bcde_cache.json`.
Runtime: Exp027 external root, `cache_2bcde_v1`.
Completed at 2026-09-11 15:51 UTC (08:51 Pacific), status
`cache_ready_for_training`. All 2730 CTs / 5106 findings passed integrity and
baseline checks. Selected arrays and metadata occupy 1.36 TiB; existing CT
arrays are shared. All prior experiment artifacts remain immutable.

## Loading future training inputs

After completion, `manifests/<category>/train.json`, `A.json`, `B.json` and
`full.json` contain stable finding keys. Use the standalone
`exp027_multicategory_data.FindingStore` with this cache's runtime directory.
`get(key)` returns `(ct, base_logit, binary_gt, finding_index, metadata)`;
arrays are read-only memory maps. The finding index includes full-finding
TP/FP/FN/GT counts, baseline Dice, deletion eligibility and tile coordinates.
Missing or changed verified inputs raise an error; this loader never runs
VoxTell. Future trainers keep their own loss/schedule configuration separately.

The original split has one 2d patient in both training and validation A:
`train_2936`. Affected keys are explicitly flagged in each category view and
the final inventory. Cache coverage is complete by original dataset membership;
resolve that exposure before making a patient-held-out 2d training comparison.
