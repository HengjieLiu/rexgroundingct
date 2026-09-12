---
created: 2026-09-11
updated: 2026-09-11
status: cache_ready_for_training
---

# Completed frozen Exp007 cache for 2b–2e

The cache completed at **2026-09-11 15:51:04 UTC / 08:51 Pacific**. All
2730 CTs and 5106 findings passed the final coverage, source/array integrity,
geometry, per-finding baseline and deletion-index checks. The coordinator
container `rex027_cache_2bcde_v1_resume_dice` exited successfully with code 0.

- Training: 2553 CTs / 4854 findings; all required logits generated.
- Validation: 177 CTs / 252 findings, imported from the existing strict cache
  without additional inference.
- New selected arrays and metadata: 1,496,591,017,141 bytes (1.361 TiB), with CT
  arrays referenced from the existing native preprocessing cache.
- Successful execution wall time: 7 h 26 min 16.5 s, including concurrent CPU
  import/verification, GPU smoke, export and final publication. It shared GPUs
  and storage with the continuing 2a run.
- All 29 CPU tests and eight retained GPU smoke cases passed. The initial Dice
  smoothing defect and recovery evidence remain documented in
  [verification.md](verification.md); frozen inference and storage were unchanged.

## Verified baseline mean per-finding Dice

| Category | Train | A | B | Full validation |
| --- | ---: | ---: | ---: | ---: |
| 2b | 0.374777 | 0.360920 | 0.352978 | 0.357030 |
| 2c | 0.339244 | 0.416814 | 0.413587 | 0.415254 |
| 2d | 0.402410 | 0.393874 | 0.395930 | 0.394980 |
| 2e | 0.426179 | 0.404691 | 0.425653 | 0.410408 |

All full-validation values reproduce the pinned strict source, including the
existing `1e-6` Dice smoothing. These are frozen-base results, not newly trained
refinement results. Complete aggregate and per-finding tables are in runtime
`reports/baselines.csv` and `reports/baseline_findings.csv`.

## Membership and deletion eligibility

| Category | Train findings | A findings | B findings | Deletion-eligible train | Deletion-eligible validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2b | 1367 | 25 | 24 | 1365 | 49 / 49 |
| 2c | 1507 | 31 | 29 | 1506 | 60 / 60 |
| 2d | 1743 | 61 | 71 | 1728 | 129 / 132 |
| 2e | 237 | 8 | 3 | 236 | 11 / 11 |

The 19 training and three validation findings with no base-positive voxels
remain cached and are marked ineligible for deletion-only training. No finding
was dropped from the data inventory or validation summaries.

The original split includes 2d patient `train_2936` in training and validation A
on different CTs: one training finding and two A findings. All affected keys are
flagged in the category manifests and final inventory. Resolve that exposure
before calling a future 2d comparison patient-held-out. A/B remain patient
disjoint; 2e B has only three findings.

## Handoff

The category train/A/B/full manifests passed exact key coverage and A+B
recomposition checks. A real cache-only loader probe succeeded on one training
and one validation finding in each category, preserving channel identity and
matching CT/logit/GT geometry. Current producer/context integrity also passed.

Use the loader documented in [README.md](README.md) to train from these inputs.
The final [dashboard](../runtime/cache_2bcde_v1/reports/live_dashboard.md) and
[verified inventory](../runtime/cache_2bcde_v1/input_manifest.json) are available.
No category-specific refinement training or model/threshold selection ran.
