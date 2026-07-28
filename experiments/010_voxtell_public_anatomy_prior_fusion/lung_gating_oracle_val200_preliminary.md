---
created: 2026-07-27
updated: 2026-07-27
status: preliminary_result
experiment_id: 010_voxtell_public_anatomy_prior_fusion
source_prediction_experiment: 006_voxtell_cached_native_v123_lr_ablation
---

# Preliminary Oracle Whole-Lung Gating On Exp006 v123_cached_e5_d4 Val200

## Summary

This note records a preliminary read-only oracle analysis that post-processes
experiment 006 `v123_cached_e5_d4` epoch-100 fixed val200 predictions with
TotalSegmentator whole-lung masks.

The gate is "oracle" because it uses the validation ground-truth target
containment fraction to decide which findings are safe to restrict to the
TotalSegmentator whole-lung region. This is useful as a design probe, but it is
not directly challenge-deployable.

## Inputs

- Prediction source: experiment 006 `v123_cached_e5_d4`, epoch 100, fixed
  val200, threshold `0.5`.
- Anatomy source: experiment 010 TotalSegmentator `total_fast` 3 mm val200
  audit.
- Whole-lung labels: TotalSegmentator labels `{10, 11, 12, 13, 14}`.
- Hit threshold: global Dice `>= 0.1`.
- Gating rule for eligible findings:

```text
prediction := prediction AND whole_lung_mask
```

Ungated findings remain unchanged.

## Primary Results

| Setting | Gated findings | Dice/finding | Delta Dice | Hit rate | Hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, no gate | 0 / 381 | 0.3241 | 0 | 0.7612 | 290 / 381 |
| Known gate, 100% contained | 105 / 381 | 0.3306 | +0.0065 | 0.7690 | 293 / 381 |
| Known gate, >=95% contained | 207 / 381 | 0.3340 | +0.0099 | 0.7743 | 295 / 381 |
| Known gate, >=90% contained | 253 / 381 | 0.3338 | +0.0097 | 0.7743 | 295 / 381 |

## Gated Subset Behavior

| Gate | Gated subset Dice before -> after | Hits before -> after | Improved | Worsened |
| --- | ---: | ---: | ---: | ---: |
| 100% | 0.3147 -> 0.3382 | 76 -> 79 | 70 | 0 |
| >=95% | 0.3181 -> 0.3362 | 152 -> 157 | 148 | 22 |
| >=90% | 0.3231 -> 0.3376 | 188 -> 193 | 167 | 46 |

## Interpretation

Whole-lung hard gating helps in this oracle analysis, but the benefit saturates
around the `>=95%` containment cohort. The `>=90%` cohort gates more findings,
but it also introduces more clipped or otherwise worsened cases, so Dice is
slightly below the `>=95%` setting while hit count is unchanged.

For future challenge-valid anatomy fusion, this supports using anatomy as a
selective or soft spatial constraint rather than blanket lung clipping. A
deployable version would need to infer routing from prompt text, category,
anatomy prediction, model confidence, or a learned policy, without using
ground-truth containment.

## Caveats

- This analysis uses validation ground-truth containment to choose the gated
  subset, so it is an upper-bound/design probe.
- The TotalSegmentator audit used the 3 mm `total_fast` output, not the later
  high-resolution thorax cache.
- Findings outside the selected oracle cohort were left unchanged.
- No probability threshold sweep was performed; results use the existing
  exp006 binary predictions at threshold `0.5`.
