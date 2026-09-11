---
created: 2026-07-29
updated: 2026-09-07
status: active
---

# Side Experiments

This directory holds bounded research investigations that should not enter the
canonical `experiments/` numbering, registry, runtime sync, or launcher
workflow.

The category threshold sweep lives in
[`sideexp006_category_threshold_tuning`](sideexp006_category_threshold_tuning/README.md).
It reads frozen val200 logits and produces executed review notebooks for raw
Dice and hit rate. A1 is complete; B1/D1 tooling and an execution contract
are available, with no completed follow-on result recorded. E1 remains a
separate future recipe.

Side experiments use their own stable IDs:

```text
sideexpNNN_short_description
```

They may contain small scripts, manifests, tests, reports, and derived JSON or
CSV artifacts. Medical images, masks, predictions, checkpoints, and other
heavyweight runtime outputs remain outside Git.

## Current Side Experiments

- `sideexp001_validation_probe_design`: investigates whether a category-aware
  validation probe can predict full `val200` category behavior more reliably
  than the legacy random `val20`.
- `sideexp002_multimodel_ensemble_selection`: records the val200 model roster
  and hard-mask complementarity evidence, exports reusable pre-sigmoid logits,
  and performs cross-validated multi-model ensemble selection.
- `sideexp003_ensemble_method_hub`: provides the unified checkpoint catalog,
  generated val200/category leaderboards, shared logit-cache provenance, and
  immutable method/run specifications for subsequent ensemble work.
- `sideexp005_postprocessing_merge`: compares frozen d1/d2/d3/d11/d12 on
  val200 using official CT-RATE anatomy. Collaborator v1 is d11; collaborator
  strict v2 is applied to d11 to produce d12. Separate r002 is armed to generate
  test d11/d12 after the verified val report, regardless of scores, without ZIPs.
  See [the test execution contract](sideexp005_postprocessing_merge/test300_execution_spec.md)
  and the SideExp005 README for live state and aggregate report locations.
  Separate r003/r004 compare top-eight e1/e2/e3/e11/e12 against d on val200 and
  automatically prepare test folders after the val report and Wave 2 validation.
  See [the e-series contract](sideexp005_postprocessing_merge/e_series_execution_spec.md).
  <!-- sideexp005-e-val200 --> val200: 1000 verified files; [de_val200 report](sideexp005_postprocessing_merge/de_val200_report.md).
  <!-- sideexp005-e-test300 --> test300: 1500 verified files; [e_test300 report](sideexp005_postprocessing_merge/e_test300_report.md).
