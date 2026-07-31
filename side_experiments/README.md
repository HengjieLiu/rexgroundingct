---
created: 2026-07-29
updated: 2026-07-29
status: active
---

# Side Experiments

This directory holds bounded research investigations that should not enter the
canonical `experiments/` numbering, registry, runtime sync, or launcher
workflow.

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
