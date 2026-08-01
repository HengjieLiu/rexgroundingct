---
created: 2026-07-31
updated: 2026-07-31
experiment_id: asd_020_factorial_pathology_anatomy_binding
---

# Factorial Pathology–Anatomy Region Binding

This experiment tests whether frozen VoxTell region features can identify the
conjunction of the requested pathology and anatomy. Training-only released
masks and validated boxes from `anatomical_cot.json` create real-region
quartets:

```text
correct pathology + correct anatomy
correct pathology + wrong anatomy
wrong pathology   + correct anatomy
wrong pathology   + wrong anatomy
```

The same anchor text is used across a quartet, preventing a text-only model
from reading the label from wording. The scorer must beat sentence-only,
pathology-only, anatomy-only, and additive-without-interaction controls before
it can influence predictions.

Any prediction influence is a bounded, zero-initialized component-logit
residual. VoxTell and Qwen remain frozen. This is not Qwen LoRA, direct query
addition, a generic attention loss, or a hard component classifier.

<!-- experimentctl:state:start -->
## Execution state (generated)

- Status: `blocked`
- Outcome: `pending`
- Current stage: `none`
- State revision: `1`
- Updated: `2026-08-01T05:04:50.605768Z`

Blockers:

- `dependency_unsatisfied`: asd_000_evidence_lock_error_atlas: dependency status is ready
<!-- experimentctl:state:end -->
