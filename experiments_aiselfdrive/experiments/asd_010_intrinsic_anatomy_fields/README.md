---
created: 2026-07-31
updated: 2026-07-31
status: blocked
experiment_id: asd_010_intrinsic_anatomy_fields
---

# Intrinsic Thoracic Anatomy Fields

This experiment replaces brittle hard lobe clipping with continuous,
patient-specific thoracic coordinate fields. It first asks a CPU/cached-logit
question: can correct side, lobe, craniocaudal, pleural, fissural, and
centrality fields remove prompt-off-location false positives while retaining
at least 99% of released GT?

Only a passing diagnostic may unlock training. The learned path is a bounded
late logit residual:

```text
alpha = 2 * tanh(raw_alpha)
z_final = z_base + alpha * tanh(E_anatomy)
```

`raw_alpha=0` takes an exact bypass and returns `z_base` unchanged. Ambiguous,
bilateral, diffuse, hilar, peribronchial, and unsupported prompts also take the
identity path.

The experiment is blocked on GO outcomes from
`asd_000_evidence_lock_error_atlas` and
`asd_005_train_anatomy_asset_extension`.

<!-- experimentctl:state:start -->
## Execution state (generated)

- Status: `blocked`
- Outcome: `pending`
- Current stage: `none`
- State revision: `0`
- Updated: `2026-07-31T00:00:00Z`

Blockers:

- Dependency asd_000_evidence_lock_error_atlas must finish with outcome go.
- Dependency asd_005_train_anatomy_asset_extension must finish with outcome go.
<!-- experimentctl:state:end -->
