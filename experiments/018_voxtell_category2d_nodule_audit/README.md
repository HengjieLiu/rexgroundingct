---
created: 2026-08-21
updated: 2026-08-21
status: audit_complete
experiment_id: "018_voxtell_category2d_nodule_audit"
---

# Experiment 018: Official Category-2d Nodule Method Audit

This is an audit-only experiment. It establishes a reproducible Dice-first
baseline for pulmonary nodules/masses, defined strictly as official category
`2d` in the fixed 200-case validation set, plus a hash-pinned text and
annotation-density comparison of the released training and validation splits.
It does not produce a checkpoint, run inference, write predictions, or launch
training.

## Ownership

- Canonical config:
  `configs/experiments/018_voxtell_category2d_nodule_audit.json`
- Read-only audit generator:
  `scripts/rexgroundingct/audit_category2d_nodule_methods.py`
- Hash-pinned source contract:
  `experiments/018_voxtell_category2d_nodule_audit/audit_sources.json`
- Human audit:
  `experiments/018_voxtell_category2d_nodule_audit/report.md`
- Machine-readable audit:
  `experiments/018_voxtell_category2d_nodule_audit/nodule_method_audit.json`

## Scope

- Official nodule population: category `2d`, pulmonary nodules/masses.
- Fixed val200 category-2d population: `132` findings in `119` cases.
- Primary ranking: threshold-`0.5`, single-model checkpoints, Dice first and
  hits second.
- Train–validation distribution comparison: all `2,992` released train and
  `200` validation CT cases. Its unit is the released CT case, not patient;
  the source has no reliable patient identifier.
- Location labels are text-only findings descriptions. No lung/lobe masks,
  segmentation masks, CT images, coordinates, predictions, or image-derived
  localization are used.
- Annotation-density metrics use released `entity_counts` metadata only, not
  a segmentation-mask inspection. Test is cited for its documented exhaustive
  annotation policy but excluded because this metadata snapshot has no test
  `entity_counts`.
- Explicit exclusions: category `1e`, Exp004's legacy regex nodule stratum,
  ensembles, threshold oracles, and incomplete evaluation evidence.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/audit_category2d_nodule_methods.py
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/audit_category2d_nodule_methods.py --check
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/test_audit_category2d_nodule_methods.py
```

The source manifest pins every evaluator used for a scored row by path and
SHA-256. The generator recomposes category metrics from finding-level raw
evaluator records before rendering the report and JSON snapshot. The
train–validation comparison additionally requires the declared read-only raw
challenge metadata path to be present with its manifest-pinned SHA-256.

## Current conclusion

No completed direct category-`2d` fine-tune has exceeded Exp007's broad
phase-2 continuation on fixed-val200 category-2d Dice. A training matrix is a
later decision outside this audit.
