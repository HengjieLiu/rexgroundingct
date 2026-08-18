---
created: 2026-08-14
updated: 2026-08-14
status: audit_complete
experiment_id: "015_voxtell_isotropic_resolution_audit"
---

# Experiment 015: Native Resolution Audit

This repo-local audit records native CT spacing, released-label geometry, and
candidate isotropic storage scale before building the 0.7 mm VoxTell cache.
It reads NIfTI headers and metadata only; it does not resample images, build
caches, write predictions, or launch finetuning.

## Ownership

- Canonical config:
  `configs/experiments/015_voxtell_isotropic_resolution_audit.json`
- Audit script:
  `scripts/rexgroundingct/audit_native_resolution.py`
- Repo-local report:
  `experiments/015_voxtell_isotropic_resolution_audit/report.md`
- Machine-readable summary:
  `experiments/015_voxtell_isotropic_resolution_audit/native_resolution_summary.json`
- Case table:
  `experiments/015_voxtell_isotropic_resolution_audit/native_resolution_cases.csv`

## Status

- Status: `audit_complete`
- Cases audited: `3492` across train, val, and test.
- CT missing: `0`.
- Materialized-HU header failures: `0`.
- Train/val missing labels: `0`.
- Test labels missing as expected: `300`.
- Label shape mismatches: `0`.

## Interpretation

The audit supports using CT headers as physical spacing truth. Released labels
are grid-aligned to matched CTs by shape and finding count, but their NIfTI
affines/zooms are not physical-resolution evidence.

The 0.7 mm estimate is intentionally conservative because it is full-FOV and
uncompressed. Exp016 performs crop-to-nonzero before resampling and stores
targets as compressed sparse `.npz` files.
