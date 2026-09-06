---
created: 2026-09-05
updated: 2026-09-05
status: preflight_complete_awaiting_manifest_approval
experiment_id: "023_ct_rate_ts_total_rex_test300_audit"
---

# Experiment 023: CT-RATE `ts_total` ReX Test300 Audit

This is a separate data-acquisition and native-geometry audit for the
ReXGroundingCT MICCAI test cohort. It acquires only official CT-RATE
`ts_total` masks, validates whether they correspond to the existing native CT
volumes, and stops before any downstream use. It does not amend or depend on
the Exp020 validation cohort, runtime artifacts, or results. Its entrypoint
hash-pins the unchanged Exp020 audit engine as an implementation dependency in
its own Exp023 source lock.

## Pinned Preflight Contract

- ReX test population: `300` unique scans.
- CT-RATE source split: `300` `train_fixed`, `0` `valid_fixed`.
- Requested `ts_total` payload: `300` files totaling `626,543,520` bytes.
- Expected overlap with the known unavailable CT-RATE masks: `0`.
- ReXGroundingCT revision:
  `4aee42ef7ce5ee70fab3e43f9d2c14ece1cd5c80`.
- CT-RATE revision:
  `deeca4d89e9f978d4d1bccd88a55071ddbb146bb`.

The authenticated preflight must independently re-query the exact immutable
paths, compare the local metadata bytes to the pinned ReX object, verify the
source count and byte total, and write an external source lock before it
transfers anything. A download starts only after explicit approval of the
prepared private manifest SHA-256.

No fixed evaluation manifest is read or created for this challenge test cohort.
Test finding labels remain withheld and out of scope.

## Ownership

- Canonical config:
  `configs/experiments/023_ct_rate_ts_total_rex_test300_audit.json`
- Audit entrypoint:
  `scripts/rexgroundingct/audit_ct_rate_ts_total_rex_test300.py`
- Focused tests:
  `scripts/rexgroundingct/test_audit_ct_rate_ts_total_rex_test300.py`
- External runtime root:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/023_ct_rate_ts_total_rex_test300_audit`
- Repo-local aggregate/redacted report: `report.md`.

## Runtime And Git Policy

The external runtime root owns raw downloaded masks, staging files, detailed
manifests, per-case hashes, derived lung masks, visual-QC overlays, logs, and
detailed exception records. Downloaded official source masks are preserved
unchanged.

Tracked artifacts are limited to the config, execution spec, implementation,
tests, and a small aggregate/redacted report. Case identifiers, case-level
paths or results, NIfTIs, derived masks, overlays, logs, hashes, and detailed
exceptions remain external and must not appear in Git-facing documentation or
reports.

## Hard Gates

1. Both gated repositories must authenticate successfully before remote-path
   checks; authorization failures are never classified as missing upstream.
2. Download is targeted, revision-pinned, staged, resumable, and atomic; it
   requires explicit approval of the prepared manifest hash.
3. A geometry mismatch quarantines the case. It never authorizes resampling,
   source-header rewriting, or downstream use.
4. Lung derivation is blocked until a CT-RATE-producer, versioned, hash-pinned
   `ts_total` label mapping is available. Without it the audit reports
   `LUT_PROVENANCE_BLOCKED`, not PASS.
5. A final PASS requires source provenance, index-grid, world/header,
   integrity, label, and recorded manual visual-QC status to be complete for
   all 300 requested masks.
6. Exact qform/sform field population is reported separately from effective
   native geometry. The narrowly documented `SFORM_UNSET_QFORM_ALIGNED`
   convention may support QC-only native lung derivation only after every
   other gate passes; it never waives a conflicting transform, unit,
   orientation, grid, provenance, or label result.

## Current Status

The implementation and authenticated no-download preflight are complete. The
sealed private manifest contains all 300 requested masks, `626,543,520` remote
bytes, 300 locally provenance-matched CTs, and a passing LUT gate. Its required
download-approval SHA-256 is
`498de533faee91d980b91a530ad02aae78117552b45ae6215a6018e3eb1912ee`.

No source mask has been acquired or audited. `report.md` is intentionally an
aggregate-only preflight record and is not technical PASS evidence. No
training, inference, preprocessing-cache, submission, or test-time anatomy
integration is authorized by this experiment.
