---
created: 2026-09-03
updated: 2026-09-04
status: audit_complete_pending_manual_visual_review
experiment_id: "020_ct_rate_ts_total_rex_val200_audit"
---

# Experiment 020: CT-RATE `ts_total` ReX Validation Audit

This is a data-acquisition and native-geometry audit only. It must establish
whether the official CT-RATE `ts_total` masks correspond to the exact native CT
volumes already used for the ReXGroundingCT MICCAI validation cohort. It does
not modify training, inference, preprocessing, or the existing anatomy caches.

## Pinned Preflight Contract

- ReX validation population: `200` unique scans.
- CT-RATE source split: `200` `train_fixed`, `0` `valid_fixed`.
- Requested `ts_total` payload: `200` files totaling `415,039,487` bytes.
- ReXGroundingCT revision:
  `4aee42ef7ce5ee70fab3e43f9d2c14ece1cd5c80`.
- CT-RATE revision:
  `deeca4d89e9f978d4d1bccd88a55071ddbb146bb`.

The next authenticated preflight must query the exact paths at these immutable
revisions and write a source lock before it transfers anything. It must show
the resolved count and bytes, then wait for an explicit manifest-hash approval.

## Ownership

- Canonical config:
  `configs/experiments/020_ct_rate_ts_total_rex_val200_audit.json`
- Planned audit entrypoint:
  `scripts/rexgroundingct/audit_ct_rate_ts_total_rex_val200.py`
- Planned focused tests:
  `scripts/rexgroundingct/test_audit_ct_rate_ts_total_rex_val200.py`
- External runtime root:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/020_ct_rate_ts_total_rex_val200_audit`
- Repo-local aggregate report after a permitted closeout: `report.md`.

## Runtime And Git Policy

The external runtime root owns raw downloaded masks, staging files, detailed
manifests, per-case hashes, derived lung masks, visual QC overlays, logs, and
detailed exception records. Preserve downloaded source masks unchanged.

Tracked artifacts are limited to this contract, code, tests, and a small
aggregate/redacted report. Do not put case identifiers, case-level paths,
case-level results, NIfTIs, derived masks, overlays, logs, or detailed
exceptions in Git-facing documentation or reports.

## Hard Gates

1. Both gated repositories must authenticate successfully before remote-path
   checks; authorization failures are never classified as missing upstream.
2. Download is targeted, revision-pinned, staged, resumable, and atomic; it
   requires an explicit approval of the prepared manifest hash.
3. A geometry mismatch quarantines the case. It does not authorize automatic
   resampling or any downstream use.
4. Lung derivation is blocked until a CT-RATE-producer, versioned, hash-pinned
   `ts_total` label mapping is available. Without it the audit must report
   `LUT_PROVENANCE_BLOCKED`, not PASS.
5. A final PASS requires source provenance, index-grid, world/header, integrity,
   label, and manual visual-QC status to be independently complete.
6. Exact qform/sform field population is reported separately from effective
   native geometry. QC-only derivation may tolerate an unset segmentation
   sform only when its coded qform exactly matches the CT, each qform agrees
   with its best affine, and the CT's coded sform also agrees with that affine.
   This narrow metadata convention never waives a competing transform, unit,
   orientation, grid, provenance, or label failure.

## Completed Technical Audit

The sealed audit completed on 2026-09-04 with 200/200 verified source pairs.
Every pair has a matching full local-CT SHA-256, a valid source mask, matching
native index grid/world affine/spacing, and valid `ts_total` labels under the
CT-RATE-producer-bound TotalSegmentator 2.7.0 mapping. The masks consistently
leave their sform unset while retaining the exact coded qform; this is recorded
as `SFORM_UNSET_QFORM_ALIGNED`, not silently normalized or treated as exact
header equality.

QC-only native left/right lung masks were written externally for all 200 pairs.
The deterministic visual set contains 21 panels: 20 representatives plus one
fragmentation-flagged lung mask. A human review of those external panels is
still required before the report can be marked PASS. See `report.md` for the
redacted aggregate outcome; no downstream integration is authorized.
