---
created: 2026-09-05
updated: 2026-09-05
status: preflight_complete_awaiting_manifest_approval
experiment_id: "023_ct_rate_ts_total_rex_test300_audit"
---

# CT-RATE `ts_total` / ReX Test300 Audit

## Status

**PREFLIGHT_COMPLETE_AWAITING_MANIFEST_APPROVAL**

The authenticated no-download preflight resolved all 300 requested source
objects at the pinned revisions, totaling `626,543,520` bytes. All 300 local
CTs have matching pinned provenance sidecars and the producer LUT gate passed.
The sealed private manifest approval SHA-256 is
`498de533faee91d980b91a530ad02aae78117552b45ae6215a6018e3eb1912ee`.

This is not evidence that any source mask has been downloaded, validated, or
approved for use. Explicit manifest approval, the full audit, and recorded
human visual review are still required.

## Locked Intended Contract

| Contract item | Expected value |
| --- | --- |
| ReX population | 300 unique `test` scans |
| CT-RATE fixed split mapping | 300 `train_fixed`; 0 `valid_fixed` |
| Official task | `ts_total` only |
| Expected pinned payload | 300 files; 626,543,520 bytes |
| ReXGroundingCT revision | `4aee42ef7ce5ee70fab3e43f9d2c14ece1cd5c80` |
| CT-RATE revision | `deeca4d89e9f978d4d1bccd88a55071ddbb146bb` |
| Known unavailable-mask overlap | 0 expected |
| Evaluation-manifest use | None; test finding labels remain withheld |

The runtime preflight must independently prove each source assertion and write
the private source lock before transfer. A static contract, local metadata
check, or matching filename is not a substitute for authenticated source
provenance.

## Required Result Table

The eventual redacted closeout must report only aggregate counts for requested
and valid source masks, local CT provenance, gzip/NIfTI integrity, index-grid
and world/header agreement, spacing/orientation, allowed `ts_total` labels,
QC-only native lung derivation, exception count, and manual visual-review
outcome. It must explicitly state any non-PASS terminal status.

Case identifiers, individual paths, source hashes, raw or derived masks,
overlays, logs, per-case outcomes, and detailed exception records remain under
the external runtime root and are not reproduced here.

## Decision Boundary

Even a final PASS authorizes only the audit's recorded acquisition and
QC-only native-lung derivation. It does not authorize training, inference,
preprocessing cache changes, submission packaging, test-time anatomy use,
resampling, or source-header rewriting. Those actions require a separate
request and review.
