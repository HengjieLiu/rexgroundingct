---
created: 2026-09-05
updated: 2026-09-05
status: preflight_complete_awaiting_manifest_approval
experiment_id: "023_ct_rate_ts_total_rex_test300_audit"
---

# Codex Execution Spec: Official CT-RATE `ts_total` Test300 Audit

## Objective

Acquire only the official CT-RATE `ts_total` masks corresponding to the
ReXGroundingCT MICCAI test cohort, then determine whether they are aligned with
the exact local native CT-RATE fixed volumes. Stop before any model,
preprocessing, inference, cache, submission, or anatomy-prior integration
change.

## Prior Evidence And Cohort Contract

- The cohort contains `300` unique test scans. Filename-derived source mapping
  resolves all of them to CT-RATE `train_fixed`; `valid_fixed` contributes
  zero masks.
- The expected pinned payload is `300` files totaling `626,543,520` bytes and
  has no overlap with the known unavailable CT-RATE masks.
- The audit pins ReXGroundingCT at
  `4aee42ef7ce5ee70fab3e43f9d2c14ece1cd5c80` and CT-RATE at
  `deeca4d89e9f978d4d1bccd88a55071ddbb146bb`.
- Exp020 established the acquisition, provenance, geometry, LUT, and native
  lung-QC policy for a separate validation population. Its sealed scripts,
  source lock, runtime output, and report remain unchanged.
- Challenge-test finding labels are withheld. No evaluation manifest is used
  or created, and no test labels are read, materialized, or joined.

## Scope

In scope:

- Authenticated, revision-pinned source preflight for the ReX test population
  and CT-RATE `ts_total` paths.
- Exact 300-case manifest creation, targeted acquisition, local-CT provenance,
  integrity checks, native-grid/header audit, restricted lung QC, and manual
  visual-QC recording.
- Private external runtime source locks, case-level audit evidence, and a
  repo-local aggregate/redacted closeout report.

Out of scope:

- Full CT-RATE cloning, unrelated CT-RATE tasks, validation-manifest reuse,
  train/validation population mutation, and test-label access.
- CT or mask rewriting, resampling, training, inference, preprocessing cache
  updates, submission packaging, or test-time anatomy use.

## Inputs, Output Root, And Preflight

- Canonical config:
  `configs/experiments/023_ct_rate_ts_total_rex_test300_audit.json`
- Local ReX metadata:
  `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- Existing native CT root:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- External audit root:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/023_ct_rate_ts_total_rex_test300_audit`
- Repo-local aggregate report:
  `experiments/023_ct_rate_ts_total_rex_test300_audit/report.md`

The runtime root owns every case-level artifact. The tracked report contains
aggregate/redacted evidence only; it never contains identifiers, case paths,
case hashes, masks, overlays, or exception rows.

The audit must authenticate to both gated repositories and confirm that the
resolved immutable revisions equal the configured pins. It validates the local
metadata SHA-256, requires exactly 300 unique `test` names in metadata order,
derives each CT-RATE source path from the filename, and asserts `300`
`train_fixed` / `0` `valid_fixed`. Generic parser tests cover both valid
filename prefixes, but the observed cohort contract must remain all
`train_fixed`.

For each selected case, the preflight records remote path, object size, blob
identifier, LFS SHA-256 where present, Xet hash where present, and local CT
provenance evidence in a private runtime manifest. It rechecks `300` resolved
masks, `626,543,520` bytes, and zero known-unavailable names before generating
the source lock. It prints only the count, byte total, and bounded private
manifest preview; transfer requires the exact manifest SHA-256 approval.

## Transfer, Integrity, And Geometry Contract

Download each source mask at the pinned CT-RATE revision into runtime staging,
fully read gzip and NIfTI content, validate it, then atomically promote it into
the restricted raw-source tree. Valid existing source files remain in place.
Each case records `downloaded`, `already_valid`, `upstream_absent`,
`auth_error`, `transport_error`, or `integrity_error`; access failures are not
silently classified as absent upstream.

For every available CT/mask pair, record independently:

- remote/local source provenance and full content hashes;
- 3-D shape and index transform `inv(A_ct) @ A_seg`;
- best affine, qform and sform matrices/codes, units, axis scales,
  determinant, and handedness;
- residuals in millimetres at all eight matching-index grid corners;
- supplemental orientation codes; and
- full-read NIfTI validity, finite integer labels, observed labels, and
  allowed-label validation status.

Index-grid and world/header status are separate. Any mismatch is quarantined
and added to visual QC; it never triggers automatic resampling. Physical mask
volume uses `abs(det(affine[:3,:3])) / 1000.0` mL.

The only narrowly permitted QC-only header convention is an unset segmentation
sform paired with an exactly matching coded qform, when both qforms agree with
their best affine, the CT's coded sform agrees with its best affine, and every
other provenance, unit, orientation, grid, label, and integrity gate passes.
Record this as `SFORM_UNSET_QFORM_ALIGNED`; no other form disagreement is
accepted.

## LUT, Lung QC, And Manual Review

Whole-lung union requires a CT-RATE-producer-issued, versioned `ts_total`
mapping whose immutable source and SHA-256 are recorded in the source lock.
Hard-coded IDs, Exp019 mappings, or a local TotalSegmentator mapping without
producer provenance are prohibited. If the evidence is unavailable or
inconsistent, report `LUT_PROVENANCE_BLOCKED`, skip lung derivation, and do
not claim final PASS.

After the LUT gate passes, derive native left/right lung masks externally and
compute voxel counts, physical volumes, 26-connected-component counts,
largest-component fractions, and deterministic outlier flags. Generate visual
QC for every exceptional case plus a deterministic sample of 20 representatives.
A human records the external visual-review outcome before any final conclusion.

## Verification And Stop Conditions

Before an authenticated full run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/test_audit_ct_rate_ts_total_rex_test300.py
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/audit_ct_rate_ts_total_rex_test300.py --help
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/audit_ct_rate_ts_total_rex_test300.py --preflight
python -m json.tool configs/experiments/023_ct_rate_ts_total_rex_test300_audit.json >/dev/null
python scripts/rexgroundingct/check_repo_workflow.py
```

Focused tests cover test-population locking, generic filename-prefix mapping,
absence of an evaluation-manifest dependency, runtime/source-lock ownership and
tamper rejection, transfer statuses, geometry/header gates, LUT/lung QC,
deterministic visual selection, and report redaction.

The authenticated no-download preflight completed with all 300 requested
objects resolved, `626,543,520` bytes, 300 local CT provenance matches, and a
passing LUT-provenance gate. The sealed manifest approval SHA-256 is
`498de533faee91d980b91a530ad02aae78117552b45ae6215a6018e3eb1912ee`; no
transfer may begin without that exact approval.

`PASS` requires all 300 requested masks to have verified provenance,
download/integrity success, index-grid and world/header results, label
validation, required manual visual review, and verified LUT evidence for lung
claims. `PARTIAL`, `AUTH_BLOCKED`, `INTEGRITY_ERROR`, and
`LUT_PROVENANCE_BLOCKED` are valid terminal conclusions. No conclusion
authorizes downstream integration without a later, explicit request.

## Closeout

Keep detailed manifests, source masks, derived masks, overlays, logs, and
exceptions in the external runtime root. Update only the aggregate/redacted
repo-local report after verified runtime evidence is available. Do not update
Exp020, `common.py`, the general experiment registry, or unrelated Exp021/022
work as part of this audit.
