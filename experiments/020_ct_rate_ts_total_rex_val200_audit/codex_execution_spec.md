---
created: 2026-09-03
updated: 2026-09-04
status: audit_complete_pending_manual_visual_review
experiment_id: "020_ct_rate_ts_total_rex_val200_audit"
---

# Codex Execution Spec: Official CT-RATE `ts_total` Validation Audit

## Objective

Acquire only the official CT-RATE `ts_total` masks for the ReXGroundingCT MICCAI
validation cohort, then determine whether they are aligned with the exact local
native CT-RATE fixed volumes. Stop before any model, preprocessing, inference,
or anatomy-prior integration change.

## Prior Evidence

- The cohort has `200` unique validation scans, all mapped by filename to
  CT-RATE `train_fixed`; no `valid_fixed` mask is requested.
- The resolved target payload is `200` files and `415,039,487` bytes.
- The audit pins ReXGroundingCT at
  `4aee42ef7ce5ee70fab3e43f9d2c14ece1cd5c80` and CT-RATE at
  `deeca4d89e9f978d4d1bccd88a55071ddbb146bb`.
- Exp015 showed why grid-alignment and NIfTI header/affine claims must be
  reported separately; an affine discrepancy alone is not permission to
  resample a source segmentation.
- The local TotalSegmentator source contains incompatible historical and newer
  total-task label maps. Its mapping is not evidence of the CT-RATE producer's
  `ts_total` encoding.

## Scope

In scope:

- Authenticated, revision-pinned source preflight for both gated datasets.
- Exact 200-case manifest creation, targeted acquisition, integrity checks,
  native-grid/header audit, and restricted visual QC.
- A left/right lung union only after a versioned CT-RATE-producer LUT is
  independently verified.

Out of scope:

- Full CT-RATE repository cloning or acquisition of unrelated tasks.
- Train, test, or non-`ts_total` mask acquisition.
- Any CT rewriting, resampling, training, inference, cache update, R231 run,
  or downstream mask use.

## Inputs And Paths

- Canonical config:
  `configs/experiments/020_ct_rate_ts_total_rex_val200_audit.json`
- Local ReX metadata:
  `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- Existing native CT root:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- External audit root:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/020_ct_rate_ts_total_rex_val200_audit`
- Repo-local aggregate report after closeout:
  `experiments/020_ct_rate_ts_total_rex_val200_audit/report.md`

The runtime root owns every case-level artifact. The tracked report must contain
only aggregate/redacted evidence and no case identifiers or paths.

## Source And Download Contract

1. Authenticate to both gated repositories. A `401`, `403`, or gated-access
   result is `auth_error`, not an absent upstream file.
2. Re-query both pinned revisions, record the full revision, remote path, size,
   blob identifier, LFS SHA-256 where present, and Xet hash where present in a
   private runtime source lock.
3. Validate the local metadata hash, assert exactly 200 unique scans, retain
   metadata order separately from evaluation order, and join later results only
   by filename.
4. Hash each local CT and compare it with pinned CT-RATE remote-object evidence
   where that comparison is available. This is the source-provenance check; a
   matching shape or affine is not a substitute.
5. Print the private manifest's count, total bytes, and representative rows.
   Download begins only after explicit approval of that manifest's SHA-256.
6. Download each target at the pinned revision into runtime staging, fully read
   gzip and NIfTI content, validate it, then atomically promote it into the
   restricted raw-source tree. Preserve any valid existing source file.

Each case records distinct download outcomes: `downloaded`, `already_valid`,
`upstream_absent`, `auth_error`, `transport_error`, or `integrity_error`. No
other failure is silently collapsed into an upstream-missing result.

## Geometry And Integrity Contract

For every available CT/mask pair, record independently:

- source provenance and remote/local object hashes;
- 3-D shape and index transform `inv(A_ct) @ A_seg`;
- best affine, qform and sform matrices/codes, units, axis scales, determinant,
  and handedness;
- residuals in millimetres at all eight matching-index grid corners;
- supplemental orientation codes; and
- full-read NIfTI validity, finite integer labels, observed labels, and allowed
  label validation status.

Index-grid and world/header status are separate columns. A mismatch is
quarantined and listed for visual QC. It never triggers automatic resampling.
Physical mask volume uses `abs(det(affine[:3,:3])) / 1000.0` mL, not a product
of nominal zoom fields.

Exact qform/sform field population is also reported separately from effective
geometry. For QC-only lung derivation, the one permissible non-identical form
pattern is an unset segmentation sform paired with an exactly matching coded
qform, provided both qforms agree with their best affines and the CT's coded
sform agrees with its best affine. This is an explicitly recorded metadata
convention, not a blanket header waiver: any competing form, absent qform,
affine, unit, orientation, shape, or grid discrepancy remains quarantined.

## LUT And Lung-QC Gate

A whole-lung union requires a CT-RATE-producer-issued, versioned `ts_total`
mapping whose immutable source and hash are recorded in the runtime source lock.
Hard-coded IDs and mappings borrowed from Exp019 or the local TotalSegmentator
checkout are prohibited. If this evidence is unavailable or inconsistent, write
`LUT_PROVENANCE_BLOCKED`, skip lung derivation, and do not claim a final PASS.

After the gate passes, compute left/right voxel counts, physical volumes,
26-connected-component counts, largest-component fractions, and deterministic
outlier flags. Visual QC must include every exceptional case plus a deterministic
representative sample. Manual review state is recorded externally before a
final conclusion.

## Smoke Gate And Verification

Before the authenticated full run:

- run focused synthetic tests for remote-status classification, atomic staging,
  gzip/NIfTI integrity, geometry transforms/corner residuals, LUT blocking,
  lobe-union QC, and deterministic visual-QC selection;
- run a one-case, no-download, no-write preflight in the validated container;
- confirm the runtime cache and token location are writable without storing a
  credential in a file or log.

Expected commands after the planned script exists:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/test_audit_ct_rate_ts_total_rex_val200.py
PYTHONDONTWRITEBYTECODE=1 python scripts/rexgroundingct/audit_ct_rate_ts_total_rex_val200.py --preflight
python scripts/rexgroundingct/check_experiment_consistency.py --experiments 020_ct_rate_ts_total_rex_val200_audit
python scripts/rexgroundingct/check_repo_workflow.py
```

## Success Criteria And Stop Conditions

PASS requires every requested mask to have verified source provenance,
download/integrity success, explicit index-grid and world/header results,
validated label semantics, required visual-QC review, and a verified LUT when
lung claims are included. `PARTIAL`, `AUTH_BLOCKED`, `INTEGRITY_ERROR`, and
`LUT_PROVENANCE_BLOCKED` are valid report conclusions. No conclusion authorizes
downstream integration without a later request.

## Completed Audit Snapshot

The final sealed run verified all 200 requested source masks and local CTs:
full CT content provenance, gzip/NIfTI integrity, index-grid geometry,
world-affine geometry, spacing, and allowed labels all pass for every pair.
All source masks use the documented `SFORM_UNSET_QFORM_ALIGNED` convention:
their coded qforms and effective affines match the CTs exactly, while only the
redundant segmentation sform is unset. The report preserves this metadata
difference explicitly.

All 200 QC-only native left/right lung-mask pairs were derived externally. One
fragmentation flag selected one additional visual panel alongside the 20
deterministic representatives. The conclusion remains
`PASS_PENDING_MANUAL_VISUAL_REVIEW` until a human records the external visual
review outcome.

## Closeout Plan

Keep detailed manifests, masks, overlays, logs, and exceptions in the external
runtime root. Copy only a small aggregate/redacted Markdown summary to the
repo-local experiment folder if permitted by dataset terms. Do not update
existing experiment outputs or generated indexes as part of this audit unless a
separate approved change makes that update safe.
