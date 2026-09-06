# CT-RATE `ts_total` / ReX test300 audit

This redacted report intentionally contains aggregate evidence only; detailed
case-level manifests, source objects, and overlays remain in the restricted runtime root.

## Conclusion

**PASS_PENDING_MANUAL_VISUAL_REVIEW**

A geometry mismatch is quarantined evidence and never authorizes automatic resampling.

## Pinned sources

- ReXGroundingCT revision: `4aee42ef7ce5ee70fab3e43f9d2c14ece1cd5c80`
- CT-RATE revision: `deeca4d89e9f978d4d1bccd88a55071ddbb146bb`
- Approved remote-manifest SHA-256: `498de533faee91d980b91a530ad02aae78117552b45ae6215a6018e3eb1912ee`
- Producer LUT status: `PASS`

## Coverage and provenance

- Requested masks: 300
- Download statuses: downloaded=300
- Local CT source-provenance statuses: FULL_HASH_MATCH=300
- Full gzip/NIfTI integrity statuses: VALID=300

## Native geometry

- Index-grid statuses: PASS=300
- World/header statuses: PASS=300
- Spacing statuses: PASS=300
- qform/sform/unit compatibility statuses: SFORM_UNSET_QFORM_ALIGNED=300
- Header metadata statuses: SEGMENTATION_SFORM_UNSET=300
- Label-QC statuses: PASS=300
- Cases selected as exceptions for visual QC: 3
- Visual QC images generated: 23

## Review state

- Manual visual review: not yet recorded

ReX test finding masks remain withheld and untouched. No model, preprocessing, inference, submission behavior, or anatomy-prior integration was modified by this audit.
