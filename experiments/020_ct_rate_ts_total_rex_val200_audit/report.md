# CT-RATE `ts_total` / ReX validation audit

This redacted report intentionally contains aggregate evidence only; detailed
case-level manifests, source objects, and overlays remain in the restricted runtime root.

## Conclusion

**PASS_PENDING_MANUAL_VISUAL_REVIEW**

A geometry mismatch is quarantined evidence and never authorizes automatic resampling.

## Pinned sources

- ReXGroundingCT revision: `4aee42ef7ce5ee70fab3e43f9d2c14ece1cd5c80`
- CT-RATE revision: `deeca4d89e9f978d4d1bccd88a55071ddbb146bb`
- Approved remote-manifest SHA-256: `179d7294d67c736757a4da42b4572473a2d7cb28a8456717f1aaa05bdd8ebeb2`
- Producer LUT status: `PASS`

## Coverage and provenance

- Requested masks: 200
- Download statuses: already_valid=200
- Local CT source-provenance statuses: FULL_HASH_MATCH=200
- Full gzip/NIfTI integrity statuses: VALID=200

## Native geometry

- Index-grid statuses: PASS=200
- World/header statuses: PASS=200
- Spacing statuses: PASS=200
- qform/sform/unit compatibility statuses: SFORM_UNSET_QFORM_ALIGNED=200
- Header metadata statuses: SEGMENTATION_SFORM_UNSET=200
- Label-QC statuses: PASS=200
- Cases selected as exceptions for visual QC: 1
- Visual QC images generated: 21

## Review state

- Manual visual review: not yet recorded

No model, preprocessing, inference, or anatomy-prior integration was modified by this audit.
