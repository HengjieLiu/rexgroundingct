---
created: 2026-09-04
updated: 2026-09-04
status: pass_pending_manual_visual_review
experiment_id: "020_ct_rate_ts_total_rex_val200_audit"
---

# CT-RATE `ts_total` / ReX Validation Audit

## Technical result

**PASS_PENDING_MANUAL_VISUAL_REVIEW**

The audited source set is exactly the 200-case ReXGroundingCT MICCAI validation
cohort, all mapped to CT-RATE `train_fixed`. The targeted `ts_total` set was
sealed at CT-RATE revision `deeca4d89e9f978d4d1bccd88a55071ddbb146bb` and
ReXGroundingCT revision `4aee42ef7ce5ee70fab3e43f9d2c14ece1cd5c80`.

| Check | Result |
| --- | --- |
| Requested and valid source masks | 200 / 200 |
| Full local-CT SHA-256 provenance matches | 200 / 200 |
| Full gzip/NIfTI integrity | 200 / 200 |
| Same index grid and world affine | 200 / 200 |
| Same spacing and orientation | 200 / 200 |
| Allowed `ts_total` labels | 200 / 200 |
| QC-only native left/right lung-mask pairs derived | 200 / 200 |

The producer-bound TotalSegmentator 2.7.0 LUT passed. Every pair has the same
benign header-metadata convention: the CT and segmentation have identical
coded qforms and effective affines, the CT's coded sform agrees with that
affine, and the segmentation sform is unset. This is reported as
`SFORM_UNSET_QFORM_ALIGNED`; it is not a resampling or header-rewriting step.

Lung-QC found one fragmentation flag and no robust native-volume outliers.
Twenty deterministic representative panels plus that flagged panel were
generated externally and sealed to the audit result.

## Required next review

A human must inspect the 21 external visual-QC panels and record the outcome
before this becomes a final PASS. Until then, this audit does not authorize any
training, inference, preprocessing, resampling, anatomy-prior integration, or
other downstream use. Detailed manifests, source masks, derived masks, and
overlays remain outside Git in the experiment runtime root.
