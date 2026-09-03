---
created: 2026-08-22
updated: 2026-08-22
status: audit_complete
experiment_id: "019_voxtell_iso07_lung_bbox_coverage_audit"
---

# Experiment 019: 0.7 mm Lung Bounding-Box Coverage Audit

This audit measures how many 0.7 mm isotropic voxels of equal expansion on all
six faces are required for a TotalSegmentator whole-lung bounding box to contain
each released validation finding mask.

## Ownership

- Canonical config:
  `configs/experiments/019_voxtell_iso07_lung_bbox_coverage_audit.json`
- Audit script:
  `scripts/rexgroundingct/audit_iso07_lung_bbox_coverage.py`
- Focused tests:
  `scripts/rexgroundingct/test_audit_iso07_lung_bbox_coverage.py`
- Execution spec:
  `experiments/019_voxtell_iso07_lung_bbox_coverage_audit/codex_execution_spec.md`
- Human report:
  `experiments/019_voxtell_iso07_lung_bbox_coverage_audit/report.md`
- Runtime directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/019_voxtell_iso07_lung_bbox_coverage_audit`
- Derived iso07 lung-mask cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/totalsegmentator_lung_iso07_v1`

## Data Contract

- Fixed validation population: 200 cases and 381 findings.
- Whole-lung mask: union of TotalSegmentator labels 10–14.
- GT masks: existing `crop_clip1024_linear_iso07_v1` `targets.npz` artifacts.
- Mask space: per-case reoriented, cropped, `F,Z,Y,X`, 0.7 mm isotropic grid.
- Expansion scalar: the maximum of the six directional bbox deficits.

TotalSegmentator is an anatomical prior from the fast 3 mm model, upsampled to
the source CT geometry; it is not a lung ground-truth annotation.

## Runtime And Git Policy

The audit reads the CTs, TotalSegmentator outputs, and cached GT masks but does
not modify them. Per-case lung masks, logs, manifests, JSON, and CSV outputs
remain under the external runtime/data roots. Only the small Markdown report and
provenance artifacts belong in the repository.
