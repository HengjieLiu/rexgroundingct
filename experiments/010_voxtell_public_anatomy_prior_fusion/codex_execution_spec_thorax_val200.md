---
created: 2026-07-26
updated: 2026-07-26
status: active
experiment_id: "010_voxtell_public_anatomy_prior_fusion"
phase: "informative_val200_thorax_cache"
---

# Exp010 Informative TotalSegmentator Thorax Val200 Cache

## Objective

Generate a second, more informative TotalSegmentator cache for the fixed
val200 cohort using high-resolution selected thoracic anatomy plus open airway,
vessel, body, and compartment tasks. The earlier 3 mm `total_fast` phase-A
inventory remains immutable and is used only as a comparison source.

## Inputs And Paths

- Fixed validation JSON:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`
- Expected validation JSON SHA256:
  `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`
- Raw CT root:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- ReX target root:
  `/data/hengjie/datasets/rexgroundingct/segmentations`
- TotalSegmentator weights:
  `/mnt/shengdata1/hengjie/models/totalsegmentator/2.16.0`
- New cache root:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/totalsegmentator_thorax_v1`
- Runtime reports:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/010_voxtell_public_anatomy_prior_fusion`

## Method

Run TotalSegmentator on original HU CTs in source geometry. Do not use VoxTell
z-score caches, crop-to-nonzero arrays, or lesion/pathology-predicting tasks.

Per case, save native-geometry packed multilabel files:

- `total_roi_labels.nii.gz`
- `lung_vessels_labels.nii.gz`
- `trunk_cavities_labels.nii.gz`
- `body_labels.nii.gz`

Each task also writes task-specific metadata, statistics, run report, inference
log, failure file when needed, lock directory, and completion marker.

## Total ROI Subset

`task=total` uses 1.5 mm, no `--fast`, and `--roi_subset` for lung lobes,
trachea, esophagus, heart, aorta, pulmonary vein, SVC/IVC, central great
vessels, T1-T12, C7, spinal cord, ribs, sternum, costal cartilage, clavicles,
scapulae, liver, spleen, stomach, and bilateral adrenal glands.

Separate open tasks:

- `lung_vessels`: all four labels.
- `trunk_cavities`: all four labels.
- `body`: `body_trunc` and `body_extremities`.

Explicit exclusions:

- `lung_nodules`
- `pleural_pericard_effusion`
- lesion, opacity, tumor, hemorrhage, or other pathology-predicting tasks

## Launch Contract

- Four detached workers, GPUs `0-3`.
- One shard per GPU.
- Each worker processes tasks in order: `total`, `lung_vessels`,
  `trunk_cavities`, `body`.
- `--require-device gpu`
- `wait_free_mib=30000`
- `poll_seconds=60`
- `free_stability_checks=2`
- Wait before every case so active jobs are not displaced.
- Workers skip complete task-case outputs and use per-task locks.

## Verification And Closeout

- Static validation: fixed val JSON hash and valid `total` ROI names.
- Smoke: one val200 case through all four tasks.
- Full acceptance: `200/200` complete for each task, no geometry mismatch,
  no invalid labels, final aggregate report, target-overlap CSV, runtime CSV,
  representative visual QC, and comparison with the phase-A 3 mm cache.

Anatomy outputs are soft priors only. They must not be used to hard-clip lesion
predictions.
