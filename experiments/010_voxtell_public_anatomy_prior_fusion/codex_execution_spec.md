---
created: 2026-07-26
updated: 2026-07-26
status: active_data_audit
experiment_id: "010_voxtell_public_anatomy_prior_fusion"
---

# Codex Execution Spec

## Objective

Pin TotalSegmentator reproducibly, run its open anatomical `total` task over
the complete fixed ReXGroundingCT val200 cohort, inspect the masks visually,
and measure which labels are present and overlap challenge targets before
choosing the anatomy channels for VoxTell fusion.

## Prior Evidence

- Research design:
  `experiments/010_voxtell_public_anatomy_prior_fusion/research_report_v1.md`.
- TotalSegmentator is already used by the VoxTell paper for anatomy-aware
  prompt construction, but the public VoxTell forward pass does not consume
  anatomy masks.
- The current TotalSegmentator `total` task exposes 117 classes.
- Fixed validation cohort:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`, SHA256
  `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`.

## Scope

In scope:

- Add the official TotalSegmentator repository as a pinned submodule.
- Download open `total_fast` weights into the shared model root.
- Run 3 mm `total` inference on all 200 validation CTs.
- Save compact native-geometry multilabel masks, per-case provenance, class
  statistics, logs, timing, and GPU memory.
- Generate all-case visual QC, label-frequency tables, and target/anatomy
  overlap tables.
- Recommend a compact thoracic ontology and a 20-case high-resolution check.

Out of scope:

- VoxTell architecture changes or training.
- Disease-predicting TotalSegmentator tasks.
- Treating public anatomy as a hard lesion gate.
- Running the final 1.5 mm train/val cache before the phase-A audit is read.

## Inputs And Paths

- Canonical config:
  `configs/experiments/010_voxtell_public_anatomy_prior_fusion.json`
- Raw CT:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- ReX targets:
  `/data/hengjie/datasets/rexgroundingct/segmentations`
- TotalSegmentator weights:
  `/mnt/shengdata1/hengjie/models/totalsegmentator/2.16.0`
- Reusable phase-A cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/totalsegmentator_total_fast_3mm_v2_16_0`
- Runtime reports and figures:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/010_voxtell_public_anatomy_prior_fusion`

The Docker image layer remains under Docker daemon ownership. Model weights,
NIfTI outputs, logs, reports, and figures must live under
`/mnt/shengdata1/hengjie`.

## Data And Preprocessing Contract

- TotalSegmentator input is the original HU CT, never the VoxTell z-score
  cache.
- Phase A uses `total --fast`, whose model spacing is 3 mm.
- `--save_lowres` is not used: the packed output must be restored to the
  original CT shape and affine.
- CT interpolation is TotalSegmentator's default order 1.
- Output is one `uint8` multilabel NIfTI per case.
- Per-case validation requires exact spatial shape, affine, and axis-code
  agreement with the source CT.
- ReX target files have channel-first masks and non-spatial identity affine
  metadata. Target overlap therefore follows the already validated project
  rule: target array indices align to raw CT voxel indices, while CT affine
  controls physical orientation.
- No crop-to-nonzero or z-score is applied before TotalSegmentator.

## Method

1. Pin `external/TotalSegmentator` at tag `v2.16.0`, commit
   `099b679c76350effc701889d22f25fdbf060cea6`.
2. Record the upstream container
   `wasserth/totalsegmentator:2.16.0` at digest
   `sha256:e2229ec932d0cd38a463c6aa537cfead1406e9fe43c3bca81f476382709c6921`.
   Its CUDA 13 PyTorch build is incompatible with this host driver and is not
   used for GPU inference.
3. Build `rexgroundingct-totalsegmentator:2.16.0-cu126` from the validated
   `rexgroundingct-voxtell:cu126` base and install the pinned submodule with
   `--no-deps`.
4. Query `totalseg_info` for the live 117-class map.
5. Materialize a 200-case JSONL plan in fixed val200 order.
6. Download `total_fast` weights into the shared model root and hash the
   resulting files.
7. Smoke one case while recording runtime and peak GPU memory. Require the
   official run report to resolve `device` as `gpu`; CPU fallback is a failure.
8. Start four resumable workers, one assigned GPU and deterministic shard per
   worker. Workers wait until their GPU has at least 24,000 MiB free so active
   training is not displaced.
9. Aggregate observed labels, case frequencies, voxel volumes, failures, and
   challenge-target overlap.
10. Create 200 three-plane overlays, 10 contact sheets, a label-frequency plot,
   and a case-by-label heatmap.
11. Recommend atomic labels and merged label families for the later
    AnatomyBridge input.

## Smoke Gate

- Source CT exists and is finite HU-valued data.
- TotalSegmentator output is `uint8`, contains only IDs `0-117`, and is finite.
- Output shape, affine, spacing, and axis codes match the source CT.
- Official run report and statistics JSONs exist.
- Peak GPU use fits alongside the current workload without trainer failure.
- Re-running the worker skips the completed case.

## Success Criteria

- Exactly 200 case completion markers exist.
- Every case has one packed mask, metadata, official run report, statistics,
  and inference log.
- No geometry mismatch or failed case remains.
- The final manifest records code, image, weights, dataset, and output hashes.
- Tables report all 117 configured labels and the observed-label union.
- All 200 cases have visual QC thumbnails.
- A compact recommended thoracic ontology and explicit exclusions are
  documented.

## Stop Conditions

- Do not launch full workers while assigned GPU free memory is below the
  configured threshold.
- Stop a case on geometry mismatch or invalid label ID; do not mark it
  complete.
- Do not proceed to the high-resolution ontology cache if phase-A masks show
  systematic lobe or orientation failure.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  scripts/rexgroundingct/prepare_010_totalsegmentator_val_audit.py \
  scripts/rexgroundingct/run_totalsegmentator_val_worker.py \
  scripts/rexgroundingct/summarize_010_totalsegmentator_val_audit.py
bash -n scripts/rexgroundingct/run_010_totalsegmentator_val_audit_docker.sh
python scripts/rexgroundingct/check_experiment_consistency.py
git diff --check
```

## Closeout Plan

Write the complete runtime report under the exp010 report directory, copy a
small result summary into this repo-local experiment folder, update the
canonical config status, sync the experiment index, and revise
`research_report_v1.md` only where measured evidence changes the v1 design.
