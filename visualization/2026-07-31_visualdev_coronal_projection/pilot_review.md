---
created: 2026-07-31
updated: 2026-07-31
status: complete
---

# Coronal Projection Pilot Review

## Outcome

The representative pilot completed without model inference. It rendered 12
fixed-val200 cases, 28 findings, four prediction sets, and two CT projection
backgrounds for a total of 24 PNG figures.

Runtime output:

```text
Docker: /database/datasets/rexgroundingct/visualizations/
          2026-07-31_visualdev_coronal_projection/pilot
Host:   /data/hengjie/datasets/rexgroundingct/visualizations/
          2026-07-31_visualdev_coronal_projection/pilot
```

Authoritative small runtime artifacts are `run_manifest.json` and
`pilot_finding_metrics.csv` in that directory.

## Projection Review

The mean-intensity CT projection is the preferred background for a full-val200
render. It preserves recognizable lung, mediastinal, diaphragmatic, and chest
wall context while avoiding the single-voxel bone dominance of a true CT MIP.

The P75 projection also suppresses isolated high-density voxels, but the pilot
figures are substantially paler and flatter. It remains useful as a diagnostic
alternative but is not the recommended primary display.

Binary GT and prediction masks use the same maximum/`any` projection in both
versions, so only the CT background changes.

## Row-Count Result

The prior connected-component outlier at fixed val index 23 had a category
figure with 171 rows. The new one-row-per-finding layout renders the entire case
in four rows. The maximum pilot stress cases render in five rows and remain
readable.

## Orientation Verification

- All 12 source CTs report `LPS` voxel orientation.
- The renderer transforms CT, GT, and every prediction into RAS with the same
  CT-affine orientation transform.
- The anterior-posterior RAS axis is collapsed before applying the common
  coronal display transform.
- Synthetic tests verify superior at the top, inferior at the bottom, patient
  right on screen left, and patient left on screen right.
- Visual laterality checks agree with the finding text. In val index 23, the
  annotated left-lung consolidations appear on screen right and the annotated
  right-lung consolidation appears on screen left.

Every panel includes explicit `S`, `I`, `R`, and `L` markers.

## Metric And Artifact Verification

- Figure count: 12 P75 + 12 mean = 24.
- Every selected case has all four prediction NIfTIs.
- All three local fine-tuned checkpoint paths exist. Public VoxTell v1.1 is
  recorded by its Hugging Face model reference rather than a repo-local weight
  path.
- Displayed per-finding 3D Dice values agree with the four canonical evaluator
  JSON files to within `4.99e-7`, the expected CSV rounding difference.
- No projected Dice is reported. Projected TP/FP/FN colors are qualitative
  because structures at different depths can overlap after projection.

## Recommendation

After human review of the pilot, use the mean-intensity CT background for the
full fixed-val200 visualization. Keep P75 selectable in the renderer but do not
generate a second full-val200 gallery unless it is specifically needed.
