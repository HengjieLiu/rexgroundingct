---
created: 2026-07-31
updated: 2026-08-01
status: active
---

# Coronal Projection Visualization Development

This folder develops a compact whole-volume coronal visualization for the
fixed ReXGroundingCT val200 set. It replaces one-row-per-connected-component
figures with one row per annotated finding and compares several existing
VoxTell prediction sets without rerunning inference. It contains both a
representative pilot and the full mean-only val200 gallery organized by ReX
category.

## Pilot Models

The pilot uses the corrected, fixed-threshold val200 predictions for:

| Key | Display role | Mean global Dice per finding | Prediction source |
| --- | --- | ---: | --- |
| `public_v11` | Public VoxTell v1.1 original pretrained model; no project epoch | 0.225228 | Experiment 001 |
| `exp009_s3v1_e100` | 100+100 attention | 0.334709 | Exp009 `s3v1_fixedrho_suppress_half_quarter`, continuation epoch 100, update 10,000 |
| `noddp_best` | 100+100 plain; best no-DDP | 0.339839 | Exp009 `baseline_cont100`, continuation epoch 100, update 10,000 |
| `ddp_best` | Best DDP continuation | 0.346023 | Exp007 continuation relative epoch 50 / absolute epoch 150, continuation update 5,000 |

All prediction directories contain 200 NIfTI files. The visualization reads
those exported masks directly; it does not load model checkpoints or run
inference.

## Projection Contract

- CT, GT, and predictions are transformed into RAS voxel orientation using the
  CT affine and the same orientation transform used by the earlier validated
  coronal visualization helper.
- The anterior-posterior RAS axis is collapsed to produce a coronal view.
- CT uses either a lung-windowed 75th-percentile projection (`p75`) or a
  lung-windowed average-intensity projection (`mean`). These are less sensitive
  to isolated rib, spine, or vessel voxels than a full maximum-intensity CT
  projection.
- Binary GT and prediction masks use a true maximum projection, equivalent to
  `any` along the anterior-posterior axis.
- Display orientation follows radiology convention: superior is up and patient
  right is screen left. Every panel is marked `S`, `I`, `R`, and `L`.
- Dice labels remain the 3D Dice values. No projected Dice is reported because
  structures at different anterior-posterior depths can overlap after
  projection.

## Color Definition

> [!IMPORTANT]
> Prediction overlay colors are assigned per anterior-posterior projection ray
> after voxelwise 3D TP/FP/FN classification. Green wins whenever the ray
> contains any real voxelwise TP, so slight anterior-posterior over- or
> under-segmentation still shows overlap. Purple marks rays where FN and FP
> both occur at different anterior-posterior depths but no voxel overlaps.

| Color | Meaning |
| --- | --- |
| Green | Ray contains any real voxelwise TP, `gt & pred` |
| Red | No TP; ray contains FP only |
| Blue | No TP; ray contains FN only |
| Purple | No TP; ray contains depth-disjoint FN+FP |

## Layout

Each case produces one PNG per CT projection method. Each row is one finding,
so the fixed val200 set yields one to five rows per case. Columns are:

1. CT projection
2. GT mask projection
3. Public VoxTell v1.1
4. 100+100 attention: Exp009 `s3v1_fixedrho_suppress_half_quarter`, epoch 100
5. 100+100 plain: best no-DDP Exp009 continuation epoch 100, update 10,000
6. Best DDP: continuation relative epoch 50 / absolute epoch 150, update 5,000

Prediction panels use the four-color rule above and include the same legend in
every rendered figure.

## Reproducibility

All renderer and packaging code lives in this folder, so another clone can
regenerate the gallery without notebook state or `/tmp` helper scripts. The
required inputs are:

- the fixed val200 JSON, default
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`;
- MICCAI metadata, default
  `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`;
- GT segmentations, default
  `/data/hengjie/datasets/rexgroundingct/segmentations`;
- CT volumes, default
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`;
- four prediction directories for public VoxTell v1.1, 100+100 attention,
  100+100 plain, and Best DDP.

The renderer writes outside Git by default:

```text
Docker: /database/datasets/rexgroundingct/visualizations/
Host:   /data/hengjie/datasets/rexgroundingct/visualizations/
```

Set `REXGROUNDINGCT_VISUALIZATION_ROOT` to change the visualization root, or
pass `--output-dir` to choose the exact destination. If local data or
prediction mounts differ from the defaults, pass the corresponding `--val-json`,
`--metadata-json`, `--seg-dir`, `--ct-root`, or `--*-pred-dir` arguments.

Regenerate the full val200 mean gallery from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-07-31_visualdev_coronal_projection/coronal_projection.py \
  --mode val200-by-category
```

Then refresh the committed package:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-07-31_visualdev_coronal_projection/package_results.py
```

## Pilot

`pilot_cases.json` fixes 12 representative cases spanning one to five findings,
multiple challenge categories, fragmented masks, strong and weak predictions,
and cases favoring different model checkpoints.

The pilot is complete. See `pilot_review.md` for the output audit and the
recommendation to use the mean-intensity CT projection as the primary
full-val200 background.

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-07-31_visualdev_coronal_projection/coronal_projection.py
```

By default, generated PNGs, tables, and the run manifest are written outside
Git under the shared dataset visualization tree. The same location is:

```text
Docker: /database/datasets/rexgroundingct/visualizations/
          2026-07-31_visualdev_coronal_projection/pilot
Host:   /data/hengjie/datasets/rexgroundingct/visualizations/
          2026-07-31_visualdev_coronal_projection/pilot
```

The renderer selects the Docker path when `/database/datasets` is mounted and
otherwise uses the host path. Set `REXGROUNDINGCT_VISUALIZATION_ROOT` to
override this resolution explicitly.

Use `--dry-run` to validate the selected cases and input paths without loading
CT volumes or writing outputs.

## Full Val200 Mean Gallery by Category

The full gallery uses the same four prediction columns and orientation contract
as the pilot, but renders only the mean CT projection. It contains 342 figures
covering all 381 val200 findings. A case with findings in multiple categories
appears once in each relevant category folder, and each figure shows only the
findings assigned to that folder's category.

The output tree is:

```text
full_val200_mean_by_category/
  1a/
  1b/
  ...
  2e/
  2f_empty/
  2g/
  2h/
  category_case_index.csv
  category_case_index.md
  val200_finding_metrics.csv
  val200_finding_metrics.md
  run_manifest.json
```

Category folders contain only PNGs named with the fixed validation index and
full CT case stem, for example `1e/val000_train_19753_a_2.png`. Different scans
from the same patient remain separate figures. `2f_empty` is intentionally
present and empty because val200 has no category-2f findings.

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-07-31_visualdev_coronal_projection/coronal_projection.py \
  --mode val200-by-category
```

The output location is:

```text
Docker: /database/datasets/rexgroundingct/visualizations/
          2026-07-31_visualdev_coronal_projection/full_val200_mean_by_category
Host:   /data/hengjie/datasets/rexgroundingct/visualizations/
          2026-07-31_visualdev_coronal_projection/full_val200_mean_by_category
```

Full mode rejects any explicitly requested projection method other than
`mean`; it never creates a P75 directory or P75 figures.

## Committed Results Package

The curated gallery is packaged inside the repository at:

```text
results/
  README.md
  full_val200_mean_by_category/
    four_model_overall_metrics.csv
    four_model_category_metrics.csv
    category_case_index.csv
    val200_finding_metrics.csv
    run_manifest.json
    1a/README.md + PNGs
    ...
    2f_empty/README.md
    ...
    2h/README.md + PNGs
```

Start with the [results overview](results/README.md). It reports train,
validation, and test category counts and ratios; overall and per-category
val200 Dice/hit comparisons; model provenance; and links to the category
galleries. Category pages embed every figure in collapsed batches of at most
ten for practical GitHub viewing.

The canonical renderer outputs stay in the shared dataset tree. Package them
byte-for-byte, generate the derived tables and gallery pages, and verify the
full result contract with:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-07-31_visualdev_coronal_projection/package_results.py
```

Use `--verify-only` to recheck an existing package without writing. The 342
curated PNGs are the only generated visualization images tracked in this
repository and are stored through Git LFS.

## Verification

Run the focused synthetic tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s visualization/2026-07-31_visualdev_coronal_projection \
  -p 'test_*.py'
```

The tests cover projection behavior, radiology display orientation, the fixed
model order, category filtering, category-directory mapping, val-first figure
naming, the mean-only full-val contract, summary recomposition, gallery
batching, and the special zero-PNG `2f_empty` category.
