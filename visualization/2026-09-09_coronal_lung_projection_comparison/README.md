# Coronal whole-volume versus lung-only CT backgrounds

CPU-only extension of the July 31 coronal renderer, with full-val200 category
galleries and a preserved single-case mode. The default single case
is `val189_train_3026_a_2`, with separate figures for finding 0 (1c,
emphysema) and finding 1 (2a, fibrotic scarring). Each finding produces two
3×5 figures: whole-volume CT mean and exact-lung-only CT mean. Lung restriction
changes only the background; all GT and prediction errors remain visible.

## Layout and methods

The first row contains CT, GT, and three Exp007 prediction panels. Rows two
and three leave the first two cells blank and show category-best and iso07-best
predictions. In the pilot, prediction columns are raw, PP2 whole lungs +20 mm, and
**PP3 upper lobes +20 mm — corrected routing**. Both supports are intersected
independently with raw predictions in native CT space.

The frozen Exp024/025 routes interpreted both pilot prompts as both whole lungs,
so their original PP3 equals PP2 here. This pilot intentionally corrects PP3
to the left/right upper-lobe union (official labels 10 and 12). It preserves
all original routes, predictions, and reports. This correction is specific to
prompts that explicitly describe the upper lobes of both lungs.

The exact lung CT background uses labels 10–14, with no dilation. Windowed CT
values are averaged over lung voxels on each AP ray; rays with no lung are
black. The window is center −600 HU / width 1500 HU. Native physical aspect
ratio, superior-up/patient-right-on-screen-left orientation, and July overlay
colors/opacity are retained. Green wins for any voxelwise TP on a ray; purple
means depth-disjoint FP and FN without TP. Panel Dice is full-volume 3D Dice.
Released GT channels contain positive instance IDs; use their union (`GT > 0`)
as in the July renderer. Prediction inputs must already be binary.

## Inputs and provenance

Runtime experiment root:
`/mnt/shengdata1/hengjie/experiments/rexgroundingct`.

- Overall row: Exp007 continuation relative e50 / absolute e150, frozen full
  val200 Dice 0.3460234857111323. Raw masks come from its original evaluation;
  PP2 and corrected PP3 are reconstructed because Exp022 saved measurements
  but no processed NIfTIs.
- Category row: Exp024 `validation/b1` and `validation/b2_prompt_lung20`.
  Finding 0 uses Exp017 relative e75 / absolute e175 (1c Dice 0.2078041621);
  finding 1 uses Exp009 `s3v2_balanced_feature_half_quarter` e100 (2a Dice
  0.3444508345). These are retrospective validation-selected models.
- Iso07 row: Exp025 `validation/baseline` and `validation/whole_lung20`, from
  Exp017 relative e50 / absolute e150, full val200 Dice 0.33752638623854136.
- Official anatomy is resolved through Exp020's private native-geometry audit;
  its source status remains `PASS_PENDING_MANUAL_VISUAL_REVIEW`.
- Model identities, source paths, hashes, frozen routes, correction, grid
  checks, metrics, and output hashes are recorded in the external manifest.

## Usage

Use Python with NumPy, SciPy, nibabel, Matplotlib, and pandas (available in the
existing `rexgroundingct-voxtell:cu126` image); no GPU is required.

### Full-val200 category galleries

The full mode renders 200 cases / 381 findings as 762 PNGs under `results/`.
Each official category has separate `full_body.md` and `lung_only.md` pages
with embedded images in collapsible groups of ten, split distributions, and
nine model/policy metric rows. Category 2f has two empty pages and no PNGs.

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-09-09_coronal_lung_projection_comparison/render_comparison.py \
  --mode val200-by-category --dry-run

PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-09-09_coronal_lung_projection_comparison/render_comparison.py \
  --mode val200-by-category --workers 4 --resume

PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-09-09_coronal_lung_projection_comparison/render_comparison.py \
  --mode val200-by-category --verify-only
```

Full mode preserves the frozen Exp024/025 routes except 15 explicitly approved
bilateral upper-lobe findings, including seven newly enabled fine-region
routes. The case-insensitive phrases are `upper lobes of both lungs`,
`both upper lobes`, and `bilateral upper lobes`. Corrected PP3 uses labels 10/12
with the existing 20 mm physical dilation. All other routes are unchanged.
Panel captions name the effective support or explicitly indicate an unchanged,
ineligible policy. See the [category gallery index](results/README.md) for the
browsable package. This full-cohort correction supersedes the pilot's narrower
exact-phrase restriction only in full mode.

PNG gallery output is the approved repository-local artifact exception and is
configured for Git LFS. Derived masks and detailed per-case records go to the
external shared visualization root under this folder's name and
`full_val200_runtime/`, configurable with `--runtime-dir`. Resume verifies code,
settings, input hashes, routing, and artifact hashes before reusing a complete
case. Each CT/background and each required anatomical support is reused within
its case. Shared checkpoints are hashed once per invocation.

### Single-case pilot

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-09-09_coronal_lung_projection_comparison/render_comparison.py \
  --dry-run

PYTHONDONTWRITEBYTECODE=1 python \
  visualization/2026-09-09_coronal_lung_projection_comparison/render_comparison.py
```

Options include `--case`, `--findings 0 1`, `--output-dir`, `--runtime-root`,
`--ct-root`, `--seg-dir`, `--metadata-json`, and `--val-json`. The default
output directory is the shared dataset visualization root followed by
`2026-09-09_coronal_lung_projection_comparison/val189_train_3026_a_2`.
`REXGROUNDINGCT_VISUALIZATION_ROOT` overrides the shared root. Other cases are
accepted only when the selected prompts meet this pilot's bilateral upper-lobe
contract and all frozen input sources exist. The dry run validates file/header
and routing inputs without reading volume payloads or writing outputs.

Outputs: four PNGs by default, `metrics.csv`, `run_manifest.json`, and derived
native finding-first NIfTIs under `derived_masks/`. With a finding subset,
derived files contain only selected channels; their original IDs are recorded
in the manifest. Generated outputs remain outside Git.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s visualization/2026-09-09_coronal_lung_projection_comparison -p 'test_*.py'
```

The renderer verifies saved PP2 against reconstruction, category composition
against its source checkpoint masks, frozen original PP3 behavior, original
3D metrics, corrected Exp007 PP3 against Exp022, and identical overlay arrays
and Dice across both backgrounds. Full mode also verifies the package census,
PNG integrity, Markdown links, image batches, and all nine metric combinations.
Visual review uses the pilot and a deterministic sample covering all nonempty
categories, multiple findings, corrected and ineligible routes, and long prompts.

## Completed full-val200 galleries

The [category gallery index](results/README.md) contains all 200 cases / 381
findings: 381 PNGs per background, 762 total, at 3600×2160 pixels. The package
has 28 category pages, a 762-row figure index, 3,429 model/policy metric rows,
and compact provenance. Its total size is approximately 531 MiB.

All 15 focused tests and full-cohort verification gates passed. The final audit
confirmed exactly 15 corrected routes, seven newly enabled fine-region routes,
366 unchanged routes, and matching hashes for all 800 external derived masks.
Every PNG is unignored and receives `filter=lfs`. No files were staged,
committed, or pushed.

Codex visually inspected 21 PNGs covering all 13 nonempty categories, the
four pilot figures, multiple findings, corrected and ineligible policies,
and the longest prompt in both backgrounds. No orientation, readability, or
clipping issues were found in this sample. The review and PNG hashes are
recorded in [the manifest](results/run_manifest.json). This display review
retains the official anatomy source's `PASS_PENDING_MANUAL_VISUAL_REVIEW` status.

## Completed pilot

Generated and visually inspected all four 3600×2160 PNGs on 2026-09-09.
Eight focused tests passed, including the positive-instance GT regression.
Source/checkpoint hashes, category source-channel equality, native geometry,
saved PP2 reconstruction, original PP3 equivalence, published metrics, corrected
Exp007 PP3, derived-mask round trips, and cross-background overlay invariance
all passed. No model inference was run.

The results index, four images, four derived masks, 18-row metrics CSV, and
manifest with the completed rendering review are at:

```text
/data/hengjie/datasets/rexgroundingct/visualizations/
2026-09-09_coronal_lung_projection_comparison/val189_train_3026_a_2/
```

The renderer used the existing CPU-capable Docker image
`sha256:8ff421d05fbf6044553ba987d065a4d6fdaaaab8e2913272e620d08c4c286e0f`.
See the [Docker environment guide](../../docker/voxtell/README.md) for the
host UID/GID and input mounts. Set `MPLCONFIGDIR=/tmp/matplotlib` and
`PYTHONDONTWRITEBYTECODE=1` when mounting the repository read-only; no GPU mount
is needed. The output visualization tree must be writable.

Some processed panels are identical by construction or because no predicted
voxels fall outside the selected support. In this pilot, the category-best
emphysema mask and both iso07 masks are unchanged by either policy. Identical
panels were retained and verified, rather than introducing an additional
postprocessing rule to force a visual difference. The figure review does not
change the official anatomy source's pending-human-review status.
