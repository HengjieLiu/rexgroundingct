# Full val200: whole-body and lung-only coronal comparison

**200 cases · 381 findings · 762 figures.** Each finding has one 3×5 figure per CT background, with overall-best, category-best, and iso07-best model rows and raw/PP2/PP3 prediction columns.

[Finding metrics](finding_metrics.csv) · [Figure index](figure_index.csv) · [Provenance](run_manifest.json) · [Renderer and usage](../README.md)

## Val200 model/policy comparison

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.346023 | 296/381 | 77.69% |
| Overall best | PP2 | 0.350762 | 297/381 | 77.95% |
| Overall best | PP3 (targeted corrections) | 0.354457 | 297/381 | 77.95% |
| Category best | Raw | 0.360278 | 298/381 | 78.22% |
| Category best | PP2 | 0.365156 | 300/381 | 78.74% |
| Category best | PP3 (targeted corrections) | 0.367893 | 300/381 | 78.74% |
| Iso07 best | Raw | 0.337526 | 290/381 | 76.12% |
| Iso07 best | PP2 | 0.346117 | 292/381 | 76.64% |
| Iso07 best | PP3 (targeted corrections) | 0.351670 | 291/381 | 76.38% |

The same full-volume 3D metrics apply to both backgrounds. A hit is Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections, not unbiased performance estimates.

## Category galleries

| Category | Findings | Full body | Lung only |
| --- | --- | --- | --- |
| 1a — Bronchial wall thickening | 3 | [Gallery](1a/full_body.md) | [Gallery](1a/lung_only.md) |
| 1b — Bronchiectasis | 11 | [Gallery](1b/full_body.md) | [Gallery](1b/lung_only.md) |
| 1c — Emphysema (including Centrilobular, Paraseptal, Bullous) | 17 | [Gallery](1c/full_body.md) | [Gallery](1c/lung_only.md) |
| 1d — Septal thickening (including Interlobular, Reticulation) | 6 | [Gallery](1d/full_body.md) | [Gallery](1d/lung_only.md) |
| 1e — Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 11 | [Gallery](1e/full_body.md) | [Gallery](1e/lung_only.md) |
| 1f — Other | 4 | [Gallery](1f/full_body.md) | [Gallery](1f/lung_only.md) |
| 2a — Linear (including subsegmental atelectasis, scarring, fibrosis) | 69 | [Gallery](2a/full_body.md) | [Gallery](2a/lung_only.md) |
| 2b — Atelectasis, consolidation | 49 | [Gallery](2b/full_body.md) | [Gallery](2b/lung_only.md) |
| 2c — Groundglass opacity | 60 | [Gallery](2c/full_body.md) | [Gallery](2c/lung_only.md) |
| 2d — Pulmonary nodules/masses | 132 | [Gallery](2d/full_body.md) | [Gallery](2d/lung_only.md) |
| 2e — Pleural effusion or thickening | 11 | [Gallery](2e/full_body.md) | [Gallery](2e/lung_only.md) |
| 2f — Honeycombing | 0 | [Gallery](2f_empty/full_body.md) | [Gallery](2f_empty/lung_only.md) |
| 2g — Pneumothorax | 1 | [Gallery](2g/full_body.md) | [Gallery](2g/lung_only.md) |
| 2h — Other | 7 | [Gallery](2h/full_body.md) | [Gallery](2h/lung_only.md) |

## Color definition

> [!IMPORTANT]
> Prediction overlay colors are assigned per AP projection ray after voxelwise
> 3D TP/FP/FN classification. Green wins whenever the ray contains any real
> voxelwise TP. Purple marks depth-disjoint FP and FN without voxelwise overlap.

| Color | Meaning |
| --- | --- |
| Green | Any voxelwise TP on the ray; takes priority |
| Red | FP only, without TP or FN |
| Blue | FN only, without TP or FP |
| Purple | Depth-disjoint FP + FN, without TP |


## Method and provenance

Whole-body CT uses the July mean projection. Lung-only CT averages only exact lung voxels (labels 10–14), with black empty rays. Both use a −600/1500 HU window, superior-up radiology orientation, native physical aspect ratio, and unrestricted GT/prediction overlays.

PP2 uses frozen whole-lung eligibility with a physical 20 mm margin. PP3 uses frozen fine-region routing with the approved correction for 15 bilateral upper-lobe findings, including seven newly fine-eligible findings. Corrected support is labels 10 and 12 plus 20 mm. All other routes and original experiment artifacts remain unchanged.

PNG files are configured for Git LFS. Derived NIfTIs and detailed case records remain outside the repository. The official anatomy source retains `PASS_PENDING_MANUAL_VISUAL_REVIEW`.

## Verification

All 200 cases passed geometry, source-composition, saved-policy, baseline-metric,
overlay-invariance, and layout checks. The package contains exactly 381 PNGs per
background, with valid relative links and batches of at most ten images.
All 15 focused tests passed. The routing audit confirmed 15 corrections,
seven newly enabled fine-region routes, and 366 unchanged routes.

Codex visually reviewed 21 PNGs covering every nonempty category, both pilot
findings and backgrounds, multiple findings, corrected/ineligible policies,
and the longest prompt. No display issues were found in this sample. Review
details and image hashes are in the provenance manifest; the official anatomy
source remains pending human review.

All 762 PNGs receive `filter=lfs` and are eligible for tracking. The complete
package is approximately 531 MiB. Staging, committing, and pushing remain for
the user.

## Regenerate or verify

```bash
python visualization/2026-09-09_coronal_lung_projection_comparison/render_comparison.py --mode val200-by-category --workers 4 --resume
python visualization/2026-09-09_coronal_lung_projection_comparison/render_comparison.py --mode val200-by-category --verify-only
```
