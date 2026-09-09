# 2f — Honeycombing — Lung-only CT mean

[← Results overview](../README.md) · [Other background](full_body.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 16 | 0.21% |
| Val | 0 | 0.00% |
| Test | 0 | 0.00% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | — | — | — |
| Overall best | PP2 | — | — | — |
| Overall best | PP3 (targeted corrections) | — | — | — |
| Category best | Raw | — | — | — |
| Category best | PP2 | — | — | — |
| Category best | PP3 (targeted corrections) | — | — | — |
| Iso07 best | Raw | — | — | — |
| Iso07 best | PP2 | — | — | — |
| Iso07 best | PP3 (targeted corrections) | — | — | — |

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


Lung restriction affects only the CT background. All GT/prediction voxels remain visible. PP2 follows frozen whole-lung eligibility; PP3 follows frozen side/lobe routing except the 15 approved bilateral upper-lobe corrections. Ineligible policies leave predictions unchanged.

## Figures (0)

There are no category-2f validation findings. This gallery intentionally contains zero PNGs.
