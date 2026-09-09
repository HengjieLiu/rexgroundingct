# 2g — Pneumothorax — Full-body CT mean

[← Results overview](../README.md) · [Other background](lung_only.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 18 | 0.23% |
| Val | 1 | 0.26% |
| Test | 1 | 0.17% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.048314 | 0/1 | 0.00% |
| Overall best | PP2 | 0.048314 | 0/1 | 0.00% |
| Overall best | PP3 (targeted corrections) | 0.048314 | 0/1 | 0.00% |
| Category best | Raw | 0.125141 | 1/1 | 100.00% |
| Category best | PP2 | 0.125141 | 1/1 | 100.00% |
| Category best | PP3 (targeted corrections) | 0.125141 | 1/1 | 100.00% |
| Iso07 best | Raw | 0.039908 | 0/1 | 0.00% |
| Iso07 best | PP2 | 0.039908 | 0/1 | 0.00% |
| Iso07 best | PP3 (targeted corrections) | 0.039908 | 0/1 | 0.00% |

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

## Figures (1)

<details>
<summary>Figures 1–1 of 1</summary>

### 1. val007_train_19891_a_2 · finding 0

Right pneumothorax with pleural air largely filling the right hemithorax, most pronounced at the upper lobe level, measuring approximately 60 mm in maximal thickness; not present on the prior examination

![2g val007_train_19891_a_2 · finding 0](full_body/val007_train_19891_a_2_finding0_2g.png)

[Open full-resolution PNG](full_body/val007_train_19891_a_2_finding0_2g.png)

</details>

