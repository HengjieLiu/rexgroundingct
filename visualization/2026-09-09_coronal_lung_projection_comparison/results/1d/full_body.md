# 1d — Septal thickening (including Interlobular, Reticulation) — Full-body CT mean

[← Results overview](../README.md) · [Other background](lung_only.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 194 | 2.52% |
| Val | 6 | 1.57% |
| Test | 9 | 1.55% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.116696 | 1/6 | 16.67% |
| Overall best | PP2 | 0.116698 | 1/6 | 16.67% |
| Overall best | PP3 (targeted corrections) | 0.116698 | 1/6 | 16.67% |
| Category best | Raw | 0.146813 | 1/6 | 16.67% |
| Category best | PP2 | 0.147005 | 1/6 | 16.67% |
| Category best | PP3 (targeted corrections) | 0.146845 | 1/6 | 16.67% |
| Iso07 best | Raw | 0.118833 | 1/6 | 16.67% |
| Iso07 best | PP2 | 0.118966 | 1/6 | 16.67% |
| Iso07 best | PP3 (targeted corrections) | 0.118833 | 1/6 | 16.67% |

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

## Figures (6)

<details>
<summary>Figures 1–6 of 6</summary>

### 1. val101_train_13398_a_1 · finding 0

Mosaic attenuation pattern in both lung parenchyma

![1d val101_train_13398_a_1 · finding 0](full_body/val101_train_13398_a_1_finding0_1d.png)

[Open full-resolution PNG](full_body/val101_train_13398_a_1_finding0_1d.png)

### 2. val106_train_13111_a_2 · finding 0

Smooth interlobular septal thickening in the peripheral areas, more prominent in the lower lobes of both lungs

![1d val106_train_13111_a_2 · finding 0](full_body/val106_train_13111_a_2_finding0_1d.png)

[Open full-resolution PNG](full_body/val106_train_13111_a_2_finding0_1d.png)

### 3. val115_train_2694_a_2 · finding 1

Areas of crazy paving in the lungs

![1d val115_train_2694_a_2 · finding 1](full_body/val115_train_2694_a_2_finding1_1d.png)

[Open full-resolution PNG](full_body/val115_train_2694_a_2_finding1_1d.png)

### 4. val116_train_13119_c_1 · finding 0

Interlobar septal thickening in both lungs

![1d val116_train_13119_c_1 · finding 0](full_body/val116_train_13119_c_1_finding0_1d.png)

[Open full-resolution PNG](full_body/val116_train_13119_c_1_finding0_1d.png)

### 5. val121_train_13252_b_2 · finding 3

Peribronchial reticulonodular densities with faint borders in the lower lobes

![1d val121_train_13252_b_2 · finding 3](full_body/val121_train_13252_b_2_finding3_1d.png)

[Open full-resolution PNG](full_body/val121_train_13252_b_2_finding3_1d.png)

### 6. val195_train_2577_a_1 · finding 1

Interlobular septal thickening in both lungs, most prominent in basal segments of the upper lobes, right middle lobe, and lower lobes bilaterally

![1d val195_train_2577_a_1 · finding 1](full_body/val195_train_2577_a_1_finding1_1d.png)

[Open full-resolution PNG](full_body/val195_train_2577_a_1_finding1_1d.png)

</details>

