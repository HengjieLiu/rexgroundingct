# 1f — Other — Full-body CT mean

[← Results overview](../README.md) · [Other background](lung_only.md) · [Figure index](../figure_index.csv) · [Finding metrics](../finding_metrics.csv)

## Split distribution

| Split | Finding count | Within-split ratio |
| --- | --- | --- |
| Train | 150 | 1.95% |
| Val | 4 | 1.05% |
| Test | 4 | 0.69% |

## Val200 model/policy summary

A hit is full-volume 3D Dice ≥ 0.1. Category-best checkpoints are retrospective validation selections.

| Model row | Policy | Mean 3D Dice | Hits / findings | Hit rate |
| --- | --- | --- | --- | --- |
| Overall best | Raw | 0.212339 | 2/4 | 50.00% |
| Overall best | PP2 | 0.212339 | 2/4 | 50.00% |
| Overall best | PP3 (targeted corrections) | 0.212339 | 2/4 | 50.00% |
| Category best | Raw | 0.244367 | 2/4 | 50.00% |
| Category best | PP2 | 0.244367 | 2/4 | 50.00% |
| Category best | PP3 (targeted corrections) | 0.244367 | 2/4 | 50.00% |
| Iso07 best | Raw | 0.144214 | 1/4 | 25.00% |
| Iso07 best | PP2 | 0.144214 | 1/4 | 25.00% |
| Iso07 best | PP3 (targeted corrections) | 0.144214 | 1/4 | 25.00% |

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

## Figures (4)

<details>
<summary>Figures 1–4 of 4</summary>

### 1. val023_train_1957_a_1 · finding 0

Left main bronchus occluded

![1f val023_train_1957_a_1 · finding 0](full_body/val023_train_1957_a_1_finding0_1f.png)

[Open full-resolution PNG](full_body/val023_train_1957_a_1_finding0_1f.png)

### 2. val042_train_13092_a_2 · finding 2

Focal coarse calcification in the lower lobe pleura of the left lung

![1f val042_train_13092_a_2 · finding 2](full_body/val042_train_13092_a_2_finding2_1f.png)

[Open full-resolution PNG](full_body/val042_train_13092_a_2_finding2_1f.png)

### 3. val081_train_13471_a_1 · finding 0

Aeration differences observed bilaterally in the lower lobes

![1f val081_train_13471_a_1 · finding 0](full_body/val081_train_13471_a_1_finding0_1f.png)

[Open full-resolution PNG](full_body/val081_train_13471_a_1_finding0_1f.png)

### 4. val199_train_13177_a_2 · finding 1

Subcentimeter nonspecific parenchymal nodules in the upper lobes of both lungs

![1f val199_train_13177_a_2 · finding 1](full_body/val199_train_13177_a_2_finding1_1f.png)

[Open full-resolution PNG](full_body/val199_train_13177_a_2_finding1_1f.png)

</details>

