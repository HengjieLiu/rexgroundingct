<!-- Archive metadata in this block is not source content. -->
> **Archive metadata:** Source: [https://rexrank.ai/ReXGroundingCT/](https://rexrank.ai/ReXGroundingCT/) | Retrieved: `2026-07-22T00:09:05Z` | Raw source: [`snapshots/2026-07-22T000905Z/raw/rexrank.ai/ReXGroundingCT/index.html`](snapshots/2026-07-22T000905Z/raw/rexrank.ai/ReXGroundingCT/index.html) | SHA-256: `b5818cbbea4e6d9868a0fff6186a434cffa6b7539343de6b36dd3f98474a9704`
> **Archive note:** Interactive controls do not function in Markdown; the raw HTML remains authoritative for page behavior.

---
[Button: button]


- [ReXrankCT](https://rexrank.ai/ReXGroundingCT/#)
  - [ReXGroundingCT](https://rexrank.ai/ReXGroundingCT/index.html)
- [ReX-MLE](https://rexrank.ai/ReX-MLE/index.html)


 [ReXrank](https://rexrank.ai/)


 🏆 **ReXGroundingCT Challenge @ MICCAI 2026** — submit your model & see the live leaderboard! [Go to the Challenge →](https://rexrank.ai/ReXGroundingCT/challenge.html)


# ReXGroundingCT


## Segmentation of Findings from Free-Text Reports


### [⭐@Researchers: Submit to ReXrankCT](https://rexrank.ai/ReXGroundingCT/challenge.html#register)


### [Read the Paper](https://arxiv.org/abs/2507.22030)


## About ReXGroundingCT


ReXGroundingCT is a large-scale 3D chest CT dataset linking free-text radiology findings to pixel-level segmentations in volumetric imaging. It comprises 3,142 non-contrast chest CT scans with 8,028 annotated findings (16,301 entities) from the CT-RATE dataset. ReXGroundingCT enables sentence-level grounding for both focal and non-focal lung and pleural abnormalities across 14 categories. On ReXrank, we are hosting ReXGroundingCT's testset, which contains 100 CT scans with exhaustive radiologist annotations for all visible findings.


---


## Performance Metrics


**Global Dice:** Average Dice per finding per case


**Global HIT Rate:** Proportion of findings that have Dice >= 0.1


**Instance Precision:** TP / (TP + FP), where True Positives are instances that have a Dice >= 0.2


**Instance Recall:** TP / (TP + FN)


---


## Model Performance


| Model | Global Dice | Global HIT Rate | Instance Precision | Instance Recall | Instance F1 |
| --- | --- | --- | --- | --- | --- |
| [DAGG](https://arxiv.org/abs/2607.12602)<br>Gachon University<br>2/23/2026 | **0.253** | **0.517** | **0.211** | **0.299** | **0.247** |
| [VoxTell](https://arxiv.org/abs/2511.11450)<br>German Cancer Research Center (DKFZ)<br>Heidelberg University<br>3/24/2026 | **0.285** | **0.615** | **0.185** | **0.294** | **0.227** |
| [SAT-FT](https://arxiv.org/abs/2312.17183)<br>Shanghai Jiao Tong University | **0.205** | **0.473** | **0.062** | **0.383** | **0.107** |
| [BiomedParseV2](https://github.com/microsoft/BiomedParse/tree/v2)<br>Microsoft | **0.025** | **0.066** | **0.006** | **0.071** | **0.012** |
| MedCompose-CT<br>Microsoft Research<br>6/4/2026 | **0.297** | **0.601** | **0.127** | **0.304** | **0.179** |
| VoxTell-FT<br>HBKU<br>6/24/2026 | **0.162** | **0.371** | **0.031** | **0.277** | **0.056** |
| DVSM<br>7/13/2026 | **0.322** | **0.642** | **0.215** | **0.359** | **0.269** |


---


## Per-Category Results


 Model [Select: categoryModelSelect: DVSM, MedCompose-CT, VoxTell, DAGG, SAT-FT, VoxTell-FT, BiomedParseV2]

 Submission [Select: categorySubmissionSelect]


| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |


**Legend:** Category codes starting with **1** are typically non-focal lung/airway/pleural abnormalities; codes starting with **2** are typically focal lung/airway/pleural opacities.

## Archived Source Data

> **Archive note:** The page's per-category selector is populated by embedded JavaScript. The complete official CSV-backed values are expanded below; displayed metrics use the page's three-decimal formatting.

### Model Performance CSV

| Model | Institution | Version | Date | Global Dice | Global HIT Rate | Instance Precision | Instance Recall | Instance F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [DAGG](https://arxiv.org/abs/2607.12602) | Gachon University | -- | 2/23/2026 | 0.253 | 0.517 | 0.211 | 0.299 | 0.247 |
| [VoxTell](https://arxiv.org/abs/2511.11450) | German Cancer Research Center (DKFZ) & Heidelberg University | -- | 3/24/2026 | 0.285 | 0.615 | 0.185 | 0.294 | 0.227 |
| [SAT-FT](https://arxiv.org/abs/2312.17183) | Shanghai Jiao Tong University | -- | -- | 0.205 | 0.473 | 0.062 | 0.383 | 0.107 |
| [BiomedParseV2](https://github.com/microsoft/BiomedParse/tree/v2) | Microsoft | -- | -- | 0.025 | 0.066 | 0.006 | 0.071 | 0.012 |
| MedCompose-CT | Microsoft Research | -- | 6/4/2026 | 0.297 | 0.601 | 0.127 | 0.304 | 0.179 |
| VoxTell-FT | HBKU | -- | 6/24/2026 | 0.162 | 0.371 | 0.031 | 0.277 | 0.056 |
| DVSM | -- | -- | 7/13/2026 | 0.322 | 0.642 | 0.215 | 0.359 | 0.269 |

### Per-Category Results

<details>
<summary>DAGG</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 1 | 0.000 | 0.000 |
| **1b** - Bronchiectasis | 4 | 0.041 | 0.000 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 17 | 0.246 | 0.529 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.049 | 0.167 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 7 | 0.110 | 0.429 |
| **1f** - Other | 3 | 0.009 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 49 | 0.191 | 0.531 |
| **2b** - Atelectasis, consolidation | 36 | 0.290 | 0.583 |
| **2c** - Groundglass opacity | 36 | 0.224 | 0.472 |
| **2d** - Pulmonary nodules/masses | 58 | 0.287 | 0.638 |
| **2e** - Pleural effusion or thickening | 7 | 0.140 | 0.286 |
| **2f** - Honeycombing | 0 | 0.000 | 0.000 |
| **2g** - Pneumothorax | 1 | 0.058 | 0.000 |
| **2h** - Other | 1 | 0.434 | 1.000 |

</details>

<details>
<summary>VoxTell</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 1 | 0.000 | 0.000 |
| **1b** - Bronchiectasis | 4 | 0.127 | 0.500 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 17 | 0.088 | 0.176 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.027 | 0.167 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 7 | 0.209 | 0.429 |
| **1f** - Other | 3 | 0.182 | 0.333 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 49 | 0.237 | 0.673 |
| **2b** - Atelectasis, consolidation | 36 | 0.294 | 0.667 |
| **2c** - Groundglass opacity | 36 | 0.316 | 0.667 |
| **2d** - Pulmonary nodules/masses | 58 | 0.299 | 0.741 |
| **2e** - Pleural effusion or thickening | 7 | 0.293 | 0.429 |
| **2f** - Honeycombing | 0 | 0.000 | 0.000 |
| **2g** - Pneumothorax | 1 | 0.799 | 1.000 |
| **2h** - Other | 1 | 0.559 | 1.000 |

</details>

<details>
<summary>SAT-FT</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 1 | 0.051 | 0.000 |
| **1b** - Bronchiectasis | 4 | 0.036 | 0.000 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 17 | 0.011 | 0.059 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.053 | 0.167 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 7 | 0.153 | 0.429 |
| **1f** - Other | 3 | 0.132 | 0.333 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 49 | 0.119 | 0.388 |
| **2b** - Atelectasis, consolidation | 36 | 0.250 | 0.583 |
| **2c** - Groundglass opacity | 36 | 0.280 | 0.556 |
| **2d** - Pulmonary nodules/masses | 58 | 0.191 | 0.638 |
| **2e** - Pleural effusion or thickening | 7 | 0.135 | 0.429 |
| **2f** - Honeycombing | 0 | 0.000 | 0.000 |
| **2g** - Pneumothorax | 1 | 0.064 | 0.000 |
| **2h** - Other | 1 | 0.283 | 1.000 |

</details>

<details>
<summary>BiomedParseV2</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 1 | 0.031 | 0.000 |
| **1b** - Bronchiectasis | 4 | 0.001 | 0.000 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 17 | 0.002 | 0.000 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.009 | 0.000 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 7 | 0.000 | 0.000 |
| **1f** - Other | 3 | 0.139 | 0.667 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 49 | 0.006 | 0.000 |
| **2b** - Atelectasis, consolidation | 36 | 0.040 | 0.111 |
| **2c** - Groundglass opacity | 36 | 0.060 | 0.194 |
| **2d** - Pulmonary nodules/masses | 58 | 0.000 | 0.000 |
| **2e** - Pleural effusion or thickening | 7 | 0.047 | 0.143 |
| **2f** - Honeycombing | 0 | 0.000 | 0.000 |
| **2g** - Pneumothorax | 1 | 0.081 | 0.000 |
| **2h** - Other | 1 | 0.133 | 1.000 |

</details>

<details>
<summary>MedCompose-CT</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 1 | 0.142 | 1.000 |
| **1b** - Bronchiectasis | 4 | 0.106 | 0.500 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 17 | 0.050 | 0.118 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.091 | 0.167 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 7 | 0.209 | 0.571 |
| **1f** - Other | 3 | 0.190 | 0.333 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 49 | 0.225 | 0.612 |
| **2b** - Atelectasis, consolidation | 36 | 0.285 | 0.583 |
| **2c** - Groundglass opacity | 36 | 0.280 | 0.583 |
| **2d** - Pulmonary nodules/masses | 58 | 0.345 | 0.810 |
| **2e** - Pleural effusion or thickening | 7 | 0.282 | 0.571 |
| **2f** - Honeycombing | 0 | 0.000 | 0.000 |
| **2g** - Pneumothorax | 1 | 0.620 | 1.000 |
| **2h** - Other | 1 | 0.494 | 1.000 |

</details>

<details>
<summary>VoxTell-FT</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 1 | 0.006 | 0.000 |
| **1b** - Bronchiectasis | 4 | 0.008 | 0.000 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 17 | 0.005 | 0.000 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.034 | 0.167 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 7 | 0.107 | 0.286 |
| **1f** - Other | 3 | 0.070 | 0.333 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 49 | 0.067 | 0.224 |
| **2b** - Atelectasis, consolidation | 36 | 0.133 | 0.417 |
| **2c** - Groundglass opacity | 36 | 0.156 | 0.389 |
| **2d** - Pulmonary nodules/masses | 58 | 0.236 | 0.638 |
| **2e** - Pleural effusion or thickening | 7 | 0.129 | 0.429 |
| **2f** - Honeycombing | 0 | 0.000 | 0.000 |
| **2g** - Pneumothorax | 1 | 0.052 | 0.000 |
| **2h** - Other | 1 | 0.026 | 0.000 |

</details>

<details>
<summary>DVSM</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 1 | 0.039 | 0.000 |
| **1b** - Bronchiectasis | 4 | 0.047 | 0.000 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 17 | 0.084 | 0.235 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.142 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 7 | 0.225 | 0.429 |
| **1f** - Other | 3 | 0.201 | 0.333 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 49 | 0.284 | 0.714 |
| **2b** - Atelectasis, consolidation | 36 | 0.342 | 0.694 |
| **2c** - Groundglass opacity | 36 | 0.305 | 0.694 |
| **2d** - Pulmonary nodules/masses | 58 | 0.358 | 0.759 |
| **2e** - Pleural effusion or thickening | 7 | 0.327 | 0.571 |
| **2f** - Honeycombing | 0 | 0.000 | 0.000 |
| **2g** - Pneumothorax | 1 | 0.638 | 1.000 |
| **2h** - Other | 1 | 0.610 | 1.000 |

</details>
