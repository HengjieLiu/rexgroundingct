# ReXGroundingCT Current Summary

<!-- This digest is archive-generated. Follow links to literal transcriptions and raw evidence. -->
> Snapshot: [`snapshots/2026-07-22T001343Z/`](snapshots/2026-07-22T001343Z/) | Extracted: `2026-07-22T00:13:43Z`

## Important Source Differences

- **Submission route:** [`rexgroundingct_challenge.md`](rexgroundingct_challenge.md) says its authenticated Register & Submit form is the only submission channel for the MICCAI challenge. [`rexrankct_submission_guideline.md`](rexrankct_submission_guideline.md) separately retains email-submission instructions. Both are transcribed unchanged; use the challenge page for the current MICCAI submission channel and the guideline as its linked prediction-format reference.
- **Dataset splits:** The base ReXrank page describes a 100-scan hosted test set. The MICCAI challenge page describes separate 200-scan validation and 300-scan test splits. The official Hugging Face update record says the MICCAI split was added separately while the base `dataset.json` remained unchanged.
- **Two leaderboards:** The main ReXGroundingCT page contains the established model benchmark. The MICCAI challenge page loads a separate live public-split leaderboard from Firestore.

## Dataset Overview

## About ReXGroundingCT


ReXGroundingCT is a large-scale 3D chest CT dataset linking free-text radiology findings to pixel-level segmentations in volumetric imaging. It comprises 3,142 non-contrast chest CT scans with 8,028 annotated findings (16,301 entities) from the CT-RATE dataset. ReXGroundingCT enables sentence-level grounding for both focal and non-focal lung and pleural abnormalities across 14 categories. On ReXrank, we are hosting ReXGroundingCT's testset, which contains 100 CT scans with exhaustive radiologist annotations for all visible findings.


---

## Challenge Task

### 🔬 Task


Participants are evaluated on **free-text finding grounding**: a model receives a CT volume and a natural-language finding from a radiology report and must output a 3D segmentation mask corresponding to that description. The challenge is organized into two tracks: a *Main track* that consumes the free-text finding directly, and a more general *Overall track* (see [Challenge Tracks](https://rexrank.ai/ReXGroundingCT/challenge.html#tracks) below). In both tracks, fixed-category (class-based) segmentation that cannot separate two findings in different locations is out of scope and will be disqualified.


Findings span **14 categories** covering both typically non-focal abnormalities (bronchial wall thickening, bronchiectasis, emphysema, septal thickening, micronodules, and other diffuse abnormalities) and typically focal abnormalities (linear opacities, atelectasis/consolidation, ground-glass opacities, pulmonary nodules/masses, pleural effusion/thickening, honeycombing, pneumothorax, and other focal findings).

## Challenge Tracks

### 🎯 Challenge Tracks


The challenge is structured around two tracks, each with its own set of winners. Every eligible *Main-track* submission is automatically considered for the *Overall track* as well, so a strong free-text method can place in both.


**1. Main Track: Free-Text Grounding.** Methods in this track must consume the exact free-text finding as provided, conditioning the segmentation directly on that text (e.g., as a text embedding). The raw prompt is not rewritten into a structured representation before it reaches the model. This is the original goal of the challenge: flexible models that ground free-text findings natively.


**2. Overall Track: Free-Text & Structured.** This more general track admits both free-text methods and methods that transform the free-text finding into a structured representation (for example, parsing out location, morphology, or size with an LLM or text parser) and condition on that structured form. Every eligible Main-track submission is automatically considered here too, so a strong free-text method can place in both.


In both tracks, a model must take in more than just the finding itself as a prompt (a prompt cannot simply be “nodule”), and must be able to produce distinct segmentation masks for two findings in separate locations.


**Category information (both tracks).** Using a finding's category as an auxiliary signal is permitted in either track, whether the category is provided as part of the official test input or inferred from the free-text prompt (e.g., via a prompt-to-category classifier). For the Main track, the only requirement is that the exact free-text finding is still consumed directly by the model; the category supplements it rather than replaces it.


**Also permitted (both tracks).**


- Anatomical segmentations (lobes, lung, pleura, etc.) as additional input channels or spatial constraints for a text-grounded prediction.
- Post-processing of a text-grounded prediction, such as restricting the mask to the anatomical region named in the finding.


**A quick illustration.**


- **Allowed (Main track):** encode “small nodule in the right upper lobe” as a text embedding, condition the segmentation on it, optionally add the inferred category “nodule” as an auxiliary input, and constrain the output to the right upper lobe.
- **Allowed (Overall track only):** parse the same finding into `{location: right upper lobe, morphology: nodular, size: small}` and condition on those structured fields instead of the raw text embedding.
- **Out of scope (both tracks):** output a fixed set of class masks and select “nodule” by label, with no way to separate two nodules in different locations.


There will be three winners across the two tracks. Because Main-track methods are also considered for the Overall track, all three could end up being the same methods.

## MICCAI Dataset Splits

### 📦 Dataset


| Split | Cases | Annotations |
| --- | --- | --- |
| **Training** | 2,992 CT scans | Partial-instance (up to 3 instances per finding) |
| **Validation** | 200 CT scans | Exhaustive (all instances segmented by radiologists) |
| **Test** | 300 CT scans | Exhaustive (all instances segmented by radiologists) |


All annotations are pixel-level 3D segmentation masks linked to free-text findings extracted from radiology reports. Validation and test sets are annotated exclusively by board-certified radiologists.

## Timeline

### 📅 Timeline


Until June 2026

Pre-registration — training data publicly available


June 2026

Challenge launched — registration open, test set released


June — September 2026 — in progress

Submission phase (open now) — submit multiple runs; each is evaluated on the held-out test set and appears on the leaderboard


September 2026

Submission deadline — final leaderboard standings locked as the official results


Late September 2026

Results announced & challenge session at MICCAI 2026

## Evaluation Metrics

### 📊 Evaluation Metrics


**Ranking metric:** Average Dice Similarity Coefficient (DSC) per finding per case.


**Overlap-based metrics:**


- **Dice (primary):** Average DSC computed per finding per case
- **Hit Rate:** Proportion of findings where overall Dice ≥ 0.1
- **Instance Precision:** TP / (TP + FP), where TP is a predicted instance with Dice ≥ 0.2
- **Instance Recall:** TP / (TP + FN)
- **Instance F1:** Harmonic mean of Instance Precision and Recall


**Distance-based metrics:**


- **Distance Precision:** TP / (TP + FP), where TP is a predicted instance with ASSD (non-focal) or centroid distance (focal) ≤ 2× max voxel spacing
- **Distance Recall:** TP / (TP + FN), using the same distance matching criterion
- **Distance F1:** Harmonic mean of Distance Precision and Recall

## Main ReXGroundingCT Model Benchmark

| Model | Institution | Version | Date | Global Dice | Global HIT Rate | Instance Precision | Instance Recall | Instance F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [DAGG](https://arxiv.org/abs/2607.12602) | Gachon University | -- | 2/23/2026 | 0.253 | 0.517 | 0.211 | 0.299 | 0.247 |
| [VoxTell](https://arxiv.org/abs/2511.11450) | German Cancer Research Center (DKFZ) & Heidelberg University | -- | 3/24/2026 | 0.285 | 0.615 | 0.185 | 0.294 | 0.227 |
| [SAT-FT](https://arxiv.org/abs/2312.17183) | Shanghai Jiao Tong University | -- | -- | 0.205 | 0.473 | 0.062 | 0.383 | 0.107 |
| [BiomedParseV2](https://github.com/microsoft/BiomedParse/tree/v2) | Microsoft | -- | -- | 0.025 | 0.066 | 0.006 | 0.071 | 0.012 |
| MedCompose-CT | Microsoft Research | -- | 6/4/2026 | 0.297 | 0.601 | 0.127 | 0.304 | 0.179 |
| VoxTell-FT | HBKU | -- | 6/24/2026 | 0.162 | 0.371 | 0.031 | 0.277 | 0.056 |
| DVSM | -- | -- | 7/13/2026 | 0.322 | 0.642 | 0.215 | 0.359 | 0.269 |

## MICCAI Public-Split Leaderboard

| # | Submission | Team | Submitter | Institution | Dice | Hit Rate | Instance F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Grounder_1.1 | AIMHI | Kwang-Hyun Uhm | Gachon  University | 0.332 | 0.745 | 0.329 |
| 2 | JECT_base | iRAIL | hanbinko | SeoulNationalUniversity | 0.315 | 0.722 | 0.306 |
| — | Grounder | AIMHI | Kwang-Hyun Uhm | Gachon  University | 0.314 | 0.702 | 0.276 |
| 3 | MUW-GroundCT-v3 | MUW | Ronald Fecso | Medical University of Vienna | 0.300 | 0.695 | 0.244 |
| 4 | ThoraxTell | DKFZ | Moritz Langenberg | DKFZ Heidelberg | 0.296 | 0.689 | 0.163 |
| — | MUW-GroundCT-v2 | MUW | Ronald Fecso | Medical University of Vienna | 0.277 | 0.642 | 0.267 |
| 5 | VoXTell-FT (submission 2) | HBKU | Syed Abdullah Basit | Hamad Bin Khalifa University | 0.276 | 0.642 | 0.153 |
| 6 | VoxTell-FT | MICLab | Kristhian Aguilar | UNICAMP | 0.265 | 0.599 | 0.111 |
| — | JECT | iRAIL | hanbinko | SeoulNationalUniversity | 0.248 | 0.656 | 0.263 |
| — | MUW-GroundCT | MUW | Ronald Fecso | Medical University of Vienna | 0.242 | 0.573 | 0.186 |
| — | MUW-Text2UNet-v2 | MUW | Ronald Fecso | Medical University of Vienna | 0.188 | 0.530 | 0.122 |
| 7 | FARM-VoxTell-FT-Test | FARM | Jie Xu | Fudan Univerisity | 0.163 | 0.424 | 0.062 |
| — | FARM-VoxTell-FT | FARM | Jie Xu | Fudan Univerisity | 0.163 | 0.424 | 0.062 |
| 8 | VoXTell-FT | — | Abdullah | HBKU | 0.153 | 0.391 | 0.049 |
| — | FARM-VoxTell-FT-HZ-2 | FARM | Jie Xu | Fudan Univerisity | 0.146 | 0.440 | 0.074 |
| — | FARM-VoxTell-FT-HZ | FARM | Jie Xu | Fudan Univerisity | 0.146 | 0.440 | 0.074 |
| — | MUW-Text2UNet | MUW | Ronald Fecso | Medical University of Vienna | 0.106 | 0.301 | 0.041 |

## Official Companion Sources

- [Hugging Face dataset card](huggingface_dataset_card.md)
- [Dataset-generation repository](github_dataset_generation.md)
- [Paper record](arxiv_paper_record.md)
- [MICCAI 2026 challenge listing](miccai_2026_challenge_listing.md)
