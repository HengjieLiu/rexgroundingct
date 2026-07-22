<!-- Archive metadata in this block is not source content. -->
> **Archive metadata:** Source: [https://rexrank.ai/ReXGroundingCT/challenge.html](https://rexrank.ai/ReXGroundingCT/challenge.html) | Retrieved: `2026-07-22T00:09:05Z` | Raw source: [`snapshots/2026-07-22T000905Z/raw/rexrank.ai/ReXGroundingCT/challenge.html`](snapshots/2026-07-22T000905Z/raw/rexrank.ai/ReXGroundingCT/challenge.html) | SHA-256: `3b1264e89db26266254c3b4912d2d5e3a533ee53944405d801d341f69d85eb1d`
> **Archive note:** Interactive controls are represented as text; the public runtime leaderboard is transcribed below from its archived JSON payload.

---
[Button: button]


- [ReXrankCT](https://rexrank.ai/ReXGroundingCT/challenge.html#)
  - [ReXGroundingCT](https://rexrank.ai/ReXGroundingCT/index.html)
- [ReX-MLE](https://rexrank.ai/ReX-MLE/index.html)


 [ReXrank](https://rexrank.ai/)


# ReXGrounding Challenge


## MICCAI 2026


### Grounding Free-Text Findings to 3D CT Segmentations


### 📊 Leaderboard


Live standings on a **public 50% subset** of the held-out test set, ranked by mean Dice. The other 50% is withheld — **final results on the full test set will be revealed at MICCAI 2026.** Each submitter is ranked by their single best submission. **Click any row for its per-category breakdown.** Updated automatically as submissions are evaluated.


Loading leaderboard…


| # | Submission | Team | Submitter | Institution | Dice | Hit Rate | Instance F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |


---


### 🚀 Register & Submit


Registration is **open**. **All submissions are made here, through the Register & Submit form below** — create an account, register your team, then submit your predictions as a Google Drive link. This is the only submission channel.


The [submission guidelines](https://rexrank.ai/explore/submission_guideline_ct.html) are a reference for the required prediction *format* only — they are not a separate way to submit. Review them so your predictions are formatted correctly, then submit through the form below.


 [Button: Log In] [Button: Create Account]

 [Input: Email] [Input: Password] [Button: Log In]

[Forgot password?](https://rexrank.ai/ReXGroundingCT/challenge.html#)


 [Input: Your name (optional)] [Input: Email] [Input: Password (min 6 characters)] [Button: Create Account]

You'll get a verification email — **check your spam/junk folder** if it's not in your inbox — verify it before logging in.


 [Button: Resend verification email]


Signed in as · [Log out](https://rexrank.ai/ReXGroundingCT/challenge.html#)


#### Register your team

 [Input: Team name] [Input: Contact email] Members (name & affiliation):

 [Button: + Add member] [Button: Register Team]


#### Team:


Your predictions are evaluated on the **held-out test set** and appear on the leaderboard as official results.

 [Input: Model name (required)] [Input: Google Drive link to a single .zip of your predictions]

**⚠️ Submit a single `.zip` file** (not a folder), containing your 300 prediction `.nii.gz` files. **Set sharing to "Anyone with the link"** before submitting — restricted links, folders, and access-requested links cannot be evaluated automatically.

 [Textarea: Notes (optional)]

⏱️ Evaluation runs automatically — **expect your result on the leaderboard within ~2 hours (max)**. Your entry will show as *“⏳ Evaluating…”* while it runs. Please **do not submit again until you receive your result.**

 [Button: Submit Predictions]


---


### 🏆 About the Challenge


The **ReXGrounding Challenge** is a [MICCAI 2026](https://conferences.miccai.org/2026/) challenge designed to evaluate models on localizing unconstrained radiology findings described in natural language to precise 3D segmentation masks in volumetric chest CT.


Unlike prior challenges that focus on category-level lesion or organ segmentation, this benchmark requires models to interpret diverse clinical language — including anatomical descriptors, spatial relations, and morphological attributes — and ground it accurately in volumetric space. The dataset includes both focal and diffuse abnormalities, spans a wide range of radiological patterns, and reflects real-world reporting variability.


The challenge is built upon **CT-RATE**, a large-scale dataset of non-contrast chest CT scans paired with free-text radiology reports, and is further extended with expert-verified, pixel-level 3D segmentations corresponding to individual report findings. The challenge is hosted on the [ReXrank leaderboard](https://rexrank.ai/ReXGroundingCT/index.html).


---


### 🔬 Task


Participants are evaluated on **free-text finding grounding**: a model receives a CT volume and a natural-language finding from a radiology report and must output a 3D segmentation mask corresponding to that description. The challenge is organized into two tracks: a *Main track* that consumes the free-text finding directly, and a more general *Overall track* (see [Challenge Tracks](https://rexrank.ai/ReXGroundingCT/challenge.html#tracks) below). In both tracks, fixed-category (class-based) segmentation that cannot separate two findings in different locations is out of scope and will be disqualified.


Findings span **14 categories** covering both typically non-focal abnormalities (bronchial wall thickening, bronchiectasis, emphysema, septal thickening, micronodules, and other diffuse abnormalities) and typically focal abnormalities (linear opacities, atelectasis/consolidation, ground-glass opacities, pulmonary nodules/masses, pleural effusion/thickening, honeycombing, pneumothorax, and other focal findings).


---


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


---


### 📦 Dataset


| Split | Cases | Annotations |
| --- | --- | --- |
| **Training** | 2,992 CT scans | Partial-instance (up to 3 instances per finding) |
| **Validation** | 200 CT scans | Exhaustive (all instances segmented by radiologists) |
| **Test** | 300 CT scans | Exhaustive (all instances segmented by radiologists) |


All annotations are pixel-level 3D segmentation masks linked to free-text findings extracted from radiology reports. Validation and test sets are annotated exclusively by board-certified radiologists.


---


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


---


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


---


### 📝 How to Participate


- **Registration is open** — create an account and register your team using the form below to participate.
- Training and validation data are publicly available — you can start developing and evaluating your method now.
- Any publicly available or private training data, including pre-trained models and external datasets, may be used. All external data sources must be described in the method submission.
- All predictions on the test set must be **fully automatic** — no manual intervention, post-hoc editing, or case-specific tuning allowed.
- **Free-text grounding only — categorical segmentation is disqualified.** Your model must take each test finding's **free-text prompt exactly as provided (as-is)** and segment the specific finding that text describes. Methods that ignore, rewrite, or map the prompts to a fixed set of classes — i.e. that perform category-based segmentation rather than grounding the free text — will be **disqualified** if determined to do so.
- You can submit multiple runs; each is evaluated on the held-out test set and shown on the leaderboard. Your best submission before the deadline is your official result.
- See the [submission guidelines](https://rexrank.ai/explore/submission_guideline_ct.html) for format details.


---


### 🏅 Awards & Publication


- The challenge has two tracks (Main and Overall), each with its own winners; every Main-track submission is automatically eligible for the Overall track (see [Challenge Tracks](https://rexrank.ai/ReXGroundingCT/challenge.html#tracks)).
- Top performing teams in each track will receive **certificates** and invited **oral/spotlight presentations** at the challenge session.
- Winning teams will be recognized on the public leaderboard and in the post-challenge publication.
- Members of the winning teams qualify for **co-authorship** on the challenge publication (up to 8 authors per team).
- All teams are free to publish their own results independently with **no embargo period**.


---


### 👥 Organizers


- Mohammed Baharoon — Harvard Medical School, USA
- Pranav Rajpurkar — Harvard Medical School, USA
- Luyang Luo — Harvard Medical School, USA
- Xiaoman Zhang — Harvard Medical School, USA
- Mahmoud Hussain Alabbad — King Fahad Hospital, Saudi Arabia
- Sungeun Kim — Harvard Medical School, USA


---


### 📄 Resources


- [ReXGroundingCT Leaderboard](https://rexrank.ai/ReXGroundingCT/index.html)
- [Dataset on HuggingFace](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT)
- [Evaluation Code](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT/blob/main/rexrank_eval.py)
- [Submission Guidelines](https://rexrank.ai/explore/submission_guideline_ct.html)
- [ReXGroundingCT Paper](https://arxiv.org/abs/2507.22030)


---


### 📧 Contact


For questions about the challenge, please contact [Mohammed Baharoon](mailto:MohammedSalimAB@outlook.com).


---


### 📚 References


**ReXGroundingCT:**


```
@article{baharoon2026rexgroundingct,
  title={ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports},
  author={Baharoon, Mohammed and Luo, Luyang and Moritz, Michael and Kumar, Abhinav and Kim, Sung Eun and Zhang, Xiaoman and Zhu, Miao and Alabbad, Mahmoud Hussain and Alhazmi, Maha Sbayel and Mistry, Neel P and others},
  journal={NEJM AI},
  pages={AIdbp2501220},
  year={2026},
  publisher={Massachusetts Medical Society}
}
```


**CT-RATE:**


```
@article{hamamci2026generalist,
  title={Generalist foundation models from a multimodal dataset for 3D computed tomography},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Wang, Chenyu and Almas, Furkan and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and Dogan, Irem and Durugol, Omer Faruk and Hou, Benjamin and Shit, Suprosanna and others},
  journal={Nature Biomedical Engineering},
  pages={1--19},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```


- [ReXrank](https://rexrank.ai/)

## Archived Live Leaderboard

> **Archive note:** This table was decoded from the official page's intentionally public, read-only Firestore `leaderboard` collection at the snapshot time. Ranking follows the page's JavaScript exactly.

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

### Per-Submission Category Results

<details>
<summary>Grounder_1.1 - Kwang-Hyun Uhm</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.143 | 0.600 |
| **1b** - Bronchiectasis | 5 | 0.112 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.132 | 0.438 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.156 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.218 | 0.500 |
| **1f** - Other | 2 | 0.006 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.281 | 0.803 |
| **2b** - Atelectasis, consolidation | 46 | 0.353 | 0.783 |
| **2c** - Groundglass opacity | 39 | 0.371 | 0.795 |
| **2d** - Pulmonary nodules/masses | 97 | 0.365 | 0.835 |
| **2e** - Pleural effusion or thickening | 13 | 0.360 | 0.615 |
| **2g** - Pneumothorax | 1 | 0.123 | 1.000 |
| **2h** - Other | 1 | 0.673 | 1.000 |

</details>

<details>
<summary>JECT_base - hanbinko</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.072 | 0.400 |
| **1b** - Bronchiectasis | 5 | 0.079 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.102 | 0.312 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.160 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.199 | 0.600 |
| **1f** - Other | 2 | 0.031 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.296 | 0.803 |
| **2b** - Atelectasis, consolidation | 46 | 0.358 | 0.783 |
| **2c** - Groundglass opacity | 39 | 0.365 | 0.744 |
| **2d** - Pulmonary nodules/masses | 97 | 0.324 | 0.804 |
| **2e** - Pleural effusion or thickening | 13 | 0.309 | 0.615 |
| **2g** - Pneumothorax | 1 | 0.444 | 1.000 |
| **2h** - Other | 1 | 0.552 | 1.000 |

</details>

<details>
<summary>Grounder - Kwang-Hyun Uhm</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.096 | 0.400 |
| **1b** - Bronchiectasis | 5 | 0.113 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.153 | 0.438 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.139 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.159 | 0.300 |
| **1f** - Other | 2 | 0.015 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.284 | 0.803 |
| **2b** - Atelectasis, consolidation | 46 | 0.358 | 0.739 |
| **2c** - Groundglass opacity | 39 | 0.342 | 0.744 |
| **2d** - Pulmonary nodules/masses | 97 | 0.319 | 0.763 |
| **2e** - Pleural effusion or thickening | 13 | 0.365 | 0.692 |
| **2g** - Pneumothorax | 1 | 0.638 | 1.000 |
| **2h** - Other | 1 | 0.610 | 1.000 |

</details>

<details>
<summary>MUW-GroundCT-v3 - Ronald Fecso</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.108 | 0.600 |
| **1b** - Bronchiectasis | 5 | 0.054 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.074 | 0.250 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.156 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.222 | 0.600 |
| **1f** - Other | 2 | 0.051 | 0.500 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.273 | 0.803 |
| **2b** - Atelectasis, consolidation | 46 | 0.340 | 0.739 |
| **2c** - Groundglass opacity | 39 | 0.346 | 0.744 |
| **2d** - Pulmonary nodules/masses | 97 | 0.291 | 0.711 |
| **2e** - Pleural effusion or thickening | 13 | 0.355 | 0.769 |
| **2g** - Pneumothorax | 1 | 0.387 | 1.000 |
| **2h** - Other | 1 | 0.551 | 1.000 |

</details>

<details>
<summary>ThoraxTell - Moritz Langenberg</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.162 | 0.800 |
| **1b** - Bronchiectasis | 5 | 0.054 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.055 | 0.125 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.098 | 0.500 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.121 | 0.500 |
| **1f** - Other | 2 | 0.000 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.266 | 0.820 |
| **2b** - Atelectasis, consolidation | 46 | 0.322 | 0.674 |
| **2c** - Groundglass opacity | 39 | 0.325 | 0.769 |
| **2d** - Pulmonary nodules/masses | 97 | 0.303 | 0.753 |
| **2e** - Pleural effusion or thickening | 13 | 0.326 | 0.538 |
| **2g** - Pneumothorax | 1 | 0.579 | 1.000 |
| **2h** - Other | 1 | 0.308 | 1.000 |

</details>

<details>
<summary>MUW-GroundCT-v2 - Ronald Fecso</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.148 | 0.600 |
| **1b** - Bronchiectasis | 5 | 0.097 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.067 | 0.250 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.109 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.187 | 0.500 |
| **1f** - Other | 2 | 0.045 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.252 | 0.754 |
| **2b** - Atelectasis, consolidation | 46 | 0.320 | 0.696 |
| **2c** - Groundglass opacity | 39 | 0.314 | 0.692 |
| **2d** - Pulmonary nodules/masses | 97 | 0.266 | 0.660 |
| **2e** - Pleural effusion or thickening | 13 | 0.304 | 0.615 |
| **2g** - Pneumothorax | 1 | 0.215 | 1.000 |
| **2h** - Other | 1 | 0.544 | 1.000 |

</details>

<details>
<summary>VoXTell-FT (submission 2) - Syed Abdullah Basit</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.155 | 0.800 |
| **1b** - Bronchiectasis | 5 | 0.098 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.014 | 0.062 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.127 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.151 | 0.400 |
| **1f** - Other | 2 | 0.067 | 0.500 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.223 | 0.656 |
| **2b** - Atelectasis, consolidation | 46 | 0.293 | 0.696 |
| **2c** - Groundglass opacity | 39 | 0.327 | 0.769 |
| **2d** - Pulmonary nodules/masses | 97 | 0.280 | 0.711 |
| **2e** - Pleural effusion or thickening | 13 | 0.322 | 0.615 |
| **2g** - Pneumothorax | 1 | 0.243 | 1.000 |
| **2h** - Other | 1 | 0.430 | 1.000 |

</details>

<details>
<summary>VoxTell-FT - Kristhian Aguilar</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.034 | 0.000 |
| **1b** - Bronchiectasis | 5 | 0.088 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.025 | 0.125 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.118 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.134 | 0.300 |
| **1f** - Other | 2 | 0.024 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.180 | 0.541 |
| **2b** - Atelectasis, consolidation | 46 | 0.317 | 0.674 |
| **2c** - Groundglass opacity | 39 | 0.363 | 0.692 |
| **2d** - Pulmonary nodules/masses | 97 | 0.258 | 0.742 |
| **2e** - Pleural effusion or thickening | 13 | 0.348 | 0.615 |
| **2g** - Pneumothorax | 1 | 0.164 | 1.000 |
| **2h** - Other | 1 | 0.579 | 1.000 |

</details>

<details>
<summary>JECT - hanbinko</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.126 | 0.400 |
| **1b** - Bronchiectasis | 5 | 0.119 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.088 | 0.312 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.152 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.134 | 0.300 |
| **1f** - Other | 2 | 0.039 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.242 | 0.770 |
| **2b** - Atelectasis, consolidation | 46 | 0.286 | 0.761 |
| **2c** - Groundglass opacity | 39 | 0.224 | 0.615 |
| **2d** - Pulmonary nodules/masses | 97 | 0.263 | 0.732 |
| **2e** - Pleural effusion or thickening | 13 | 0.157 | 0.462 |
| **2g** - Pneumothorax | 1 | 0.424 | 1.000 |
| **2h** - Other | 1 | 0.350 | 1.000 |

</details>

<details>
<summary>MUW-GroundCT - Ronald Fecso</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.078 | 0.400 |
| **1b** - Bronchiectasis | 5 | 0.061 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.039 | 0.062 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.120 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.170 | 0.500 |
| **1f** - Other | 2 | 0.000 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.229 | 0.672 |
| **2b** - Atelectasis, consolidation | 46 | 0.314 | 0.696 |
| **2c** - Groundglass opacity | 39 | 0.323 | 0.667 |
| **2d** - Pulmonary nodules/masses | 97 | 0.198 | 0.546 |
| **2e** - Pleural effusion or thickening | 13 | 0.238 | 0.615 |
| **2g** - Pneumothorax | 1 | 0.362 | 1.000 |
| **2h** - Other | 1 | 0.383 | 1.000 |

</details>

<details>
<summary>MUW-Text2UNet-v2 - Ronald Fecso</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.027 | 0.000 |
| **1b** - Bronchiectasis | 5 | 0.048 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.017 | 0.000 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.084 | 0.333 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.159 | 0.200 |
| **1f** - Other | 2 | 0.000 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.151 | 0.574 |
| **2b** - Atelectasis, consolidation | 46 | 0.205 | 0.522 |
| **2c** - Groundglass opacity | 39 | 0.258 | 0.692 |
| **2d** - Pulmonary nodules/masses | 97 | 0.175 | 0.608 |
| **2e** - Pleural effusion or thickening | 13 | 0.298 | 0.615 |
| **2g** - Pneumothorax | 1 | 0.173 | 1.000 |
| **2h** - Other | 1 | 0.157 | 1.000 |

</details>

<details>
<summary>FARM-VoxTell-FT-Test - Jie Xu</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.094 | 0.600 |
| **1b** - Bronchiectasis | 5 | 0.083 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.047 | 0.125 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.058 | 0.167 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.106 | 0.200 |
| **1f** - Other | 2 | 0.019 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.048 | 0.115 |
| **2b** - Atelectasis, consolidation | 46 | 0.177 | 0.500 |
| **2c** - Groundglass opacity | 39 | 0.255 | 0.667 |
| **2d** - Pulmonary nodules/masses | 97 | 0.188 | 0.577 |
| **2e** - Pleural effusion or thickening | 13 | 0.118 | 0.385 |
| **2g** - Pneumothorax | 1 | 0.237 | 1.000 |
| **2h** - Other | 1 | 0.376 | 1.000 |

</details>

<details>
<summary>FARM-VoxTell-FT - Jie Xu</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.094 | 0.600 |
| **1b** - Bronchiectasis | 5 | 0.083 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.047 | 0.125 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.058 | 0.167 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.106 | 0.200 |
| **1f** - Other | 2 | 0.019 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.048 | 0.115 |
| **2b** - Atelectasis, consolidation | 46 | 0.177 | 0.500 |
| **2c** - Groundglass opacity | 39 | 0.255 | 0.667 |
| **2d** - Pulmonary nodules/masses | 97 | 0.188 | 0.577 |
| **2e** - Pleural effusion or thickening | 13 | 0.118 | 0.385 |
| **2g** - Pneumothorax | 1 | 0.237 | 1.000 |
| **2h** - Other | 1 | 0.376 | 1.000 |

</details>

<details>
<summary>VoXTell-FT - Abdullah</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.022 | 0.000 |
| **1b** - Bronchiectasis | 5 | 0.011 | 0.000 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.007 | 0.000 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.023 | 0.000 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.063 | 0.200 |
| **1f** - Other | 2 | 0.004 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.059 | 0.213 |
| **2b** - Atelectasis, consolidation | 46 | 0.118 | 0.348 |
| **2c** - Groundglass opacity | 39 | 0.165 | 0.436 |
| **2d** - Pulmonary nodules/masses | 94 | 0.231 | 0.670 |
| **2e** - Pleural effusion or thickening | 13 | 0.129 | 0.462 |
| **2g** - Pneumothorax | 1 | 0.052 | 0.000 |
| **2h** - Other | 1 | 0.026 | 0.000 |
| **unknown** - Uncategorized | 3 | 0.095 | 0.333 |

</details>

<details>
<summary>FARM-VoxTell-FT-HZ-2 - Jie Xu</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.150 | 0.800 |
| **1b** - Bronchiectasis | 5 | 0.077 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.073 | 0.250 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.078 | 0.167 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.118 | 0.300 |
| **1f** - Other | 2 | 0.007 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.071 | 0.246 |
| **2b** - Atelectasis, consolidation | 46 | 0.169 | 0.587 |
| **2c** - Groundglass opacity | 39 | 0.193 | 0.615 |
| **2d** - Pulmonary nodules/masses | 97 | 0.163 | 0.546 |
| **2e** - Pleural effusion or thickening | 13 | 0.003 | 0.000 |
| **2g** - Pneumothorax | 1 | 0.024 | 0.000 |
| **2h** - Other | 1 | 0.559 | 1.000 |

</details>

<details>
<summary>FARM-VoxTell-FT-HZ - Jie Xu</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.150 | 0.800 |
| **1b** - Bronchiectasis | 5 | 0.077 | 0.200 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.073 | 0.250 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.078 | 0.167 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.118 | 0.300 |
| **1f** - Other | 2 | 0.007 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.071 | 0.246 |
| **2b** - Atelectasis, consolidation | 46 | 0.169 | 0.587 |
| **2c** - Groundglass opacity | 39 | 0.193 | 0.615 |
| **2d** - Pulmonary nodules/masses | 97 | 0.163 | 0.546 |
| **2e** - Pleural effusion or thickening | 13 | 0.003 | 0.000 |
| **2g** - Pneumothorax | 1 | 0.024 | 0.000 |
| **2h** - Other | 1 | 0.559 | 1.000 |

</details>

<details>
<summary>MUW-Text2UNet - Ronald Fecso</summary>

| Category | n | Dice | Hit Rate |
| --- | --- | --- | --- |
| **1a** - Bronchial wall thickening | 5 | 0.058 | 0.200 |
| **1b** - Bronchiectasis | 5 | 0.014 | 0.000 |
| **1c** - Emphysema (including Centrilobular, Paraseptal, Bullous) | 16 | 0.009 | 0.000 |
| **1d** - Septal thickening (including Interlobular, Reticulation) | 6 | 0.049 | 0.167 |
| **1e** - Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic) | 10 | 0.114 | 0.300 |
| **1f** - Other | 2 | 0.000 | 0.000 |
| **2a** - Linear (including subsegmental atelectasis, scarring, fibrosis) | 61 | 0.125 | 0.426 |
| **2b** - Atelectasis, consolidation | 46 | 0.178 | 0.565 |
| **2c** - Groundglass opacity | 39 | 0.262 | 0.590 |
| **2d** - Pulmonary nodules/masses | 97 | 0.007 | 0.031 |
| **2e** - Pleural effusion or thickening | 13 | 0.173 | 0.615 |
| **2g** - Pneumothorax | 1 | 0.080 | 0.000 |
| **2h** - Other | 1 | 0.009 | 0.000 |

</details>
