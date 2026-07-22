<!-- Archive metadata in this block is not source content. -->
> **Archive metadata:** Source: [https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT) | Retrieved: `2026-07-22T00:13:43Z` | Raw source: [`snapshots/2026-07-22T001343Z/raw/huggingface.co/datasets/rajpurkarlab/ReXGroundingCT.html`](snapshots/2026-07-22T001343Z/raw/huggingface.co/datasets/rajpurkarlab/ReXGroundingCT.html) | SHA-256: `e4a0ecee73305b3510b9d3ca1f1bfcbe473a04d36800b3abb5aeceb47d7fb47f`
> **Archive note:** Interactive controls do not function in Markdown; the raw HTML remains authoritative for page behavior.

---
[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg) Hugging Face](https://huggingface.co/)
[Input: Search models, datasets, users...]


- [Models](https://huggingface.co/models)
- [Datasets](https://huggingface.co/datasets)
- [Spaces](https://huggingface.co/spaces)
- [Buckets new](https://huggingface.co/storage)
- [Docs](https://huggingface.co/docs)
- [Enterprise](https://huggingface.co/enterprise)
- [Pricing](https://huggingface.co/pricing)
- - Website - [Tasks](https://huggingface.co/tasks) - [HuggingChat](https://huggingface.co/chat) - [Collections](https://huggingface.co/collections) - [Languages](https://huggingface.co/languages) - [Organizations](https://huggingface.co/organizations) - Community - [Blog](https://huggingface.co/blog) - [Posts](https://huggingface.co/posts) - [Daily Papers](https://huggingface.co/papers) - [Hardware](https://huggingface.co/hardware) - [Learn](https://huggingface.co/learn) - [Discord](https://huggingface.co/join/discord) - [Forum](https://discuss.huggingface.co/) - [GitHub](https://github.com/huggingface) - Solutions - [Team & Enterprise](https://huggingface.co/enterprise) - [Hugging Face PRO](https://huggingface.co/pro) - [Enterprise Support](https://huggingface.co/support) - [Inference Providers](https://huggingface.co/inference/models) - [Inference Endpoints](https://huggingface.co/inference-endpoints) - [Storage Buckets](https://huggingface.co/storage)
- ---
- [Log In](https://huggingface.co/login)
- [Sign Up](https://huggingface.co/join)


# [Datasets:](https://huggingface.co/datasets)

---


 [![](https://cdn-avatars.huggingface.co/v1/production/uploads/1657667248177-62cdfe70342b1d5dab92e58f.png)](https://huggingface.co/rajpurkarlab)
 [rajpurkarlab](https://huggingface.co/rajpurkarlab)
/


[ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT)

[Button: like] [Button: 45]


[Button: Follow
 ![](https://cdn-avatars.huggingface.co/v1/production/uploads/1657667248177-62cdfe70342b1d5dab92e58f.png) Rajpurkar Lab / Harvard Medical AI] [Button: 138]


ArXiv:
[Button: arxiv: 2507.22030]

[Button: arxiv: 2403.17834]


License:
[Button: cc-by-nc-sa-4.0]


 [Dataset card](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT)[Files Files and versions
 xet](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT/tree/main)[Community
4](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT/discussions)


## You need to agree to share your contact information to access this dataset


This repository is publicly accessible, but you have to accept the conditions to access its files and content.


[Log in](https://huggingface.co/login?next=/datasets/rajpurkarlab/ReXGroundingCT) or [Sign Up](https://huggingface.co/join?next=/datasets/rajpurkarlab/ReXGroundingCT) to review the conditions and access this dataset content.


- [Updates](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#updates)

- [Dataset Structure](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#dataset-structure)
  - [Field Descriptions](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#field-descriptions)
- [Segmentation Masks](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#segmentation-masks)

- [Citations](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#citations)


# [#rexgroundingct](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#rexgroundingct) ReXGroundingCT


[ReXGroundingCT](https://www.arxiv.org/abs/2507.22030) is a dataset designed to link free-text radiology findings with pixel-level segmentations in 3D chest CT scans. Each sample consists of a volumetric CT scan, associated segmentation masks for one or more findings, and detailed textual descriptions.
The dataset has segmentations for 8,028 findings across 14 different categories in 3,142 CT scans. There are 2,992 scans allocated for training, 50 for public validation, and 100 held privately to be hosted for a public leaderboard on [ReXrank](https://rexrank.ai/ReXGroundingCT/index.html).


---


## [#updates](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#updates) Updates


- **2026-06-14** — ReXGroundingCT MICCAI 2026 challenge launches.
- **2026-06-13** — Added `MICCAI_challenge_dataset.json` for the ReXGroundingCT MICCAI 2026 challenge: a `val` split (200 scans, segmentation masks released) and a held-out `test` split (300 scans — names + findings only, masks withheld). The base `dataset.json` (train/val/test) is unchanged.


---


## [#dataset-structure](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#dataset-structure) Dataset Structure


Each item in the dataset is represented as a JSON entry with the following structure:


```json
{
    "name": "train_1741_b_2.nii.gz",
    "findings": {
        "0": "Irregularly circumscribed nodular consolidation area adjacent to the diaphragm in the basal segment of the lower lobe of the right lung"
    },
    "entity_counts": {
        "0": 1
    },
    "shape": [512, 512, 238],
    "pixels": {
        "0": 15571
    },
    "categories": {
        "0": "2b"
    },
    "protocol": "protocol1"
}
```


### [#field-descriptions](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#field-descriptions) Field Descriptions


- **`name`**: The filename of the CT volume (e.g., `train_1741_b_2.nii.gz`).
- **`findings`**: A dictionary mapping finding indices in the segmentation mask to free-text radiology findings.
- **`entity_counts`**: Number of segmented entities for each finding.
- **`shape`**: The shape of the CT volume in `[H, W, D]` (height, width, depth).
- **`pixels`**: Number of non-zero pixels for each finding's segmentation mask.
- **`categories`**: The category code assigned to each finding (e.g., `"2b"`).
- **`protocol`**: Annotation protocol used for the case (e.g., `protocol1`, `protocol2`).


---


## [#segmentation-masks](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#segmentation-masks) Segmentation Masks


Each CT scan has an associated segmentation mask volume with the shape:


```
(F, H, W, D)
```


- **F**: Number of findings in the scan
- **H, W, D**: Spatial dimensions of the CT volume (matching the `shape` field)


For example, a scan with 3 findings and shape `[512, 512, 238]` will have a segmentation mask of shape `[3, 512, 512, 238]`, where each slice along the F dimension corresponds to one finding. For a given finding, each unique pixel value corresponds to a different entity for that finding.


We also provide `reports_dataset.json`, which contains the full extracted reports, including negative findings and findings outside the lung and pleura.


> **Note**: the `MLHC_dataset_version.json` file is a smaller sample of the dataset used in the paper [State-of-the-Art Text-Prompted Medical Segmentation Models Struggle to Ground Chest CT Findings](https://static1.squarespace.com/static/59d5ac1780bd5ef9c396eda6/t/689b89d2eaccd04601af00b6/1755023826975/61_camera_ready+-+Mohammed+Baharoon.pdf) published in MLHC 2025.


## [#citations](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT#citations) Citations


If you find this dataset useful, please cite the following papers:


```
@article{baharoon2025rexgroundingct,
  title={ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports},
  author={Baharoon, Mohammed and Luo, Luyang and Moritz, Michael and Kumar, Abhinav and Kim, Sung Eun and Zhang, Xiaoman and Zhu, Miao and Alabbad, Mahmoud Hussain and Alhazmi, Maha Sbayel and Mistry, Neel P and others},
  journal={arXiv preprint arXiv:2507.22030},
  year={2025}
}

@article{hamamci2024developing,
  title={Developing generalist foundation models from a multimodal dataset for 3d computed tomography},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Wang, Chenyu and Almas, Furkan and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and Doga, Irem and Durugol, Omer Faruk and Dai, Weicheng and Xu, Murong and others},
  journal={arXiv preprint arXiv:2403.17834},
  year={2024}
}
```


[Button: Copy to bucket new]


**Downloads last month**


2,830


 [Total file size:

3.57 GB](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT)


## Models trained or fine-tuned on rajpurkarlab/ReXGroundingCT


[![](https://cdn-avatars.huggingface.co/v1/production/uploads/5f19d8d8925b9863e28ad654/VjNmS90XpxqznNWOHzLl_.png)


#### Kentucky-Open-Science/DALE-CT-1S


 Feature Extraction • 0.3B • Updated Jun 5 • 2](https://huggingface.co/Kentucky-Open-Science/DALE-CT-1S)


[![](https://cdn-avatars.huggingface.co/v1/production/uploads/5f19d8d8925b9863e28ad654/VjNmS90XpxqznNWOHzLl_.png)


#### Kentucky-Open-Science/Ker-VLJEPA-3B


 Text Generation • Updated Mar 25](https://huggingface.co/Kentucky-Open-Science/Ker-VLJEPA-3B)


[![](https://cdn-avatars.huggingface.co/v1/production/uploads/5f19d8d8925b9863e28ad654/VjNmS90XpxqznNWOHzLl_.png)


#### Kentucky-Open-Science/DALE-CT-2S


 Feature Extraction • 0.3B • Updated Jun 5](https://huggingface.co/Kentucky-Open-Science/DALE-CT-2S)


## Papers for rajpurkarlab/ReXGroundingCT


[#### ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports


 Paper • 2507.22030 • Published Jul 29, 2025 • 4](https://huggingface.co/papers/2507.22030)

[#### A foundation model utilizing chest CT volumes and radiology reports for supervised-level zero-shot detection of abnormalities


 Paper • 2403.17834 • Published Mar 26, 2024 • 5](https://huggingface.co/papers/2403.17834)


[Button: System theme]


Company
 [TOS](https://huggingface.co/terms-of-service) [Privacy](https://huggingface.co/privacy) [About](https://huggingface.co/huggingface) [Careers](https://apply.workable.com/huggingface/) [/](https://huggingface.co/)
Website
 [Models](https://huggingface.co/models) [Datasets](https://huggingface.co/datasets) [Spaces](https://huggingface.co/spaces) [Pricing](https://huggingface.co/pricing) [Docs](https://huggingface.co/docs)
