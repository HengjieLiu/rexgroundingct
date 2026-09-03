# Exp018: Official Category-2d Nodule Method Audit

- Status: `audit_complete` (source snapshot `2026-08-21_train_val_category2d`)
- Mode: read-only audit; no checkpoint production, inference, training, or GPU launch.
- Primary population: single-model checkpoints only evaluated on fixed val200 at mask threshold `0.5`.

## Scope and metric contract

Official category `2d` is **Pulmonary nodules/masses**. It contains exactly `132` findings in `119` of the fixed `200` val200 cases. `1e` micronodules/tree-in-bud and Exp004's legacy regex-based `nodule` stratum are not this primary population.

Dice is the unweighted mean of raw `global_dice` across the `132` official records. Hits recompute `global_dice >= 0.1`. The deterministic primary order is dice descending, hits descending, candidate id ascending.

## Train–validation category-2d distribution

The archived official challenge policy is correct at the **instance** level: training is `Partial-instance (up to 3 instances per finding)`, while validation and test are `Exhaustive (all instances segmented by radiologists)`. The hash-pinned rendered snapshot is `challenge_info/rexgroundingct_challenge.md`; its raw archive is `challenge_info/snapshots/2026-07-22T001343Z/raw/rexrank.ai/ReXGroundingCT/challenge.html`.

That policy does not imply fewer free-text finding descriptions per CT case. The released source has CT-case filenames rather than a reliable patient identifier, so every comparison below is per **CT case**, not per patient.

**Text-only location limitation:** Text only: locations are derived only from findings free text. It does not use actual lung or lobe masks, segmentation masks, CT images, coordinates, predictions, or image-derived localization.

**Entity-count limitation:** Released entity_counts metadata only; no segmentation masks are loaded. It reports segmented entities per finding, not an independently verified lesion census. The field definition is hash-pinned from `challenge_info/snapshots/2026-07-22T001343Z/raw/huggingface.co/datasets/rajpurkarlab/ReXGroundingCT.html`.

### Side-by-side population and annotation density

| Metric | Train | Validation | Validation − train |
| --- | ---: | ---: | ---: |
| CT cases | 2,992 | 200 | -2,792 |
| 2d-positive CT cases | 1,305 | 119 | -1,186 |
| 2d-positive CT-case share | 43.6% | 59.5% | +15.9 pp |
| 2d finding descriptions | 1,743 | 132 | -1,611 |
| 2d findings per CT case | 0.583 | 0.660 | +0.077 |
| 2d findings per 2d-positive CT case | 1.336 | 1.109 | -0.226 |
| Released annotated 2d entities | 3,185 | 466 | -2,719 |
| Released entities per 2d finding | 1.827 | 3.530 | +1.703 |
| Released entities per CT case | 1.065 | 2.330 | +1.265 |
| Released entities per 2d-positive CT case | 2.441 | 3.916 | +1.475 |

### Released entity-count distribution

Each row is a released `entity_counts` value for one official 2d finding. This is metadata, not a new segmentation-mask audit.

| Segmented entities per finding | Train | Validation | Validation − train |
| --- | ---: | ---: | ---: |
| 1 | 887/1743 (50.9%) | 51/132 (38.6%) | -12.3 pp |
| 2 | 316/1743 (18.1%) | 21/132 (15.9%) | -2.2 pp |
| 3 | 504/1743 (28.9%) | 14/132 (10.6%) | -18.3 pp |
| 4+ | 36/1743 (2.1%) | 46/132 (34.8%) | +32.8 pp |

The expected density gap is visible: validation has `3.530` released segmented entities per 2d finding versus `1.827` in training. Do not treat an entity count of `3` as proof of omitted lesions: it is a censoring-risk indicator. The released train metadata also contains `36` 2d findings above three (maximum `5`), so the documented up-to-three policy is not a hard metadata invariant.

#### Observed training values above three

| Released train entity count | Official 2d findings |
| ---: | ---: |
| 4 | 26 |
| 5 | 10 |

The machine-readable comparison block retains all sorted `(CT case, finding index, entity count)` records for these observed training values; they are reported as metadata exceptions, not capped or discarded.

Test is cited for its documented exhaustive policy but excluded from these entity tables because this metadata snapshot has no test entity_counts.

### Text-only location comparison

The primary table assigns exactly one normalized text label per finding using fixed precedence. Rows are sorted by training share descending, then label; a finding description can cover multiple nodules.

| Rank | Normalized text location | Train (n / 1743) | Validation (n / 132) | Validation − train |
| ---: | --- | ---: | ---: | ---: |
| 1 | Bilateral / both lungs | 669 (38.4%) | 61 (46.2%) | +7.8 pp |
| 2 | Right lower lobe | 206 (11.8%) | 14 (10.6%) | -1.2 pp |
| 3 | No location stated | 177 (10.2%) | 3 (2.3%) | -7.9 pp |
| 4 | Left lower lobe | 176 (10.1%) | 16 (12.1%) | +2.0 pp |
| 5 | Right upper lobe | 173 (9.9%) | 11 (8.3%) | -1.6 pp |
| 6 | Right lung, lobe not stated | 153 (8.8%) | 5 (3.8%) | -5.0 pp |
| 7 | Left upper lobe | 107 (6.1%) | 8 (6.1%) | -0.1 pp |
| 8 | Right middle lobe | 72 (4.1%) | 9 (6.8%) | +2.7 pp |
| 9 | Laterobasal, side not stated | 9 (0.5%) | 1 (0.8%) | +0.2 pp |
| 10 | Right apical (bleb wording) | 1 (0.1%) | 1 (0.8%) | +0.7 pp |
| 11 | Left lung fissure | 0 (0.0%) | 1 (0.8%) | +0.8 pp |
| 12 | Right minor fissure / perifissural | 0 (0.0%) | 2 (1.5%) | +1.5 pp |

### Additional matching free-text location cues

This table preserves every matching cue in a finding description, including a named lobe embedded in a bilateral description. It is multi-label and non-exclusive; counts do not sum to the split totals.

| Rank | Textual location cue | Dimension | Train (n / 1743) | Validation (n / 132) | Validation − train |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Bilateral / both lungs | laterality/distribution | 669 (38.4%) | 61 (46.2%) | +7.8 pp |
| 2 | Right-lung wording | laterality | 276 (15.8%) | 10 (7.6%) | -8.3 pp |
| 3 | Right lower lobe | lobe | 217 (12.4%) | 16 (12.1%) | -0.3 pp |
| 4 | Basal / laterobasal / posterobasal / anterobasal / mediobasal | within-lung qualifier | 199 (11.4%) | 20 (15.2%) | +3.7 pp |
| 5 | Left lower lobe | lobe | 184 (10.6%) | 19 (14.4%) | +3.8 pp |
| 6 | Right upper lobe | lobe | 183 (10.5%) | 12 (9.1%) | -1.4 pp |
| 7 | Subpleural | surface/relationship | 172 (9.9%) | 19 (14.4%) | +4.5 pp |
| 8 | Anterior / anterobasal | within-lung qualifier | 154 (8.8%) | 9 (6.8%) | -2.0 pp |
| 9 | Lateral / laterobasal | within-lung qualifier | 150 (8.6%) | 16 (12.1%) | +3.5 pp |
| 10 | Posterior / posterobasal | within-lung qualifier | 148 (8.5%) | 16 (12.1%) | +3.6 pp |
| 11 | Left-lung wording | laterality | 140 (8.0%) | 4 (3.0%) | -5.0 pp |
| 12 | Superior | within-lung qualifier | 137 (7.9%) | 10 (7.6%) | -0.3 pp |
| 13 | Left upper lobe | lobe | 114 (6.5%) | 11 (8.3%) | +1.8 pp |
| 14 | Right middle lobe | lobe | 83 (4.8%) | 10 (7.6%) | +2.8 pp |
| 15 | Fissural / perifissural | surface/relationship | 70 (4.0%) | 6 (4.5%) | +0.5 pp |
| 16 | Medial / mediobasal | within-lung qualifier | 42 (2.4%) | 3 (2.3%) | -0.1 pp |
| 17 | Peripheral | surface/relationship | 38 (2.2%) | 6 (4.5%) | +2.4 pp |
| 18 | Apical / apex | within-lung qualifier | 34 (2.0%) | 2 (1.5%) | -0.4 pp |
| 19 | Lingular | within-lung qualifier | 32 (1.8%) | 3 (2.3%) | +0.4 pp |
| 20 | Pleural-based | surface/relationship | 18 (1.0%) | 3 (2.3%) | +1.2 pp |
| 21 | Intrapulmonary | relationship | 7 (0.4%) | 2 (1.5%) | +1.1 pp |
| 22 | Diaphragmatic / juxtadiaphragmatic | within-lung qualifier | 1 (0.1%) | 3 (2.3%) | +2.2 pp |

Category membership, not a `nodul*` text regex, defines this population. The validation-only text audit is retained unchanged in `nodule_method_audit.json`; the comparison block records the same fixed rules and split-level counts.

## Current Pareto references

| Role | Candidate | Checkpoint | 2d Dice | Hits | Hit rate |
| --- | --- | --- | ---: | ---: | ---: |
| Dice-first reference | `exp007_cont_e050_abs_e150` | relative e50 / absolute e150 | 0.394936 | 115/132 | 0.871 |
| Hit-rate guardrail | `exp017_phase1_e075` | e75 | 0.392786 | 118/132 | 0.894 |

The two rows express the current trade-off: Exp007's phase-2 continuation reaches the highest Dice, while Exp017 phase 1 epoch 75 preserves three more 2d hits.

## Primary fixed-val200 single-checkpoint leaderboard

Every row below is a single model evaluated at the fixed `0.5` mask threshold. The frozen runtime scan found `58` evaluator files; `58` meet the primary contract and `0` are unranked. The legacy 20-model catalog supplies history, but its raw evaluator JSONs are hash-checked and recomposed here; newer runtime results are included as additional rows.

| Rank | Candidate | Method family | Checkpoint | Scope | 2d Dice | Hits | Hit rate |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | `exp007_cont_e050_abs_e150` | Exp007 phase-2 continuation | relative e50 / absolute e150 | `broad_continuation` | 0.394936 | 115/132 | 0.871 |
| 2 | `exp017_phase1_e075` | Exp017 0.7 mm phase 1 | e75 | `broad_preprocessing_variant` | 0.392786 | 118/132 | 0.894 |
| 3 | `exp017_cont_e050_abs_e150` | Exp017 0.7 mm phase-2 continuation | relative e50 / absolute e150 | `broad_preprocessing_variant` | 0.381093 | 115/132 | 0.871 |
| 4 | `exp007_cont_e075_abs_e175` | Exp007 phase-2 continuation | relative e75 / absolute e175 | `broad_continuation` | 0.379058 | 109/132 | 0.826 |
| 5 | `exp009_baseline_e100` | Exp009 baseline_cont100 | e100 | `broad_single_model` | 0.378417 | 113/132 | 0.856 |
| 6 | `exp008_shared_e100` | Exp008 v1_sharedfusion_softguide | e100 | `broad_single_model` | 0.376547 | 113/132 | 0.856 |
| 7 | `exp007_cont_e100_abs_e200` | Exp007 phase-2 continuation | relative e100 / absolute e200 | `broad_continuation` | 0.374798 | 110/132 | 0.833 |
| 8 | `exp007_ddp_e050` | Exp007 ddp_bs4 | e50 | `broad_single_model` | 0.374119 | 109/132 | 0.826 |
| 9 | `exp008_joint_e100` | Exp008 v3_dualfusion_softguide_joint | e100 | `broad_single_model` | 0.373578 | 114/132 | 0.864 |
| 10 | `exp008_dual_e100` | Exp008 v1_dualfusion_softguide | e100 | `broad_single_model` | 0.371461 | 111/132 | 0.841 |
| 11 | `exp017_phase1_e050` | Exp017 0.7 mm phase 1 | e50 | `broad_preprocessing_variant` | 0.371460 | 115/132 | 0.871 |
| 12 | `exp008_precision_e100` | Exp008 v2_dualfusion_precision | e100 | `broad_single_model` | 0.371033 | 112/132 | 0.848 |
| 13 | `exp007_ddp_e075` | Exp007 ddp_bs4 | e75 | `broad_single_model` | 0.369589 | 111/132 | 0.841 |
| 14 | `exp007_ddp_e100` | Exp007 ddp_bs4 | e100 | `broad_single_model` | 0.368074 | 109/132 | 0.826 |
| 15 | `exp012_r01_category_2a_replay50_e100` | Exp012 R01 category specialist | e100 | `non_2d_specialist` | 0.367895 | 114/132 | 0.864 |
| 16 | `exp009_s3v1_e100` | Exp009 s3v1_fixedrho_suppress_half_quarter | e100 | `broad_single_model` | 0.367206 | 110/132 | 0.833 |
| 17 | `exp014_ddp_bs16_e050` | Exp014 native DDP batch16 | e50 | `broad_batch_size_ablation` | 0.366663 | 111/132 | 0.841 |
| 18 | `exp017_phase1_e100` | Exp017 0.7 mm phase 1 | e100 | `broad_preprocessing_variant` | 0.365712 | 117/132 | 0.886 |
| 19 | `exp017_cont_e025_abs_e125` | Exp017 0.7 mm phase-2 continuation | relative e25 / absolute e125 | `broad_preprocessing_variant` | 0.365361 | 116/132 | 0.879 |
| 20 | `exp014_ddp_bs16_e075` | Exp014 native DDP batch16 | e75 | `broad_batch_size_ablation` | 0.365012 | 109/132 | 0.826 |
| 21 | `exp012_r02_category_2d_replay50_e100` | Exp012 R02 direct 2d replay50 | e100 | `direct_2d_finetune` | 0.363793 | 110/132 | 0.833 |
| 22 | `exp011_clip_zscore_e100` | Exp011 v123_e5d4_clip1024_zscore | e100 | `broad_single_model` | 0.363435 | 113/132 | 0.856 |
| 23 | `exp011_native_zscore_e100` | Exp011 v123_e5d4_zscore | e100 | `broad_single_model` | 0.363338 | 112/132 | 0.848 |
| 24 | `exp012_r01_category_1alldiffuse_replay50_e100` | Exp012 R01 category specialist | e100 | `non_2d_specialist` | 0.362938 | 110/132 | 0.833 |
| 25 | `exp009_s3v2_e100` | Exp009 s3v2_balanced_feature_half_quarter | e100 | `broad_single_model` | 0.362037 | 113/132 | 0.856 |
| 26 | `exp006_e5d4_e100` | Exp006 v123_cached_e5_d4 | e100 | `broad_single_model` | 0.361274 | 114/132 | 0.864 |
| 27 | `exp006_e6d4_e100` | Exp006 v123_cached_e6_d4 | e100 | `broad_single_model` | 0.361056 | 110/132 | 0.833 |
| 28 | `exp017_phase1_e025` | Exp017 0.7 mm phase 1 | e25 | `broad_preprocessing_variant` | 0.361005 | 113/132 | 0.856 |
| 29 | `exp008_shared_e080` | Exp008 v1_sharedfusion_softguide | e80 | `broad_single_model` | 0.360891 | 110/132 | 0.833 |
| 30 | `exp008_joint_e080` | Exp008 v3_dualfusion_softguide_joint | e80 | `broad_single_model` | 0.360166 | 113/132 | 0.856 |
| 31 | `exp012_r02_category_1alldiffuse_replay25_e100` | Exp012 R02 category specialist | e100 | `non_2d_specialist` | 0.360115 | 111/132 | 0.841 |
| 32 | `exp011_linear_hu_e100` | Exp011 v123_e5d4_clip1024_linear | e100 | `broad_single_model` | 0.359148 | 113/132 | 0.856 |
| 33 | `exp017_cont_e075_abs_e175` | Exp017 0.7 mm phase-2 continuation | relative e75 / absolute e175 | `broad_preprocessing_variant` | 0.358592 | 112/132 | 0.848 |
| 34 | `exp009_s3v3_e100` | Exp009 s3v3_logit_residual_half_quarter | e100 | `broad_single_model` | 0.357764 | 113/132 | 0.856 |
| 35 | `exp008_precision_e080` | Exp008 v2_dualfusion_precision | e80 | `broad_single_model` | 0.357404 | 108/132 | 0.818 |
| 36 | `exp012_r01_category_2c_replay50_e100` | Exp012 R01 category specialist | e100 | `non_2d_specialist` | 0.356942 | 108/132 | 0.818 |
| 37 | `exp014_ddp_bs16_e025` | Exp014 native DDP batch16 | e25 | `broad_batch_size_ablation` | 0.355198 | 111/132 | 0.841 |
| 38 | `exp012_r01_category_2b_replay50_e100` | Exp012 R01 category specialist | e100 | `non_2d_specialist` | 0.351100 | 107/132 | 0.811 |
| 39 | `exp008_dual_e080` | Exp008 v1_dualfusion_softguide | e80 | `broad_single_model` | 0.350696 | 111/132 | 0.841 |
| 40 | `exp013_category_1all_diffuse_target100_e100` | Exp013 public category-only specialist | e100 | `non_2d_specialist` | 0.348174 | 109/132 | 0.826 |
| 41 | `exp007_ddp_e025` | Exp007 native DDP batch4 | e25 | `broad_continuation` | 0.347757 | 112/132 | 0.848 |
| 42 | `exp011_e4d4_clip_linear_e100` | Exp011 e4d4 CT normalization | e100 | `broad_normalization_variant` | 0.347408 | 112/132 | 0.848 |
| 43 | `exp012_r02_category_1alldiffuse_replay10_e100` | Exp012 R02 category specialist | e100 | `non_2d_specialist` | 0.345439 | 106/132 | 0.803 |
| 44 | `exp013_category_2all_focal_target100_e100` | Exp013 public category-only specialist | e100 | `broader_2d_inclusive_finetune` | 0.344977 | 111/132 | 0.841 |
| 45 | `exp011_e4d4_clip_zscore_e100` | Exp011 e4d4 CT normalization | e100 | `broad_normalization_variant` | 0.344919 | 110/132 | 0.833 |
| 46 | `exp011_e4d4_native_zscore_e100` | Exp011 e4d4 CT normalization | e100 | `broad_normalization_variant` | 0.340492 | 113/132 | 0.856 |
| 47 | `exp013_category_2d_target100_e100` | Exp013 public direct 2d target100 | e100 | `direct_2d_finetune` | 0.339718 | 108/132 | 0.818 |
| 48 | `exp006_e7d4_e100` | Exp006 cached-native LR ablation | e100 | `broad_lr_ablation` | 0.338986 | 106/132 | 0.803 |
| 49 | `exp012_r02_category_1alldiffuse_replay00_e100` | Exp012 R02 category specialist | e100 | `non_2d_specialist` | 0.338838 | 108/132 | 0.818 |
| 50 | `exp007_cont_e025_abs_e125` | Exp007 phase-2 continuation | relative e25 / absolute e125 | `broad_continuation` | 0.337559 | 109/132 | 0.826 |
| 51 | `exp004_native192_cont100_e100` | Exp004 native192 continuation | e100 | `broad_sampling_variant` | 0.310126 | 102/132 | 0.773 |
| 52 | `exp003_v123_opt_poscrop_emptyloss_e100` | Exp003 rescue ablation | e100 | `broad_single_model` | 0.292579 | 100/132 | 0.758 |
| 53 | `exp006_e7d6_e100` | Exp006 cached-native LR ablation | e100 | `broad_lr_ablation` | 0.292365 | 99/132 | 0.750 |
| 54 | `exp013_category_2bc_target100_e100` | Exp013 public category-only specialist | e100 | `non_2d_specialist` | 0.291144 | 97/132 | 0.735 |
| 55 | `exp003_v1234_opt_poscrop_emptyloss_ds_e100` | Exp003 rescue ablation | e100 | `broad_single_model` | 0.289643 | 99/132 | 0.750 |
| 56 | `exp003_v12_opt_poscrop_e100` | Exp003 rescue ablation | e100 | `broad_single_model` | 0.286618 | 96/132 | 0.727 |
| 57 | `exp003_v1_opt_e100` | Exp003 rescue ablation | e100 | `broad_single_model` | 0.284191 | 91/132 | 0.689 |
| 58 | `exp004_iso2mm_global192_cont100_e100` | Exp004 2 mm global-context continuation | e100 | `broad_sampling_variant` | 0.066344 | 26/132 | 0.197 |

## Direct official-2d fine-tune inventory

These are intentionally separate from the primary ranking's broad-model comparison. Target-census rows cover the official 132 `2d` records but only the 119 affected cases (or a union containing them), so they are partial evidence rather than full-val200 model selection evidence.

| Method | Training contract | Target-census-only evidence | Full fixed-val200 checkpoint | Readout |
| --- | --- | --- | --- | --- |
| `category_2d_replay50` | Exp009-source continuation; 50 targeted 2d and 50 natural-replay events per epoch. | e0 source checkpoint (2d slice of union evaluator): 0.378452; 113/132 | e100: 0.363793; 110/132 | Target-Dice selection retained the e0 Exp009 source; the completed e100 full-val200 result regressed. |
| `category_2d_target100` | Public VoxTell v1.1 start; 100% targeted official-2d events and no replay. | e0 target-census checkpoint: 0.226821; 70/132; e20 target-census checkpoint: 0.313835; 103/132; e40 target-census checkpoint: 0.310738; 107/132; e60 target-census checkpoint: 0.301824; 105/132; e80 target-census checkpoint: 0.352313; 105/132 | e100: 0.339718; 108/132 | All e0/e20/e40/e60/e80 target-census checkpoints are partial evidence; e80 is the target-Dice selection, while e100 is reported separately on full val200. |

## Broader nodule-inclusive fine-tunes

These methods include `2d` in a broader target set; they are not direct 2d-only methods.

| Method | Target categories | Fixed val200 2d | Note |
| --- | --- | --- | --- |
| `category_2all_focal_target100` | 2a, 2b, 2c, 2d, 2e, 2f, 2g, 2h | 0.344977; 111/132 | Public-start 100%-targeted focal-category specialist; direct-2d claims do not apply. |

## Exclusions and caveats

- **Ensembles:** Exp003's four-model probability ensemble (2d Dice 0.303438; 101/132 hits at 0.50) and SideExp002 ensemble results are excluded because the primary table ranks single checkpoints only.
- **Validation-selected threshold oracles:** SideExp002's all-20 ensemble global threshold 0.55 (2d Dice 0.379972; 112/132) and category-selected 2d threshold 0.90 oracle (0.388159; 109/132) are descriptive validation selections, not fixed-0.5 single-model evidence.
- **Legacy Exp003 threshold provenance:** Project policy fixes single-model thresholds at 0.5 unless explicitly performing an ensemble sweep, so the four Exp003 single-model rows are ranked. Their raw evaluator files record the global hit cutoff (0.1), not a contemporaneous segmentation-mask cutoff; the project-fixed 0.5 policy is hash-pinned in the Exp018 audit contract and the legacy code chain remains forensic rather than direct command capture.
- **Incompatible nodule definitions:** Category 1e micronodules/tree-in-bud and Exp004's legacy regex-based nodule stratum (N=150) are not official category 2d (N=132).
- **Incomplete evaluations:** Target-only Exp013 checkpoint censuses are kept as partial evidence, not primary rows; Exp014 epoch 100 and Exp017 phase-2 epoch 100 have no fixed-val200 evaluator in this frozen snapshot.
- **Validation reuse:** All rankings are fixed-val200 descriptive comparisons. They do not establish challenge-test generalization and should not be treated as a final test-set model selection.

## Threshold provenance

Each primary row carries one of the hash-validated 0.5 sources below in the machine-readable snapshot. Raw evaluator JSON records the global hit threshold, so this additional provenance is required for the segmentation-mask threshold claim.

| Evidence ID | Source | JSON field | Value | SHA-256 |
| --- | --- | --- | ---: | --- |
| `exp004_primary_threshold` | `configs/experiments/004_voxtell_v123_native_vs_2mm_global_context_ft.json` | `validation.primary_threshold` | 0.5 | `2978c06c77c50329453af8d29c0510643ed121a1cf3183464ccb0805e4ecade3` |
| `exp006_primary_threshold` | `configs/experiments/006_voxtell_cached_native_v123_lr_ablation.json` | `validation.primary_threshold` | 0.5 | `b8e587a36a2a51e4ee9696a1690f557883f0d1deeae6c77602b225ebf154a210` |
| `exp007_primary_threshold` | `configs/experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched.json` | `validation.primary_threshold` | 0.5 | `ff7d1d22c5e1887167156df886cc1e792a3318cbb39fc5147e4a45e4baf1b349` |
| `exp011_primary_threshold` | `configs/experiments/011_voxtell_v123_e4d4_ct_normalization_ablation.json` | `validation.primary_threshold` | 0.5 | `f98028e8810c44fb30a8883569f79ac84cabd2ff57100743474127ee00bb1235` |
| `exp012_final_threshold` | `configs/experiments/012_voxtell_category_specialists_replay50_cont100.json` | `validation.final_threshold` | 0.5 | `8700da1116914e3d71933abafe859138f31d5550eca72519e53c9e51e4998782` |
| `exp013_final_threshold` | `configs/experiments/013_voxtell_public_category_only_specialists.json` | `validation.final_threshold` | 0.5 | `cd424e71122e3431f7a460421b4a772b7e9ff7381b89d519198f23ca8b7c17f7` |
| `exp014_primary_threshold` | `configs/experiments/014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched.json` | `validation.primary_threshold` | 0.5 | `18ea20f511a8e4ef956efe98d232a10c02b191b2a48fb5f5455c40ca4d71dafe` |
| `exp017_primary_threshold` | `configs/experiments/017_voxtell_iso07_hu_ddp_bs4_update_matched.json` | `validation.primary_threshold` | 0.5 | `32e2b702490255245c249e9b759b8a57aa442beccf3098908459edb2643f046f` |
| `exp018_project_fixed_threshold_policy` | `configs/experiments/018_voxtell_category2d_nodule_audit.json` | `official_nodule_contract.primary_mask_threshold` | 0.5 | `08eea7163a955011b46d32db2113274d6ed20713da889182c45a3f88dd817402` |
| `sideexp002_fixed_mask_threshold` | `side_experiments/sideexp002_multimodel_ensemble_selection/outputs/baseline_per_category_performance.json` | `definitions.mask_threshold` | 0.5 | `cf9dadd4ac3b3ad869ab889596309ee61ff19afa8dab281249a4012696f3cfec` |

## Evidence-based conclusion

No direct official-category-2d fine-tune has exceeded the broad Exp007 phase-2 continuation on fixed val200 Dice. The broad reference is `exp007_cont_e050_abs_e150` at `0.394936; 115/132`; the best completed direct 2d fine-tune is `exp012_r02_category_2d_replay50_e100` at `0.363793; 110/132`. A training matrix and GPU launch are intentionally outside Exp018.

## Source provenance

All evaluator source hashes below were validated before the metrics were recomposed. `primary_leaderboard` rows are raw fixed-val200 evaluator sources; unranked and target-census evidence is listed separately.

| Role | Row | Evaluator path | SHA-256 |
| --- | --- | --- | --- |
| primary_leaderboard | `exp007_cont_e050_abs_e150` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/eval_epoch050_val200/eval/val_quick_global_eval.json` | `a912a823af552e3b245d0cb03dee822fb54f923d4216d84e5539f315d014cadd` |
| primary_leaderboard | `exp017_phase1_e075` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_iso07_hu_ddp_bs4_20260815T092119Z/ddp_bs4/eval_epoch075_val200/eval/val_quick_global_eval.json` | `3b83ad1b13dc2c476bd147ca298132c499adeb892e52d354b1f2e78871f7aaf6` |
| primary_leaderboard | `exp017_cont_e050_abs_e150` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_cont100_from_iso07_epoch100_20260816T165351Z/ddp_bs4/eval_epoch050_val200/eval/val_quick_global_eval.json` | `12b78c2df274e3a7e0295377ec180b1df350eb3a3bbd080bafdbd51335a0653b` |
| primary_leaderboard | `exp007_cont_e075_abs_e175` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/eval_epoch075_val200/eval/val_quick_global_eval.json` | `c626a8144f68b7fa24d84d9467a2c7c01518f5217c57d5320f5ddb2d63d83fac` |
| primary_leaderboard | `exp009_baseline_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/baseline_cont100/eval_epoch100_val200/eval/val_quick_global_eval.json` | `dde9f3ef77b32f3859e4c3f684a55d71baed4e36950ce4e4af02fd23fc022a3d` |
| primary_leaderboard | `exp008_shared_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation/runs/exp008_dual_branch_20260726T182647Z/v1_sharedfusion_softguide/eval_epoch100_val200/eval/val_quick_global_eval.json` | `34ef0eb159c61818f3e905819e221786edfdcd772afa5321a6256c7d5e03382c` |
| primary_leaderboard | `exp007_cont_e100_abs_e200` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/eval_epoch100_val200/eval/val_quick_global_eval.json` | `69b66b9ba344644149cda47258158e825550a663126d92b6e0939209c57572fa` |
| primary_leaderboard | `exp007_ddp_e050` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_full_20260725T231624Z/ddp_bs4/eval_epoch050_val200/eval/val_quick_global_eval.json` | `1e71f12e5d41f6dd3715565a3a4144a35bcc6ea0ed203a14db1c2e671977d89d` |
| primary_leaderboard | `exp008_joint_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation/runs/exp008_dual_branch_20260726T182647Z/v3_dualfusion_softguide_joint/eval_epoch100_val200/eval/val_quick_global_eval.json` | `a0d3cf68851806797ed6b20e7cdb84d73241d8396158ff4853df5f84b07fa019` |
| primary_leaderboard | `exp008_dual_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation/runs/exp008_dual_branch_20260726T182647Z/v1_dualfusion_softguide/eval_epoch100_val200/eval/val_quick_global_eval.json` | `ef741d90b38234e7baf477fd002ab6be9c5e5f534f8c17b718abafe2812ca249` |
| primary_leaderboard | `exp017_phase1_e050` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_iso07_hu_ddp_bs4_20260815T092119Z/ddp_bs4/eval_epoch050_val200/eval/val_quick_global_eval.json` | `4a72349e774187e551788d9dd5b293fe272199400f6ca47961bafb7d06a3488b` |
| primary_leaderboard | `exp008_precision_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation/runs/exp008_dual_branch_20260726T182647Z/v2_dualfusion_precision/eval_epoch100_val200/eval/val_quick_global_eval.json` | `c8edad1d1256c0366b1fdbedc6ffb1684edf4fb2446274db97fdfc86e76ed66d` |
| primary_leaderboard | `exp007_ddp_e075` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_full_20260725T231624Z/ddp_bs4/eval_epoch075_val200/eval/val_quick_global_eval.json` | `e694f52a30e81e1bac2c5764b53e17065a8ed755b63c93ef7ebfb8fec93494ba` |
| primary_leaderboard | `exp007_ddp_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_full_20260725T231624Z/ddp_bs4/eval_epoch100_val200/eval/val_quick_global_eval.json` | `52a639fc13acef4742ce8432f69b79f99e3de3ddd2926838cf3077a0a060c35b` |
| primary_leaderboard | `exp012_r01_category_2a_replay50_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_20260801T215219Z/category_2a_replay50/eval_epoch100_val200/eval/val_quick_global_eval.json` | `413d6a3ad57ba4e81668b8e7623fd1790ecab70a716d3e0bd174c4741556dcaa` |
| primary_leaderboard | `exp009_s3v1_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/s3v1_fixedrho_suppress_half_quarter/eval_epoch100_val200/eval/val_quick_global_eval.json` | `ba5634190cf14da69a4d5f45ba19e825ff237d5596961c7756428ad0dd0e85d2` |
| primary_leaderboard | `exp014_ddp_bs16_e050` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched/runs/exp014_ddp_bs16_update_matched_20260804T051845Z/ddp_bs16/eval_epoch050_val200/eval/val_quick_global_eval.json` | `8c19307c81c75789bd0ae69f22280a4e2e5459a807057ea363a8ddd88ef38502` |
| primary_leaderboard | `exp017_phase1_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_iso07_hu_ddp_bs4_20260815T092119Z/ddp_bs4/eval_epoch100_val200/eval/val_quick_global_eval.json` | `e7eb19e788f91d6eceb98d64e6bb78df11ecd6c18c5f9cb10e9a0268d1f9eaa8` |
| primary_leaderboard | `exp017_cont_e025_abs_e125` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_cont100_from_iso07_epoch100_20260816T165351Z/ddp_bs4/eval_epoch025_val200/eval/val_quick_global_eval.json` | `d4cf09ed2e2068d9b7d042422871eae9c320cbaec460f1a8284d2c02ae0dee65` |
| primary_leaderboard | `exp014_ddp_bs16_e075` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched/runs/exp014_ddp_bs16_update_matched_20260804T051845Z/ddp_bs16/eval_epoch075_val200/eval/val_quick_global_eval.json` | `23996a668bc74ed07dd1ba470c84f64e40a5ed840cba31da18395147937b6547` |
| primary_leaderboard | `exp012_r02_category_2d_replay50_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_r02_replay_ratio_20260802T122657Z/category_2d_replay50/eval_epoch100_val200/eval/val_quick_global_eval.json` | `d9ab844e97cce6cc366ac7841c9208515fb3401d5ac40b1974d43deb1cc6438f` |
| primary_leaderboard | `exp011_clip_zscore_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/011_voxtell_v123_e4d4_ct_normalization_ablation/runs/exp011_ct_norm_e5d4_20260728T225333Z/v123_e5d4_clip1024_zscore/eval_epoch100_val200/eval/val_quick_global_eval.json` | `5a8a4f6eec61df0d3b1ee6f64ad5cca955665b6fa9c1a3e71764e5923438e10c` |
| primary_leaderboard | `exp011_native_zscore_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/011_voxtell_v123_e4d4_ct_normalization_ablation/runs/exp011_ct_norm_e5d4_20260728T225333Z/v123_e5d4_zscore/eval_epoch100_val200/eval/val_quick_global_eval.json` | `b15afec8bece31868ee0bed3f162014b03ee14da0574058900dc8b22a4fc391f` |
| primary_leaderboard | `exp012_r01_category_1alldiffuse_replay50_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_20260801T215219Z/category_1alldiffuse_replay50/eval_epoch100_val200/eval/val_quick_global_eval.json` | `2fc01ba31473af5ce92f19f1085bdbe3bc00cf584a8bf8d122b675e2a18be2a8` |
| primary_leaderboard | `exp009_s3v2_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/s3v2_balanced_feature_half_quarter/eval_epoch100_val200/eval/val_quick_global_eval.json` | `67f479054b2a4e81c4c9b0d2acfd1bed94878f003ea308e69eb06200824ca056` |
| primary_leaderboard | `exp006_e5d4_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/runs/exp006_cached_native_lr_20260725T050001Z/v123_cached_e5_d4/eval_epoch100_val200/eval/val_quick_global_eval.json` | `a0282d36cc8246a2a996908d5af94e6b3e1bb0fe3542d52bf2bc1e3c0778e93b` |
| primary_leaderboard | `exp006_e6d4_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/runs/exp006_cached_native_lr_20260725T050001Z/v123_cached_e6_d4/eval_epoch100_val200/eval/val_quick_global_eval.json` | `0db404a00a4a4b5f7feab661c72099b38d0335a69217612411bfd98c2b6e78b2` |
| primary_leaderboard | `exp017_phase1_e025` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_iso07_hu_ddp_bs4_20260815T092119Z/ddp_bs4/eval_epoch025_val200/eval/val_quick_global_eval.json` | `7f888c2fbec21d4ea8dae5ce7de520c89ac3eddd9a7fd92d44afd8c9e5878ae3` |
| primary_leaderboard | `exp008_shared_e080` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation/runs/exp008_dual_branch_20260726T182647Z/v1_sharedfusion_softguide/eval_epoch080_val200/eval/val_quick_global_eval.json` | `ec103783c29068d75d1cba4192460c6966e59b86c05d3cc4314ff034741ad707` |
| primary_leaderboard | `exp008_joint_e080` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation/runs/exp008_dual_branch_20260726T182647Z/v3_dualfusion_softguide_joint/eval_epoch080_val200/eval/val_quick_global_eval.json` | `2d9e5dc812f3ce9bd341ab8a4bdeac73a57006b2a0e091cf9d867c86a82e16ab` |
| primary_leaderboard | `exp012_r02_category_1alldiffuse_replay25_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_r02_replay_ratio_20260802T122657Z/category_1alldiffuse_replay25/eval_epoch100_val200/eval/val_quick_global_eval.json` | `39c3de5d917267920d0ce5b9ad11d1b30edaeda246c3e93c5b5eebca53329480` |
| primary_leaderboard | `exp011_linear_hu_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/011_voxtell_v123_e4d4_ct_normalization_ablation/runs/exp011_ct_norm_e5d4_20260728T225333Z/v123_e5d4_clip1024_linear/eval_epoch100_val200/eval/val_quick_global_eval.json` | `865ec9183ab8d62711972976f2a975a164b73455e6c1c3454f0c628b9ad5d9fd` |
| primary_leaderboard | `exp017_cont_e075_abs_e175` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched/runs/exp017_cont100_from_iso07_epoch100_20260816T165351Z/ddp_bs4/eval_epoch075_val200/eval/val_quick_global_eval.json` | `8ae99e2c5ce98eb522625f04dcdc9ecff0c0f836e2cc2fa3d7233f9655046168` |
| primary_leaderboard | `exp009_s3v3_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation/runs/exp009_s3_attention_20260727T081747Z/s3v3_logit_residual_half_quarter/eval_epoch100_val200/eval/val_quick_global_eval.json` | `95c2ed99cf25c81bc8bfff99170acbe816f5d1bd09f9c4b95f564ad5582cd85a` |
| primary_leaderboard | `exp008_precision_e080` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation/runs/exp008_dual_branch_20260726T182647Z/v2_dualfusion_precision/eval_epoch080_val200/eval/val_quick_global_eval.json` | `9d3b70252fa5cd61e685ce86c25556ee9464d8a77a5faee776b2aee12777978b` |
| primary_leaderboard | `exp012_r01_category_2c_replay50_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_20260801T215219Z/category_2c_replay50/eval_epoch100_val200/eval/val_quick_global_eval.json` | `39d9188589b4b263f13d3e1277322453cc694d7401f6abacde37bc3082dfc000` |
| primary_leaderboard | `exp014_ddp_bs16_e025` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched/runs/exp014_ddp_bs16_update_matched_20260804T051845Z/ddp_bs16/eval_epoch025_val200/eval/val_quick_global_eval.json` | `8ae1f302a945c6acd69a6f5cc6112ec1aab29a046dc2c1d4e0a3e5651bb30b41` |
| primary_leaderboard | `exp012_r01_category_2b_replay50_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_20260801T215219Z/category_2b_replay50/eval_epoch100_val200/eval/val_quick_global_eval.json` | `df5501b72181894f4d8b19aaf10ae3329f9155f1532e6d19e5a7ea1accccc175` |
| primary_leaderboard | `exp008_dual_e080` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation/runs/exp008_dual_branch_20260726T182647Z/v1_dualfusion_softguide/eval_epoch080_val200/eval/val_quick_global_eval.json` | `4e97cd9584c2e58786b7db1c608707e35f5bad63071476736c55d70bf45c3799` |
| primary_leaderboard | `exp013_category_1all_diffuse_target100_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists/runs/exp013_category_only_voxtell_20260803T061201Z/category_1all_diffuse_target100/eval_epoch100_val200/eval/val_quick_global_eval.json` | `8ff421e9a8ca4eb90dfb5a013f05d325363daa0ffac69ce778eaec57588226dd` |
| primary_leaderboard | `exp007_ddp_e025` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_full_20260725T231624Z/ddp_bs4/eval_epoch025_val200/eval/val_quick_global_eval.json` | `730e8e500e7fc0eb1fb6bccb4a0f81d9ba1c11158394a29e6e13c02f4766149b` |
| primary_leaderboard | `exp011_e4d4_clip_linear_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/011_voxtell_v123_e4d4_ct_normalization_ablation/runs/exp011_ct_norm_e4d4_20260728T085103Z/v123_e4d4_clip1024_linear/eval_epoch100_val200/eval/val_quick_global_eval.json` | `9a4f76e7fd0a53c2cf526262f007535738b6ffa28ab1f634c636998127aeaa57` |
| primary_leaderboard | `exp012_r02_category_1alldiffuse_replay10_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_r02_replay_ratio_20260802T122657Z/category_1alldiffuse_replay10/eval_epoch100_val200/eval/val_quick_global_eval.json` | `cdc991c79e12e26f3458355392439cc15e7b7f39d4e1d0e076ba7483ee028f50` |
| primary_leaderboard | `exp013_category_2all_focal_target100_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists/runs/exp013_category_only_voxtell_20260803T061201Z/category_2all_focal_target100/eval_epoch100_val200/eval/val_quick_global_eval.json` | `90c274d2b29bb8ef5a164cf9b4c29f81989ab73a682cc5c97aa0aefb30a90e12` |
| primary_leaderboard | `exp011_e4d4_clip_zscore_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/011_voxtell_v123_e4d4_ct_normalization_ablation/runs/exp011_ct_norm_e4d4_20260728T085103Z/v123_e4d4_clip1024_zscore/eval_epoch100_val200/eval/val_quick_global_eval.json` | `2c479d0b530a0ce684bcbc9adac705d40f2c01702c5b783736e0f2fed604c4bb` |
| primary_leaderboard | `exp011_e4d4_native_zscore_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/011_voxtell_v123_e4d4_ct_normalization_ablation/runs/exp011_ct_norm_e4d4_20260728T085103Z/v123_e4d4_zscore/eval_epoch100_val200/eval/val_quick_global_eval.json` | `c05b9d55e06f0fc7e2ef46588030f5224edafee6af7b86f5500cbe9182969c94` |
| primary_leaderboard | `exp013_category_2d_target100_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists/runs/exp013_category_only_voxtell_20260803T061201Z/category_2d_target100/eval_epoch100_val200/eval/val_quick_global_eval.json` | `7cc2d6fe7cca130fdb07e5e20d2a1198030456d0e056837fc9b83de77c6e16a5` |
| primary_leaderboard | `exp006_e7d4_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/runs/exp006_cached_native_lr_20260725T050001Z/v123_cached_e7_d4/eval_epoch100_val200/eval/val_quick_global_eval.json` | `2b50a5a01237a5c582af061fcc0bd8fbf6affa611117fdd5c341e70f5e4c95fd` |
| primary_leaderboard | `exp012_r02_category_1alldiffuse_replay00_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_r02_replay_ratio_20260802T122657Z/category_1alldiffuse_replay00/eval_epoch100_val200/eval/val_quick_global_eval.json` | `65d82d5505ce1faff927eb316e5637b5308b959e4baf12b7d5980ae95dd25e2c` |
| primary_leaderboard | `exp007_cont_e025_abs_e125` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/eval_epoch025_val200/eval/val_quick_global_eval.json` | `bb2b283916d09d7b84ec7d9033de99ea314ac9ea8af9ec5dfe62de6514997963` |
| primary_leaderboard | `exp004_native192_cont100_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft/runs/exp004_paired_20260724T081000Z/native192_cont100/eval_epoch100_val200/eval/val_quick_global_eval.json` | `1643066407aafa5e430340a2b384f8c656ed5b6f0baaaa311da2d2a1899bc5b3` |
| primary_leaderboard | `exp003_v123_opt_poscrop_emptyloss_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v123_opt_poscrop_emptyloss/full_100ep/eval_epoch100_val200/eval/val_quick_global_eval.json` | `23cd7d48106cf40ca458ee772eaebcb6ca53d0b5bed70fa389a0702db98ae7a1` |
| primary_leaderboard | `exp006_e7d6_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/runs/exp006_cached_native_lr_20260725T050001Z/v123_cached_e7_d6/eval_epoch100_val200/eval/val_quick_global_eval.json` | `a8da7ccef459fb8478e91e404f770072868c94d7e24f8a59a9cdefac275f921c` |
| primary_leaderboard | `exp013_category_2bc_target100_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists/runs/exp013_category_only_voxtell_20260803T061201Z/category_2bc_target100/eval_epoch100_val200/eval/val_quick_global_eval.json` | `b9ba4c0f49a1bd7cf7bab15fd06bd2857d052c8e5be3fe52cab770ccbb4cd044` |
| primary_leaderboard | `exp003_v1234_opt_poscrop_emptyloss_ds_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v1234_opt_poscrop_emptyloss_ds/full_100ep/eval_epoch100_val200/eval/val_quick_global_eval.json` | `32c5c02ffd8e861deb9eb6ecde2f95509a45ef5578eda2bc4f3bc6cdb3d67dc1` |
| primary_leaderboard | `exp003_v12_opt_poscrop_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v12_opt_poscrop/full_100ep/eval_epoch100_val200/eval/val_quick_global_eval.json` | `354938d34d8980b714d2bdc141d36ceb05def1ceb26103fcf66562e05f50f1f5` |
| primary_leaderboard | `exp003_v1_opt_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v1_opt/full_100ep/eval_epoch100_val200/eval/val_quick_global_eval.json` | `301b4545958a2e1a55d31bee8e0a6654c693143854aa34c87bf5fea953cdc980` |
| primary_leaderboard | `exp004_iso2mm_global192_cont100_e100` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft/runs/exp004_paired_20260724T081000Z/iso2mm_global192_cont100/eval_epoch100_val200/eval/val_quick_global_eval.json` | `1bdbe64e70c99c7c30399421e7d28d7ec99df925d246077f91646366e166ae2c` |
| direct_2d_target_census_evidence | `category_2d_replay50:e0 source checkpoint (2d slice of union evaluator)` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100/runs/exp012_category_specialists_r02_replay_ratio_20260802T122657Z/category_2d_replay50/eval_epoch000_union/eval/val_quick_global_eval.json` | `1775f03bfa194689c21a1a8e899ba1f5a0efbf3e7c57d5afec40f1bd93a0eaa6` |
| direct_2d_target_census_evidence | `category_2d_target100:e0 target-census checkpoint` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists/runs/exp013_category_only_voxtell_20260803T061201Z/category_2d_target100/eval_epoch000_target/eval/val_quick_global_eval.json` | `ac61380eb92f790548366b4c985db0619beff6f9e54ec71671bd4b268637d6a5` |
| direct_2d_target_census_evidence | `category_2d_target100:e20 target-census checkpoint` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists/runs/exp013_category_only_voxtell_20260803T061201Z/category_2d_target100/eval_epoch020_target/eval/val_quick_global_eval.json` | `fded7c7d24c7bd4eb85b0c2746f008f8f378badf479b4490acc1bf0419e0032c` |
| direct_2d_target_census_evidence | `category_2d_target100:e40 target-census checkpoint` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists/runs/exp013_category_only_voxtell_20260803T061201Z/category_2d_target100/eval_epoch040_target/eval/val_quick_global_eval.json` | `a9f89d4c87525b24d676234e3fc55202a9c9cd94d79f2f1cb3ca4d060419058c` |
| direct_2d_target_census_evidence | `category_2d_target100:e60 target-census checkpoint` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists/runs/exp013_category_only_voxtell_20260803T061201Z/category_2d_target100/eval_epoch060_target/eval/val_quick_global_eval.json` | `9ce8f273db778278a4c4983d32ba6754002ea9bf9cdc12528a584317cf32c75f` |
| direct_2d_target_census_evidence | `category_2d_target100:e80 target-census checkpoint` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists/runs/exp013_category_only_voxtell_20260803T061201Z/category_2d_target100/eval_epoch080_target/eval/val_quick_global_eval.json` | `9873ad15c4e5e0295f1b9ff99d0a05ad4573d8a1188f23b2b6a3028d10912494` |

The source catalog itself is `side_experiments/sideexp002_multimodel_ensemble_selection/outputs/baseline_per_category_performance.json` with SHA-256 `cf9dadd4ac3b3ad869ab889596309ee61ff19afa8dab281249a4012696f3cfec`. The fixed category map is `configs/evaluation/rexgroundingct_val200_seed20260723.json` with SHA-256 `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`.
