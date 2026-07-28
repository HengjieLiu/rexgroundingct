---
created: 2026-07-27
updated: 2026-07-27
status: preliminary_result
experiment_id: 010_voxtell_public_anatomy_prior_fusion
source_prediction_experiment: 006_voxtell_cached_native_v123_lr_ablation
---

# Prompt-Only Whole-Lung Gating On Exp006 v123_cached_e5_d4 Val200

## Summary

This report evaluates whether deterministic prompt-only rules can choose
which findings are safe for hard whole-lung post-processing. Routes are
computed from the free-text finding string before any ground-truth
containment or prediction metrics are read.

The tested post-processing is:

```text
prediction := prediction AND TotalSegmentator whole_lung_mask
```

The primary conclusion is:

- No prompt-only hard whole-lung policy passed the predeclared deployment-candidate checks.
- Treat hard whole-lung gating as a validation-only finding for now; prefer soft anatomy fusion or a learned/selective policy.

## Inputs

- Fixed validation JSON: `/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json`.
- Test prompt source: `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`.
- Prediction source: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/runs/exp006_cached_native_lr_20260725T050001Z/v123_cached_e5_d4/eval_epoch100_val200/predictions`.
- Anatomy cache: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/totalsegmentator_total_fast_3mm_v2_16_0/cases`.
- Whole-lung labels: TotalSegmentator `{10, 11, 12, 13, 14}`.
- Frozen router JSON SHA256: `261dd928947c0e91dd3137a5934d67588aa60a98281c66bcaf01e0db53707f2b`.

## Prompt-Only Policy Results

| Policy | Val selected | Test selected | Dice/finding | Delta Dice | Hit rate | Hits | Target retained | Removed targets | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline, no gate | 0 / 381 | - | 0.3241 | 0 | 0.7612 | 290 / 381 | - | - | - |
| `strict_clean_lung` | 271 / 381 | 365 / 582 | 0.3252 | +0.0011 | 0.7664 | 292 / 381 | 0.8924 | 0 | no |
| `allow_peripheral` | 277 / 381 | 385 / 582 | 0.3252 | +0.0011 | 0.7664 | 292 / 381 | 0.8932 | 0 | no |
| `broad_lung_prompt` | 287 / 381 | 405 / 582 | 0.3255 | +0.0014 | 0.7664 | 292 / 381 | 0.8943 | 0 | no |

## Oracle Reference

| Setting | Dice/finding | Hit rate | Hits |
| --- | ---: | ---: | ---: |
| Baseline, no gate | 0.3241 | 0.7612 | 290 / 381 |
| Oracle 100% contained | 0.3306 | 0.7690 | 293 / 381 |
| Oracle >=95% contained | 0.3340 | 0.7743 | 295 / 381 |
| Oracle >=90% contained | 0.3338 | 0.7743 | 295 / 381 |

## Selected-Subset Behavior

| Policy | Dice before -> after | Hits before -> after | Improved | Worsened | Hit gains/losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `strict_clean_lung` | 0.3265 -> 0.3281 | 213 -> 215 | 152 | 89 | +6 / -4 |
| `allow_peripheral` | 0.3252 -> 0.3267 | 216 -> 218 | 156 | 89 | +6 / -4 |
| `broad_lung_prompt` | 0.3198 -> 0.3217 | 221 -> 223 | 165 | 89 | +6 / -4 |

## Target Containment

| Policy | Selected | 100% | >=95% | >=90% | >=50% | Min fraction | Mean fraction | Aggregate retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict_clean_lung` | 271 | 74 | 153 | 187 | 263 | 0.0755 | 0.9006 | 0.8924 |
| `allow_peripheral` | 277 | 76 | 157 | 192 | 269 | 0.0755 | 0.9011 | 0.8932 |
| `broad_lung_prompt` | 287 | 82 | 166 | 201 | 279 | 0.0755 | 0.9040 | 0.8943 |

## Clipped Examples

Examples below are selected findings whose target had less than 95% whole-lung containment.

### `strict_clean_lung`

| Case | Finding | Category | Target in lung | Dice before -> after | Prompt |
| --- | ---: | --- | ---: | ---: | --- |
| train_13189_a_1.nii.gz | 1 | 2d | 0.0755 | 0.0039 -> 0.0039 | Vascular structure extending to the right lower lobe from the aorta, consistent with pulmonary sequestration |
| train_19325_a_2.nii.gz | 3 | 2b | 0.2122 | 0.2380 -> 0.1358 | Focal area of consolidation in the lingula |
| train_18964_a_1.nii.gz | 0 | 2b | 0.2609 | 0.1039 -> 0.0440 | Consolidation with air bronchograms in the right lower lobe |
| train_3027_a_2.nii.gz | 0 | 2a | 0.3501 | 0.0875 -> 0.0600 | Pulmonary scarring present |
| train_1957_a_1.nii.gz | 3 | 2b | 0.4052 | 0.5414 -> 0.2986 | Consolidation in the right lung posterobasal segment |
| train_19456_a_1.nii.gz | 1 | 2a | 0.4288 | 0.6309 -> 0.4041 | Linear scarring/atelectasis in the medial segment of the right middle lobe |

### `allow_peripheral`

| Case | Finding | Category | Target in lung | Dice before -> after | Prompt |
| --- | ---: | --- | ---: | ---: | --- |
| train_13189_a_1.nii.gz | 1 | 2d | 0.0755 | 0.0039 -> 0.0039 | Vascular structure extending to the right lower lobe from the aorta, consistent with pulmonary sequestration |
| train_19325_a_2.nii.gz | 3 | 2b | 0.2122 | 0.2380 -> 0.1358 | Focal area of consolidation in the lingula |
| train_18964_a_1.nii.gz | 0 | 2b | 0.2609 | 0.1039 -> 0.0440 | Consolidation with air bronchograms in the right lower lobe |
| train_3027_a_2.nii.gz | 0 | 2a | 0.3501 | 0.0875 -> 0.0600 | Pulmonary scarring present |
| train_1957_a_1.nii.gz | 3 | 2b | 0.4052 | 0.5414 -> 0.2986 | Consolidation in the right lung posterobasal segment |
| train_19456_a_1.nii.gz | 1 | 2a | 0.4288 | 0.6309 -> 0.4041 | Linear scarring/atelectasis in the medial segment of the right middle lobe |

### `broad_lung_prompt`

| Case | Finding | Category | Target in lung | Dice before -> after | Prompt |
| --- | ---: | --- | ---: | ---: | --- |
| train_13189_a_1.nii.gz | 1 | 2d | 0.0755 | 0.0039 -> 0.0039 | Vascular structure extending to the right lower lobe from the aorta, consistent with pulmonary sequestration |
| train_19325_a_2.nii.gz | 3 | 2b | 0.2122 | 0.2380 -> 0.1358 | Focal area of consolidation in the lingula |
| train_18964_a_1.nii.gz | 0 | 2b | 0.2609 | 0.1039 -> 0.0440 | Consolidation with air bronchograms in the right lower lobe |
| train_3027_a_2.nii.gz | 0 | 2a | 0.3501 | 0.0875 -> 0.0600 | Pulmonary scarring present |
| train_1957_a_1.nii.gz | 3 | 2b | 0.4052 | 0.5414 -> 0.2986 | Consolidation in the right lung posterobasal segment |
| train_19456_a_1.nii.gz | 1 | 2a | 0.4288 | 0.6309 -> 0.4041 | Linear scarring/atelectasis in the medial segment of the right middle lobe |

## Interpretation

Prompt-only hard whole-lung gating is not safe enough as a test-default policy under the predeclared checks.

Compared with the oracle gate, prompt-only routing has to trade off recall
risk against false-positive removal without seeing target containment.
The safest next direction is soft anatomy fusion or learned selective
routing, with hard post-processing reserved for narrowly validated
prompt cohorts.

## Runtime Artifacts

- Frozen router JSON: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/010_voxtell_public_anatomy_prior_fusion/reports/prompt_only_lung_gate_router_frozen.json`.
- Frozen router CSV: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/010_voxtell_public_anatomy_prior_fusion/reports/prompt_only_lung_gate_router_frozen.csv`.
- Per-finding CSV: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/010_voxtell_public_anatomy_prior_fusion/reports/prompt_only_lung_gating_val200_per_finding.csv`.
- Summary JSON: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/010_voxtell_public_anatomy_prior_fusion/reports/prompt_only_lung_gating_val200_summary.json`.
