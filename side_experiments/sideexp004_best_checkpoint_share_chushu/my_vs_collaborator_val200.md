# SideExp004 My Best vs Collaborator Best on Val200

This report compares independently selected best Dice values: the winner for a category need not be the same model as the overall-Dice winner.

Shared fixed map: 200 cases / 381 findings, manifest SHA `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`.
My source: `side_experiments/sideexp003_ensemble_method_hub/checkpoint_catalog.json` (`bd7fb876bd90c4f9e2e61ce33a4fcaf3d49493e411e3cab1fdae33f8c68551a8`). Collaborator selection source: `side_experiments/sideexp004_best_checkpoint_share_chushu/best_checkpoint_ensemble_share_20260907/outputs/best_checkpoint_ensemble_share_20260907/selected_models.json` (`67de079b51fbf01152286137e2956806065bf9d3ed4361d3763204743e8a3867`).

## Primary comparison — source-declared raw-like collaborator results

The primary collaborator cohort has 20 rows marked `raw_like` in the supplied share bundle. This is the closest available checkpoint-oriented comparison, not a strict fixed-threshold/checkpoint-SHA provenance equivalence claim.

| Metric | Findings | My best candidate | My Dice | My hits | Collaborator best result | Collaborator Dice | Collaborator hits | Δ mine − collaborator | Higher Dice |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Overall | 381 | `exp007_ddp_bs4_e050_a499ad1c` (`eligible`) | 0.346023 | 296/381 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | 0.345081 | 290/381 | +0.000943 | mine |
| 1a | 3 | `exp017_ddp_bs4_e025_9e07869a` (`eligible`) | 0.154380 | 2/3 | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | 0.120209 | 1/3 | +0.034170 | mine |
| 1b | 11 | `exp012_category_2d_replay50_e100_c4e325e3` (`eligible`) | 0.185143 | 5/11 | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | 0.162315 | 4/11 | +0.022828 | mine |
| 1c | 17 | `exp021_ddp_bs4_e025_4f5717d4` (`eligible`) | 0.212666 | 10/17 | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | 0.201640 | 7/17 | +0.011026 | mine |
| 1d | 6 | `exp026_ddp_bs4_e010_755ae85c` (`eligible`) | 0.148635 | 2/6 | `s3_full200_every1000__update_06000__8a24d746` | 0.143115 | 1/6 | +0.005520 | mine |
| 1e | 11 | `exp026_ddp_bs4_e030_7635a9ff` (`eligible`) | 0.212354 | 7/11 | `attention_scale_screen_1_2_u500__update500__646afe72` | 0.223830 | 8/11 | -0.011476 | collaborator |
| 1f | 4 | `exp007_ddp_bs4_e100_d56474a6` (`eligible`) | 0.244367 | 2/4 | `attention_scale_screen_1_2_u500__update500__646afe72` | 0.259066 | 2/4 | -0.014700 | collaborator |
| 2a | 69 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` (`eligible`) | 0.344451 | 56/69 | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | 0.343231 | 59/69 | +0.001220 | mine |
| 2b | 49 | `exp013_category_2all_focal_target100_e100_ea299444` (`eligible`) | 0.374337 | 38/49 | `s3_full200_every1000__update_06000__8a24d746` | 0.367622 | 37/49 | +0.006716 | mine |
| 2c | 60 | `exp007_ddp_bs4_e050_a499ad1c` (`eligible`) | 0.414803 | 49/60 | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | 0.403797 | 51/60 | +0.011006 | mine |
| 2d | 132 | `exp007_ddp_bs4_e050_a499ad1c` (`eligible`) | 0.394936 | 115/132 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | 0.400448 | 114/132 | -0.005512 | collaborator |
| 2e | 11 | `exp007_ddp_bs4_e020_9fba9484` (`eligible`) | 0.504427 | 9/11 | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | 0.489315 | 9/11 | +0.015113 | mine |
| 2g | 1 | `exp004_iso2mm_global192_cont100_e100_6d94fb99` (`eligible`) | 0.187388 | 1/1 | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | 0.151452 | 1/1 | +0.035936 | mine |
| 2h | 7 | `exp017_ddp_bs4_e075_39aabab2` (`eligible`) | 0.371772 | 4/7 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | 0.268723 | 3/7 | +0.103049 | mine |

## Pipeline-inclusive reference — all normalized collaborator variants

This reference includes four supplied pipeline/post-processing rows. It is useful for observing achieved evaluation results, but is not directly comparable to SideExp003 checkpoint-only inference.

| Metric | Findings | My best candidate | My Dice | My hits | Collaborator best result | Collaborator Dice | Collaborator hits | Δ mine − collaborator | Higher Dice |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Overall | 381 | `exp007_ddp_bs4_e050_a499ad1c` (`eligible`) | 0.346023 | 296/381 | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | 0.353632 | 298/381 | -0.007608 | collaborator |
| 1a | 3 | `exp017_ddp_bs4_e025_9e07869a` (`eligible`) | 0.154380 | 2/3 | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | 0.120209 | 1/3 | +0.034170 | mine |
| 1b | 11 | `exp012_category_2d_replay50_e100_c4e325e3` (`eligible`) | 0.185143 | 5/11 | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | 0.162315 | 4/11 | +0.022828 | mine |
| 1c | 17 | `exp021_ddp_bs4_e025_4f5717d4` (`eligible`) | 0.212666 | 10/17 | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | 0.201640 | 7/17 | +0.011026 | mine |
| 1d | 6 | `exp026_ddp_bs4_e010_755ae85c` (`eligible`) | 0.148635 | 2/6 | `s3_full200_every1000__update_06000__8a24d746` | 0.143115 | 1/6 | +0.005520 | mine |
| 1e | 11 | `exp026_ddp_bs4_e030_7635a9ff` (`eligible`) | 0.212354 | 7/11 | `attention_scale_screen_1_2_u500__update500__646afe72` | 0.223830 | 8/11 | -0.011476 | collaborator |
| 1f | 4 | `exp007_ddp_bs4_e100_d56474a6` (`eligible`) | 0.244367 | 2/4 | `attention_scale_screen_1_2_u500__update500__646afe72` | 0.259066 | 2/4 | -0.014700 | collaborator |
| 2a | 69 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` (`eligible`) | 0.344451 | 56/69 | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | 0.343231 | 59/69 | +0.001220 | mine |
| 2b | 49 | `exp013_category_2all_focal_target100_e100_ea299444` (`eligible`) | 0.374337 | 38/49 | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | 0.376728 | 37/49 | -0.002391 | collaborator |
| 2c | 60 | `exp007_ddp_bs4_e050_a499ad1c` (`eligible`) | 0.414803 | 49/60 | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | 0.423390 | 52/60 | -0.008587 | collaborator |
| 2d | 132 | `exp007_ddp_bs4_e050_a499ad1c` (`eligible`) | 0.394936 | 115/132 | `prompt_location_u10k__semantic_A_e10_s10_fw10` | 0.407002 | 116/132 | -0.012066 | collaborator |
| 2e | 11 | `exp007_ddp_bs4_e020_9fba9484` (`eligible`) | 0.504427 | 9/11 | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | 0.489315 | 9/11 | +0.015113 | mine |
| 2g | 1 | `exp004_iso2mm_global192_cont100_e100_6d94fb99` (`eligible`) | 0.187388 | 1/1 | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | 0.166508 | 1/1 | +0.020879 | mine |
| 2h | 7 | `exp017_ddp_bs4_e075_39aabab2` (`eligible`) | 0.371772 | 4/7 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | 0.268723 | 3/7 | +0.103049 | mine |

## Interpretation boundaries

- `2f` has zero findings in this fixed val200 split and is therefore unavailable.
- My candidates use SideExp003's current catalog ordering (Dice, then hits, then candidate ID); each selected winner's eligibility appears in the table.
- The collaborator bundle's per-finding keys and category labels exactly match the fixed manifest, but it does not provide evaluator configuration, threshold provenance, or checkpoint SHA-256 values.
- Per-category maxima are same-validation selection summaries, not a held-out routed-ensemble estimate.
