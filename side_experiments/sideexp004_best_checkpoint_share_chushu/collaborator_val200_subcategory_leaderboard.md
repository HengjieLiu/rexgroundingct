# SideExp004 Collaborator Val200 Official-Category Leaderboard

Generated deterministically by `build_collaborator_val200_leaderboards.py`; category metrics are recomposed from exact `(file, finding_idx)` alignment to the fixed val200 manifest.

Each category cell is `Dice (hits/findings)`. `2f` is unavailable because the fixed val200 cohort has zero `2f` findings.

## Wide table by normalized collaborator result

| Method family | Model / result | Type | Overall Dice | 1a | 1b | 1c | 1d | 1e | 1f | 2a | 2b | 2c | 2d | 2e | 2f | 2g | 2h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | pipeline | 0.353632 (298/381) | 0.088293 (1/3) | 0.134037 (3/11) | 0.194139 (9/17) | 0.138649 (1/6) | 0.193473 (6/11) | 0.206275 (2/4) | 0.325866 (58/69) | 0.376728 (37/49) | 0.423390 (52/60) | 0.402569 (116/132) | 0.449526 (8/11) | — | 0.166508 (1/1) | 0.187238 (4/7) |
| `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | pipeline | 0.347978 (295/381) | 0.100525 (1/3) | 0.132812 (3/11) | 0.190411 (8/17) | 0.129927 (1/6) | 0.211620 (6/11) | 0.194188 (2/4) | 0.339494 (59/69) | 0.344745 (37/49) | 0.387986 (50/60) | 0.407002 (116/132) | 0.467844 (8/11) | — | 0.080137 (0/1) | 0.164083 (4/7) |
| `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | pipeline | 0.347393 (295/381) | 0.107114 (1/3) | 0.129121 (4/11) | 0.185522 (7/17) | 0.125425 (1/6) | 0.211283 (7/11) | 0.182704 (2/4) | 0.343054 (60/69) | 0.344608 (37/49) | 0.388218 (50/60) | 0.404960 (114/132) | 0.451319 (8/11) | — | 0.103514 (1/1) | 0.183057 (3/7) |
| `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | pipeline | 0.345792 (290/381) | 0.096253 (1/3) | 0.141532 (4/11) | 0.164228 (9/17) | 0.108265 (1/6) | 0.191246 (6/11) | 0.190544 (2/4) | 0.334383 (56/69) | 0.361318 (37/49) | 0.377608 (49/60) | 0.402664 (114/132) | 0.460513 (8/11) | — | 0.064150 (0/1) | 0.268376 (3/7) |
| `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | raw-like | 0.345081 (290/381) | 0.095835 (1/3) | 0.141299 (4/11) | 0.164773 (9/17) | 0.108409 (1/6) | 0.190923 (6/11) | 0.191283 (2/4) | 0.333885 (56/69) | 0.361996 (37/49) | 0.377825 (49/60) | 0.400448 (114/132) | 0.460493 (8/11) | — | 0.066180 (0/1) | 0.268723 (3/7) |
| `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | raw-like | 0.342487 (297/381) | 0.105967 (1/3) | 0.155907 (4/11) | 0.164487 (7/17) | 0.111079 (1/6) | 0.191891 (7/11) | 0.182694 (2/4) | 0.342213 (57/69) | 0.359895 (38/49) | 0.378444 (51/60) | 0.390200 (117/132) | 0.450200 (8/11) | — | 0.139431 (1/1) | 0.228301 (3/7) |
| `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | raw-like | 0.338356 (293/381) | 0.108101 (1/3) | 0.152098 (4/11) | 0.152731 (8/17) | 0.123505 (1/6) | 0.188796 (6/11) | 0.248897 (2/4) | 0.335863 (59/69) | 0.353945 (38/49) | 0.380582 (50/60) | 0.386684 (113/132) | 0.466307 (8/11) | — | 0.082393 (0/1) | 0.128531 (3/7) |
| `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | raw-like | 0.338098 (292/381) | 0.093428 (1/3) | 0.158871 (5/11) | 0.172764 (8/17) | 0.142531 (1/6) | 0.212769 (7/11) | 0.158065 (1/4) | 0.322858 (56/69) | 0.342946 (37/49) | 0.396684 (49/60) | 0.391210 (117/132) | 0.453004 (8/11) | — | 0.110049 (1/1) | 0.058170 (1/7) |
| `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | raw-like | 0.335650 (296/381) | 0.103936 (1/3) | 0.142996 (4/11) | 0.192192 (9/17) | 0.130999 (1/6) | 0.205812 (7/11) | 0.194217 (2/4) | 0.327846 (56/69) | 0.360254 (37/49) | 0.385784 (50/60) | 0.378398 (118/132) | 0.462424 (8/11) | — | 0.114800 (1/1) | 0.047550 (2/7) |
| `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | raw-like | 0.335547 (289/381) | 0.093587 (1/3) | 0.122309 (3/11) | 0.189170 (7/17) | 0.134952 (1/6) | 0.194211 (7/11) | 0.220170 (2/4) | 0.335810 (58/69) | 0.357843 (38/49) | 0.386623 (50/60) | 0.369265 (111/132) | 0.467595 (8/11) | — | 0.090787 (0/1) | 0.184969 (3/7) |
| `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | raw-like | 0.335535 (289/381) | 0.093588 (1/3) | 0.122329 (3/11) | 0.189176 (7/17) | 0.134958 (1/6) | 0.194263 (7/11) | 0.219922 (2/4) | 0.335831 (58/69) | 0.357832 (38/49) | 0.386579 (50/60) | 0.369242 (111/132) | 0.467603 (8/11) | — | 0.090750 (0/1) | 0.184978 (3/7) |
| `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | raw-like | 0.335377 (293/381) | 0.099951 (1/3) | 0.127437 (3/11) | 0.187812 (8/17) | 0.129927 (1/6) | 0.204921 (6/11) | 0.194188 (2/4) | 0.332698 (59/69) | 0.341985 (37/49) | 0.378622 (49/60) | 0.381454 (115/132) | 0.471122 (8/11) | — | 0.080076 (0/1) | 0.146920 (4/7) |
| `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | raw-like | 0.334679 (291/381) | 0.104205 (1/3) | 0.143829 (4/11) | 0.189215 (9/17) | 0.119109 (1/6) | 0.183280 (6/11) | 0.224551 (2/4) | 0.317883 (57/69) | 0.355269 (37/49) | 0.375859 (48/60) | 0.380652 (114/132) | 0.467252 (8/11) | — | 0.112442 (1/1) | 0.197179 (3/7) |
| `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | raw-like | 0.334677 (291/381) | 0.106599 (1/3) | 0.126899 (4/11) | 0.153357 (6/17) | 0.118632 (1/6) | 0.209044 (9/11) | 0.234889 (2/4) | 0.337699 (57/69) | 0.353452 (36/49) | 0.385016 (51/60) | 0.373465 (113/132) | 0.462882 (8/11) | — | 0.025322 (0/1) | 0.157508 (3/7) |
| `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | raw-like | 0.334454 (292/381) | 0.106724 (1/3) | 0.127370 (4/11) | 0.183051 (7/17) | 0.125424 (1/6) | 0.203230 (6/11) | 0.182704 (2/4) | 0.333957 (59/69) | 0.341839 (37/49) | 0.376212 (49/60) | 0.379950 (114/132) | 0.454419 (8/11) | — | 0.103390 (1/1) | 0.179092 (3/7) |
| `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | raw-like | 0.334277 (292/381) | 0.102746 (1/3) | 0.140149 (4/11) | 0.145947 (7/17) | 0.122006 (1/6) | 0.192098 (7/11) | 0.234953 (2/4) | 0.333169 (58/69) | 0.345747 (37/49) | 0.403797 (51/60) | 0.371090 (112/132) | 0.456456 (8/11) | — | 0.046378 (0/1) | 0.147765 (4/7) |
| `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | raw-like | 0.333336 (289/381) | 0.076618 (1/3) | 0.115093 (4/11) | 0.180170 (7/17) | 0.127616 (1/6) | 0.202404 (7/11) | 0.165436 (1/4) | 0.321158 (55/69) | 0.359269 (38/49) | 0.393185 (50/60) | 0.381331 (116/132) | 0.435232 (8/11) | — | 0.074168 (0/1) | 0.033677 (1/7) |
| `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | raw-like | 0.332785 (295/381) | 0.107993 (1/3) | 0.156697 (4/11) | 0.161768 (9/17) | 0.120777 (1/6) | 0.172756 (6/11) | 0.237692 (2/4) | 0.339020 (60/69) | 0.348239 (38/49) | 0.379224 (51/60) | 0.376613 (113/132) | 0.418914 (8/11) | — | 0.089337 (0/1) | 0.113986 (2/7) |
| `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | raw-like | 0.332320 (284/381) | 0.120209 (1/3) | 0.145117 (4/11) | 0.201640 (7/17) | 0.119816 (1/6) | 0.191669 (7/11) | 0.156838 (1/4) | 0.343231 (59/69) | 0.348726 (37/49) | 0.376944 (49/60) | 0.367452 (107/132) | 0.450535 (8/11) | — | 0.141811 (1/1) | 0.112266 (2/7) |
| `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | raw-like | 0.329409 (290/381) | 0.093258 (1/3) | 0.144178 (4/11) | 0.167764 (7/17) | 0.130758 (1/6) | 0.190271 (7/11) | 0.176912 (1/4) | 0.328278 (56/69) | 0.367385 (38/49) | 0.384449 (51/60) | 0.359573 (111/132) | 0.489315 (9/11) | — | 0.151452 (1/1) | 0.069204 (3/7) |
| `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | raw-like | 0.328925 (290/381) | 0.076260 (1/3) | 0.110398 (3/11) | 0.167100 (8/17) | 0.141116 (1/6) | 0.177063 (6/11) | 0.230357 (2/4) | 0.319096 (58/69) | 0.365346 (37/49) | 0.390301 (49/60) | 0.361020 (112/132) | 0.454680 (8/11) | — | 0.124079 (1/1) | 0.171856 (4/7) |
| `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | raw-like | 0.328351 (294/381) | 0.077663 (1/3) | 0.129214 (4/11) | 0.187505 (8/17) | 0.143115 (1/6) | 0.191462 (7/11) | 0.218559 (2/4) | 0.319582 (58/69) | 0.367622 (37/49) | 0.390162 (50/60) | 0.355601 (113/132) | 0.447122 (8/11) | — | 0.148108 (1/1) | 0.134392 (4/7) |
| `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | raw-like | 0.326589 (287/381) | 0.103388 (1/3) | 0.129498 (4/11) | 0.166674 (7/17) | 0.141166 (1/6) | 0.223830 (8/11) | 0.259066 (2/4) | 0.309165 (56/69) | 0.355237 (37/49) | 0.393130 (50/60) | 0.355349 (110/132) | 0.466174 (8/11) | — | 0.115578 (1/1) | 0.148644 (2/7) |
| `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | raw-like | 0.325187 (292/381) | 0.096538 (1/3) | 0.162315 (4/11) | 0.142846 (7/17) | 0.125674 (1/6) | 0.204750 (7/11) | 0.150833 (1/4) | 0.320338 (58/69) | 0.362513 (38/49) | 0.373368 (51/60) | 0.363833 (114/132) | 0.466737 (9/11) | — | 0.063439 (0/1) | 0.041577 (1/7) |

## Ranked within each official category — all normalized variants

Pipeline rows are retained and labeled in this inclusive result-level view.

| Category | Findings | Rank | Dice | Hits | Type | Method family | Model / result | Checkpoint provenance |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 1a | 3 | 1 | 0.120209 | 1/3 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1a | 3 | 2 | 0.108101 | 1/3 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1a | 3 | 3 | 0.107993 | 1/3 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1a | 3 | 4 | 0.107114 | 1/3 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 1a | 3 | 5 | 0.106724 | 1/3 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1a | 3 | 6 | 0.106599 | 1/3 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1a | 3 | 7 | 0.105967 | 1/3 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1a | 3 | 8 | 0.104205 | 1/3 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1a | 3 | 9 | 0.103936 | 1/3 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1a | 3 | 10 | 0.103388 | 1/3 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1a | 3 | 11 | 0.102746 | 1/3 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1a | 3 | 12 | 0.100525 | 1/3 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 1a | 3 | 13 | 0.099951 | 1/3 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1a | 3 | 14 | 0.096538 | 1/3 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 1a | 3 | 15 | 0.096253 | 1/3 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 1a | 3 | 16 | 0.095835 | 1/3 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1a | 3 | 17 | 0.093588 | 1/3 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1a | 3 | 18 | 0.093587 | 1/3 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1a | 3 | 19 | 0.093428 | 1/3 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1a | 3 | 20 | 0.093258 | 1/3 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1a | 3 | 21 | 0.088293 | 1/3 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 1a | 3 | 22 | 0.077663 | 1/3 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1a | 3 | 23 | 0.076618 | 1/3 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1a | 3 | 24 | 0.076260 | 1/3 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1b | 11 | 1 | 0.162315 | 4/11 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 1b | 11 | 2 | 0.158871 | 5/11 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1b | 11 | 3 | 0.156697 | 4/11 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1b | 11 | 4 | 0.155907 | 4/11 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1b | 11 | 5 | 0.152098 | 4/11 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1b | 11 | 6 | 0.145117 | 4/11 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1b | 11 | 7 | 0.144178 | 4/11 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1b | 11 | 8 | 0.143829 | 4/11 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1b | 11 | 9 | 0.142996 | 4/11 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1b | 11 | 10 | 0.141532 | 4/11 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 1b | 11 | 11 | 0.141299 | 4/11 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1b | 11 | 12 | 0.140149 | 4/11 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1b | 11 | 13 | 0.134037 | 3/11 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 1b | 11 | 14 | 0.132812 | 3/11 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 1b | 11 | 15 | 0.129498 | 4/11 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1b | 11 | 16 | 0.129214 | 4/11 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1b | 11 | 17 | 0.129121 | 4/11 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 1b | 11 | 18 | 0.127437 | 3/11 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1b | 11 | 19 | 0.127370 | 4/11 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1b | 11 | 20 | 0.126899 | 4/11 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1b | 11 | 21 | 0.122329 | 3/11 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1b | 11 | 22 | 0.122309 | 3/11 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1b | 11 | 23 | 0.115093 | 4/11 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1b | 11 | 24 | 0.110398 | 3/11 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1c | 17 | 1 | 0.201640 | 7/17 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1c | 17 | 2 | 0.194139 | 9/17 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 1c | 17 | 3 | 0.192192 | 9/17 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1c | 17 | 4 | 0.190411 | 8/17 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 1c | 17 | 5 | 0.189215 | 9/17 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1c | 17 | 6 | 0.189176 | 7/17 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1c | 17 | 7 | 0.189170 | 7/17 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1c | 17 | 8 | 0.187812 | 8/17 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1c | 17 | 9 | 0.187505 | 8/17 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1c | 17 | 10 | 0.185522 | 7/17 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 1c | 17 | 11 | 0.183051 | 7/17 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1c | 17 | 12 | 0.180170 | 7/17 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1c | 17 | 13 | 0.172764 | 8/17 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1c | 17 | 14 | 0.167764 | 7/17 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1c | 17 | 15 | 0.167100 | 8/17 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1c | 17 | 16 | 0.166674 | 7/17 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1c | 17 | 17 | 0.164773 | 9/17 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1c | 17 | 18 | 0.164487 | 7/17 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1c | 17 | 19 | 0.164228 | 9/17 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 1c | 17 | 20 | 0.161768 | 9/17 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1c | 17 | 21 | 0.153357 | 6/17 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1c | 17 | 22 | 0.152731 | 8/17 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1c | 17 | 23 | 0.145947 | 7/17 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1c | 17 | 24 | 0.142846 | 7/17 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 1d | 6 | 1 | 0.143115 | 1/6 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1d | 6 | 2 | 0.142531 | 1/6 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1d | 6 | 3 | 0.141166 | 1/6 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1d | 6 | 4 | 0.141116 | 1/6 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1d | 6 | 5 | 0.138649 | 1/6 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 1d | 6 | 6 | 0.134958 | 1/6 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1d | 6 | 7 | 0.134952 | 1/6 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1d | 6 | 8 | 0.130999 | 1/6 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1d | 6 | 9 | 0.130758 | 1/6 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1d | 6 | 10 | 0.129927 | 1/6 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 1d | 6 | 11 | 0.129927 | 1/6 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1d | 6 | 12 | 0.127616 | 1/6 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1d | 6 | 13 | 0.125674 | 1/6 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 1d | 6 | 14 | 0.125425 | 1/6 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 1d | 6 | 15 | 0.125424 | 1/6 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1d | 6 | 16 | 0.123505 | 1/6 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1d | 6 | 17 | 0.122006 | 1/6 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1d | 6 | 18 | 0.120777 | 1/6 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1d | 6 | 19 | 0.119816 | 1/6 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1d | 6 | 20 | 0.119109 | 1/6 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1d | 6 | 21 | 0.118632 | 1/6 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1d | 6 | 22 | 0.111079 | 1/6 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1d | 6 | 23 | 0.108409 | 1/6 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1d | 6 | 24 | 0.108265 | 1/6 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 1e | 11 | 1 | 0.223830 | 8/11 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1e | 11 | 2 | 0.212769 | 7/11 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1e | 11 | 3 | 0.211620 | 6/11 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 1e | 11 | 4 | 0.211283 | 7/11 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 1e | 11 | 5 | 0.209044 | 9/11 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1e | 11 | 6 | 0.205812 | 7/11 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1e | 11 | 7 | 0.204921 | 6/11 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1e | 11 | 8 | 0.204750 | 7/11 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 1e | 11 | 9 | 0.203230 | 6/11 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1e | 11 | 10 | 0.202404 | 7/11 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1e | 11 | 11 | 0.194263 | 7/11 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1e | 11 | 12 | 0.194211 | 7/11 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1e | 11 | 13 | 0.193473 | 6/11 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 1e | 11 | 14 | 0.192098 | 7/11 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1e | 11 | 15 | 0.191891 | 7/11 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1e | 11 | 16 | 0.191669 | 7/11 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1e | 11 | 17 | 0.191462 | 7/11 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1e | 11 | 18 | 0.191246 | 6/11 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 1e | 11 | 19 | 0.190923 | 6/11 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1e | 11 | 20 | 0.190271 | 7/11 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1e | 11 | 21 | 0.188796 | 6/11 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1e | 11 | 22 | 0.183280 | 6/11 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1e | 11 | 23 | 0.177063 | 6/11 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1e | 11 | 24 | 0.172756 | 6/11 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1f | 4 | 1 | 0.259066 | 2/4 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1f | 4 | 2 | 0.248897 | 2/4 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1f | 4 | 3 | 0.237692 | 2/4 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1f | 4 | 4 | 0.234953 | 2/4 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1f | 4 | 5 | 0.234889 | 2/4 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1f | 4 | 6 | 0.230357 | 2/4 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1f | 4 | 7 | 0.224551 | 2/4 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1f | 4 | 8 | 0.220170 | 2/4 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1f | 4 | 9 | 0.219922 | 2/4 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1f | 4 | 10 | 0.218559 | 2/4 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1f | 4 | 11 | 0.206275 | 2/4 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 1f | 4 | 12 | 0.194217 | 2/4 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1f | 4 | 13 | 0.194188 | 2/4 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1f | 4 | 14 | 0.194188 | 2/4 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 1f | 4 | 15 | 0.191283 | 2/4 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1f | 4 | 16 | 0.190544 | 2/4 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 1f | 4 | 17 | 0.182704 | 2/4 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 1f | 4 | 18 | 0.182704 | 2/4 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1f | 4 | 19 | 0.182694 | 2/4 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1f | 4 | 20 | 0.176912 | 1/4 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1f | 4 | 21 | 0.165436 | 1/4 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1f | 4 | 22 | 0.158065 | 1/4 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1f | 4 | 23 | 0.156838 | 1/4 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1f | 4 | 24 | 0.150833 | 1/4 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2a | 69 | 1 | 0.343231 | 59/69 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2a | 69 | 2 | 0.343054 | 60/69 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 2a | 69 | 3 | 0.342213 | 57/69 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2a | 69 | 4 | 0.339494 | 59/69 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 2a | 69 | 5 | 0.339020 | 60/69 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2a | 69 | 6 | 0.337699 | 57/69 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2a | 69 | 7 | 0.335863 | 59/69 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2a | 69 | 8 | 0.335831 | 58/69 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2a | 69 | 9 | 0.335810 | 58/69 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2a | 69 | 10 | 0.334383 | 56/69 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 2a | 69 | 11 | 0.333957 | 59/69 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2a | 69 | 12 | 0.333885 | 56/69 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2a | 69 | 13 | 0.333169 | 58/69 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2a | 69 | 14 | 0.332698 | 59/69 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2a | 69 | 15 | 0.328278 | 56/69 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2a | 69 | 16 | 0.327846 | 56/69 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2a | 69 | 17 | 0.325866 | 58/69 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 2a | 69 | 18 | 0.322858 | 56/69 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2a | 69 | 19 | 0.321158 | 55/69 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2a | 69 | 20 | 0.320338 | 58/69 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2a | 69 | 21 | 0.319582 | 58/69 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2a | 69 | 22 | 0.319096 | 58/69 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2a | 69 | 23 | 0.317883 | 57/69 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2a | 69 | 24 | 0.309165 | 56/69 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2b | 49 | 1 | 0.376728 | 37/49 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 2b | 49 | 2 | 0.367622 | 37/49 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2b | 49 | 3 | 0.367385 | 38/49 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2b | 49 | 4 | 0.365346 | 37/49 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2b | 49 | 5 | 0.362513 | 38/49 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2b | 49 | 6 | 0.361996 | 37/49 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2b | 49 | 7 | 0.361318 | 37/49 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 2b | 49 | 8 | 0.360254 | 37/49 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2b | 49 | 9 | 0.359895 | 38/49 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2b | 49 | 10 | 0.359269 | 38/49 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2b | 49 | 11 | 0.357843 | 38/49 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2b | 49 | 12 | 0.357832 | 38/49 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2b | 49 | 13 | 0.355269 | 37/49 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2b | 49 | 14 | 0.355237 | 37/49 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2b | 49 | 15 | 0.353945 | 38/49 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2b | 49 | 16 | 0.353452 | 36/49 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2b | 49 | 17 | 0.348726 | 37/49 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2b | 49 | 18 | 0.348239 | 38/49 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2b | 49 | 19 | 0.345747 | 37/49 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2b | 49 | 20 | 0.344745 | 37/49 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 2b | 49 | 21 | 0.344608 | 37/49 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 2b | 49 | 22 | 0.342946 | 37/49 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2b | 49 | 23 | 0.341985 | 37/49 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2b | 49 | 24 | 0.341839 | 37/49 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2c | 60 | 1 | 0.423390 | 52/60 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 2c | 60 | 2 | 0.403797 | 51/60 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2c | 60 | 3 | 0.396684 | 49/60 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2c | 60 | 4 | 0.393185 | 50/60 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2c | 60 | 5 | 0.393130 | 50/60 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2c | 60 | 6 | 0.390301 | 49/60 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2c | 60 | 7 | 0.390162 | 50/60 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2c | 60 | 8 | 0.388218 | 50/60 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 2c | 60 | 9 | 0.387986 | 50/60 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 2c | 60 | 10 | 0.386623 | 50/60 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2c | 60 | 11 | 0.386579 | 50/60 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2c | 60 | 12 | 0.385784 | 50/60 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2c | 60 | 13 | 0.385016 | 51/60 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2c | 60 | 14 | 0.384449 | 51/60 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2c | 60 | 15 | 0.380582 | 50/60 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2c | 60 | 16 | 0.379224 | 51/60 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2c | 60 | 17 | 0.378622 | 49/60 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2c | 60 | 18 | 0.378444 | 51/60 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2c | 60 | 19 | 0.377825 | 49/60 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2c | 60 | 20 | 0.377608 | 49/60 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 2c | 60 | 21 | 0.376944 | 49/60 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2c | 60 | 22 | 0.376212 | 49/60 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2c | 60 | 23 | 0.375859 | 48/60 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2c | 60 | 24 | 0.373368 | 51/60 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2d | 132 | 1 | 0.407002 | 116/132 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 2d | 132 | 2 | 0.404960 | 114/132 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 2d | 132 | 3 | 0.402664 | 114/132 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 2d | 132 | 4 | 0.402569 | 116/132 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 2d | 132 | 5 | 0.400448 | 114/132 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2d | 132 | 6 | 0.391210 | 117/132 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2d | 132 | 7 | 0.390200 | 117/132 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2d | 132 | 8 | 0.386684 | 113/132 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2d | 132 | 9 | 0.381454 | 115/132 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2d | 132 | 10 | 0.381331 | 116/132 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2d | 132 | 11 | 0.380652 | 114/132 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2d | 132 | 12 | 0.379950 | 114/132 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2d | 132 | 13 | 0.378398 | 118/132 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2d | 132 | 14 | 0.376613 | 113/132 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2d | 132 | 15 | 0.373465 | 113/132 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2d | 132 | 16 | 0.371090 | 112/132 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2d | 132 | 17 | 0.369265 | 111/132 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2d | 132 | 18 | 0.369242 | 111/132 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2d | 132 | 19 | 0.367452 | 107/132 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2d | 132 | 20 | 0.363833 | 114/132 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2d | 132 | 21 | 0.361020 | 112/132 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2d | 132 | 22 | 0.359573 | 111/132 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2d | 132 | 23 | 0.355601 | 113/132 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2d | 132 | 24 | 0.355349 | 110/132 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2e | 11 | 1 | 0.489315 | 9/11 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2e | 11 | 2 | 0.471122 | 8/11 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2e | 11 | 3 | 0.467844 | 8/11 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 2e | 11 | 4 | 0.467603 | 8/11 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2e | 11 | 5 | 0.467595 | 8/11 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2e | 11 | 6 | 0.467252 | 8/11 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2e | 11 | 7 | 0.466737 | 9/11 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2e | 11 | 8 | 0.466307 | 8/11 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2e | 11 | 9 | 0.466174 | 8/11 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2e | 11 | 10 | 0.462882 | 8/11 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2e | 11 | 11 | 0.462424 | 8/11 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2e | 11 | 12 | 0.460513 | 8/11 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 2e | 11 | 13 | 0.460493 | 8/11 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2e | 11 | 14 | 0.456456 | 8/11 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2e | 11 | 15 | 0.454680 | 8/11 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2e | 11 | 16 | 0.454419 | 8/11 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2e | 11 | 17 | 0.453004 | 8/11 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2e | 11 | 18 | 0.451319 | 8/11 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 2e | 11 | 19 | 0.450535 | 8/11 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2e | 11 | 20 | 0.450200 | 8/11 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2e | 11 | 21 | 0.449526 | 8/11 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 2e | 11 | 22 | 0.447122 | 8/11 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2e | 11 | 23 | 0.435232 | 8/11 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2e | 11 | 24 | 0.418914 | 8/11 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2g | 1 | 1 | 0.166508 | 1/1 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 2g | 1 | 2 | 0.151452 | 1/1 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2g | 1 | 3 | 0.148108 | 1/1 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2g | 1 | 4 | 0.141811 | 1/1 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2g | 1 | 5 | 0.139431 | 1/1 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2g | 1 | 6 | 0.124079 | 1/1 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2g | 1 | 7 | 0.115578 | 1/1 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2g | 1 | 8 | 0.114800 | 1/1 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2g | 1 | 9 | 0.112442 | 1/1 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2g | 1 | 10 | 0.110049 | 1/1 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2g | 1 | 11 | 0.103514 | 1/1 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 2g | 1 | 12 | 0.103390 | 1/1 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2g | 1 | 13 | 0.090787 | 0/1 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2g | 1 | 14 | 0.090750 | 0/1 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2g | 1 | 15 | 0.089337 | 0/1 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2g | 1 | 16 | 0.082393 | 0/1 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2g | 1 | 17 | 0.080137 | 0/1 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 2g | 1 | 18 | 0.080076 | 0/1 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2g | 1 | 19 | 0.074168 | 0/1 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2g | 1 | 20 | 0.066180 | 0/1 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2g | 1 | 21 | 0.064150 | 0/1 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 2g | 1 | 22 | 0.063439 | 0/1 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2g | 1 | 23 | 0.046378 | 0/1 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2g | 1 | 24 | 0.025322 | 0/1 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2h | 7 | 1 | 0.268723 | 3/7 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2h | 7 | 2 | 0.268376 | 3/7 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable |
| 2h | 7 | 3 | 0.228301 | 3/7 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2h | 7 | 4 | 0.197179 | 3/7 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2h | 7 | 5 | 0.187238 | 4/7 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable |
| 2h | 7 | 6 | 0.184978 | 3/7 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2h | 7 | 7 | 0.184969 | 3/7 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2h | 7 | 8 | 0.183057 | 3/7 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable |
| 2h | 7 | 9 | 0.179092 | 3/7 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2h | 7 | 10 | 0.171856 | 4/7 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2h | 7 | 11 | 0.164083 | 4/7 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable |
| 2h | 7 | 12 | 0.157508 | 3/7 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2h | 7 | 13 | 0.148644 | 2/7 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2h | 7 | 14 | 0.147765 | 4/7 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2h | 7 | 15 | 0.146920 | 4/7 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2h | 7 | 16 | 0.134392 | 4/7 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2h | 7 | 17 | 0.128531 | 3/7 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2h | 7 | 18 | 0.113986 | 2/7 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2h | 7 | 19 | 0.112266 | 2/7 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2h | 7 | 20 | 0.069204 | 3/7 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2h | 7 | 21 | 0.058170 | 1/7 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2h | 7 | 22 | 0.047550 | 2/7 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2h | 7 | 23 | 0.041577 | 1/7 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2h | 7 | 24 | 0.033677 | 1/7 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |

## Ranked within each official category — source-declared raw-like subset

This is the primary source-declared checkpoint-result cohort used by the comparison report.

| Category | Findings | Rank | Dice | Hits | Type | Method family | Model / result | Checkpoint provenance |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 1a | 3 | 1 | 0.120209 | 1/3 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1a | 3 | 2 | 0.108101 | 1/3 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1a | 3 | 3 | 0.107993 | 1/3 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1a | 3 | 4 | 0.106724 | 1/3 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1a | 3 | 5 | 0.106599 | 1/3 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1a | 3 | 6 | 0.105967 | 1/3 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1a | 3 | 7 | 0.104205 | 1/3 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1a | 3 | 8 | 0.103936 | 1/3 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1a | 3 | 9 | 0.103388 | 1/3 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1a | 3 | 10 | 0.102746 | 1/3 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1a | 3 | 11 | 0.099951 | 1/3 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1a | 3 | 12 | 0.096538 | 1/3 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 1a | 3 | 13 | 0.095835 | 1/3 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1a | 3 | 14 | 0.093588 | 1/3 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1a | 3 | 15 | 0.093587 | 1/3 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1a | 3 | 16 | 0.093428 | 1/3 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1a | 3 | 17 | 0.093258 | 1/3 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1a | 3 | 18 | 0.077663 | 1/3 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1a | 3 | 19 | 0.076618 | 1/3 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1a | 3 | 20 | 0.076260 | 1/3 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1b | 11 | 1 | 0.162315 | 4/11 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 1b | 11 | 2 | 0.158871 | 5/11 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1b | 11 | 3 | 0.156697 | 4/11 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1b | 11 | 4 | 0.155907 | 4/11 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1b | 11 | 5 | 0.152098 | 4/11 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1b | 11 | 6 | 0.145117 | 4/11 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1b | 11 | 7 | 0.144178 | 4/11 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1b | 11 | 8 | 0.143829 | 4/11 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1b | 11 | 9 | 0.142996 | 4/11 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1b | 11 | 10 | 0.141299 | 4/11 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1b | 11 | 11 | 0.140149 | 4/11 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1b | 11 | 12 | 0.129498 | 4/11 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1b | 11 | 13 | 0.129214 | 4/11 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1b | 11 | 14 | 0.127437 | 3/11 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1b | 11 | 15 | 0.127370 | 4/11 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1b | 11 | 16 | 0.126899 | 4/11 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1b | 11 | 17 | 0.122329 | 3/11 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1b | 11 | 18 | 0.122309 | 3/11 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1b | 11 | 19 | 0.115093 | 4/11 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1b | 11 | 20 | 0.110398 | 3/11 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1c | 17 | 1 | 0.201640 | 7/17 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1c | 17 | 2 | 0.192192 | 9/17 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1c | 17 | 3 | 0.189215 | 9/17 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1c | 17 | 4 | 0.189176 | 7/17 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1c | 17 | 5 | 0.189170 | 7/17 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1c | 17 | 6 | 0.187812 | 8/17 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1c | 17 | 7 | 0.187505 | 8/17 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1c | 17 | 8 | 0.183051 | 7/17 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1c | 17 | 9 | 0.180170 | 7/17 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1c | 17 | 10 | 0.172764 | 8/17 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1c | 17 | 11 | 0.167764 | 7/17 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1c | 17 | 12 | 0.167100 | 8/17 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1c | 17 | 13 | 0.166674 | 7/17 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1c | 17 | 14 | 0.164773 | 9/17 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1c | 17 | 15 | 0.164487 | 7/17 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1c | 17 | 16 | 0.161768 | 9/17 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1c | 17 | 17 | 0.153357 | 6/17 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1c | 17 | 18 | 0.152731 | 8/17 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1c | 17 | 19 | 0.145947 | 7/17 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1c | 17 | 20 | 0.142846 | 7/17 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 1d | 6 | 1 | 0.143115 | 1/6 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1d | 6 | 2 | 0.142531 | 1/6 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1d | 6 | 3 | 0.141166 | 1/6 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1d | 6 | 4 | 0.141116 | 1/6 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1d | 6 | 5 | 0.134958 | 1/6 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1d | 6 | 6 | 0.134952 | 1/6 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1d | 6 | 7 | 0.130999 | 1/6 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1d | 6 | 8 | 0.130758 | 1/6 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1d | 6 | 9 | 0.129927 | 1/6 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1d | 6 | 10 | 0.127616 | 1/6 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1d | 6 | 11 | 0.125674 | 1/6 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 1d | 6 | 12 | 0.125424 | 1/6 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1d | 6 | 13 | 0.123505 | 1/6 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1d | 6 | 14 | 0.122006 | 1/6 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1d | 6 | 15 | 0.120777 | 1/6 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1d | 6 | 16 | 0.119816 | 1/6 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1d | 6 | 17 | 0.119109 | 1/6 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1d | 6 | 18 | 0.118632 | 1/6 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1d | 6 | 19 | 0.111079 | 1/6 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1d | 6 | 20 | 0.108409 | 1/6 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1e | 11 | 1 | 0.223830 | 8/11 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1e | 11 | 2 | 0.212769 | 7/11 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1e | 11 | 3 | 0.209044 | 9/11 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1e | 11 | 4 | 0.205812 | 7/11 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1e | 11 | 5 | 0.204921 | 6/11 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1e | 11 | 6 | 0.204750 | 7/11 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 1e | 11 | 7 | 0.203230 | 6/11 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1e | 11 | 8 | 0.202404 | 7/11 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1e | 11 | 9 | 0.194263 | 7/11 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1e | 11 | 10 | 0.194211 | 7/11 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1e | 11 | 11 | 0.192098 | 7/11 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1e | 11 | 12 | 0.191891 | 7/11 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1e | 11 | 13 | 0.191669 | 7/11 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1e | 11 | 14 | 0.191462 | 7/11 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1e | 11 | 15 | 0.190923 | 6/11 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1e | 11 | 16 | 0.190271 | 7/11 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1e | 11 | 17 | 0.188796 | 6/11 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1e | 11 | 18 | 0.183280 | 6/11 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1e | 11 | 19 | 0.177063 | 6/11 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1e | 11 | 20 | 0.172756 | 6/11 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1f | 4 | 1 | 0.259066 | 2/4 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 1f | 4 | 2 | 0.248897 | 2/4 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 1f | 4 | 3 | 0.237692 | 2/4 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 1f | 4 | 4 | 0.234953 | 2/4 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 1f | 4 | 5 | 0.234889 | 2/4 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 1f | 4 | 6 | 0.230357 | 2/4 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 1f | 4 | 7 | 0.224551 | 2/4 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 1f | 4 | 8 | 0.220170 | 2/4 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 1f | 4 | 9 | 0.219922 | 2/4 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 1f | 4 | 10 | 0.218559 | 2/4 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 1f | 4 | 11 | 0.194217 | 2/4 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 1f | 4 | 12 | 0.194188 | 2/4 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 1f | 4 | 13 | 0.191283 | 2/4 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 1f | 4 | 14 | 0.182704 | 2/4 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 1f | 4 | 15 | 0.182694 | 2/4 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 1f | 4 | 16 | 0.176912 | 1/4 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 1f | 4 | 17 | 0.165436 | 1/4 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 1f | 4 | 18 | 0.158065 | 1/4 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 1f | 4 | 19 | 0.156838 | 1/4 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 1f | 4 | 20 | 0.150833 | 1/4 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2a | 69 | 1 | 0.343231 | 59/69 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2a | 69 | 2 | 0.342213 | 57/69 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2a | 69 | 3 | 0.339020 | 60/69 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2a | 69 | 4 | 0.337699 | 57/69 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2a | 69 | 5 | 0.335863 | 59/69 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2a | 69 | 6 | 0.335831 | 58/69 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2a | 69 | 7 | 0.335810 | 58/69 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2a | 69 | 8 | 0.333957 | 59/69 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2a | 69 | 9 | 0.333885 | 56/69 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2a | 69 | 10 | 0.333169 | 58/69 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2a | 69 | 11 | 0.332698 | 59/69 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2a | 69 | 12 | 0.328278 | 56/69 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2a | 69 | 13 | 0.327846 | 56/69 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2a | 69 | 14 | 0.322858 | 56/69 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2a | 69 | 15 | 0.321158 | 55/69 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2a | 69 | 16 | 0.320338 | 58/69 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2a | 69 | 17 | 0.319582 | 58/69 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2a | 69 | 18 | 0.319096 | 58/69 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2a | 69 | 19 | 0.317883 | 57/69 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2a | 69 | 20 | 0.309165 | 56/69 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2b | 49 | 1 | 0.367622 | 37/49 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2b | 49 | 2 | 0.367385 | 38/49 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2b | 49 | 3 | 0.365346 | 37/49 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2b | 49 | 4 | 0.362513 | 38/49 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2b | 49 | 5 | 0.361996 | 37/49 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2b | 49 | 6 | 0.360254 | 37/49 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2b | 49 | 7 | 0.359895 | 38/49 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2b | 49 | 8 | 0.359269 | 38/49 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2b | 49 | 9 | 0.357843 | 38/49 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2b | 49 | 10 | 0.357832 | 38/49 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2b | 49 | 11 | 0.355269 | 37/49 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2b | 49 | 12 | 0.355237 | 37/49 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2b | 49 | 13 | 0.353945 | 38/49 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2b | 49 | 14 | 0.353452 | 36/49 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2b | 49 | 15 | 0.348726 | 37/49 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2b | 49 | 16 | 0.348239 | 38/49 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2b | 49 | 17 | 0.345747 | 37/49 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2b | 49 | 18 | 0.342946 | 37/49 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2b | 49 | 19 | 0.341985 | 37/49 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2b | 49 | 20 | 0.341839 | 37/49 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2c | 60 | 1 | 0.403797 | 51/60 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2c | 60 | 2 | 0.396684 | 49/60 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2c | 60 | 3 | 0.393185 | 50/60 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2c | 60 | 4 | 0.393130 | 50/60 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2c | 60 | 5 | 0.390301 | 49/60 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2c | 60 | 6 | 0.390162 | 50/60 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2c | 60 | 7 | 0.386623 | 50/60 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2c | 60 | 8 | 0.386579 | 50/60 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2c | 60 | 9 | 0.385784 | 50/60 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2c | 60 | 10 | 0.385016 | 51/60 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2c | 60 | 11 | 0.384449 | 51/60 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2c | 60 | 12 | 0.380582 | 50/60 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2c | 60 | 13 | 0.379224 | 51/60 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2c | 60 | 14 | 0.378622 | 49/60 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2c | 60 | 15 | 0.378444 | 51/60 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2c | 60 | 16 | 0.377825 | 49/60 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2c | 60 | 17 | 0.376944 | 49/60 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2c | 60 | 18 | 0.376212 | 49/60 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2c | 60 | 19 | 0.375859 | 48/60 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2c | 60 | 20 | 0.373368 | 51/60 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2d | 132 | 1 | 0.400448 | 114/132 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2d | 132 | 2 | 0.391210 | 117/132 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2d | 132 | 3 | 0.390200 | 117/132 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2d | 132 | 4 | 0.386684 | 113/132 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2d | 132 | 5 | 0.381454 | 115/132 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2d | 132 | 6 | 0.381331 | 116/132 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2d | 132 | 7 | 0.380652 | 114/132 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2d | 132 | 8 | 0.379950 | 114/132 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2d | 132 | 9 | 0.378398 | 118/132 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2d | 132 | 10 | 0.376613 | 113/132 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2d | 132 | 11 | 0.373465 | 113/132 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2d | 132 | 12 | 0.371090 | 112/132 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2d | 132 | 13 | 0.369265 | 111/132 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2d | 132 | 14 | 0.369242 | 111/132 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2d | 132 | 15 | 0.367452 | 107/132 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2d | 132 | 16 | 0.363833 | 114/132 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2d | 132 | 17 | 0.361020 | 112/132 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2d | 132 | 18 | 0.359573 | 111/132 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2d | 132 | 19 | 0.355601 | 113/132 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2d | 132 | 20 | 0.355349 | 110/132 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2e | 11 | 1 | 0.489315 | 9/11 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2e | 11 | 2 | 0.471122 | 8/11 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2e | 11 | 3 | 0.467603 | 8/11 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2e | 11 | 4 | 0.467595 | 8/11 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2e | 11 | 5 | 0.467252 | 8/11 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2e | 11 | 6 | 0.466737 | 9/11 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2e | 11 | 7 | 0.466307 | 8/11 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2e | 11 | 8 | 0.466174 | 8/11 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2e | 11 | 9 | 0.462882 | 8/11 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2e | 11 | 10 | 0.462424 | 8/11 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2e | 11 | 11 | 0.460493 | 8/11 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2e | 11 | 12 | 0.456456 | 8/11 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2e | 11 | 13 | 0.454680 | 8/11 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2e | 11 | 14 | 0.454419 | 8/11 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2e | 11 | 15 | 0.453004 | 8/11 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2e | 11 | 16 | 0.450535 | 8/11 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2e | 11 | 17 | 0.450200 | 8/11 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2e | 11 | 18 | 0.447122 | 8/11 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2e | 11 | 19 | 0.435232 | 8/11 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2e | 11 | 20 | 0.418914 | 8/11 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2g | 1 | 1 | 0.151452 | 1/1 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2g | 1 | 2 | 0.148108 | 1/1 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2g | 1 | 3 | 0.141811 | 1/1 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2g | 1 | 4 | 0.139431 | 1/1 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2g | 1 | 5 | 0.124079 | 1/1 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2g | 1 | 6 | 0.115578 | 1/1 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2g | 1 | 7 | 0.114800 | 1/1 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2g | 1 | 8 | 0.112442 | 1/1 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2g | 1 | 9 | 0.110049 | 1/1 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2g | 1 | 10 | 0.103390 | 1/1 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2g | 1 | 11 | 0.090787 | 0/1 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2g | 1 | 12 | 0.090750 | 0/1 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2g | 1 | 13 | 0.089337 | 0/1 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2g | 1 | 14 | 0.082393 | 0/1 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2g | 1 | 15 | 0.080076 | 0/1 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2g | 1 | 16 | 0.074168 | 0/1 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
| 2g | 1 | 17 | 0.066180 | 0/1 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2g | 1 | 18 | 0.063439 | 0/1 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2g | 1 | 19 | 0.046378 | 0/1 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2g | 1 | 20 | 0.025322 | 0/1 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2h | 7 | 1 | 0.268723 | 3/7 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable |
| 2h | 7 | 2 | 0.228301 | 3/7 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable |
| 2h | 7 | 3 | 0.197179 | 3/7 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable |
| 2h | 7 | 4 | 0.184978 | 3/7 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable |
| 2h | 7 | 5 | 0.184969 | 3/7 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable |
| 2h | 7 | 6 | 0.179092 | 3/7 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable |
| 2h | 7 | 7 | 0.171856 | 4/7 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared |
| 2h | 7 | 8 | 0.157508 | 3/7 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable |
| 2h | 7 | 9 | 0.148644 | 2/7 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared |
| 2h | 7 | 10 | 0.147765 | 4/7 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable |
| 2h | 7 | 11 | 0.146920 | 4/7 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable |
| 2h | 7 | 12 | 0.134392 | 4/7 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable |
| 2h | 7 | 13 | 0.128531 | 3/7 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable |
| 2h | 7 | 14 | 0.113986 | 2/7 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable |
| 2h | 7 | 15 | 0.112266 | 2/7 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable |
| 2h | 7 | 16 | 0.069204 | 3/7 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable |
| 2h | 7 | 17 | 0.058170 | 1/7 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared |
| 2h | 7 | 18 | 0.047550 | 2/7 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable |
| 2h | 7 | 19 | 0.041577 | 1/7 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable |
| 2h | 7 | 20 | 0.033677 | 1/7 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable |
