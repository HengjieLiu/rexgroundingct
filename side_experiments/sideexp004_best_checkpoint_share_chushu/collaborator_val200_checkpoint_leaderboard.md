# SideExp004 Collaborator Val200 Result Leaderboard

Generated deterministically by `build_collaborator_val200_leaderboards.py`; do not edit ranking rows by hand.

Fixed validation manifest: `configs/evaluation/rexgroundingct_val200_seed20260723.json` (`7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`), 200 cases / 381 findings.
Cohort: 24 normalized selected collaborator result variants (20 source-declared raw-like; 4 pipeline).

`raw-like` is copied from the supplied bundle; it is not a SideExp003-style eligibility or fixed-threshold provenance claim. The catalog records the source-file hashes and any reported checkpoint path.

## All normalized collaborator result variants

This result-level table retains distinct pipeline evaluations of a reported checkpoint so the supplied bundle remains auditable.

| Rank | Dice | Hits | Type | Method family | Model / result | Checkpoint provenance | Result hash |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 1 | 0.353632 | 298/381 | pipeline | `lobe_roi_segmentation_loss_continue_5000_to10000` | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | declared, not locally verifiable | `5b09b79b1ac7` |
| 2 | 0.347978 | 295/381 | pipeline | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_u10k__semantic_A_e10_s10_fw10` | declared, not locally verifiable | `43d488e67b54` |
| 3 | 0.347393 | 295/381 | pipeline | `nonattention_baseline_selection` | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | declared, not locally verifiable | `a1b37791e35d` |
| 4 | 0.345792 | 290/381 | pipeline | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | declared, not locally verifiable | `de0cd3974b06` |
| 5 | 0.345081 | 290/381 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable | `1bf09c0e9071` |
| 6 | 0.342487 | 297/381 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable | `e490373c0b15` |
| 7 | 0.338356 | 293/381 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable | `0b0b51710dd3` |
| 8 | 0.338098 | 292/381 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared | `1cf9a47d94cf` |
| 9 | 0.335650 | 296/381 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable | `21802344efc1` |
| 10 | 0.335547 | 289/381 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable | `4d50029fa5db` |
| 11 | 0.335535 | 289/381 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable | `77cbe37ff1d7` |
| 12 | 0.335377 | 293/381 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable | `476e16cb4caf` |
| 13 | 0.334679 | 291/381 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable | `d5e523342ceb` |
| 14 | 0.334677 | 291/381 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable | `753ef2d13ae0` |
| 15 | 0.334454 | 292/381 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable | `f9c103ca318c` |
| 16 | 0.334277 | 292/381 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable | `8dc6626e918b` |
| 17 | 0.333336 | 289/381 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable | `f3a82624a43c` |
| 18 | 0.332785 | 295/381 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable | `4a367e790a53` |
| 19 | 0.332320 | 284/381 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable | `41e10d95e343` |
| 20 | 0.329409 | 290/381 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable | `82d735f12815` |
| 21 | 0.328925 | 290/381 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared | `b7f764fa5206` |
| 22 | 0.328351 | 294/381 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable | `8a24d74647b8` |
| 23 | 0.326589 | 287/381 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared | `646afe726f7b` |
| 24 | 0.325187 | 292/381 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable | `92378e4cdb4f` |

## Source-declared raw-like checkpoint-result subset

This is the primary cohort for the closest available checkpoint-oriented comparison. It contains no pipeline rows, but still lacks source threshold and checkpoint-SHA provenance.

| Rank | Dice | Hits | Type | Method family | Model / result | Checkpoint provenance | Result hash |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 1 | 0.345081 | 290/381 | raw-like | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | declared, not locally verifiable | `1bf09c0e9071` |
| 2 | 0.342487 | 297/381 | raw-like | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd` | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | declared, not locally verifiable | `e490373c0b15` |
| 3 | 0.338356 | 293/381 | raw-like | `word_global_v11_b3_low_lr_poly_full200_selected` | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | declared, not locally verifiable | `0b0b51710dd3` |
| 4 | 0.338098 | 292/381 | raw-like | `sampler_ablation_restricted_vs_inclusive_5k` | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | not declared | `1cf9a47d94cf` |
| 5 | 0.335650 | 296/381 | raw-like | `less_fan_conditioning_10k` | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | declared, not locally verifiable | `21802344efc1` |
| 6 | 0.335547 | 289/381 | raw-like | `category_continuous_router_from_s3_u9000` | `category_continuous_router_from_s3_u9000__best__4d50029f` | declared, not locally verifiable | `4d50029fa5db` |
| 7 | 0.335535 | 289/381 | raw-like | `controlled_v11_selected_full200` | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | declared, not locally verifiable | `77cbe37ff1d7` |
| 8 | 0.335377 | 293/381 | raw-like | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000` | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | declared, not locally verifiable | `476e16cb4caf` |
| 9 | 0.334679 | 291/381 | raw-like | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000` | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | declared, not locally verifiable | `d5e523342ceb` |
| 10 | 0.334677 | 291/381 | raw-like | `b3_bico_e345_v11` | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | declared, not locally verifiable | `753ef2d13ae0` |
| 11 | 0.334454 | 292/381 | raw-like | `nonattention_baseline_selection` | `nonattention_baseline_selection__full381_update10000__f9c103ca` | declared, not locally verifiable | `f9c103ca318c` |
| 12 | 0.334277 | 292/381 | raw-like | `b3_plus_less_v11` | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | declared, not locally verifiable | `8dc6626e918b` |
| 13 | 0.333336 | 289/381 | raw-like | `segmote_voxtell_m0_mote_v11_5k` | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | declared, not locally verifiable | `f3a82624a43c` |
| 14 | 0.332785 | 295/381 | raw-like | `word_global_v11_0to2000` | `word_global_v11_0to2000__b3_u4000__4a367e79` | declared, not locally verifiable | `4a367e790a53` |
| 15 | 0.332320 | 284/381 | raw-like | `c2_less_semantic_coadapt_v11` | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | declared, not locally verifiable | `41e10d95e343` |
| 16 | 0.329409 | 290/381 | raw-like | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000` | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | declared, not locally verifiable | `82d735f12815` |
| 17 | 0.328925 | 290/381 | raw-like | `attention_scale_screen_matched_comparison` | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | not declared | `b7f764fa5206` |
| 18 | 0.328351 | 294/381 | raw-like | `s3_full200_every1000` | `s3_full200_every1000__update_06000__8a24d746` | declared, not locally verifiable | `8a24d74647b8` |
| 19 | 0.326589 | 287/381 | raw-like | `attention_scale_screen_1_2_u500` | `attention_scale_screen_1_2_u500__update500__646afe72` | not declared | `646afe726f7b` |
| 20 | 0.325187 | 292/381 | raw-like | `qwen_early_coadaptation_from_v11` | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | declared, not locally verifiable | `92378e4cdb4f` |

## Provenance boundary

The collaborator bundle reports repository commit `1fd6445daab285f373b346912f0fba1e878dc0c5`. It was not inserted into SideExp003's canonical catalog.

The bundle contains a larger 309-result aggregate index, but only these 24 normalized folders provide the full per-finding and all-category evidence required here.
