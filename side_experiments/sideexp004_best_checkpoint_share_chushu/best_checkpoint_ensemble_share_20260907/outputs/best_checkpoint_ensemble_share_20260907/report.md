# Best Full200 checkpoints and category-ensemble audit

Generated from completed results containing exactly 381 per-finding rows. Fast30/subsets, incomplete runs, prediction volumes, and checkpoint binaries are excluded from this share bundle.

## Current leaders

| Rank | Model/result | Type | Dice | Hits | 2a | 2b | 2c | 2d |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `lobe_roi_segmentation_loss_continue_5000_to10000__semantic_v2_summary_tables__5b09b79b` | post-process/pipeline | 0.353632 | 298/381 | 0.325866 | 0.376728 | 0.423390 | 0.402569 |
| 2 | `prompt_location_u10k__semantic_A_e10_s10_fw10` | post-process/pipeline | 0.347978 | 295/381 | 0.339494 | 0.344745 | 0.387986 | 0.407002 |
| 3 | `nonattention_baseline_selection__semantic_v2_summary_tables__a1b37791` | post-process/pipeline | 0.347393 | 295/381 | 0.343054 | 0.344608 | 0.388218 | 0.404960 |
| 4 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__de0cd397` | post-process/pipeline | 0.345792 | 290/381 | 0.334383 | 0.361318 | 0.377608 | 0.402664 |
| 5 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | raw-like | 0.345081 | 290/381 | 0.333885 | 0.361996 | 0.377825 | 0.400448 |
| 6 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__update_10000__66d820f1` | post-process/pipeline | 0.345062 | 290/381 | 0.333887 | 0.361990 | 0.377812 | 0.400405 |
| 7 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_s3_1over1_allcat_update10000__0baf1191` | raw-like | 0.343143 | 288/381 | 0.332074 | 0.359962 | 0.380966 | 0.396697 |
| 8 | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | raw-like | 0.342487 | 297/381 | 0.342213 | 0.359895 | 0.378444 | 0.390200 |
| 9 | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update6000__87fb07eb` | raw-like | 0.340492 | 294/381 | 0.340409 | 0.362704 | 0.375581 | 0.390787 |
| 10 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_s3_1over1_allcat_update4000__65f82531` | raw-like | 0.338484 | 286/381 | 0.311248 | 0.348529 | 0.407392 | 0.385649 |
| 11 | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | raw-like | 0.338356 | 293/381 | 0.335863 | 0.353945 | 0.380582 | 0.386684 |
| 12 | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | raw-like | 0.338098 | 292/381 | 0.322858 | 0.342946 | 0.396684 | 0.391210 |
| 13 | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__6366b010` | raw-like | 0.338061 | 292/381 | 0.323025 | 0.343222 | 0.396118 | 0.391196 |
| 14 | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update8000__79e72982` | raw-like | 0.337161 | 293/381 | 0.333761 | 0.354106 | 0.384546 | 0.383157 |
| 15 | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update6600__8ea52dfc` | raw-like | 0.336338 | 293/381 | 0.335088 | 0.351552 | 0.382574 | 0.381561 |
| 16 | `s3_1over1_allcat_v11_e1e-5_d1e-4_s3e1e-4_1gpu_bs1_acc4_10kupd__update_09000__5d4fb46a` | post-process/pipeline | 0.336326 | 289/381 | 0.336599 | 0.357118 | 0.386180 | 0.371700 |
| 17 | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update6000__b93bd690` | raw-like | 0.336315 | 293/381 | 0.334760 | 0.354234 | 0.386369 | 0.379432 |
| 18 | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update6800__280ed488` | raw-like | 0.335910 | 295/381 | 0.335261 | 0.352096 | 0.381785 | 0.380309 |
| 19 | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update5000__e2494247` | raw-like | 0.335834 | 298/381 | 0.326306 | 0.351368 | 0.397793 | 0.373303 |
| 20 | `less_fan_conditioning_10k__c0_b0_update6800__21802344` | raw-like | 0.335650 | 296/381 | 0.327846 | 0.360254 | 0.385784 | 0.378398 |

## Focus-category specialists

### Category 2a (N=69)

| Rank | Model | Global Dice | Category Dice | Hits |
|---:|---|---:|---:|---:|
| 1 | `c2_less_semantic_coadapt_v11__pathology_continue_update_7600__41e10d95` | 0.332320 | 0.343231 | 284/381 |
| 2 | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | 0.342487 | 0.342213 | 297/381 |
| 3 | `word_global_v11_0to2000__b3_u4000__4a367e79` | 0.332785 | 0.339020 | 295/381 |
| 4 | `b3_bico_e345_v11__b3_bico_e345_update5000__753ef2d1` | 0.334677 | 0.337699 | 291/381 |
| 5 | `b3_candidate_router_ablation_v11__functional_k1_u4000__b822578d` | 0.321228 | 0.336838 | 291/381 |
| 6 | `s3_low_lr_full200_every1000__update_08000__be24d0fc` | 0.333368 | 0.336728 | 295/381 |
| 7 | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | 0.338356 | 0.335863 | 293/381 |
| 8 | `controlled_v11_selected_full200__s3_update9000__77cbe37f` | 0.335535 | 0.335831 | 289/381 |

### Category 2b (N=49)

| Rank | Model | Global Dice | Category Dice | Hits |
|---:|---|---:|---:|---:|
| 1 | `s3_full200_every1000__update_06000__8a24d746` | 0.328351 | 0.367622 | 294/381 |
| 2 | `qwen_lora_top6_global_from_v11_low_lr_e1e-6_d1e-5_from_u5000__update_07400__82d735f1` | 0.329409 | 0.367385 | 290/381 |
| 3 | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | 0.328925 | 0.365346 | 290/381 |
| 4 | `qwen_early_coadaptation_from_v11__C1_update_5000__92378e4c` | 0.325187 | 0.362513 | 292/381 |
| 5 | `qwen_complementarity_full200__q0__2898dd8b` | 0.328515 | 0.362125 | 288/381 |
| 6 | `s3_low_lr_full200_every1000__update_08000__be24d0fc` | 0.333368 | 0.362091 | 295/381 |
| 7 | `c_global_lambda01_v11__b0_update5000__e620bf6e` | 0.328407 | 0.362076 | 288/381 |
| 8 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | 0.345081 | 0.361996 | 290/381 |

### Category 2c (N=60)

| Rank | Model | Global Dice | Category Dice | Hits |
|---:|---|---:|---:|---:|
| 1 | `b3_plus_less_v11__b3_plus_less_update5000__8dc6626e` | 0.334277 | 0.403797 | 292/381 |
| 2 | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | 0.338098 | 0.396684 | 292/381 |
| 3 | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | 0.333336 | 0.393185 | 289/381 |
| 4 | `attention_scale_screen_1_2_u500__update500__646afe72` | 0.326589 | 0.393130 | 287/381 |
| 5 | `attention_scale_screen_control_u500__update500__bb5ba37f` | 0.328343 | 0.391470 | 287/381 |
| 6 | `a2_fan_preserve_semantic_grounding_e3b_v11_5000__update_02200__196b8417` | 0.315713 | 0.390550 | 289/381 |
| 7 | `attention_scale_screen_matched_comparison__base_update6000__b7f764fa` | 0.328925 | 0.390301 | 290/381 |
| 8 | `s3_full200_every1000__update_06000__8a24d746` | 0.328351 | 0.390162 | 294/381 |

### Category 2d (N=132)

| Rank | Model | Global Dice | Category Dice | Hits |
|---:|---|---:|---:|---:|
| 1 | `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` | 0.345081 | 0.400448 | 290/381 |
| 2 | `sampler_ablation_restricted_vs_inclusive_5k__update_04400__1cf9a47d` | 0.338098 | 0.391210 | 292/381 |
| 3 | `resampling_fixed_d0p75_h0p7_w0p7_v1_v11_controlled_10kupd__resample_d075_baseline_update10000__e490373c` | 0.342487 | 0.390200 | 297/381 |
| 4 | `word_global_v11_b3_low_lr_poly_full200_selected__b3_low_lr_update5000__0b0b5171` | 0.338356 | 0.386684 | 293/381 |
| 5 | `prompt_location_alignment_loss_v1_noattention_from_update8000_to10000__full381_update10000__476e16cb` | 0.335377 | 0.381454 | 293/381 |
| 6 | `segmote_voxtell_m0_mote_v11_5k__segmote_m0_update3400_thr0p40_full200__f3a82624` | 0.333336 | 0.381331 | 289/381 |
| 7 | `a2_fan_preserve_semantic_grounding_v11_continue_5000_to10000__update_09500__d5e52334` | 0.334679 | 0.380652 | 291/381 |
| 8 | `nonattention_baseline_selection__full381_update10000__f9c103ca` | 0.334454 | 0.379950 | 292/381 |

## Recommended ensemble strategy

For checkpoint-only deployment, use **hard category routing**, with `resampling_fixed_d1p0_h0p7_w0p7_v1_v11_controlled_10kupd__resample_baseline_update10000__1bf09c0e` as the default and switch only 2a/2b/2c/2d according to `focus_category_router_raw_checkpoints.json`. Its same-Full200 validation estimate is **0.351587 Dice, 295/381 hits**.

If existing deterministic semantic post-processing is allowed, `focus_category_router_with_postprocess.json` gives the broader pipeline oracle: **0.358313 Dice, 299/381 hits**.

This is an oracle estimate because the same Full200 set selected the specialists and measured the result. For a defensible ensemble, freeze the routing table using a development split (or leave-one-case-fold-out selection), then evaluate once on untouched test data. Per-finding Dice alone cannot justify voxel-probability averaging; that would require aligned raw probability maps and a separately calibrated fusion rule.

## Share contents

- `all_full200_results.json/tsv`: every discovered unique complete Full200 result.
- `best_raw_checkpoint_per_method.json/tsv`: best raw-like result per top-level method family.
- `selected_models.json/tsv`: global leaders plus focus-category specialists included below `models/`.
- `models/<id>/result.json`, `per_category.json`, `per_finding.json`: normalized shareable result records.
- `focus_category_router_raw_checkpoints.json`: raw checkpoint-only mapping and validation estimate.
- `focus_category_router_with_postprocess.json`: broader pipeline mapping and validation estimate.

Repository commit: `1fd6445daab285f373b346912f0fba1e878dc0c5`.
