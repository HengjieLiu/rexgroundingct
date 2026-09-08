# a001_top16_after_wave4_fullval

Wait for fresh ranks 1–16, then run the top-16 uniform and five-scope seeded Caruana diagnostics.

Poll interval: `600 seconds`; workers: `5`.

| Scope | Fixed seed ranks | Fixed seed candidates |
| --- | --- | --- |
| all | 1, 2, 3, 4 | `exp007_ddp_bs4_e050_a499ad1c, exp009_baseline_cont100_e100_50e631f1, exp017_ddp_bs4_e050_4f36d9bb, exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` |
| 2a | 4, 3, 15, 2 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801, exp017_ddp_bs4_e050_4f36d9bb, exp017_ddp_bs4_e100_f11d238f, exp009_baseline_cont100_e100_50e631f1` |
| 2b | 7, 4, 11, 1 | `exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3, exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801, exp017_ddp_bs4_e075_39aabab2, exp007_ddp_bs4_e050_a499ad1c` |
| 2c | 1, 16, 13, 8 | `exp007_ddp_bs4_e050_a499ad1c, exp008_v1_sharedfusion_softguide_e080_ed0cdd0a, exp012_category_2a_replay50_e100_8dcdac04, exp017_ddp_bs4_e050_a50a69dc` |
| 2d | 1, 3, 2, 5 | `exp007_ddp_bs4_e050_a499ad1c, exp017_ddp_bs4_e050_4f36d9bb, exp009_baseline_cont100_e100_50e631f1, exp008_v1_sharedfusion_softguide_e100_61bcec41` |

This is an optimistic full-val diagnostic and writes no derived logits.
