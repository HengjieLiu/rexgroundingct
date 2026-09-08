# j001_top20_val200_fresh Fresh Val200 Logit Export

Roster: `top20_val200_dice_20260907T211605Z` (`04aaff359ba3e1e6748a2e0851f64a828906c28b0ed642b673088064275ba96c`).
Job-spec SHA-256: `b379536c17dcb8f4d45fb075f80adbb22e1db66799cdb99499dd0f7eef542cc1`.
Reuse policy: `fresh_only`; legacy caches are neither read nor used as fallback.

| Wave | GPU | Rank | Candidate | Cache key |
| ---: | ---: | ---: | --- | --- |
| 1 | 0 | 1 | `exp007_ddp_bs4_e050_a499ad1c` | `v2_d6dcddf705766db234bf631fa0925bfd69a5e18388767448b02a31b6484fcdbe` |
| 1 | 1 | 2 | `exp009_baseline_cont100_e100_50e631f1` | `v2_2cef6f9ce93dc1670cb31c78771674cb5c8645268d84baa797db7e11027a0051` |
| 1 | 2 | 3 | `exp017_ddp_bs4_e050_4f36d9bb` | `v2_d9eed4850d1a784ae877e393462193eaa10e9488e98838c4804c6d5094c291ca` |
| 1 | 3 | 4 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` | `v2_810f5b294b0b7d01d9ed818a7d30f254516d9f8003548e09c2392164398c1c5d` |
| 2 | 0 | 5 | `exp008_v1_sharedfusion_softguide_e100_61bcec41` | `v2_a4ba8f9594d3f159053c54f9f0dc557a3df32e963f0afc220de1aa6a0789ecfd` |
| 2 | 1 | 6 | `exp008_v1_dualfusion_softguide_e100_9ba13980` | `v2_48c3f564a6ebae6814165e3621e8c1d13b387d83991319f550eb581868c86a23` |
| 2 | 2 | 7 | `exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3` | `v2_fb2790b8cd31a9819dcfef1b9c68145864c539a64eec17c74c37a5775740e704` |
| 2 | 3 | 8 | `exp017_ddp_bs4_e050_a50a69dc` | `v2_d521b95e6b10eeb58c863744eaaa3f7c56867759e12c08e8191a8bfe2ef9fc69` |
| 3 | 0 | 9 | `exp007_ddp_bs4_e100_d56474a6` | `v2_985c56db1deb0d9b65bca73c028bbdca374ef190f8f131de415684fee18ccb16` |
| 3 | 1 | 10 | `exp008_v3_dualfusion_softguide_joint_e100_f98374db` | `v2_3a6dabb84f3e415e57b3f02a5958f80f840498d49dcaf90238f5c8d25cab6339` |
| 3 | 2 | 11 | `exp017_ddp_bs4_e075_39aabab2` | `v2_7d2798c3054c72c59a2b8c895a80caac57b2404f9cba37596ea4c555cca947b1` |
| 3 | 3 | 12 | `exp008_v2_dualfusion_precision_e100_e67e2bc2` | `v2_460bc47530b0c94b23df24a80e92d70ad8c883a0cf9d9e6b341c3a49993a0d5c` |
| 4 | 0 | 13 | `exp012_category_2a_replay50_e100_8dcdac04` | `v2_970aa27766b0d1aca142747f872cb382dc4006a266b28b8ee6c61c2d6674a88b` |
| 4 | 1 | 14 | `exp007_ddp_bs4_e050_24e8ab93` | `v2_091cf4237a0cc6fb40c47b6a741f64cc14a97c83fd87e1e32fde0eaf6dd7d88c` |
| 4 | 2 | 15 | `exp017_ddp_bs4_e100_f11d238f` | `v2_cd261a722f97fbf9dcd3c6d4c062e45746413439f29b1bd7c644dcf1b0be7bd0` |
| 4 | 3 | 16 | `exp008_v1_sharedfusion_softguide_e080_ed0cdd0a` | `v2_a444cfc47c7d46575ff278fd3c89e6afec9ddbd4f9fc9b730d7da0b43a7ffbfd` |
| 5 | 0 | 17 | `exp009_s3v3_logit_residual_half_quarter_e100_a0167aaa` | `v2_81a6233dfb516fa0cd32463e31a3f3a24ff9b1c00c2cc08b091e5f792c92bc4f` |
| 5 | 1 | 18 | `exp007_ddp_bs4_e080_b5dc2c6f` | `v2_d2aa331c85c0314d41bfbd6cfeb5ef7533e7ae257d009819bc0ac3a4bf63aee3` |
| 5 | 2 | 19 | `exp007_ddp_bs4_e075_7f0b6a5f` | `v2_3aff543e68410d50b0228c0f243b7a914422ec82b1f07f558be95712f433163d` |
| 5 | 3 | 20 | `exp007_ddp_bs4_e075_e3ec77b2` | `v2_31979324c66a5b776d13f791a88fd83fdb2ab06b3dfefdda8c2b3462e234f521` |

The launcher requires 20,000 MiB free before loading each wave and 4,096 MiB free after the retained largest-case smoke. It automatically continues only after every member of the previous wave is `strict_passed`.
