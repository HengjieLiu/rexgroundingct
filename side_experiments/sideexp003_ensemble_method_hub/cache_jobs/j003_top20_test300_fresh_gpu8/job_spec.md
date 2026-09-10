# j003_top20_test300_fresh_gpu8 Fresh test300 Logit Export

Roster: `top20_val200_dice_20260907T211605Z` (`04aaff359ba3e1e6748a2e0851f64a828906c28b0ed642b673088064275ba96c`).
Job-spec SHA-256: `50ac85fb91a37a748be45e82e27d4ed09862645cdb06258b07f67fdd5bf8e912`.
Reuse policy: `fresh_only`; legacy caches are neither read nor used as fallback.

| Wave | GPU | Rank | Candidate | Cache key |
| ---: | ---: | ---: | --- | --- |
| 1 | 0 | 1 | `exp007_ddp_bs4_e050_a499ad1c` | `v3_6af052b3370aee14cede4058d20f34e491900cd444656537c7af8bb63cd48fb2` |
| 1 | 1 | 2 | `exp009_baseline_cont100_e100_50e631f1` | `v3_5bc31807a3490244cca848c4879205b4417ff7726d7ccdfa253b8f0dae4dcfff` |
| 1 | 2 | 3 | `exp017_ddp_bs4_e050_4f36d9bb` | `v3_bb856dd6c4b492f5c697ba3885f062463036a839c208d35852c0f7dd953f38f9` |
| 1 | 3 | 4 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` | `v3_144c527113b06c49a2155aafe894ce2f330e8648ab334fcb35e1461c25441a42` |
| 2 | 0 | 5 | `exp008_v1_sharedfusion_softguide_e100_61bcec41` | `v3_f5e9701bc48393fd1abd23e2e0fb6e2945f7a9e1d499ad336c923723b4582c62` |
| 2 | 1 | 6 | `exp008_v1_dualfusion_softguide_e100_9ba13980` | `v3_bf73e668e52d87718bc4cb27d2e25ef78bd5d7dd97d8d88086060c36516374b5` |
| 2 | 2 | 7 | `exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3` | `v3_e31c44bc50326740b11f456f226dff82b8f9542badd87ae419b32e2fca99f8fd` |
| 2 | 3 | 8 | `exp017_ddp_bs4_e050_a50a69dc` | `v3_f7e926387c575c40041910636a74f5cff32407b52aa4822abacb5c20759f3d77` |
| 3 | 0 | 9 | `exp007_ddp_bs4_e100_d56474a6` | `v3_123a38cf0730a7c85c57d87f7191730e620f81fcafb58673b6215da069989cd0` |
| 3 | 1 | 10 | `exp008_v3_dualfusion_softguide_joint_e100_f98374db` | `v3_5b830e791e0247e54d56f3f3c0f2144adb0ed6b50493ad5b823bdbcaf2369c28` |
| 3 | 2 | 11 | `exp017_ddp_bs4_e075_39aabab2` | `v3_afa32ab54ca0d2da86248f2c9c12796dc744ccbf9634c5b2d52b25d061150f61` |
| 3 | 3 | 12 | `exp008_v2_dualfusion_precision_e100_e67e2bc2` | `v3_efdc71c38be38c4a1c2527245612eab6bf402c15846754f05ac002640a8ddcf9` |
| 4 | 0 | 13 | `exp012_category_2a_replay50_e100_8dcdac04` | `v3_e6e3fc6d1c87d95fae37907daaf10886abcc9d0738ece0194a8b7962f0ab3bc4` |
| 4 | 1 | 14 | `exp007_ddp_bs4_e050_24e8ab93` | `v3_f10f62dde51290cd31de214a741639effc57ea164c5ea643b67bb2dc1ad2878b` |
| 4 | 2 | 15 | `exp017_ddp_bs4_e100_f11d238f` | `v3_25266637f39b1ba244cc93920422addd7f94ef679962c19d55d11de9432d5b86` |
| 4 | 3 | 16 | `exp008_v1_sharedfusion_softguide_e080_ed0cdd0a` | `v3_8c90f5dee96e8119bfde59afbe658a650939d091772f2d6d3b83c0151a3f690f` |
| 5 | 0 | 17 | `exp009_s3v3_logit_residual_half_quarter_e100_a0167aaa` | `v3_2c11a5cbffa07605e94ca7cebffe3c42f1b9a3b12485dec0b5ededa5c5c03355` |
| 5 | 1 | 18 | `exp007_ddp_bs4_e080_b5dc2c6f` | `v3_3f27a2f85201e6071d5de7173561aef7b7de2bfe99b760a4d9853c559f298a8f` |
| 5 | 2 | 19 | `exp007_ddp_bs4_e075_7f0b6a5f` | `v3_751760f4e8e0f8a4f98fb13c2ca87c7987318a10d35961bea3bed68a54503d75` |
| 5 | 3 | 20 | `exp007_ddp_bs4_e075_e3ec77b2` | `v3_54a86efa0d49fb30da04ef8f8271a70a6751297ea331d36693b9b3d18a928f07` |

The launcher requires 20,000 MiB free before loading each wave and 4,096 MiB free after the retained largest-case smoke. It automatically continues only after every member of the previous wave is `strict_passed`.
