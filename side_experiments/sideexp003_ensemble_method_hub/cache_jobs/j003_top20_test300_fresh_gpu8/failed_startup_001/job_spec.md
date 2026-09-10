# j003_top20_test300_fresh_gpu8 Fresh test300 Logit Export

Roster: `top20_val200_dice_20260907T211605Z` (`04aaff359ba3e1e6748a2e0851f64a828906c28b0ed642b673088064275ba96c`).
Job-spec SHA-256: `981c4471643905188bc46b7cb47f1a1db3de6db81ac6914cf106b27237ece12a`.
Reuse policy: `fresh_only`; legacy caches are neither read nor used as fallback.

| Wave | GPU | Rank | Candidate | Cache key |
| ---: | ---: | ---: | --- | --- |
| 1 | 0 | 1 | `exp007_ddp_bs4_e050_a499ad1c` | `v3_0f5def821936593c377e0437898fa985d168be11ccf74d709006295a2afdb3fc` |
| 1 | 1 | 2 | `exp009_baseline_cont100_e100_50e631f1` | `v3_0622174043f2b436e4fadece6ef599d5784851d50ac94a4de4e05c1a45ca280c` |
| 1 | 2 | 3 | `exp017_ddp_bs4_e050_4f36d9bb` | `v3_265335d93708e4e6ee9b1b0050cea12c1c6021591607e5c82ac193909bed400c` |
| 1 | 3 | 4 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` | `v3_c476eaf5dd463f1be8d8063e5609fe6b9ead6f77715eb47498303a5d18cff5b4` |
| 2 | 0 | 5 | `exp008_v1_sharedfusion_softguide_e100_61bcec41` | `v3_2596f1c6c90db2954e109a436099f22a5f9a3074cc3a20e08e8b11ae99c2b2cc` |
| 2 | 1 | 6 | `exp008_v1_dualfusion_softguide_e100_9ba13980` | `v3_0510acd4050acd1fd7fc37eb65ac3dcd814591908968450a575c023d66f8aebb` |
| 2 | 2 | 7 | `exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3` | `v3_72a3de129ecfe73736fddfc052c785dae3bead9bf8deb468d0e5ee756af65349` |
| 2 | 3 | 8 | `exp017_ddp_bs4_e050_a50a69dc` | `v3_e47797c502bec722853252d58e024901c5138ef32b6213b23a5bbc4b10b87aa3` |
| 3 | 0 | 9 | `exp007_ddp_bs4_e100_d56474a6` | `v3_4b4ce12acfbc488063fb83126531bb2b81715610ebb6f5bf936ed8a82372f9d6` |
| 3 | 1 | 10 | `exp008_v3_dualfusion_softguide_joint_e100_f98374db` | `v3_0ab8b76842c1be65d9ddbbd8f13335289581748cd8ecfa42b4e43fda52202b3c` |
| 3 | 2 | 11 | `exp017_ddp_bs4_e075_39aabab2` | `v3_50ce0eff6538cbae36e175233bb4af2d163ff9b5beda1284ef1d12288195ef1a` |
| 3 | 3 | 12 | `exp008_v2_dualfusion_precision_e100_e67e2bc2` | `v3_093fef16bf64763a9db9502c9b350e65d6da323a56343a11900f7006e80ba989` |
| 4 | 0 | 13 | `exp012_category_2a_replay50_e100_8dcdac04` | `v3_991d008c89ba04f601733743e4c6dbb8b68aa4f98f012a67d418efc1e1205481` |
| 4 | 1 | 14 | `exp007_ddp_bs4_e050_24e8ab93` | `v3_9a5ff1b517033bac72f317fad5a4de6a10616b26fb10008b83dde5db2d239c12` |
| 4 | 2 | 15 | `exp017_ddp_bs4_e100_f11d238f` | `v3_8b9c1adcb8b7e072487e6b5f80d244ca8cb508a4bac7e2c191461861a0f64cd9` |
| 4 | 3 | 16 | `exp008_v1_sharedfusion_softguide_e080_ed0cdd0a` | `v3_c4fa0b91ba37c4db098b5cd34541a1bd0ba7b10e79c41565b79ab4916b5b80a5` |
| 5 | 0 | 17 | `exp009_s3v3_logit_residual_half_quarter_e100_a0167aaa` | `v3_a868ae39835cc52ee62585ab7f6933c1328d072f1565d6e7e639fa534bdcd59c` |
| 5 | 1 | 18 | `exp007_ddp_bs4_e080_b5dc2c6f` | `v3_a16ba37a3896c0289f98057b8343c4567d2de1fe04adcecc77f16dde298c645b` |
| 5 | 2 | 19 | `exp007_ddp_bs4_e075_7f0b6a5f` | `v3_dc2ee94fe9a656fe1f78d8ec53f1701921da65295c6a3f848e6057079dc3b43c` |
| 5 | 3 | 20 | `exp007_ddp_bs4_e075_e3ec77b2` | `v3_6402a7e2bba6440a9d63475fa6157f5c9a88ac3f7f19296e4914b067a9f724f8` |

The launcher requires 20,000 MiB free before loading each wave and 4,096 MiB free after the retained largest-case smoke. It automatically continues only after every member of the previous wave is `strict_passed`.
