# j002_top20_val200_fresh_r09_r20 Fresh Val200 Logit Export

Roster: `top20_val200_dice_20260907T211605Z` (`04aaff359ba3e1e6748a2e0851f64a828906c28b0ed642b673088064275ba96c`).
Job-spec SHA-256: `9b7b967ad6bdbe51deaa4830b1a3776c8b0948e2305a0c2e9c373206f95b0020`.
Reuse policy: `fresh_only`; legacy caches are neither read nor used as fallback.

| Wave | GPU | Rank | Candidate | Cache key |
| ---: | ---: | ---: | --- | --- |
| 3 | 0 | 9 | `exp007_ddp_bs4_e100_d56474a6` | `v2_9ab1a17db9f41ab535d898c6ac5144e517e6b9c3c327f6ed8795c5a9f39beb05` |
| 3 | 1 | 10 | `exp008_v3_dualfusion_softguide_joint_e100_f98374db` | `v2_99fcd35504e45e2463d9a1c31f7f34b6f26a943484f5424c41c0aa25d5590e16` |
| 3 | 2 | 11 | `exp017_ddp_bs4_e075_39aabab2` | `v2_a7061b160d18dec1df8333cb25befac839fd988f637d6e50d9633bd7110f9d6d` |
| 3 | 3 | 12 | `exp008_v2_dualfusion_precision_e100_e67e2bc2` | `v2_339bbb581330c405e36224951245084041d42040ebe0630d6a8f62c7a98734ba` |
| 4 | 0 | 13 | `exp012_category_2a_replay50_e100_8dcdac04` | `v2_79b4bbd8d2f2d9348b53a0a931de077b6d29ed1616bc23ea8997771f3e839279` |
| 4 | 1 | 14 | `exp007_ddp_bs4_e050_24e8ab93` | `v2_a02086943b20265be0d4c7e965775951fd1404ebd0101eed482848b0e285ee6a` |
| 4 | 2 | 15 | `exp017_ddp_bs4_e100_f11d238f` | `v2_ece4981b3b05be6666c3f128e1b77736d95e1e0e62e76c5a0ace43ba3d92fd0b` |
| 4 | 3 | 16 | `exp008_v1_sharedfusion_softguide_e080_ed0cdd0a` | `v2_353f8e40457b2678e7fd8bbb0addeecbcad67ddc58dd0b53725877ce5d47e95d` |
| 5 | 0 | 17 | `exp009_s3v3_logit_residual_half_quarter_e100_a0167aaa` | `v2_7dc791e199e57afdba1d2cca3e9aff45d9a008a54b95b755562c4c60f9b3cc56` |
| 5 | 1 | 18 | `exp007_ddp_bs4_e080_b5dc2c6f` | `v2_279286fa73875ed0dddeb4f46298c8cc198836611a004d79ca4ef9bd6d714210` |
| 5 | 2 | 19 | `exp007_ddp_bs4_e075_7f0b6a5f` | `v2_e57eccfec0b3d17d51d8dd13bf731c4fc450ad63b0adff26bb5908774c3023ee` |
| 5 | 3 | 20 | `exp007_ddp_bs4_e075_e3ec77b2` | `v2_bd782a9a86a690ca50607a450b63d4bd93c467feda3f25c9efa4084c463886a9` |

The launcher requires 20,000 MiB free before loading each wave and 4,096 MiB free after the retained largest-case smoke. It automatically continues only after every member of the previous wave is `strict_passed`.

## Audited continuation

Parent job: `j001_top20_val200_fresh` (`b379536c17dcb8f4d45fb075f80adbb22e1db66799cdb99499dd0f7eef542cc1`).
Parent source bundle: `ecfd609aa7cf6382bfef94467a2d6105ab46725a92c94301126ea23fe9580faa`.
Continuation source bundle: `ce59a4d5021a6969d2fc63fe9b19d2fc8b1c1d3d81f4b26defbdec680d38bb28`.
Source-drift audit: `9ff8b54d858579bffef45b26e9b0e7e3f2a79136a8dfdd5c40072b66a31cea0b`.
Completed parent caches remain immutable; every continuation candidate uses a new cache key bound to the continuation source bundle.
