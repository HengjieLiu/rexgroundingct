# Top-4 Caruana Replacement Diagnostic

Status: `diagnostic_only`; selection and evaluation both use full val200.

Wall time: `101.03 min`.
No derived logits were materialized.

| K | Added | Basket | Dice | Hits | Marginal Dice |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `exp007_ddp_bs4_e050_a499ad1c` | `exp007_ddp_bs4_e050_a499ad1c` | 0.345868 | 295 | +0.000000 |
| 2 | `exp017_ddp_bs4_e050_4f36d9bb` | `exp007_ddp_bs4_e050_a499ad1c,exp017_ddp_bs4_e050_4f36d9bb` | 0.360140 | 295 | +0.014272 |
| 3 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` | `exp007_ddp_bs4_e050_a499ad1c,exp017_ddp_bs4_e050_4f36d9bb,exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` | 0.358606 | 295 | -0.001533 |
| 4 | `exp017_ddp_bs4_e050_4f36d9bb` | `exp007_ddp_bs4_e050_a499ad1c,exp017_ddp_bs4_e050_4f36d9bb,exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801,exp017_ddp_bs4_e050_4f36d9bb` | 0.358433 | 292 | -0.000173 |

Final weights: `{"exp007_ddp_bs4_e050_a499ad1c": 0.25, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 0.25, "exp017_ddp_bs4_e050_4f36d9bb": 0.5}`.

The full-val Dice peak is K=2 (0.360140), not the forced final K=4 basket.

## All addition trials

| K | Candidate | Dice | Hits | Selected |
| ---: | --- | ---: | ---: | :---: |
| 2 | `exp007_ddp_bs4_e050_a499ad1c` | 0.345868 | 295 | |
| 2 | `exp009_baseline_cont100_e100_50e631f1` | 0.354983 | 297 | |
| 2 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` | 0.351182 | 300 | |
| 2 | `exp017_ddp_bs4_e050_4f36d9bb` | 0.360140 | 295 | yes |
| 3 | `exp007_ddp_bs4_e050_a499ad1c` | 0.350438 | 295 | |
| 3 | `exp009_baseline_cont100_e100_50e631f1` | 0.358145 | 293 | |
| 3 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` | 0.358606 | 295 | yes |
| 3 | `exp017_ddp_bs4_e050_4f36d9bb` | 0.344965 | 288 | |
| 4 | `exp007_ddp_bs4_e050_a499ad1c` | 0.357756 | 299 | |
| 4 | `exp009_baseline_cont100_e100_50e631f1` | 0.357506 | 296 | |
| 4 | `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801` | 0.353478 | 296 | |
| 4 | `exp017_ddp_bs4_e050_4f36d9bb` | 0.358433 | 292 | yes |

## Timing

| Pass | Wall min | Source/sigmoid min | Generation min | Evaluation min | GT read min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Add K=2 | 48.494 | 34.681 | 2.702 | 0.847 | 0.741 |
| Add K=3 | 28.773 | 15.143 | 3.032 | 0.737 | 0.777 |
| Add K=4 | 23.765 | 9.709 | 3.629 | 0.737 | 0.798 |
| Total | 101.032 | 59.533 | 9.363 | 2.321 | 2.316 |

Total active CPU time was 73.941 minutes. Logical source bytes were
889,036,210,176 (827.98 GiB); process read bytes were 353,386,721,280
(329.12 GiB), reflecting substantial OS page-cache reuse in later rounds.

## Final K=4 official-category metrics

| Category | Findings | Dice | Hits | Hit rate |
| --- | ---: | ---: | ---: | ---: |
| 1a | 3 | 0.117397 | 1 | 0.333333 |
| 1b | 11 | 0.165291 | 4 | 0.363636 |
| 1c | 17 | 0.191066 | 8 | 0.470588 |
| 1d | 6 | 0.127822 | 1 | 0.166667 |
| 1e | 11 | 0.179043 | 6 | 0.545455 |
| 1f | 4 | 0.189031 | 2 | 0.500000 |
| 2a | 69 | 0.362893 | 57 | 0.826087 |
| 2b | 49 | 0.366035 | 36 | 0.734694 |
| 2c | 60 | 0.400574 | 50 | 0.833333 |
| 2d | 132 | 0.409025 | 116 | 0.878788 |
| 2e | 11 | 0.429437 | 8 | 0.727273 |
| 2f | 0 | — | 0 | — |
| 2g | 1 | 0.027590 | 0 | 0.000000 |
| 2h | 7 | 0.271363 | 3 | 0.428571 |
