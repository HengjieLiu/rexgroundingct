# Top-16 Per-Scope Seeded Caruana Diagnostic

Status: `diagnostic_only`; selection and evaluation use the same full val200 set and are optimistic.

## Scope `all`

Fixed K=4 seed: `exp007_ddp_bs4_e050_a499ad1c, exp009_baseline_cont100_e100_50e631f1, exp017_ddp_bs4_e050_4f36d9bb, exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801`.

| K | Added | Dice | Hits | Hit rate | Marginal Dice | Best so far | Multiplicities |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 4 | `fixed_seed` | 0.357504 | 296 | 0.776903 | +0.000000 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp009_baseline_cont100_e100_50e631f1": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1}` |
| 5 | `exp017_ddp_bs4_e050_a50a69dc` | 0.360958 | 298 | 0.782152 | +0.003454 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp009_baseline_cont100_e100_50e631f1": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 1}` |
| 6 | `exp017_ddp_bs4_e075_39aabab2` | 0.365395 | 298 | 0.782152 | +0.004438 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp009_baseline_cont100_e100_50e631f1": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 1, "exp017_ddp_bs4_e075_39aabab2": 1}` |
| 7 | `exp017_ddp_bs4_e050_a50a69dc` | 0.365726 | 298 | 0.782152 | +0.000331 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp009_baseline_cont100_e100_50e631f1": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 2, "exp017_ddp_bs4_e075_39aabab2": 1}` |
| 8 | `exp007_ddp_bs4_e050_a499ad1c` | 0.366540 | 299 | 0.784777 | +0.000814 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 2, "exp009_baseline_cont100_e100_50e631f1": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 2, "exp017_ddp_bs4_e075_39aabab2": 1}` |
## Scope `2a`

Fixed K=4 seed: `exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801, exp017_ddp_bs4_e050_4f36d9bb, exp017_ddp_bs4_e100_f11d238f, exp009_baseline_cont100_e100_50e631f1`.

| K | Added | Dice | Hits | Hit rate | Marginal Dice | Best so far | Multiplicities |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 4 | `fixed_seed` | 0.362824 | 58 | 0.840580 | +0.000000 | yes | `{"exp009_baseline_cont100_e100_50e631f1": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e100_f11d238f": 1}` |
| 5 | `exp017_ddp_bs4_e050_a50a69dc` | 0.365518 | 59 | 0.855072 | +0.002694 | yes | `{"exp009_baseline_cont100_e100_50e631f1": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 1, "exp017_ddp_bs4_e100_f11d238f": 1}` |
| 6 | `exp012_category_2a_replay50_e100_8dcdac04` | 0.369313 | 59 | 0.855072 | +0.003796 | yes | `{"exp009_baseline_cont100_e100_50e631f1": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp012_category_2a_replay50_e100_8dcdac04": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 1, "exp017_ddp_bs4_e100_f11d238f": 1}` |
| 7 | `exp017_ddp_bs4_e075_39aabab2` | 0.370389 | 59 | 0.855072 | +0.001075 | yes | `{"exp009_baseline_cont100_e100_50e631f1": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp012_category_2a_replay50_e100_8dcdac04": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 1, "exp017_ddp_bs4_e075_39aabab2": 1, "exp017_ddp_bs4_e100_f11d238f": 1}` |
| 8 | `exp008_v1_sharedfusion_softguide_e080_ed0cdd0a` | 0.371421 | 59 | 0.855072 | +0.001032 | yes | `{"exp008_v1_sharedfusion_softguide_e080_ed0cdd0a": 1, "exp009_baseline_cont100_e100_50e631f1": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp012_category_2a_replay50_e100_8dcdac04": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 1, "exp017_ddp_bs4_e075_39aabab2": 1, "exp017_ddp_bs4_e100_f11d238f": 1}` |
## Scope `2b`

Fixed K=4 seed: `exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3, exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801, exp017_ddp_bs4_e075_39aabab2, exp007_ddp_bs4_e050_a499ad1c`.

| K | Added | Dice | Hits | Hit rate | Marginal Dice | Best so far | Multiplicities |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 4 | `fixed_seed` | 0.370373 | 38 | 0.775510 | +0.000000 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e075_39aabab2": 1}` |
| 5 | `exp017_ddp_bs4_e050_a50a69dc` | 0.371852 | 38 | 0.775510 | +0.001479 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_a50a69dc": 1, "exp017_ddp_bs4_e075_39aabab2": 1}` |
| 6 | `exp017_ddp_bs4_e050_4f36d9bb` | 0.373111 | 38 | 0.775510 | +0.001259 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 1, "exp017_ddp_bs4_e075_39aabab2": 1}` |
| 7 | `exp017_ddp_bs4_e050_a50a69dc` | 0.373078 | 38 | 0.775510 | -0.000033 | no | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 2, "exp017_ddp_bs4_e075_39aabab2": 1}` |
| 8 | `exp007_ddp_bs4_e050_a499ad1c` | 0.373256 | 37 | 0.755102 | +0.000178 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 2, "exp009_s3v1_fixedrho_suppress_half_quarter_e100_263f27d3": 1, "exp009_s3v2_balanced_feature_half_quarter_e100_a3e93801": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 2, "exp017_ddp_bs4_e075_39aabab2": 1}` |
## Scope `2c`

Fixed K=4 seed: `exp007_ddp_bs4_e050_a499ad1c, exp008_v1_sharedfusion_softguide_e080_ed0cdd0a, exp012_category_2a_replay50_e100_8dcdac04, exp017_ddp_bs4_e050_a50a69dc`.

| K | Added | Dice | Hits | Hit rate | Marginal Dice | Best so far | Multiplicities |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 4 | `fixed_seed` | 0.420702 | 50 | 0.833333 | +0.000000 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp008_v1_sharedfusion_softguide_e080_ed0cdd0a": 1, "exp012_category_2a_replay50_e100_8dcdac04": 1, "exp017_ddp_bs4_e050_a50a69dc": 1}` |
| 5 | `exp017_ddp_bs4_e075_39aabab2` | 0.426851 | 51 | 0.850000 | +0.006149 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp008_v1_sharedfusion_softguide_e080_ed0cdd0a": 1, "exp012_category_2a_replay50_e100_8dcdac04": 1, "exp017_ddp_bs4_e050_a50a69dc": 1, "exp017_ddp_bs4_e075_39aabab2": 1}` |
| 6 | `exp017_ddp_bs4_e050_a50a69dc` | 0.426372 | 51 | 0.850000 | -0.000479 | no | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp008_v1_sharedfusion_softguide_e080_ed0cdd0a": 1, "exp012_category_2a_replay50_e100_8dcdac04": 1, "exp017_ddp_bs4_e050_a50a69dc": 2, "exp017_ddp_bs4_e075_39aabab2": 1}` |
| 7 | `exp007_ddp_bs4_e050_a499ad1c` | 0.427622 | 51 | 0.850000 | +0.001250 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 2, "exp008_v1_sharedfusion_softguide_e080_ed0cdd0a": 1, "exp012_category_2a_replay50_e100_8dcdac04": 1, "exp017_ddp_bs4_e050_a50a69dc": 2, "exp017_ddp_bs4_e075_39aabab2": 1}` |
| 8 | `exp008_v1_sharedfusion_softguide_e080_ed0cdd0a` | 0.428111 | 51 | 0.850000 | +0.000489 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 2, "exp008_v1_sharedfusion_softguide_e080_ed0cdd0a": 2, "exp012_category_2a_replay50_e100_8dcdac04": 1, "exp017_ddp_bs4_e050_a50a69dc": 2, "exp017_ddp_bs4_e075_39aabab2": 1}` |
## Scope `2d`

Fixed K=4 seed: `exp007_ddp_bs4_e050_a499ad1c, exp017_ddp_bs4_e050_4f36d9bb, exp009_baseline_cont100_e100_50e631f1, exp008_v1_sharedfusion_softguide_e100_61bcec41`.

| K | Added | Dice | Hits | Hit rate | Marginal Dice | Best so far | Multiplicities |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 4 | `fixed_seed` | 0.402057 | 115 | 0.871212 | +0.000000 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp008_v1_sharedfusion_softguide_e100_61bcec41": 1, "exp009_baseline_cont100_e100_50e631f1": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1}` |
| 5 | `exp017_ddp_bs4_e050_a50a69dc` | 0.408198 | 116 | 0.878788 | +0.006141 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp008_v1_sharedfusion_softguide_e100_61bcec41": 1, "exp009_baseline_cont100_e100_50e631f1": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 1}` |
| 6 | `exp017_ddp_bs4_e050_a50a69dc` | 0.414497 | 117 | 0.886364 | +0.006299 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp008_v1_sharedfusion_softguide_e100_61bcec41": 1, "exp009_baseline_cont100_e100_50e631f1": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 2}` |
| 7 | `exp017_ddp_bs4_e075_39aabab2` | 0.412719 | 118 | 0.893939 | -0.001778 | no | `{"exp007_ddp_bs4_e050_a499ad1c": 1, "exp008_v1_sharedfusion_softguide_e100_61bcec41": 1, "exp009_baseline_cont100_e100_50e631f1": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 2, "exp017_ddp_bs4_e075_39aabab2": 1}` |
| 8 | `exp007_ddp_bs4_e050_a499ad1c` | 0.415954 | 118 | 0.893939 | +0.003235 | yes | `{"exp007_ddp_bs4_e050_a499ad1c": 2, "exp008_v1_sharedfusion_softguide_e100_61bcec41": 1, "exp009_baseline_cont100_e100_50e631f1": 1, "exp017_ddp_bs4_e050_4f36d9bb": 1, "exp017_ddp_bs4_e050_a50a69dc": 2, "exp017_ddp_bs4_e075_39aabab2": 1}` |
No derived ensemble logits were materialized.
