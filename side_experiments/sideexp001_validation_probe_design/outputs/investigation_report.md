# SideExp001 Category-Aware Validation Probe Investigation

## Outcome

- Recommendation status: **accepted**
- Selected case count: `80`
- Selected finding count: `195`
- Census cutoff: `17`
- Frozen pre-holdout selection SHA256: `f01d79f935be2e9ed6a92ee4e1dc8117594608069ac850c28f93c60b7510e1f6`

The selector used Exp006/007/008 only. Exp009/011 were loaded after
the case set was frozen and were not used for retuning.

## Legacy Val20 Finding

- Reconstruction check: `20` of `22` paired evaluations met the absolute Dice tolerance `0.0001` with identical hit counts.
- Maximum absolute Dice difference: `0.00011715`.
- Mean Dice bias versus val200: `+0.0519`.
- Dice category-composition component: `-0.0200`.
- Dice within-category case-selection component: `+0.0719`.
- Mean hit-rate bias versus val200: `-0.0336`.

## Validation Versus Full Test Distribution

The full released test metadata contains 300 cases and 582 findings.
The public live leaderboard uses an undisclosed 150-case subset with
302 findings; its category counts are secondary reference only.

| Category | Val n (%) | Full test n (%) | Test-val diff | Ratio | Flag | Public n (%) | Public-test diff |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1a | 3 (0.79%) | 6 (1.03%) | +0.244 pp | 1.31 |  | 5 (1.66%) | +0.625 pp |
| 1b | 11 (2.89%) | 11 (1.89%) | -0.997 pp | 0.65 | VAL-TEST SHIFT | 5 (1.66%) | -0.234 pp |
| 1c | 17 (4.46%) | 27 (4.64%) | +0.177 pp | 1.04 |  | 16 (5.30%) | +0.659 pp |
| 1d | 6 (1.57%) | 9 (1.55%) | -0.028 pp | 0.98 |  | 6 (1.99%) | +0.440 pp |
| 1e | 11 (2.89%) | 16 (2.75%) | -0.138 pp | 0.95 |  | 10 (3.31%) | +0.562 pp |
| 1f | 4 (1.05%) | 4 (0.69%) | -0.363 pp | 0.65 | LOW-COUNT SHIFT WARNING | 2 (0.66%) | -0.025 pp |
| 2a | 69 (18.11%) | 113 (19.42%) | +1.306 pp | 1.07 |  | 61 (20.20%) | +0.783 pp |
| 2b | 49 (12.86%) | 89 (15.29%) | +2.431 pp | 1.19 | VAL-TEST SHIFT | 46 (15.23%) | -0.060 pp |
| 2c | 60 (15.75%) | 87 (14.95%) | -0.800 pp | 0.95 |  | 39 (12.91%) | -2.035 pp |
| 2d | 132 (34.65%) | 190 (32.65%) | -2.000 pp | 0.94 |  | 97 (32.12%) | -0.527 pp |
| 2e | 11 (2.89%) | 27 (4.64%) | +1.752 pp | 1.61 | VAL-TEST SHIFT | 13 (4.30%) | -0.335 pp |
| 2f | 0 (0.00%) | 0 (0.00%) | +0.000 pp | n/a |  | 0 (0.00%) | +0.000 pp |
| 2g | 1 (0.26%) | 1 (0.17%) | -0.091 pp | 0.65 | LOW-COUNT SHIFT WARNING | 1 (0.33%) | +0.159 pp |
| 2h | 7 (1.84%) | 2 (0.34%) | -1.494 pp | 0.19 | LOW-COUNT SHIFT WARNING | 1 (0.33%) | -0.013 pp |

- Validation/full-test total variation: `0.0591`.
- Validation/full-test Jensen-Shannon divergence: `0.0080` bits.
- Full-test/public-subset total variation: `0.0323`.

## Accuracy-Cost Frontier

| Cases | Census cutoff | Mandatory cases | Findings | CV error | CV macro Dice | CV macro hit | Frontier | Knee | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 20 | 4 | 8 | 55 | 0.1287 | 0.0517 | 0.1097 | yes |  |  |
| 30 | 7 | 19 | 86 | 0.0733 | 0.0272 | 0.0554 | yes |  |  |
| 40 | 7 | 19 | 95 | 0.0645 | 0.0256 | 0.0487 | yes |  |  |
| 50 | 11 | 43 | 139 | 0.0225 | 0.0085 | 0.0145 | yes | yes |  |
| 70 | 17 | 55 | 181 | 0.0200 | 0.0063 | 0.0119 | yes |  |  |
| 80 | 17 | 55 | 195 | 0.0159 | 0.0036 | 0.0101 | yes |  | yes |
| 100 | 17 | 55 | 234 | 0.0104 | 0.0031 | 0.0061 | yes |  |  |
| 120 | 17 | 55 | 259 | 0.0098 | 0.0030 | 0.0057 | yes |  |  |

## Frozen Holdout Audit

- Holdout acceptance: `passed`.
- Macro category Dice RMSE: `0.0038`.
- 90th-percentile category Dice RMSE: `0.0130`.
- Worst-category Dice RMSE: `0.0212`.
- Macro category hit-rate RMSE: `0.0049`.
- 90th-percentile category hit-rate RMSE: `0.0192`.
- Worst-category hit-rate RMSE: `0.0240`.

## Interpretation Limits

- Test-reweighted metrics are extrapolations from validation outcomes, not measured test performance.
- The live leaderboard subset membership is undisclosed; its ratios cannot identify which test cases are evaluated.
- Category `2f` has no validation or test findings and cannot be represented in a validation probe.
