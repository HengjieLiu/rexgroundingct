# Side Experiment 002 Hard-Mask Complementarity

This report quantifies why individual val200 Dice alone is insufficient for pruning the ensemble pool. It uses the existing threshold-0.5 masks for the same 200 cases and 381 findings; no inference is rerun.

> **Audit correction:** an earlier exploratory comparison aligned rows by their serialized JSON order. The evaluator writes cases in multiprocessing completion order, so that comparison overstated disagreement. Every value below is corrected by joining on `(case name, finding index)`.

> **Interpretation limit:** union hits and best-of-model oracle Dice use hindsight to choose a successful or higher-Dice model per finding. They are upper bounds that demonstrate complementary errors, not scores achievable by averaging.

## Definitions

- Dice correlation: Pearson correlation between aligned per-finding Dice vectors.
- Hit disagreement: one model has Dice at least `0.1` and the other does not.
- Model-only hits: findings hit by that model and missed by its pair.
- Union hits: findings hit by either model.
- Oracle Dice: mean of the larger per-finding Dice from the two models.

## Exp008 Epoch 80 Versus Epoch 100

| Pair | Dice correlation | Hit disagreements | Epoch-80-only hits | Second-only hits | Union hits | Oracle Dice | Oracle gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `exp008_shared_e080` vs `exp008_shared_e100` | 0.969813 | 16 / 381 | 6 | 10 | 293 | 0.351453 | +0.014863 |
| `exp008_dual_e080` vs `exp008_dual_e100` | 0.968635 | 8 / 381 | 5 | 3 | 291 | 0.351982 | +0.015933 |
| `exp008_precision_e080` vs `exp008_precision_e100` | 0.967150 | 16 / 381 | 6 | 10 | 293 | 0.348686 | +0.015401 |
| `exp008_joint_e080` vs `exp008_joint_e100` | 0.968615 | 10 / 381 | 5 | 5 | 293 | 0.350704 | +0.016184 |

Each epoch-80 snapshot uniquely hits 5-6 findings missed by its epoch-100 continuation. Correctly aligned correlations are roughly 0.97, so these snapshots are related but not identical.

## Exp011 Normalization Arms Versus Exp009 Baseline

| Pair | Dice correlation | Hit disagreements | Exp011-only hits | Second-only hits | Union hits | Oracle Dice | Oracle gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `exp011_native_zscore_e100` vs `exp009_baseline_e100` | 0.941121 | 18 / 381 | 7 | 11 | 295 | 0.356690 | +0.016851 |
| `exp011_clip_zscore_e100` vs `exp009_baseline_e100` | 0.933662 | 19 / 381 | 8 | 11 | 296 | 0.356461 | +0.016621 |
| `exp011_linear_hu_e100` vs `exp009_baseline_e100` | 0.912341 | 19 / 381 | 9 | 10 | 297 | 0.358028 | +0.018189 |

The three independently trained Exp011 arms recover 7-9 findings missed by the highest-Dice individual model.

## Exp011 Pairwise

| Pair | Dice correlation | Hit disagreements | First-only hits | Second-only hits | Union hits | Oracle Dice | Oracle gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `exp011_native_zscore_e100` vs `exp011_clip_zscore_e100` | 0.979327 | 11 / 381 | 5 | 6 | 290 | 0.338351 | +0.012257 |
| `exp011_native_zscore_e100` vs `exp011_linear_hu_e100` | 0.964269 | 13 / 381 | 5 | 8 | 292 | 0.339157 | +0.013063 |
| `exp011_clip_zscore_e100` vs `exp011_linear_hu_e100` | 0.980707 | 8 / 381 | 3 | 5 | 290 | 0.337047 | +0.011678 |

## Selected Cross-Family Pairs

| Pair | Dice correlation | Hit disagreements | First-only hits | Second-only hits | Union hits | Oracle Dice | Oracle gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `exp006_e6d4_e100` vs `exp008_precision_e080` | 0.887578 | 36 / 381 | 17 | 19 | 300 | 0.356820 | +0.026115 |
| `exp007_ddp_e050` vs `exp008_precision_e080` | 0.922693 | 27 / 381 | 16 | 11 | 299 | 0.360412 | +0.027145 |
| `exp009_baseline_e100` vs `exp008_joint_e080` | 0.960363 | 12 / 381 | 6 | 6 | 294 | 0.355758 | +0.015919 |

The Exp007 epoch-50 and Exp008 precision epoch-80 pair disagrees on 27 hit decisions and covers 299 of 381 findings.

## Greedy Best-of-Model Oracle

This deliberately optimistic sequence repeatedly adds the model that maximizes per-finding hindsight Dice.

| Step | Added model | Oracle Dice | Increment |
| ---: | --- | ---: | ---: |
| 1 | `exp009_baseline_e100` | 0.339839 | +0.339839 |
| 2 | `exp007_ddp_e075` | 0.363779 | +0.023940 |
| 3 | `exp008_shared_e080` | 0.375374 | +0.011595 |
| 4 | `exp007_ddp_e050` | 0.383406 | +0.008032 |
| 5 | `exp006_e6d4_e100` | 0.388250 | +0.004844 |
| 6 | `exp007_ddp_e100` | 0.391108 | +0.002858 |
| 7 | `exp008_joint_e080` | 0.393505 | +0.002397 |
| 8 | `exp009_s3v3_e100` | 0.395416 | +0.001911 |
| 9 | `exp011_linear_hu_e100` | 0.396745 | +0.001329 |
| 10 | `exp009_s3v1_e100` | 0.397940 | +0.001195 |
| 11 | `exp008_shared_e100` | 0.398869 | +0.000929 |
| 12 | `exp008_precision_e080` | 0.399661 | +0.000792 |
| 13 | `exp011_clip_zscore_e100` | 0.400387 | +0.000726 |
| 14 | `exp006_e5d4_e100` | 0.400992 | +0.000605 |
| 15 | `exp008_dual_e080` | 0.401584 | +0.000592 |
| 16 | `exp009_s3v2_e100` | 0.402029 | +0.000445 |
| 17 | `exp011_native_zscore_e100` | 0.402421 | +0.000392 |
| 18 | `exp008_precision_e100` | 0.402705 | +0.000284 |
| 19 | `exp008_dual_e100` | 0.402912 | +0.000207 |
| 20 | `exp008_joint_e100` | 0.403015 | +0.000103 |

The realizable gain must be measured from saved logits with cross-validated averaging, weights, and thresholds.
