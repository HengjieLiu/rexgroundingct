# Dice-first audit: accepting TP removal in exchange for FP removal

Status: **exploration_pending_user_review**. The user clarified that improved mean Dice is the primary goal and explicitly accepts TP removal when its FP-removal benefit increases Dice. This supersedes the proposed blanket 99% pooled / 95% per-finding retention requirements as default acceptance criteria. No threshold, checkpoint or training run is adopted by this audit.

All new analysis uses saved counts and scores on CPU; no new model inference or training was run.

## Mathematical tradeoff

For one finding, let T be base TP, F base FP, and G total GT foreground (T + base FN). If the editor removes a TP voxels and b FP voxels:

```text
D_base   = 2T / (T + F + G)
D_edited = 2(T-a) / (T + F + G - a - b)

D_edited > D_base  iff  T*b > a*(F+G), for T > 0.
```

Writing alpha=a/T and beta=b/F, the condition is `beta*F > alpha*(F+G)`. Removing 10% of TP and 50% of FP improves a nonzero-TP finding whenever `F > G/4`. The comparison is finding-specific: an FP-rich finding can tolerate more TP loss than an FP-poor finding. The equations omit the negligible existing 1e-6 Dice smoothing; numeric reports retain it. A zero-base-TP finding cannot gain actual overlap through deletion alone.

Equivalently, the deleted set must contain an FP proportion greater than `1 - D_base/2` to improve that finding's Dice. This is a condition on the error composition of the edits, not a model probability threshold.

## The user's 50% FP / 10% TP example

Applying those fractions to each finding's actual base counts gives:

| Subset | Base mean Dice | Hypothetical mean Dice | Change |
| --- | --- | --- | --- |
| A | 0.339416 | 0.382137 | +0.042721 |
| B | 0.334950 | 0.377053 | +0.042103 |
| full | 0.337215 | 0.379632 | +0.042416 |

The full improvement is +0.042416 Dice, or +4.24 percentage points. These are fractional-count illustrations, not achieved model results. Among the 67 findings with nonzero base TP, 53 improve and 14 worsen; two zero-TP findings remain without overlap, apart from negligible smoothing changes. Under uniform 50% FP deletion in every finding, the macro-Dice break-even TP-removal fraction is approximately 22.86%.

At the cohort-count level, base TP=382,990, FP=933,611 and GT=1,047,485. Pooled Dice is 0.324007, which is a different metric from the primary mean Dice 0.337215. The 50% FP / 10% TP pooled counts imply pooled Dice 0.370839. Pooled Dice improves at 50% FP removal while TP removal is below 23.56%, but this does not determine the primary mean-finding score.

## Identical pooled budgets can have very different mean Dice

All three constructions remove 50% of FP within every finding and exactly 10% of the total base TP. Only the allocation of TP deletion differs. These are examples, not optimization bounds or deployable GT-based policies.

| TP-deletion allocation | Full mean Dice | Findings losing all existing TP |
| --- | --- | --- |
| 10% of each finding's TP | 0.379632 | 0 |
| Spend the TP budget on smallest base-TP findings first | 0.271418 | 30 |
| Spend the TP budget on largest base-TP findings first | 0.403240 | 1 |

This is why overall TP retention alone should not govern a mean-finding-Dice objective. Keep per-finding Dice changes and lost/gained hits visible, while allowing beneficial TP removal.

An existing per-finding result also demonstrates that a TP-retention flag does
not necessarily indicate worse Dice. In run 4 at threshold 0.90,
`train_19180_a_2.nii.gz::1` retains only 84.62% of its base TP, yet its Dice
increases from 0.118925 to 0.135489 because its FP reduction compensates.

## What the actual epoch-20 models achieve at a 10% TP-loss budget

Each threshold below is derived retrospectively from the full cohort's labeled TP scores. Strict score > threshold and tie-aware discrete counts allow at most 10% pooled TP deletion. These thresholds are analysis points, not held-out selection.

| Run | Removal threshold | TP removed, full | FP removed, full | A Dice | B Dice | Full Dice | Full change | Hits gained / lost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run1_train | 0.625041 | 10.0000% | 22.4490% | 0.334465 | 0.336836 | 0.335633 | -0.001582 | 0 / 3 |
| run2_train_val_a | 0.463816 | 10.0000% | 20.4043% | 0.337738 | 0.338083 | 0.337908 | +0.000692 | 0 / 3 |
| run3_val_a | 0.500815 | 10.0000% | 25.9387% | 0.363104 | 0.340140 | 0.351788 | +0.014573 | 1 / 0 |
| run4_val_all | 0.421558 | 10.0000% | 26.2859% | 0.332680 | 0.353541 | 0.342959 | +0.005744 | 0 / 3 |

Run 3 achieves the clearest observed example of the requested tradeoff: approximately 25.94% FP deletion with 10% TP deletion, full Dice 0.351788, and one gained hit with no lost baseline hits. B Dice is 0.340140 versus B base 0.334950; the larger full gain includes fitted A cases. Run 4 fits both A and B.

The current models do not reach 50% FP deletion while retaining 90% of base TP. At approximately 50% FP deletion their results are:

| Run | Removal threshold | TP removed, full | FP removed, full | B Dice | Full Dice |
| --- | --- | --- | --- | --- | --- |
| run1_train | 0.457939 | 30.2893% | 49.9999% | 0.314495 | 0.311136 |
| run2_train_val_a | 0.324518 | 30.4089% | 49.9999% | 0.315722 | 0.313692 |
| run3_val_a | 0.341712 | 26.3258% | 49.9999% | 0.325952 | 0.343422 |
| run4_val_all | 0.256719 | 25.1181% | 49.9999% | 0.337908 | 0.328608 |

Allowing TP loss can improve Dice, but more deletion is not automatically better. For example, run 3 at 50% FP deletion still improves full Dice, yet it loses much of the gain obtained around its 10% TP-deletion operating point and its B Dice falls below baseline.

## Retrospective epoch-20 Dice maxima in a denser threshold grid

Editor scores were swept from 0.000 to 1.000 in 0.005 steps for every saved checkpoint. The following are per-run full-cohort grid maxima, in fixed run order. They use full-cohort labels to locate the maximum and must not be presented as independent generalization estimates. No checkpoint or threshold is promoted.

| Run | Removal threshold | A Dice | B Dice | Full Dice | TP removed | FP removed |
| --- | --- | --- | --- | --- | --- | --- |
| run1_train | 0.810 | 0.343714 | 0.340055 | 0.341911 | 0.7911% | 3.5724% |
| run2_train_val_a | 0.600 | 0.344922 | 0.343014 | 0.343982 | 1.6128% | 5.2323% |
| run3_val_a | 0.495 | 0.363173 | 0.340094 | 0.351801 | 10.4191% | 26.6576% |
| run4_val_all | 0.570 | 0.336497 | 0.354609 | 0.345422 | 4.1231% | 13.1918% |

The exact JSON also contains A-Dice-maximizing grid points applied unchanged to B. B has been inspected repeatedly during development and is gradient-held-out only in runs 1–3; A is fitted in runs 2–4.

## Implication for the next loss/pipeline proposal

- Use mean per-finding Dice improvement as the primary selection target, with A-derived threshold procedures and B reporting. TP-retention percentages, severe per-finding Dice losses and hits gained/lost remain diagnostics or explicit soft regularizers, rather than automatic rejection at 99%/95% retention.
- Compare a differentiable Dice objective on the retained mask, anchored by an auxiliary deletion BCE, against the current BCE-only objective. With blended removal probability p, retained foreground is `q = base_mask * (1-p)`. Compute `1 - (2*sum(q*GT)+eps)/(sum(q)+sum(GT)+eps)` per finding and average across findings. A TP deletion can be worthwhile when the accompanying FP deletion sufficiently improves this objective.
- Patch training needs a deliberate estimate of whole-finding Dice or accumulation across tiles. Ordinary Dice on an arbitrary sampled patch is not identical to the final mean per-finding metric. Keep true instance IDs and per-finding accounting so large voxel counts do not silently turn the objective into pooled Dice.
- Reconsider the previously suggested extra TP penalty/top-TP loss as ablations, not the leading criterion. Excessive preservation pressure can prevent Dice-improving edits. Ranking, annotation review, full-volume calibration and context-consistency checks remain relevant.
- Better score discrimination is still needed to approach the 50% FP / 10% TP target. Relaxing the decision threshold alone cannot produce a tradeoff outside a model's measured curve.

The theoretical deletion-only oracle that removes all annotated FP while retaining all base TP has mean Dice 0.590656 on this cohort. This is a label-informed ceiling, not a model forecast.

## Artifacts and verification

![Uniform-count scenario and actual model tradeoffs](../runtime/deletion_four_arm_20ep/reports/dice_tradeoff_audit/dice_vs_tp_fp_tradeoff.png)

- [Full analysis JSON](../runtime/deletion_four_arm_20ep/reports/dice_tradeoff_audit/results.json)
- [All dense threshold metrics, CSV](../runtime/deletion_four_arm_20ep/reports/dice_tradeoff_audit/dense_threshold_metrics.csv)
- [Figure PDF](../runtime/deletion_four_arm_20ep/reports/dice_tradeoff_audit/dice_vs_tp_fp_tradeoff.pdf)
- [Executed CPU analysis script](../runtime/deletion_four_arm_20ep/reports/dice_tradeoff_audit/generate_audit.py)

All 1,104 saved per-finding score-file hashes passed. The dense sweep reproduces the original six-threshold A/B/full Dice and TP/FP rates for all sixteen evaluations. The CSV contains 9,948 rows: 9,648 editor aggregate rows plus 300 dense frozen-base-control rows (base probability 0.50–0.995). Configurations, checkpoints, caches and historical results remain unchanged.
