# Completed A-only four-loss comparison

Status: **pending_user_review**. Finished 2026-09-11 02:14:41 UTC
(September10 19:14 Pacific), after116.2 minutes including preparation.
All four arms completed2000 updates, all16 evaluations and all16 dense threshold
analyses. The completion audit passed8000 updates, checkpoint/schedule identity,
and exact reproduction of historical A-only BCE weights and metrics at all
four milestones. No threshold/checkpoint is selected; no continuation ran.

All four models fit the same35 A findings with identical pristine initialization
and patch schedules. B's34 findings are excluded from gradients but have already
informed development. Scores are mean per-finding Dice. Full-cohort results mix
fitting and held-out development exposure and are secondary diagnostics.

## Final epoch20 results

| Loss | A Dice, threshold0.50 | B Dice, threshold0.50 | A Dice, threshold0.90 | B Dice, threshold0.90 |
| --- | --- | --- | --- | --- |
| Frozen base | 0.339416 | 0.334950 | 0.339416 | 0.334950 |
| F+2K: current BCE | 0.363095 | 0.340140 | 0.340498 | 0.335697 |
| F+K: equal BCE group weights | 0.358500 | 0.323392 | 0.343571 | 0.338488 |
| D+0.25(F+2K)/3 | 0.369552 | 0.344549 | 0.344767 | 0.337133 |
| D+0.25(F+K)/2 | 0.368905 | 0.338723 | 0.345947 | 0.337392 |

F and K are separate FP-deletion and TP-preservation BCE means. D is the
finding-aware single-patch Dice surrogate, with the outside prediction held at
base. The Dice recipes also rescale BCE; the comparison does not isolate adding
Dice while holding the BCE coefficient fixed.

At threshold0.50 the Dice+2×TP auxiliary recipe improves A by0.030136 and B
by0.009599 over their cached baselines. Against the original BCE recipe at the
same threshold, the increments are0.006457 on A and0.004410 on B. This is a
modest held-out development improvement, with a larger gain on fitted A.
One seed and34 B findings do not establish robustness.

## B deletion tradeoff at threshold0.50

| Loss | B Dice change vs base | B FP removed | B TP removed | B findings improved / worsened | B hits gained / lost |
| --- | --- | --- | --- | --- | --- |
| F+2K | +0.005190 | 21.24% | 11.14% | 22 / 12 | 1 / 0 |
| F+K | -0.011558 | 38.63% | 25.25% | 15 / 19 | 0 / 0 |
| D+0.25(F+2K)/3 | +0.009599 | 23.01% | 11.07% | 21 / 13 | 0 / 0 |
| D+0.25(F+K)/2 | +0.003773 | 31.95% | 16.52% | 20 / 14 | 1 / 0 |

FP/TP removal percentages pool voxels within B; Dice averages findings. The
Dice+2×TP recipe removes more B FP than current BCE at approximately the same
pooled TP loss. Some findings still worsen. Original BCE gains one B hit here,
while Dice+2×TP preserves the baseline hit set. Retention flags are diagnostics,
not acceptance gates.

Equal BCE weights alone induce excessive deletion at0.50 despite A improving.
At0.90 its B Dice improves to0.338488, so the0.50 result does not establish that
its score ordering is unusable. The operating point materially changes the
comparison. Dice+equal weights also edits more at0.50, with a smaller B Dice gain
than the2×TP auxiliary recipe at that same threshold.

## Retrospective dense-grid diagnostics, final checkpoint

These are grid calculations, not deployed thresholds or independent estimates.
A is fitted. B labels must not be used to select a threshold and then describe
that maximum as unbiased held-out evidence. Rows retain experimental order.

| Loss | A-Dice-maximizing threshold | A Dice there | B Dice at that threshold | Retrospective B maximum / threshold |
| --- | --- | --- | --- | --- |
| F+2K | 0.440 | 0.363861 | 0.337770 | 0.341339 / 0.595 |
| F+K | 0.615 | 0.364525 | 0.337583 | 0.342418 / 0.750 |
| D+0.25(F+2K)/3 | 0.360 | 0.373934 | 0.338638 | 0.344913 / 0.550 |
| D+0.25(F+K)/2 | 0.485 | 0.369131 | 0.337584 | 0.342896 / 0.685 |

Optimizing the threshold on fitted A can favor more aggressive editing than
helps B. Stronger A fit does not automatically transfer through threshold
calibration. Inspect operating-point effects and discrimination separately.

## Durable records

- [Complete final 201-threshold tables with baseline, comparison figure and all-checkpoint CSV](threshold_sweep_final.md)
- [Final dashboard](../runtime/deletion_loss_ablation_a_20ep/reports/live_dashboard.md)
- [Fixed-order milestone metrics, CSV](../runtime/deletion_loss_ablation_a_20ep/reports/fixed_order_results.csv)
- [Fixed-order milestone metrics, JSON](../runtime/deletion_loss_ablation_a_20ep/reports/fixed_order_results.json)
- [Dense201-threshold summaries, per-finding records and plots](../runtime/deletion_loss_ablation_a_20ep/reports/threshold_sweep/index.md)
- [Completion and historical-reference audit](../runtime/deletion_loss_ablation_a_20ep/reports/completion.json)
- [Loss and execution specification](codex_execution_spec.md)

Results remain for user review. The next research question is whether the Dice
recipe's observed benefit repeats across seeds and patient splits; this report
does not start additional training or adopt a threshold.
