---
created: 2026-09-05
status: audit_complete_for_user_review
---

# Exp022: official anatomy on Exp007 val200

All 200 scans / 381 findings reproduce Exp007 cont e050 / abs e150: **Dice 0.346023; 296/381 hits**. No source masks, predictions, training or inference changed.

Official CT-RATE masks remain pending human visual review. Findings are exploratory on a previously used validation cohort, not held-out gains or deployment approval.

Diagnostic overlay status: AI_INSPECTED_HUMAN_REVIEW_PENDING.

## Main findings

- Indiscriminate exact lung clipping: Dice **0.3381** (Δ -0.0079); 129 findings worsen and 6 existing hits are lost. An average gain, if present, is not a safety guarantee.
- Prompt-routed exact fine-region clipping: Dice **0.3449** (Δ -0.0012); 116 findings worsen and 5 hits are lost. Explicit location does not guarantee anatomical-mask containment.
- Highest observed prompt-routed comparison: `prompt_fine:mask:20`, Dice **0.3573** (Δ +0.0113); 120 improve, 3 worsen; hit gains/losses +2/−0. This row is selected after looking at val200, not an unbiased improvement estimate or an approved method.
- A broader fixed comparator, `prompt_lung:mask:20`, reaches **0.3508**, with 0 measured Dice regressions and 0 existing TP voxels removed. Yet its minimum GT coverage is 90.7%: no regression on current predictions does not prove complete anatomical coverage of target voxels the model already missed.
- Compare exact masks against margins and boxes category by category. Dense/atelectatic lung, pleuroparenchymal boundaries, fissures and unsupported non-lung targets have different failure mechanisms. A broad box can preserve tissue that a lung segmentation omits, but also retains more false positives.
- No policy is adopted. Use this audit to choose a later controlled experiment and an acceptable harm budget after reviewing the individual failures.

## Full-cohort comparisons

| Policy | Selected (subset Dice before→after) | Full Dice | Δ Dice [95% CT interval] | Hits | Gain/loss | Better/worse | Mean/min GT retained | GT <95% / excluded | TP/FP removed |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |
| all_lung:bbox:0 | 381 (0.3460→0.3502) | 0.3502 | +0.0042 [+0.0021, +0.0067] | 297/381 | +1/−0 | 95/5 | 0.9986/0.7802 | 2/0 | 22353/786111 |
| all_lung:bbox:10 | 381 (0.3460→0.3501) | 0.3501 | +0.0041 [+0.0021, +0.0066] | 297/381 | +1/−0 | 80/0 | 0.9997/0.9272 | 1/0 | 1/720253 |
| all_lung:bbox:20 | 381 (0.3460→0.3500) | 0.3500 | +0.0040 [+0.0019, +0.0065] | 297/381 | +1/−0 | 75/0 | 1.0000/0.9984 | 0/0 | 0/653028 |
| all_lung:bbox:5 | 381 (0.3460→0.3502) | 0.3502 | +0.0042 [+0.0021, +0.0067] | 297/381 | +1/−0 | 84/1 | 0.9995/0.8495 | 1/0 | 1050/744272 |
| all_lung:mask:0 | 381 (0.3460→0.3381) | 0.3381 | -0.0079 [-0.0136, -0.0022] | 291/381 | +1/−6 | 159/129 | 0.8931/0.0000 | 153/1 | 946766/2055990 |
| all_lung:mask:10 | 381 (0.3460→0.3509) | 0.3509 | +0.0048 [+0.0025, +0.0075] | 297/381 | +1/−0 | 92/5 | 0.9925/0.0000 | 13/1 | 59791/1053178 |
| all_lung:mask:20 | 381 (0.3460→0.3510) | 0.3510 | +0.0050 [+0.0026, +0.0077] | 297/381 | +1/−0 | 89/2 | 0.9955/0.0118 | 7/0 | 10262/1005364 |
| all_lung:mask:5 | 381 (0.3460→0.3503) | 0.3503 | +0.0043 [+0.0019, +0.0071] | 297/381 | +1/−0 | 97/16 | 0.9849/0.0000 | 22/1 | 151982/1183357 |
| prompt_fine:bbox:0 | 361 (0.3491→0.3604) | 0.3567 | +0.0107 [+0.0064, +0.0155] | 298/381 | +2/−0 | 116/6 | 0.9919/0.3974 | 8/0 | 1895/755110 |
| prompt_fine:bbox:10 | 361 (0.3491→0.3596) | 0.3559 | +0.0099 [+0.0056, +0.0147] | 298/381 | +2/−0 | 104/1 | 0.9941/0.4481 | 5/0 | 37/696103 |
| prompt_fine:bbox:20 | 361 (0.3491→0.3593) | 0.3556 | +0.0096 [+0.0054, +0.0143] | 298/381 | +2/−0 | 98/1 | 0.9961/0.4481 | 3/0 | 37/646049 |
| prompt_fine:bbox:5 | 361 (0.3491→0.3600) | 0.3563 | +0.0103 [+0.0060, +0.0151] | 298/381 | +2/−0 | 106/1 | 0.9935/0.4481 | 5/0 | 51/718608 |
| prompt_fine:mask:0 | 361 (0.3491→0.3479) | 0.3449 | -0.0012 [-0.0086, +0.0059] | 292/381 | +1/−5 | 170/116 | 0.8890/0.0000 | 151/2 | 808163/1670125 |
| prompt_fine:mask:10 | 361 (0.3491→0.3601) | 0.3565 | +0.0104 [+0.0061, +0.0154] | 298/381 | +2/−0 | 125/8 | 0.9874/0.4481 | 16/0 | 291639/1142030 |
| prompt_fine:mask:20 | 361 (0.3491→0.3611) | 0.3573 | +0.0113 [+0.0069, +0.0164] | 298/381 | +2/−0 | 120/3 | 0.9940/0.4481 | 7/0 | 129820/1056237 |
| prompt_fine:mask:5 | 361 (0.3491→0.3597) | 0.3561 | +0.0100 [+0.0052, +0.0152] | 297/381 | +1/−0 | 131/15 | 0.9781/0.0000 | 27/1 | 402452/1210260 |
| prompt_lung:bbox:0 | 361 (0.3491→0.3535) | 0.3501 | +0.0041 [+0.0020, +0.0066] | 297/381 | +1/−0 | 86/2 | 0.9995/0.9496 | 1/0 | 565/614864 |
| prompt_lung:bbox:10 | 361 (0.3491→0.3533) | 0.3499 | +0.0039 [+0.0019, +0.0064] | 297/381 | +1/−0 | 73/0 | 1.0000/0.9973 | 0/0 | 0/575485 |
| prompt_lung:bbox:20 | 361 (0.3491→0.3532) | 0.3498 | +0.0038 [+0.0018, +0.0063] | 297/381 | +1/−0 | 69/0 | 1.0000/0.9998 | 0/0 | 0/514726 |
| prompt_lung:bbox:5 | 361 (0.3491→0.3534) | 0.3501 | +0.0040 [+0.0020, +0.0065] | 297/381 | +1/−0 | 75/0 | 1.0000/0.9886 | 0/0 | 0/593487 |
| prompt_lung:mask:0 | 361 (0.3491→0.3435) | 0.3407 | -0.0053 [-0.0105, -0.0003] | 292/381 | +1/−5 | 155/118 | 0.9116/0.2139 | 136/0 | 328929/1212705 |
| prompt_lung:mask:10 | 361 (0.3491→0.3541) | 0.3508 | +0.0048 [+0.0025, +0.0074] | 297/381 | +1/−0 | 84/2 | 0.9980/0.6862 | 4/0 | 617/778809 |
| prompt_lung:mask:20 | 361 (0.3491→0.3541) | 0.3508 | +0.0047 [+0.0025, +0.0074] | 297/381 | +1/−0 | 81/0 | 0.9996/0.9065 | 2/0 | 0/760909 |
| prompt_lung:mask:5 | 361 (0.3491→0.3538) | 0.3504 | +0.0044 [+0.0021, +0.0071] | 297/381 | +1/−0 | 88/11 | 0.9933/0.5024 | 12/0 | 21669/809958 |
| prompt_side:bbox:0 | 361 (0.3491→0.3589) | 0.3552 | +0.0092 [+0.0051, +0.0139] | 298/381 | +2/−0 | 105/3 | 0.9977/0.4481 | 2/0 | 602/708757 |
| prompt_side:bbox:10 | 361 (0.3491→0.3585) | 0.3549 | +0.0089 [+0.0047, +0.0136] | 298/381 | +2/−0 | 93/1 | 0.9984/0.4481 | 1/0 | 37/667967 |
| prompt_side:bbox:20 | 361 (0.3491→0.3583) | 0.3547 | +0.0087 [+0.0046, +0.0134] | 298/381 | +2/−0 | 89/1 | 0.9984/0.4481 | 1/0 | 37/616237 |
| prompt_side:bbox:5 | 361 (0.3491→0.3587) | 0.3551 | +0.0090 [+0.0049, +0.0137] | 298/381 | +2/−0 | 95/1 | 0.9983/0.4481 | 1/0 | 37/683209 |
| prompt_side:mask:0 | 361 (0.3491→0.3493) | 0.3462 | +0.0002 [-0.0064, +0.0066] | 293/381 | +2/−5 | 164/114 | 0.9101/0.2139 | 136/0 | 328966/1299790 |
| prompt_side:mask:10 | 361 (0.3491→0.3599) | 0.3562 | +0.0102 [+0.0058, +0.0152] | 298/381 | +2/−0 | 103/3 | 0.9964/0.4481 | 5/0 | 654/867230 |
| prompt_side:mask:20 | 361 (0.3491→0.3599) | 0.3562 | +0.0102 [+0.0058, +0.0152] | 298/381 | +2/−0 | 100/1 | 0.9980/0.4481 | 3/0 | 37/849330 |
| prompt_side:mask:5 | 361 (0.3491→0.3595) | 0.3559 | +0.0099 [+0.0055, +0.0149] | 298/381 | +2/−0 | 106/12 | 0.9917/0.4481 | 13/0 | 21706/898379 |

## Category conclusions

| Category | Findings | Baseline Dice | Recommendation |
| --- | ---: | ---: | --- |
| 1a | 3 | 0.0969 | Too unsupported for hard restriction: keep the prompt-routed no-op; do not substitute a lung or trachea mask for the target compartment. |
| 1b | 11 | 0.1459 | Promising only for a later controlled experiment. Highest observed prompt-routed Dice is `prompt_fine:mask:0` (Δ +0.0039; 2 worsened; 0 lost hits). This is a validation-selected description, not a preselected method or held-out estimate. Some findings still worsen; do not deploy from this result alone. |
| 1c | 17 | 0.1569 | Promising only for a later controlled experiment. Highest observed prompt-routed Dice is `prompt_fine:mask:5` (Δ +0.0135; 0 worsened; 0 lost hits). This is a validation-selected description, not a preselected method or held-out estimate. No measured regression under this comparison; this is not a general safety guarantee. |
| 1d | 6 | 0.1167 | Insufficient evidence; descriptive only. Highest observed prompt-routed Dice is `prompt_fine:mask:0` (Δ +0.0001; 0 worsened; 0 lost hits). This is a validation-selected description, not a preselected method or held-out estimate. No measured regression under this comparison; this is not a general safety guarantee. |
| 1e | 11 | 0.1942 | Promising only for a later controlled experiment. Highest observed prompt-routed Dice is `prompt_fine:mask:0` (Δ +0.0060; 1 worsened; 0 lost hits). This is a validation-selected description, not a preselected method or held-out estimate. Some findings still worsen; do not deploy from this result alone. |
| 1f | 4 | 0.2123 | Insufficient evidence; descriptive only. Highest observed prompt-routed Dice is `prompt_fine:mask:0` (Δ +0.0000; 0 worsened; 0 lost hits). This is a validation-selected description, not a preselected method or held-out estimate. No measured regression under this comparison; this is not a general safety guarantee. |
| 2a | 69 | 0.3376 | Promising only for a later controlled experiment. Highest observed prompt-routed Dice is `prompt_fine:mask:5` (Δ +0.0118; 2 worsened; 0 lost hits). This is a validation-selected description, not a preselected method or held-out estimate. Some findings still worsen; do not deploy from this result alone. |
| 2b | 49 | 0.3577 | Promising only for a later controlled experiment. Highest observed prompt-routed Dice is `prompt_fine:mask:20` (Δ +0.0105; 0 worsened; 0 lost hits). This is a validation-selected description, not a preselected method or held-out estimate. No measured regression under this comparison; this is not a general safety guarantee. |
| 2c | 60 | 0.4148 | Promising only for a later controlled experiment. Highest observed prompt-routed Dice is `prompt_side:mask:10` (Δ +0.0090; 0 worsened; 0 lost hits). This is a validation-selected description, not a preselected method or held-out estimate. No measured regression under this comparison; this is not a general safety guarantee. |
| 2d | 132 | 0.3949 | Promising only for a later controlled experiment. Highest observed prompt-routed Dice is `prompt_fine:mask:5` (Δ +0.0152; 1 worsened; 0 lost hits). This is a validation-selected description, not a preselected method or held-out estimate. Some findings still worsen; do not deploy from this result alone. |
| 2e | 11 | 0.4121 | Too unsupported for hard restriction: keep the prompt-routed no-op; do not substitute a lung or trachea mask for the target compartment. |
| 2f | 0 | — | Insufficient evidence: no findings. |
| 2g | 1 | 0.0483 | Too unsupported for hard restriction: keep the prompt-routed no-op; do not substitute a lung or trachea mask for the target compartment. |
| 2h | 7 | 0.1658 | Insufficient evidence; descriptive only. Highest observed prompt-routed Dice is `prompt_fine:bbox:0` (Δ +0.0770; 0 worsened; 0 lost hits). This is a validation-selected description, not a preselected method or held-out estimate. No measured regression under this comparison; this is not a general safety guarantee. |

## Interpretation and evidence

The 381 complete prompts were individually interpreted and frozen before new outcomes. Unsupported pleural, central-airway-wall, and aberrant-vessel targets bypass prompt routing. Bilateral distributions are not narrowed to their largest/dominant lesion. Lobe masks are broader parents, not bronchopulmonary segment masks.

All policies use fixed 0/5/10/20 mm mask/bbox margins; no GT-selected per-finding routing. Best observed category rows are labeled validation-selected descriptions. Raw TP/FP totals count voxels, not physical volumes. Selected-subset and per-case metrics, all-label GT overlap and all risk/extent/locality breakdowns are in the external report. Sparse categories cannot support confident generalization.

External report: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/022_exp007_official_anatomy_val200_audit/reports/report.md`.
All-prompt appendix: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/022_exp007_official_anatomy_val200_audit/reports/all_findings.md`.
Visual review: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/022_exp007_official_anatomy_val200_audit/reports/visual_review.md`.

Reproduction: run the measurement entrypoint against its frozen runtime seal, then `report_022_official_anatomy.py`. Source hashes, the preserved original seal and the documented GT-instance-format correction are external. Report-script hash: `b30715540eee6645808bd7608ab403e4d45e03cdf575bfdceaaa66cfa8094f64`.

**Stop for user review. No policy is adopted.**
