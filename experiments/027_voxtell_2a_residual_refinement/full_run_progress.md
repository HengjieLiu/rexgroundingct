---
created: 2026-09-10
updated: 2026-09-10
status: stopped_by_user_pending_review
---

# Full-run results through the requested epoch-50 stop

Stopped at the user's request on 2026-09-10 at 15:27:08 UTC (08:27 Pacific),
after every epoch-50 evaluation completed. All four models have exactly 5,000
updates and complete evaluations at epochs 10, 20, 30, 40 and 50. No update
5,001 occurred. All artifacts are preserved and GPUs 0–3 are idle.

Scores below are completed 69-finding evaluations. Run order is fixed; no
checkpoint or model is selected. A is training data for runs 2–4; B for run 4.
Full scores for runs 2–3 mix training and held-out findings.

| Epoch | Run | Dice A | Dice B | Dice full | Delta from cached base, full | Hits full |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cached base | — | 0.3394159 | 0.3349501 | 0.3372154 | 0 | 59/69 |
| 10 | run1_train | 0.3338657 | 0.3267986 | 0.3303834 | -0.0068320 | 58/69 |
| 10 | run2_train_val_a | 0.3386372 | 0.3323239 | 0.3355263 | -0.0016891 | 59/69 |
| 10 | run3_val_a | 0.3358736 | 0.3238694 | 0.3299585 | -0.0072569 | 59/69 |
| 10 | run4_val_all | 0.3329513 | 0.3217724 | 0.3274429 | -0.0097725 | 59/69 |
| 20 | run1_train | 0.3359168 | 0.3298620 | 0.3329333 | -0.0042821 | 59/69 |
| 20 | run2_train_val_a | 0.3371424 | 0.3273474 | 0.3323159 | -0.0048995 | 59/69 |
| 20 | run3_val_a | 0.3326123 | 0.3169773 | 0.3249081 | -0.0123073 | 60/69 |
| 20 | run4_val_all | 0.3324442 | 0.3270112 | 0.3297671 | -0.0074483 | 60/69 |
| 30 | run1_train | 0.3393981 | 0.3349719 | 0.3372171 | +0.0000017 | 59/69 |
| 30 | run2_train_val_a | 0.3397412 | 0.3344660 | 0.3371418 | -0.0000736 | 59/69 |
| 30 | run3_val_a | 0.3430545 | 0.3180986 | 0.3307574 | -0.0064580 | 60/69 |
| 30 | run4_val_all | 0.3334217 | 0.3265190 | 0.3300204 | -0.0071950 | 61/69 |
| 40 | run1_train | 0.3366836 | 0.3301053 | 0.3334421 | -0.0037733 | 59/69 |
| 40 | run2_train_val_a | 0.3385997 | 0.3261615 | 0.3324707 | -0.0047447 | 59/69 |
| 40 | run3_val_a | 0.3382661 | 0.3039863 | 0.3213746 | -0.0158408 | 60/69 |
| 40 | run4_val_all | 0.3375684 | 0.3321082 | 0.3348779 | -0.0023375 | 60/69 |
| 50 | run1_train | 0.3366113 | 0.3293772 | 0.3330467 | -0.0041687 | 59/69 |
| 50 | run2_train_val_a | 0.3326867 | 0.3196950 | 0.3262850 | -0.0109304 | 58/69 |
| 50 | run3_val_a | 0.3340796 | 0.3024115 | 0.3184750 | -0.0187404 | 58/69 |
| 50 | run4_val_all | 0.3394726 | 0.3355343 | 0.3375320 | +0.0003166 | 59/69 |

Epoch-50 results remain pending user review. A is used for training in runs
2–4; B in run 4. The small run-4 full Dice difference is an in-sample fitting
diagnostic. No ranking or checkpoint selection is made. The user superseded
the remaining 100-epoch schedule. See [stop verification](user_stop_after_val50.md).

[Live dashboard](runtime/full_fp32_100ep/reports/live_dashboard.md) · [Full-run verification](full_run_verification.md)
