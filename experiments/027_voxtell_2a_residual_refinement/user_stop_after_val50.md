---
created: 2026-09-10
updated: 2026-09-10
status: stopped_by_user_pending_review
---

# User-requested stop after epoch 50

Stopped at 2026-09-10T15:27:08.262335+00:00. All arms have exactly 5,000 optimizer updates and complete 69-finding evaluations. No update 5,001 occurred.

All checkpoints, schedules and histories are preserved. Results remain pending user review; there is no automatic ranking.

| Run | A Dice | B Dice | Full Dice | Full hits |
| --- | ---: | ---: | ---: | ---: |
| run1_train | 0.3366113 | 0.3293772 | 0.3330467 | 59/69 |
| run2_train_val_a | 0.3326867 | 0.3196950 | 0.3262850 | 58/69 |
| run3_val_a | 0.3340796 | 0.3024115 | 0.3184750 | 58/69 |
| run4_val_all | 0.3394726 | 0.3355343 | 0.3375320 | 59/69 |

At 15:18:35 UTC only coordinator PID 2755098 was suspended; evaluators and the CPU reporter continued. CPU finalizer PID 3134835 verified all four summaries, exact A/B/full recomposition, checkpoint hashes, complete journals and absence of later checkpoints. It then terminated the coordinator through its existing interruption handler. The technical interruption status is preserved separately; this was an intentional stop, not an evaluation failure.

The immutable original 10,000-update config and schedules remain as provenance. The remaining schedule is superseded. Any further training requires a new user instruction.

All coordinator, evaluator and reporter processes exited. At verification, GPUs 0–3 each showed 0% utilization and 2 MiB allocated. The dashboard now shows `stopped_by_user`.

- [Stop evidence](runtime/full_fp32_100ep/stop_after_val50.json)
- [Finalizer log](runtime/full_fp32_100ep/logs/stop_after_val50.log)
- [Final dashboard](runtime/full_fp32_100ep/reports/live_dashboard.md)
