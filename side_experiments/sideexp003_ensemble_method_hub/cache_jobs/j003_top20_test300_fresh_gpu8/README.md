---
created: 2026-09-09
updated: 2026-09-09
status: active
---

# Top-20 test300 preparation on GPU8

**Current instruction (2026-09-09): finish Wave 1 only; hold ranks 5–20.**
The finish guard is armed. Scheduler PID `2103906` is intentionally stopped;
the four Wave 1 Docker workers continue unpaused. Do not manually resume the
scheduler or use the restart commands below while this hold is in force.
Guard PID `2974265` waits for complete publication/validation, then terminates
the scheduler and writes `stopped_after_wave1`. See
[the hold specification](../../finish_test300_wave1_spec.md), runtime
`finish_wave1_state.json` and `logs/finish_wave1.log`. Final closeout is
`wave1_hold_report.md`; later waves require a new decision after profiling.

This job exports the same frozen checkpoints as the paired val200 caches,
retaining ranks 1–20 in five four-GPU waves. Read
[the execution contract](../../test300_execution_spec.md) before running it.
Test labels are withheld; the outputs contain native CT logits and storage
validation evidence, with no test Dice/hit metrics or ensemble decisions.

`job_spec.json` freezes the source bundle, image, checkpoint/config identities,
test-only native and existing iso07 preprocessing, prompt order, and storage
contract after preprocessing completes. `validation_report.json` records the
release checks. The coordinator copies its compact `inventory.json` and
`report.md` here when all 20 caches pass.

Runtime progress, logs, control files, per-wave timings, and the live inventory:
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/cache/jobs/j003_top20_test300_fresh_gpu8/`.

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py cache watch \
  --job j003_top20_test300_fresh_gpu8 --once
```

The original coordinator failed when Docker could not pause the CPU search.
The user now accepts slowdown and requires the CPU search to remain untouched.
Restart the same frozen inference job through the separate recovery launcher:

```bash
python side_experiments/sideexp003_ensemble_method_hub/resume_test300_no_pause.py \
  --job side_experiments/sideexp003_ensemble_method_hub/cache_jobs/j003_top20_test300_fresh_gpu8/job_spec.json \
  --session-id no_pause_YYYYMMDDTHHMMSSZ
```

Read [the recovery policy](../../test300_no_pause_execution_spec.md). The
original `hub.py cache run` command still contains its frozen pause policy.
The coordinator validates and reuses its completed publications and local
staging. It rejects changed provenance and preserves the 20 TiB shared reserve.
