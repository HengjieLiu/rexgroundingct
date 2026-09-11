---
created: 2026-09-10
updated: 2026-09-10
status: active
---

# j005: streaming test300 continuation, ranks 9–20

**Live since 2026-09-10T20:10:52Z:** Wave 2 finished and the automatic gates
passed. Coordinator PID `765487` is running Wave 3 on all four GPUs; transition
PID `604748` records monitoring. All 16 retained publication outputs matched
exactly. The private sequential benchmark took 226.9 s versus 338.0 s for the
reference (33% shorter); this does not establish full-wave overlap speedup.
The frozen [job specification](job_spec.json) now exists. The lifecycle notes
below describe the gates that ran before launch.

Implements [the approved contract](../../stream_test300_spec.md). j004 finishes
Wave 2 unchanged under its pidfd finish guard. Its coordinator remains held;
do not restart j004 to launch later waves. j003 and d1/d2/d3 stay unchanged.

The detached `stream_test300_transition.py run` waits for the authoritative
`stopped_after_wave2` handoff, checks private retained-logit parity and cold
publication, runs regressions/repository checks, freezes `job_spec.json`, then
automatically starts `stream_test300.py run`. A failed gate stops the transition
without resuming either parent. The job specification deliberately does not
exist until these gates pass.

Four GPUs each feed one CPU case worker with two pending handoffs maximum.
Native arrays receive in-memory float32/float16 hashes and exact zero-mask
proofs before local staging. One checkpoint-wide dtype is selected from 300
small records. A separate single publisher verifies each local source in one
sequential conversion/write pass, then verifies the temporary shared output in
one pass before promotion. No second complete cache is created locally.

The next GPU wave starts once four staging manifests are complete and their
workers exit, while publication continues in rank order. Publication reads are
capped at 32 MiB/s during GPU work and paused only after sustained measured I/O
contention. Disk admission reserves the unpublished backlog plus the next
wave, retaining 20 TiB shared and 64 GiB available RAM. No unrelated container
is paused. All work stays on branch `gpu8`.

```bash
# Before job_spec.json exists:
python side_experiments/sideexp003_ensemble_method_hub/stream_test300_transition.py watch --once

# After the transition freezes and launches j005:
python side_experiments/sideexp003_ensemble_method_hub/stream_test300.py watch --job j005_test300_r09_r20_streaming_gpu8 --once
```

Runtime root:
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/cache/jobs/j005_test300_r09_r20_streaming_gpu8`.

- `transition_plan.json`: source-pinned automatic handoff policy.
- `transition_state.json`, `transition_monitor.jsonl`: hold/preflight/launch state and disk counters.
- `five_minute_reports.jsonl`: five-minute progress records, including the first 30 minutes after launch.
- `preflight/report.json`, `preflight/report.md`: exact reference parity, one-source-pass and physical-read evidence; no new inference.
- `automatic_launch.json`, `commands/`, `logs/`: exact launch commands and ownership.
- `state.json`, `monitor.jsonl`: staging/publication/verification counters and workload-weighted ETA.
- `staging_manifests/`: durable checkpoint-wide storage proofs, retained after owned staging cleanup.
- `publication_progress/`, `publisher_state.json`: copy and verification phases, bytes and timings.
- `inventory.json`: final 12 validated caches with paired val200 and eight-parent lineage.

Local owned staging and private benchmark copies are under
`/data/hengjie/sideexp003_staging/j005_test300_r09_r20_streaming_gpu8`.
No test labels, Dice, ensemble changes or submission actions are part of this job.

The private benchmark compares one sequential publisher for both old and new
paths. Only private fsynced copies receive DONTNEED advice; physical read
counters must confirm cold local reads. Later reference passes and destination
readback may benefit from cache. The small benchmark does not establish a
full-wave overlap speedup; live throughput replaces its initial estimate.
