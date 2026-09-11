---
created: 2026-09-10
updated: 2026-09-10
status: active
---

# Finish Wave 2 and stream test300 Waves 3–5

The user authorizes finishing j004 Wave 2 unchanged, holding its coordinator,
then resuming ranks 9–20 under `j005_test300_r09_r20_streaming_gpu8` after
validation. The new GPU waves may overlap publication of earlier waves.

First arm `finish_test300_wave2.py`: pin and stop only the exact j004
coordinator with a pidfd; retain its independent workers. Poll every 30 seconds.
Require four exited-successful workers and strict 300-case/582-prompt cache
records. Retire the stopped coordinator, acquire both parent locks, preserve
logs and write `stopped_after_wave2`. Fail closed if identity, state or later
wave progress changes. Never alter Wave 2 numerical work or published data.

New staging keeps cached-equivalent preprocessing, model precision/windows,
text embedding behavior, numeric prompts and CT-native geometry unchanged.
Four GPUs each feed one CPU case worker with two pending handoffs maximum;
all processes use NUMPY_MADVISE_HUGEPAGE=0. Compute finite/range checks,
float32 and candidate-float16 hashes and exact threshold-zero preservation
while arrays are in memory. Persist proofs only after atomic float32 staging.
One dtype per checkpoint is selected from all 300 small proof records.

A separate one-process publisher reads each source once in chunks, verifies
its hash while converting and writing a temporary shared NPY, then verifies
that destination in one read before atomic promotion. No complete extra cache
is written locally. Record source bytes, destination bytes, hashes and phases.
Restart verifies committed outputs and reuses valid local stages; no full
inference repetition, no unconditional whole-cache scan, no cleanup until
complete publication validation. A failed partial copy retains local sources.

Start the next GPU wave only after all four preceding GPU workers exit with
300/582 staging proofs. Publication runs concurrently in rank order. Limit
local publication reads to 32 MiB/s while GPUs run; unlimited when idle.
Compare normalized per-case local input/staging I/O cost with the first wave's
pre-publication baseline. Two minute checks above 2x pause only the publisher;
two below 1.5x resume it. No unrelated process is paused.

Retain GPU gates 20,000 MiB before launch / 4,096 MiB after both largest
smokes; maintain 64 GiB available RAM and the shared 20 TiB reserve. Admission
reserves staging and worst-case final bytes for the whole unpublished backlog
plus a new wave, using proved dtype where known and float32 otherwise.

New source-bound cache keys and a frozen job retain checkpoint/config,
preprocessing, dataset, prompt, geometry and pinned-image identities, with
explicit j003/j004 completed-cache lineage. Keep branch gpu8, original frozen
sources and d1/d2/d3 unchanged. No labels, Dice, ensembles or submissions.

Implement and run small tests while Wave 2 finishes. Defer disk-heavy parity
and private-file cold-read benchmarks until its guard completes; never flush
global caches or evict production files. Require value/hash/geometry/storage
parity with retained benchmark data, one source pass, restart/ownership/corrupt
input tests, overlap/throttle/capacity tests and repository checks. Freeze
finished implementation before automatic launch. Record five-minute progress
for the first 30 minutes, then minute telemetry and revised ETA. Completion:
12 new strict caches of 300 CTs / 582 prompts plus eight parent caches.
