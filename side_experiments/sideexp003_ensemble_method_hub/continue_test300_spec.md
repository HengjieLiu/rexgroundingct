---
created: 2026-09-10
updated: 2026-09-10
status: active
---

# GPU8 test300 Waves 2–5 continuation

The user authorizes ranks 5–20 after p001, using four CPU workers, and asks
for 30 minutes of observation with progress/ETA every five minutes.
Job `j004_test300_r05_r20_cpu4_gpu8` retains the original j003 ranks,
checkpoint/config identities, numeric prompts, inference precision, window
settings, cached-equivalent native / existing iso07 preprocessing, and native
CT geometry. No normalization, resampling, embedding precision, model method,
or ensemble choices change. No test masks or metrics are accessed.

The separate source-bound job uses `NUMPY_MADVISE_HUGEPAGE=0` before importing
NumPy in every process. Four GPU processes each feed one spawned CPU worker
(four CPU workers total), each with one numerical-library thread. Each GPU
keeps at most two unfinished crop handoffs and overlaps the next inference
with CPU restoration/staging. Handoffs use atomic local NPY files and hashed
records. Existing frozen inference and geometry functions remain unchanged.
This is a bounded continuous pipeline, not a collect-16 barrier. Exact output
equivalence is checked against the captured p001 reference before launch.

Retain two largest smoke cases per model; require 20,000 MiB before launch,
4,096 MiB after smoke, and all four smoke workers before releasing full work.
Use 64 GiB RAM headroom with serialized CPU memory reservations. Check full
wave float32 publication footprint plus local staging/handoff headroom before
each wave; preserve 20 TiB shared free. Insufficient resources wait, with no
automatic deletion or interruption of unrelated jobs.

Re-use frozen j003 publication/validation: checkpoint-wide float16 only when
all zero-threshold masks survive; otherwise float32, same inference. Atomic
cross-filesystem publication, full 300-case/582-prompt validation, then cleanup
only of this job's staging. Restart validates stages or committed crop records;
conflicting source, cache, or live ownership fails. Original j003 remains held
and unchanged; j004 has new keys and its own launch lock, also holding j003's
launch lock throughout the continuation to prevent duplicate scheduling.

Runtime: shared SideExp003 `cache/jobs/j004_test300_r05_r20_cpu4_gpu8`;
local: `/data/hengjie/sideexp003_staging/j004_test300_r05_r20_cpu4_gpu8`.
Use the original pinned image. Save commands, environment, minute state,
per-case GPU/CPU timings, per-wave timings, and paired val/test inventory.
Frozen sources must validate before every wave and in each worker. Failure
stops only owned containers; verified artifacts remain restartable. No global
THP changes, production reranking, uploads, commits, or changes to d1/d2/d3.

Validation: synthetic handoff, provenance failure, limits, and restart tests;
real saved-crop parity without another benchmark inference; SideExp003 tests
and repository checks. Reproduction: `python continue_test300.py prepare`,
then detached `python continue_test300.py run`; `watch` reads live status.
Initial 31–63 h estimate is provisional until production smoke and ordinary
case timings measure the allocation-fixed GPU path. First smoke cases are
deliberately large and must not be extrapolated as average-case throughput.
