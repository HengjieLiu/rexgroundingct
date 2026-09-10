---
created: 2026-09-10
status: active
---

# Test300 pipeline profiling execution specification

User-authorized bounded job `p001_test300_pipeline16_gpu8`: one inference pass
for 16 model–case pairs, ranks 5–8 of frozen j003, four cases per GPU. Production
Waves 2–5 remain held. No production cache, submission, scoring, or inference
source is modified. Preprocessing is cached-equivalent and unchanged: native
full-crop z-score or frozen iso07 fixed-HU, existing sliding windows and CT
orientation. No segmentation access or test metrics.

Select largest preprocessed input volume, largest native finding-array, median
and lower-quartile workload per model. Deduplicate and fill with next largest.
Freeze source hashes, image ID, dataset, candidates, metadata, selections,
full-cohort workload features and space budget before launch.

Runtime: `/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/profiling_jobs/p001_test300_pipeline16_gpu8`.
Scratch: `/data/hengjie/sideexp003_staging/p001_test300_pipeline16_gpu8`.
Retain exact pre-restoration crop logits for CPU replay. Reference inference
uses the original functions with timing wrappers; diagnostic spooling is timed
separately. CUDA events measure network and transfer intervals without inserting
a synchronization at every patch. Inclusive child timers are never added to
their parent timers. System telemetry every five seconds, progress every minute.

After the four GPU workers finish and exit successfully, replay CPU stages with
4, 8, then 16 spawn workers and one numerical thread each. Warm the same crop
files before each pass. Enforce a conservative per-case allocation reservation
and 64 GiB available-RAM floor. Recompute geometry, clipping, staging, global
per-checkpoint dtype selection, publication and strict validation. Compare exact
native float32 hashes and geometry against reference and exact storage masks.
Sample dtype decisions cannot establish the full 300-case dtype decision.

Every file write uses only the job namespace and atomic destination rename;
keep the 20 TiB shared reserve and a 1 GiB per-write margin. Model/case start
records forbid silently repeating an interrupted inference; verified crop
records can recover CPU work without another model pass. A job lock rejects
duplicate owners; failure prevents later dispatch and never controls unrelated
containers. GPU gates are 20,000 MiB before launch and 4,096 MiB after retained
largest-case smoke outputs. No automatic production resume.

Interfaces: `profile_test300.py prepare|run|watch|report --job PATH`; internal
`gpu --rank N` and `replay`. `prepare` writes immutable JSON. `run` owns detached
GPU and CPU containers and telemetry. `watch` is read-only. `report` summarizes
per-case timings, CPU throughput/memory, bottlenecks and a workload-weighted ETA
range; overlap remains an estimate with one GPU pass.

Before launch: synthetic selection, timing, geometry/precision parity, forced
float32 fallback, source drift, duplicate lock, interrupted recovery and resource
gate tests; SideExp003 tests and repository checks. After completion: sync small
JSON/CSV/Markdown reports, record limitations and next decision in current status.
