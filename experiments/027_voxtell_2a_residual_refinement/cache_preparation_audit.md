---
created: 2026-09-09
updated: 2026-09-09
status: audited
---

# Benchmark cache coverage and cost

The existing shared native CT/target preprocessing cache was reused. The new
Exp027 cache contains frozen Exp007 2a logits and binary finding masks for the
cases required by the four deterministic 100-update schedules, plus every
validation 2a finding. It does not contain logits for the entire training 2a pool.

| Source | Cached CTs | Cached 2a findings | Dedicated cache GiB |
| --- | ---: | ---: | ---: |
| Original train | 133 of 801 eligible CTs | 228 of 1,120 findings | 65.982 |
| Validation A | 32 of 32 eligible CTs | 35 of 35 findings | 10.154 |
| Validation B | 31 of 31 eligible CTs | 34 of 34 findings | 8.768 |
| Total | 196 | 297 | 84.904 |

The benchmark actually samples 140 distinct training findings and evaluates all
69 validation findings. Preparing a selected CT retains all of its original 2a
channels, so 228 training findings are cached even though only 140 distinct ones
appear in the benchmark schedules. Unselected training CTs have not undergone
this experiment's frozen-base logit export.

Disk usage was measured by summing the sizes of existing files. Logits occupy
56.516 GiB, binary targets 28.258 GiB, and metadata/foreground-coordinate pools
0.130 GiB. All 196 case logit arrays have FP16 storage provenance. The referenced
CT image files occupy another 73.350 GiB in the existing shared preprocessing
cache; Exp027 references them without duplicating them. FP32 benchmarking shares
these exact input files through a cache symlink, so it does not duplicate this
84.904 GiB cache or regenerate logits in a different precision.

## Preparation time

The original build was interrupted by a validation-cache checksum-reader bug.
Launch and cache-completion timestamps reconstruct these active intervals:

- Attempt 3: 2026-09-10 01:28:03.491 to 01:31:11.001 UTC, approximately 187.5 s;
  21 training CT caches survived this attempt.
- Attempt 4: launch 2026-09-10 03:12:59.154 UTC to final cache completion
  03:33:12.266 UTC, approximately 1,213.1 s; it retained those 21 cases and
  completed the remaining 175.

Together these are about **23.3 minutes of active elapsed preparation with four
workers**. This estimate includes startup and some failed-attempt work. It is
reconstructed from timestamps, not a clean uninterrupted fresh-cache timing.
The elapsed clock time between the two launches includes an additional repair
pause and must not be interpreted as GPU preparation throughput.

Later benchmark attempts reuse and verify all completed files. Their reported
`preparation_seconds` measures cache reuse/hash verification, not the original
logit generation. The cost of building the earlier shared CT preprocessing and
SideExp003 validation-logit caches is also outside these intervals.

## RAM and GPU memory

Peak host RAM and GPU memory were not recorded for the original cache generation
stage, so no retrospective peak is asserted. Arrays are memory-mapped from disk
for refiner sampling; the complete cache is not transferred to a GPU. The
separately measured FP16 refiner training peak is 10.682 GiB per GPU, which is
not a measurement of the frozen-VoxTell cache-generation stage.

Evidence: runtime `prepared.json`, `benchmark/*/schedule.json`, all
`cache/*/metadata.json` and array file sizes, and archived launch/status records
under `launch_attempts/003_validation_payload_hash` and `004_context_lock`.
