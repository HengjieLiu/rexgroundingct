---
created: 2026-09-07
updated: 2026-09-07
status: active
---

# Cache Jobs

Tracked cache-job folders contain immutable job specifications and compact
timing/final reports. Large arrays, progress, smoke records, control sentinels,
and logs live under the matching `/mnt` `cache/jobs/<job-id>/` path.

Job `j001_top20_val200_fresh` is fresh-only and schedules the frozen top 20 in
five consecutive four-GPU waves. Existing SideExp002 caches are historical
provenance only.

Job `j002_top20_val200_fresh_r09_r20` is the audited continuation after a
repository source-bundle change. It imports no arrays: it verifies ranks 1–8 as
strict parent evidence, then creates new source-bound cache keys for original
ranks 9–20 and schedules only original waves 3–5. See its
`source_drift_audit.json` before running it.

Job `j003_top20_test300_fresh_gpu8` exports the same 20 frozen checkpoints for
300 test CTs and 582 prompts. It uses a separate native test preprocessing
cache, local float32 staging, and verified shared publication. Its CT geometry
and withheld-label contract have new `v3_` cache identities. The coordinator
preserves the concurrent CPU resume and records paired val/test provenance.

Job `j004_test300_r05_r20_cpu4_gpu8` continues original test ranks 5–20 after
profiling, with the process-local NumPy allocation fix and four CPU workers.
Its new source-bound v4 keys preserve all original checkpoint and data
contracts; j003 Wave 1 and d1/d2/d3 remain unchanged.
