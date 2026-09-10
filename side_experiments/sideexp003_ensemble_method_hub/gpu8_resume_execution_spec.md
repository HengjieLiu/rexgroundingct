# GPU8 benchmark and seeded-search resume

Authorized on 2026-09-08. Work stays on branch `gpu8`; no merge or push.

Resume `a001_top16_after_wave4_fullval` through K=16 for all five scopes,
preserving its frozen numerical implementation, roster, seeds, and thresholds.
The parent has 200 initial/K5/K6/K7/K8 partials and 27 K9 partials.

`resume_seeded_caruana.py` owns an exclusive shared launch lock, an archived
snapshot, a 20-minute CPU benchmark, optional child preparation, and a detached
supervised worker. It never changes the two frozen scoring modules. Benchmark
20 completed K8 cases at size-quantile midpoints, using their original K7
baskets. Compare 5/10/20 processes with quotas 6/11/21, then repeat the baseline;
skip 20 processes unless time remains for that repeat. Each trial uses fresh
scratch partials. All trial rows must equal the archived reference exactly.
Require >=15% improvement against the faster completed baseline; choose the
smallest qualifying configuration within 5% of the fastest. Incomplete timing
evidence falls back to 5 workers; correctness failures stop all production work.

Five workers resumes the original execution in place. Higher concurrency uses
`a002_gpu8_top16_seed4_resume` / `r002_top16_seed4_gpu8_resume`, with a new
execution identity, validated partial imports, and a per-file import receipt.
Only metric JSON is copied. Original partials and all source logits remain
unchanged. Child specifications use `resume_spec.json`, explicitly consumed by
the auxiliary helper rather than the original wave-gated job launcher.

Before launch, reconstruct all existing completed curves and verify their
candidate trials, support, weights, and winners. Hold the parent lock throughout
benchmarking and production. Require 64 GiB available RAM and 20 TiB free disk
at launch; check disk, frozen provenance, and progress every minute. A unique
container is stopped and its exit confirmed before releasing ownership on
interruption. Never retry a failed calculation automatically.

Closeout requires 200 cases per pass through K16, unchanged inherited curves,
scope support 381/69/49/60/132, valid final weights and peaks, and separate
historical, replay, benchmark, and resumed timing records. All results remain
optimistic same-val200 diagnostics; no derived logits or final recipe promotion.
