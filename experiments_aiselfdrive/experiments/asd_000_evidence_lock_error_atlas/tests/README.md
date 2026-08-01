# Tests

Tests cover source/lineage drift, summary-independent mask recomputation,
honest val80/val120 roles, RAS/left-right geometry, conservative prompt
parsing, tri-state labels, laterality-only outside-lung safety, vectorized
component reconstruction, null observation counts, deterministic clustered
bootstrap, accepted-logit validation, and fail-closed Docker fallback command
construction. Revision-2 contract tests additionally derive the exact
37,666,291,712-byte val80 float16 layout from all 80 prediction headers, verify
7/195 exact-case versus 195/195 normalized val80 bank coverage (14/381 versus
381/381 on val200), pin transitive VoxTell sources, exercise owned-container
timeout cleanup, and validate controller-bound failed evidence.
