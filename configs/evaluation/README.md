---
created: 2026-07-23
updated: 2026-07-23
status: active
---

# Evaluation Configs

This folder stores small, fixed evaluator-compatible JSON files that are safe to
commit. They are used when an experiment needs a stable validation probe instead
of selecting the first `N` cases from metadata order at run time.

For experiment 002 and later comparable fine-tuning runs,
`rexgroundingct_val20_seed20260723.json` is the standard small validation
probe. For experiment 003, `rexgroundingct_val200_seed20260723.json` is the
fixed-order 200-case validation probe used at the 100-epoch checkpoint.

Each manifest records the source metadata hash, seed, shuffle rule, selected
source indices, selected case names, and output hash.

Experiment 012 uses exhaustive target-category censuses plus fixed-val80 union
sets at its intermediate barriers. Their case/finding counts, source hashes,
and zero train/validation overlap audit are recorded in
`rexgroundingct_exp012_category_evaluation_subsets.manifest.json`.
