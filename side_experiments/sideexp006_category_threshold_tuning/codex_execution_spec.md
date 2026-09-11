# A1 val200 category threshold sweep

Created: 2026-09-10. Authorized: implementation and A1 execution, following the
user-approved plan. This is SideExp006, outside the canonical experiment registry.

## Inputs and scientific contract

- Fixed split: `configs/evaluation/rexgroundingct_val200_seed20260723.json`,
  SHA256 `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`;
  200 cases / 381 findings, matched by name and numeric finding index.
- A1: Exp007 continuation epoch 50 / absolute epoch 150, checkpoint SHA256
  `a499ad1c0fada9a7e4d78e352fa5510d31295e82749e00ef0a4eee6da685e4aa`.
- Cache: SideExp003 `cache/logits/by_cache_key/` under its shared runtime,
  key `v2_d6dcddf705766db234bf631fa0925bfd69a5e18388767448b02a31b6484fcdbe`.
- Labels: `/data/hengjie/datasets/rexgroundingct/segmentations/`.
- Evaluator reference: `/data/hengjie/datasets/rexgroundingct/rexrank_eval.py`,
  global-only semantics: union masks, epsilon 1e-6, both-empty Dice 1,
  finding HIT iff Dice >= 0.1. Average findings, not case averages.
- Preprocessing is unchanged: existing native evaluator-layout logits from
  crop_zscore_native_v1; no new normalization, resampling, windows, or inference.
  Validate full native shapes and prompt ordering; do not reinterpret GT affine.
- Sweep integer percentages 5 through 95 in steps of 5. A1 compares float32
  logits against float32 logit cutoffs, inclusively. No anatomy or CC filtering.
- Float16 storage exactly preserves the same-pass 0.50 masks only; the sweep
  characterizes stored logits and makes no all-threshold float32 equivalence claim.
- Baseline gate: Dice 0.34586800803778955 within 1e-10, 295/381 hits exactly.
  Historical 0.3460234857111323 / 296 hits is a separately labeled diagnostic.

## Execution and artifacts

Create a source-bound run manifest before computation. Validate metadata,
cache manifests, array headers/sizes/logical-value hashes, finite values, and GT hashes.
Read sources only; two spawned CPU workers, one numerical thread per worker,
NUMPY_MADVISE_HUGEPAGE=0. Chunk scoring, memory-map source arrays, and load each
case's labels once. No GPUs, probability volumes, or threshold-mask exports.

Use an exclusive flock for run ownership. Each atomic per-case result includes
the run identity and content digest. Resume only matching, validated records.
Failure leaves partial evidence and no completion marker. Source drift is a
hard failure. Data hashes are checked as part of each case's first evaluation.

Runtime:
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp006_category_threshold_tuning/runs/r001_a1_val200/`.
Outputs: per-case JSON, finding/category/overall CSV, comparison CSV, summary,
run manifest, progress/completion records, executed IPYNB and standalone HTML.

Recipe interface: A1 singleton; B1 frozen category routing with explicit missing
cache failure; D1/E1 frozen four/eight-member equal probability averages. Only
A1 is executed. Future recipes require distinct output roots and preserve
native geometry, category joins, and no-inference scope.

## Notebook and gates

Execute all notebook cells from the completed tables. Include provenance,
definitions, baseline parity, overall curves, a 14-category grid, best-Dice and
best-hit comparisons, and a clear no-data 2f panel. Mark n<10 as small samples.
Break tied maxima by distance to 0.50 then lower threshold. Maxima describe
same-validation tuning, not adopted thresholds or unbiased estimates.

Before A1: focused numerical, malformed-input, resume, and recipe tests plus
source preflight and a retained real first-case smoke with per-finding 0.50
parity. Array SHA256 follows SideExp003's C-order value bytes, not NPY file
bytes. Bounded dispatch stops scheduling after failure. The first hash-check
attempt accepted zero cases and is preserved under `attempts/initial_hash_check`.
Before closeout: 7239 unique finding-threshold records,
category/overall recomposition, baseline gate, successful executed notebook
without error cells, HTML export, canonical repository workflow check and
Git whitespace checks. Report small closeout evidence and artifact paths.

Instance precision/recall/F1, new inference, B1/D1/E1 sweeps, submission recipe
changes, and test outputs are outside this execution.
