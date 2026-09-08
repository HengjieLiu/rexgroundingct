---
created: 2026-09-07
updated: 2026-09-07
status: active
---

# SideExp003 Ensemble Method Hub

SideExp003 is the single tracked control plane for future RexGroundingCT
ensembles. It owns the fixed-val200 checkpoint catalog, reproducible method/run
contracts, and the provenance for the shared logit cache. Large arrays, logs,
and progress files live only under:

```text
/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/
  sideexp003_ensemble_method_hub/
```

This is intentionally a side experiment. Do not add it to
`experiments/registry.yaml` or `configs/experiments/`.

## Current milestone

Foundation steps 1–4 are implemented. The user has additionally authorized a
fresh-only val200 export for the frozen top 20 leaderboard checkpoints. This
job deliberately ignores all SideExp002 logits, runs five consecutive waves of
four checkpoints, and publishes only new content-addressed SideExp003 caches.

Ranks 1–8 completed under `j001_top20_val200_fresh`. Repository source changed
before Wave 3 could launch, so ranks 9–20 continue under the separately hashed
`j002_top20_val200_fresh_r09_r20` job. Its source-drift audit verifies the eight
parent strict caches, retains original ranks and wave numbers 3–5, and assigns
new cache keys bound to the continuation source bundle. It never relabels or
overwrites a `j001` cache.

The cache job alone does not authorize greedy search. Later explicit user
decisions authorized metrics-only top-4 and top-8 full-val diagnostics. They
are not OOF results, do not set `Mmax`, and cannot become formal recipes.

## Catalog workflow

`checkpoint_catalog.json` is the only ranking data source. Both Markdown files
are atomically rendered from it and must not be edited by hand.

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  catalog add --experiment 027 --dry-run

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  catalog add --experiment 027 --apply

python side_experiments/sideexp003_ensemble_method_hub/hub.py check
```

The scanner accepts only single-checkpoint fixed-val200 evaluators with exactly
200 cases and 381 findings. It recomposes all metrics after joining to the
official category map by `(case name, finding index)`. Checkpoints are unique by
SHA-256. Conflicting duplicate evaluators stop the update and print an override
stub; the tool never guesses which source is canonical. Merge a reviewed choice
into `catalog_overrides.json`, including a non-empty rationale, then rerun the
dry-run. Empty or stale override paths remain hard failures.

Rows with missing checkpoint or threshold provenance remain visible as
`needs_review`, but roster construction excludes them by default.

## Runtime layout

```text
sideexp003_ensemble_method_hub/
├── cache/logits/
│   ├── legacy_sideexp002/
│   └── by_cache_key/
├── cache/jobs/<job-id>/
└── methods/<method>/runs/rNNN_<roster-slug>/
```

Tracked run folders mirror the method/run relative path and contain immutable
specifications, result JSON, reports, and the final decision. Runtime folders
contain only large caches, progress, logs, and materialized logits.

Cache keys bind dataset, checkpoint, config/inference contract,
preprocessing-manifest, source bundle, container image, and storage contract.
Fresh exports stage evaluator-layout `(F,X,Y,Z)` float32 pre-sigmoid logits
clipped to `[-30,30]`. They publish float16 only when an offline cast preserves
the threshold-zero mask at every voxel; otherwise the same staged inference is
published as float32. Dtype fallback never invokes the model twice.

## Fresh top-20 cache workflow

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  roster freeze --top 20 --metric overall_dice \
  --roster-id top20_val200_dice_20260907T211605Z

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache plan --roster top20_val200_dice_20260907T211605Z \
  --split val200 --wave-size 4 --reuse-policy fresh-only --apply

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache run --job j001_top20_val200_fresh --auto-continue

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache watch --job j001_top20_val200_fresh --interval 60
```

When a frozen source bundle changes between waves, create a separate audited
continuation rather than changing the original job specification:

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache continue --parent-job j001_top20_val200_fresh \
  --job-id j002_top20_val200_fresh_r09_r20 --start-rank 9 --dry-run

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache continue --parent-job j001_top20_val200_fresh \
  --job-id j002_top20_val200_fresh_r09_r20 --start-rank 9 --apply

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  cache run --job j002_top20_val200_fresh_r09_r20 --auto-continue
```

Each worker receives one physical GPU. Every wave retains its largest-case
prediction as a smoke result, requires 20,000 MiB free before launch and 4,096
MiB after the smoke, and does not release the remaining cases until all four
workers pass. Any candidate failure prevents the next wave from starting.

## Top-4 preliminary comparison

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble compare --roster top20_val200_dice_20260907T211605Z \
  --job j001_top20_val200_fresh --run-id r001_top4_fresh_val200 --dry-run

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble compare --roster top20_val200_dice_20260907T211605Z \
  --job j001_top20_val200_fresh --run-id r001_top4_fresh_val200 --apply
```

The worker is CPU-only and single-process. It streams sigmoid probabilities,
uses threshold `>= 0.5`, records generation and evaluation timing separately,
and writes metrics only. Its full-val Caruana path starts at rank 1 and adds
three basket positions with replacement.

For an independently frozen top-k uniform diagnostic:

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble uniform --roster top20_val200_dice_20260907T211605Z \
  --job j001_top20_val200_fresh --top 8 \
  --run-id r002_top8_fresh_val200 --dry-run

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble uniform --roster top20_val200_dice_20260907T211605Z \
  --job j001_top20_val200_fresh --top 8 \
  --run-id r002_top8_fresh_val200 --apply
```

Test300 uses the same split-aware artifact schema but remains `deferred`. Its
labels are withheld, so future caches can validate 300-case/582-prompt coverage
and geometry but cannot report Dice or hits.

## Wave-4-gated top-16 seeded diagnostic

The authorized top-16 diagnostic combines strict ranks 1–8 from `j001` with
strict ranks 9–16 from `j002`. It polls every ten minutes, starts while Wave 5
continues, computes the top-16 uniform result, and then runs independent
Caruana-with-replacement curves for `all/2a/2b/2c/2d` from frozen per-scope
K=4 seeds. Five CPU workers shard cases and share every model read across
scopes. No derived logits are written.

```bash
python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble seeded-plan \
  --roster top20_val200_dice_20260907T211605Z \
  --source-job j001_top20_val200_fresh \
  --source-job j002_top20_val200_fresh_r09_r20 \
  --top 16 --workers 5 \
  --job a001_top16_after_wave4_fullval --dry-run

python side_experiments/sideexp003_ensemble_method_hub/hub.py \
  ensemble seeded-run --job a001_top16_after_wave4_fullval \
  --poll-interval 600 --auto-start
```

Use `ensemble seeded-watch --job a001_top16_after_wave4_fullval --once` for a
one-shot status snapshot. These results are optimistic same-val diagnostics,
not OOF estimates or final recipes.

See [codex_execution_spec.md](codex_execution_spec.md) for gates and
[prior_ensemble_summary.md](prior_ensemble_summary.md) for historical evidence.
