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

The cache job does not authorize greedy search. After all 20 fresh caches pass
their strict gate, the hub stops at the existing `Mmax` user gate.

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

Each worker receives one physical GPU. Every wave retains its largest-case
prediction as a smoke result, requires 20,000 MiB free before launch and 4,096
MiB after the smoke, and does not release the remaining cases until all four
workers pass. Any candidate failure prevents the next wave from starting.

Test300 uses the same split-aware artifact schema but remains `deferred`. Its
labels are withheld, so future caches can validate 300-case/582-prompt coverage
and geometry but cannot report Dice or hits.

See [codex_execution_spec.md](codex_execution_spec.md) for gates and
[prior_ensemble_summary.md](prior_ensemble_summary.md) for historical evidence.
