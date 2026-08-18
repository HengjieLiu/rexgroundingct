---
created: 2026-08-13
updated: 2026-08-13
status: active
---

# Val200 Leaderboards

This folder stores dated snapshots of local fixed-val200 model rankings. Keep
one Markdown file per refresh so changes in ranking, inclusion criteria, and
runtime provenance remain reviewable over time.

## Current Snapshots

- `2026-08-13_top20_val200_no_ensemble.md`: top 20 non-ensemble evaluated
  checkpoints ranked by fixed-val200 mean global Dice per finding.

## Refresh Policy

- Rank by `summary.mean_global_dice_per_finding` from
  `val_quick_global_eval.json`.
- Use fixed val200:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`.
- Exclude ensemble, probability-average, multiscale, and postprocess-gating
  outputs unless a future snapshot explicitly changes scope.
- Treat each evaluated checkpoint as one ranked model, even when multiple
  checkpoints come from the same experiment arm.
- Record the refresh date, filter rule, source paths or reports, and notable
  caveats in every snapshot.
