# SideExp004 Best Checkpoint Share — Chushu

This SideExp004 folder is a local, provenance-aware analysis of the collaborator
val200 share bundle. It does not modify or enroll records in the canonical
SideExp003 checkpoint catalog.

## Scope

The input bundle contains 309 aggregate Full200 results, but only its 24
normalized `selected_models.json` entries have all 381 per-finding rows and
full official-category data. The generated reports therefore cover those 24
result variants only:

- 20 are source-declared `raw_like` results.
- 4 are source-declared pipeline/post-processing results.
- All 24 exactly align to the fixed val200 manifest: 200 cases, 381 findings,
  and all nonempty official categories. `2f` has zero findings.

The primary comparison uses the source-declared raw-like subset. The inclusive
pipeline table is descriptive only. `raw_like` is not equivalent to SideExp003
`eligible`: the supplied bundle lacks evaluator/configuration, mask-threshold,
and checkpoint-SHA provenance. Reported checkpoint paths are not locally
verifiable here.

## Inputs and generated artifacts

The immutable collaborator input is under:

```text
best_checkpoint_ensemble_share_20260907/outputs/
best_checkpoint_ensemble_share_20260907/
```

`build_collaborator_val200_leaderboards.py` validates that bundle against
`configs/evaluation/rexgroundingct_val200_seed20260723.json` and reads the
current SideExp003 catalog only to render the comparison.

Its generated files are:

- `collaborator_selected24_catalog.json`: normalized records, input hashes,
  coverage, and provenance status.
- `collaborator_val200_checkpoint_leaderboard.md`: overall inclusive and
  source-declared raw-like rankings.
- `collaborator_val200_subcategory_leaderboard.md`: wide and ranked official
  category tables for both cohorts.
- `my_vs_collaborator_val200.md`: independent overall/per-category maxima for
  SideExp003 versus collaborator rows, plus the pipeline-inclusive reference.

Do not hand-edit generated files.

## Commands

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp004_best_checkpoint_share_chushu/\
  build_collaborator_val200_leaderboards.py --dry-run

PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp004_best_checkpoint_share_chushu/\
  build_collaborator_val200_leaderboards.py --apply

PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp004_best_checkpoint_share_chushu/\
  build_collaborator_val200_leaderboards.py --check

PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  side_experiments/sideexp004_best_checkpoint_share_chushu/\
  test_build_collaborator_val200_leaderboards.py
```
