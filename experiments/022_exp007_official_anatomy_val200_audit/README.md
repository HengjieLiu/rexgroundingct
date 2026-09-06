---
created: 2026-09-05
updated: 2026-09-05
status: audit_complete_for_user_review
---

# Exp007 official-anatomy val200 benefit-harm audit

This is a repo-local index for a runtime experiment. Edit the canonical
config in the repo, and keep large runtime artifacts under `/mnt/shengdata1`.

## Folder Map

- `README.md`: this experiment's status and ownership rules.
- `codex_execution_spec.md`: active or historical Codex execution spec.
- `metrics_summary.json`: synced metrics and provenance summary.
- `sync_manifest.json`: hashes, paths, and sync provenance.
- `report.md`: small copied runtime report when available.
- `runtime`: ignored symlink to heavyweight runtime outputs.

## Status

- Status: `audit_complete_for_user_review`
- Canonical config: `configs/experiments/022_exp007_official_anatomy_val200_audit.json`
- Execution spec: `experiments/022_exp007_official_anatomy_val200_audit/codex_execution_spec.md`
- Runtime directory: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/022_exp007_official_anatomy_val200_audit`
- No runtime symlink was created; use the explicit external path above.

## Review and boundaries

Read `report.md` for the repository-safe aggregate. The external runtime
`reports/report.md` contains all category discussions; `reports/all_findings.md`
contains every one of the 381 prompts and all fixed comparisons;
`reports/visual_review.md` records inspection of 40 selected diagnostic panels.

All 200 native grids and all 381 baseline scores pass. The 12,192
finding-policy comparisons and 19 focused tests pass. The highest-observed
prompt-routed result is exploratory Dice 0.357341, with three regressions;
this is not an adopted policy or a held-out estimate. Official-mask human QC
remains pending. No source masks, predictions, training or inference changed.

## Synced Small Artifacts

- Report snapshot: `experiments/022_exp007_official_anatomy_val200_audit/report.md`
- Metrics summary: `metrics_summary.json`
- Sync manifest: `sync_manifest.json`

## Ownership Rule

- Configs are canonical in the repo.
- Runtime config files are snapshots and should not be edited by hand.
- Logs, predictions, checkpoints, and raw evaluator outputs stay on `/mnt/shengdata1`.
- Re-run the sync script after an evaluation or training run writes a new report.
- Keep `codex_execution_spec.md` current before substantial long-running work.
