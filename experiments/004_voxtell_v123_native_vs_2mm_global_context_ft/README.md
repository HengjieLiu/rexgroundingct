---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# VoxTell v123 native versus 2 mm global-context continuation

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

- Status: `evaluation_complete`
- Canonical config: `configs/experiments/004_voxtell_v123_native_vs_2mm_global_context_ft.json`
- Execution spec: `experiments/004_voxtell_v123_native_vs_2mm_global_context_ft/codex_execution_spec.md`
- Runtime directory: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft`
- Runtime link: `runtime` is ignored by git and points to the runtime directory.

## Synced Small Artifacts

- Report snapshot: `experiments/004_voxtell_v123_native_vs_2mm_global_context_ft/report.md`
- Metrics summary: `metrics_summary.json`
- Sync manifest: `sync_manifest.json`

## Ownership Rule

- Configs are canonical in the repo.
- Runtime config files are snapshots and should not be edited by hand.
- Logs, predictions, checkpoints, and raw evaluator outputs stay on `/mnt/shengdata1`.
- Re-run the sync script after an evaluation or training run writes a new report.
- Keep `codex_execution_spec.md` current before substantial long-running work.
