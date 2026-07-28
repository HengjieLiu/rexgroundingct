---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# VoxTell cached-native v123 e5/d4 DDP batch4 update-matched run

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
- Canonical config: `configs/experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched.json`
- Execution spec: `experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/codex_execution_spec.md`
- Runtime directory: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched`
- Runtime link: `runtime` is ignored by git and points to the runtime directory.

## Synced Small Artifacts

- Report snapshot: `experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/report.md`
- Metrics summary: `metrics_summary.json`
- Sync manifest: `sync_manifest.json`

## Ownership Rule

- Configs are canonical in the repo.
- Runtime config files are snapshots and should not be edited by hand.
- Logs, predictions, checkpoints, and raw evaluator outputs stay on `/mnt/shengdata1`.
- Re-run the sync script after an evaluation or training run writes a new report.
- Keep `codex_execution_spec.md` current before substantial long-running work.
