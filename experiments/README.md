---
created: 2026-07-22
updated: 2026-09-06
status: active
---

# ReXGroundingCT Experiments

This directory is a lightweight repo-local index. It keeps configs and
small summaries easy to inspect while large outputs remain on `/mnt/shengdata1`.

## Folder Map

- `README.md`: experiment index overview and drift policy.
- `registry.yaml`: generated machine-readable experiment summary.
- `<experiment-id>/README.md`: one experiment's status and ownership rules.
- `<experiment-id>/codex_execution_spec.md`: active or historical Codex execution spec.
- `<experiment-id>/metrics_summary.json`: small synced metrics/provenance summary.
- `<experiment-id>/sync_manifest.json`: synced hash and path manifest.
- `<experiment-id>/report.md`: small copied runtime report when available.
- `<experiment-id>/runtime`: ignored symlink to heavyweight runtime outputs.

## Drift Policy

- Edit canonical configs under `configs/experiments/`.
- Write or update `codex_execution_spec.md` before substantial long-running experiment work.
- Runtime configs are hashed snapshots copied at run start.
- `experiments/*/runtime` symlinks are ignored by git.
- Use `scripts/rexgroundingct/check_experiment_consistency.py` before committing.

## Experiments

| ID | Status | Report | Runtime |
| --- | --- | --- | --- |
| `001_voxtell_v1_1_miccai200_val_eval` | `evaluation_complete` | `experiments/001_voxtell_v1_1_miccai200_val_eval/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval` |
| `002_voxtell_text_ft_miccai_train_val` | `runtime_initialized` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val` |
| `003_voxtell_rex_ft_rescue_ablation` | `runtime_initialized` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation` |
| `004_voxtell_v123_native_vs_2mm_global_context_ft` | `evaluation_complete` | `experiments/004_voxtell_v123_native_vs_2mm_global_context_ft/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/004_voxtell_v123_native_vs_2mm_global_context_ft` |
| `005_voxtell_global_proposal_local_cascade` | `evaluation_complete` | `experiments/005_voxtell_global_proposal_local_cascade/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/005_voxtell_global_proposal_local_cascade` |
| `006_voxtell_cached_native_v123_lr_ablation` | `evaluation_complete` | `experiments/006_voxtell_cached_native_v123_lr_ablation/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation` |
| `007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched` | `phase2_continuation_active` | `experiments/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched` |
| `008_voxtell_dual_branch_proposal_refinement_ablation` | `evaluation_complete` | `experiments/008_voxtell_dual_branch_proposal_refinement_ablation/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation` |
| `009_voxtell_s3_attention_coupling_ablation` | `evaluation_complete` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation` |
| `010_voxtell_public_anatomy_prior_fusion` | `evaluation_complete` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/010_voxtell_public_anatomy_prior_fusion` |
| `011_voxtell_v123_e4d4_ct_normalization_ablation` | `active` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/011_voxtell_v123_e4d4_ct_normalization_ablation` |
| `012_voxtell_category_specialists_replay50_cont100` | `active` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/012_voxtell_category_specialists_replay50_cont100` |
| `013_voxtell_public_category_only_specialists` | `active` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/013_voxtell_public_category_only_specialists` |
| `014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched` | `runtime_initialized` | `experiments/014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched` |
| `015_voxtell_isotropic_resolution_audit` | `audit_complete` | `experiments/015_voxtell_isotropic_resolution_audit/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/015_voxtell_isotropic_resolution_audit` |
| `016_voxtell_iso07_hu_preprocessing` | `preprocessing_complete` | `experiments/016_voxtell_iso07_hu_preprocessing/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/016_voxtell_iso07_hu_preprocessing` |
| `017_voxtell_iso07_hu_ddp_bs4_update_matched` | `phase2_continuation_ready` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/017_voxtell_iso07_hu_ddp_bs4_update_matched` |
| `018_voxtell_category2d_nodule_audit` | `audit_complete` | `experiments/018_voxtell_category2d_nodule_audit/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/018_voxtell_category2d_nodule_audit` |
| `019_voxtell_iso07_lung_bbox_coverage_audit` | `audit_complete` | `experiments/019_voxtell_iso07_lung_bbox_coverage_audit/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/019_voxtell_iso07_lung_bbox_coverage_audit` |
| `020_ct_rate_ts_total_rex_val200_audit` | `audit_complete_pending_manual_visual_review` | `experiments/020_ct_rate_ts_total_rex_val200_audit/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/020_ct_rate_ts_total_rex_val200_audit` |
| `021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100` | `implementation` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100` |
| `022_exp007_official_anatomy_val200_audit` | `audit_complete_for_user_review` | `experiments/022_exp007_official_anatomy_val200_audit/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/022_exp007_official_anatomy_val200_audit` |
| `023_ct_rate_ts_total_rex_test300_audit` | `preflight_complete_awaiting_manifest_approval` | `experiments/023_ct_rate_ts_total_rex_test300_audit/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/023_ct_rate_ts_total_rex_test300_audit` |
| `024_test_inference_anatomy_audit` | `complete` | `experiments/024_test_inference_anatomy_audit/aggregate_report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit` |
| `025_iso07_best_anatomy_audit` | `complete` | `experiments/025_iso07_best_anatomy_audit/report.md` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/025_iso07_best_anatomy_audit` |
| `026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr` | `implemented_delayed_launch` |  | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr` |
| `027_voxtell_2a_residual_refinement` | `cache_preparation` | `` | `/mnt/shengdata1/hengjie/experiments/rexgroundingct/027_voxtell_2a_residual_refinement/full_fp32_100ep` |
