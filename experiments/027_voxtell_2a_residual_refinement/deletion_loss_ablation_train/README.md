# Four losses fitted on the original training set

The user authorized implementation, verification and four-GPU execution of
100 epochs × 100 updates per model. Launched at 2026-09-11 04:20:37 UTC
(September 10, 21:20 Pacific). The four CPU preparation workers verify all
required cached inputs first; the four trainers start automatically after the
complete-input gate passes. Live status is available in the dashboard below.

- [Execution specification](codex_execution_spec.md)
- [Verification and launch](verification.md)
- [Live dashboard](../runtime/deletion_loss_ablation_train_100ep/reports/live_dashboard.md)
- [Live 3×3 figure](../runtime/deletion_loss_ablation_train_100ep/reports/live_dashboard.png)
- [Full threshold tables and tradeoffs](../runtime/deletion_loss_ablation_train_100ep/reports/threshold_sweep/index.md)
- [Matched-update comparison with A-only training](../runtime/deletion_loss_ablation_train_100ep/reports/comparison_with_a.md)
- [Historical train-only reference checks](../runtime/deletion_loss_ablation_train_100ep/reports/reference_checks.json)
- [Completion audit](../runtime/deletion_loss_ablation_train_100ep/reports/completion.json)

All arms use the same 1,119 eligible training findings. A's 35 findings and B's
34 findings are excluded from gradients; both are repeatedly inspected
development data. The full cohort combines A+B, with no training exposure in
this run. An epoch denotes 100 optimizer updates, not a full dataset pass.

Canonical config: `configs/experiments/027_deletion_loss_ablation_train.json`.
Large artifacts remain under Exp027's external root in
`deletion_loss_ablation_train_100ep`. Historical arrays, models, reports and
provenance remain unchanged. No automatic ranking or threshold selection.
