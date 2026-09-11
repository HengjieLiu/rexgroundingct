# Four concurrent deletion-only runs

The user authorized implementation and execution on 2026-09-10. Stage 1 stops
after all four epoch 20 evaluations for user review; no automatic continuation.

Stage 1 completed successfully at 2026-09-10 20:08:43 UTC. All four runs and
their evaluations are finished; results remain pending user review.

- [Completed results in fixed run order](results.md)
- [Dice-first tradeoff audit: accepting TP loss for better Dice](dice_first_tradeoff_audit.md)
- [Proposed next training pipeline and loss audit](v2_training_proposal.md)
- [All saved thresholds, checkpoints and A/B/full results](all_thresholds.md)
- [Frozen-base probability sweep: 0.25–0.95](../base_probability_threshold_sweep.md)
- [Four-run training-loss comparison](../runtime/deletion_four_arm_20ep/reports/loss_comparison/four_run_training_loss.png)
- [Training-loss comparison PDF](../runtime/deletion_four_arm_20ep/reports/loss_comparison/four_run_training_loss.pdf)
- [Execution specification](codex_execution_spec.md)
- [Accepted method plan](../deletion_four_arm_plan.md)
- [Live dashboard](../runtime/deletion_four_arm_20ep/reports/live_dashboard.md)
- [Live figure](../runtime/deletion_four_arm_20ep/reports/live_dashboard.png)
- [Verification and launch](verification.md)
- Config: `configs/experiments/027_deletion_four_arm.json`.
- Launcher: `scripts/rexgroundingct/run_027_deletion_four_host.sh`.
- Implementation: `deletion027_data.py`, `deletion027_worker.py`,
  `deletion027_report.py`, `run_027_deletion_four.py` in `scripts/rexgroundingct/`.

All models are fresh FP32 initializations. Cached base inputs and the stopped
residual experiment are preserved. A is development data; B is held out from
refiner gradients in runs 1–3 and in-sample for run 4. Ranking remains with the user.
