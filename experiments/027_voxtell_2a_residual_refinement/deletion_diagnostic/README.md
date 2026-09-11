# Deletion-only fitting diagnostic

The user authorized coding and running this small diagnostic on 2026-09-10.
It is separate from the stopped four-arm Exp027 residual experiment. No larger
training run is authorized by this diagnostic.

- [Execution specification](codex_execution_spec.md)
- [Results](results.md)
- [Proposed four-arm training follow-up](../deletion_four_arm_plan.md)
- [Curated metrics and provenance](metrics_summary.json)
- [Live diagnostic report](../runtime/deletion_diagnostic_v1/report.md)
- [Training/evaluation figure](../runtime/deletion_diagnostic_v1/curves.png)
- Canonical config: `configs/experiments/027_deletion_diagnostic.json`.
- Implementation: `scripts/rexgroundingct/fit_027_deletion.py`.
- Tests: `scripts/rexgroundingct/test_027_deletion.py`.
- CPU closeout checks: `scripts/rexgroundingct/analyze_027_deletion.py`.

All fitting and full-volume results use eight deliberately selected half-A
findings. They measure fitting capacity and inference consistency, not held-out
performance. Half B is excluded from selection, gradients and evaluation.
