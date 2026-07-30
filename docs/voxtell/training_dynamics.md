---
created: 2026-07-28
updated: 2026-07-28
status: active
---

# VoxTell Training-Dynamics Analysis

## Purpose

Training-loss plots are derived artifacts, not substitutes for fixed validation
metrics. Use them to inspect optimization stability and within-experiment
differences. Do not compare absolute loss values across experiments when their
objectives differ.

The canonical cross-experiment gallery is:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/
  comparisons/training_dynamics/README.md
```

Each experiment owns its detailed figures under:

```text
<runtime-experiment>/reports/training_dynamics/
```

This keeps plots beside their source experiment while giving humans and agents
one gallery for Exp008, Exp009, and Exp011.

## Plot Contract

- X-axis: optimizer update, fixed to `0..10000`.
- Thin lines: raw per-update loss.
- Bold lines: trailing 100-update mean.
- Vertical lines: declared evaluation milestones.
- Y-axis: panel-specific `99.5`th-percentile robust cap. The number of raw
  observations above the visible cap is recorded in the panel and JSON.
- Segmented, recovery, and final metric files are merged by `global_update`.
- Duplicate updates must agree numerically, and each arm must form a contiguous
  prefix beginning at update 1.
- Active Exp011 e5d4 plots stop at the latest completed
  `milestones/epoch*/report.complete` barrier. In-progress segment updates are
  intentionally excluded.

Exp008, Exp009, and Exp011 have different objectives:

- Exp008 total loss combines final v123, weighted proposal, and optional
  final-precision terms.
- Exp009 S3 arms combine base v123 with an S3 auxiliary term; its continuation
  baseline has no S3 auxiliary term.
- Exp011 logs one aggregate v123 objective. Its detailed view therefore compares
  normalization and encoder-LR profiles rather than unavailable subcomponents.

Compare loss values within an experiment panel only. Validation Dice and hit
rate remain the model-selection evidence.

## Generation

Validate inputs without writing:

```bash
python scripts/rexgroundingct/plot_training_dynamics.py \
  --experiments 008 009 011 \
  --dry-run
```

Generate or idempotently refresh all canonical outputs:

```bash
python scripts/rexgroundingct/plot_training_dynamics.py \
  --experiments 008 009 011
```

The plotter records the source metric paths and SHA256 hashes, run groups,
observed update ranges, script hash, and a plotted-input hash. It writes PNG,
CSV, JSON, and Markdown files atomically. If the plotted inputs and all expected
outputs are unchanged, it exits with `already_current`.

## Exp011 Refresher

The Exp011 refresher is a separate host CPU process:

```bash
bash scripts/rexgroundingct/run_training_dynamics_refresher.sh
```

It polls completed report markers every 60 seconds, exposes no CUDA devices, and
never gates or signals training. It refreshes after new Exp011 milestones and
exits after the final experiment-completion marker has been processed.

Use preparation-only inspection with:

```bash
DRY_RUN=1 \
bash scripts/rexgroundingct/run_training_dynamics_refresher.sh
```

The canonical Exp011 native e5d4 panel includes Exp006
`v123_cached_e5_d4` as a dashed replication reference.
