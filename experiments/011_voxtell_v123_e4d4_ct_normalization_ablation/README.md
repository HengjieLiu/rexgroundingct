# VoxTell v123 CT Normalization And Encoder-LR Ablation

This repo-local folder records experiment 011 intent and small provenance
artifacts. Heavy caches and runtime outputs remain under `/mnt/shengdata1`.

## Status

- Status: `active`
- Canonical config:
  `configs/experiments/011_voxtell_v123_e4d4_ct_normalization_ablation.json`
- Runtime directory:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/011_voxtell_v123_e4d4_ct_normalization_ablation`
- Training guard: the launcher requires explicit `START_TRAINING=1`.
- The user authorized training at `2026-07-28T07:28:16Z`. Both generated
  caches completed their final audits at `2026-07-28T08:48Z`; every
  preparation and GPU-headroom gate subsequently passed.
- No optimizer update or training run had started at the time of authorization.
- Completed e4d4 run group:
  `exp011_ct_norm_e4d4_20260728T085103Z`.
- The e5d4 rerun is authorized and uses a new
  `exp011_ct_norm_e5d4_<timestamp>` run group.

## Scientific Question

At paired v123 e4/d4 and e5/d4 fine-tuning settings, does VoxTell benefit from:

1. clipping HU outliers before full-volume z-score; or
2. replacing z-score with a fixed clipped-HU linear mapping?

See `docs/voxtell/ct_hu_normalization_analysis.md` for the validation-wide
evidence and model interpretation.

## Ownership

- Config and execution spec are canonical in Git.
- Standard caches live under the shared VoxTell preprocessing root.
- Checkpoints, logs, predictions, and evaluator outputs stay outside Git.
- Run the experiment-index sync after training and evaluation produce results.

## Guarded Launch

Preparation-only mode does not expose GPUs:

```bash
DETACH=0 \
bash scripts/rexgroundingct/run_011_ct_normalization_ablation_docker.sh
```

The completed e4d4 launcher remains available for provenance. Launch or resume
the authorized e5d4 state machine with:

```bash
START_E5D4=1 \
bash scripts/rexgroundingct/run_011_e5d4_host_orchestrator.sh
```

The host orchestrator runs detached by default and creates a separate Docker
container for every training, evaluation, and reporting stage. Re-running the
same command resumes the active e5d4 group idempotently.

## Milestone Barriers

Each LR profile is segmented at epochs `5/20/40/60/80/100`. All three arms first
reach the same milestone and exit cleanly, then each arm runs fixed val20 on
its assigned GPU. The next training segment starts only after all three
evaluations and the canonical combined report update complete successfully.
Epoch-100 val200 starts after the epoch-100 val20 barrier.

Each milestone checkpoint preserves model weights, SGD momentum, AMP scaler,
global update, polynomial LR horizon, schedule cursor, and Python/NumPy/PyTorch
random-number states. GPU 3 is never exposed to the experiment. The report
retains the completed e4d4 results, adds e5d4 results as they arrive, and keeps
exp006 native e5d4 as a replication reference.

## Training Dynamics

Canonical e4d4/e5d4 normalization and paired encoder-LR loss figures are
generated under
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/011_voxtell_v123_e4d4_ct_normalization_ablation/reports/training_dynamics/`.
The CPU-only refresher reads completed report barriers and does not gate or
signal training. In-progress segment updates are excluded until their
`report.complete` marker exists. The shared gallery is
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/comparisons/training_dynamics/README.md`.
