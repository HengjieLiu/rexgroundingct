---
created: 2026-07-29
updated: 2026-07-29
status: active
side_experiment_id: sideexp001_validation_probe_design
---

# Category-Aware Validation Probe Design

This CPU-only side experiment measures why the fixed random `val20` differs
from `val200`, compares category-dependent case-sampling strategies, and
selects a probe size from the category-accuracy versus case-count frontier.

It is deliberately separate from canonical experiment numbering. It does not
modify `configs/evaluation/`, `configs/experiments/`, `experiments/registry.yaml`,
or any training launcher.

## Inputs

- Full challenge metadata:
  `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- Legacy fixed probes under `configs/evaluation/`
- Per-case quick/global evaluator JSONs from Exp006/007/008/009/011
- Archived public-leaderboard Firestore response under `challenge_info/`

The released metadata contains all 300 test cases and 582 test prompts and
categories. Their masks remain hidden. The live public leaderboard evaluates
an undisclosed 150-case subset with 302 findings; those subset counts are
reported only as a secondary reference.

## Reproduce

Run the analysis from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp001_validation_probe_design/analyze_validation_probe.py
```

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s side_experiments/sideexp001_validation_probe_design \
  -p 'test_*.py'
```

The analysis is deterministic. Re-running it with unchanged inputs must
reproduce the same output hashes.

## Outputs

Small tracked artifacts are written under `outputs/`:

- `investigation_report.md`
- `analysis_summary.json`
- `split_category_audit.csv`
- `strategy_results.csv`
- `legacy_reconstruction.csv`
- `selected_category_errors.csv`
- an evaluator-compatible selected probe and manifest when all selection and
  holdout gates pass

No inference is launched. Candidate metrics are reconstructed from existing
per-case `val200` evaluation results after auditing reconstruction against
independently executed legacy `val20` evaluations.

After evaluating the accepted probe, summarize raw and category-weighted
metrics with:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp001_validation_probe_design/summarize_probe_eval.py \
  --probe-json <accepted-probe.json> \
  --probe-manifest <accepted-probe.manifest.json> \
  --eval-json <val_quick_global_eval.json> \
  --output-json <probe_summary.json>
```

The full-test-weighted result is always labeled as an extrapolation from
validation outcomes, not measured test performance.
