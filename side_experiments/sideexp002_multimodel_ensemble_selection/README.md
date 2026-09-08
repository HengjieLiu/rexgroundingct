---
created: 2026-07-29
updated: 2026-09-07
status: active
side_experiment_id: sideexp002_multimodel_ensemble_selection
---

# Multi-Model Logit Ensemble Selection

This side experiment selects and evaluates ensembles from the 20 Exp006,
Exp007, Exp008, Exp009, and Exp011 checkpoints whose fixed val200 mean global
Dice per finding is at least `0.320000`.

It is deliberately outside canonical experiment numbering. It does not modify
`configs/experiments/`, `experiments/registry.yaml`, or the canonical experiment
sync workflow. Test-set inference is out of scope.

## Tracked Artifacts

- `candidate_manifest.json`: immutable candidate, model, evaluation, cache, and
  hash provenance.
- `frozen_sources/`: hash-named provenance snapshots used only when a declared
  canonical source changes after the roster is frozen.
- `analyze_candidates.py`: reconstructs published metrics and regenerates the
  model-selection and hard-mask complementarity records.
- `summarize_baseline_categories.py`: ranks the 20 fixed val200 baselines
  globally and by official ReX category.
- `summarize_all_val200_cutoff030.py`: revalidates the frozen 58-row primary
  audit roster, applies an overall Dice cutoff of `0.30`, and computes the
  retrospective per-category model oracle.
- `collaborator_attention_manifest.json`: immutable four-decimal transcription
  of the paired collaborator Baseline, All-category, and Strict-2b2c table.
- `compare_collaborator_attention_variance.py`: compares the paired attention
  effects with size-matched category variation from the 20-model baseline pool.
- `make_folds.py`: builds deterministic category-balanced case folds.
- `export_logits.py`: exports evaluator-aligned final pre-sigmoid logits.
- `launch_logit_exports.py`: resumable four-GPU candidate launcher.
- `wait_then_launch.py`: validation-aware follow-up queue launcher.
- `revalidate_completed_exports.py`: fail-soft, inference-free revalidation of
  existing arrays.
- `audit_logit_exports.py`: deterministic all-20 structural, storage, and
  historical-diagnostic audit.
- `finalize_when_complete.py`: strict audit and val200-search handoff.
- `search_ensembles.py`: streaming probability/logit ensemble evaluation and
  cross-validated selection.
- `analyze_all20_thresholds.py`: resumable fixed-grid threshold analysis for
  the uniform probability ensemble of all 20 stored logit exports.
- `outputs/candidate_model_selection.md`: ranked 20-model roster.
- `outputs/hard_mask_complementarity.md`: the evidence for retaining diverse
  lower-ranked candidates.
- `outputs/hard_mask_complementarity_summary.json`: machine-readable evidence.
- `outputs/folds.json`: deterministic five-fold case assignment.
- `outputs/baseline_per_category_performance.md`: full global and category
  Dice/hit rankings and observed ranges.
- `outputs/baseline_per_category_performance.json`: machine-readable companion
  to the baseline category report.
- `outputs/all_methods_val200_cutoff030_category_oracle.md`: all 50 eligible
  fixed-val200 methods ranked globally and by category, plus oracle results.
- `outputs/all_methods_val200_cutoff030_category_oracle.json`:
  machine-readable companion to the cutoff/oracle report.
- `outputs/collaborator_attention_vs_20model_variance.md`: full category,
  cross-category, Exp009-method, and trajectory-context comparison.
- `outputs/collaborator_attention_vs_20model_variance.json`: machine-readable
  companion to the collaborator attention comparison.
- `outputs/all20_uniform_probability_threshold_report.md`: full global and
  category threshold sweep for the fixed all-20 ensemble.
- `outputs/all20_uniform_probability_threshold_summary.json`: machine-readable
  companion to the all-20 threshold report.

## Runtime Boundary

Heavy arrays, manifests, locks, and logs live under:

```text
/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp002_multimodel_ensemble_selection
```

As of 2026-09-07, the `logits/` entry at that location is a compatibility
symlink. Its physical storage is managed by SideExp003 at:

```text
/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/cache/logits/legacy_sideexp002
```

The logical SideExp002 path remains stable so immutable export manifests and
existing no-inference analyses continue to resolve without edits. Do not replace
or rewrite legacy arrays; SideExp003 creates a new versioned strict export when
one of the seven analysis-only caches is selected for a final recipe.

Each case is stored as an atomic, memory-mappable `.npy` array in evaluator
layout `(F, X, Y, Z)`. The default representation is float16 logits clipped to
`[-30, 30]`. A candidate falls back to float32 only if float16 storage changes
its same-pass threshold-zero masks.
Canonical source files are never rewritten by this side experiment. If one
changes later, the exporter accepts only a byte-identical snapshot with the
SHA256 already recorded in the candidate manifest.

## Reproduce Tracked Evidence

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/analyze_candidates.py

PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/make_folds.py

PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/summarize_baseline_categories.py

PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/summarize_all_val200_cutoff030.py

PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/compare_collaborator_attention_variance.py
```

Run unit tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s side_experiments/sideexp002_multimodel_ensemble_selection \
  -p 'test_*.py'
```

## Export and Search

The launcher never runs more than one Side Experiment 002 inference worker per
GPU. Its default 32-GiB memory wait can be explicitly disabled for approved
co-scheduling with training:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/launch_logit_exports.py \
  --smoke --skip-checkpoint-hash

PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/launch_logit_exports.py \
  --no-memory-wait
```

`wait_then_launch.py` can arm a follow-up queue behind an already-running
candidate. It waits for export plus validation completion, not for free GPU
memory, and invokes the same locked no-memory-wait launcher.

The five-architecture smoke test gates serialization against the same in-memory
prediction. Its comparison with the historical evaluation is diagnostic because
separate mixed-precision CUDA runs can differ slightly. The same rule applies to
complete exports: historical Dice and hit differences are retained as warnings
and never reject otherwise intact logits.

After exports, generate the consolidated audit:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/audit_logit_exports.py
```

Existing complete arrays can be revalidated without invoking a model:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/revalidate_completed_exports.py \
  --candidate-id exp006_e5d4_e100
```

After all candidate manifests are complete:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/search_ensembles.py
```

The fixed all-20 diagnostic can deliberately use every structurally intact
export even when the strict same-pass reproduction proof is unavailable:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/analyze_all20_thresholds.py \
  --allow-unaccepted
```

It uniformly averages per-model sigmoid probabilities and evaluates thresholds
`0.10`, `0.20`, `0.25` through `0.75` at `0.05` spacing, `0.80`, and `0.90`.
Its per-case metric cache is external and resumable; it does not write another
full probability volume. The override bypasses gate status only: candidate
provenance, recorded array-hash verification, case completeness, file sizes,
dtype, geometry, and finite streamed values remain mandatory.

For unattended completion, `finalize_when_complete.py` waits for all 20 strict
acceptance records, writes the JSON/Markdown audit, and only then starts the
val200 cross-validated search. It never invokes test-set inference.

Probability averaging is the default primary domain. Logit averaging remains a
cross-validated secondary candidate. Hard masks are never averaged.
