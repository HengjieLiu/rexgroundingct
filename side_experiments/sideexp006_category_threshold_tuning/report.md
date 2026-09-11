# A1 category threshold review — complete

The raw A1 val200 sweep completed 200 cases / 381 findings at 19 thresholds,
producing 7,239 verified finding-threshold records. Full sweep wall time was
759.3 seconds with two CPU workers and a retained smoke case. No inference ran.

Runtime and review artifacts:

`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp006_category_threshold_tuning/runs/r001_a1_val200/`

- [Executed notebook](</mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp006_category_threshold_tuning/runs/r001_a1_val200/a1_val200_thresholds.executed.ipynb>)
- [HTML review](</mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp006_category_threshold_tuning/runs/r001_a1_val200/a1_val200_thresholds.html>)
- `finding_metrics.csv`, `category_metrics.csv`, `overall_metrics.csv`, and
  `comparison.csv`: complete counts, curves, and descriptive maxima.
- `run_manifest.json`, `implementation_manifest.json`, `execution_history.json`,
  `source/`, `sweep.log`, `completion.json`, and `notebook_verification.json`:
  frozen provenance and execution evidence.

| Overall comparison | Threshold | Metric |
| --- | ---: | ---: |
| Saved-cache Dice baseline | 0.50 | 0.34586800803778955 |
| Best observed global Dice | 0.60 | 0.3459823617843387 |
| Saved-cache hit baseline | 0.50 | 295/381 (77.4278%) |
| Best observed global hit rate | 0.70 | 297/381 (77.9528%) |

The global Dice gain is only 0.00011435. Category maxima differ: category 2d
peaks at 0.95 with Dice 0.399286 (baseline 0.394980, n=132), while category 2e
peaks at 0.20 with Dice 0.424396 (baseline 0.410408, n=11). These are descriptive
same-validation tuning results; no thresholds have been adopted. Category 2f
has no findings; the notebook marks every category with n<10 as a small sample.

The historical separate-inference baseline remains 0.3460234857111323 / 296
hits and is displayed separately. All tuning deltas use the saved-cache
baseline, which reproduced exactly. Float16 storage guarantees same-pass
0.50 mask parity; this sweep characterizes the stored values at other thresholds.

Verification: 15 focused tests passed, including a synthetic executed-notebook
integration test. Six real finding/threshold comparisons matched the existing
global-only evaluator within 1e-12; all 200 cases also passed cached 0.50
per-finding parity. Category totals and Dice recomposed correctly. All five
notebook code cells executed with zero errors and two embedded figures;
the exported figures were visually reviewed. Canonical repository workflow
checks passed with 19 pre-existing experiment-index warnings.

The initial reader incorrectly compared full NPY-file hashes with logical
array-value hashes. That attempt accepted no metric records, was stopped, and
is preserved in `attempts/initial_hash_check/`. The corrected implementation
has a logical-hash regression test and bounded failure dispatch.

B1/D1/E1 recipe resolution is implemented for separate future runs. B1 requires
additional strict cache preparation; D1/E1 have existing component caches.
Instance metrics, test predictions, and submission changes remain deferred.
