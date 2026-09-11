# Category threshold tuning

Independent threshold analysis of frozen val200 logit caches. The first run
evaluates A1 at 0.05–0.95 with 0.05 spacing, reporting raw mean finding Dice
and hit rate. Instance metrics and test prediction changes are deferred.

**A1 complete:** [report and artifact links](report.md). All 7,239 records
verified; the saved-cache 0.50 Dice/hit baseline reproduced exactly. The
executed notebook and HTML are ready for review; no thresholds are adopted.

Read `codex_execution_spec.md` for the scientific and execution contract.
`sweep.py` owns source validation, recipes, resumable counts, and CSV summaries.
`notebook.py` executes the review notebook from the completed metric tables.
The scripts do not modify the SideExp003 exporters or caches.

Runtime root:
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp006_category_threshold_tuning/runs/r001_a1_val200/`

Commands, inside the existing VoxTell image with repository and data mounted:

```bash
python side_experiments/sideexp006_category_threshold_tuning/sweep.py --recipe a1 --preflight
python -m unittest discover -s side_experiments/sideexp006_category_threshold_tuning -p 'test_*.py'
python side_experiments/sideexp006_category_threshold_tuning/sweep.py --recipe a1 --smoke
python side_experiments/sideexp006_category_threshold_tuning/sweep.py --recipe a1 --workers 2
python side_experiments/sideexp006_category_threshold_tuning/notebook.py --run-root /mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp006_category_threshold_tuning/runs/r001_a1_val200
```

Future recipes `b1`, `d1`, and `e1` require an explicit `--output-root`.
B1 preserves the registered category routing and accepts a frozen
`--source-manifest` mixing strict full caches and explicitly scoped subset caches.
`targeted_export.py` prepares only its 81 missing routed findings; see
[b1_d1_execution_spec.md](b1_d1_execution_spec.md). D1/E1 use registered paired val200
caches in frozen model order and equal sigmoid-probability weights. No missing
cache is generated automatically. A new source or scoring identity needs a
new run directory; existing case records are never silently mixed.

Array provenance uses SideExp003's C-order array-value SHA256 (excluding the
NPY header). Header/shape/size checks are separate. Dispatch is bounded to the
worker count so a failed case stops scheduling additional work. Each A1 case
also reproduces the cached per-finding 0.50 reference before its results are saved.

All arrays, logs, executed notebooks, HTML, and per-finding results remain
external. Only source, tests, specifications, and a small closeout belong here.
