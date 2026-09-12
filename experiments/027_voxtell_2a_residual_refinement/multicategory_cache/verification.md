---
created: 2026-09-11
updated: 2026-09-11
status: cache_ready_for_training
---

# Verification and execution

All 29 multicategory CPU tests passed in the existing
`rexgroundingct-voxtell:cu126` image with no GPU devices. Coverage includes real
category census and the inherited patient-overlap audit, finding/channel
ordering, CT deduplication, disjoint shards, original geometry, clipping and
signed-zero storage fallback, historical tile-index equivalence, small volumes,
empty predictions, corruption rejection, partial and complete recovery,
cache-only loading, completion gating, A+B summaries, report locking and
explicit GPU/memory guards. The repository workflow check passed with 19
pre-existing missing-runtime/sync warnings; whitespace and shell syntax passed.

The metadata-only Docker preparation succeeded without GPU access. All 26
source hashes bound by the running 2a loss experiment were rechecked unchanged.
The original-split 2d overlap is explicitly recorded: training
`train_2936_a_1.nii.gz::3`, validation `train_2936_b_2.nii.gz::0` and `::1` in A.
No finding is dropped from this cache. Patient-held-out training decisions
remain a later task.

The labeled synthetic Markdown dashboard was inspected with a completed smoke
worker, delayed peers, pending ETA, memory readings, fixed category/worker order
and exposure notes. Evidence and source snapshots are under runtime
`verification/`. Launched at 2026-09-11 08:21 UTC in
`rex027_cache_2bcde_v1` / `27c66fa0a54f`. Initial status is `smoke`; completion
remains pending until every required artifact passes verification.

The first real validation case exposed an omitted `1e-6` Dice smoothing term
in the new summary code: masks were exactly identical, but Dice differed by
up to `1.1e-10`. The coordinator stopped all cache workers before any completion
marker was published. Restored the exact existing evaluator formula, added a
regression test and reran all 29 CPU tests and the workflow check successfully.
Failed code/context/index evidence is preserved in the first attempt's
`dice_formula_diagnosis/`; the numerical cache producer and storage contract did
not change. The corrected execution context is
`28cdfa9b9ce6e6f22875cca523bd7a1142c22a88b343c6137789bc5093ac559d`.
Explicit recovery launched at 08:24:47 UTC in
`rex027_cache_2bcde_v1_resume_dice` / `5164d2f0609a`.
Validation imports then passed exact per-finding baseline and mask checks.

All eight retained GPU smoke cases passed. The coordinator released all four
workers to full export at 2026-09-11 08:32:03 UTC. Peak reserved GPU memory was
24.57 GiB and the minimum sampled free memory was 9.74 GiB, satisfying both
the 26 GiB allocator cap and 4 GiB reserve. Validation continues on CPU; cache
completion remains pending. The live dashboard reflects current verified counts.

Final update: all 2730 CTs / 5106 findings passed verification at
2026-09-11 15:51:04 UTC, and the container exited with code 0. The final
inventory, baseline tables and category manifests were published successfully.
Current source/context integrity, exact A+B key recomposition and eight real
cache-only loader probes (train/validation for every category) passed. See the
[completion report](results.md) for counts, storage, baselines and exposure notes.

## Commands

```bash
bash scripts/rexgroundingct/run_027_multicategory_cache_host.sh dry-run
bash scripts/rexgroundingct/run_027_multicategory_cache_host.sh prepare
bash scripts/rexgroundingct/run_027_multicategory_cache_host.sh import-validation
START_GPU_WORK=1 DETACH=1 bash scripts/rexgroundingct/run_027_multicategory_cache_host.sh orchestrate --gpus 0 1 2 3 --share-gpus
MULTICATEGORY_CONTAINER_NAME=rex027_cache_2bcde_v1_resume START_GPU_WORK=1 DETACH=1 bash scripts/rexgroundingct/run_027_multicategory_cache_host.sh orchestrate --gpus 0 1 2 3 --share-gpus --resume
bash scripts/rexgroundingct/run_027_multicategory_cache_host.sh verify --resume
bash scripts/rexgroundingct/run_027_multicategory_cache_host.sh report
```

`export` is an alias for the guarded full coordinator. `import-validation` and
`verify` use CPU workers only. Internal GPU workers are owned by the coordinator.
Large evidence files, per-attempt logs and completion records are retained under
the external `cache_2bcde_v1` runtime. The inventory is authoritative only after
full verification; successful code tests alone do not mean the cache is ready.
