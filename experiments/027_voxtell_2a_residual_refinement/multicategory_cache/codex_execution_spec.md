---
created: 2026-09-11
updated: 2026-09-11
status: authorized
experiment_id: "027_voxtell_2a_residual_refinement"
---

# Reusable 2b–2e cache execution specification

## Objective and authorization

Prepare complete frozen Exp007 inputs for future refinement of 2b, 2c, 2d and
2e. The user approved implementation followed by concurrent cache export on
GPUs 0–3 alongside the running 2a experiment. Do not start new refinement
training, change the existing experiment's lifecycle, or modify its hashed
source files. This specification is recorded before implementation.

## Inputs and exact coverage

Use checkpoint SHA256
`a499ad1c0fada9a7e4d78e352fa5510d31295e82749e00ef0a4eee6da685e4aa`,
the pinned challenge metadata and `crop_zscore_native_v1` source manifest from
the canonical Exp027 configuration. Import validation from the strict
SideExp003 cache for this same checkpoint; no validation GPU inference.

In category order 2b/2c/2d/2e, training CT/finding counts are
911/1367, 1129/1507, 1305/1743 and 182/237. Validation counts are
40/49, 53/60, 119/132 and 11/11. The union is 2553 training CTs/4854 findings
plus 177 validation CTs/252 findings: 2730 CTs and 5106 findings total.

Reuse the existing 200-CT patient split. A/B finding counts are 25/24, 31/29,
61/71 and 8/3. These are development subsets, with especially limited 2e B
evidence. No split optimization or label exclusion is introduced here.

Implementation audit found an inherited exception to train/validation patient
separation: `train_2936` has training `train_2936_a_1.nii.gz::3` (2d), and
validation `train_2936_b_2.nii.gz::0` and `::1` (2d, half A). Preserve the exact
requested cache scope, explicitly pin this exception and reject any additional
overlap. Record affected keys in cache views and the final inventory. Before
future 2d training, resolve this exposure; do not describe its A scores as
patient-held-out without that adjustment. A/B themselves remain disjoint.

Canonical config: `configs/experiments/027_voxtell_2bcde_cache.json`.
Runtime: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/027_voxtell_2a_residual_refinement/cache_2bcde_v1`.
Only code/configs/small summaries belong in Git; arrays/logs/figures remain
external. Reference existing normalized CT arrays rather than duplicate them.

## Preprocessing and numerical contract

Preprocessing is cached-equivalent native VoxTell: full cropped-volume z-score
once, no resampling, established CT orientation transforms, 192-cubed windows
with 50% overlap. Generate training logits once per relevant CT with its full
original numerically sorted finding/prompt sequence, selecting all requested
category channels only after inference. Separate findings remain separate;
their own instance IDs become binary foreground.

Reuse the established frozen predictor settings, including text encoding and
segmentation precision. The FP32 refinement default does not change the base
producer. Clip stored logits to [-30,30]. Preserve validation source dtype;
training storage is FP16 only if its >=0 mask exactly matches clipped FP32,
otherwise retain FP32. Record source and array hashes, dtype and same-pass
mask reproduction. Expected added storage is about 1.4 TiB with binary GT,
up to roughly 2.3 TiB with FP32 logits.

The cache contract binds producer code, checkpoint, preprocessing, metadata and
selected channels, independently of training losses, schedules and A/B views.
Record split provenance separately. Keep existing 2a cache provenance unchanged.

## Execution, concurrency and recovery

Expose prepare, export, verify, report, dry-run, orchestrate and explicit resume.
Metadata/validation preparation is CPU-only; the loader cannot invoke VoxTell.
Use four disjoint deterministic training shards, one worker per GPU. Require
32 GiB free before each worker, a 26 GiB PyTorch allocator cap, and at least
4 GiB actual free GPU memory during monitored export. Smoke each shard's
largest native volume and largest volume-times-original-prompt workload first,
retain their outputs, and release the full group only after all smoke checks.
Insufficient memory, OOM, non-finite output or integrity failure stops the cache
group and records evidence; do not alter precision, windows or the 2a job.

Use a coordinator lock, per-CT locks, atomic arrays and a final completion
marker binding array/metadata/tile-index hashes. Verify completed cases on
resume. Finish or rebuild partial cases; invalid completed artifacts fail.
Publish per-worker counts, current CT, timings, storage, GPU memory, failures
and measured ETA through a single CPU Markdown report writer. Sharing can
slow both experiments; do not promise the previous 2a ETA.

```bash
START_GPU_WORK=1 DETACH=1 \
bash scripts/rexgroundingct/run_027_multicategory_cache_host.sh \
orchestrate --gpus 0 1 2 3 --share-gpus
```

Repeat with `--resume` for explicit recovery. No automatic failed-job retry.

## Training readiness and verification

Write per-category train/A/B/full manifests, memory-mapped cache-only access,
full-finding TP/FP/FN/GT counts and exact 192-cubed deletion tile indexes using
bounded-memory calculations. Keep zero-base-positive findings, separately
marking deletion eligibility. Save mean per-finding baseline Dice and underlying
counts for every source/subset; use original geometry for validation checks.

Required validation baselines rounded to six decimals are 0.357030, 0.415254,
0.394980 and 0.410408. Every validation finding must match its strict source.
Verify complete identifiers, prompt ordering, patient separation, geometry,
source hashes, array hashes, clipping/dtype, binary GT, tile coverage and
whole-finding counts. CPU Docker tests with GPUs disabled cover deduplication,
nonconsecutive finding IDs, orientation, empty predictions, storage identity,
partial recovery, corruption rejection, cache-only loading, reports and launch
gates. Inspect a labeled synthetic report; run repository and whitespace checks.

After complete verification, publish `input_manifest.json`, category manifests,
baseline CSV/JSON, timing/storage inventory, report and stable README links.
Set `cache_ready_for_training`. Category training, ranking and selection remain
out of scope. Failures and partial state must never be labeled ready.
