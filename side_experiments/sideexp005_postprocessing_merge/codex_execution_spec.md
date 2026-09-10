# SideExp005 execution spec

## Objective and prior evidence

Compare five fixed masks per CT on val200, with complete category benefit/harm
records. The baseline is SideExp003 uniform_global/r001_top4_fresh_val200:
finding Dice 0.3575041942161927 and 296/381 hits. Existing caches contain native
CT-index logits but no ensemble prediction volumes. The collaborator source
audit and ZIP are preserved, not retuned against these results.

## Inputs, ownership and preprocessing

Canonical configuration: `config.json`, run ID
`r001_d1_val200_frozen_postprocessing`. Runtime is the matching run beneath
`/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp005_postprocessing_merge/`.
Source logits, model checkpoints, GT, CT headers, Exp020 anatomy and Exp024
val prompt routing remain read-only. Freeze hashes of source manifests,
dataset, prompts, code, checkpoints' recorded identities and image ID.

No normalization, resampling, inference or sliding-window policy changes occur.
Use published cache precision. Native-zscore and iso07 model preprocessing
remain those of the frozen checkpoints. Do not orient arrays a second time.
Validate FXYZ cache shapes against native CT and GT index layouts; official
anatomy must match the CT affine and grid. GT's legacy identity header does
not replace the native CT geometry. Source manual visual review status remains
PASS_PENDING_MANUAL_VISUAL_REVIEW under the user's explicit reuse instruction.

## Methods

- d1: existing float32 sigmoid, fixed rank-order sum, chunks of 4,194,304,
  threshold sum >= 2.0. No dense probability or ensemble logit cache.
- d2: existing eligible whole-lung+20mm behavior, independently from d1.
- d3: existing eligible selected side/lobe+20mm behavior, independently from d1.
- d11: collaborator exact-lobe+15 / side+15 / whole+10, with its original
  parser, fallback and absence of category exclusions.
- d12: d11 intersected with collaborator frozen strict-v2 support only for
  hard-eligible exact signatures: apical, basal, superior_within_base,
  laterobasal, subpleural_peripheral+posterior. Other findings retain d11.

Official labels 10/11/12/13/14 map to LUL/LLL/RUL/RML/RLL. Preserve source
distance thresholds, world-axis calculations, union/intersection semantics
and empty-mask behavior. Input adapters may cache repeated support masks and
avoid computing unused supports only after exact reference parity tests.

## Execution and recovery

Separate flock and detached CPU-only container; four processes, one numerical
thread each. Set NUMPY_MADVISE_HUGEPAGE=0 before NumPy import as validated by
SideExp003 p001 profiling. No global kernel changes or other-job interruption.
Mount immutable data read-only; do not mount GPUs or Docker socket in workers.

Preflight freezes a runtime manifest, verifies 200/381 coverage, input headers,
anatomy hashes, prompt routing and source identity. Validate cache array-byte
hashes during their normal d1 reads. Atomically publish a per-case journal
before destination renames; restart accepts only owned, hash-valid artifacts.
Conflicting files, a live owner, source drift or geometry failure stop the run.
SIGTERM stops further dispatch and records interruption without deleting data.

Materialize and score all d1 cases first; require baseline/category reproduction
within 1e-12 and exact hits. Only then run postprocessing, retaining smoke
outputs for the largest CT and largest finding array before the remaining cases.
Reserve 250 GiB shared headroom plus 20 TiB free reserve before launch; check
reserve during execution. Never automatically delete existing caches.

Record read/hash, averaging, anatomy support, NIfTI write, scoring and output
validation timings. Emit state every minute and workload-normalized ETA after
ten completed cases per phase. The previous 68-minute baseline is historical,
not a prediction for this allocation setting and concurrent workload.

## Validation and success criteria

Test reference parity (original collaborator entrypoints and Exp024 transform),
threshold ties, precision, chunks, ordering, negation/history/ambiguity,
bilateral and multilobe routes, category bypass, spacing, axes and distance
boundaries. Test provenance failure, geometry, space, duplicate ownership,
partial publication/restart, and completion. Expected text routing is
110/67/204 lobe/side/whole; v2 selects 49 and leaves 332 identical to d11.

Require 1000 hash-valid native binary uint8 NIfTIs and 1905 finding scores.
Every postprocessed output is a subset of d1; d12 is also a subset of d11.
Report both finding-wise and case-wise overall Dice, finding HIT >= 0.1, all
14 categories (2f empty/NA), and improved/decreased/unchanged counts at 1e-12.
Report HIT gains/losses, TP/FP removed, new empty masks, and d12 versus d11.
Keep detailed prompts, case paths and ranked finding changes external. Only
aggregate summaries and provenance links enter Git. No instance metrics.

## Closeout and review gate

Run focused tests, relevant SideExp003/postprocessing/register regressions and
repository checks. Verify all artifacts, publish aggregate report and timing
inventory, then update the side-experiment index/current status. Do not use
canonical experiment sync. Reserve d11/d12 in the submission register with
separate validation links and test-pending status; old d123 completion cannot
mark them ready. Test generation is a separate user-reviewed phase.
