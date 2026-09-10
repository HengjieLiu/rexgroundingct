---
created: 2026-09-09
updated: 2026-09-09
status: audit_complete_recommendations_pending
---

# Exp027 data-loading audit

The largest observed loading problem is consistent with cold network-file reads.
The next implementation should target predictable access to scheduled data and
overlap CPU patch preparation with GPU updates. This audit reads existing
histories, schedules, code and storage metadata; it does not change the running
FP32 benchmark or perform competing I/O/GPU performance workloads.

## Measurements and interpretation

Both trials used identical 100-update schedules, including finding, source,
patch mode and patch RNG seed. The loader code and its output dtypes are the
same for FP16 and FP32. `data_seconds` covers metadata/mapping access, patch
reads, augmentation and CPU array construction. It excludes host-to-GPU copies,
model work, and the subsequent JSON/history writes.

| Run | FP16 load mean / p95, s | Later FP32 load mean / p95, s | FP16 load share of training | FP32 load share |
| --- | --- | --- | ---: | ---: |
| 1 | 0.975 / 4.077 | 0.080 / 0.102 | 70.4% | 7.4% |
| 2 | 1.106 / 3.907 | 0.081 / 0.105 | 73.6% | 7.7% |
| 3 | 1.237 / 3.755 | 0.086 / 0.110 | 75.6% | 8.2% |
| 4 | 1.468 / 4.100 | 0.086 / 0.107 | 78.6% | 8.0% |

FP16 had 29–47 loads exceeding one second per arm, with a worst load of 5.82 s.
FP32 had no load exceeding one second; its worst was 0.169 s. These are sequential
observations, not a controlled cold/warm experiment: the first trial and its
evaluation can warm filesystem pages, and resource contention can differ.
They strongly suggest cache residency as a major factor. FP32 precision itself
does not accelerate this unchanged CPU loader.

The files reside on an NFS4 mount. At audit time the host exposed about 503 GiB
RAM with about 485 GiB available (including reclaimable file cache), and local
NVMe had about 231 GiB free. These are snapshots, not reserved resources. The
benchmark inputs total about 158.25 GiB including referenced CT files. The
validation-only subset is about 41.9 GiB. See [the cache inventory](cache_preparation_audit.md).

The non-loading portion of a step averaged 0.378–0.386 s in FP16 and 0.956–0.990 s
in FP32. This includes transfers, update work and synchronization, so it is not
pure GPU kernel timing. FP32 had lower total training wall time despite more
expensive update work because its input loading was much faster. Raw training
wall time therefore cannot be used as a fair FP16/FP32 compute-speed comparison.

Eliminating all measured loading cost from the later FP32 trial would yield only
about 1.08–1.09× training throughput with everything else fixed. The comparable
arithmetic ceiling for the earlier stalled FP16 trial is 3.38–4.67×. Neither is
a measured optimization gain. A warmed FP16 replay is needed to establish its
steady-state baseline before claiming a loader or precision speedup.

## Findings in the implementation

1. [The training loop](../../scripts/rexgroundingct/exp027_runner.py) calls
   `store.patch` synchronously, then performs three blocking `.to(device)` calls,
   then updates the model. There is no background loader or prefetch queue.
2. [FindingStore](../../scripts/rexgroundingct/exp027_data.py) retains only two
   CTs' parsed metadata and three memory maps per CT. It repeatedly reopens
   files and reparses large foreground-coordinate lists when cases recur.
   This Python cache is different from the kernel's shared page cache; eviction
   of a memory-map object does not prove the underlying data left RAM.
3. `extract_patch` allocates a float32 validity mask on each of its three calls.
   Two of those masks are discarded. Patch preparation also builds intermediate
   arrays, converts/flips them into contiguous float32 storage, then copies CT
   and logits again through `np.stack`.
4. The sampler already has a deterministic per-event RNG seed. This supports
   prefetch without changing which patches are trained on, provided the consumer
   preserves event order and checkpoint cursors count successful optimizer calls.
5. Every update rewrites and fsyncs the entire growing training-history JSON.
   This cost is outside `data_seconds` and currently small, but its cumulative
   serialization/write volume grows quadratically with the number of updates.

Serial replay of the actual 100-update schedules gives these metadata/mapping
cache hits. These are simulations, not physical page-cache measurements:

| Run | Current capacity 2 | Capacity 32 | Capacity 64 | Capacity 128 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 4 | 4 | 5 |
| 2 | 5 | 19 | 28 | 28 |
| 3 | 8 | 70 | 70 | 70 |
| 4 | 7 | 48 | 53 | 53 |

Larger handle/metadata caches help the repeatedly sampled validation cohorts
most. They cannot remove first-touch reads for run 1's 95 distinct CTs. With
multiple loader processes, each worker sees only part of the sequence; measure
the resulting cache behavior rather than treating this serial simulation as a
throughput forecast.

## Recommendations, in priority order

### 1. Make access to scheduled inputs predictable

Prewarm the repeatedly used validation working set once through shared file
pages, and profile its startup cost separately. For larger training schedules,
stage a bounded set of upcoming cases from NFS to local NVMe, retaining canonical
files and hashes on the external runtime root. The staging set must include
the referenced CT files as well as logits, masks and metadata. A local copy of
the logit cache alone leaves CT reads on NFS.

Use a space/byte budget and read-ahead from the frozen event schedule. Validation's
approximately 41.9 GiB is a practical first target; benchmark staging must record
copy time and available space. Avoid separate eager whole-volume copies per
training/loader process. The whole 801-CT pool needs a size inventory before an
all-RAM or all-local-disk design is adopted. Expected benefit: fewer multi-second
stalls; the exact gain remains to be measured.

### 2. Add bounded, deterministic CPU prefetch

Start with one CPU producer per arm, then compare two if first-touch latency
still starves the GPU. Keep two to four prepared patches ahead; do not immediately
create many workers that compete for the same NFS bandwidth. A map-style dataset
indexed by the existing schedule can use ordered DataLoader workers and a
dedicated generator. Keep workers CPU-only, use spawn after CUDA initialization,
and retain the existing event-specific patch seeds.

Each current float32 batch payload is 108 MiB: two input channels plus target
and validity mask. Four ready patches per arm across four arms require about
1.69 GiB just for ready payloads; active, worker, shared-memory and pinned-copy
buffers add to that. Bound and measure total RAM. PyTorch documents background
loading and prefetch controls in its [DataLoader reference](https://docs.pytorch.org/docs/2.8/data.html)
and [performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

Persist the consumed schedule cursor, not the producer's prefetched cursor.
Discard unconsumed prefetched batches at interruption/barriers and reconstruct
them from the schedule on resume. Preserve order, source mixture, padding masks,
joint flips and AMP retry behavior. Use the same already-fetched batch for an
overflow retry.

### 3. Retain metadata/mappings and compact coordinate pools

Separate the metadata cache from the volume-map cache. Retain all small-cohort
metadata (32 A CTs / 63 full-validation CTs); use a configurable byte/handle
budget for training cases. Parse foreground coordinates once into compact int32
arrays, preserving their order and exact sampling behavior. Check open-file
limits and actual worker RSS. Avoid multiplying large Python coordinate lists
across processes; PyTorch documents this issue for
[multiprocess datasets](https://docs.pytorch.org/docs/2.8/data.html#multi-process-data-loading).

### 4. Build each patch directly into its final buffers

Allocate the final CT/logit tensor, target and one validity mask once. Copy/cast
selected voxels directly into them, applying the same flips and padding values.
Remove two unused validity-mask allocations (54 MiB of unnecessary mask writes
per update) and the extra input stack/copy. This is a targeted improvement to
the roughly 80–86 ms warmed loading path, not a solution to cold network reads.

### 5. Optimize CPU-to-GPU transfer after measuring it separately

Use a bounded pinned-memory pipeline and `non_blocking=True`. Pinning in the
main training thread can add a blocking copy; do it through the loader/prefetch
path. Actual GPU transfer/compute overlap requires pinned source buffers and
a separate copy stream with correct lifetime/event synchronization, as described
in [PyTorch's transfer guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
This targets time outside the currently recorded `data_seconds`; measure it
before attributing the network-file stalls to PCIe copies.

### 6. Address long-run logging, then consider storage-format changes

Use an append-only training event log with stable update IDs and periodic atomic
dashboard snapshots, preserving checkpoint-based recovery. Whole-history JSON
rewrites need not happen every update for a dashboard refreshed every 30 seconds.

A chunked/compressed patch-oriented storage format is a later option if measured
NFS transfer volume still dominates after prefetch and bounded staging. It adds
conversion, decompression and provenance complexity and needs comparison with
the existing memory-mapped NPY representation. Preserve voxel values and geometry.

## Proposed next experiment and constraints

After the current FP32 evaluation finishes, first instrument a loader replay to
separate metadata parsing, map opening, CT/logit/GT patch reads, allocations/casts,
and augmentation. For GPU tests separately time loader wait, transfer, update,
and reporting. Compare the baseline with the selected improvements on identical
events at the same precision, all four arms concurrently. Include clearly labeled
first-touch and warmed passes and report startup/staging separately. No global
cache flush on a shared machine is needed; disclose uncontrolled cache state.

Verify byte-identical patch tensors, finding order, source mixture and padding,
plus checkpoint/resume equivalence with unconsumed prefetch and AMP retries.
Measure mean/p95/max loading wait, throughput, RAM and file-cache residency,
local space, NFS bytes and GPU utilization. Pick worker counts from this evidence.

The current cache contract fingerprints all Exp027 production modules, including
the trainer. A loader-only change must retain original array producer provenance
and use an explicit compatible-input migration/new session; it should not cause
unnecessary VoxTell re-exports. A later cleanup can separate export dependencies
from training-code provenance while preserving strict input hashes.

Training-loader gains do not imply the same gain in full-volume evaluation.
In evaluation, memory-map reads occur lazily during inference and metrics, so its
small `loading_seconds` field does not measure all storage traffic. Evaluation
deserves separate profiling if it becomes the next bottleneck.

## Reproduction

The audit is reproducible with:

```bash
python scripts/rexgroundingct/audit_027_data_loading.py
```

This only reads histories, schedules and system metadata, then writes small
records under [`runtime/audits/data_loading`](runtime/audits/data_loading).
The actual schedule equality checks passed for all four arms. Recommendations
are not implemented; no new performance test or cache staging was launched.
