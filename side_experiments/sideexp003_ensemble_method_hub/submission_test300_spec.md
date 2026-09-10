---
created: 2026-09-09
updated: 2026-09-09
status: active
---

# Wave 1 top-four test submission execution contract

Job s001_top4_test300_d123 consumes only ranks 1–4 of the immutable j003 test
job. It waits for stopped_after_wave1, coordinator exit and four strict
300-case/582-prompt publications. Later waves remain held. No inference,
label access, scoring or checkpoint selection is performed.

Four CPU case workers use one numerical thread each, published fp16/fp32
memory maps and chunks of 4,194,304 elements. The existing val200 float32
sigmoid is accumulated in rank order and thresholded at sum >= 2.0 (d1).
Existing Exp024 routes and physical 20 mm dilation define eligible whole-lung
support (d2) and prompt-selected support (d3), both independently from d1.
Identical label supports are shared within a case. Native CT headers are
copied without reorientation or preprocessing. Preprocessing is unchanged:
all model preprocessing was already performed by the frozen source export.

Frozen inputs include source files, pinned image, dataset, routes, audit,
anatomy hashes, native geometry, checkpoint identities and paired val caches.
Completed cache manifests are bound once the dependency gate opens. Every
input array hash is checked while streaming its bytes for averaging. No
full ensemble float cache is written. Output transactions record validated
temporary-file hashes before atomic rename, allowing recovery after any
individual publication. Unknown files or foreign ownership fail closed.

Runtime records belong to SideExp003/submission_jobs/s001_top4_test300_d123.
Only d1/d2/d3 folders and their ZIPs are written beside Exp024 a/b outputs.
Each ZIP stores exactly 300 nii.gz files at its root; CRC, file hashes,
geometry, binary uint8 and 582-prompt coverage are verified before completion.
Retain largest CT and largest finding-array smoke cases before other work.
Reserve 350 GiB in addition to 20 TiB free shared storage; never delete caches.

Poll the dependency every 30 seconds, persist progress every minute, and
estimate remaining case work after ten cases using processed voxel workload.
Reading, hashing, probability averaging, anatomy support, NIfTI writing and
validation timings are separate. The detached CPU-only Docker supervisor
has no Docker socket or access to challenge segmentation labels. Failure
leaves durable records and published files for an explicit same-job restart.
Tests and repository checks precede freezing and arming. Source changes
require a new frozen specification. This job never uploads submissions.
