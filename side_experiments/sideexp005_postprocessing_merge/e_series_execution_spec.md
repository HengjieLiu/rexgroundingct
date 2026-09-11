# Frozen top-eight e-series execution

Approved 2026-09-10. Stable investigation: `sideexp005_postprocessing_merge`.
Configuration: [e_series_config.json](e_series_config.json). Entry point:
`python side_experiments/sideexp005_postprocessing_merge/e_series.py arm`.
`preflight`, `watch --once`, and `report` are also available. Numerical stages
are explicit in `e_series_compute.py preflight|run|report --stage val200|test300
--config <frozen-runtime/config.json>`.

## Inputs and immutable methods

Freeze original ranks 1–8 from `roster_top8_val200_fresh_j001.json`, their
checkpoint identities, paired validation manifests and original cache order.
Test inputs are ranks 1–4 from j003 v3 caches and 5–8 from j004 v4 caches.
Strict cache completion, hashes, split, prompt order, native geometry and
preprocessing provenance are required. Hash array contents while reading;
the poller reads only small manifests and worker status.

e1 accumulates the existing float32 sigmoid in rank order, eight equal weights,
4,194,304-element chunks, and thresholds the probability sum at >= 4.0.
e2 and e3 apply our eligible whole-lung / prompt-selected support with 20 mm
dilation independently to e1. e11 applies collaborator semantic-v1 to e1;
e12 applies strict semantic-v2 to e11. Reuse the original r001 source snapshot
and unchanged `methods.postprocess`, retaining all eligibility, fallback,
dilation and coordinate behavior. Use official CT-RATE labels 10–14 and
split-specific frozen prompts/routes. No inference or orientation restoration.
Anatomy status remains `PASS_PENDING_MANUAL_VISUAL_REVIEW`.

## Schedule and resources

`r003_top8_val200_frozen_postprocessing` starts immediately after checks and
preflight. Reconstruct 200 e1 predictions / 381 findings, then require Dice
0.35234375344586844, 291 HITs and matching category metrics within 1e-12.
Only then apply e2/e3/e11/e12. Retain smoke cases for the largest native CT and
largest finding array. A baseline mismatch blocks downstream processing.

The detached coordinator checks Wave 2 immediately and every 600 seconds,
recording timestamped progress, publication status, failures and available ETA.
`r004_top8_test300_frozen_postprocessing` starts after verified val report
collection, all eight strict 300-case / 582-prompt cache manifests and successful
exits of the four Wave 2 workers. Later waves and the j004 coordinator may remain
running. Validation score quality never gates test generation.

Each stage uses four CPU workers, one numerical thread each,
`NUMPY_MADVISE_HUGEPAGE=0`, the configured immutable image, a 96 GiB container
limit and no GPUs. Test execution has no segmentation-label mount. Worker
progress is recorded every minute; workload-normalized ETA starts after ten
new completed cases per phase. Record read/hash, averaging, anatomy, write,
score and verification time independently.

Require 250 GiB val / 350 GiB test headroom above 20 TiB shared free reserve
before launch and each dispatch. Space shortages wait and retry. Separate
locks, frozen source/configuration, durable prepared/complete case journals,
destination temporary files and atomic renames protect ownership and restart.
Only provenance- and hash-verified outputs may be reused. Incompatible inputs
and failed workers record explicit failure. No other job is paused or deleted.

## Outputs and acceptance

Runtime, private findings, logs and val predictions stay in the respective
external SideExp005 `runs/` directories named above. Test predictions go to
Exp024 `outputs/e1`, `e2`, `e3`, `e11`, `e12`. Create no ZIPs or uploads.
Require five validated val folders x 200 files and five test folders x 300 files,
binary uint8, finding-first native CT geometry/header, exact prompt coverage
and per-file hashes. No test metrics are calculated.

Publish `de_val200_report.md` without changing the d-only report. Include all
ten d/e variants, overall finding-wise and case-wise Dice, HIT, all 14 categories
(2f unavailable), e-postprocessor benefit/harm versus e1 and e12 versus e11.
Compare each e variant with its matched d variant using paired finding Dice/HIT
without subset assumptions. Classify changes at 1e-12; category partitions must
sum to their finding counts. Reuse verified d records. Link private routing,
largest changes, inventories, timing and reproduction evidence.

Register family e with eight equal 1/8 weights and explicit parents e1 -> none,
e2/e3/e11 -> e1, e12 -> e11. Keep test readiness, val evidence, ZIP policy
`skipped_by_request`, and manual submission history separate. The verified
collector updates aggregate documents and submission readiness automatically.

Before arming, test averaging parity/ties/mixed precision/chunks, numeric order,
frozen-method parity and subsets, Wave 2 gating while later waves run, incomplete
publication/failures, ownership, interrupted publication, space recovery, paired
reports and registry regressions. Run relevant existing repository checks.
