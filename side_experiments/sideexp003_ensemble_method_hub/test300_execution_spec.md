# GPU8 top-20 test300 base-logit export

Authorized 2026-09-09. Implement and execute `j003_top20_test300_fresh_gpu8`
on branch `gpu8`, using the frozen `top20_val200_dice_20260907T211605Z`
checkpoint roster. Preserve the existing CPU resume and frozen scoring code.

Export 300 test CTs / 582 ordered prompts per checkpoint. Test geometry comes
only from CT headers; never read test segmentation labels or calculate test
Dice/hits. Keep original checkpoint/config provenance and explicitly bind the
test preprocessing and CT-native restoration contract in new cache keys.

Build a separate `crop_zscore_native_test300_v1` cache using the unchanged
`crop_zscore_native_v1` method (300 images, no targets, 112.80 GiB). Reuse the
completed iso07 cache. Never modify the original preprocessing manifests.

Stage clipped float32 logits under `/data/hengjie/sideexp003_staging/`.
Publish content-addressed caches under the existing shared SideExp003 cache
root. Cast to float16 only after proving exact threshold-zero mask preservation;
otherwise publish float32 from the same inference. Use destination-local atomic
files across filesystems. Retain local staging until all published hashes,
geometry, prompt coverage and storage proofs pass. Resume only identical jobs.

Five waves contain ranks 1–4, 5–8, 9–12, 13–16, 17–20, one model per GPU.
Retain largest-input and largest-output smoke cases. Require 20,000 MiB free
before launch and 4,096 MiB after smoke. GPU export starts after the existing
CPU worker benchmark completes. Monitor once a minute, preserve 20 TiB shared
free space, and reserve each wave's worst-case float32 publication footprint.
Pause only the identified CPU production container on sustained >2x normalized
export cost; release an owned pause on recovery below 1.5x or export exit.

Expected final float16 arrays: 2.13 TiB; shared peak including preprocessing:
2.24 TiB. Four local float32 stages need 0.85 TiB (allocate 1 TiB). All-float32
shared output plus preprocessing needs 4.37 TiB; pause for capacity rather than
breach the 20 TiB reserve. Estimated export work is 19–20 hours; budget 24–30
hours including preparation/contention and replace ETA after Wave 1.

Before release: test split isolation, ordering, orientation, dtype fallback,
atomic cross-filesystem publication, interrupted-stage reuse, ownership and
disk gates; run SideExp003 and repository checks. Freeze final source hashes
before launch. Final inventory must cover 20 x 300 cases / 582 prompts, link
paired strict val200 evidence, report actual bytes and timing, and leave the
ensemble recipe and submission decisions unchanged.
