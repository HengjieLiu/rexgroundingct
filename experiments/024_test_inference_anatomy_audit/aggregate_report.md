# Exp024 aggregate report

Status: complete.  The b validation gate passed, and all six test prediction
sets were generated and verified in the external runtime.  This tracked
summary is intentionally not a test performance claim.

The reconstructed b1 validation result is Dice **0.360278032575** with
**298/381** hits, matching the published `0.360278032575` / `298/381` oracle.
At the fixed +20 mm margin, prompt-eligible whole-lung support gives Dice
**0.365156393439** (+0.004878360864; 300/381 hits) and prompt-supported
side/lobe support gives **0.367795760985** (+0.007517728410; 300/381 hits).
Exact whole-lung clipping is harmful: **0.351270187170** (−0.009007845405;
295/381 hits), with a paired CT bootstrap 95% interval of
[−0.014825832, −0.003851399].  The +20 mm intervals are [0.002666744,
0.007628018] (whole-lung) and [0.002308722, 0.012848933] (fine), using 2,000
resamples and seed `20260905`.

These are validation diagnostics only.  The category checkpoint oracle is
retrospective, validation-selected, and optimistic; it is not an unbiased
test estimate.  The external report remains authoritative for detailed
category and per-finding evidence.

Official test TotalSegmentator masks remain
`PASS_PENDING_MANUAL_VISUAL_REVIEW` until a separately documented human review.

The six native-resolution output directories are under
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/outputs`
(`a1`, `a2`, `a3`, `b1`, `b2`, `b3`).  Each contains exactly 300 official
`.nii.gz` files and 582 finding channels.  Native shape, affine, finding-axis,
binary-`uint8`, and original-CT-header checks passed for every set.  The
machine-readable completion record is
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/completion.json`,
with the full output/hash manifest in
`/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/test_output_manifest.json`.
No test ground truth was read or created, so no test Dice or ranking claim is
made.  No submission ZIP was created.
