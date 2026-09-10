# CPU allocation follow-up

During the authorized baseline, all four GPU workers spent approximately 90%
CPU in kernel code, with no disk I/O or major faults. Five-second host counters
showed 524 failed direct compactions, zero successes and 168 huge-page fallbacks.
THP enabled and defrag policies are both `madvise`; NumPy 2.4.4 enables huge-page
advice by default. This is a host fragmentation/allocator interaction hypothesis,
not evidence that GPU inference itself is slow.

A 256 MiB synthetic float32 allocation/fill/sum, alternating process-local NumPy
advice on/off, measured:

| Advice | Wall seconds | System CPU seconds |
|---|---:|---:|
| on | 15.2156 | 13.2062 |
| off | 0.0740 | 0.0422 |
| on | 16.2870 | 14.1071 |
| off | 0.0728 | 0.0488 |

All four sums were identical (67108864). No host sysctl, global THP setting,
existing process, or production job was modified.

`profile_test300_nohuge.py` waits for the original 4/8/16 CPU baseline to finish,
then repeats only CPU replay at 4/8/16 with `NUMPY_MADVISE_HUGEPAGE=0`. It uses the
same functions, retained crop inputs and reference hashes; no extra inference.
Outputs use separate `nohuge_cpu4`, `nohuge_cpu8`, `nohuge_cpu16` namespaces.
Reserve checks include three extra native-float32 output/staging sets. The
follow-up does not resume production Waves 2–5. Compare wall times, stage times,
exact values/masks and kernel counters before recommending a production change.

## Real-case validation

The largest-output iso07 case `train_12972_a_2.nii.gz` reproduced native array
SHA `e07700ef593eb9c447db36b228c2e4931cda25e9635a826c351cecb0e3e49626` exactly.
Restoration fell from 76.36 to 8.26 seconds; orientation fell from 359.84 to
17.03 seconds. The replay used one numerical thread versus four in reference.
Reading/hashing the cold crop took 26.66 s, local writing 23.31 s, and full CPU
restoration/staging 79.98 s. Publication and strict validation took another
25.94 s (dtype scan was outside that last timer). The sample supported float16
with exact storage masks; this does not establish full-cohort iso07 dtype.

The default public embedding bank contains no exact lowercased matches for
582 test prompts or 381 val prompts. Text embeddings cost 222–268 s across four
cases per model, including recurring backbone CPU/GPU transfers. That cost is
separate from the demonstrated NumPy allocation bottleneck; its contribution to
historical val/test differences is not established by this benchmark.
