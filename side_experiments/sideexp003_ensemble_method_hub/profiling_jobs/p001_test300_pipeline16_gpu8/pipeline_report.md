# Test300 pipeline diagnosis and CPU benchmark

All 16 inference jobs and six 16-case CPU comparisons passed. Production Waves 2–5 remain held.

The demonstrated bottleneck is large-array allocation with NumPy huge-page advice on this host. Kernel compaction, rather than GPU segmentation computation alone, caused long CPU stalls. A 256 MiB allocation improved from 15–16 s to 0.073–0.074 s when advice was disabled locally. The real largest iso07 output preserved its exact native float32 hash while restoration plus orientation fell from 436 s to 25 s.

| NumPy huge-page advice | CPU workers | Total min | Restore/stage min | Precision scan min | Publish/validate min | Max active | Peak worker GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| on | 4 | 14.62 | 12.28 | 1.14 | 1.17 | 4 | 15.96 |
| on | 8 | 14.44 | 11.96 | 1.44 | 1.03 | 8 | 15.94 |
| on | 16 | 12.08 | 10.11 | 0.97 | 0.98 | 16 | 15.88 |
| off | 4 | 4.80 | 2.56 | 1.01 | 1.21 | 4 | 15.96 |
| off | 8 | 5.95 | 3.12 | 1.03 | 1.78 | 8 | 15.95 |
| off | 16 | 5.31 | 3.08 | 0.71 | 1.50 | 16 | 15.94 |

Fastest measured CPU replay: **4 workers with `NUMPY_MADVISE_HUGEPAGE=0`**. The maximum-memory scheduler retains 64 GiB available RAM. Each worker uses one numerical thread. Warm-input preparation is excluded from these wall times and recorded separately.

For a production continuation, carry the process-local allocation setting into both GPU workers and CPU workers, then use a bounded continuous CPU queue rather than waiting for 16 GPU outputs. Keep all inference/geometry/storage checks. The replay measures the CPU pool; it does not prove a full-pipeline overlap speedup. Prompt embedding generation and recurring CPU/GPU backbone transfers are another substantial cost and should be evaluated separately before changing their precision or cache contract.

| Reference stage (summed across 16 jobs) | Minutes |
|---|---:|
| input_total | 1.17 |
| embeddings_child | 16.45 |
| cuda_network | 9.84 |
| restore_total | 11.26 |
| orientation_total | 21.81 |
| finite_range_clip | 0.19 |
| local_write | 3.01 |
| local_array_hash | 0.28 |
| diagnostic_spool | 3.72 |

Network and embedding measurements are child intervals inside crop prediction; do not add them to their parent. Diagnostic spool time is extra benchmark work. Shared publication and precision scans are in the per-case CSV and comparison records.

Conditional planning envelope for Waves 2–5: **31.1–63.3 hours** with the observed text-embedding cost. This retains the measured unfixed GPU/text phases. It is neither a measured runtime for the modified GPU path nor a statistical confidence interval. Replace it using retained GPU smoke timings before treating it as a production ETA.

- Four-case architectural samples proxy Waves 3–5.
- Retains measured recurring text-embedding cost; no embedding optimization assumed.
- One inference pass: GPU/CPU overlap and helper-allocation improvement are unverified.
- Native output bytes scale CPU work; publication is added as a final barrier.
- Warm small-sample I/O can understate full-cache storage cost.

Validated contracts: native float32 hash and CT geometry parity, numeric finding ordering, per-checkpoint sample dtype agreement, finite/clipped values, exact threshold-zero masks after publication, atomic writes and hashes. No labels or Dice, no extra model inference, no production cache publication, no global THP changes. All sampled checkpoints permitted float16; full-cohort fallback remains undecided.

Reproduction: `python side_experiments/sideexp003_ensemble_method_hub/profile_test300.py prepare`, then `run`; arm `profile_test300_nohuge.py run` for the CPU-only follow-up and use `profile_test300_report.py --wait` for this report. Existing job IDs are immutable; inspect preserved records before any recovery.

## Recorded system evidence

| Mode | Kernel share of worker CPU | Host I/O wait | Peak summed RSS GiB |
|---|---:|---:|---:|
| cpu4 | 86.7% | 0.2% | 38.2 |
| cpu8 | 92.6% | 0.2% | 60.3 |
| cpu16 | 90.8% | 0.3% | 69.3 |
| nohuge_cpu4 | 21.9% | 0.5% | 36.0 |
| nohuge_cpu8 | 23.4% | 1.0% | 43.1 |
| nohuge_cpu16 | 25.1% | 1.5% | 73.2 |

GPU zero-utilization fractions during measured case intervals (including diagnostic spooling): rank 5: 75.4%, rank 6: 80.8%, rank 7: 84.3%, rank 8: 83.5%.

The replay driver serializes the four per-checkpoint precision scans; this is a measured limit of this harness, not a claim that the original four GPU workers scanned serially. Keep per-checkpoint dtype agreement while distributing these scans in a future coordinator. Dual-branch inference also converts the unused proposal output to CPU before selecting the final branch; removing that transfer is a separate, untested optimization.

Each concurrency setting was run once, in fixed order. Allocation history, NUMA placement and shared I/O can affect the ordering. The repeated synthetic on/off control and exact-hash real-case replay provide the causal allocation evidence.

- Host VM and iowait counters include unrelated processes.
- GPU utilization is sampled during case intervals and includes diagnostic spooling.
- RSS sums can double-count shared mappings; worker peak RSS is also reported separately.
