---
created: 2026-09-09
updated: 2026-09-09
status: reproduced_and_fixed
---

# Exp027 warm-up gradient overflow

The failed benchmark had a finite forward loss. Its fifth synthetic warm-up
update overflowed two of the final convolution's sixteen weight gradients under
FP16 AMP scaling. No measured training updates or evaluations had started.
This was reproduced with the exact 192³ architecture, seed, optimizer, Docker
image and default AMP scale. The diagnosis uses synthetic inputs, so the failure
does not implicate the cached CTs, logits or incomplete training annotations.

## Settings compared with previous training

| Setting | Exp027 residual refiner | Exp007 continuation used for the frozen base |
| --- | --- | --- |
| Optimizer | AdamW | SGD with Nesterov, momentum 0.99 |
| Learning rate | 1e-4 for the whole refiner | Encoder 1e-5, decoder 1e-4 |
| LR schedule | Constant during timing | 100-update LR warm-up, polynomial decay with power 0.9 over 10,000 updates |
| Gradient clipping | Global L2 norm 1.0 | Global L2 norm 12.0 |
| AMP scale initially | 65,536 | Default GradScaler initialization |
| Overflow behavior before this fix | Clipping raised before GradScaler could recover | GradScaler could skip the optimizer step and decrease its scale |

Exp026 also uses encoder LR 1e-5, decoder LR 1e-4 and clipping at 12, with fixed
LR and no LR warm-up. LR values across AdamW and SGD are not equivalent step
sizes. The refiner already clips more tightly than these previous runs.

Evidence: the Exp007 continuation's runtime `ddp_bs4/run_manifest.json`, canonical
Exp007/026/027 configs, and the historical trainer at commit
`658d7b703d601b9fc99a2b4b1bcd3380a218460c` (the recorded Exp007 source revision).
The installed PyTorch version is `2.8.0+cu126`; its GradScaler defaults to
initial scale 65,536, backoff 0.5 and growth interval 2,000.

## Reproduction

| Diagnostic | Outcome |
| --- | --- |
| Original strict update, initial scale 65,536 | Four successful synthetic updates; fifth fails with loss 1.636476874 and two infinite `head.weight` gradients |
| Strict clipping at 1 or at 12 on those gradients | Both reject the same non-finite gradients |
| AMP skips the failed attempt, halves scale to 32,768, retries the same patch | Fifth optimizer update succeeds; loss stays 1.636476874, unscaled gradient norm 2.144594 |
| Initial scale 16,384 as a diagnostic control | Five successful synthetic updates with finite losses/gradients |

The recovered head gradient has maximum absolute value 1.0673828125. Its
65,536-scaled magnitude exceeds FP16's finite range. Gradient clipping occurs
after backward and unscaling, so changing the clipping threshold cannot repair
an overflow that already occurred. Recovery at the same weights, data and LR
establishes AMP gradient overflow as the immediate cause; LR reduction is not
needed to resolve this reproduced failure. This does not establish stability
over an entire real-data run.

## Implementation correction

Retain AdamW LR 1e-4, clipping at 1.0, FP16 forward/FP32 losses, the default AMP
scale and all accepted method settings. Check gradients after unscaling. For
an AMP overflow, let GradScaler skip the optimizer call and lower its scale,
then retry the same patch with the same RNG state. Check finite gradients and
clip strictly before any real optimizer update. A post-step hook verifies
exactly one actual optimizer call before the schedule/history cursor advances.

Record each overflow's scale, affected parameters and finite loss in the worker
log and successful-update history. Retry time is included in the measured step.
Stop on non-finite forward loss, non-finite gradients without AMP, non-finite
gradients at scale <= 1, or more than 16 scale backoffs for one update. Never
replace non-finite values or silently count skipped attempts as updates.

Synthetic warm-up now records its successful updates and restores pristine
model, optimizer, scaler and RNG state even if warm-up fails. Warm-up scale
calibration is discarded as required by the benchmark design. Each arm retains
its own subsequent scaler state in resumable checkpoints.

## Verification and artifacts

- 17 contract tests and 15 CPU PyTorch tests passed, including injected overflow,
  exact optimizer-call counting, same-patch/RNG replay, persistent-invalid stop,
  checkpoint continuation with a backed-off scaler, and warm-up restoration.
- A real 192³ GPU check passed using the production update/warm-up functions.
  It restored pristine state after five successful warm-up updates, then took
  five disposable synthetic optimizer updates. Its final weights exactly
  matched the independent adaptive-scaling diagnostic control.
- Repository workflow and whitespace checks passed. Existing unrelated
  experiment-index warnings remain unchanged.

Raw artifacts, scripts and logs are under
[`runtime/diagnostics/amp_20260910`](runtime/diagnostics/amp_20260910):
[`diagnostic.json`](runtime/diagnostics/amp_20260910/diagnostic.json) and
[`production_fix_validation.json`](runtime/diagnostics/amp_20260910/production_fix_validation.json).
These disposable synthetic updates are excluded from experimental results and
timing. The failed attempt and cache producer provenance are preserved before
retrying the previously authorized four-arm timing benchmark.
