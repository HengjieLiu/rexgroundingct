---
created: 2026-09-09
updated: 2026-09-09
status: accepted_for_implementation
---

# Exp027 design decisions

This is the durable record of the user's pre-coding discussion. Implement the
four conditions below; final ranking and checkpoint selection belong to the
user. Code completion does not authorize a timing test or training launch.

| Question, in priority order | Accepted recommendation and rationale |
| --- | --- |
| 1. How should partial training annotations be supervised? | Ordinary Dice + BCE on released binary masks in all arms. Unlabeled real voxels count as background. Training annotations can omit instances; this does not establish incomplete boundaries. Defer partial-label masking and error/boundary weighting to separate experiments. |
| 2. How should validation be divided? | Patient-grouped halves A/B of the full 200-case cohort, seed 20260909. Search 5,000 partitions; balance 2a counts, scans, then volume quartile and entity-count strata. Freeze and hash the result. |
| 3. What is one example? | One CT and original 2a finding, its own logits and binary union of its instance IDs. Never merge separate same-category prompts or synthesize negative prompts. |
| 4. How much residual freedom? | Uncapped additive residual. Zero-initialize the final layer and penalize mean absolute residual with coefficient 0.005. The earlier ±2 tanh cap is superseded: it could hide fitting capacity by forbidding correction of confident errors. Stored base logits retain the existing [-30,30] clipping contract. |
| 5. What spatial support? | Native 192-cubed patches, 50% GT foreground / 25% base prediction / 25% random context. Empty predictions fall back to random sampling. Full preprocessed CT tiled inference, not prediction-bbox inference; preserve the baseline crop/restore contract. |
| 6. How should run 2 mix sources? | Exactly 50% original train / 50% validation A per 100-update block. Uniform finding sampling within sources prevents the small exhaustive subset from disappearing in simple concatenation. |
| 7. How much training and evaluation? | Equal updates and batch size across arms. One reporting epoch is 100 updates. The user chooses total updates/epochs and evaluation milestones after timings; 5,000 updates is not an accepted budget. Benchmark 100 updates per arm concurrently, then one complete 2a evaluation per arm. |
| 8. Which architecture and tile size? | Two-channel 3D CNN, 16 features, GN(4), GELU, four two-convolution residual blocks with dilations 1/2/4/1, zero-initialized one-channel head. User explicitly chose 192³, matching VoxTell; 96³ is superseded. |
| 9. Which preprocessing/base outputs? | Frozen Exp007 continuation 50 / absolute 150; native cached z-score CT and final blended logits. Reuse strict SideExp003 validation cache. Joint flips only; no intensity augmentation or resampling. Compare with the exact consumed cache baseline. |
| 10. Which loss/optimizer? | Dice + BCEWithLogits + 0.005 mean absolute residual, valid real voxels only, Dice epsilon 1e-6 including empty patches. AdamW lr 1e-4, wd 1e-4, betas .9/.999, eps 1e-8; constant LR initially, batch 1, FP16 AMP/FP32 losses, norm clipping 1.0. |
| 11. Which evaluation and reporting? | Threshold .5; all 69 validation 2a findings on 63 scans once per model/checkpoint, aggregated into A/B/full. Report paired Dice, hits, voxel precision/recall, edits, residuals and timing; label training exposure. No automatic ranking or best-checkpoint selection. |
| 12. Which execution/review policy? | Same initial weights, reproducible separate sampling streams, one independent model per GPU, synchronous train/eval barriers, resumable full state. All results remain pending user review. |

## Four conditions and interpretation

| Arm | Training source | A exposure | B exposure | Full validation |
| --- | --- | --- | --- | --- |
| run1_train | Original train 2a | Held out from refiner | Held out from refiner | Held out from refiner |
| run2_train_val_a | Train 2a + A, 50/50 | Training | Held out from refiner | Mixed |
| run3_val_a | A only | Training | Held out from refiner | Mixed |
| run4_val_all | A+B | Training | Training | Training |

The user retains run 4 as an empirical fitting-capacity diagnostic and will
interpret and rank all four runs. A failed fit requires checking convergence,
pipeline correctness and architecture/loss limitations. It is not a mathematical
upper bound on every possible residual method. Exp007 was already selected on
full val200; the new split does not erase that prior selection exposure.
Source comparisons also differ in data amount/distribution and base-model
training exposure, so they do not isolate annotation completeness causally.

## Subsequent accepted choices

- Implement and CPU-test first. Stop at `awaiting_timing_approval`.
- No GPU smoke, base-logit generation, benchmark, background launch poller or
  full training before explicit authorization.
- Benchmark all four arms concurrently after approval. Separate preparation,
  synthetic warm-up, 100 measured updates, and full 2a evaluation costs.
- After benchmarking stop at `awaiting_schedule_decision`; the user sets the
  full-run budget and validation cadence. Full runs start fresh by default.
- Dashboard is local `live_dashboard.md` plus `live_dashboard.png` (3×3 panels),
  updated every 30 seconds and on milestones, with JSON/CSV underlying records.
  The browser-page and TensorBoard alternatives were not selected.

## Reference evidence

- [Exp007](../007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/README.md)
- [Annotation evidence](../../docs/brainstorm/2026-07-23_rex_training_partial_instance_annotation_evidence.md)
- [Cache provenance](../../side_experiments/sideexp003_ensemble_method_hub/checkpoint_catalog.json)
- [Challenge protocol](https://rexrank.ai/ReXGroundingCT/challenge.html)

Audited metadata: train 2a has 1,120 findings / 801 scans / 741 filename-derived
patient IDs; val 2a has 69 findings / 63 scans / 61 patient IDs. Historical
checkpoint 2a Dice is 0.337641; strict cached inference is 0.337215 with 59/69
hits. Use the latter for paired comparisons.

## Numerical implementation correction after approved launch

The user requested diagnosis of the warm-up failure and comparison with prior
training's clipping. [The diagnosis](numerical_diagnosis.md) established a finite
loss with FP16 scaled-gradient overflow. Preserve LR 1e-4 and norm clipping 1.0.
Use the previous trainer's AMP skip/backoff behavior with bounded same-patch/RNG
retry and explicit actual-optimizer-call accounting. A skipped attempt is not
an update or a consumed finding. Log every retry; persistent invalid values
still stop the group. Warm-up restores the original scaler along with all
other pristine state. This corrects numerical handling without changing the
accepted data, model, loss, optimizer or timing schedule.

## Subsequent FP32 comparison authorization

After discussing the AMP motivation, the user requested the same benchmark in
FP32 after the FP16 trial finishes. Preserve the FP16 results and run all four
FP32 arms from the same initial weights and schedules in a separate runtime.
Use autocast/scaling disabled and TF32 acceleration disabled for this diagnostic.
Keep cached base-logit storage unchanged. Compare numerical failures, speed,
memory and A/B/full metrics; this does not select a final precision or training
schedule automatically. An explicitly authorized controller waits for complete
FP16 results and released GPUs before launching FP32.

## Full-run decisions after FP16/FP32 review

User accepted FP32 for all four full runs and as the future refinement default.
The chosen schedule is 100 updates per epoch for 100 epochs, with evaluation
every 10 epochs. Full training starts from pristine common weights, after the
entire 801-CT / 1,120-finding training 2a cache is complete and verified. Keep
all 63 validation CTs / 69 findings cached. The board publishes training updates
without waiting for epochs/barriers and provisional per-finding A/B/full scores
with matched-subset baselines, replacing them with each arm's final result.
The user explicitly selected provisional plus final validation reporting.
