---
created: 2026-07-26
updated: 2026-07-27
status: complete
experiment_id: "008_voxtell_dual_branch_proposal_refinement_ablation"
---

# Codex Execution Spec

## Objective

Determine whether an inclusive proposal decoder can improve VoxTell
segmentation through soft multiscale guidance, and test whether shared fusion,
branch-specific fusion adaptation, modest precision pressure, or joint
guide-gradient flow is preferable.

## Runtime Amendment: Pause After Epoch 80

On 2026-07-26, the active run was amended to pause after all four epoch-80
val20 evaluations rather than immediately continue to epoch 100.

- Preserve the original 10,000-update LR horizon; do not redefine the run as an
  8,000-update schedule.
- Save and validate each immutable update-8000 checkpoint.
- Require network weights, optimizer momentum, AMP scaler, global update,
  schedule cursor, and deterministic data order to remain resumable.
- Complete final and proposal val20 evaluation for all four epoch-80 models.
- Consume zero events from updates 8001-10000 while paused.
- Defer epoch-100 val20 and val200 until an explicit resume.
- Write an ETA/status record every 600 seconds and a dedicated epoch-80 pause
  report.

The normal launcher now supports `PAUSE_AFTER_EPOCH=80`. The already-running
launcher predates that option, so per-arm
`control/pause_before_update_010000` markers provide the equivalent boundary
before any epoch-100 model/CUDA initialization or sample consumption.

## Runtime Completion

The four arms resumed on 2026-07-27 from their immutable update-8000
checkpoints after checkpoint hashes and resume state were audited.

- Resume used `--resume-checkpoint`, preserving SGD momentum, AMP scaler,
  optimizer parameter groups, and the deterministic schedule cursor.
- The first resumed encoder LR was `2.37e-6`, matching the next point in the
  original 10,000-update poly schedule. Warmup was not restarted.
- All arms completed update `10000`; final checkpoints report epoch `100`,
  global update `10000`, encoder/decoder base LRs `1e-5/1e-4`, and terminal LR
  `0`.
- Epoch-100 val20 final/proposal evaluation and fixed val200 evaluation
  completed synchronously for all arms.
- Canonical results are recorded in
  `experiments/008_voxtell_dual_branch_proposal_refinement_ablation/report.md`.

## Prior Evidence

- Source model: experiment 006 `v123_cached_e5_d4` epoch 100.
- Source val20: Dice `0.390702`, hit rate `0.741935` (`23/31`).
- Source val200: Dice `0.324134`, hit rate `0.761155` (`290/381`).
- Experiment 007 provides a verified 40,000-event deterministic schedule whose
  first 10,000 events reproduce the schedule consumed by experiment 006.
- Technical method note:
  `docs/voxtell/asymmetric_proposal_refinement.md`.
- Exact variant comparison and training reference:
  `docs/voxtell/exp008_variant_reference.md`.

## Inputs And Paths

- Canonical config:
  `configs/experiments/008_voxtell_dual_branch_proposal_refinement_ablation.json`
- Source model:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/runs/exp006_cached_native_lr_20260725T050001Z/v123_cached_e5_d4/model_epoch100`
- Native cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1`
- Runtime root:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/008_voxtell_dual_branch_proposal_refinement_ablation`
- Fixed val20:
  `configs/evaluation/rexgroundingct_val20_seed20260723.json`
- Fixed val200:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`

## Data And Preprocessing Contract

- Classification: cached-equivalent to the VoxTell native baseline.
- Orientation: existing validated LPS/RAS target handling.
- Normalization: crop-to-nonzero, then z-score the complete cropped volume once.
- Sampling: native-resolution `192^3` foreground-positive patches.
- Prompt policy: three slots; multi-finding uses two positive plus one negative,
  one-finding uses one positive plus two negatives.
- Schedule: reindex source events `10000-19999` while preserving original event
  indices; all arms consume the same 10,000 events.

## Method

GPU assignment:

| GPU | Variant |
| ---: | --- |
| 0 | `v1_sharedfusion_softguide` |
| 1 | `v1_dualfusion_softguide` |
| 2 | `v2_dualfusion_precision` |
| 3 | `v3_dualfusion_softguide_joint` |

All arms copy the source decoder into proposal and refinement paths. New guide
and residual-adapter outputs start at zero. The final v123 objective remains at
weight 1.0. Proposal v123 plus recall-Tversky is weighted by 0.5. Only v2 adds
precision-Tversky with weight 0.1.

Training uses batch 1, no DDP, SGD Nesterov, encoder LR `1e-5`, non-encoder LR
`1e-4`, 100-update warmup, poly decay power `0.9`, weight decay `3e-5`, momentum
`0.99`, gradient clipping `12`, and 10,000 optimizer updates.

Each arm is supervised independently. It trains to one milestone, exits and
releases CUDA, completes same-GPU val20, then resumes full optimizer/scaler
state. Epoch 100 val20 is followed by val200. There is no evaluation sidecar.

## Smoke Gate

- Materialize and audit all four epoch-0 models.
- Confirm bitwise source-to-proposal and source-to-final logit equivalence at
  all five scales on one fixed `192^3` patch for every variant.
- Require epoch-0 val20 hit rate and hit/finding counts to match the stored
  Exp006 result exactly. Permit an absolute Dice difference of at most `1e-4`
  because independent mixed-precision sliding-window runs on separate GPUs
  show small aggregate variation even when same-patch logits are bitwise
  identical; record every observed delta.
- Test dual output shapes, loss direction, empty targets, detach/joint
  gradients, orientation export, and checkpoint resume.
- Run one finite optimizer update for every variant.
- Benchmark inference window batch sizes `1/2/4` on the joint variant; use the
  largest common passing size under the recorded speed, memory, and output
  agreement limits.
- Enable decoder activation checkpointing for every arm only if the common
  batch-1 training smoke otherwise exceeds GPU memory.

## Success Criteria

- Epoch-0 val20 final metrics match the source.
- Every arm completes updates `2000/4000/6000/8000/10000`.
- Final and proposal val20 summaries exist at epochs `20/40/60/80/100`.
- Final val200 exists at epoch 100.
- No selected positive target is empty.
- No arm trains while its GPU is evaluating.
- The report includes loss components, branch diagnostics, memory, time, and
  the missing single-branch continuation-control limitation.

For the current pause amendment, interim acceptance is:

- all four update-8000 checkpoints are resumable;
- val20 final/proposal summaries exist at epochs `0/20/40/60/80`;
- all four arms report paused before target update 10000;
- the epoch-80 pause report records the complete trajectory and resume command.

## Verification

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/rexgroundingct/*.py
bash -n scripts/rexgroundingct/run_008_dual_branch_ablation.sh
bash -n scripts/rexgroundingct/run_008_dual_branch_ablation_docker.sh
python scripts/rexgroundingct/check_experiment_consistency.py
git diff --check
```

## Closeout

Write a runtime comparison report and JSON summary, sync the repo-local
experiment index, and update `docs/current_status.md` only after evidence
changes the next model decision.
