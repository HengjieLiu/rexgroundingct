---
created: 2026-09-06
updated: 2026-09-06
status: active
experiment_id: "026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr"
---

# Codex Execution Spec

## Objective

Measure whether removing exp007's polynomial learning-rate decay improves the
cached-native v123 e5/d4 DDP batch4 recipe when initialized directly from the
public VoxTell v1.1 checkpoint.

## Prior Evidence

- Exp006's strongest public-start arm used encoder LR `1e-5` and decoder LR
  `1e-4`.
- Exp007 used the same differential LRs with a 100-update warmup and poly
  power `0.9`; its epoch-100 val200 result was Dice `0.3310` and hit rate
  `0.7585`.
- The native cache manifest is verified at
  `fd6787a18b9f12ef68035bd9a2dcca30c56322b3957e073bf513cbd3e71af4c3`.
- The reused 40,000-event DDP schedule is verified at
  `acdfed8dd8d9908e9ea33d6790d47f8f7e3cbeed4bad326274d115d38e96a0aa`.

## Scope

In scope:

- Public VoxTell v1.1 initialization.
- Exp007-compatible cached-native v123 preprocessing, sampler, loss, model,
  and DDP batch4 setup.
- Fixed encoder/decoder LRs with no warmup or LR decay.
- Full val200 barriers at epochs `5, 10, ..., 100`.

Out of scope:

- Fine-tuned initialization from exp006/exp007.
- Preprocessing, architecture, loss, prompt-policy, or batch-size changes.
- Probability-map persistence.
- Test-set inference or submission packaging.

## Inputs And Paths

- Canonical config:
  `configs/experiments/026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr.json`
- Metadata: `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- CT root: `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- Segmentation root: `/data/hengjie/datasets/rexgroundingct/segmentations`
- Runtime root: `/mnt/shengdata1/hengjie/experiments/rexgroundingct/026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr`
- Fixed val200: `configs/evaluation/rexgroundingct_val200_seed20260723.json`

## Data And Preprocessing Contract

- Classification: cached-equivalent to the VoxTell native baseline.
- Cache: `crop_zscore_native_v1`, shared under `/mnt/shengdata1`.
- Orientation: existing VoxTell/nnU-Net orientation correction and evaluator
  export path.
- Normalization: crop-to-nonzero, full cropped-volume z-score once, then native
  `192^3` sampling.
- Resampling: none.
- Sampler: Exp007's verified 40,000-event v123 positive-crop schedule.

## Method

Use four DDP processes, local batch 1, gradient accumulation 1, and effective
global batch 4. Train from the public model directory resolved by VoxTell's
`download_voxtell_model` helper. Use SGD Nesterov, momentum `0.99`, weight decay
`3e-5`, gradient clipping `12`, and v123 empty-target weighted-BCE loss.

Each five-epoch segment advances 500 optimizer updates. At the segment end,
save the immutable full-state checkpoint, release the trainer GPUs, evaluate all
200 fixed validation cases across four GPUs, and resume from that checkpoint.

The host launch is scheduled two hours after preparation with the user-level
systemd one-shot command documented in the handoff. Existing GPU containers are
not treated as a blocking condition.

## Evaluation Execution Contract

- Milestones: epochs `5, 10, ..., 100`; updates `500, 1000, ..., 10000`.
- Evaluation: fixed val200 JSON, threshold `0.5`, 200 cases / 381 findings.
- Resource policy: synchronous barrier using all four GPUs.
- Resume state: model, optimizer, AMP scaler, global update, sample schedule,
  and distributed RNG state from the full checkpoint.
- Locking: per-evaluation `.eval.lock`; completed evaluations are reused and
  incomplete predictions are safely resumed.

## Smoke Gate

After the two-hour delay and before full training:

- Validate the materialized schedule and cache.
- Run the sample-test path.
- Run DDP update-2/update-4 resume smoke.
- Run one-case inference from the smoke checkpoint.

## Success Criteria

- 10,000 optimizer updates complete.
- Twenty immutable milestone checkpoints exist.
- Twenty full val200 summaries complete with 200 predictions each.
- Training metrics show constant encoder/decoder LRs.
- Public checkpoint provenance and runtime config hashes are recorded.

## Verification

```bash
bash -n scripts/rexgroundingct/run_026_ddp_bs4_fixed_lr.sh
bash -n scripts/rexgroundingct/run_026_ddp_bs4_fixed_lr_docker.sh
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/rexgroundingct/common.py scripts/rexgroundingct/summarize_026_results.py
python scripts/rexgroundingct/check_experiment_consistency.py --experiments 026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr
git diff --check
```

## Closeout Plan

After completion, generate the fixed-LR report, run the full experiment sync
and consistency checks, and update `docs/current_status.md` only if the result
changes the preferred VoxTell training schedule.
