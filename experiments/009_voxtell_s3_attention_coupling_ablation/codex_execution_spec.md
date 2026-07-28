---
created: 2026-07-27
updated: 2026-07-27
status: active
experiment_id: "009_voxtell_s3_attention_coupling_ablation"
---

# Codex Execution Spec

## Objective

Compare a same-source exp006 continuation baseline with three half-quarter S3
attention coupling strategies. All revised S3 arms use `1/2` and `1/4`; the
controlled variable is how S3 affects VoxTell features or logits.

## Prior Evidence

- Source model: experiment 006 `v123_cached_e5_d4` epoch 100.
- Source val20: Dice `0.390702`, hit rate `0.741935` (`23/31`).
- Source val200: Dice `0.324134`, hit rate `0.761155` (`290/381`).
- Original S3 can localize voxel-text matches but its suppressive gate cannot
  boost true-positive regions directly.
- The first all-scale exp009 smoke attempt showed that `1/1` S3 attention is
  activation-memory dominated at `192^3`: all three S3 arms OOMed during the
  first backward pass on 48 GiB GPUs while the baseline smoke passed. The active
  all-scale container `rex009_s3_attention_20260727T080246Z` was stopped before
  launching this revised plan, leaving its OOM runtime logs intact.
- Experiment 008 established the same-source continuation pattern, epoch-0
  equivalence gate, and synchronous milestone train/eval/report loop.

## Scope

In scope:

- Continue from exp006 `v123_cached_e5_d4` epoch-100 weights.
- Run four independent single-GPU batch1 arms immediately, even while other
  segmentation work is active.
- Use a no-impact S3 initialization and a linear coupling ramp to full strength
  by update 2000 / epoch 20.
- Evaluate val20 and write a report at epochs `0`, `5`, `20`, `40`, `60`,
  `80`, and `100`; evaluate val200 and report at epoch `100`.

Out of scope:

- Full-resolution `1/1` S3 attention; it is deferred until a memory-reduced
  attention implementation exists.
- Scale ablation; all revised S3 variants use the same `1/2 + 1/4` scale set.
- Prompt-specific adaptive S3 strength; defer to a follow-up experiment if
  exp009 is inconclusive.
- Silent fallback on OOM. If an arm OOMs, record the OOM and stop that arm.

## Inputs And Paths

- Canonical config:
  `configs/experiments/009_voxtell_s3_attention_coupling_ablation.json`
- Source model:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/006_voxtell_cached_native_v123_lr_ablation/runs/exp006_cached_native_lr_20260725T050001Z/v123_cached_e5_d4/model_epoch100`
- Native cache:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/crop_zscore_native_v1`
- Runtime root:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/009_voxtell_s3_attention_coupling_ablation`
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

| GPU | Variant | S3 scales | Coupling |
| ---: | --- | --- | --- |
| 0 | `baseline_cont100` | none | exp006 e5/d4 100-epoch continuation control |
| 1 | `s3v1_fixedrho_suppress_half_quarter` | `1/2`, `1/4` | original S3 fixed-rho suppressive gate |
| 2 | `s3v2_balanced_feature_half_quarter` | `1/2`, `1/4` | balanced boost/suppress feature modulation |
| 3 | `s3v3_logit_residual_half_quarter` | `1/2`, `1/4` | direct zero-init logit residual |

S3 attention map:

```text
A = sigmoid(scale * cosine(visual_voxel, text_prompt) + bias)
```

Coupling:

```text
s3v1: skip' = skip * ((1 - rho_eff) + rho_eff * A), rho_fixed = 0.25
s3v2: skip' = skip * (1 + alpha_eff * (2A - 1)), alpha_target = 0.25
s3v3: logit' = logit + beta_eff * R([logit, up(A_1/2), up(A_1/4)])
```

For all S3 arms:

```text
effective_strength = target_strength * min(1, global_update / 2000)
```

At update 0 the effective strength is zero, so the S3 arms should be
prediction-equivalent to the source model. S3 auxiliary supervision starts at
update 1 so the attention projections can learn while coupling ramps.

Training uses batch 1, no DDP, SGD Nesterov, encoder LR `1e-5`, non-encoder LR
`1e-4`, 100-update warmup, poly decay power `0.9`, weight decay `3e-5`, momentum
`0.99`, gradient clipping `12`, and 10,000 optimizer updates.

The main segmentation objective remains v123. The S3 auxiliary loss supervises
all nonempty positive prompt channels using pooled/dilated localization and
hard background margin at `1/2` and `1/4` only. The full-resolution sampled
foreground/hard-background alignment term is disabled for this revised run.

Each arm trains to one milestone, exits and releases CUDA, completes same-GPU
val20 plus the milestone report, then resumes full optimizer/scaler state.
Epoch 100 val20 is followed by val200. There is no evaluation sidecar.

## Smoke Gate

- Log GPU memory before launch; do not wait for other running segmentation
  jobs to finish.
- Materialize and audit epoch-0 models.
- Require epoch-0 val20 hit count to match the stored exp006 result exactly;
  permit Dice tolerance `1e-4`.
- Run one finite optimizer update for every arm.
- Run checkpoint save/load/resume smoke for every arm.
- Compile the modified training, inference, S3 materialization, and summary
  scripts before launch.

## Success Criteria

- Val20 summaries exist at epochs `0`, `5`, `20`, `40`, `60`, `80`, and `100`.
- Val200 summaries exist at epoch `100`.
- Milestone reports are refreshed after every val20 and final val200.
- Required diagnostics include attention mean/std/min/max per scale, S3
  effective strength, logit scale, bias, auxiliary losses, memory, and update
  time.
- No arm silently changes batch size, scales, or architecture after OOM.

## Verification

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  scripts/rexgroundingct/voxtell_s3_attention.py \
  scripts/rexgroundingct/materialize_s3_attention_model.py \
  scripts/rexgroundingct/summarize_009_results.py \
  scripts/rexgroundingct/train_text_conditioned_voxtell.py \
  scripts/rexgroundingct/run_voxtell_val_inference.py
bash -n scripts/rexgroundingct/run_009_s3_attention_coupling_ablation.sh
bash -n scripts/rexgroundingct/run_009_s3_attention_coupling_ablation_docker.sh
git diff --check
```

## Closeout Plan

Write a runtime comparison report and JSON summary, sync the repo-local
experiment index, and update `docs/current_status.md` only after evidence
changes the next model decision.
