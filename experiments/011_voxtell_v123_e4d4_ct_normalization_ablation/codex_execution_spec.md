---
created: 2026-07-27
updated: 2026-07-27
status: active
experiment_id: 011_voxtell_v123_e4d4_ct_normalization_ablation
---

# Experiment 011 Execution Specification

## Objective

Compare native z-score, clipped z-score, and fixed linear HU preprocessing
under paired encoder/decoder LR profiles `1e-4/1e-4` and `1e-5/1e-4`.

The preparation phase generated and audited the two new caches and implemented
the experiment. The user explicitly authorized training at
`2026-07-28T07:28:16Z`, conditional on both caches and all preparation gates
passing. Those gates passed, and run group
`exp011_ct_norm_e4d4_20260728T085103Z` is active. The launcher guard remains in
place so a partial cache cannot start a trainer.

The e4d4 group completed. The user subsequently authorized a three-arm e5d4
rerun using only GPUs 0-2, synchronous val20 barriers, final val200, and an
atomic combined-report refresh after every completed evaluation barrier.

## Prior Evidence

- Experiment 006 established the cached-native v123 LR-ablation workflow.
- The fixed val200 HU audit found materialized HU values, substantial
  z-score variation, and sentinel/outlier cases that crop-to-nonzero does not
  remove.
- VoxTell's first encoder blocks use affine `InstanceNorm3d`, so robust
  clipping may matter more than a purely affine input remapping.
- Full evidence:
  `docs/voxtell/ct_hu_normalization_analysis.md`.

## Arms

| Arm | GPU | Cache | Encoder/decoder LR |
| --- | ---: | --- | ---: |
| `v123_e4d4_zscore` | 0 | `crop_zscore_native_v1` | `1e-4 / 1e-4` |
| `v123_e4d4_clip1024_zscore` | 1 | `crop_clip1024_zscore_native_v1` | `1e-4 / 1e-4` |
| `v123_e4d4_clip1024_linear` | 2 | `crop_clip1024_linear_native_v1` | `1e-4 / 1e-4` |
| `v123_e5d4_zscore` | 0 | `crop_zscore_native_v1` | `1e-5 / 1e-4` |
| `v123_e5d4_clip1024_zscore` | 1 | `crop_clip1024_zscore_native_v1` | `1e-5 / 1e-4` |
| `v123_e5d4_clip1024_linear` | 2 | `crop_clip1024_linear_native_v1` | `1e-5 / 1e-4` |

All arms initialize from public VoxTell v1.1. They do not continue from an
exp003, exp004, exp006, or later fine-tuned checkpoint.

## Data And Preprocessing Contract

Shared:

1. Read fixed CT NIfTIs using the established image orientation path.
2. Convert released targets from evaluator `(F,X,Y,Z)` layout to VoxTell
   `(F,Z,Y,X)` using the same affine-derived transform.
3. Crop the raw HU image to nonzero once and apply the identical crop to all
   targets.
4. Normalize the complete cropped image once.
5. Sample or slide native-resolution `192^3` windows from the cached image.

Normalization:

- Native z-score: no HU clipping; image padding `0`.
- Clipped z-score: clip to `[-1024,1024]`, then full-crop z-score; padding `0`.
- Fixed linear HU: `clip(HU,-1024,1024)/1024`; padding `-1`.
- Target padding is always zero.
- Patch-local z-score is forbidden.
- DICOM slope/intercept must not be applied again.

Both generated caches must contain 3,192 cases, 8,068 nonempty targets, and
zero geometry, target-hash, HU-header, or normalization audit failures relative
to `crop_zscore_native_v1`.

## Reproducibility

- Seed: `20260723`.
- Shared schedule: 10,000 events.
- Required schedule SHA256:
  `f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776`.
- Every arm consumes the identical case, prompt, target, and patch-start event
  sequence.
- Multi-finding cases use two positive prompts and one negative.
- One-finding cases use one positive prompt and two negatives.
- Every selected positive target is nonempty in its sampled patch.
- Fixed val20 and val200 JSON paths are passed directly; `--limit` is forbidden.

## Optimization And Loss

- 100 epochs, 100 optimizer updates per epoch, batch size 1, no DDP.
- SGD Nesterov, momentum `0.99`, weight decay `3e-5`.
- e4d4 encoder/decoder LR: `1e-4/1e-4`.
- e5d4 encoder/decoder LR: `1e-5/1e-4`.
- Warmup 100 updates, polynomial decay power `0.9`, gradient clipping `12`.
- Nonempty target: Dice plus weighted BCE.
- Empty target: weighted BCE only with loss weight `0.5`.
- BCE foreground/boundary/background weights: `1.0/1.5/0.5`.
- Boundary radius: `10`.
- Deep supervision: `[1, 1/2, 1/4, 1/8, 1/16]`, normalized.

## Checkpoints And Evaluation

Save immutable checkpoints at updates `500/2000/4000/6000/8000/10000` and a
rolling recovery checkpoint every 500 updates.

Only GPUs 0, 1, and 2 are exposed to the experiment. Training is split into
segments ending at epochs `5/20/40/60/80/100`. At each milestone:

1. All three training processes exit after atomically saving full state.
2. All three val20 evaluations run concurrently, one on each assigned GPU.
3. A global barrier waits for every val20 summary.
4. The canonical intermediate JSON and Markdown reports are refreshed.
5. All arms resume from their immutable milestone checkpoint.

Resume restores network weights, SGD momentum, AMP scaler, Python/NumPy/PyTorch
CPU and CUDA RNG streams, global update, polynomial-LR horizon, and
deterministic schedule cursor. Epoch-100 val200 runs on all three GPUs only
after epoch-100 val20 succeeds. Evaluation uses threshold `0.5`, shared locks,
and completed-output checks.

The e5d4 profile is controlled by a host-side milestone state machine. Each
training segment, evaluation barrier, and report update runs in a separate
short-lived container. Stable checkpoints, locks, completion markers, and
full-state resume make every stage idempotent. The combined report retains
e4d4, adds e5d4, and includes exp006 native e5d4 as a replication reference.

## Preparation Gate

Permitted before explicit training approval:

- cache unit and real-case tests;
- full cache generation and cross-cache audit;
- HU audit;
- imports, compilation, schedule/config checks;
- sample-only patch tests with no model or optimizer;
- generated-command and coordinator dry runs.

Not permitted before authorization:

- optimizer updates;
- batch-probe or one-update training smoke;
- trainer launch;
- validation inference or evaluation launch.

The launcher defaults to preparation-only and requires `START_TRAINING=1`.
The active run used guarded full mode after the acceptance checks in this
document passed. Full mode additionally requires exactly three visible devices
and at least `32 GiB` free on each before creating a run group.

## Acceptance

Preparation is complete when:

- both new cache roots have valid manifests and root `.complete` markers;
- cache and HU audit outputs reproduce required counts;
- all three experiment arms generate valid, identical-event sample tests;
- canonical config, launcher, coordinator, and report scaffold pass repository
  checks;
- no training or evaluation process has been started.
