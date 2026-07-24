---
created: 2026-07-23
updated: 2026-07-23
status: active
run_status: "probe_complete_train_smoke_complete"
---

# Single-GPU Probe Results

Runtime directory:

`/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val`

## Embedding Bank

The train+val prompt embedding bank was completed before probing:

- Path: `config/rex_text_embeddings.npz`
- Labels: `6467`
- Embedding shape: `(6467, 2560)`
- Dtype: `float16`
- Precompute wall time for the successful run: about `7 min 16 sec`

## Batch-Size Probe

Run:

`runs/codex_batch_probe_20260723T034249Z`

Probe settings:

- GPU: `0`
- Device: NVIDIA RTX 6000 Ada Generation
- Patch size: `192 x 192 x 192`
- Probe steps per batch size: `2`
- AMP: enabled
- Gradient accumulation: `1`
- Precomputed text embeddings: enabled

| Batch size | Status | Mean update seconds | Peak allocated GiB | Peak reserved GiB |
| --- | --- | ---: | ---: | ---: |
| `1` | ok | `5.085` | `25.875` | `26.154` |
| `2` | ok | `16.375` | `38.378` | `42.064` |
| `3` | OOM | n/a | `44.373` | `46.049` |

Largest fitting single-GPU batch size is `2`.

Practical recommendation for the first baseline is batch size `1`, because it is
faster per patch and leaves much more memory headroom. Batch size `2` fits but
is slower per patch in this probe and leaves little margin for augmentation or
validation-side memory variance.

## Train Smoke

Run:

`runs/codex_train_smoke_20260723T034434Z`

Smoke settings:

- Epochs: `1`
- Steps per epoch: `2`
- Batch size: `1`
- Checkpoint every update: yes, for export-path testing

Results:

- Completed optimizer updates: `2`
- Elapsed seconds: `21.247`
- Mean update seconds including checkpoint/export overhead: `10.624`
- Peak allocated GiB: `25.875`
- Peak reserved GiB: `26.154`
- Final checkpoint written.
- Inference-compatible `model/` directory written.

The smoke was intentionally too short for model quality. It only verifies that
sampling, loss, optimizer step, checkpointing, and model export execute.

## DDP Implication

For this experiment, one epoch means `100` optimizer updates. With batch size
`1`, the first single-GPU baseline sees `100` case-patches and `300`
prompt-target channels per epoch.

Four-GPU DDP with per-GPU batch size `1` would see `400` case-patches and
`1200` prompt-target channels per epoch if `steps_per_epoch` remains `100`.
That is worthwhile for throughput if we accept the larger global batch and
larger per-epoch sample count. For a fixed sampled-patch budget, reduce
`steps_per_epoch` by the number of GPUs.

Do not split one `192^3` patch across GPUs. The useful multi-GPU path is normal
DDP data parallelism, one or more complete case-patches per GPU.
