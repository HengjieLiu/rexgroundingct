---
created: 2026-07-25
updated: 2026-07-25
status: active
---

# DDP And Effective Batch Size

This note records the project convention for comparing single-GPU batch size,
gradient accumulation, and multi-GPU DistributedDataParallel (DDP) fine-tuning.

## Concepts

DDP means one training process per GPU. Each process owns a full copy of the
model, consumes its own local mini-batch, computes gradients, and then
all-reduces gradients with the other ranks before the optimizer step. After the
step, all ranks have the same model weights.

Effective global batch size is:

```text
global_batch = per_gpu_batch_size * world_size * gradient_accumulation
```

For the VoxTell ReXGroundingCT fine-tuning line:

```text
batch1 single GPU: per_gpu_batch=1, world_size=1, grad_accum=1 -> global_batch=1
DDP bs4:           per_gpu_batch=1, world_size=4, grad_accum=1 -> global_batch=4
1 GPU accum4:      per_gpu_batch=1, world_size=1, grad_accum=4 -> global_batch=4
```

Gradient accumulation is mainly for simulating a larger global batch when a
single GPU cannot hold the desired batch at once. DDP is mainly for distributing
the same optimizer update across multiple GPUs so the wall-clock time per
sample is lower.

## DDP Versus Gradient Accumulation

DDP with four GPUs and batch1 per GPU is close to one GPU with batch1 and
gradient accumulation 4 when these are all true:

- the same four scheduled samples are used for each optimizer update;
- the loss is scaled correctly before `backward()`;
- the model does not contain BatchNorm behavior that depends on local batch
  statistics;
- the LR scheduler, optimizer step counter, EMA, scaler, and checkpoint cadence
  advance once per optimizer update, not once per local micro-batch;
- stochastic choices are controlled by a materialized schedule or by rank-aware
  deterministic seeding.

They are still not perfectly identical in practice. DDP introduces all-reduce
communication, rank-specific execution order, and small floating-point
nondeterminism. Gradient accumulation keeps one process and has no cross-GPU
communication, but it takes roughly four forward/backward passes per update on
one GPU.

## Comparison Criteria

A fair comparison depends on the question.

Sample-matched asks whether seeing the same number of training patches is
enough:

```text
bs1 epoch100: 100 epochs * 100 updates/epoch * 1 sample/update = 10,000 samples
DDP bs4 epoch25: 25 epochs * 100 updates/epoch * 4 samples/update = 10,000 samples
```

This is sample-matched but not update-matched. The DDP run has only 2,500
optimizer updates and is only 25% through a 10,000-update LR horizon.

Update-matched asks whether the same number of optimizer steps is better with a
larger global batch:

```text
bs1 epoch100: 10,000 updates and 10,000 samples
DDP bs4 epoch100: 10,000 updates and 40,000 samples
```

This is update-matched but not sample-matched. It measures whether the larger
global batch and four times as many sampled patches improve the final recipe.

Wall-clock comparison asks how much useful validation performance is obtained
per hour. For challenge work this matters because fast iteration can beat a
theoretically cleaner but slow comparison.

## Project Schedule Rule

For materialized schedules, the global event stream is consumed by update and
rank:

```text
event_index = update * global_batch + rank * local_batch + local_offset
```

For DDP bs4 with local batch1 and no accumulation:

```text
rank0 update0 -> event 0
rank1 update0 -> event 1
rank2 update0 -> event 2
rank3 update0 -> event 3
rank0 update1 -> event 4
```

This keeps all ranks deterministic and makes resume after segmented validation
unambiguous.

## Recommendation For VoxTell ReXGroundingCT

Use single-GPU batch1 runs when testing risky modeling changes, because they are
simple to inspect and avoid distributed failure modes.

Use DDP batch4 when a recipe is stable and the question is whether more samples
per optimizer update improve accuracy or throughput. Report both:

- sample-matched checkpoints, such as DDP epoch25 versus single-GPU epoch100;
- update-matched checkpoints, such as DDP epoch100 versus single-GPU epoch100.

For segmented DDP runs, pause training at evaluation checkpoints, run validation
with all GPUs, then resume from the full optimizer/scaler checkpoint. This gives
clean val200 signals without concurrent eval competing with the DDP trainer.
