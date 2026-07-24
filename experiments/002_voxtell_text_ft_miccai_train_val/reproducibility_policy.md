---
created: 2026-07-23
updated: 2026-07-23
status: active
seed: 20260723
---

# Experiment 002 Reproducibility Policy

This policy makes the first fine-tuning comparisons repeatable enough to debug
training behavior and compare batch/DDP arms without hidden data-order changes.

## Fixed Seed

Use seed `20260723` unless a later named experiment intentionally changes it.
The seed is passed to Python, NumPy, PyTorch, and the experiment 002 sampler.

The plain seeded sampler is repeatable for the same process layout, but it is
not sufficient for comparing batch size or DDP, because different batch sizes
and ranks consume RNG draws in different orders. For comparable training runs,
use a materialized sample schedule.

## Fixed Val20 Probe

The standard quick validation probe is:

`configs/evaluation/rexgroundingct_val20_seed20260723.json`

Generation rule:

```bash
python scripts/rexgroundingct/prepare_fixed_probe_subset.py \
  --split val \
  --seed 20260723 \
  --size 20 \
  --output-json configs/evaluation/rexgroundingct_val20_seed20260723.json \
  --manifest-json configs/evaluation/rexgroundingct_val20_seed20260723.manifest.json
```

The script loads the validation split from
`/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`, applies
`np.random.default_rng(20260723).permutation(200)`, and takes the first `20`
indices. Do not replace this with `--limit 20`, because that means metadata
order, not the fixed seeded probe.

## Training Schedule

For comparable training runs, generate a JSONL patch-event schedule and pass it
to the trainer with `--sample-schedule`.

For the 5-epoch batch/DDP comparison:

```bash
python scripts/rexgroundingct/prepare_training_schedule.py \
  --seed 20260723 \
  --events 1000 \
  --output-jsonl /mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val/config/train_schedule_seed20260723_5ep_100steps_gb2.jsonl \
  --manifest-json /mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val/config/train_schedule_seed20260723_5ep_100steps_gb2.manifest.json
```

The `1000` events equal `5 epochs * 100 optimizer updates * global batch 2`.
Each event records the case, prompt slots, positive/negative target mapping,
foreground-anchor metadata, patch size, source split index, and a per-event
`patch_seed`. The final patch start is computed deterministically from
`patch_seed` after the case is loaded, reoriented, cropped, and normalized.

Consumption rule:

- `b1_gpu0`: consumes events `0..499`, one patch per optimizer update.
- `b2_gpu1`: consumes events `0..999`, two patches per optimizer update.
- `ddp2_b1_gpu23`: consumes the same pairs as `b2_gpu1`; rank 0 consumes the
  first event in each pair and rank 1 consumes the second.

This keeps the single-GPU batch-2 and DDP global patch streams identical. The
batch-1 arm sees the prefix because it has half the global patch budget at the
same number of optimizer updates.

## Fallback Sampling

The unresolved paper-alignment issue remains labeled as
`confusion_not_solved_but_moving_forward`: multi-finding cases use `2` positive
and `1` negative prompt slot; one-finding cases use `1` positive and `2`
negative prompt slots.

Additional implementation rule for one-finding cases: force foreground anchoring
on the sole positive target so the positive target is non-empty in the sampled
patch whenever the released full-volume mask is non-empty. This makes fallback
samples less likely to become pure-negative training examples.
