---
created: 2026-07-23
updated: 2026-07-23
status: active
comparison: "five_epoch_batch_ddp"
---

# Five-Epoch Batch/DDP Comparison

This comparison tests whether larger single-GPU batch size or 2-GPU DDP is
worthwhile for the experiment 002 `192^3` z-score baseline.

## Epoch Unit

Each arm uses:

- `5` epochs
- `100` optimizer updates per epoch
- `500` optimizer updates total
- `grad_accum = 1`
- seed `20260723`
- a shared materialized train schedule for data-order reproducibility
- fixed LR with no learning-rate decay: `--lr-schedule fixed`

In DDP, `100` steps means `100` synchronized optimizer updates. It does not mean
`50` steps per GPU. Each DDP update consumes one local batch on each GPU, then
averages gradients.

## Arms

| Arm | Visible GPUs | Launch mode | Per-GPU batch | Global batch | Steps/epoch | Patches/epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `b1_gpu0` | `0` | single process | `1` | `1` | `100` | `100` |
| `b2_gpu1` | `1` | single process | `2` | `2` | `100` | `200` |
| `ddp2_b1_gpu23` | `2,3` | 2-process DDP | `1` | `2` | `100` | `200` |

The DDP arm intentionally uses per-GPU batch `1`, not `2`, so its global batch
matches the single-GPU batch-2 arm. DDP with per-GPU batch `2` would be global
batch `4` and should be treated as a separate optimization experiment.

## Reproducible Data Order

The launcher generates or reuses:

`/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val/config/train_schedule_seed20260723_5ep_100steps_gb2.jsonl`

This schedule has `1000` patch events:
`5 epochs * 100 optimizer updates * global batch 2`.

The schedule stores case/prompt/anchor choices and a per-event `patch_seed`.
The trainer computes the final patch start from that seed after case
preprocessing, so schedule generation does not preload CT volumes.

- `b1_gpu0` consumes the first `500` events.
- `b2_gpu1` consumes all `1000` events in pairs.
- `ddp2_b1_gpu23` consumes the same pairs as `b2_gpu1`, split by DDP rank.

The launcher also uses the fixed validation probe:

`configs/evaluation/rexgroundingct_val20_seed20260723.json`

## Run

Inside the VoxTell Docker container:

```bash
bash /workspace/scripts/rexgroundingct/run_002_five_epoch_batch_ddp_comparison.sh
```

By default, the three arms run in parallel on disjoint GPUs. Set
`RUN_PARALLEL=0` to run them sequentially.

Validation is optional and off by default. Set `RUN_VAL=1` to run the fixed
20-case quick/global validation probe after each arm:

```bash
RUN_VAL=1 VAL_LIMIT=20 \
  bash /workspace/scripts/rexgroundingct/run_002_five_epoch_batch_ddp_comparison.sh
```

## Outputs

The launcher writes one group directory:

`/mnt/shengdata1/hengjie/experiments/rexgroundingct/002_voxtell_text_ft_miccai_train_val/runs/five_epoch_batch_ddp_comparison_<timestamp>`

Each arm has its own run directory containing:

- `logs/train.log`
- `reports/training_metrics.json`
- `reports/training_report.md`
- `checkpoints/checkpoint_final.pth`
- `model/`

Compare:

- elapsed seconds;
- mean update seconds;
- peak allocated/reserved GiB;
- mean/later loss, interpreted cautiously because these are short runs;
- validation quick/global Dice if `RUN_VAL=1`.
