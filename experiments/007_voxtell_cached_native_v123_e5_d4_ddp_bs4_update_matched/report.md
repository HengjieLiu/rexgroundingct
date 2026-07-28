# Experiment 007 DDP Batch4 Update-Matched Report

Run group: `exp007_full_20260725T231624Z`

## Val200 Progress

| Checkpoint | Comparison role | Dice | Hit rate | Hits / findings |
| ---: | --- | ---: | ---: | ---: |
| 25 | sample-matched to exp006 epoch100 | 0.3091 | 0.7454 | 284 / 381 |
| 50 | intermediate | 0.3333 | 0.7559 | 288 / 381 |
| 75 | intermediate | 0.3318 | 0.7612 | 290 / 381 |
| 100 | update-matched to exp006 epoch100 | 0.3310 | 0.7585 | 289 / 381 |

## References

| Model | Dice | Hit rate | Hits / findings |
| --- | ---: | ---: | ---: |
| exp006_v123_cached_e5_d4_epoch100 | 0.3241 | 0.7612 | 290 / 381 |
| exp003_v123_epoch100 | 0.2833 | 0.6903 | 263 / 381 |

## Training

- Status: `completed`
- Completed updates: `10000` / `10000`
- World size: `4`
- Per-GPU batch size: `1`
- Gradient accumulation: `1`
- Last loss: `0.2207`
- Final segment seconds/update: `5.689`

## Training Time

Training time excludes validation pauses between exp007 segments.
Model-level training time uses training-script elapsed timers. Checkpoint rows use cumulative update timers plus checkpoint mtimes, because exp006 did not emit separate elapsed metrics at every checkpoint.

| Model | Global batch | Updates | Effective samples | Training time | Sec/update | Samples/sec | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp007_ddp_bs4_e5_d4 | 4 | 10000 | 40000 | 15:57:35 | 5.745 | 0.696 | 30.44 |
| exp006_bs1_e5_d4 | 1 | 10000 | 10000 | 14:00:27 | 5.043 | 0.198 | 25.88 |

## Checkpoint Training Time

| Model | Epoch | Update | Cumulative update-timer time | Cumulative sec/update | Checkpoint saved |
| --- | ---: | ---: | ---: | ---: | --- |
| exp007_ddp_bs4_e5_d4 | 25.0 | 2500 | 04:03:27 | 5.843 | 2026-07-25 20:21:27 PDT |
| exp007_ddp_bs4_e5_d4 | 50.0 | 5000 | 08:01:49 | 5.782 | 2026-07-26 00:38:23 PDT |
| exp007_ddp_bs4_e5_d4 | 75.0 | 7500 | 11:58:09 | 5.745 | 2026-07-26 04:54:07 PDT |
| exp007_ddp_bs4_e5_d4 | 100.0 | 10000 | 15:54:29 | 5.727 | 2026-07-26 09:09:23 PDT |
| exp006_bs1_e5_d4 | 5.0 | 500 | 00:30:53 | 3.706 | 2026-07-24 22:33:04 PDT |
| exp006_bs1_e5_d4 | 20.0 | 2000 | 02:18:52 | 4.166 | 2026-07-25 00:22:16 PDT |
| exp006_bs1_e5_d4 | 40.0 | 4000 | 05:00:40 | 4.510 | 2026-07-25 03:05:37 PDT |
| exp006_bs1_e5_d4 | 60.0 | 6000 | 07:57:40 | 4.777 | 2026-07-25 06:04:10 PDT |
| exp006_bs1_e5_d4 | 80.0 | 8000 | 10:49:00 | 4.868 | 2026-07-25 08:56:58 PDT |
| exp006_bs1_e5_d4 | 100.0 | 10000 | 13:52:52 | 4.997 | 2026-07-25 12:02:13 PDT |

| Segment end epoch | Completed updates | Last100 loss | Sec/update |
| ---: | ---: | ---: | ---: |
| 25 | 2500 | 0.3914 | 5.863 |
| 50 | 5000 | 0.3742 | 5.738 |
| 75 | 7500 | 0.3608 | 5.692 |
| 100 | 10000 | 0.3555 | 5.689 |
