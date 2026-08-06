# Experiment 014 DDP Batch16 Update-Matched Report

Run group: `exp014_ddp_bs16_update_matched_20260804T051845Z`

## Val200 Progress

| Checkpoint | Comparison role | Dice | Hit rate | Hits / findings |
| ---: | --- | ---: | ---: | ---: |
| 25 | sample-matched to exp007 epoch100 | 0.3230 | 0.7612 | n/a |
| 50 | intermediate | 0.3207 | 0.7507 | n/a |
| 75 | intermediate | 0.3258 | 0.7559 | n/a |

## References

| Model | Dice | Hit rate | Hits / findings |
| --- | ---: | ---: | ---: |
| exp007_ddp_bs4_epoch100 | 0.3310 | 0.7585 | n/a |
| exp006_v123_cached_e5_d4_epoch100 | 0.3241 | 0.7612 | n/a |

## Training

- Completed updates: `7500`
- World size: `4`
- Per-GPU batch size: `1`
- Gradient accumulation: `4`
- Effective global batch size: `16`
- Last loss: `0.3111`
- Mean seconds/update: `19.074`
- Samples/second: `2.517`
