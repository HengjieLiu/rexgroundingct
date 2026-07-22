# VoxTell v1.1 MICCAI Val Corrected Orientation Quick Evaluation

## Summary

- Cases: `200`
- Findings/prompts: `381`
- Challenge metric, mean global Dice per finding: `0.225228`
- Secondary mean global Dice per case: `0.234287`
- Hit rate: `0.535433`
- Hits/misses: `204` / `177`
- Instance precision/recall/F1: `not computed in quick/global eval`

## Timing

- 4-GPU inference wall time: about `20:04`
- Setup-to-last-prediction time: about `20:14`
- Quick/global evaluation time: about `18-20s`
- Setup-to-report total time: about `20:34`

## Evaluation Mode

This is a corrected-orientation global-only evaluation on the 200-case MICCAI
validation split. It computes Dice and hit rate per finding, but does not compute
connected components or instance precision/recall/F1.

The challenge ranking metric is **Mean global Dice per finding**. Mean global
Dice per case is reported only as a secondary analysis metric.

## Orientation Cleanup Note

The previous canonical Experiment 001 outputs had an orientation export bug. The
old predictions, logs, official eval JSON, and stale report were deleted. The
corrected 200-case predictions and quick/global eval are now the standard
Experiment 001 outputs. The verified single-case diagnostic remains under
`diagnostics/orientation_case_val_idx001` as provenance.

## Comparison Context

| Source | Dice | Hit | Instance F1 |
| --- | ---: | ---: | ---: |
| VoxTell v1.1 corrected orientation quick/global val | 0.225 | 0.535 | n/a |
| VoxTell paper Figure 4 fine-tuned val | 0.282 | 0.678 | n/a |
| ReXrank main benchmark VoxTell | 0.285 | 0.615 | 0.227 |
| MICCAI public leaderboard ThoraxTell | 0.296 | 0.689 | 0.163 |

Notes:
- VoxTell paper Figure 4 reports fine-tuned validation metrics with HIT5, not the official HIT@0.1.
- MICCAI leaderboard rows are contextual because the public leaderboard split differs from local MICCAI validation.
- Public `voxtell_v1.1` is not guaranteed to match a ReXGroundingCT-fine-tuned checkpoint.
