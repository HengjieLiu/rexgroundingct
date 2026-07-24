# Human-Readable Training Specification

Experiment 004 is a continuation test, not a restart from VoxTell v1.1. Both
arms begin from exactly the same exp003 v123 epoch-100 network weights and use
fresh optimization state.

| Setting | Exp004 value | Same as exp003 v123? | Same as VoxTell pretraining? |
| --- | --- | --- | --- |
| Tensor patch | `192^3` | Yes | Yes |
| Prompts per sample | 3 | Yes | Yes |
| One-finding fallback | 1 positive + 2 negatives | Yes | Not established by the paper |
| Positive crop guarantee | Every selected positive is nonempty | Yes | More explicit than the paper |
| Batch / accumulation | 1 / 1 | Yes | No |
| Updates per epoch | 100 | Yes | Project-specific |
| Optimizer | SGD Nesterov, momentum 0.99 | Yes | Fine-tuning recipe |
| Encoder / other LR | `1e-7` / `1e-6` | Yes | No |
| Warmup / decay | 100 updates / poly 0.9 | Yes, restarted | No |
| Loss | v123 Dice + weighted BCE; empty BCE only at 0.5 | Yes | No |
| Deep supervision | `[1,.5,.25,.125,.0625]` normalized | Yes | Similar structure |
| Native normalization | Crop, then full cropped-volume z-score once | Yes | Yes |
| 2 mm normalization | Native full-volume z-score before resampling | New | No |
| 2 mm physical FOV | 384 mm per axis | New | No |

The native arm is the control for another 100 epochs of training. Therefore,
differences between arms should be attributable primarily to physical-space
sampling and inference geometry, while differences from the original exp003
baseline also include continuation time and a restarted learning-rate cycle.
