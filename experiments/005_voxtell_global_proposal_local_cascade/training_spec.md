# Human-Readable Training Specification

Stage 1 is trained from scratch because it is a lightweight global proposal
network, not a resized VoxTell checkpoint. It reuses the external coarse
branch's prompt-conditioned U-Net pattern with instance normalization, which is
appropriate for batch size 1.

| Setting | Value |
| --- | --- |
| Full-FOV input | `192^3` at 4 mm isotropic |
| Effective FOV | 768 mm per axis |
| Normalization | Full native cropped-volume z-score once, before resampling |
| Text | Existing fixed Qwen prompt embeddings |
| Sampling | Same three prompt slots and deterministic order as exp004 |
| Optimizer | AdamW, LR `3e-4`, weight decay `1e-4` |
| Schedule | 100-update warmup, then cosine decay |
| Loss | Weighted BCE + Tversky + valid-center probability-mass ranking objective |
| Empty prompt | BCE only, weight `0.25` |
| Updates | 5,000 (`50 x 100`) |
| Checkpoint/report interval | 500 updates |
| Local stage | Frozen exp003 v123 epoch-100 model |
| Candidate count | Top 3 for segmentation; top 1/3/5 for proposal metrics |

This experiment is intentionally shorter than the full VoxTell continuations.
The proposal model is small, starts from random weights, and has inspectable
val20 inclusion reports every five standardized epochs.

The ranking term is the negative log fraction of all predicted probability
mass that falls inside valid proposal centers. A raw maximum-inside-target
objective was rejected during smoke testing because a random `192^3` heatmap
almost always contains an accidental near-one maximum.
