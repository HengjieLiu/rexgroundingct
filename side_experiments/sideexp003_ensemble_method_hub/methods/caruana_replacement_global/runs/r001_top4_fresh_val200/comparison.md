# Uniform Top-4/Top-8 vs. Caruana-Replacement Diagnostic

This is an optimistic full-val diagnostic, not an OOF estimate or a formal recipe.

| Method | Dice | Hits | Marginal |
| --- | ---: | ---: | ---: |
| Uniform top-4 | 0.357504 | 296 | — |
| Uniform top-8 | 0.352344 | 291 | -0.005160 vs. top-4 |
| Caruana K=1 | 0.345868 | 295 | — |
| Caruana K=2 | **0.360140** | 295 | +0.014272 |
| Caruana K=3 | 0.358606 | 295 | -0.001533 |
| Caruana K=4 | 0.358433 | 292 | -0.000173 |

Uniform top-8 took 98.218 minutes. Uniform top-4 took 68.154 minutes, and
Caruana's three search passes took 101.032 minutes in total.

The earlier top-4 comparison worker (uniform followed by Caruana) took 169.189
minutes end to end.

The forced K=4 sequence is
`rank1, rank3, rank4, rank3`, with frequency weights `rank1=0.25`,
`rank3=0.50`, and `rank4=0.25`. The best full-val point is K=2.

No derived ensemble logits were written.
