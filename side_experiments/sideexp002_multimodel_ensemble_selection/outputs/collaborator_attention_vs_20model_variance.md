# Collaborator Attention Variants Versus 20-Model Variation

The collaborator's three runs are a paired comparison on the exact same val200 evaluation: initialization, seed, data order, schedule/settings, and evaluation are paired, with attention configuration as the only intended intervention. The 20-model reference is a heterogeneous empirical variation envelope, **not** a random-seed variance estimate.

> Percentiles below are descriptive reference positions, not p-values or confidence intervals. Without repeated matched seeds or collaborator per-finding outputs, the analysis cannot establish statistical causality.

## Take-Home Message

Attention appears to **redistribute category performance rather than uniformly improve the model**. The strongest credible signal is a targeted improvement for `2b` (atelectasis/consolidation): both paired attention variants improve it, and the effect is unusually large relative to the 20-model envelope. `2d` (pulmonary nodules/masses) moves consistently downward, suggesting a possible tradeoff, while most other changes are compatible with ordinary model variation. Rare-category swings are too unstable to attribute. Repeated matched seeds are still required for a causal claim.

## Concise Evidence Assessment

- Category 2b is the strongest targeted attention-associated signal: both paired attention variants improve it and its size-matched spread is unusually high relative to the 20-model envelope.
- Category 2d moves downward for both attention variants, but its magnitude is not exceptional; it is a possible attention tradeoff, not strong outlier evidence.
- Most other category changes are mixed or remain inside the empirical 20-model variation envelope.
- The recomposed overall shifts are ordinary relative to the 190 20-model pairs, so these results do not show a broad attention benefit.
- Low-support swings, especially 1f, 2g, and 2h, cannot support attention attribution.

The clearest result is `2b`: collaborator spread `0.020700` is at the `98.4`th matched-triplet percentile, and both attention variants improve the baseline. In contrast, `2d` moves down in both variants but its spread is only at the `52.7`th percentile.

## Evaluation and Category Support

- Split: `rexgroundingct_val200_seed20260723` (200 cases, 381 findings).
- Mask threshold: `0.5`; category/global Dice is the unweighted mean across findings.
- High-support interpretation threshold: `N >= 20`.
- `2f` honeycombing is unavailable because val200 has no findings.

| Code | Official category | N | Interpretation tier |
| --- | --- | ---: | --- |
| 1a | Bronchial wall thickening | 3 | Low support; no attribution |
| 1b | Bronchiectasis | 11 | Low support; no attribution |
| 1c | Emphysema | 17 | Low support; no attribution |
| 1d | Septal thickening / reticulation | 6 | Low support; no attribution |
| 1e | Micronodules / tree-in-bud | 11 | Low support; no attribution |
| 1f | Other diffuse lung/airway/pleural abnormality | 4 | Low support; no attribution |
| 2a | Linear opacity, scarring, fibrosis | 69 | High support |
| 2b | Atelectasis / consolidation | 49 | High support |
| 2c | Ground-glass opacity | 60 | High support |
| 2d | Pulmonary nodules / masses | 132 | High support |
| 2e | Pleural effusion / thickening | 11 | Low support; no attribution |
| 2f | Honeycombing | 0 | Unavailable |
| 2g | Pneumothorax | 1 | Low support; no attribution |
| 2h | Other focal lung/airway/pleural finding | 7 | Low support; no attribution |

## Collaborator Table Transcription

Scores and reported deltas preserve the screenshot's four-decimal display. A recomputed delta may differ by `0.0001` because the source scores were rounded independently; the declared tolerance is `0.0002`.

| Code | Category (N) | Baseline | All-category (reported / recomputed Δ) | Strict-2b2c (reported / recomputed Δ) | Displayed best | Check |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 1a | Bronchial wall thickening (3) | 0.1067 | 0.0936 (-0.0131 / -0.0131) | 0.0944 (-0.0123 / -0.0123) | Baseline | Pass |
| 1b | Bronchiectasis (11) | 0.1274 | 0.1223 (-0.0050 / -0.0051) | 0.1575 (+0.0301 / +0.0301) | Strict | Pass |
| 1c | Emphysema (17) | 0.1831 | 0.1892 (+0.0061 / +0.0061) | 0.1540 (-0.0291 / -0.0291) | All-cat | Pass |
| 1d | Septal thickening (6) | 0.1254 | 0.1350 (+0.0095 / +0.0096) | 0.1267 (+0.0013 / +0.0013) | All-cat | Pass |
| 1e | Micronodules/tree-in-bud (11) | 0.2032 | 0.1943 (-0.0090 / -0.0089) | 0.2110 (+0.0078 / +0.0078) | Strict | Pass |
| 1f | Other diffuse (4) | 0.1827 | 0.2199 (+0.0372 / +0.0372) | 0.2402 (+0.0575 / +0.0575) | Strict* | Pass |
| 2a | Linear opacity/fibrosis (69) | 0.3340 | 0.3358 (+0.0019 / +0.0018) | 0.3299 (-0.0041 / -0.0041) | Close | Pass |
| 2b | Atelectasis/consolidation (49) | 0.3418 | 0.3578 (+0.0160 / +0.0160) | 0.3625 (+0.0207 / +0.0207) | Strict | Pass |
| 2c | Ground-glass opacity (60) | 0.3762 | 0.3866 (+0.0104 / +0.0104) | 0.3782 (+0.0020 / +0.0020) | All-cat | Pass |
| 2d | Nodules/masses (132) | 0.3799 | 0.3692 (-0.0107 / -0.0107) | 0.3669 (-0.0130 / -0.0130) | Baseline | Pass |
| 2e | Pleural effusion/thickening (11) | 0.4544 | 0.4676 (+0.0132 / +0.0132) | 0.4459 (-0.0085 / -0.0085) | All-cat | Pass |
| 2g | Pneumothorax (1) | 0.1034 | 0.0908 (-0.0126 / -0.0126) | 0.1184 (+0.0150 / +0.0150) | Insufficient sample | Pass |
| 2h | Other focal (7) | 0.1791 | 0.1850 (+0.0059 / +0.0059) | 0.1013 (-0.0778 / -0.0778) | All-cat | Pass |
| 2f | Honeycombing (0) | — | — | — | Unavailable | Pass |

## Recomposed Overall Scores

These scores are recomposed from category means and supports; no separate collaborator global result was supplied.

| Variant | Support-weighted Dice | Δ vs baseline | Macro category Dice | Macro Δ |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.334439 | +0.000000 | 0.238254 | +0.000000 |
| All-category | 0.335516 | +0.001077 | 0.242085 | +0.003831 |
| Strict-2b2c | 0.330857 | -0.003582 | 0.237454 | -0.000800 |

## Full Category Variance Comparison

`Triplet pct` compares the three collaborator scores' range with all 1,140 three-model ranges from our pool. `A pct` and `S pct` compare each absolute attention-baseline delta with all 190 within-category 20-model differences. Percentiles use tie-aware empirical midranks.

| Code | N | B / A / S Dice | Collaborator mean ± sample SD | Collaborator range | Best / worst variant | Sample variance C / 20 | 20-model mean ± sample SD | 20-model IQR / MAD | 20-model range | Range ratio | Triplet pct | A pct | S pct | Assessment |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1a | 3 | 0.1067 / 0.0936 / 0.0944 | 0.098233 ± 0.007343 | 0.013100 | Baseline / All-category | 0.00005392 / 0.00016675 | 0.110120 ± 0.012913 | 0.023673 / 0.011165 | 0.037624 | 0.348 | 18.0 | 48.4 | 46.3 | Low support; no attribution |
| 1b | 11 | 0.1274 / 0.1223 / 0.1575 | 0.135733 ± 0.019022 | 0.035200 | Strict-2b2c / All-category | 0.00036184 / 0.00034397 | 0.152490 ± 0.018546 | 0.024323 / 0.010793 | 0.061657 | 0.571 | 57.1 | 16.8 | 72.6 | Low support; no attribution |
| 1c | 17 | 0.1831 / 0.1892 / 0.1540 | 0.175433 ± 0.018811 | 0.035200 | All-category / Strict-2b2c | 0.00035384 / 0.00038930 | 0.167025 ± 0.019731 | 0.018786 / 0.006927 | 0.070393 | 0.500 | 71.8 | 32.6 | 75.8 | Low support; no attribution |
| 1d | 6 | 0.1254 / 0.1350 / 0.1267 | 0.129033 ± 0.005208 | 0.009600 | All-category / Baseline | 0.00002712 / 0.00007172 | 0.118237 ± 0.008469 | 0.009300 / 0.005045 | 0.028884 | 0.332 | 34.5 | 60.5 | 7.9 | Low support; no attribution |
| 1e | 11 | 0.2032 / 0.1943 / 0.2110 | 0.202833 ± 0.008356 | 0.016700 | Strict-2b2c / All-category | 0.00006982 / 0.00012967 | 0.181278 ± 0.011387 | 0.010420 / 0.006057 | 0.043356 | 0.385 | 49.6 | 46.8 | 38.4 | Low support; no attribution |
| 1f | 4 | 0.1827 / 0.2199 / 0.2402 | 0.214267 ± 0.029161 | 0.057500 | Strict-2b2c / Baseline | 0.00085036 / 0.00068558 | 0.173743 ± 0.026184 | 0.003964 / 0.002201 | 0.074842 | 0.768 | 61.8 | 66.3 | 77.9 | Low support; no attribution |
| 2a | 69 | 0.3340 / 0.3358 / 0.3299 | 0.333233 ± 0.003024 | 0.005900 | All-category / Strict-2b2c | 0.00000914 / 0.00006787 | 0.333488 ± 0.008238 | 0.010344 / 0.003938 | 0.028158 | 0.210 | 16.6 | 12.6 | 33.7 | Mixed direction; not outlier |
| 2b | 49 | 0.3418 / 0.3578 / 0.3625 | 0.354033 ± 0.010852 | 0.020700 | Strict-2b2c / Baseline | 0.00011776 / 0.00002837 | 0.354552 ± 0.005326 | 0.004576 / 0.002241 | 0.021792 | 0.950 | 98.4 | 96.8 | 99.5 | Attention-associated outlier |
| 2c | 60 | 0.3762 / 0.3866 / 0.3782 | 0.380333 ± 0.005518 | 0.010400 | All-category / Baseline | 0.00003045 / 0.00006741 | 0.388037 ± 0.008210 | 0.011163 / 0.005798 | 0.027714 | 0.375 | 30.9 | 61.6 | 10.5 | Consistent positive; not outlier |
| 2d | 132 | 0.3799 / 0.3692 / 0.3669 | 0.372000 ± 0.006938 | 0.013000 | Baseline / Strict-2b2c | 0.00004813 / 0.00005335 | 0.365362 ± 0.007304 | 0.010430 / 0.005155 | 0.027721 | 0.469 | 52.7 | 68.4 | 77.4 | Possible tradeoff; not outlier |
| 2e | 11 | 0.4544 / 0.4676 / 0.4459 | 0.455967 ± 0.010935 | 0.021700 | All-category / Strict-2b2c | 0.00011956 / 0.00069811 | 0.443912 ± 0.026422 | 0.033591 / 0.016192 | 0.092982 | 0.233 | 16.3 | 29.5 | 20.5 | Low support; no attribution |
| 2f | 0 | — | — | — | — | — | — | — | — | — | — | — | — | Unavailable |
| 2g | 1 | 0.1034 / 0.0908 / 0.1184 | 0.104200 ± 0.013817 | 0.027600 | Strict-2b2c / All-category | 0.00019092 / 0.00020892 | 0.060462 ± 0.014454 | 0.016849 / 0.007386 | 0.068826 | 0.401 | 71.8 | 48.9 | 56.3 | Low support; no attribution |
| 2h | 7 | 0.1791 / 0.1850 / 0.1013 | 0.155133 ± 0.046714 | 0.083700 | All-category / Strict-2b2c | 0.00218222 / 0.00465816 | 0.157238 ± 0.068251 | 0.053075 / 0.027472 | 0.208711 | 0.401 | 42.9 | 6.3 | 61.1 | Low support; no attribution |

## High-Support Categories (N >= 20)

- `2a` (Linear opacity, scarring, fibrosis, N=69): All-category +0.0018, Strict-2b2c -0.0041; matched-triplet percentile 16.6. **Mixed direction; not outlier.**
- `2b` (Atelectasis / consolidation, N=49): All-category +0.0160, Strict-2b2c +0.0207; matched-triplet percentile 98.4. **Attention-associated outlier.**
- `2c` (Ground-glass opacity, N=60): All-category +0.0104, Strict-2b2c +0.0020; matched-triplet percentile 30.9. **Consistent positive; not outlier.**
- `2d` (Pulmonary nodules / masses, N=132): All-category -0.0107, Strict-2b2c -0.0130; matched-triplet percentile 52.7. **Possible tradeoff; not outlier.**

## Low-Support Categories

No attention-attribution label is assigned below 20 findings. In particular, `2g` has one finding; `1f` has four; and `2h` has seven. Their large-looking swings can be dominated by one or a few examples.

| Code | N | All-category Δ | Strict-2b2c Δ | Collaborator range | Matched-triplet pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1a | 3 | -0.013100 | -0.012300 | 0.013100 | 18.0 |
| 1b | 11 | -0.005100 | +0.030100 | 0.035200 | 57.1 |
| 1c | 17 | +0.006100 | -0.029100 | 0.035200 | 71.8 |
| 1d | 6 | +0.009600 | +0.001300 | 0.009600 | 34.5 |
| 1e | 11 | -0.008900 | +0.007800 | 0.016700 | 49.6 |
| 1f | 4 | +0.037200 | +0.057500 | 0.057500 | 61.8 |
| 2e | 11 | +0.013200 | -0.008500 | 0.021700 | 16.3 |
| 2g | 1 | -0.012600 | +0.015000 | 0.027600 | 71.8 |
| 2h | 7 | +0.005900 | -0.077800 | 0.083700 | 42.9 |

## Cross-Category Direction and Magnitude

Across all 13 represented categories, the variants move in the same nonzero direction for `6/13` categories (Pearson delta correlation `0.306119`). Across high-support categories `2a, 2b, 2c, 2d`, they agree for `3/4` (correlation `0.923186`).

| Variant | Support-weighted RMS Δ | Pair percentile | Macro RMS Δ | Pair percentile | Absolute recomposed overall Δ | Pair percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-category | 0.010878 | 15.3 | 0.014250 | 15.3 | 0.001077 | 11.1 |
| Strict-2b2c | 0.018244 | 58.4 | 0.030686 | 59.5 | 0.003582 | 38.9 |

## Secondary Context: Exp009 Attention Ablation

> Method-ablation context only; these arms do not estimate random-seed variance.

Values are category Dice changes from `exp009_baseline_e100` for the three S3 attention arms.

| Code | Exp009 baseline | S3v1 Δ | S3v2 Δ | S3v3 Δ | Four-arm range |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1a | 0.115071 | +0.003156 | +0.009543 | +0.007920 | 0.009543 |
| 1b | 0.157151 | -0.004272 | +0.004568 | +0.015649 | 0.019921 |
| 1c | 0.177998 | -0.004027 | +0.000089 | +0.004603 | 0.008630 |
| 1d | 0.125360 | +0.001248 | +0.000960 | +0.003662 | 0.003662 |
| 1e | 0.178662 | -0.000496 | +0.007514 | -0.007257 | 0.014771 |
| 1f | 0.161189 | -0.002401 | +0.058696 | +0.001637 | 0.061096 |
| 2a | 0.340520 | -0.007383 | +0.003931 | -0.000821 | 0.011314 |
| 2b | 0.356849 | +0.005559 | +0.004477 | -0.000261 | 0.005819 |
| 2c | 0.390826 | -0.003031 | -0.002740 | -0.000566 | 0.003031 |
| 2d | 0.378417 | -0.011211 | -0.016380 | -0.020653 | 0.020653 |
| 2e | 0.448143 | +0.000960 | +0.030662 | +0.005606 | 0.030662 |
| 2g | 0.072083 | -0.021352 | -0.016047 | -0.019648 | 0.021352 |
| 2h | 0.233286 | +0.009816 | -0.011159 | -0.024303 | 0.034120 |

## Secondary Context: Optimization Trajectories

> Within-lineage optimization-trajectory context; repeated snapshots are dependent and do not estimate random-seed variance.

Exp007 reports the range over epochs 50/75/100. Exp008 cells are signed epoch-100 minus epoch-80 category Dice changes.

| Code | Exp007 e50/e75/e100 range | Exp007 e100−e50 | Exp008 shared Δ | dual Δ | precision Δ | joint Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1a | 0.015432 | -0.015432 | +0.020313 | +0.020256 | +0.020131 | +0.021991 |
| 1b | 0.030069 | -0.030069 | +0.019178 | +0.009084 | +0.003063 | +0.005822 |
| 1c | 0.059881 | +0.036056 | -0.003891 | -0.001606 | +0.000370 | -0.007857 |
| 1d | 0.010382 | -0.004226 | -0.003883 | -0.003507 | -0.006309 | +0.002075 |
| 1e | 0.036515 | -0.016247 | +0.005058 | +0.009041 | +0.020532 | +0.040808 |
| 1f | 0.020611 | +0.000128 | -0.001253 | -0.004566 | -0.005740 | -0.006610 |
| 2a | 0.010700 | +0.002459 | +0.001244 | -0.006055 | -0.001205 | -0.006330 |
| 2b | 0.002105 | +0.000806 | -0.009377 | -0.002062 | -0.005342 | -0.006189 |
| 2c | 0.006736 | +0.000004 | -0.007635 | -0.009744 | -0.018659 | -0.010226 |
| 2d | 0.006045 | -0.006045 | +0.015657 | +0.020764 | +0.013628 | +0.013412 |
| 2e | 0.025436 | -0.020573 | -0.007688 | +0.004850 | +0.025696 | +0.005319 |
| 2g | 0.048762 | +0.044555 | +0.019434 | +0.002069 | +0.007028 | +0.008146 |
| 2h | 0.020492 | -0.020492 | +0.005378 | +0.036023 | +0.013356 | +0.072539 |

## Interpretation Boundary

- The paired collaborator design makes attention the primary intended intervention, so directionally consistent, unusually large effects are more suggestive than ordinary mixed changes.
- The 20-model pool mixes experiments, architectures, normalization, and trajectory snapshots. It is an empirical plausibility envelope, not an estimate of independent-seed noise.
- Exp009 is method-ablation context; Exp007/Exp008 snapshots are dependent optimization-trajectory context. Neither is random-seed variance.
- Formal attribution would require repeated matched seeds, ideally with aligned per-finding collaborator outputs.

## Provenance and Reproduction

- Collaborator manifest SHA256: `5d1b791333f4270aa83d35805ebb659f892a38d0b6c87c6be5cbe5267439e14c`
- 20-model baseline summary SHA256: `cf9dadd4ac3b3ad869ab889596309ee61ff19afa8dab281249a4012696f3cfec`
- Dataset SHA256: `7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897`
- Deterministic result SHA256: `f550b065b22fdc9232f38e9e46cbc88489732ce26dbcfd402be6f4fc3153ef92`

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  side_experiments/sideexp002_multimodel_ensemble_selection/compare_collaborator_attention_variance.py
```
