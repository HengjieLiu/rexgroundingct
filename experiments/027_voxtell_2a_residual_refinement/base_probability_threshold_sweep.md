# Frozen Exp007 2a probability-threshold sweep

User-requested probability thresholds 0.25–0.95 in steps of 0.05. This is a binary foreground threshold on the frozen base logits; the deletion module is not used. Status: **pending_user_review**.

Checkpoint: Exp007 continuation epoch 50 / absolute epoch 150, SHA `a499ad1c0fada9a7e4d78e352fa5510d31295e82749e00ef0a4eee6da685e4aa`. All 69 validation 2a findings across 63 CTs are included; A has 35 findings and B has 34.

Predicted foreground = sigmoid(cached logit) >= threshold, evaluated equivalently in logit space. The entire cached native extent is included, so thresholds below 0.50 can recover additional foreground. The established clipped [-30,30] cache and its recorded storage dtype are preserved; there is no new model inference or quantization. Clipping cannot change decisions within the requested threshold range.

Dice is the unweighted mean over findings, using smoothing 1e-6. The native geometry has no resampling; original finding IDs, prompt/channel order and GT voxel counts are checked. All 63 CT cache logit/target hashes passed, and all cached extents equal the full original reoriented CT extents. Threshold 0.50 reproduces every cached baseline finding; A+B recomposition and monotonic TP/prediction counts passed.

![Threshold sweep](runtime/analysis/base_probability_threshold_sweep/threshold_sweep.png)

| Probability threshold | Logit threshold | A Dice | B Dice | Full Dice | Full change vs 0.50 | Full hits | Full precision | Full recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25 | -1.098612 | 0.335217 | 0.328937 | 0.332122 | -0.005093 | 60/69 | 33.005% | 50.266% |
| 0.30 | -0.847298 | 0.336266 | 0.330503 | 0.333426 | -0.003789 | 59/69 | 33.516% | 49.549% |
| 0.35 | -0.619039 | 0.337287 | 0.331729 | 0.334548 | -0.002667 | 59/69 | 33.989% | 48.906% |
| 0.40 | -0.405465 | 0.338025 | 0.332932 | 0.335515 | -0.001700 | 59/69 | 34.430% | 48.304% |
| 0.45 | -0.200671 | 0.338782 | 0.334081 | 0.336466 | -0.000750 | 59/69 | 34.865% | 47.727% |
| 0.50 | 0.000000 | 0.339416 | 0.334950 | 0.337215 | +0.000000 | 59/69 | 35.282% | 47.186% |
| 0.55 | 0.200671 | 0.340001 | 0.335697 | 0.337881 | +0.000665 | 59/69 | 35.704% | 46.621% |
| 0.60 | 0.405465 | 0.340245 | 0.336334 | 0.338318 | +0.001103 | 59/69 | 36.121% | 46.019% |
| 0.65 | 0.619039 | 0.340449 | 0.336932 | 0.338716 | +0.001500 | 59/69 | 36.578% | 45.412% |
| 0.70 | 0.847298 | 0.340628 | 0.337530 | 0.339101 | +0.001886 | 59/69 | 37.074% | 44.748% |
| 0.75 | 1.098612 | 0.340673 | 0.337990 | 0.339351 | +0.002136 | 59/69 | 37.590% | 44.030% |
| 0.80 | 1.386294 | 0.340495 | 0.338246 | 0.339387 | +0.002172 | 59/69 | 38.215% | 43.151% |
| 0.85 | 1.734601 | 0.340220 | 0.338092 | 0.339172 | +0.001956 | 59/69 | 38.993% | 42.053% |
| 0.90 | 2.197225 | 0.339332 | 0.337385 | 0.338372 | +0.001157 | 58/69 | 40.042% | 40.507% |
| 0.95 | 2.944439 | 0.335226 | 0.335102 | 0.335165 | -0.002051 | 57/69 | 42.050% | 37.607% |

Precision and recall above are also means over findings. Hits use Dice >= 0.10. The exports include raw TP/FP/FN counts, pooled Dice, per-finding changes, and hits gained/lost. Threshold ranking and selection remain with the user.

The extension to 0.95 exactly reproduced all 33 previous aggregate rows and
759 per-finding rows for 0.25–0.75. The original report, exports and executed
script are preserved under `runtime/analysis/base_probability_threshold_sweep/range_025_075/`.
The maximum full-cohort Dice in this grid is 0.339387 at probability 0.80,
only 0.000036 above 0.75 and 0.002172 above the original 0.50 baseline.

- [All 45 A/B/full summaries: CSV](runtime/analysis/base_probability_threshold_sweep/summary.csv)
- [All 1035 per-finding rows: CSV](runtime/analysis/base_probability_threshold_sweep/per_finding.csv)
- [Results, hashes and provenance: JSON](runtime/analysis/base_probability_threshold_sweep/results.json)
- [Figure PDF](runtime/analysis/base_probability_threshold_sweep/threshold_sweep.pdf)

CPU analysis completed in 18.1 seconds before plotting. Reproduce with `python scripts/rexgroundingct/audit_027_base_probability_thresholds.py --workers 4` in the existing VoxTell image with GPUs disabled.
