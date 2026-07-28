# Experiment 008 Dual-Branch Ablation

## Final Segmentation Val20

| Variant | e0 | e20 | e40 | e60 | e80 | e100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1_sharedfusion_softguide` | 0.3907/0.7419 | 0.3135/0.6774 | 0.3246/0.7097 | 0.3295/0.6774 | 0.3689/0.6774 | 0.3696/0.7097 |
| `v1_dualfusion_softguide` | 0.3907/0.7419 | 0.3419/0.7419 | 0.3612/0.7742 | 0.3124/0.7097 | 0.3791/0.7097 | 0.3722/0.7097 |
| `v2_dualfusion_precision` | 0.3907/0.7419 | 0.3217/0.6774 | 0.3357/0.6774 | 0.3024/0.7419 | 0.3769/0.7097 | 0.3710/0.7097 |
| `v3_dualfusion_softguide_joint` | 0.3907/0.7419 | 0.2846/0.6774 | 0.3572/0.7419 | 0.3585/0.7419 | 0.3780/0.7419 | 0.3813/0.7419 |

Cells are `Dice / hit rate`.

## Val200

Starting-point reference: Exp006 `v123_cached_e5_d4` epoch 100, Dice `0.3241`, hit rate `0.7612` (`290/381`).

| Variant | e80 | e100 |
| --- | ---: | ---: |
| `v1_sharedfusion_softguide` | 0.3328/0.7428 (283/381) | 0.3366/0.7533 (287/381) |
| `v1_dualfusion_softguide` | 0.3304/0.7559 (288/381) | 0.3360/0.7507 (286/381) |
| `v2_dualfusion_precision` | 0.3307/0.7428 (283/381) | 0.3333/0.7533 (287/381) |
| `v3_dualfusion_softguide_joint` | 0.3308/0.7559 (288/381) | 0.3345/0.7559 (288/381) |

Cells are `Dice / hit rate (hits/findings)`.

## Interpretation Limit

No same-source single-branch continuation control is present; improvement over epoch 0 cannot be attributed entirely to the dual-branch architecture.
