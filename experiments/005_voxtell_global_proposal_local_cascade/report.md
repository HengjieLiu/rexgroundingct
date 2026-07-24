# Experiment 005 Final Comparison

| Variant | Proposal hit@3 | Full inclusion@3 | Coverage@3 | Cascade Dice | Cascade hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict_inclusion | 0.6719 | 0.2073 | 0.4294 | 0.1524 | 0.4042 |
| inclusion_margin16mm | 0.7559 | 0.2257 | 0.5013 | 0.1624 | 0.4436 |

Stage 1 is selected by full inclusion and target coverage; end-to-end selection
also considers cascade Dice and hit rate at the fixed threshold 0.5.
