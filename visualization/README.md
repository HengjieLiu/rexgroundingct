---
created: 2026-07-23
updated: 2026-08-01
status: active
---

# Visualization

This folder contains lightweight, repo-tracked visualization notebooks and
helpers. Heavy generated figures, executed notebooks, and derived tables should
be written to the experiment runtime tree under `/mnt/shengdata1`, not committed
to Git. The curated val200 coronal-projection result below is a narrow exception:
its review gallery is intentionally versioned through Git LFS, while all other
generated visualization PNGs remain ignored.

## Current Notebook

- `2026-07-23_val20_coronal_gt_pred_v123_epoch100.ipynb`: first 20 cases from
  the reshuffled validation set, using v123 epoch-100 predictions.
- `2026-07-24_val200_category_gt_v123_pretrained_coronal.ipynb`: all 200
  reshuffled validation cases, grouped by patient and category, using GT, v123
  epoch-100 predictions, and pretrained VoxTell v1.1 predictions.

The notebook uses `rex_val20_coronal_viz.py` to keep orientation handling,
connected-component measurements, Dice labels, and plotting logic auditable.

## Coronal Projection Development

- `2026-07-31_visualdev_coronal_projection/`: one-row-per-finding whole-volume
  coronal projection pilot. It compares public VoxTell and three fine-tuned
  checkpoints, uses bone-resistant CT projections plus binary mask MIPs, and
  enforces radiology orientation with superior up and patient right on screen
  left. Its full-val200 mean gallery is organized into official category
  folders. See the committed [results overview](2026-07-31_visualdev_coronal_projection/results/README.md)
  for split distributions, four-model comparisons, and category galleries.

## Inputs

- Reshuffled validation order:
  `configs/evaluation/rexgroundingct_val200_seed20260723.json`
- MICCAI metadata:
  `/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json`
- Segmentations:
  `/data/hengjie/datasets/rexgroundingct/segmentations`
- CT volumes:
  `/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct`
- v123 epoch-100 predictions:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v123_opt_poscrop_emptyloss/full_100ep/eval_epoch100_val200/predictions`
- pretrained VoxTell v1.1 predictions:
  `/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval/predictions`

## Outputs

Default generated outputs go to:

```text
/mnt/shengdata1/hengjie/experiments/rexgroundingct/003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/v123_opt_poscrop_emptyloss/full_100ep/visualizations/2026-07-23_val20_coronal_gt_pred_v123_epoch100
```

Expected files include summary CSV/Markdown tables, per-patient PNG figures, and
a JSON run manifest.

The val200 category-grouped notebook writes to:

```text
/data/hengjie/datasets/rexgroundingct/visualizations/2026-07-24_val200_category_gt_v123_pretrained_coronal
```

Its figure filenames start with the official MICCAI category code, for example
`1e_000_train_19753_a_2.png`.
