"""Finding-aware, single-patch deletion objectives; no hard threshold in gradients."""
import torch

from fit_027_deletion import deletion_loss
from deletion027_ablation_data import LOSSES


def loss_for_arm(logits, base, target, valid, counts, arm):
    if arm not in LOSSES:
        raise ValueError("Unknown loss arm")
    # This preserves historical BCE operation ordering and gradients exactly.
    weight = 2. if arm.endswith("tp2") else 1.
    bce, parts = deletion_loss(logits, base, target, valid, 1., weight)
    t0, f0, g0 = (int(counts[k]) for k in ("base_tp", "base_fp", "total_gt"))
    tp = (base >= 0) & (target > 0) & (valid > 0)
    fp = (base >= 0) & (target == 0) & (valid > 0)
    if min(t0, f0, g0-t0) < 0 or int(tp.sum()) > t0 or int(fp.sum()) > f0:
        raise ValueError("Invalid whole-finding counts")
    probability = logits.float().sigmoid()
    a, b = probability[tp].sum(), probability[fp].sum()
    dice = 1 - (2*(t0-a)+1e-6)/(t0+f0+g0-a-b+1e-6)
    normalized = bce/(1+weight)
    loss = dice + .25*normalized if arm.startswith("dice_") else bce
    return loss, {**parts, "dice_loss": dice, "normalized_bce": normalized,
                  "soft_tp_removed": a, "soft_fp_removed": b}
