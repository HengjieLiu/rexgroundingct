"""Small uncapped 3D residual network and identity-preserving tiled inference."""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from exp027_data import extract_patch


class ResidualBlock(nn.Module):
    def __init__(self, channels, groups, dilation):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation),
                                  nn.GroupNorm(groups, channels), nn.GELU(),
                                  nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation),
                                  nn.GroupNorm(groups, channels))

    def forward(self, x):
        return F.gelu(x + self.body(x))


class ResidualRefiner(nn.Module):
    def __init__(self, channels=16, groups=4, dilations=(1, 2, 4, 1)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(2, channels, 3, padding=1), nn.GroupNorm(groups, channels), nn.GELU())
        self.blocks = nn.Sequential(*(ResidualBlock(channels, groups, d) for d in dilations))
        self.head = nn.Conv3d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def make_model(config):
    spec = config["model"]
    return ResidualRefiner(spec["channels"], spec["groups"], spec["dilations"])


def refiner_loss(base, residual, target, valid, l1=.005, eps=1e-6):
    z, delta, y, mask = base.float(), residual.float(), target.float(), valid.float()
    if not (z.shape == delta.shape == y.shape == mask.shape):
        raise ValueError("Loss tensors must have equal shapes")
    dims = tuple(range(1, z.ndim))
    count = mask.sum(dims)
    if torch.any(count == 0):
        raise ValueError("Loss patch has no real voxels")
    final = z + delta
    prob = torch.sigmoid(final)
    bce = ((F.binary_cross_entropy_with_logits(final, y, reduction="none") * mask).sum(dims) / count).mean()
    soft = ((2 * (prob * y * mask).sum(dims) + eps) /
            ((prob * mask).sum(dims) + (y * mask).sum(dims) + eps)).mean()
    magnitude = ((delta.abs() * mask).sum(dims) / count).mean()
    pred = final >= 0
    hard = ((2 * (pred * y * mask).sum(dims) + eps) /
            ((pred * mask).sum(dims) + (y * mask).sum(dims) + eps)).mean()
    return bce + 1 - soft + l1 * magnitude, {
        "bce": bce, "dice_loss": 1 - soft, "residual_mean_abs": magnitude,
        "residual_penalty": l1 * magnitude, "patch_dice": hard,
    }


def tile_starts(shape, size, overlap=.5):
    if len(shape) != 3 or len(size) != 3 or not 0 <= overlap < 1:
        raise ValueError("Invalid tiling geometry")
    axes = []
    for n, p in zip(shape, size):
        if n < 1 or p < 1:
            raise ValueError("Nonpositive tile or image dimension")
        if n <= p:
            axes.append([-(p - n) // 2])
        else:
            count = int(math.ceil((n - p) / (p * (1 - overlap)))) + 1
            axes.append(sorted(set(int(round(x)) for x in np.linspace(0, n - p, count))))
    return list(itertools.product(*axes))


def gaussian_weights(size):
    weights = np.ones(size, dtype=np.float32)
    for axis, width in enumerate(size):
        coordinate = np.arange(width, dtype=np.float32) - (width - 1) / 2
        one = np.exp(-.5 * (coordinate / (width / 8)) ** 2)
        shape = [1, 1, 1]
        shape[axis] = width
        weights *= one.reshape(shape)
    return np.maximum(weights / weights.max(), 1e-7)


@torch.no_grad()
def refine_volume(model, image, base, size, device, *, overlap=.5, amp=True, progress=None):
    if image.shape != base.shape or len(image.shape) != 3:
        raise ValueError("CT and logit geometry mismatch")
    model.eval()
    summed = np.zeros(base.shape, dtype=np.float32)
    denominator = np.zeros(base.shape, dtype=np.float32)
    weight = gaussian_weights(size)
    starts_list = tile_starts(base.shape, size, overlap)
    for index, starts in enumerate(starts_list):
        ct, _ = extract_patch(image, starts, size, 0)
        z, _ = extract_patch(base, starts, size, -30)
        x = torch.from_numpy(np.stack([ct, z]).astype(np.float32))[None].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            delta = model(x)
        value = delta[0, 0].float().cpu().numpy()
        if not np.isfinite(value).all():
            raise FloatingPointError("Non-finite predicted residual")
        dest, src = [], []
        for n, p, start in zip(base.shape, size, starts):
            lo, hi = max(0, start), min(n, start + p)
            dest.append(slice(lo, hi))
            src.append(slice(lo - start, hi - start))
        dest, src = tuple(dest), tuple(src)
        summed[dest] += value[src] * weight[src]
        denominator[dest] += weight[src]
        if progress:
            progress(index + 1, len(starts_list))
    if np.any(denominator <= 0):
        raise RuntimeError("Tiling left uncovered voxels")
    summed /= denominator
    final = np.asarray(base, dtype=np.float32) + summed
    if not np.isfinite(final).all():
        raise FloatingPointError("Non-finite final logits")
    return final, summed
