#!/usr/bin/env python3
"""Full-FOV proposal cache, model, and candidate geometry for experiment 005."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import REX_SEG_DIR
from train_text_conditioned_voxtell import crop_target, reorient_target_to_voxtell_layout
from voxtell_2mm import cache_case_key, load_cached_case


GLOBAL_SHAPE = (192, 192, 192)
GLOBAL_SPACING_MM = 4.0
LOCAL_PATCH_SIZE = (192, 192, 192)


def global_case_dir(cache_root: Path, name: str) -> Path:
    return Path(cache_root) / "cases" / cache_case_key(name)


def global_case_paths(cache_root: Path, name: str) -> dict[str, Path]:
    root = global_case_dir(cache_root, name)
    return {
        "root": root,
        "image": root / "image.npy",
        "native_targets": root / "native_targets.npz",
        "metadata": root / "metadata.json",
        "complete": root / ".complete",
    }


def center_pad_slices(shape: tuple[int, int, int]) -> tuple[list[int], list[int], tuple[slice, ...]]:
    if any(size > target for size, target in zip(shape, GLOBAL_SHAPE)):
        raise ValueError(f"4 mm shape {shape} does not fit global grid {GLOBAL_SHAPE}")
    before = [(target - size) // 2 for size, target in zip(shape, GLOBAL_SHAPE)]
    after = [target - size - left for size, target, left in zip(shape, GLOBAL_SHAPE, before)]
    slices = tuple(slice(left, left + size) for left, size in zip(before, shape))
    return before, after, slices


def downsample_2mm_image_to_global(image_2mm: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int], list[int], list[int]]:
    shape_4mm = tuple(int(math.ceil(size / 2.0)) for size in image_2mm.shape[1:])
    resized = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(image_2mm))[None].float(),
        size=shape_4mm,
        mode="trilinear",
        align_corners=False,
    )[0, 0].numpy()
    before, after, slices = center_pad_slices(shape_4mm)
    output = np.zeros(GLOBAL_SHAPE, dtype=np.float16)
    output[slices] = resized.astype(np.float16, copy=False)
    return output[None], shape_4mm, before, after


def global_index_to_native_start(
    global_index: int,
    axis: int,
    metadata: dict[str, Any],
) -> int:
    pad_before = int(metadata["pad_before_zyx"][axis])
    shape_4mm = int(metadata["shape_4mm_zyx"][axis])
    native_size = int(metadata["native_cropped_shape_zyx"][axis])
    local = min(max(int(global_index) - pad_before, 0), shape_4mm - 1)
    native_center = (local + 0.5) * native_size / float(shape_4mm) - 0.5
    max_start = max(0, native_size - LOCAL_PATCH_SIZE[axis])
    return int(np.clip(round(native_center - (LOCAL_PATCH_SIZE[axis] - 1) / 2.0), 0, max_start))


def global_point_to_native_starts(
    point_zyx: tuple[int, int, int] | list[int],
    metadata: dict[str, Any],
) -> list[int]:
    return [
        global_index_to_native_start(int(point_zyx[axis]), axis, metadata)
        for axis in range(3)
    ]


def eligible_global_axis_range(
    axis: int,
    start_low: int,
    start_high: int,
    metadata: dict[str, Any],
) -> list[int]:
    valid_low = int(metadata["pad_before_zyx"][axis])
    valid_high = valid_low + int(metadata["shape_4mm_zyx"][axis]) - 1
    eligible = [
        index
        for index in range(valid_low, valid_high + 1)
        if start_low <= global_index_to_native_start(index, axis, metadata) <= start_high
    ]
    if eligible:
        return [min(eligible), max(eligible)]
    midpoint = (start_low + start_high) / 2.0
    nearest = min(
        range(valid_low, valid_high + 1),
        key=lambda index: abs(global_index_to_native_start(index, axis, metadata) - midpoint),
    )
    return [nearest, nearest]


def proposal_boxes_for_native_target(
    target: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    coords = np.argwhere(target > 0)
    if coords.size == 0:
        raise ValueError("Proposal target must be nonempty")
    bbox_min = coords.min(axis=0).astype(int)
    bbox_max = coords.max(axis=0).astype(int)
    inclusion_ranges: list[list[int]] = []
    hit_ranges: list[list[int]] = []
    inclusion_possible = True
    for axis, (low, high, dim, patch) in enumerate(
        zip(bbox_min, bbox_max, target.shape, LOCAL_PATCH_SIZE)
    ):
        max_start = max(0, int(dim) - patch)
        inclusion_low = max(0, int(high) - patch + 1)
        inclusion_high = min(int(low), max_start)
        hit_low = max(0, int(low) - patch + 1)
        hit_high = min(int(high), max_start)
        if inclusion_low > inclusion_high:
            inclusion_possible = False
        inclusion_ranges.append([inclusion_low, inclusion_high])
        hit_ranges.append([hit_low, hit_high])

    selected = inclusion_ranges if inclusion_possible else hit_ranges
    global_box = [
        eligible_global_axis_range(axis, bounds[0], bounds[1], metadata)
        for axis, bounds in enumerate(selected)
    ]
    return {
        "native_bbox_zyx": [bbox_min.tolist(), bbox_max.tolist()],
        "native_voxels": int(coords.shape[0]),
        "inclusion_possible": inclusion_possible,
        "valid_native_start_ranges_zyx": inclusion_ranges,
        "hit_native_start_ranges_zyx": hit_ranges,
        "proposal_target_policy": (
            "full_inclusion_valid_centers" if inclusion_possible else "hit_valid_centers_fallback"
        ),
        "proposal_box_global_zyx": global_box,
    }


def load_native_targets_from_metadata(
    name: str,
    metadata_2mm: dict[str, Any],
    seg_dir: Path = REX_SEG_DIR,
) -> np.ndarray:
    gt_img = nib.load(str(Path(seg_dir) / name))
    target_fxyz = np.asanyarray(gt_img.dataobj).astype(np.float32, copy=False)
    ct_properties = {
        "nibabel_stuff": {
            key: np.asarray(value, dtype=np.float64)
            for key, value in metadata_2mm["ct_properties"]["nibabel_stuff"].items()
        }
    }
    target_fzyx, _ = reorient_target_to_voxtell_layout(target_fxyz, ct_properties, name)
    target = crop_target(target_fzyx, metadata_2mm["crop_bbox_zyx"])
    return np.ascontiguousarray((target > 0).astype(np.uint8, copy=False))


def preprocess_global_case(
    entry: dict[str, Any],
    cache_2mm_root: Path,
    seg_dir: Path = REX_SEG_DIR,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_2mm, _targets_2mm, metadata_2mm = load_cached_case(cache_2mm_root, entry["name"])
    image, shape_4mm, before, after = downsample_2mm_image_to_global(image_2mm)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "name": entry["name"],
        "split": entry.get("_split"),
        "split_index": entry.get("_index"),
        "global_shape_zyx": list(GLOBAL_SHAPE),
        "global_spacing_mm": GLOBAL_SPACING_MM,
        "source_2mm_shape_zyx": [int(value) for value in image_2mm.shape[1:]],
        "shape_4mm_zyx": list(shape_4mm),
        "pad_before_zyx": before,
        "pad_after_zyx": after,
        "valid_global_bbox_zyx": [
            [int(left), int(left + size - 1)] for left, size in zip(before, shape_4mm)
        ],
        "native_cropped_shape_zyx": metadata_2mm["native_cropped_shape_zyx"],
        "native_original_reoriented_shape_zyx": metadata_2mm["original_reoriented_shape_zyx"],
        "native_crop_bbox_zyx": metadata_2mm["crop_bbox_zyx"],
        "orientation": metadata_2mm["orientation"],
        "ct_properties": metadata_2mm["ct_properties"],
        "prompts": [
            entry["findings"][key] for key in sorted(entry["findings"], key=lambda value: int(value))
        ],
        "categories": [
            entry.get("categories", {}).get(key)
            for key in sorted(entry["findings"], key=lambda value: int(value))
        ],
        "normalization": "inherited_full_native_crop_zscore_before_2mm_then_4mm_resampling",
        "image_resampling": "2mm_to_4mm_trilinear_align_corners_false_then_zero_center_pad",
    }
    native_targets = load_native_targets_from_metadata(entry["name"], metadata_2mm, seg_dir)
    metadata["targets"] = [
        proposal_boxes_for_native_target(target, metadata) for target in native_targets
    ]
    return image, native_targets, metadata


def load_global_case(
    cache_root: Path,
    name: str,
    load_targets: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    paths = global_case_paths(cache_root, name)
    if not paths["complete"].is_file():
        raise FileNotFoundError(f"Incomplete global proposal cache case: {paths['root']}")
    image = np.load(paths["image"], allow_pickle=False)
    targets = None
    if load_targets:
        with np.load(paths["native_targets"], allow_pickle=False) as data:
            targets = data["targets"]
    metadata = json.loads(paths["metadata"].read_text())
    return image, targets, metadata


def proposal_target_from_box(
    box_zyx: list[list[int]],
    margin_voxels: int = 0,
    valid_bbox_zyx: list[list[int]] | None = None,
) -> np.ndarray:
    target = np.zeros(GLOBAL_SHAPE, dtype=np.uint8)
    slices = []
    for axis, (low, high) in enumerate(box_zyx):
        axis_low = max(0, int(low) - margin_voxels)
        axis_high = min(GLOBAL_SHAPE[axis] - 1, int(high) + margin_voxels)
        if valid_bbox_zyx is not None:
            axis_low = max(axis_low, int(valid_bbox_zyx[axis][0]))
            axis_high = min(axis_high, int(valid_bbox_zyx[axis][1]))
        slices.append(slice(axis_low, axis_high + 1))
    target[tuple(slices)] = 1
    return target


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
    )


class GlobalProposalUNet(nn.Module):
    """Small FiLM-conditioned U-Net adapted from the external coarse branch."""

    def __init__(self, base_channels: int = 8, text_embedding_dim: int = 2560) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.enc1 = _conv_block(1, base_channels)
        self.down1 = nn.Conv3d(base_channels, base_channels * 2, 3, 2, 1, bias=False)
        self.enc2 = _conv_block(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 4, 3, 2, 1, bias=False)
        self.bottleneck = _conv_block(base_channels * 4, base_channels * 4)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = _conv_block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = _conv_block(base_channels * 2, base_channels)
        self.text_film = nn.Sequential(
            nn.Linear(text_embedding_dim, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 2),
        )
        self.out = nn.Conv3d(base_channels, 1, 1)

    def forward(self, image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        if text_embedding.ndim == 4:
            text_embedding = text_embedding.squeeze(2)
        x1 = self.enc1(image)
        x2 = self.enc2(self.down1(x1))
        xb = self.bottleneck(self.down2(x2))
        x = self.dec2(torch.cat([self.up2(xb), x2], dim=1))
        features = self.dec1(torch.cat([self.up1(x), x1], dim=1))
        batch, prompts, _ = text_embedding.shape
        gamma, beta = self.text_film(text_embedding).chunk(2, dim=-1)
        gamma = gamma.view(batch, prompts, -1, 1, 1, 1)
        beta = beta.view(batch, prompts, -1, 1, 1, 1)
        conditioned = features[:, None] * (1.0 + gamma) + beta
        logits = self.out(conditioned.reshape(batch * prompts, features.shape[1], *features.shape[2:]))
        return logits.reshape(batch, prompts, *features.shape[2:])


def extract_candidate_starts(
    probabilities: np.ndarray,
    metadata: dict[str, Any],
    top_k: int = 5,
    nms_radius: int = 8,
) -> list[dict[str, Any]]:
    scores = np.ascontiguousarray(probabilities, dtype=np.float32).copy()
    valid = np.zeros(GLOBAL_SHAPE, dtype=bool)
    valid_slices = tuple(
        slice(int(low), int(high) + 1) for low, high in metadata["valid_global_bbox_zyx"]
    )
    valid[valid_slices] = True
    scores[~valid] = -1.0
    candidates: list[dict[str, Any]] = []
    seen_starts: set[tuple[int, int, int]] = set()
    max_attempts = max(top_k * 64, 256)
    for _ in range(max_attempts):
        flat_index = int(np.argmax(scores))
        score = float(scores.flat[flat_index])
        if score < 0:
            break
        point = np.unravel_index(flat_index, GLOBAL_SHAPE)
        suppression = tuple(
            slice(max(0, axis - nms_radius), min(size, axis + nms_radius + 1))
            for axis, size in zip(point, GLOBAL_SHAPE)
        )
        scores[suppression] = -1.0
        starts = global_point_to_native_starts(point, metadata)
        starts_key = tuple(starts)
        if starts_key in seen_starts:
            continue
        seen_starts.add(starts_key)
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "score": score,
                "global_point_zyx": [int(value) for value in point],
                "native_patch_starts_zyx": starts,
            }
        )
        if len(candidates) >= top_k:
            break
    return candidates


def candidate_region_metrics(
    candidates: list[dict[str, Any]],
    native_target: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float | bool]:
    total = int(native_target.sum())
    output: dict[str, float | bool] = {}
    for k in ks:
        union = np.zeros(native_target.shape, dtype=bool)
        full_in_one = False
        for candidate in candidates[:k]:
            starts = candidate["native_patch_starts_zyx"]
            slices = tuple(
                slice(start, min(start + patch, dim))
                for start, patch, dim in zip(starts, LOCAL_PATCH_SIZE, native_target.shape)
            )
            union[slices] = True
            patch_covered = int(np.count_nonzero(native_target[slices]))
            full_in_one = full_in_one or patch_covered == total
        covered = int(np.count_nonzero(native_target.astype(bool) & union))
        output[f"hit_at_{k}"] = covered > 0
        output[f"full_inclusion_at_{k}"] = full_in_one
        output[f"target_coverage_at_{k}"] = covered / total if total else 0.0
    return output
