#!/usr/bin/env python3
"""Paper-aligned VoxTell text-conditioned fine-tuning for ReXGroundingCT.

This trainer keeps the public VoxTell architecture and trains it directly with
free-text prompt embeddings. The default baseline is intentionally narrow:
single-GPU, z-score normalized, `192^3` patches, three prompt slots per sample,
and the ReXGroundingCT orientation fix applied before patch sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from voxtell.inference.predictor import VoxTellPredictor, download_voxtell_model

from common import (
    CT_ROOT,
    EXP_ROOT,
    REX_METADATA,
    REX_SEG_DIR,
    command_string,
    ct_rate_abs_path,
    env_snapshot,
    git_commit,
    load_split_entries,
    sha256_file,
    sorted_prompts,
    utc_now_iso,
    write_json,
    REPO_ROOT,
    VOXTELL_SUBMODULE,
)
from poll_ct_subset import snapshot as ct_snapshot


DEFAULT_PATCH_SIZE = (192, 192, 192)
DEFAULT_DEEP_SUPERVISION_WEIGHTS = (1.0, 0.5, 0.25, 0.125, 0.0625)
EXPERIMENT_ID = "002_voxtell_text_ft_miccai_train_val"


@dataclass(frozen=True)
class PromptSlot:
    prompt: str
    target_index: int | None
    is_positive: bool


@dataclass
class CaseData:
    name: str
    image: np.ndarray
    targets: np.ndarray
    prompts: list[str]
    bbox: list[list[int]]
    orientation: dict[str, Any]


@dataclass
class PatchSample:
    image: np.ndarray
    target: np.ndarray
    prompts: list[str]
    stats: Counter
    event: dict[str, Any] | None = None


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = bool(args.distributed or world_size > 1)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            str(args.local_rank if args.local_rank is not None else args.gpu),
        )
    )
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return distributed, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serializable_ornt(ornt: np.ndarray) -> list[list[int | float]]:
    return [[int(axis), float(direction)] for axis, direction in ornt.tolist()]


def resolve_model_dir(model_dir: Path | None) -> Path:
    if model_dir is not None:
        return model_dir
    env_model = os.environ.get("VOXTELL_MODEL")
    if env_model:
        return Path(env_model)
    return Path(download_voxtell_model())


def parse_patch_size(values: list[int]) -> tuple[int, int, int]:
    patch_size = tuple(int(value) for value in values)
    if len(patch_size) != 3:
        raise ValueError("--patch-size must contain exactly 3 integers")
    if any(value <= 0 for value in patch_size):
        raise ValueError("--patch-size values must be positive")
    return patch_size


def parse_checkpoint_updates(value: str | None) -> set[int]:
    if value is None or not value.strip():
        return set()
    updates: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        update = int(item)
        if update <= 0:
            raise ValueError("--checkpoint-updates values must be positive")
        updates.add(update)
    return updates


def reorient_target_to_voxtell_layout(
    target_fxyz: np.ndarray,
    ct_properties: dict[str, Any],
    name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map released GT masks from raw `(F,X,Y,Z)` to VoxTell `(F,Z,Y,X)`."""
    if target_fxyz.ndim == 3:
        target_fxyz = target_fxyz[None]
    if target_fxyz.ndim != 4:
        raise ValueError(f"{name}: expected 4D target `(F,X,Y,Z)`, got {target_fxyz.shape}")

    nibabel_stuff = ct_properties.get("nibabel_stuff", {})
    original_affine = nibabel_stuff.get("original_affine")
    reoriented_affine = nibabel_stuff.get("reoriented_affine", original_affine)
    if original_affine is None or reoriented_affine is None:
        raise ValueError(f"{name}: missing CT affine metadata from NibabelIOWithReorient")

    original_ornt = io_orientation(original_affine)
    reoriented_ornt = io_orientation(reoriented_affine)
    transform = ornt_transform(original_ornt, reoriented_ornt)
    target_reoriented_fxyz = np.stack(
        [
            apply_orientation(target_fxyz[index], transform)
            for index in range(target_fxyz.shape[0])
        ],
        axis=0,
    )
    target_fzyx = np.transpose(target_reoriented_fxyz, (0, 3, 2, 1))
    target_fzyx = np.ascontiguousarray((target_fzyx > 0).astype(np.float32, copy=False))

    metadata = {
        "applied_transform": "original_ct_FXYZ_to_reoriented_FXYZ_then_nnunet_FZYX",
        "ct_original_axcodes": list(nib.aff2axcodes(original_affine)),
        "ct_reoriented_axcodes": list(nib.aff2axcodes(reoriented_affine)),
        "ct_original_orientation": serializable_ornt(original_ornt),
        "ct_reoriented_orientation": serializable_ornt(reoriented_ornt),
        "orientation_transform": serializable_ornt(transform),
        "raw_target_shape": [int(dim) for dim in target_fxyz.shape],
        "training_target_shape": [int(dim) for dim in target_fzyx.shape],
    }
    return target_fzyx, metadata


def crop_target(target: np.ndarray, bbox: list[list[int]]) -> np.ndarray:
    slices = tuple(slice(int(start), int(stop)) for start, stop in bbox)
    return np.ascontiguousarray(target[(slice(None), *slices)])


class CaseCache:
    def __init__(self, max_size: int):
        self.max_size = max(0, int(max_size))
        self._items: OrderedDict[str, CaseData] = OrderedDict()

    def get(self, key: str) -> CaseData | None:
        if self.max_size == 0 or key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key: str, value: CaseData) -> None:
        if self.max_size == 0:
            return
        if key in self._items:
            self._items.pop(key)
        self._items[key] = value
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)


class RexVoxTellPatchSampler:
    def __init__(
        self,
        entries: list[dict[str, Any]],
        negative_prompt_pool: list[str],
        ct_root: Path,
        seg_dir: Path,
        preprocessed_cache_dir: Path | None,
        patch_size: tuple[int, int, int],
        foreground_oversample_prob: float,
        seed: int,
        cache_size: int,
        require_positive_crop: bool = False,
        min_positive_voxels: int = 1,
        max_positive_crop_attempts: int = 64,
        max_sample_attempts: int = 128,
    ) -> None:
        if not entries:
            raise ValueError("Sampler requires at least one training entry")
        if not negative_prompt_pool:
            raise ValueError("Sampler requires a non-empty negative prompt pool")
        self.entries = entries
        self.negative_prompt_pool = negative_prompt_pool
        self.ct_root = ct_root
        self.seg_dir = seg_dir
        self.preprocessed_cache_dir = preprocessed_cache_dir
        self.patch_size = patch_size
        self.foreground_oversample_prob = float(foreground_oversample_prob)
        self.rng = np.random.default_rng(seed)
        self.reader = NibabelIOWithReorient()
        self.normalizer = ZScoreNormalization(intensityproperties={})
        self.cache = CaseCache(cache_size)
        self.require_positive_crop = bool(require_positive_crop)
        self.min_positive_voxels = int(min_positive_voxels)
        self.max_positive_crop_attempts = int(max_positive_crop_attempts)
        self.max_sample_attempts = int(max_sample_attempts)
        if self.min_positive_voxels < 1:
            raise ValueError("min_positive_voxels must be >= 1")
        if self.max_positive_crop_attempts < 1:
            raise ValueError("max_positive_crop_attempts must be >= 1")
        if self.max_sample_attempts < 1:
            raise ValueError("max_sample_attempts must be >= 1")

    def load_case(self, entry: dict[str, Any]) -> CaseData:
        name = entry["name"]
        cached = self.cache.get(name)
        if cached is not None:
            return cached

        if self.preprocessed_cache_dir is not None:
            from voxtell_preprocessed_cache import load_cached_case

            image, targets, metadata = load_cached_case(
                self.preprocessed_cache_dir,
                name,
                require_targets=True,
            )
            if targets is None:
                raise ValueError(f"{name}: cached case does not contain training targets")
            prompts = sorted_prompts(entry)
            if len(prompts) != int(targets.shape[0]):
                raise ValueError(f"{name}: {len(prompts)} prompts but {targets.shape[0]} cached targets")
            case = CaseData(
                name=name,
                image=np.ascontiguousarray(image, dtype=np.float32),
                targets=np.ascontiguousarray(targets, dtype=np.float32),
                prompts=prompts,
                bbox=metadata["crop_bbox_zyx"],
                orientation=metadata["orientation"],
            )
            self.cache.put(name, case)
            return case

        ct_path = ct_rate_abs_path(name, self.ct_root)
        gt_path = self.seg_dir / name
        if not ct_path.is_file():
            raise FileNotFoundError(f"Missing CT: {ct_path}")
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing segmentation: {gt_path}")

        image, ct_properties = self.reader.read_images([str(ct_path)])
        image = image.astype(np.float32, copy=True)

        gt_img = nib.load(str(gt_path))
        target_fxyz = np.asanyarray(gt_img.dataobj).astype(np.float32, copy=False)
        target_fzyx, orientation = reorient_target_to_voxtell_layout(target_fxyz, ct_properties, name)
        if tuple(image.shape[1:]) != tuple(target_fzyx.shape[1:]):
            raise ValueError(
                f"{name}: image shape {image.shape[1:]} != orientation-fixed target shape {target_fzyx.shape[1:]}"
            )

        image, _, bbox = crop_to_nonzero(image, None)
        image = self.normalizer.run(image, None).astype(np.float32, copy=False)
        bbox_list = [[int(v) for v in axis] for axis in bbox]
        target_fzyx = crop_target(target_fzyx, bbox_list)
        prompts = sorted_prompts(entry)
        if len(prompts) != int(target_fzyx.shape[0]):
            raise ValueError(
                f"{name}: {len(prompts)} prompts but {target_fzyx.shape[0]} target channels"
            )

        case = CaseData(
            name=name,
            image=np.ascontiguousarray(image),
            targets=np.ascontiguousarray(target_fzyx),
            prompts=prompts,
            bbox=bbox_list,
            orientation=orientation,
        )
        self.cache.put(name, case)
        return case

    def sample_slots(self, case: CaseData) -> tuple[list[PromptSlot], Counter]:
        stats: Counter = Counter()
        n_findings = len(case.prompts)
        if n_findings < 1:
            raise ValueError(f"{case.name}: no findings available")

        if n_findings >= 2:
            positive_indices = self.rng.choice(n_findings, size=2, replace=False).tolist()
            n_negative = 1
            stats["multi_finding_samples"] += 1
        else:
            positive_indices = [0]
            n_negative = 2
            stats["one_finding_fallback_samples"] += 1

        positive_slots = [
            PromptSlot(case.prompts[index], int(index), True)
            for index in positive_indices
        ]
        case_prompt_keys = {prompt.lower() for prompt in case.prompts}
        candidates = [prompt for prompt in self.negative_prompt_pool if prompt not in case_prompt_keys]
        if not candidates:
            candidates = self.negative_prompt_pool
        replace = len(candidates) < n_negative
        negative_prompts = self.rng.choice(candidates, size=n_negative, replace=replace).tolist()
        negative_slots = [PromptSlot(str(prompt), None, False) for prompt in negative_prompts]

        slots = positive_slots + negative_slots
        self.rng.shuffle(slots)
        stats["positive_slots"] += len(positive_slots)
        stats["negative_slots"] += len(negative_slots)
        return slots, stats

    def random_starts(
        self,
        spatial_shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        rng = rng or self.rng
        starts: list[int] = []
        for dim, patch in zip(spatial_shape, self.patch_size):
            if dim <= patch:
                starts.append(0)
            else:
                starts.append(int(rng.integers(0, dim - patch + 1)))
        return starts

    def anchored_starts(
        self,
        coord: np.ndarray,
        spatial_shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        rng = rng or self.rng
        starts: list[int] = []
        for axis, (dim, patch) in enumerate(zip(spatial_shape, self.patch_size)):
            if dim <= patch:
                starts.append(0)
                continue
            low = max(0, int(coord[axis]) - patch + 1)
            high = min(int(coord[axis]), dim - patch)
            if low > high:
                start = max(0, min(int(coord[axis]) - patch // 2, dim - patch))
            else:
                start = int(rng.integers(low, high + 1))
            starts.append(start)
        return starts

    def starts_containing_points(
        self,
        points: np.ndarray,
        spatial_shape: tuple[int, int, int],
        rng: np.random.Generator | None = None,
    ) -> list[int] | None:
        rng = rng or self.rng
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
        starts: list[int] = []
        for axis, (dim, patch) in enumerate(zip(spatial_shape, self.patch_size)):
            if dim <= patch:
                starts.append(0)
                continue
            low = max(0, int(points[:, axis].max()) - patch + 1)
            high = min(int(points[:, axis].min()), dim - patch)
            if low > high:
                return None
            starts.append(int(rng.integers(low, high + 1)))
        return starts

    def positive_point_candidates(
        self,
        coords: np.ndarray,
        rng: np.random.Generator,
        max_candidates: int = 32,
    ) -> np.ndarray:
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected positive coords with shape (N, 3), got {coords.shape}")
        if len(coords) <= max_candidates:
            return coords

        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        center = (bbox_min.astype(np.float32) + bbox_max.astype(np.float32)) / 2.0
        landmarks = [bbox_min, bbox_max, center]
        for z in (bbox_min[0], bbox_max[0]):
            for y in (bbox_min[1], bbox_max[1]):
                for x in (bbox_min[2], bbox_max[2]):
                    landmarks.append(np.asarray([z, y, x], dtype=np.float32))

        chosen: list[int] = []
        seen: set[int] = set()
        coords_float = coords.astype(np.float32, copy=False)
        for point in landmarks:
            distances = np.sum((coords_float - np.asarray(point, dtype=np.float32)) ** 2, axis=1)
            index = int(np.argmin(distances))
            if index not in seen:
                seen.add(index)
                chosen.append(index)

        remaining = max(0, max_candidates - len(chosen))
        if remaining:
            random_indices = rng.choice(len(coords), size=remaining, replace=False).tolist()
            for index in random_indices:
                index = int(index)
                if index not in seen:
                    seen.add(index)
                    chosen.append(index)
        return coords[np.asarray(chosen[:max_candidates], dtype=np.int64)]

    def positive_crop_starts(
        self,
        case: CaseData,
        slots: list[PromptSlot],
        spatial_shape: tuple[int, int, int],
        rng: np.random.Generator | None,
        stats: Counter,
    ) -> list[int] | None:
        rng = rng or self.rng
        positive_slots = [
            slot for slot in slots
            if slot.is_positive and slot.target_index is not None
        ]
        if not positive_slots:
            stats["positive_crop_no_positive_slots"] += 1
            return None

        coords_per_target: list[np.ndarray] = []
        for slot in positive_slots:
            assert slot.target_index is not None
            coords = np.argwhere(case.targets[slot.target_index] > 0)
            if coords.size == 0:
                stats["positive_crop_empty_full_mask"] += 1
                return None
            coords_per_target.append(coords)

        stats["positive_crop_requested"] += 1
        for attempt in range(1, self.max_positive_crop_attempts + 1):
            points = np.stack(
                [coords[int(rng.integers(0, len(coords)))] for coords in coords_per_target],
                axis=0,
            )
            starts = self.starts_containing_points(points, spatial_shape, rng)
            if starts is None:
                continue
            stats["positive_crop_success"] += 1
            stats["positive_crop_attempts"] += attempt
            return starts

        candidate_lists = [
            self.positive_point_candidates(coords, rng)
            for coords in coords_per_target
        ]
        if len(candidate_lists) == 1:
            points_to_try = candidate_lists[0][rng.permutation(len(candidate_lists[0]))]
            for offset, point in enumerate(points_to_try, start=1):
                starts = self.starts_containing_points(point[None, :], spatial_shape, rng)
                if starts is not None:
                    stats["positive_crop_success"] += 1
                    stats["positive_crop_attempts"] += self.max_positive_crop_attempts + offset
                    return starts
        elif len(candidate_lists) == 2:
            left_candidates = candidate_lists[0][rng.permutation(len(candidate_lists[0]))]
            right_candidates = candidate_lists[1][rng.permutation(len(candidate_lists[1]))]
            checks = 0
            for left in left_candidates:
                for right in right_candidates:
                    checks += 1
                    starts = self.starts_containing_points(
                        np.stack([left, right], axis=0),
                        spatial_shape,
                        rng,
                    )
                    if starts is not None:
                        stats["positive_crop_success"] += 1
                        stats["positive_crop_attempts"] += self.max_positive_crop_attempts + checks
                        return starts

        stats["positive_crop_failed"] += 1
        stats["positive_crop_attempts"] += self.max_positive_crop_attempts
        return None

    def extract_patch(self, array: np.ndarray, starts: list[int]) -> np.ndarray:
        output = np.zeros((array.shape[0], *self.patch_size), dtype=array.dtype)
        crop_slices = []
        insert_slices = []
        for axis, (start, patch) in enumerate(zip(starts, self.patch_size)):
            stop = min(start + patch, array.shape[axis + 1])
            crop_slices.append(slice(start, stop))
            insert_slices.append(slice(0, stop - start))
        cropped = array[(slice(None), *crop_slices)]
        output[(slice(None), *insert_slices)] = cropped
        return output

    def sample_event(self, event_index: int | None = None) -> dict[str, Any]:
        last_error: str | None = None
        for _ in range(self.max_sample_attempts):
            entry = self.entries[int(self.rng.integers(0, len(self.entries)))]
            case = self.load_case(entry)
            slots, stats = self.sample_slots(case)
            stats["samples"] += 1

            positive_slots = [slot for slot in slots if slot.is_positive]
            anchor_slot: PromptSlot | None = None
            one_finding_foreground_forced = bool(stats.get("one_finding_fallback_samples", 0))
            foreground_forced = self.require_positive_crop or one_finding_foreground_forced
            foreground_requested = (
                foreground_forced
                or bool(self.rng.random() < self.foreground_oversample_prob)
            )
            if foreground_requested and positive_slots:
                if one_finding_foreground_forced:
                    anchor_slot = positive_slots[0]
                    stats["one_finding_forced_foreground_anchor"] += 1
                elif self.require_positive_crop:
                    anchor_slot = positive_slots[int(self.rng.integers(0, len(positive_slots)))]
                    stats["positive_crop_forced_foreground_anchor"] += 1
                else:
                    anchor_slot = positive_slots[int(self.rng.integers(0, len(positive_slots)))]
                stats["foreground_oversample_requested"] += 1

            spatial_shape = tuple(int(dim) for dim in case.image.shape[1:])
            anchor_slot_index: int | None = None
            if self.require_positive_crop:
                starts = self.positive_crop_starts(
                    case=case,
                    slots=slots,
                    spatial_shape=spatial_shape,
                    rng=self.rng,
                    stats=stats,
                )
                if starts is None:
                    last_error = f"{case.name}: selected positive prompts cannot share a positive crop"
                    continue
            elif anchor_slot is not None and anchor_slot.target_index is not None:
                anchor_mask = case.targets[anchor_slot.target_index] > 0
                coords = np.argwhere(anchor_mask)
                if coords.size:
                    coord = coords[int(self.rng.integers(0, len(coords)))]
                    starts = self.anchored_starts(coord, spatial_shape)
                    stats["foreground_anchor_success"] += 1
                else:
                    starts = self.random_starts(spatial_shape)
                    stats["foreground_anchor_empty_full_mask"] += 1
            else:
                starts = self.random_starts(spatial_shape)
                stats["random_patch_samples"] += 1

            for slot_index, slot in enumerate(slots):
                if anchor_slot is not None and slot.target_index == anchor_slot.target_index:
                    anchor_slot_index = slot_index
                    break

            return {
                "event_index": event_index,
                "case_name": case.name,
                "source_split": entry.get("_split"),
                "source_split_index": entry.get("_index"),
                "patch_size": [int(value) for value in self.patch_size],
                "spatial_shape": [int(dim) for dim in spatial_shape],
                "patch_starts": [int(value) for value in starts],
                "prompt_slots": [
                    {
                        "prompt": slot.prompt,
                        "target_index": None if slot.target_index is None else int(slot.target_index),
                        "is_positive": bool(slot.is_positive),
                    }
                    for slot in slots
                ],
                "foreground_requested": bool(foreground_requested),
                "foreground_forced": bool(foreground_forced),
                "require_positive_crop": bool(self.require_positive_crop),
                "anchor_slot_index": anchor_slot_index,
                "anchor_target_index": None
                if anchor_slot is None or anchor_slot.target_index is None
                else int(anchor_slot.target_index),
                "stats": dict(stats),
            }
        raise RuntimeError(
            f"Failed to sample a valid patch event after {self.max_sample_attempts} attempts. "
            f"Last error: {last_error}"
        )

    def sample_from_event(self, event: dict[str, Any]) -> PatchSample:
        event = dict(event)
        if "patch_size" in event and tuple(int(value) for value in event["patch_size"]) != tuple(self.patch_size):
            raise ValueError(f"Scheduled event patch size {event['patch_size']} != sampler patch size {self.patch_size}")
        matching = [entry for entry in self.entries if entry["name"] == event["case_name"]]
        if not matching:
            raise KeyError(f"Scheduled case {event['case_name']!r} is not available to this sampler")
        case = self.load_case(matching[0])

        slots = [
            PromptSlot(
                prompt=str(slot["prompt"]),
                target_index=None if slot.get("target_index") is None else int(slot["target_index"]),
                is_positive=bool(slot["is_positive"]),
            )
            for slot in event["prompt_slots"]
        ]
        if len(slots) != 3:
            raise ValueError(f"Scheduled event must contain exactly 3 prompt slots, got {len(slots)}")
        for slot in slots:
            if slot.target_index is None:
                continue
            if slot.target_index < 0 or slot.target_index >= case.targets.shape[0]:
                raise ValueError(
                    f"{case.name}: scheduled target index {slot.target_index} out of range "
                    f"for {case.targets.shape[0]} targets"
                )
            expected_prompt = case.prompts[slot.target_index]
            if slot.prompt != expected_prompt:
                raise ValueError(
                    f"{case.name}: scheduled prompt {slot.prompt!r} does not match "
                    f"target {slot.target_index} prompt {expected_prompt!r}"
                )

        stats: Counter = Counter(event.get("stats", {}))
        if "patch_starts" in event:
            starts = [int(value) for value in event["patch_starts"]]
            if len(starts) != 3:
                raise ValueError(f"Scheduled event has invalid patch_starts: {starts}")
        else:
            if "patch_seed" not in event:
                raise ValueError("Scheduled event must contain either patch_starts or patch_seed")
            event_rng = np.random.default_rng(int(event["patch_seed"]))
            spatial_shape = tuple(int(dim) for dim in case.image.shape[1:])
            if self.require_positive_crop or bool(event.get("require_positive_crop", False)):
                starts = self.positive_crop_starts(
                    case=case,
                    slots=slots,
                    spatial_shape=spatial_shape,
                    rng=event_rng,
                    stats=stats,
                )
                if starts is None:
                    positive_targets = [
                        slot.target_index for slot in slots
                        if slot.is_positive and slot.target_index is not None
                    ]
                    raise RuntimeError(
                        f"{case.name}: scheduled event {event.get('event_index')} cannot produce "
                        f"a patch containing all positive targets {positive_targets}"
                    )
            elif (anchor_target_index := event.get("anchor_target_index")) is not None:
                anchor_mask = case.targets[int(anchor_target_index)] > 0
                coords = np.argwhere(anchor_mask)
                if coords.size:
                    coord = coords[int(event_rng.integers(0, len(coords)))]
                    starts = self.anchored_starts(coord, spatial_shape, rng=event_rng)
                    stats["foreground_anchor_success"] += 1
                else:
                    starts = self.random_starts(spatial_shape, rng=event_rng)
                    stats["foreground_anchor_empty_full_mask"] += 1
            else:
                starts = self.random_starts(spatial_shape, rng=event_rng)
                stats["random_patch_samples"] += 1
            event["spatial_shape"] = [int(dim) for dim in spatial_shape]
            event["patch_starts"] = [int(value) for value in starts]

        selected_targets = []
        zero_channel = np.zeros(case.targets.shape[1:], dtype=np.float32)
        for slot_index, slot in enumerate(slots):
            if slot.target_index is None:
                selected_targets.append(zero_channel)
                continue
            selected_targets.append(case.targets[slot.target_index])

        target_full = np.stack(selected_targets, axis=0).astype(np.float32, copy=False)
        image_patch = self.extract_patch(case.image, starts).astype(np.float32, copy=False)
        target_patch = (self.extract_patch(target_full, starts) > 0).astype(np.float32, copy=False)

        stats["scheduled_samples" if event.get("event_index") is not None else "unscheduled_samples"] += 1
        positive_nonempty = 0
        positive_below_minimum = 0
        for slot_index, slot in enumerate(slots):
            if slot.is_positive and target_patch[slot_index].any():
                positive_nonempty += 1
            if slot.is_positive and target_patch[slot_index].sum() < self.min_positive_voxels:
                positive_below_minimum += 1
        stats["positive_nonempty_patch_channels"] += positive_nonempty
        if self.require_positive_crop or bool(event.get("require_positive_crop", False)):
            stats["positive_crop_verified_samples"] += 1
            if positive_below_minimum:
                stats["positive_crop_verification_failed"] += 1
                raise RuntimeError(
                    f"{case.name}: scheduled event {event.get('event_index')} has "
                    f"{positive_below_minimum} positive target(s) below min voxels "
                    f"{self.min_positive_voxels} in patch starts {starts}"
                )
        anchor_slot_index = event.get("anchor_slot_index")
        if anchor_slot_index is not None and target_patch[int(anchor_slot_index)].any():
            stats["anchor_nonempty_patch_channels"] += 1

        return PatchSample(
            image=np.ascontiguousarray(image_patch),
            target=np.ascontiguousarray(target_patch),
            prompts=[slot.prompt for slot in slots],
            stats=stats,
            event=event,
        )

    def sample(self) -> PatchSample:
        return self.sample_from_event(self.sample_event())


def build_negative_prompt_pool(entries: list[dict[str, Any]]) -> list[str]:
    return sorted({prompt.lower() for entry in entries for prompt in sorted_prompts(entry)})


def load_embedding_bank(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing embedding bank: {path}. Run run_002_precompute_text_embeddings.sh first."
        )
    data = np.load(path)
    labels = [str(label).lower() for label in data["labels"]]
    embeddings = data["embeddings"]
    if len(labels) != int(embeddings.shape[0]):
        raise ValueError(f"{path}: labels and embeddings length mismatch")
    bank = {
        label: np.asarray(embeddings[index], dtype=np.float32)
        for index, label in enumerate(labels)
    }
    dims = {int(value.shape[0]) for value in bank.values()}
    if len(dims) != 1:
        raise ValueError(f"{path}: inconsistent embedding dimensions: {sorted(dims)}")
    return bank


def verify_embedding_coverage(bank: dict[str, np.ndarray], prompts: list[str]) -> None:
    missing = sorted({prompt.lower() for prompt in prompts if prompt.lower() not in bank})
    if missing:
        preview = ", ".join(repr(prompt) for prompt in missing[:10])
        raise KeyError(f"Embedding bank is missing {len(missing)} prompts; first missing: {preview}")


def batch_text_embeddings(
    prompt_batches: list[list[str]],
    bank: dict[str, np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    arrays = []
    for prompts in prompt_batches:
        arrays.append(np.stack([bank[prompt.lower()] for prompt in prompts], axis=0))
    return torch.from_numpy(np.stack(arrays, axis=0)).to(device=device, dtype=torch.float32)


def parse_deep_supervision_weights(value: str | None) -> tuple[float, ...]:
    if value is None or not value.strip():
        return DEFAULT_DEEP_SUPERVISION_WEIGHTS
    weights = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not weights:
        raise ValueError("--deep-supervision-weights must contain at least one value")
    if any(weight < 0 for weight in weights):
        raise ValueError("--deep-supervision-weights cannot contain negative values")
    if sum(weights) <= 0:
        raise ValueError("--deep-supervision-weights must sum to a positive value")
    return weights


def dice_loss_per_prompt_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    target = target.float()
    dims = tuple(range(2, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return dice_loss_per_prompt_from_logits(logits, target, eps=eps).mean()


def make_bce_voxel_weights(
    target: torch.Tensor,
    foreground_weight: float,
    boundary_weight: float,
    background_weight: float,
    boundary_radius: int,
) -> torch.Tensor:
    weights = torch.full_like(target, float(background_weight))
    foreground = target > 0.5
    weights = torch.where(foreground, torch.full_like(weights, float(foreground_weight)), weights)
    if boundary_radius <= 0 or boundary_weight == background_weight:
        return weights
    kernel = 2 * int(boundary_radius) + 1
    dilated = F.max_pool3d(target.float(), kernel_size=kernel, stride=1, padding=int(boundary_radius)) > 0.5
    outer_boundary = dilated & ~foreground
    return torch.where(outer_boundary, torch.full_like(weights, float(boundary_weight)), weights)


def bce_per_prompt_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    voxel_weighting: bool,
    foreground_weight: float,
    boundary_weight: float,
    background_weight: float,
    boundary_radius: int,
) -> torch.Tensor:
    bce_per_voxel = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    if not voxel_weighting:
        return bce_per_voxel.mean(dim=tuple(range(2, bce_per_voxel.ndim)))
    voxel_weights = make_bce_voxel_weights(
        target.float(),
        foreground_weight=foreground_weight,
        boundary_weight=boundary_weight,
        background_weight=background_weight,
        boundary_radius=boundary_radius,
    )
    weighted_sum = (bce_per_voxel * voxel_weights).sum(dim=tuple(range(2, bce_per_voxel.ndim)))
    norm = voxel_weights.sum(dim=tuple(range(2, voxel_weights.ndim))).clamp_min(1e-8)
    return weighted_sum / norm


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    bce_per_prompt = bce_per_prompt_with_logits(
        logits,
        target,
        voxel_weighting=args.voxel_bce_weighting,
        foreground_weight=args.bce_foreground_weight,
        boundary_weight=args.bce_boundary_weight,
        background_weight=args.bce_background_weight,
        boundary_radius=args.bce_boundary_radius,
    )
    dice_per_prompt = dice_loss_per_prompt_from_logits(logits, target)
    if args.loss_mode == "dice_bce":
        return (bce_per_prompt + dice_per_prompt).mean()
    if args.loss_mode == "empty_bce_only":
        target_nonempty = target.float().reshape(*target.shape[:2], -1).sum(dim=2) > 0.5
        nonempty_loss = bce_per_prompt + dice_per_prompt
        empty_loss = bce_per_prompt * float(args.empty_target_loss_weight)
        return torch.where(target_nonempty, nonempty_loss, empty_loss).mean()
    raise ValueError(f"Unsupported loss mode: {args.loss_mode}")


def deep_supervision_loss(
    outputs: torch.Tensor | list[torch.Tensor],
    target: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    configured_weights = parse_deep_supervision_weights(args.deep_supervision_weights)
    if len(configured_weights) < len(outputs):
        raise ValueError(
            f"Got {len(outputs)} deep-supervision outputs but only "
            f"{len(configured_weights)} configured weights"
        )
    weights = torch.as_tensor(
        configured_weights[: len(outputs)],
        device=target.device,
        dtype=torch.float32,
    )
    weights = weights / weights.sum()
    total = torch.zeros((), device=target.device, dtype=torch.float32)
    for weight, logits in zip(weights, outputs):
        target_at_scale = target
        if tuple(logits.shape[2:]) != tuple(target.shape[2:]):
            target_at_scale = F.interpolate(target.float(), size=logits.shape[2:], mode="nearest")
        total = total + weight * segmentation_loss(logits, target_at_scale, args)
    return total


def set_deep_supervision(network: torch.nn.Module, enabled: bool) -> None:
    model = network.module if hasattr(network, "module") else network
    model.deep_supervision = enabled
    if hasattr(model, "decoder"):
        model.decoder.deep_supervision = enabled


def optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    update_index_zero_based: int,
    total_updates: int,
    power: float,
    schedule: str,
    warmup_updates: int,
) -> float:
    step = int(update_index_zero_based) + 1
    warmup_updates = max(0, int(warmup_updates))
    if warmup_updates > 0 and step <= warmup_updates:
        factor = step / float(warmup_updates)
    elif schedule == "fixed":
        factor = 1.0
    elif schedule == "poly":
        if total_updates <= 0:
            factor = 1.0
        else:
            progress = (step - warmup_updates) / float(max(1, total_updates - warmup_updates))
            factor = (1.0 - min(max(progress, 0.0), 1.0)) ** power
    else:
        raise ValueError(f"Unsupported LR schedule: {schedule}")
    for group in optimizer.param_groups:
        group["lr"] = float(group.get("base_lr", base_lr)) * factor
    return float(optimizer.param_groups[0]["lr"])


def cuda_memory_report(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "cuda": False,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
            "max_allocated_gib": 0.0,
            "max_reserved_gib": 0.0,
            "device_name": None,
        }
    torch.cuda.synchronize(device)
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "cuda": True,
        "max_allocated_bytes": allocated,
        "max_reserved_bytes": reserved,
        "max_allocated_gib": allocated / float(1024**3),
        "max_reserved_gib": reserved / float(1024**3),
        "device_name": torch.cuda.get_device_name(device),
    }

def reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_batch(
    sampler: RexVoxTellPatchSampler,
    batch_size: int,
    embedding_bank: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Counter]:
    samples = [sampler.sample() for _ in range(batch_size)]
    images = torch.from_numpy(np.stack([sample.image for sample in samples], axis=0)).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    targets = torch.from_numpy(np.stack([sample.target for sample in samples], axis=0)).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    text_embeddings = batch_text_embeddings([sample.prompts for sample in samples], embedding_bank, device)
    stats: Counter = Counter()
    for sample in samples:
        stats.update(sample.stats)
    return images, targets, text_embeddings, stats


def run_optimizer_update(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sampler: RexVoxTellPatchSampler,
    embedding_bank: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    grad_accum: int,
    amp: bool,
    args: argparse.Namespace,
) -> tuple[float, Counter, float | None]:
    optimizer.zero_grad(set_to_none=True)
    stats: Counter = Counter()
    loss_values = []
    for _ in range(grad_accum):
        images, targets, text_embeddings, batch_stats = make_batch(
            sampler=sampler,
            batch_size=batch_size,
            embedding_bank=embedding_bank,
            device=device,
        )
        stats.update(batch_stats)
        with torch.autocast(device.type, enabled=device.type == "cuda" and amp):
            outputs = network(images, text_embeddings)
            loss = deep_supervision_loss(outputs, targets, args)
            scaled_loss = loss / grad_accum
        scaler.scale(scaled_loss).backward()
        loss_values.append(float(loss.detach().cpu()))
    grad_norm = None
    if args.clip_grad_norm and args.clip_grad_norm > 0:
        scaler.unscale_(optimizer)
        parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, float(args.clip_grad_norm))
        grad_norm = float(grad_norm_tensor.detach().cpu())
    scaler.step(optimizer)
    scaler.update()
    return float(np.mean(loss_values)), stats, grad_norm


def optimizer_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "optimizer": "SGD",
        "lr": args.lr,
        "encoder_lr": args.encoder_lr,
        "decoder_lr": args.decoder_lr,
        "lr_schedule": args.lr_schedule,
        "poly_power": args.poly_power,
        "warmup_updates": args.warmup_updates,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "nesterov": args.nesterov,
        "clip_grad_norm": args.clip_grad_norm,
    }


def loss_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "loss_mode": args.loss_mode,
        "empty_target_loss_weight": args.empty_target_loss_weight,
        "voxel_bce_weighting": args.voxel_bce_weighting,
        "bce_foreground_weight": args.bce_foreground_weight,
        "bce_boundary_weight": args.bce_boundary_weight,
        "bce_background_weight": args.bce_background_weight,
        "bce_boundary_radius": args.bce_boundary_radius,
        "deep_supervision_weights": list(parse_deep_supervision_weights(args.deep_supervision_weights)),
    }


def preprocessing_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.preprocessed_cache_dir is None:
        return {
            "mode": "on_the_fly",
            "preprocessed_cache_dir": None,
            "preprocess_id": "voxtell_default_on_the_fly",
            "normalization": "voxtell_default_crop_nonzero_zscore",
        }

    manifest_path = args.preprocessed_cache_dir / "manifest.json"
    config: dict[str, Any] = {
        "mode": "preprocessed_cache",
        "preprocessed_cache_dir": str(args.preprocessed_cache_dir),
        "manifest_path": str(manifest_path),
        "preprocess_id": "unknown",
        "normalization": "unknown",
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        config.update(
            {
                "manifest_sha256": sha256_file(manifest_path),
                "preprocess_id": manifest.get("preprocess_id", "unknown"),
                "normalization": manifest.get("normalization", "unknown"),
                "cases": manifest.get("cases"),
                "targets": manifest.get("targets"),
                "empty_targets": manifest.get("empty_targets"),
                "foreground_fallback_targets": manifest.get("foreground_fallback_targets"),
                "splits": manifest.get("splits"),
            }
        )
    else:
        config["manifest_missing"] = True
    return config


def sampler_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "foreground_oversample_probability": args.foreground_oversample_prob,
        "require_positive_crop": args.require_positive_crop,
        "min_positive_voxels": args.min_positive_voxels,
        "max_positive_crop_attempts": args.max_positive_crop_attempts,
        "max_sample_attempts": args.max_sample_attempts,
        "preprocessed_cache_dir": (
            str(args.preprocessed_cache_dir) if args.preprocessed_cache_dir is not None else None
        ),
        "preprocessing": preprocessing_config(args),
    }


def save_checkpoint(
    path: Path,
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_update: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = network.module if hasattr(network, "module") else network
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    preprocessing = preprocessing_config(args)
    torch.save(
        {
            "epoch": int(epoch),
            "global_update": int(global_update),
            "network_weights": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler.state_dict(),
            "experiment": args.experiment_id,
            "patch_size": list(parse_patch_size(args.patch_size)),
            "normalization": preprocessing["normalization"],
            "preprocessed_cache_dir": (
                str(args.preprocessed_cache_dir) if args.preprocessed_cache_dir is not None else None
            ),
            "preprocessing_config": preprocessing,
            "sampler_fallback": "one_finding_uses_1_positive_2_negatives",
            "lr_schedule": args.lr_schedule,
            "optimizer_config": optimizer_config(args),
            "loss_config": loss_config(args),
            "checkpoint_updates": sorted(parse_checkpoint_updates(args.checkpoint_updates)),
        },
        tmp,
    )
    os.replace(tmp, path)


def point_latest_checkpoint(latest: Path, checkpoint: Path) -> None:
    """Atomically make the rolling checkpoint a hard link to an immutable save."""
    latest.parent.mkdir(parents=True, exist_ok=True)
    tmp = latest.with_name(f".{latest.name}.tmp.{os.getpid()}")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    os.link(checkpoint, tmp)
    os.replace(tmp, latest)


def materialize_inference_model(model_dir: Path, checkpoint: Path, output_model_dir: Path) -> None:
    output_model_dir.mkdir(parents=True, exist_ok=True)
    (output_model_dir / "fold_0").mkdir(exist_ok=True)
    shutil.copy2(model_dir / "plans.json", output_model_dir / "plans.json")
    shutil.copy2(checkpoint, output_model_dir / "fold_0" / "checkpoint_final.pth")


def write_training_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        f"# {metrics.get('experiment', 'ReXGroundingCT')} Training Report",
        "",
        f"- Status: `{metrics['status']}`",
        f"- Epochs: `{metrics['epochs']}`",
        f"- Steps per epoch: `{metrics['steps_per_epoch']}` optimizer updates",
        f"- Total optimizer updates: `{metrics['total_updates']}`",
        f"- Batch size per update: `{metrics['batch_size']}` case-patches",
        f"- Gradient accumulation: `{metrics['grad_accum']}`",
        f"- LR schedule: `{metrics['lr_schedule']}`",
        f"- LR groups: `{metrics.get('lrs_final')}`",
        f"- Loss mode: `{metrics.get('loss_config', {}).get('loss_mode')}`",
        f"- Deep supervision weights: `{metrics.get('loss_config', {}).get('deep_supervision_weights')}`",
        f"- Require positive crop: `{metrics.get('sampler_config', {}).get('require_positive_crop')}`",
        f"- Patch size: `{metrics['patch_size']}`",
        f"- Sample schedule: `{metrics.get('sample_schedule', {}).get('path') if metrics.get('sample_schedule') else 'none'}`",
        f"- Elapsed seconds: `{metrics['elapsed_seconds']:.3f}`",
        f"- Updates per second: `{metrics['updates_per_second']:.6f}`",
        f"- Peak CUDA allocated GiB: `{metrics['memory']['max_allocated_gib']:.3f}`",
        f"- Peak CUDA reserved GiB: `{metrics['memory']['max_reserved_gib']:.3f}`",
        f"- Mean loss: `{metrics['mean_loss']:.6f}`",
        f"- Last loss: `{metrics['last_loss']:.6f}`",
        "",
        "## Sample Composition",
        "",
    ]
    for key, value in sorted(metrics["sample_stats"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            f"- Final checkpoint: `{metrics['checkpoint_final']}`",
            f"- Inference model dir: `{metrics['inference_model_dir']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_probe_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Experiment 002 Single-GPU Batch Probe",
        "",
        f"- Status: `{report['status']}`",
        f"- Largest successful batch size: `{report.get('largest_successful_batch_size')}`",
        f"- Patch size: `{report['patch_size']}`",
        "",
        "| Batch size | Status | Seconds/update | Peak allocated GiB | Peak reserved GiB | Loss |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        lines.append(
            "| {batch} | {status} | {seconds} | {allocated} | {reserved} | {loss} |".format(
                batch=result["batch_size"],
                status=result["status"],
                seconds=f"{result.get('mean_update_seconds', 0.0):.3f}" if result["status"] == "ok" else "n/a",
                allocated=f"{result.get('memory', {}).get('max_allocated_gib', 0.0):.3f}",
                reserved=f"{result.get('memory', {}).get('max_reserved_gib', 0.0):.3f}",
                loss=f"{result.get('mean_loss', 0.0):.6f}" if result["status"] == "ok" else "n/a",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def prepare_runtime_dirs(run_dir: Path) -> None:
    for child in ["config", "logs", "checkpoints", "reports", "eval", "model", "predictions"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def runtime_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    readiness: dict[str, Any],
    mode: str,
    rank: int,
    world_size: int,
    local_rank: int,
    model_dir: Path | None,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment_id,
        "mode": mode,
        "command": command_string(),
        "env": env_snapshot(),
        "repo_root": str(REPO_ROOT),
        "repo_commit": git_commit(REPO_ROOT),
        "voxtell_submodule": str(VOXTELL_SUBMODULE),
        "voxtell_commit": git_commit(VOXTELL_SUBMODULE),
        "config": config,
        "ct_readiness": readiness,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "model_dir": str(model_dir) if model_dir is not None else None,
        "run_dir": str(args.run_dir),
        "optimizer_config": optimizer_config(args),
        "loss_config": loss_config(args),
        "sampler_config": sampler_config(args),
        "preprocessing_config": preprocessing_config(args),
    }


def build_sampler(
    entries: list[dict[str, Any]],
    negative_prompt_pool: list[str],
    args: argparse.Namespace,
    seed: int,
) -> RexVoxTellPatchSampler:
    return RexVoxTellPatchSampler(
        entries=entries,
        negative_prompt_pool=negative_prompt_pool,
        ct_root=args.ct_root,
        seg_dir=args.seg_dir,
        preprocessed_cache_dir=args.preprocessed_cache_dir,
        patch_size=parse_patch_size(args.patch_size),
        foreground_oversample_prob=args.foreground_oversample_prob,
        seed=seed,
        cache_size=args.case_cache_size,
        require_positive_crop=args.require_positive_crop,
        min_positive_voxels=args.min_positive_voxels,
        max_positive_crop_attempts=args.max_positive_crop_attempts,
        max_sample_attempts=args.max_sample_attempts,
    )


def load_sample_schedule(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing sample schedule: {path}")
    events: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("event_index") is None:
                event["event_index"] = line_index
            if int(event["event_index"]) != len(events):
                raise ValueError(
                    f"{path}: event_index {event['event_index']} does not match "
                    f"zero-based line order {len(events)}"
                )
            events.append(event)
    if not events:
        raise ValueError(f"Sample schedule is empty: {path}")
    return events


class ScheduledPatchSampler:
    """Consume a global patch-event stream deterministically across ranks."""

    def __init__(
        self,
        base_sampler: RexVoxTellPatchSampler,
        events: list[dict[str, Any]],
        rank: int,
        world_size: int,
        local_events_per_update: int,
        required_global_events: int,
    ) -> None:
        if local_events_per_update < 1:
            raise ValueError("local_events_per_update must be >= 1")
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if required_global_events > len(events):
            raise ValueError(
                f"Schedule has {len(events)} events but this run requires "
                f"{required_global_events} global patch events"
            )
        self.base_sampler = base_sampler
        self.events = events
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_events_per_update = int(local_events_per_update)
        self.required_global_events = int(required_global_events)
        self.local_sample_count = 0
        self.global_events_per_update = self.local_events_per_update * self.world_size

    def sample(self) -> PatchSample:
        local_index = self.local_sample_count
        update_index = local_index // self.local_events_per_update
        within_local_update = local_index % self.local_events_per_update
        event_index = (
            update_index * self.global_events_per_update
            + self.rank * self.local_events_per_update
            + within_local_update
        )
        if event_index >= self.required_global_events:
            raise IndexError(
                f"Schedule exhausted on rank {self.rank}: event {event_index} "
                f">= required {self.required_global_events}"
            )
        self.local_sample_count += 1
        return self.base_sampler.sample_from_event(self.events[event_index])


def sample_schedule_summary(
    args: argparse.Namespace,
    events: list[dict[str, Any]] | None,
    world_size: int,
    required_global_events: int | None = None,
    local_events_per_update: int | None = None,
) -> dict[str, Any] | None:
    if args.sample_schedule is None or events is None:
        return None
    if local_events_per_update is None:
        local_events_per_update = int(args.batch_size * args.grad_accum)
    if required_global_events is None:
        total_updates = int(args.epochs * args.steps_per_epoch)
        required_global_events = int(total_updates * local_events_per_update * world_size)
    return {
        "path": str(args.sample_schedule),
        "sha256": sha256_file(args.sample_schedule),
        "available_events": len(events),
        "required_global_events": required_global_events,
        "local_events_per_update": int(local_events_per_update),
        "global_events_per_update": int(local_events_per_update * world_size),
        "consumption_rule": "event_index = update * global_events_per_update + rank * local_events_per_update + local_offset",
    }


def create_predictor_and_network(
    args: argparse.Namespace,
    model_dir: Path,
    device: torch.device,
) -> tuple[VoxTellPredictor, torch.nn.Module]:
    predictor = VoxTellPredictor(
        model_dir=str(model_dir),
        device=device,
        embedding_bank=str(args.embeddings),
        use_precomputed_embeddings=False,
    )
    network = predictor.network.to(device)
    set_deep_supervision(network, True)
    network.train()
    return predictor, network


def configure_optimizer(args: argparse.Namespace, network: torch.nn.Module) -> torch.optim.Optimizer:
    encoder_lr = args.encoder_lr
    decoder_lr = args.decoder_lr
    if encoder_lr is not None or decoder_lr is not None:
        encoder_lr = float(args.lr if encoder_lr is None else encoder_lr)
        decoder_lr = float(args.lr if decoder_lr is None else decoder_lr)
        encoder_params = []
        decoder_params = []
        for name, parameter in network.named_parameters():
            if not parameter.requires_grad:
                continue
            normalized_name = name.removeprefix("module.")
            if normalized_name.startswith("encoder."):
                encoder_params.append(parameter)
            else:
                decoder_params.append(parameter)
        groups = []
        if encoder_params:
            groups.append(
                {
                    "params": encoder_params,
                    "lr": encoder_lr,
                    "base_lr": encoder_lr,
                    "name": "encoder",
                }
            )
        if decoder_params:
            groups.append(
                {
                    "params": decoder_params,
                    "lr": decoder_lr,
                    "base_lr": decoder_lr,
                    "name": "decoder",
                }
            )
        return torch.optim.SGD(
            groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    return torch.optim.SGD(
        [
            {
                "params": [parameter for parameter in network.parameters() if parameter.requires_grad],
                "lr": args.lr,
                "base_lr": args.lr,
                "name": "all",
            }
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )


def run_sample_test(args: argparse.Namespace, train_entries: list[dict[str, Any]], negative_pool: list[str]) -> int:
    base_sampler = build_sampler(train_entries, negative_pool, args, args.seed)
    schedule_events = load_sample_schedule(args.sample_schedule) if args.sample_schedule else None
    sampler: Any = base_sampler
    if schedule_events is not None:
        sampler = ScheduledPatchSampler(
            base_sampler=base_sampler,
            events=schedule_events,
            rank=0,
            world_size=1,
            local_events_per_update=1,
            required_global_events=args.sample_test_steps,
        )
    stats: Counter = Counter()
    examples = []
    for _ in range(args.sample_test_steps):
        sample = sampler.sample()
        stats.update(sample.stats)
        if len(examples) < 3:
            examples.append(
                {
                    "image_shape": list(sample.image.shape),
                    "target_shape": list(sample.target.shape),
                    "prompts": sample.prompts,
                    "target_nonempty": [bool(sample.target[index].any()) for index in range(sample.target.shape[0])],
                    "event_index": None if sample.event is None else sample.event.get("event_index"),
                }
            )
    report = {
        "status": "completed",
        "samples": args.sample_test_steps,
        "patch_size": list(parse_patch_size(args.patch_size)),
        "sample_schedule": sample_schedule_summary(
            args,
            schedule_events,
            world_size=1,
            required_global_events=args.sample_test_steps,
            local_events_per_update=1,
        ),
        "sample_stats": dict(stats),
        "examples": examples,
    }
    write_json(args.run_dir / "reports" / "sample_test.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def run_batch_probe(
    args: argparse.Namespace,
    train_entries: list[dict[str, Any]],
    negative_pool: list[str],
    embedding_bank: dict[str, np.ndarray],
    model_dir: Path,
    device: torch.device,
    distributed: bool,
    rank: int,
    main_process: bool,
) -> int:
    if distributed:
        raise ValueError("Batch probe is defined for one GPU only. Run without --distributed.")
    if args.sample_schedule is not None:
        raise ValueError("Batch probe does not use --sample-schedule; probe unscheduled throughput instead.")

    _predictor, network = create_predictor_and_network(args, model_dir, device)
    optimizer = configure_optimizer(args, network)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    sampler = build_sampler(train_entries, negative_pool, args, args.seed + rank)

    report: dict[str, Any] = {
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "patch_size": list(parse_patch_size(args.patch_size)),
        "min_batch_size": args.min_batch_size,
        "max_batch_size": args.max_batch_size,
        "probe_steps": args.probe_steps,
        "results": [],
        "largest_successful_batch_size": None,
    }

    for batch_size in range(args.min_batch_size, args.max_batch_size + 1):
        reset_cuda_peak(device)
        losses = []
        update_seconds = []
        stats: Counter = Counter()
        try:
            for _ in range(args.probe_steps):
                synchronize(device)
                start = time.perf_counter()
                loss, update_stats, _grad_norm = run_optimizer_update(
                    network=network,
                    optimizer=optimizer,
                    scaler=scaler,
                    sampler=sampler,
                    embedding_bank=embedding_bank,
                    device=device,
                    batch_size=batch_size,
                    grad_accum=args.grad_accum,
                    amp=args.amp,
                    args=args,
                )
                synchronize(device)
                update_seconds.append(time.perf_counter() - start)
                losses.append(loss)
                stats.update(update_stats)
            memory = cuda_memory_report(device)
            result = {
                "batch_size": batch_size,
                "status": "ok",
                "mean_update_seconds": float(np.mean(update_seconds)),
                "mean_loss": float(np.mean(losses)),
                "memory": memory,
                "sample_stats": dict(stats),
            }
            report["largest_successful_batch_size"] = batch_size
            report["results"].append(result)
            if main_process:
                write_json(args.run_dir / "reports" / "batch_probe.json", report)
                write_probe_report(args.run_dir / "reports" / "batch_probe.md", report)
        except BaseException as exc:
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            status = "oom" if is_oom_error(exc) else "error"
            report["results"].append(
                {
                    "batch_size": batch_size,
                    "status": status,
                    "error": f"{type(exc).__name__}: {exc}",
                    "memory": cuda_memory_report(device) if device.type == "cuda" else {},
                }
            )
            if main_process:
                write_json(args.run_dir / "reports" / "batch_probe.json", report)
                write_probe_report(args.run_dir / "reports" / "batch_probe.md", report)
            if status == "oom":
                break
            raise

    ok_results = [result for result in report["results"] if result["status"] == "ok"]
    if ok_results:
        fastest = min(
            ok_results,
            key=lambda result: result["mean_update_seconds"] / max(1, result["batch_size"]),
        )
        report["recommended_batch_size"] = fastest["batch_size"]
        report["recommendation_basis"] = "fastest_observed_seconds_per_case_patch"
        report["largest_successful_batch_size"] = max(result["batch_size"] for result in ok_results)

    if main_process:
        write_json(args.run_dir / "reports" / "batch_probe.json", report)
        write_probe_report(args.run_dir / "reports" / "batch_probe.md", report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def run_train(
    args: argparse.Namespace,
    train_entries: list[dict[str, Any]],
    negative_pool: list[str],
    embedding_bank: dict[str, np.ndarray],
    model_dir: Path,
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
    local_rank: int,
    main_process: bool,
    manifest: dict[str, Any],
) -> int:
    _predictor, network = create_predictor_and_network(args, model_dir, device)
    if distributed:
        network = DDP(
            network,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    optimizer = configure_optimizer(args, network)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    total_updates = int(args.epochs * args.steps_per_epoch)
    checkpoint_updates = parse_checkpoint_updates(args.checkpoint_updates)
    if checkpoint_updates:
        out_of_horizon = [update for update in sorted(checkpoint_updates) if update > total_updates]
        if out_of_horizon:
            raise ValueError(
                f"--checkpoint-updates contains updates beyond this run horizon "
                f"({total_updates}): {out_of_horizon}"
            )
    schedule_events = load_sample_schedule(args.sample_schedule) if args.sample_schedule else None
    if schedule_events is None:
        sampler: Any = build_sampler(train_entries[rank::world_size], negative_pool, args, args.seed + rank)
    else:
        base_sampler = build_sampler(train_entries, negative_pool, args, args.seed)
        local_events_per_update = int(args.batch_size * args.grad_accum)
        required_global_events = int(total_updates * local_events_per_update * world_size)
        sampler = ScheduledPatchSampler(
            base_sampler=base_sampler,
            events=schedule_events,
            rank=rank,
            world_size=world_size,
            local_events_per_update=local_events_per_update,
            required_global_events=required_global_events,
        )
        manifest["sample_schedule"] = sample_schedule_summary(args, schedule_events, world_size)

    global_update = 0
    losses: list[float] = []
    update_records: list[dict[str, Any]] = []
    sample_stats: Counter = Counter()
    reset_cuda_peak(device)
    start_time = time.perf_counter()

    progress = tqdm(
        total=total_updates,
        desc=f"{args.experiment_id} training updates",
        disable=not main_process,
    )
    for epoch in range(1, args.epochs + 1):
        for step_in_epoch in range(1, args.steps_per_epoch + 1):
            lr = set_learning_rate(
                optimizer=optimizer,
                base_lr=args.lr,
                update_index_zero_based=global_update,
                total_updates=total_updates,
                power=args.poly_power,
                schedule=args.lr_schedule,
                warmup_updates=args.warmup_updates,
            )
            synchronize(device)
            update_start = time.perf_counter()
            loss, stats, grad_norm = run_optimizer_update(
                network=network,
                optimizer=optimizer,
                scaler=scaler,
                sampler=sampler,
                embedding_bank=embedding_bank,
                device=device,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                amp=args.amp,
                args=args,
            )
            synchronize(device)
            elapsed = time.perf_counter() - update_start
            loss_tensor = torch.tensor(loss, device=device)
            if distributed:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensor /= world_size
            loss = float(loss_tensor.detach().cpu())
            global_update += 1
            sample_stats.update(stats)
            losses.append(loss)
            if main_process:
                update_records.append(
                    {
                        "global_update": global_update,
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "loss": loss,
                        "lr": lr,
                        "lrs": optimizer_lrs(optimizer),
                        "grad_norm": grad_norm,
                        "update_seconds": elapsed,
                    }
                )
                progress.update(1)
                progress.set_postfix(loss=f"{loss:.4f}", lr=f"{optimizer_lr(optimizer):.2e}")
                immutable_checkpoint_written = False
                checkpoint_step_path = None
                write_immutable_checkpoint = bool(
                    global_update in checkpoint_updates
                    or (
                        args.checkpoint_every_updates
                        and global_update % args.checkpoint_every_updates == 0
                    )
                )
                if write_immutable_checkpoint:
                    checkpoint_step_path = (
                        args.run_dir / "checkpoints" / f"checkpoint_update_{global_update:06d}.pth"
                    )
                    save_checkpoint(
                        checkpoint_step_path,
                        network,
                        optimizer,
                        scaler,
                        epoch,
                        global_update,
                        args,
                    )
                    immutable_checkpoint_written = True
                latest_interval = (
                    args.latest_checkpoint_every_updates
                    if args.latest_checkpoint_every_updates is not None
                    else args.checkpoint_every_updates
                )
                if latest_interval and global_update % latest_interval == 0:
                    latest_path = args.run_dir / "checkpoints" / "checkpoint_latest.pth"
                    if immutable_checkpoint_written and checkpoint_step_path is not None:
                        point_latest_checkpoint(latest_path, checkpoint_step_path)
                    else:
                        save_checkpoint(
                            latest_path,
                            network,
                            optimizer,
                            scaler,
                            epoch,
                            global_update,
                            args,
                        )
                    partial_metrics = {
                        "status": "running",
                        "global_update": global_update,
                        "mean_loss": float(np.mean(losses)),
                        "last_loss": float(losses[-1]),
                        "last_checkpoint": str(
                            checkpoint_step_path
                            if checkpoint_step_path is not None
                            else args.run_dir / "checkpoints" / "checkpoint_latest.pth"
                        ),
                        "sample_stats": dict(sample_stats),
                        "updates": update_records,
                    }
                    write_json(args.run_dir / "reports" / "training_metrics.json", partial_metrics)

    if distributed:
        dist.barrier()
    if main_process:
        progress.close()
        elapsed_total = time.perf_counter() - start_time
        final_checkpoint = args.run_dir / "checkpoints" / "checkpoint_final.pth"
        save_checkpoint(final_checkpoint, network, optimizer, scaler, args.epochs, global_update, args)
        inference_model_dir = None
        if args.materialize_final_model:
            inference_model_dir = args.run_dir / "model"
            materialize_inference_model(model_dir, final_checkpoint, inference_model_dir)
        metrics = {
            "status": "completed",
            "created_at_utc": utc_now_iso(),
            "experiment": args.experiment_id,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "total_updates": total_updates,
            "completed_updates": global_update,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "world_size": world_size,
            "patch_size": list(parse_patch_size(args.patch_size)),
            "lr_schedule": args.lr_schedule,
            "optimizer_config": optimizer_config(args),
            "loss_config": loss_config(args),
            "sampler_config": sampler_config(args),
            "preprocessing_config": preprocessing_config(args),
            "checkpoint_updates": sorted(checkpoint_updates),
            "elapsed_seconds": elapsed_total,
            "updates_per_second": global_update / elapsed_total if elapsed_total > 0 else None,
            "mean_update_seconds": elapsed_total / global_update if global_update else None,
            "mean_loss": float(np.mean(losses)),
            "last_loss": float(losses[-1]),
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            "lr_final": optimizer_lr(optimizer),
            "lrs_final": optimizer_lrs(optimizer),
            "memory": cuda_memory_report(device),
            "sample_schedule": sample_schedule_summary(args, schedule_events, world_size),
            "sample_stats": dict(sample_stats),
            "updates": update_records,
            "checkpoint_final": str(final_checkpoint),
            "inference_model_dir": str(inference_model_dir) if inference_model_dir is not None else None,
        }
        write_json(args.run_dir / "reports" / "training_metrics.json", metrics)
        write_training_report(args.run_dir / "reports" / "training_report.md", metrics)
        manifest.update(
            {
                "status": "completed",
                "training_metrics": str(args.run_dir / "reports" / "training_metrics.json"),
                "training_report": str(args.run_dir / "reports" / "training_report.md"),
                "checkpoint_final": str(final_checkpoint),
                "inference_model_dir": (
                    str(inference_model_dir) if inference_model_dir is not None else None
                ),
                "updated_at_utc": utc_now_iso(),
            }
        )
        write_json(args.run_dir / "run_manifest.json", manifest)
        print(f"Wrote final checkpoint: {final_checkpoint}")
        if inference_model_dir is not None:
            print(f"Wrote inference model: {inference_model_dir}")
        print(f"Peak allocated GiB: {metrics['memory']['max_allocated_gib']:.3f}")
        print(f"Elapsed seconds: {elapsed_total:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "batch-probe", "sample-test"], default="train")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/configs/experiments/002_voxtell_text_ft_miccai_train_val.json"),
    )
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--exp-dir", type=Path, default=EXP_ROOT / EXPERIMENT_ID)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--patch-size", nargs=3, type=int, default=list(DEFAULT_PATCH_SIZE))
    parser.add_argument("--allow-non192-patch", action="store_true")
    parser.add_argument("--foreground-oversample-prob", type=float, default=0.85)
    parser.add_argument("--case-cache-size", type=int, default=4)
    parser.add_argument("--sample-schedule", type=Path, default=None)
    parser.add_argument("--preprocessed-cache-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["fixed", "poly"], default="fixed")
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-grad-norm", type=float, default=0.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every-updates", type=int, default=100)
    parser.add_argument(
        "--checkpoint-updates",
        default="",
        help="Optional comma-separated immutable checkpoint update numbers.",
    )
    parser.add_argument("--latest-checkpoint-every-updates", type=int, default=None)
    parser.add_argument(
        "--materialize-final-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--loss-mode", choices=["dice_bce", "empty_bce_only"], default="dice_bce")
    parser.add_argument("--empty-target-loss-weight", type=float, default=0.5)
    parser.add_argument("--voxel-bce-weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bce-foreground-weight", type=float, default=1.0)
    parser.add_argument("--bce-boundary-weight", type=float, default=1.5)
    parser.add_argument("--bce-background-weight", type=float, default=0.5)
    parser.add_argument("--bce-boundary-radius", type=int, default=10)
    parser.add_argument(
        "--deep-supervision-weights",
        default=",".join(str(value) for value in DEFAULT_DEEP_SUPERVISION_WEIGHTS),
    )
    parser.add_argument("--require-positive-crop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-positive-voxels", type=int, default=1)
    parser.add_argument("--max-positive-crop-attempts", type=int, default=64)
    parser.add_argument("--max-sample-attempts", type=int, default=128)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--probe-steps", type=int, default=2)
    parser.add_argument("--sample-test-steps", type=int, default=8)
    parser.add_argument("--allow-experimental-train", action="store_true")
    args = parser.parse_args()

    patch_size = parse_patch_size(args.patch_size)
    if patch_size != DEFAULT_PATCH_SIZE and not args.allow_non192_patch:
        raise ValueError(
            "Experiment 002 baseline is fixed to 192^3. Non-192 fine-tuning also "
            "requires positional-embedding and inference export handling; pass "
            "--allow-non192-patch only for an explicit future ablation."
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.warmup_updates < 0:
        raise ValueError("--warmup-updates must be >= 0")
    if args.clip_grad_norm < 0:
        raise ValueError("--clip-grad-norm must be >= 0")
    if args.checkpoint_every_updates < 0:
        raise ValueError("--checkpoint-every-updates must be >= 0")
    if args.latest_checkpoint_every_updates is not None and args.latest_checkpoint_every_updates < 0:
        raise ValueError("--latest-checkpoint-every-updates must be >= 0")
    if args.empty_target_loss_weight < 0:
        raise ValueError("--empty-target-loss-weight must be >= 0")
    if args.bce_boundary_radius < 0:
        raise ValueError("--bce-boundary-radius must be >= 0")
    if args.min_positive_voxels < 1:
        raise ValueError("--min-positive-voxels must be >= 1")
    if args.max_positive_crop_attempts < 1:
        raise ValueError("--max-positive-crop-attempts must be >= 1")
    if args.max_sample_attempts < 1:
        raise ValueError("--max-sample-attempts must be >= 1")
    parse_deep_supervision_weights(args.deep_supervision_weights)
    if args.epochs < 1 and args.mode == "train":
        raise ValueError("--epochs must be >= 1")
    if args.steps_per_epoch < 1 and args.mode == "train":
        raise ValueError("--steps-per-epoch must be >= 1")
    if not (0.0 <= args.foreground_oversample_prob <= 1.0):
        raise ValueError("--foreground-oversample-prob must be between 0 and 1")
    parse_checkpoint_updates(args.checkpoint_updates)

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    if args.run_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_dir = args.exp_dir / "runs" / f"{args.mode}_{timestamp}"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    prepare_runtime_dirs(args.run_dir)
    if args.embeddings is None:
        args.embeddings = args.exp_dir / "config" / "rex_text_embeddings.npz"

    distributed, rank, world_size, local_rank = setup_distributed(args)
    main_process = is_main_process(rank)
    seed_everything(args.seed + rank)
    if args.mode in {"train", "batch-probe"} and not args.allow_experimental_train:
        if main_process:
            print("Training/probing is gated. Pass --allow-experimental-train.")
        cleanup_distributed()
        return 3

    device = torch.device(f"cuda:{local_rank if distributed else args.gpu}" if torch.cuda.is_available() else "cpu")
    model_dir = None if args.mode == "sample-test" else resolve_model_dir(args.model_dir)
    config = json.loads(args.config.read_text()) if args.config.exists() else {}
    readiness = ct_snapshot(args.metadata, args.ct_root, ["train", "val"])
    train_entries = load_split_entries(args.metadata, ["train"])
    val_entries = load_split_entries(args.metadata, ["val"])
    negative_pool = build_negative_prompt_pool(train_entries)
    all_train_prompts = [prompt for entry in train_entries for prompt in sorted_prompts(entry)]

    if main_process:
        manifest = runtime_manifest(args, config, readiness, args.mode, rank, world_size, local_rank, model_dir)
        manifest.update(
            {
                "status": "starting",
                "train_cases": len(train_entries),
                "val_cases": len(val_entries),
                "negative_prompt_pool_size": len(negative_pool),
                "embedding_bank": str(args.embeddings),
                "sample_schedule_path": str(args.sample_schedule) if args.sample_schedule else None,
            }
        )
        write_json(args.run_dir / "run_manifest.json", manifest)
    else:
        manifest = {}

    if not readiness["ready"]:
        if main_process:
            print("CT train+val data is not complete. Wrote manifest and exited.")
            print(f"Present {readiness['present_files']}/{readiness['expected_files']} files")
        cleanup_distributed()
        return 2

    if args.mode == "sample-test":
        code = run_sample_test(args, train_entries, negative_pool)
        cleanup_distributed()
        return code

    embedding_bank = load_embedding_bank(args.embeddings)
    verify_embedding_coverage(embedding_bank, all_train_prompts)

    if args.mode == "batch-probe":
        code = run_batch_probe(
            args=args,
            train_entries=train_entries,
            negative_pool=negative_pool,
            embedding_bank=embedding_bank,
            model_dir=model_dir,
            device=device,
            distributed=distributed,
            rank=rank,
            main_process=main_process,
        )
    else:
        code = run_train(
            args=args,
            train_entries=train_entries,
            negative_pool=negative_pool,
            embedding_bank=embedding_bank,
            model_dir=model_dir,
            device=device,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            main_process=main_process,
            manifest=manifest,
        )
    cleanup_distributed()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
