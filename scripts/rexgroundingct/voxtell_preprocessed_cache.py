#!/usr/bin/env python3
"""Shared VoxTell preprocessing cache helpers for ReXGroundingCT."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.orientations import io_orientation, ornt_transform
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization

from common import CT_ROOT, REX_SEG_DIR, ct_rate_abs_path, sorted_prompts
from train_text_conditioned_voxtell import (
    crop_target,
    reorient_target_to_voxtell_layout,
    serializable_ornt,
)


CACHE_SCHEMA_VERSION = 1
NATIVE_PREPROCESS_ID = "crop_zscore_native_v1"
ISO2MM_PREPROCESS_ID = "crop_zscore_2mm_v1"
SUPPORTED_PREPROCESS_IDS = (NATIVE_PREPROCESS_ID, ISO2MM_PREPROCESS_ID)
TARGET_SPACING_2MM_ZYX = (2.0, 2.0, 2.0)
STANDARD_CACHE_ROOT = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell"
)


def cache_case_key(name: str) -> str:
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def cache_case_dir(cache_root: Path, name: str) -> Path:
    return Path(cache_root) / "cases" / cache_case_key(name)


def cache_case_paths(cache_root: Path, name: str) -> dict[str, Path]:
    root = cache_case_dir(cache_root, name)
    return {
        "root": root,
        "image": root / "image.npy",
        "targets": root / "targets.npz",
        "metadata": root / "metadata.json",
        "complete": root / ".complete",
    }


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npy")
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def output_shape_for_spacing(
    native_shape_zyx: tuple[int, int, int],
    native_spacing_zyx: tuple[float, float, float],
    target_spacing_zyx: tuple[float, float, float] = TARGET_SPACING_2MM_ZYX,
) -> tuple[int, int, int]:
    return tuple(
        max(1, int(round(float(size) * float(spacing) / float(target))))
        for size, spacing, target in zip(native_shape_zyx, native_spacing_zyx, target_spacing_zyx)
    )


def resample_image_align_corners_false(
    image_czyx: np.ndarray,
    output_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(image_czyx))[None].float()
    resized = F.interpolate(
        tensor,
        size=output_shape_zyx,
        mode="trilinear",
        align_corners=False,
    )[0]
    return np.ascontiguousarray(resized.numpy().astype(np.float32, copy=False))


def _foreground_center_splat(
    source_mask: np.ndarray,
    output_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    output = np.zeros(output_shape_zyx, dtype=np.uint8)
    coords = np.argwhere(source_mask > 0)
    if coords.size == 0:
        return output
    source_shape = np.asarray(source_mask.shape, dtype=np.float64)
    output_shape = np.asarray(output_shape_zyx, dtype=np.float64)
    mapped = np.floor((coords.astype(np.float64) + 0.5) * output_shape / source_shape).astype(
        np.int64
    )
    mapped = np.clip(mapped, 0, np.asarray(output_shape_zyx, dtype=np.int64) - 1)
    output[tuple(mapped.T)] = 1
    return output


def positive_point_candidates(mask: np.ndarray) -> list[list[int]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return []
    center = (coords.min(axis=0).astype(np.float32) + coords.max(axis=0).astype(np.float32)) / 2.0
    candidates = [int(np.argmin(np.sum((coords.astype(np.float32) - center) ** 2, axis=1)))]
    for axis in range(3):
        candidates.extend([int(np.argmin(coords[:, axis])), int(np.argmax(coords[:, axis]))])
    unique: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    for index in candidates:
        point = tuple(int(value) for value in coords[index])
        if point not in seen:
            seen.add(point)
            unique.append(list(point))
    return unique


def resample_targets_nearest_with_fallback(
    targets_fzyx: np.ndarray,
    output_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, list[int]]:
    tensor = torch.from_numpy(np.ascontiguousarray(targets_fzyx))[None].float()
    resized = F.interpolate(tensor, size=output_shape_zyx, mode="nearest-exact")[0]
    output = (resized.numpy() > 0.5).astype(np.uint8, copy=False)
    fallback_indices: list[int] = []
    for index in range(targets_fzyx.shape[0]):
        if targets_fzyx[index].any() and not output[index].any():
            output[index] = _foreground_center_splat(targets_fzyx[index], output_shape_zyx)
            fallback_indices.append(index)
    return np.ascontiguousarray(output), fallback_indices


def _image_orientation_metadata(ct_properties: dict[str, Any], name: str) -> dict[str, Any]:
    nibabel_stuff = ct_properties.get("nibabel_stuff", {})
    original_affine = nibabel_stuff.get("original_affine")
    reoriented_affine = nibabel_stuff.get("reoriented_affine", original_affine)
    if original_affine is None or reoriented_affine is None:
        raise ValueError(f"{name}: missing CT affine metadata from NibabelIOWithReorient")
    original_ornt = io_orientation(original_affine)
    reoriented_ornt = io_orientation(reoriented_affine)
    transform = ornt_transform(original_ornt, reoriented_ornt)
    return {
        "applied_transform": "image_only_original_ct_to_reoriented_metadata",
        "ct_original_axcodes": list(nib.aff2axcodes(original_affine)),
        "ct_reoriented_axcodes": list(nib.aff2axcodes(reoriented_affine)),
        "ct_original_orientation": serializable_ornt(original_ornt),
        "ct_reoriented_orientation": serializable_ornt(reoriented_ornt),
        "orientation_transform": serializable_ornt(transform),
    }


def _load_native_crop_zscore(
    entry: dict[str, Any],
    ct_root: Path,
    seg_dir: Path,
    allow_missing_targets: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    name = entry["name"]
    ct_path = ct_rate_abs_path(name, ct_root)
    gt_path = seg_dir / name
    if not ct_path.is_file():
        raise FileNotFoundError(f"Missing CT: {ct_path}")

    reader = NibabelIOWithReorient()
    image, ct_properties = reader.read_images([str(ct_path)])
    image = image.astype(np.float32, copy=True)
    original_reoriented_shape = tuple(int(value) for value in image.shape[1:])

    target_fzyx: np.ndarray | None = None
    orientation = _image_orientation_metadata(ct_properties, name)
    gt_shape_fxyz: list[int] | None = None
    if gt_path.is_file():
        gt_img = nib.load(str(gt_path))
        target_fxyz = np.asanyarray(gt_img.dataobj).astype(np.float32, copy=False)
        target_fzyx, orientation = reorient_target_to_voxtell_layout(
            target_fxyz,
            ct_properties,
            name,
        )
        if tuple(image.shape[1:]) != tuple(target_fzyx.shape[1:]):
            raise ValueError(
                f"{name}: image shape {image.shape[1:]} != target shape {target_fzyx.shape[1:]}"
            )
        gt_shape_fxyz = [int(value) for value in gt_img.shape]
    elif not allow_missing_targets:
        raise FileNotFoundError(f"Missing segmentation: {gt_path}")

    image, _, bbox = crop_to_nonzero(image, None)
    image = ZScoreNormalization(intensityproperties={}).run(image, None)
    image = np.ascontiguousarray(image.astype(np.float32, copy=False))
    bbox_list = [[int(value) for value in axis] for axis in bbox]
    native_shape = tuple(int(value) for value in image.shape[1:])

    targets: np.ndarray | None = None
    source_counts: list[int] | None = None
    positive_points: list[list[list[int]]] | None = None
    if target_fzyx is not None:
        targets = (crop_target(target_fzyx, bbox_list) > 0).astype(np.uint8, copy=False)
        targets = np.ascontiguousarray(targets)
        source_counts = [int(target.sum()) for target in targets]
        if any(count <= 0 for count in source_counts):
            raise RuntimeError(f"{name}: native preprocessing contains an empty target")
        positive_points = [positive_point_candidates(target) for target in targets]

    prompts = sorted_prompts(entry)
    if targets is not None and len(prompts) != targets.shape[0]:
        raise ValueError(f"{name}: {len(prompts)} prompts but {targets.shape[0]} targets")

    native_spacing = tuple(float(value) for value in ct_properties["spacing"])
    nibabel_stuff = ct_properties.get("nibabel_stuff", {})
    metadata: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "preprocess_id": NATIVE_PREPROCESS_ID,
        "name": name,
        "split": entry.get("_split"),
        "split_index": entry.get("_index"),
        "ct_path": str(ct_path),
        "gt_path": str(gt_path) if gt_path.is_file() else None,
        "has_targets": targets is not None,
        "prompt_count": len(prompts),
        "prompts": prompts,
        "normalization": "crop_to_nonzero_then_full_cropped_volume_zscore_once",
        "image_resampling": None,
        "mask_resampling": None,
        "original_reoriented_shape_zyx": list(original_reoriented_shape),
        "crop_bbox_zyx": bbox_list,
        "native_cropped_shape_zyx": list(native_shape),
        "resampled_shape_zyx": list(native_shape),
        "source_target_voxels": source_counts,
        "resampled_target_voxels": source_counts,
        "resampled_target_positive_point_candidates": positive_points,
        "orientation": orientation,
        "ct_properties": {
            "spacing": list(native_spacing),
            "nibabel_stuff": {
                "original_affine": np.asarray(nibabel_stuff["original_affine"]).tolist(),
                "reoriented_affine": np.asarray(nibabel_stuff["reoriented_affine"]).tolist(),
            },
        },
        "gt_shape_fxyz": gt_shape_fxyz,
        "image_nbytes": int(image.nbytes),
        "targets_nbytes": int(targets.nbytes) if targets is not None else 0,
        "image_sha256": sha256_array(image),
        "targets_sha256": sha256_array(targets) if targets is not None else None,
    }
    return image, targets, metadata


def _preprocess_case_to_2mm(
    entry: dict[str, Any],
    ct_root: Path,
    seg_dir: Path,
    allow_missing_targets: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    image, targets, metadata = _load_native_crop_zscore(
        entry,
        ct_root=ct_root,
        seg_dir=seg_dir,
        allow_missing_targets=allow_missing_targets,
    )
    native_spacing = tuple(float(value) for value in metadata["ct_properties"]["spacing"])
    native_shape = tuple(int(value) for value in image.shape[1:])
    output_shape = output_shape_for_spacing(native_shape, native_spacing)
    image_2mm = resample_image_align_corners_false(image, output_shape)

    targets_2mm: np.ndarray | None = None
    fallback_indices: list[int] = []
    output_counts: list[int] | None = None
    positive_points: list[list[list[int]]] | None = None
    if targets is not None:
        targets_2mm, fallback_indices = resample_targets_nearest_with_fallback(targets, output_shape)
        output_counts = [int(target.sum()) for target in targets_2mm]
        if any(count <= 0 for count in output_counts):
            raise RuntimeError(
                f"{entry['name']}: 2 mm preprocessing contains an empty target after fallback"
            )
        positive_points = [positive_point_candidates(target) for target in targets_2mm]

    metadata.update(
        {
            "preprocess_id": ISO2MM_PREPROCESS_ID,
            "normalization": "crop_to_nonzero_then_full_cropped_volume_zscore_once_before_resampling",
            "image_resampling": {
                "source_spacing_zyx_mm": list(native_spacing),
                "target_spacing_zyx_mm": list(TARGET_SPACING_2MM_ZYX),
                "mode": "torch_trilinear",
                "align_corners": False,
                "antialias": False,
            },
            "mask_resampling": {
                "mode": "torch_nearest_exact",
                "fallback": "source_foreground_voxel_center_splat_if_nonempty_target_disappears",
                "fallback_target_indices": fallback_indices,
            },
            "resampled_shape_zyx": list(output_shape),
            "resampled_target_voxels": output_counts,
            "resampled_target_positive_point_candidates": positive_points,
            "image_nbytes": int(image_2mm.nbytes),
            "targets_nbytes": int(targets_2mm.nbytes) if targets_2mm is not None else 0,
            "image_sha256": sha256_array(image_2mm),
            "targets_sha256": sha256_array(targets_2mm) if targets_2mm is not None else None,
        }
    )
    return image_2mm, targets_2mm, metadata


def preprocess_case(
    entry: dict[str, Any],
    preprocess_id: str,
    ct_root: Path = CT_ROOT,
    seg_dir: Path = REX_SEG_DIR,
    allow_missing_targets: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if preprocess_id == NATIVE_PREPROCESS_ID:
        return _load_native_crop_zscore(entry, ct_root, seg_dir, allow_missing_targets)
    if preprocess_id == ISO2MM_PREPROCESS_ID:
        return _preprocess_case_to_2mm(entry, ct_root, seg_dir, allow_missing_targets)
    valid = ", ".join(SUPPORTED_PREPROCESS_IDS)
    raise ValueError(f"Unsupported preprocess_id {preprocess_id!r}; valid: {valid}")


def write_cached_case(
    cache_root: Path,
    entry: dict[str, Any],
    preprocess_id: str,
    ct_root: Path = CT_ROOT,
    seg_dir: Path = REX_SEG_DIR,
    overwrite: bool = False,
    allow_missing_targets: bool = False,
) -> dict[str, Any]:
    paths = cache_case_paths(cache_root, entry["name"])
    if paths["complete"].is_file() and not overwrite:
        return json.loads(paths["metadata"].read_text())
    paths["root"].mkdir(parents=True, exist_ok=True)
    image, targets, metadata = preprocess_case(
        entry,
        preprocess_id=preprocess_id,
        ct_root=ct_root,
        seg_dir=seg_dir,
        allow_missing_targets=allow_missing_targets,
    )
    _atomic_save_npy(paths["image"], image)
    if targets is not None:
        _atomic_save_npz(paths["targets"], targets=targets)
    elif paths["targets"].exists():
        paths["targets"].unlink()
    _atomic_write_json(paths["metadata"], metadata)
    complete_tmp = paths["complete"].with_name(f".complete.tmp.{os.getpid()}")
    complete_hash = metadata["targets_sha256"] or metadata["image_sha256"]
    complete_tmp.write_text(complete_hash + "\n")
    os.replace(complete_tmp, paths["complete"])
    return metadata


def load_cached_case(
    cache_root: Path,
    name: str,
    mmap_image: bool = False,
    require_targets: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    paths = cache_case_paths(cache_root, name)
    if not paths["complete"].is_file():
        raise FileNotFoundError(f"Incomplete or missing VoxTell cache for {name}: {paths['root']}")
    metadata = json.loads(paths["metadata"].read_text())
    image = np.load(paths["image"], mmap_mode="r" if mmap_image else None, allow_pickle=False)
    targets = None
    if paths["targets"].is_file():
        with np.load(paths["targets"], allow_pickle=False) as data:
            targets = data["targets"]
    elif require_targets:
        raise FileNotFoundError(f"Cached case has no targets: {paths['targets']}")
    return image, targets, metadata
