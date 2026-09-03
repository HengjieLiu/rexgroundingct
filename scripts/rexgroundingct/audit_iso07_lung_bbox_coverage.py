#!/usr/bin/env python3
"""Audit 0.7 mm whole-lung bbox coverage for fixed ReXGroundingCT val200."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from common import (
    CT_ROOT,
    EXP_ROOT,
    REPO_ROOT,
    REX_SEG_DIR,
    canonical_config_path,
    command_string,
    read_json,
    sha256_file,
    snapshot_experiment_config,
    update_run_manifest,
)


EXPERIMENT_ID = "019_voxtell_iso07_lung_bbox_coverage_audit"
SCRIPT_VERSION = 1
DEFAULT_CONFIG = canonical_config_path(EXPERIMENT_ID)
DEFAULT_VAL_JSON = REPO_ROOT / "configs/evaluation/rexgroundingct_val200_seed20260723.json"
DEFAULT_TOTAL_CACHE = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/"
    "totalsegmentator_total_fast_3mm_v2_16_0"
)
DEFAULT_ISO07_CACHE = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/"
    "crop_clip1024_linear_iso07_v1"
)
DEFAULT_LUNG_CACHE = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/"
    "totalsegmentator_lung_iso07_v1"
)
DEFAULT_RUNTIME_ROOT = EXP_ROOT / EXPERIMENT_ID
DEFAULT_TARGET_SPACING = (0.7, 0.7, 0.7)
LUNG_LABEL_IDS = (10, 11, 12, 13, 14)
DIRECTION_ORDER = ("L", "R", "A", "P", "S", "I")
CATEGORY_ORDER = (
    "1a",
    "1b",
    "1c",
    "1d",
    "1e",
    "1f",
    "2a",
    "2b",
    "2c",
    "2d",
    "2e",
    "2f",
    "2g",
    "2h",
)
CATEGORY_INFO = {
    "1a": ("diffuse", "Bronchial wall thickening"),
    "1b": ("diffuse", "Bronchiectasis"),
    "1c": ("diffuse", "Emphysema"),
    "1d": ("diffuse", "Septal thickening / reticulation"),
    "1e": ("diffuse", "Micronodules / tree-in-bud"),
    "1f": ("diffuse", "Other diffuse lung/airway/pleural abnormality"),
    "2a": ("focal", "Linear opacity, scarring, fibrosis"),
    "2b": ("focal", "Atelectasis / consolidation"),
    "2c": ("focal", "Ground-glass opacity"),
    "2d": ("focal", "Pulmonary nodules / masses"),
    "2e": ("focal", "Pleural effusion / thickening"),
    "2f": ("focal", "Honeycombing"),
    "2g": ("focal", "Pneumothorax"),
    "2h": ("focal", "Other focal lung/airway/pleural finding"),
}
GROUPS = {
    "all diffuse": tuple(code for code in CATEGORY_ORDER if code.startswith("1")),
    "all local/focal": tuple(code for code in CATEGORY_ORDER if code.startswith("2")),
    "all findings": CATEGORY_ORDER,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("experiment") != EXPERIMENT_ID:
        raise ValueError(f"Unexpected config experiment: {config.get('experiment')!r}")
    return config


def load_validation_rows(path: Path, expected_sha256: str | None) -> list[dict[str, Any]]:
    observed_sha256 = sha256_file(path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise ValueError(
            f"Validation manifest hash mismatch: expected={expected_sha256} "
            f"observed={observed_sha256}"
        )
    payload = json.loads(path.read_text())
    rows = payload.get("test")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list under 'test' in {path}")
    output: list[dict[str, Any]] = []
    for val_order, row in enumerate(rows):
        copied = dict(row)
        copied["val_order"] = val_order
        copied["case_key"] = str(row["name"]).removesuffix(".nii.gz")
        output.append(copied)
    return output


def sorted_finding_ids(row: dict[str, Any]) -> list[str]:
    findings = row.get("findings") or {}
    return sorted((str(key) for key in findings), key=int)


def bbox_from_mask(mask: np.ndarray) -> list[list[int]]:
    """Return inclusive ``[[lo, hi], ...]`` bounds in array-axis order."""
    coordinates = np.argwhere(mask > 0)
    if coordinates.size == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask")
    lower = coordinates.min(axis=0).astype(np.int64)
    upper = coordinates.max(axis=0).astype(np.int64)
    return [[int(lo), int(hi)] for lo, hi in zip(lower, upper, strict=True)]


def bbox_contains(outer: Sequence[Sequence[int]], inner: Sequence[Sequence[int]]) -> bool:
    return all(
        int(outer[axis][0]) <= int(inner[axis][0])
        and int(inner[axis][1]) <= int(outer[axis][1])
        for axis in range(3)
    )


def directional_deficits(
    lung_bbox_zyx: Sequence[Sequence[int]],
    target_bbox_zyx: Sequence[Sequence[int]],
) -> dict[str, int]:
    """Compute outward target-bbox deficits for a RAS ``F,Z,Y,X`` array."""
    lung_z, lung_y, lung_x = lung_bbox_zyx
    target_z, target_y, target_x = target_bbox_zyx
    return {
        "L": max(0, int(lung_x[0]) - int(target_x[0])),
        "R": max(0, int(target_x[1]) - int(lung_x[1])),
        "A": max(0, int(target_y[1]) - int(lung_y[1])),
        "P": max(0, int(lung_y[0]) - int(target_y[0])),
        "S": max(0, int(target_z[1]) - int(lung_z[1])),
        "I": max(0, int(lung_z[0]) - int(target_z[0])),
    }


def isotropic_expansion(deficits: dict[str, int]) -> int:
    missing = [direction for direction in DIRECTION_ORDER if direction not in deficits]
    if missing:
        raise ValueError(f"Missing directional deficits: {missing}")
    values = [int(deficits[direction]) for direction in DIRECTION_ORDER]
    if any(value < 0 for value in values):
        raise ValueError(f"Directional deficits must be nonnegative: {deficits}")
    return max(values)


def expand_bbox(
    bbox_zyx: Sequence[Sequence[int]],
    expansion: int,
    shape_zyx: Sequence[int],
) -> list[list[int]]:
    if expansion < 0:
        raise ValueError(f"Expansion must be nonnegative, got {expansion}")
    if len(shape_zyx) != 3:
        raise ValueError(f"Expected a 3D shape, got {shape_zyx}")
    return [
        [
            max(0, int(bbox_zyx[axis][0]) - int(expansion)),
            min(int(shape_zyx[axis]) - 1, int(bbox_zyx[axis][1]) + int(expansion)),
        ]
        for axis in range(3)
    ]


def _foreground_center_splat(mask: np.ndarray, output_shape_zyx: Sequence[int]) -> np.ndarray:
    coordinates = np.argwhere(mask > 0)
    if coordinates.size == 0:
        return np.zeros(tuple(int(value) for value in output_shape_zyx), dtype=np.uint8)
    source_shape = np.asarray(mask.shape, dtype=np.float64)
    output_shape = np.asarray(output_shape_zyx, dtype=np.float64)
    mapped = np.floor((coordinates.astype(np.float64) + 0.5) * output_shape / source_shape).astype(
        np.int64
    )
    mapped = np.clip(mapped, 0, output_shape.astype(np.int64) - 1)
    output = np.zeros(tuple(int(value) for value in output_shape_zyx), dtype=np.uint8)
    output[tuple(mapped.T)] = 1
    return output


def resample_binary_mask_nearest_exact(
    mask_zyx: np.ndarray,
    output_shape_zyx: Sequence[int],
    torch_threads: int = 1,
) -> tuple[np.ndarray, bool]:
    """Resample a binary mask with the Exp016 nearest-exact implementation."""
    if mask_zyx.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got {mask_zyx.shape}")
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(max(1, int(torch_threads)))
    source = np.ascontiguousarray(mask_zyx, dtype=np.float32)
    with torch.no_grad():
        resized = functional.interpolate(
            torch.from_numpy(source)[None, None],
            size=tuple(int(value) for value in output_shape_zyx),
            mode="nearest-exact",
        )[0, 0]
    output = np.ascontiguousarray((resized.cpu().numpy() > 0.5).astype(np.uint8))
    fallback_used = False
    if mask_zyx.any() and not output.any():
        output = _foreground_center_splat(mask_zyx, output_shape_zyx)
        fallback_used = True
    return output, fallback_used


def _import_nifti_orientation_tools() -> tuple[Any, Any, Any, Any]:
    import nibabel as nib
    from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

    return nib, apply_orientation, io_orientation, ornt_transform


def _case_paths(
    row: dict[str, Any],
    total_cache: Path,
    iso07_cache: Path,
    lung_cache: Path,
    runtime_root: Path,
) -> dict[str, Path]:
    case_key = str(row["case_key"])
    return {
        "ct": CT_ROOT / "dataset" / f"{case_key.split('_')[0]}_fixed" / "_".join(case_key.split("_")[:2]) / "_".join(case_key.split("_")[:3]) / row["name"],
        "gt": REX_SEG_DIR / row["name"],
        "total": total_cache / "cases" / case_key / "total_labels.nii.gz",
        "total_metadata": total_cache / "cases" / case_key / "metadata.json",
        "iso_metadata": iso07_cache / "cases" / case_key / "metadata.json",
        "targets": iso07_cache / "cases" / case_key / "targets.npz",
        "lung_dir": lung_cache / "cases" / case_key,
        "lung_mask": lung_cache / "cases" / case_key / "lung_mask.npz",
        "lung_metadata": lung_cache / "cases" / case_key / "metadata.json",
        "case_result": runtime_root / "cases" / case_key / "finding_bbox_audit.json",
        "case_complete": runtime_root / "cases" / case_key / ".complete",
    }


def _required_case_files(paths: dict[str, Path]) -> list[Path]:
    return [
        paths["ct"],
        paths["gt"],
        paths["total"],
        paths["total_metadata"],
        paths["iso_metadata"],
        paths["targets"],
    ]


def _validate_iso_metadata(metadata: dict[str, Any], name: str) -> None:
    spacing = metadata.get("image_resampling", {}).get("target_spacing_zyx_mm")
    if [round(float(value), 6) for value in spacing or []] != list(DEFAULT_TARGET_SPACING):
        raise ValueError(f"{name}: unexpected iso07 target spacing: {spacing}")
    if metadata.get("mask_resampling", {}).get("mode") != "torch_nearest_exact":
        raise ValueError(f"{name}: unexpected GT mask resampling mode")
    if metadata.get("orientation", {}).get("ct_reoriented_axcodes") != ["R", "A", "S"]:
        raise ValueError(f"{name}: expected RAS reorientation metadata")
    for key in ("crop_bbox_zyx", "native_cropped_shape_zyx", "resampled_shape_zyx"):
        if key not in metadata:
            raise ValueError(f"{name}: missing iso07 metadata field {key}")


def _validate_case_headers(row: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    nib, _, _, _ = _import_nifti_orientation_tools()
    ct_image = nib.load(str(paths["ct"]))
    total_image = nib.load(str(paths["total"]))
    if tuple(ct_image.shape[:3]) != tuple(total_image.shape[:3]):
        raise ValueError(
            f"{row['name']}: CT/TotalSegmentator shape mismatch "
            f"{ct_image.shape} versus {total_image.shape}"
        )
    if not np.allclose(ct_image.affine, total_image.affine, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{row['name']}: CT/TotalSegmentator affine mismatch")
    return {
        "ct_shape_xyz": [int(value) for value in ct_image.shape[:3]],
        "ct_axcodes": list(nib.aff2axcodes(ct_image.affine)),
        "total_shape_xyz": [int(value) for value in total_image.shape[:3]],
    }


def _load_target_array(path: Path, expected_channels: int, expected_shape: Sequence[int], name: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "targets" not in archive.files:
            raise ValueError(f"{name}: targets.npz lacks the 'targets' array")
        targets = np.asarray(archive["targets"])
    if targets.ndim != 4:
        raise ValueError(f"{name}: expected targets with shape FZYX, got {targets.shape}")
    if targets.shape[0] != expected_channels:
        raise ValueError(
            f"{name}: target channel count {targets.shape[0]} != findings {expected_channels}"
        )
    if tuple(targets.shape[1:]) != tuple(int(value) for value in expected_shape):
        raise ValueError(
            f"{name}: target shape {targets.shape[1:]} != iso07 shape {expected_shape}"
        )
    binary = np.ascontiguousarray(targets > 0)
    if any(not target.any() for target in binary):
        raise ValueError(f"{name}: at least one target is empty")
    return binary


def _make_lung_iso07_mask(
    total_path: Path,
    iso_metadata: dict[str, Any],
    torch_threads: int,
) -> tuple[np.ndarray, int, int, bool]:
    nib, apply_orientation, io_orientation, ornt_transform = _import_nifti_orientation_tools()
    total_image = nib.load(str(total_path))
    total_raw = np.asanyarray(total_image.dataobj)
    if total_raw.ndim != 3:
        raise ValueError(f"TotalSegmentator mask must be 3D, got {total_raw.shape}")
    lung_native = np.isin(total_raw, LUNG_LABEL_IDS).astype(np.uint8, copy=False)
    raw_voxels = int(lung_native.sum())
    if raw_voxels <= 0:
        raise ValueError(f"{total_path}: merged lung mask is empty")

    nibabel_stuff = iso_metadata["ct_properties"]["nibabel_stuff"]
    original_affine = np.asarray(nibabel_stuff["original_affine"], dtype=np.float64)
    reoriented_affine = np.asarray(nibabel_stuff["reoriented_affine"], dtype=np.float64)
    transform = ornt_transform(io_orientation(original_affine), io_orientation(reoriented_affine))
    lung_reoriented_fxyz = apply_orientation(lung_native, transform)
    lung_zyx = np.ascontiguousarray(np.transpose(lung_reoriented_fxyz, (2, 1, 0)))
    crop_bbox = iso_metadata["crop_bbox_zyx"]
    crop_slices = tuple(slice(int(axis[0]), int(axis[1])) for axis in crop_bbox)
    lung_crop = np.ascontiguousarray(lung_zyx[crop_slices])
    expected_native_shape = tuple(int(value) for value in iso_metadata["native_cropped_shape_zyx"])
    if tuple(lung_crop.shape) != expected_native_shape:
        raise ValueError(
            f"{total_path}: lung crop shape {lung_crop.shape} != metadata {expected_native_shape}"
        )
    output_shape = tuple(int(value) for value in iso_metadata["resampled_shape_zyx"])
    lung_iso07, fallback_used = resample_binary_mask_nearest_exact(
        lung_crop,
        output_shape,
        torch_threads=torch_threads,
    )
    if not lung_iso07.any():
        raise ValueError(f"{total_path}: iso07 lung mask is empty")
    return lung_iso07, raw_voxels, int(lung_iso07.sum()), fallback_used


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _write_lung_mask(
    paths: dict[str, Path],
    lung_mask: np.ndarray,
    metadata: dict[str, Any],
    overwrite: bool,
) -> None:
    paths["lung_dir"].mkdir(parents=True, exist_ok=True)
    if paths["lung_mask"].exists() and not overwrite:
        return
    temporary = paths["lung_mask"].with_name(f".{paths['lung_mask'].name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, lung_mask=lung_mask.astype(np.uint8, copy=False))
    os.replace(temporary, paths["lung_mask"])
    write_json_atomic(paths["lung_metadata"], metadata)


def _bbox_string(bbox: Sequence[Sequence[int]]) -> str:
    return "z=[%d,%d]; y=[%d,%d]; x=[%d,%d]" % (
        int(bbox[0][0]),
        int(bbox[0][1]),
        int(bbox[1][0]),
        int(bbox[1][1]),
        int(bbox[2][0]),
        int(bbox[2][1]),
    )


def _finding_record(
    row: dict[str, Any],
    finding_id: str,
    target: np.ndarray,
    lung_bbox: list[list[int]],
    shape_zyx: Sequence[int],
) -> dict[str, Any]:
    target_bbox = bbox_from_mask(target)
    deficits = directional_deficits(lung_bbox, target_bbox)
    expansion = isotropic_expansion(deficits)
    expanded = expand_bbox(lung_bbox, expansion, shape_zyx)
    if not bbox_contains(expanded, target_bbox):
        raise AssertionError(
            f"{row['name']} finding {finding_id}: expanded bbox does not contain target"
        )
    category = str((row.get("categories") or {}).get(finding_id, "unknown"))
    focality, category_name = CATEGORY_INFO.get(category, ("unknown", "Unknown"))
    return {
        "val_order": int(row["val_order"]),
        "case": str(row["name"]),
        "case_key": str(row["case_key"]),
        "finding_id": finding_id,
        "category": category,
        "category_name": category_name,
        "focality": focality,
        "finding": str((row.get("findings") or {}).get(finding_id, "")),
        "target_voxels": int(target.sum()),
        "lung_bbox_zyx": lung_bbox,
        "target_bbox_zyx": target_bbox,
        "expanded_lung_bbox_zyx": expanded,
        "deficit_L_voxels": int(deficits["L"]),
        "deficit_R_voxels": int(deficits["R"]),
        "deficit_A_voxels": int(deficits["A"]),
        "deficit_P_voxels": int(deficits["P"]),
        "deficit_S_voxels": int(deficits["S"]),
        "deficit_I_voxels": int(deficits["I"]),
        "isotropic_expansion_voxels": int(expansion),
        "isotropic_expansion_mm": float(expansion * DEFAULT_TARGET_SPACING[0]),
        "contained_before_expansion": bool(expansion == 0),
    }


def process_case(
    row: dict[str, Any],
    *,
    total_cache: Path,
    iso07_cache: Path,
    lung_cache: Path,
    runtime_root: Path,
    write_lung_masks: bool,
    overwrite: bool,
    torch_threads: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = _case_paths(row, total_cache, iso07_cache, lung_cache, runtime_root)
    missing = [str(path) for path in _required_case_files(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{row['name']}: missing required inputs: {missing}")

    if paths["case_result"].is_file() and not overwrite and paths["lung_mask"].is_file():
        cached = json.loads(paths["case_result"].read_text())
        return cached["case_summary"], cached["findings"]

    header_summary = _validate_case_headers(row, paths)
    total_metadata = json.loads(paths["total_metadata"].read_text())
    iso_metadata = json.loads(paths["iso_metadata"].read_text())
    _validate_iso_metadata(iso_metadata, row["name"])
    expected_shape = tuple(int(value) for value in iso_metadata["resampled_shape_zyx"])
    targets = _load_target_array(paths["targets"], len(sorted_finding_ids(row)), expected_shape, row["name"])
    present_ids = {
        int(label)
        for label, count in (total_metadata.get("geometry", {}).get("class_voxels") or {}).items()
        if int(label) in LUNG_LABEL_IDS and int(count) > 0
    }
    if not present_ids:
        raise ValueError(f"{row['name']}: no TotalSegmentator lung-lobe labels are present")

    lung_iso07, raw_lung_voxels, iso07_lung_voxels, fallback_used = _make_lung_iso07_mask(
        paths["total"],
        iso_metadata,
        torch_threads=torch_threads,
    )
    lung_bbox = bbox_from_mask(lung_iso07)
    lung_metadata = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "case": row["name"],
        "case_key": row["case_key"],
        "val_order": int(row["val_order"]),
        "lung_label_ids": list(LUNG_LABEL_IDS),
        "present_lung_label_ids": sorted(present_ids),
        "source_total_output": str(paths["total"]),
        "source_total_output_sha256": total_metadata.get("output_sha256"),
        "iso07_metadata": str(paths["iso_metadata"]),
        "iso07_metadata_sha256": sha256_file(paths["iso_metadata"]),
        "resampled_shape_zyx": list(expected_shape),
        "target_spacing_zyx_mm": list(DEFAULT_TARGET_SPACING),
        "crop_bbox_zyx": iso_metadata["crop_bbox_zyx"],
        "raw_lung_voxels": raw_lung_voxels,
        "iso07_lung_voxels": iso07_lung_voxels,
        "lung_bbox_zyx": lung_bbox,
        "resampling_fallback_used": fallback_used,
        "mask_interpolation": "torch_nearest_exact",
        "mask_sha256": _sha256_array(lung_iso07),
    }
    if write_lung_masks:
        _write_lung_mask(paths, lung_iso07, lung_metadata, overwrite=overwrite)

    findings = [
        _finding_record(row, finding_id, targets[index], lung_bbox, expected_shape)
        for index, finding_id in enumerate(sorted_finding_ids(row))
    ]
    case_summary = {
        "case": row["name"],
        "case_key": row["case_key"],
        "val_order": int(row["val_order"]),
        "finding_count": len(findings),
        "present_lung_label_ids": sorted(present_ids),
        "raw_lung_voxels": raw_lung_voxels,
        "iso07_lung_voxels": iso07_lung_voxels,
        "resampled_shape_zyx": list(expected_shape),
        "lung_bbox_zyx": lung_bbox,
        "resampling_fallback_used": fallback_used,
        "header_summary": header_summary,
        "lung_mask_path": str(paths["lung_mask"]) if write_lung_masks else None,
    }
    case_result = {"case_summary": case_summary, "findings": findings}
    case_dir = paths["case_result"].parent
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths["case_result"], case_result)
    write_json_atomic(
        case_dir / "input_manifest.json",
        {
            "case": row["name"],
            "ct": str(paths["ct"]),
            "gt": str(paths["gt"]),
            "total": str(paths["total"]),
            "total_metadata": str(paths["total_metadata"]),
            "iso_metadata": str(paths["iso_metadata"]),
            "targets": str(paths["targets"]),
            "target_sha256": iso_metadata.get("targets_sha256"),
        },
    )
    paths["case_complete"].write_text(utc_now() + "\n")
    return case_summary, findings


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([int(row["isotropic_expansion_voxels"]) for row in rows], dtype=np.int64)
    histogram = Counter(int(value) for value in values.tolist())
    if values.size == 0:
        return {
            "n_findings": 0,
            "n_cases": 0,
            "k0_count": 0,
            "k0_fraction": None,
            "median_voxels": None,
            "p90_voxels": None,
            "p95_voxels": None,
            "max_voxels": None,
            "histogram": {},
        }
    return {
        "n_findings": int(values.size),
        "n_cases": len({row["case"] for row in rows}),
        "k0_count": int((values == 0).sum()),
        "k0_fraction": float((values == 0).mean()),
        "median_voxels": float(np.percentile(values, 50)),
        "p90_voxels": float(np.percentile(values, 90)),
        "p95_voxels": float(np.percentile(values, 95)),
        "max_voxels": int(values.max()),
        "histogram": {str(key): int(histogram[key]) for key in sorted(histogram)},
    }


def build_summary(findings: list[dict[str, Any]], case_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    summary_rows: dict[str, dict[str, Any]] = {}
    for category in CATEGORY_ORDER:
        summary_rows[category] = {
            "label": f"{category} — {CATEGORY_INFO[category][1]}",
            "categories": [category],
            **_stats([row for row in findings if row["category"] == category]),
        }
    for group, categories in GROUPS.items():
        summary_rows[group] = {
            "label": group,
            "categories": list(categories),
            **_stats([row for row in findings if row["category"] in categories]),
        }
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "completed_at_utc": utc_now(),
        "cases_processed": len(case_summaries),
        "findings_processed": len(findings),
        "fallback_cases": sum(bool(row["resampling_fallback_used"]) for row in case_summaries),
        "categories": summary_rows,
        "case_summaries": case_summaries,
    }


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _format_histogram(stats: dict[str, Any]) -> str:
    histogram = stats.get("histogram") or {}
    if not histogram:
        return "—"
    total = max(1, int(stats["n_findings"]))
    return ", ".join(
        f"{key}: {count} ({100.0 * int(count) / total:.1f}%)"
        for key, count in histogram.items()
    )


def render_report(
    summary: dict[str, Any],
    findings: list[dict[str, Any]],
    *,
    config_path: Path,
    val_json: Path,
    total_cache: Path,
    iso07_cache: Path,
    lung_cache: Path,
) -> str:
    all_stats = summary["categories"]["all findings"]
    diffuse_stats = summary["categories"]["all diffuse"]
    focal_stats = summary["categories"]["all local/focal"]
    lines = [
        "# Experiment 019 — 0.7 mm Lung Bounding-Box Coverage Audit",
        "",
        "## High-Level Result",
        "",
        f"This audit covers `{summary['cases_processed']}` validation CT cases and "
        f"`{summary['findings_processed']}` individual GT findings.",
        "",
        f"- All findings already contained by the unexpanded lung bbox: "
        f"`{all_stats['k0_count']}/{all_stats['n_findings']}` "
        f"(`{100.0 * all_stats['k0_fraction']:.1f}%`).",
        f"- All diffuse findings contained without expansion: "
        f"`{diffuse_stats['k0_count']}/{diffuse_stats['n_findings']}` "
        f"(`{100.0 * diffuse_stats['k0_fraction']:.1f}%`).",
        f"- All local/focal findings contained without expansion: "
        f"`{focal_stats['k0_count']}/{focal_stats['n_findings']}` "
        f"(`{100.0 * focal_stats['k0_fraction']:.1f}%`).",
        f"- Overall median required expansion: `{_format_number(all_stats['median_voxels'])}` "
        f"voxels (`{0.7 * float(all_stats['median_voxels']):.3f} mm`).",
        f"- Overall 95th percentile: `{_format_number(all_stats['p95_voxels'])}` "
        f"voxels (`{0.7 * float(all_stats['p95_voxels']):.3f} mm`).",
        f"- Overall maximum: `{all_stats['max_voxels']}` voxels "
        f"(`{0.7 * int(all_stats['max_voxels']):.1f} mm`).",
        "",
        "Here, `k` is one shared expansion value applied to all six faces of the "
        "lung bounding box. Direction-specific deficits are retained below.",
        "",
        "## Population And Distributions",
        "",
        "`2a–2h` are called local/focal in this report; `1a–1f` are diffuse. "
        "Categories with no validation findings remain listed.",
        "",
        "| Population | Findings | Cases | k=0 | Median k | P90 k | P95 k | Max k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    report_order = list(CATEGORY_ORDER) + ["all diffuse", "all local/focal", "all findings"]
    for population in report_order:
        stats = summary["categories"][population]
        k0 = "—" if stats["n_findings"] == 0 else f"{stats['k0_count']} ({100.0 * stats['k0_fraction']:.1f}%)"
        lines.append(
            f"| {_markdown_escape(stats['label'])} | {stats['n_findings']} | {stats['n_cases']} | "
            f"{k0} | {_format_number(stats['median_voxels'])} | "
            f"{_format_number(stats['p90_voxels'])} | {_format_number(stats['p95_voxels'])} | "
            f"{_format_number(stats['max_voxels'])} |"
        )
    lines.extend(["", "### Exact k Distributions", "", "| Population | Distribution: `k voxels: count (percent)` |", "| --- | --- |"])
    for population in report_order:
        stats = summary["categories"][population]
        lines.append(f"| {_markdown_escape(stats['label'])} | {_format_histogram(stats)} |")

    lines.extend(
        [
            "",
            "## Data And Provenance",
            "",
            f"- Generated: `{summary['completed_at_utc']}`.",
            f"- Canonical config: `{config_path}`.",
            f"- Fixed validation manifest: `{val_json}`.",
            f"- TotalSegmentator source cache: `{total_cache}`.",
            f"- Existing iso07 GT cache: `{iso07_cache}`.",
            f"- Derived iso07 lung-mask cache: `{lung_cache}`.",
            "- Lung mask: union of TotalSegmentator labels 10–14, covering the five left/right lung lobes.",
            "- TotalSegmentator output: fast 3 mm model output restored to source CT geometry; it is an anatomical prior, not lung ground truth.",
            "- GT masks: existing iso07 `targets.npz` arrays; they were not resampled again.",
            "- No reliable patient identifier is available, so the unit is a validation CT case and its individual findings.",
            "",
            "## Geometry And Calculation",
            "",
            "The lung mask is transformed with the same per-case original-CT-to-RAS orientation, `FXYZ` to `FZYX` transpose, crop, and 0.7 mm nearest-exact resampling contract used by Exp016.",
            "",
            "In the resulting RAS `F,Z,Y,X` array:",
            "",
            "- X lower/upper faces correspond to L/R.",
            "- Y lower/upper faces correspond to P/A.",
            "- Z lower/upper faces correspond to I/S.",
            "",
            "For lung bbox `B` and target bbox `T`, the deficits are:",
            "",
            "```text",
            "dL = max(0, Bx_min - Tx_min)    dR = max(0, Tx_max - Bx_max)",
            "dP = max(0, By_min - Ty_min)    dA = max(0, Ty_max - By_max)",
            "dI = max(0, Bz_min - Tz_min)    dS = max(0, Tz_max - Bz_max)",
            "k = max(dL, dR, dA, dP, dS, dI)",
            "```",
            "",
            "Bounding boxes use inclusive voxel indices. `k=0` is exact bbox containment before expansion; `0.7 × k` is the isotropic physical expansion in millimeters.",
            "",
            "## Complete Per-Finding Details",
            "",
            "| Case | Finding | Category | Target voxels | k voxels | k mm | L | R | A | P | S | I | Lung bbox zyx | GT bbox zyx |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in findings:
        lines.append(
            f"| {_markdown_escape(row['case'])} | {row['finding_id']} | {row['category']} | "
            f"{row['target_voxels']} | {row['isotropic_expansion_voxels']} | "
            f"{row['isotropic_expansion_mm']:.1f} | {row['deficit_L_voxels']} | "
            f"{row['deficit_R_voxels']} | {row['deficit_A_voxels']} | "
            f"{row['deficit_P_voxels']} | {row['deficit_S_voxels']} | "
            f"{row['deficit_I_voxels']} | {_bbox_string(row['lung_bbox_zyx'])} | "
            f"{_bbox_string(row['target_bbox_zyx'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "This audit measures bbox containment only. It does not claim that every target voxel lies inside the predicted lung mask, and it does not evaluate TotalSegmentator segmentation quality. The result describes how much isotropic context expansion is needed around the predicted whole-lung bbox to cover released GT target extents.",
            "",
        ]
    )
    return "\n".join(lines)


def _finding_csv_rows(findings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "val_order",
        "case",
        "case_key",
        "finding_id",
        "category",
        "category_name",
        "focality",
        "finding",
        "target_voxels",
        "lung_bbox_zyx",
        "target_bbox_zyx",
        "expanded_lung_bbox_zyx",
        "deficit_L_voxels",
        "deficit_R_voxels",
        "deficit_A_voxels",
        "deficit_P_voxels",
        "deficit_S_voxels",
        "deficit_I_voxels",
        "isotropic_expansion_voxels",
        "isotropic_expansion_mm",
        "contained_before_expansion",
    ]
    output: list[dict[str, Any]] = []
    for row in findings:
        copied = dict(row)
        for key in ("lung_bbox_zyx", "target_bbox_zyx", "expanded_lung_bbox_zyx"):
            copied[key] = json.dumps(copied[key], separators=(",", ":"))
        output.append({field: copied.get(field) for field in fields})
    return output


def preflight(
    rows: list[dict[str, Any]],
    *,
    total_cache: Path,
    iso07_cache: Path,
    lung_cache: Path,
    runtime_root: Path,
    expected_cases: int,
    expected_findings: int,
) -> dict[str, Any]:
    if len(rows) != expected_cases:
        raise ValueError(f"Expected {expected_cases} validation cases, got {len(rows)}")
    finding_count = sum(len(sorted_finding_ids(row)) for row in rows)
    if finding_count != expected_findings:
        raise ValueError(f"Expected {expected_findings} findings, got {finding_count}")
    missing: list[dict[str, Any]] = []
    invalid_metadata: list[str] = []
    for row in rows:
        paths = _case_paths(row, total_cache, iso07_cache, lung_cache, runtime_root)
        absent = [str(path) for path in _required_case_files(paths) if not path.is_file()]
        if absent:
            missing.append({"case": row["name"], "paths": absent})
            continue
        iso_metadata = json.loads(paths["iso_metadata"].read_text())
        try:
            _validate_iso_metadata(iso_metadata, row["name"])
        except ValueError as exc:
            invalid_metadata.append(str(exc))
    if missing or invalid_metadata:
        raise RuntimeError(
            "Preflight failed: "
            + json.dumps(
                {"missing": missing[:10], "invalid_metadata": invalid_metadata[:10]},
                indent=2,
            )
        )
    return {
        "cases": len(rows),
        "findings": finding_count,
        "missing_inputs": 0,
        "invalid_iso07_metadata": 0,
        "target_spacing_zyx_mm": list(DEFAULT_TARGET_SPACING),
        "lung_label_ids": list(LUNG_LABEL_IDS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--val-json", type=Path, default=DEFAULT_VAL_JSON)
    parser.add_argument("--total-cache", type=Path, default=DEFAULT_TOTAL_CACHE)
    parser.add_argument("--iso07-cache", type=Path, default=DEFAULT_ISO07_CACHE)
    parser.add_argument("--lung-cache", type=Path, default=DEFAULT_LUNG_CACHE)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--expected-cases", type=int, default=200)
    parser.add_argument("--expected-findings", type=int, default=381)
    parser.add_argument("--expected-val-sha256", default="7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897")
    parser.add_argument("--case", dest="case_key")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-write-lung-masks", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    rows = load_validation_rows(args.val_json, args.expected_val_sha256)
    full_preflight = preflight(
        rows,
        total_cache=args.total_cache,
        iso07_cache=args.iso07_cache,
        lung_cache=args.lung_cache,
        runtime_root=args.runtime_root,
        expected_cases=args.expected_cases,
        expected_findings=args.expected_findings,
    )
    if args.preflight_only:
        print(json.dumps(full_preflight, indent=2, sort_keys=True))
        return 0

    selected_rows = rows
    if args.case_key:
        selected_rows = [row for row in rows if row["case_key"] == args.case_key]
        if not selected_rows:
            raise SystemExit(f"Unknown validation case key: {args.case_key}")
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    if not args.no_snapshot:
        snapshot_experiment_config(EXPERIMENT_ID, exp_dir=args.runtime_root)
    update_run_manifest(
        args.runtime_root,
        {
            "audit": EXPERIMENT_ID,
            "script_version": SCRIPT_VERSION,
            "command": command_string(),
            "config_path": str(args.config),
            "config_sha256": sha256_file(args.config),
            "val_json": str(args.val_json),
            "val_json_sha256": sha256_file(args.val_json),
            "preflight": full_preflight,
            "selected_cases": len(selected_rows),
            "write_lung_masks": not args.no_write_lung_masks,
        },
    )

    case_summaries: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        print(f"[{index}/{len(selected_rows)}] {row['name']}", flush=True)
        case_summary, case_findings = process_case(
            row,
            total_cache=args.total_cache,
            iso07_cache=args.iso07_cache,
            lung_cache=args.lung_cache,
            runtime_root=args.runtime_root,
            write_lung_masks=not args.no_write_lung_masks,
            overwrite=args.overwrite,
            torch_threads=args.torch_threads,
        )
        case_summaries.append(case_summary)
        findings.extend(case_findings)
    findings.sort(key=lambda row: (int(row["val_order"]), int(row["finding_id"])))
    case_summaries.sort(key=lambda row: int(row["val_order"]))
    summary = build_summary(findings, case_summaries)
    summary["preflight"] = full_preflight
    summary["selected_case"] = args.case_key
    summary_path = args.runtime_root / "reports/iso07_lung_bbox_coverage_summary.json"
    details_path = args.runtime_root / "reports/iso07_lung_bbox_finding_details.csv"
    report_path = args.runtime_root / "reports/iso07_lung_bbox_coverage_report.md"
    write_json_atomic(summary_path, summary)
    write_csv_atomic(
        details_path,
        _finding_csv_rows(findings),
        [
            "val_order",
            "case",
            "case_key",
            "finding_id",
            "category",
            "category_name",
            "focality",
            "finding",
            "target_voxels",
            "lung_bbox_zyx",
            "target_bbox_zyx",
            "expanded_lung_bbox_zyx",
            "deficit_L_voxels",
            "deficit_R_voxels",
            "deficit_A_voxels",
            "deficit_P_voxels",
            "deficit_S_voxels",
            "deficit_I_voxels",
            "isotropic_expansion_voxels",
            "isotropic_expansion_mm",
            "contained_before_expansion",
        ],
    )
    if not args.no_report:
        report_path.write_text(
            render_report(
                summary,
                findings,
                config_path=args.config,
                val_json=args.val_json,
                total_cache=args.total_cache,
                iso07_cache=args.iso07_cache,
                lung_cache=args.lung_cache,
            )
        )
    print(json.dumps({"summary": str(summary_path), "findings": len(findings), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
