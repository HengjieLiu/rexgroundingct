#!/usr/bin/env python3
"""Audit fixed CT HU values, cropped z-scores, and validation-target intensities."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from tqdm import tqdm

from common import (
    CT_ROOT,
    EXP_ROOT,
    REX_SEG_DIR,
    command_string,
    ct_rate_abs_path,
    sha256_file,
    utc_now_iso,
)


SCRIPT_VERSION = 2
EXPERIMENT_ID = "011_voxtell_v123_e4d4_ct_normalization_ablation"
DEFAULT_DATASET_JSON = Path(
    "/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json"
)
DEFAULT_ANALYSIS_ROOT = EXP_ROOT / EXPERIMENT_ID / "analysis"
HU_REFERENCE_VALUES = (-1000.0, 0.0)
REPRESENTATIVE_CASES = (
    "train_13082_a_1.nii.gz",
    "train_13591_a_1.nii.gz",
    "train_2560_d_2.nii.gz",
    "train_18416_a_1.nii.gz",
    "train_13013_a_1.nii.gz",
)
CATEGORY_LABELS = {
    "1a": "Bronchial wall thickening",
    "1b": "Bronchiectasis",
    "1c": "Emphysema",
    "1d": "Septal thickening",
    "1e": "Micronodules",
    "1f": "Other diffuse",
    "2a": "Linear/scarring",
    "2b": "Consolidation/atelectasis",
    "2c": "Ground glass",
    "2d": "Nodules/masses",
    "2e": "Pleural effusion/thickening",
    "2f": "Honeycombing",
    "2g": "Pneumothorax",
    "2h": "Other focal",
}
PERCENTILES = (5, 25, 50, 75, 95)


def _sorted_finding_keys(entry: dict[str, Any]) -> list[str]:
    return sorted(entry["findings"], key=lambda value: int(value))


def _percentiles(values: np.ndarray) -> dict[str, float]:
    output = np.percentile(values, PERCENTILES)
    return {
        f"p{percentile:02d}": float(value)
        for percentile, value in zip(PERCENTILES, output)
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if not rows:
        temporary.write_text("")
        os.replace(temporary, path)
        return
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _effective_nifti_scaling(image: nib.spatialimages.SpatialImage) -> tuple[float, float]:
    slope = float(getattr(image.dataobj, "slope", 1.0) or 1.0)
    intercept = float(getattr(image.dataobj, "inter", 0.0) or 0.0)
    return slope, intercept


def _finding_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "voxels": int(values.size),
        **_percentiles(values),
        "mean_hu": float(np.mean(values, dtype=np.float64)),
        "std_hu": float(np.std(values, dtype=np.float64)),
        "fraction_below_minus1000": float(np.mean(values < -1000.0)),
        "fraction_above_400": float(np.mean(values > 400.0)),
        "fraction_above_1000": float(np.mean(values > 1000.0)),
    }


def _load_entries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    entries = data.get("test")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected evaluator-compatible list under 'test'")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "hu_normalization_val200_audit.json",
    )
    parser.add_argument(
        "--case-csv",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "hu_normalization_val200_cases.csv",
    )
    parser.add_argument(
        "--finding-csv",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "hu_normalization_val200_findings.csv",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--skip-expected-count-check", action="store_true")
    args = parser.parse_args()

    entries = _load_entries(args.dataset_json)
    if args.max_cases is not None:
        entries = entries[: args.max_cases]
    if not entries:
        raise ValueError("No validation entries selected")

    case_rows: list[dict[str, Any]] = []
    finding_rows: list[dict[str, Any]] = []
    finding_values: list[np.ndarray] = []
    category_values: dict[str, list[np.ndarray]] = defaultdict(list)
    dtype_counts: Counter[str] = Counter()
    scaling_counts: Counter[str] = Counter()
    axcode_counts: Counter[str] = Counter()
    shape_mismatches: list[dict[str, Any]] = []

    for case_index, entry in enumerate(tqdm(entries, desc="Auditing val CT HU values")):
        name = entry["name"]
        ct_path = ct_rate_abs_path(name, args.ct_root)
        gt_path = args.seg_dir / entry.get("seg_path", name)
        if not ct_path.is_file():
            raise FileNotFoundError(f"Missing CT: {ct_path}")
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing segmentation: {gt_path}")

        ct_image = nib.load(str(ct_path))
        gt_image = nib.load(str(gt_path))
        stored_dtype = str(np.dtype(ct_image.get_data_dtype()))
        slope, intercept = _effective_nifti_scaling(ct_image)
        axcodes = "".join(nib.aff2axcodes(ct_image.affine))
        dtype_counts[stored_dtype] += 1
        scaling_counts[f"{slope:g}/{intercept:g}"] += 1
        axcode_counts[axcodes] += 1

        image_xyz = np.asanyarray(ct_image.dataobj).astype(np.float32, copy=False)
        targets_fxyz = np.asanyarray(gt_image.dataobj)
        if targets_fxyz.ndim == 3:
            targets_fxyz = targets_fxyz[None]
        if targets_fxyz.ndim != 4 or tuple(targets_fxyz.shape[1:]) != tuple(
            image_xyz.shape
        ):
            shape_mismatches.append(
                {
                    "name": name,
                    "ct_shape_xyz": list(image_xyz.shape),
                    "gt_shape_fxyz": list(targets_fxyz.shape),
                }
            )
            continue

        cropped, _, bbox = crop_to_nonzero(image_xyz[None], None)
        cropped = np.asarray(cropped[0], dtype=np.float32)
        mean_hu = float(np.mean(cropped, dtype=np.float64))
        std_hu = float(np.std(cropped, dtype=np.float64))
        if std_hu <= 0:
            raise RuntimeError(f"{name}: cropped CT has zero intensity standard deviation")
        row = {
            "case_index": case_index,
            "name": name,
            "stored_dtype": stored_dtype,
            "effective_nifti_slope": slope,
            "effective_nifti_intercept": intercept,
            "axcodes": axcodes,
            "shape_x": int(image_xyz.shape[0]),
            "shape_y": int(image_xyz.shape[1]),
            "shape_z": int(image_xyz.shape[2]),
            "crop_x": int(cropped.shape[0]),
            "crop_y": int(cropped.shape[1]),
            "crop_z": int(cropped.shape[2]),
            "crop_bbox": json.dumps([[int(value) for value in axis] for axis in bbox]),
            "raw_min_hu": float(np.min(cropped)),
            "raw_max_hu": float(np.max(cropped)),
            "raw_mean_hu": mean_hu,
            "raw_std_hu": std_hu,
            "fraction_below_minus1024": float(np.mean(cropped < -1024.0)),
            "fraction_above_plus1024": float(np.mean(cropped > 1024.0)),
            "minus1000_hu_zscore": float((-1000.0 - mean_hu) / std_hu),
            "zero_hu_zscore": float((0.0 - mean_hu) / std_hu),
        }
        case_rows.append(row)

        finding_keys = _sorted_finding_keys(entry)
        if len(finding_keys) != targets_fxyz.shape[0]:
            raise ValueError(
                f"{name}: {len(finding_keys)} prompts but {targets_fxyz.shape[0]} targets"
            )
        for finding_index, key in enumerate(finding_keys):
            mask = targets_fxyz[finding_index] > 0
            values = np.ascontiguousarray(image_xyz[mask], dtype=np.float32)
            if values.size == 0:
                raise RuntimeError(f"{name}: finding {key} has an empty target")
            category_code = str(entry.get("categories", {}).get(key, "unknown"))
            summary = _finding_summary(values)
            finding_rows.append(
                {
                    "case_index": case_index,
                    "name": name,
                    "finding_index": finding_index,
                    "finding_key": key,
                    "category_code": category_code,
                    "category_label": CATEGORY_LABELS.get(category_code, "Other"),
                    "prompt": entry["findings"][key],
                    **summary,
                }
            )
            finding_values.append(values)
            category_values[category_code].append(values)

    if shape_mismatches:
        raise RuntimeError(f"CT/GT shape mismatches: {shape_mismatches[:10]}")

    all_gt_values = np.concatenate(finding_values)
    finding_medians = np.asarray(
        [float(row["p50"]) for row in finding_rows],
        dtype=np.float64,
    )
    category_summary: dict[str, dict[str, Any]] = {}
    for category_code, label in CATEGORY_LABELS.items():
        arrays = category_values.get(category_code, [])
        if not arrays:
            continue
        values = np.concatenate(arrays)
        category_summary[category_code] = {
            "label": label,
            "findings": len(arrays),
            "gt_voxels": int(values.size),
            **_percentiles(values),
        }

    representative = {
        row["name"]: {
            "mean_hu": row["raw_mean_hu"],
            "std_hu": row["raw_std_hu"],
            "minus1000_hu_zscore": row["minus1000_hu_zscore"],
            "zero_hu_zscore": row["zero_hu_zscore"],
            "raw_min_hu": row["raw_min_hu"],
            "raw_max_hu": row["raw_max_hu"],
            "fraction_below_minus1024": row["fraction_below_minus1024"],
        }
        for row in case_rows
        if row["name"] in REPRESENTATIVE_CASES
    }
    highest_calcification = max(
        finding_rows,
        key=lambda row: (float(row["fraction_above_400"]), float(row["p50"])),
    )
    case_means = np.asarray([row["raw_mean_hu"] for row in case_rows], dtype=np.float64)
    case_stds = np.asarray([row["raw_std_hu"] for row in case_rows], dtype=np.float64)
    minus1000_z = np.asarray(
        [row["minus1000_hu_zscore"] for row in case_rows],
        dtype=np.float64,
    )
    zero_z = np.asarray([row["zero_hu_zscore"] for row in case_rows], dtype=np.float64)

    audit = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "command": command_string(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_json": str(args.dataset_json),
        "dataset_json_sha256": sha256_file(args.dataset_json),
        "ct_root": str(args.ct_root),
        "seg_dir": str(args.seg_dir),
        "cases": len(case_rows),
        "findings": len(finding_rows),
        "gt_voxels": int(all_gt_values.size),
        "header_and_geometry": {
            "stored_dtype_counts": dict(dtype_counts),
            "effective_slope_intercept_counts": dict(scaling_counts),
            "axcode_counts": dict(axcode_counts),
            "shape_mismatches": 0,
            "materialized_hu_interpretation": (
                "The fixed NIfTI values are treated directly as HU. Do not apply "
                "DICOM rescale slope/intercept a second time."
            ),
        },
        "cropped_case_statistics": {
            "mean_hu": _percentiles(case_means),
            "std_hu": _percentiles(case_stds),
            "minus1000_hu_after_zscore": _percentiles(minus1000_z),
            "zero_hu_after_zscore": _percentiles(zero_z),
        },
        "representative_cases": representative,
        "gt_intensity_statistics": {
            "voxel_weighted_hu": _percentiles(all_gt_values),
            "finding_weighted_median_hu": _percentiles(finding_medians),
            "fraction_below_minus1000": float(np.mean(all_gt_values < -1000.0)),
            "fraction_above_400": float(np.mean(all_gt_values > 400.0)),
            "fraction_above_1000": float(np.mean(all_gt_values > 1000.0)),
        },
        "category_summary": category_summary,
        "calcification_example": highest_calcification,
        "outputs": {
            "case_csv": str(args.case_csv),
            "finding_csv": str(args.finding_csv),
        },
    }

    if not args.skip_expected_count_check and args.max_cases is None:
        expected = {"cases": 200, "findings": 381, "gt_voxels": 32_527_800}
        observed = {key: audit[key] for key in expected}
        if observed != expected:
            raise RuntimeError(f"Expected validation audit {expected}, observed {observed}")
        if dtype_counts != Counter({"int16": 200}):
            raise RuntimeError(f"Expected all int16 fixed NIfTIs, observed {dtype_counts}")
        if scaling_counts != Counter({"1/0": 200}):
            raise RuntimeError(
                "Expected effective NIfTI slope/intercept 1/0 for all cases, "
                f"observed {scaling_counts}"
            )
        if axcode_counts != Counter({"LPS": 200}):
            raise RuntimeError(f"Expected LPS orientation for all cases, observed {axcode_counts}")

    _write_csv(args.case_csv, case_rows)
    _write_csv(args.finding_csv, finding_rows)
    _write_json(args.output_json, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
