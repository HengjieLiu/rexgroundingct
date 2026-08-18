#!/usr/bin/env python3
"""Audit native CT and label resolution for isotropic VoxTell planning."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from common import (
    CT_ROOT,
    REX_METADATA,
    REX_SEG_DIR,
    REPO_EXPERIMENT_ROOT,
    ct_rate_abs_path,
    load_split_entries,
    sha256_file,
    utc_now_iso,
)


DEFAULT_EXPERIMENT_ID = "015_voxtell_isotropic_resolution_audit"
DEFAULT_OUTPUT_DIR = REPO_EXPERIMENT_ROOT / DEFAULT_EXPERIMENT_ID
DEFAULT_TARGET_SPACINGS_MM = (1.0, 0.7)
SPLIT_ORDER = ("train", "val", "test")
SUMMARY_GROUPS = {
    "train": ("train",),
    "val": ("val",),
    "test": ("test",),
    "train+val": ("train", "val"),
    "all": ("train", "val", "test"),
}
GIB = 1024**3


def _spacing_label(value: float) -> str:
    text = f"{float(value):.6g}".replace(".", "_")
    return text.replace("-", "m")


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _percentile_summary(values: list[float], digits: int = 6) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": _round_float(np.min(array), digits),
        "p05": _round_float(np.percentile(array, 5), digits),
        "p25": _round_float(np.percentile(array, 25), digits),
        "median": _round_float(np.percentile(array, 50), digits),
        "p75": _round_float(np.percentile(array, 75), digits),
        "p95": _round_float(np.percentile(array, 95), digits),
        "max": _round_float(np.max(array), digits),
    }


def _output_shape_for_spacing(
    shape_xyz: tuple[int, int, int],
    spacing_xyz_mm: tuple[float, float, float],
    target_spacing_mm: float,
) -> tuple[int, int, int]:
    return tuple(
        max(1, int(round(float(size) * float(spacing) / float(target_spacing_mm))))
        for size, spacing in zip(shape_xyz, spacing_xyz_mm, strict=True)
    )


def _shape_tuple_to_text(values: tuple[int, ...] | list[int] | None) -> str:
    if values is None:
        return ""
    return "x".join(str(int(value)) for value in values)


def _float_tuple_to_text(values: tuple[float, ...] | list[float] | None) -> str:
    if values is None:
        return ""
    return "x".join(f"{float(value):.6g}" for value in values)


def _counter_top(counter: Counter, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"value": str(value), "count": int(count)}
        for value, count in counter.most_common(limit)
    ]


def _label_status_for_split(split: str, seg_path: Path) -> str:
    if seg_path.is_file():
        return "present"
    if split == "test":
        return "expected_missing_test_label"
    return "missing"


def _case_record(
    entry: dict[str, Any],
    ct_root: Path,
    seg_dir: Path,
    target_spacings_mm: tuple[float, ...],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    split = str(entry["_split"])
    name = str(entry["name"])
    ct_path = ct_rate_abs_path(name, ct_root)
    seg_path = seg_dir / name
    finding_count = len(entry.get("findings", {}))

    record: dict[str, Any] = {
        "split": split,
        "split_index": int(entry["_index"]),
        "name": name,
        "ct_path": str(ct_path),
        "seg_path": str(seg_path),
        "finding_count": finding_count,
        "ct_exists": ct_path.is_file(),
        "label_status": _label_status_for_split(split, seg_path),
    }
    label_record: dict[str, Any] | None = None

    if not ct_path.is_file():
        return record, label_record

    ct_img = nib.load(str(ct_path))
    ct_shape_xyz = tuple(int(value) for value in ct_img.shape[:3])
    ct_spacing_xyz = tuple(float(value) for value in ct_img.header.get_zooms()[:3])
    ct_affine = np.asarray(ct_img.affine)
    ct_dtype = str(np.dtype(ct_img.get_data_dtype()))
    ct_slope = float(getattr(ct_img.dataobj, "slope", 1.0) or 1.0)
    ct_intercept = float(getattr(ct_img.dataobj, "inter", 0.0) or 0.0)
    ct_axcodes = "".join(nib.aff2axcodes(ct_affine))
    voxel_volume_mm3 = float(np.prod(ct_spacing_xyz))
    metadata_shape = entry.get("shape")
    metadata_shape_xyz = (
        tuple(int(value) for value in metadata_shape)
        if isinstance(metadata_shape, list) and len(metadata_shape) >= 3
        else None
    )

    record.update(
        {
            "ct_shape_xyz": ct_shape_xyz,
            "ct_spacing_xyz_mm": ct_spacing_xyz,
            "ct_shape_x": ct_shape_xyz[0],
            "ct_shape_y": ct_shape_xyz[1],
            "ct_shape_z": ct_shape_xyz[2],
            "ct_spacing_x_mm": ct_spacing_xyz[0],
            "ct_spacing_y_mm": ct_spacing_xyz[1],
            "ct_spacing_z_mm": ct_spacing_xyz[2],
            "ct_physical_x_mm": ct_shape_xyz[0] * ct_spacing_xyz[0],
            "ct_physical_y_mm": ct_shape_xyz[1] * ct_spacing_xyz[1],
            "ct_physical_z_mm": ct_shape_xyz[2] * ct_spacing_xyz[2],
            "ct_voxel_volume_mm3": voxel_volume_mm3,
            "ct_voxels": int(np.prod(ct_shape_xyz)),
            "ct_axcodes": ct_axcodes,
            "ct_dtype": ct_dtype,
            "ct_slope": ct_slope,
            "ct_intercept": ct_intercept,
            "ct_materialized_hu_header_passed": (
                ct_dtype == "int16" and ct_slope == 1.0 and ct_intercept == 0.0
            ),
            "metadata_shape_xyz": metadata_shape_xyz,
            "metadata_shape_matches_ct": metadata_shape_xyz == ct_shape_xyz,
            "ct_inplane_spacing_equal": (
                abs(ct_spacing_xyz[0] - ct_spacing_xyz[1]) <= 1e-4
            ),
            "ct_isotropic_spacing": (
                abs(ct_spacing_xyz[0] - ct_spacing_xyz[1]) <= 1e-4
                and abs(ct_spacing_xyz[0] - ct_spacing_xyz[2]) <= 1e-4
            ),
        }
    )

    for target in target_spacings_mm:
        out_shape = _output_shape_for_spacing(ct_shape_xyz, ct_spacing_xyz, target)
        out_voxels = int(np.prod(out_shape))
        label = _spacing_label(target)
        record.update(
            {
                f"iso_{label}_shape_xyz": out_shape,
                f"iso_{label}_shape_x": out_shape[0],
                f"iso_{label}_shape_y": out_shape[1],
                f"iso_{label}_shape_z": out_shape[2],
                f"iso_{label}_voxels": out_voxels,
                f"iso_{label}_float32_image_gib": out_voxels * 4.0 / GIB,
                f"iso_{label}_dense_uint8_targets_gib": (
                    out_voxels * finding_count / GIB if split in {"train", "val"} else 0.0
                ),
            }
        )

    if seg_path.is_file():
        gt_img = nib.load(str(seg_path))
        gt_shape = tuple(int(value) for value in gt_img.shape)
        gt_spacings = tuple(float(value) for value in gt_img.header.get_zooms()[: len(gt_shape)])
        gt_affine = np.asarray(gt_img.affine)
        expected_shape_fxyz = (finding_count, *ct_shape_xyz)
        label_record = {
            "split": split,
            "name": name,
            "label_status": "present",
            "label_shape_fxyz": gt_shape,
            "expected_label_shape_fxyz": expected_shape_fxyz,
            "label_shape_matches_ct": gt_shape == expected_shape_fxyz,
            "label_dtype": str(np.dtype(gt_img.get_data_dtype())),
            "label_zooms": gt_spacings,
            "label_axcodes": "".join(nib.aff2axcodes(gt_affine)),
            "label_affine_matches_ct": np.allclose(
                gt_affine,
                ct_affine,
                atol=1e-5,
                rtol=1e-5,
            ),
            "label_spacing_source_policy": "inherit_matched_ct_spacing",
        }
        record.update(label_record)
    else:
        record.update(
            {
                "label_shape_fxyz": None,
                "expected_label_shape_fxyz": None,
                "label_shape_matches_ct": None,
                "label_dtype": None,
                "label_zooms": None,
                "label_axcodes": None,
                "label_affine_matches_ct": None,
                "label_spacing_source_policy": (
                    "test_has_no_released_labels"
                    if split == "test"
                    else "missing_label_for_released_label_split"
                ),
            }
        )

    return record, label_record


def _select_entries(
    metadata: Path,
    splits: list[str],
    max_cases_per_split: int | None,
) -> list[dict[str, Any]]:
    entries = load_split_entries(metadata, splits)
    if max_cases_per_split is None:
        return entries
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for entry in entries:
        split = str(entry["_split"])
        if counts[split] < max_cases_per_split:
            selected.append(entry)
            counts[split] += 1
    return selected


def _summarize_ct_group(
    rows: list[dict[str, Any]],
    target_spacings_mm: tuple[float, ...],
) -> dict[str, Any]:
    loaded = [row for row in rows if row.get("ct_exists")]
    summary: dict[str, Any] = {
        "cases": len(rows),
        "ct_loaded": len(loaded),
        "ct_missing": len(rows) - len(loaded),
        "spacing_x_mm": _percentile_summary([row["ct_spacing_x_mm"] for row in loaded]),
        "spacing_y_mm": _percentile_summary([row["ct_spacing_y_mm"] for row in loaded]),
        "spacing_z_mm": _percentile_summary([row["ct_spacing_z_mm"] for row in loaded]),
        "voxel_volume_mm3": _percentile_summary([row["ct_voxel_volume_mm3"] for row in loaded]),
        "shape_x": _percentile_summary([row["ct_shape_x"] for row in loaded], digits=3),
        "shape_y": _percentile_summary([row["ct_shape_y"] for row in loaded], digits=3),
        "shape_z": _percentile_summary([row["ct_shape_z"] for row in loaded], digits=3),
        "physical_x_mm": _percentile_summary([row["ct_physical_x_mm"] for row in loaded]),
        "physical_y_mm": _percentile_summary([row["ct_physical_y_mm"] for row in loaded]),
        "physical_z_mm": _percentile_summary([row["ct_physical_z_mm"] for row in loaded]),
        "native_voxels": _percentile_summary([row["ct_voxels"] for row in loaded], digits=3),
        "finding_count": _percentile_summary([row["finding_count"] for row in loaded], digits=3),
        "inplane_equal_count_tol_1e-4": sum(
            bool(row.get("ct_inplane_spacing_equal")) for row in loaded
        ),
        "isotropic_count_tol_1e-4": sum(bool(row.get("ct_isotropic_spacing")) for row in loaded),
        "ct_dtype_counts": dict(Counter(row["ct_dtype"] for row in loaded)),
        "ct_axcode_counts": dict(Counter(row["ct_axcodes"] for row in loaded)),
        "ct_slope_intercept_counts": {
            str(key): int(value)
            for key, value in Counter(
                (row["ct_slope"], row["ct_intercept"]) for row in loaded
            ).items()
        },
        "materialized_hu_header_passed": sum(
            bool(row.get("ct_materialized_hu_header_passed")) for row in loaded
        ),
        "top_spacing_xyz_mm": _counter_top(
            Counter(
                tuple(round(float(value), 6) for value in row["ct_spacing_xyz_mm"])
                for row in loaded
            )
        ),
        "z_spacing_bins_rounded_0.1mm": {
            f"{key:.1f}": int(value)
            for key, value in sorted(
                Counter(round(float(row["ct_spacing_z_mm"]), 1) for row in loaded).items()
            )
        },
    }
    for target in target_spacings_mm:
        label = _spacing_label(target)
        summary[f"isotropic_{label}mm"] = {
            "target_spacing_mm": float(target),
            "shape_x": _percentile_summary(
                [row[f"iso_{label}_shape_x"] for row in loaded],
                digits=3,
            ),
            "shape_y": _percentile_summary(
                [row[f"iso_{label}_shape_y"] for row in loaded],
                digits=3,
            ),
            "shape_z": _percentile_summary(
                [row[f"iso_{label}_shape_z"] for row in loaded],
                digits=3,
            ),
            "voxels_per_case": _percentile_summary(
                [row[f"iso_{label}_voxels"] for row in loaded],
                digits=3,
            ),
            "float32_image_gib_total": _round_float(
                sum(float(row[f"iso_{label}_float32_image_gib"]) for row in loaded),
                digits=3,
            ),
            "dense_uint8_targets_gib_total": _round_float(
                sum(float(row[f"iso_{label}_dense_uint8_targets_gib"]) for row in loaded),
                digits=3,
            ),
        }
    return summary


def _summarize_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    released = [row for row in rows if row["split"] in {"train", "val"}]
    test = [row for row in rows if row["split"] == "test"]
    present = [row for row in rows if row.get("label_status") == "present"]
    return {
        "released_label_splits": ["train", "val"],
        "test_labels_expected": False,
        "train_val_cases": len(released),
        "train_val_label_files_present": sum(row.get("label_status") == "present" for row in released),
        "train_val_label_files_missing": sum(row.get("label_status") == "missing" for row in released),
        "test_cases": len(test),
        "test_label_files_missing_as_expected": sum(
            row.get("label_status") == "expected_missing_test_label" for row in test
        ),
        "test_label_files_unexpectedly_present": sum(
            row.get("label_status") == "present" for row in test
        ),
        "label_shape_matches_ct_count": sum(
            row.get("label_shape_matches_ct") is True for row in present
        ),
        "label_shape_mismatch_count": sum(
            row.get("label_shape_matches_ct") is False for row in present
        ),
        "label_affine_matches_ct_count": sum(
            row.get("label_affine_matches_ct") is True for row in present
        ),
        "label_affine_mismatch_count": sum(
            row.get("label_affine_matches_ct") is False for row in present
        ),
        "label_dtype_counts": dict(
            Counter(row.get("label_dtype") for row in present if row.get("label_dtype"))
        ),
        "label_axcode_counts": dict(
            Counter(row.get("label_axcodes") for row in present if row.get("label_axcodes"))
        ),
        "label_zoom_counts_top20": _counter_top(
            Counter(
                tuple(round(float(value), 6) for value in row.get("label_zooms") or [])
                for row in present
            )
        ),
        "spacing_policy": (
            "Released labels are interpreted as evaluator-layout masks on the "
            "matched CT voxel grid. Use CT header spacing/affine as physical "
            "geometry truth; label NIfTI affine/zooms are not used as spacing truth."
        ),
    }


def _build_summary(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    target_spacings_mm: tuple[float, ...],
) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    groups = {}
    for group, splits in SUMMARY_GROUPS.items():
        group_rows = [row for row in rows if row["split"] in splits]
        groups[group] = _summarize_ct_group(group_rows, target_spacings_mm)
    return {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "ct_root": str(args.ct_root),
        "seg_dir": str(args.seg_dir),
        "output_dir": str(args.output_dir),
        "splits_requested": args.splits,
        "max_cases_per_split": args.max_cases,
        "target_spacings_mm": list(target_spacings_mm),
        "case_count": len(rows),
        "split_counts": dict(split_counts),
        "ct_missing_count": sum(not row.get("ct_exists") for row in rows),
        "ct_materialized_hu_header_failures": sum(
            row.get("ct_exists") and not row.get("ct_materialized_hu_header_passed")
            for row in rows
        ),
        "metadata_shape_mismatch_count": sum(
            row.get("ct_exists") and not row.get("metadata_shape_matches_ct") for row in rows
        ),
        "ct_summaries": groups,
        "label_audit": _summarize_labels(rows),
        "storage_warning": (
            "Candidate storage estimates are full-FOV, uncompressed tensor estimates. "
            "A later cache may differ because crop-to-nonzero, compression, sparse "
            "label storage, or changed dtype can reduce or change actual disk use."
        ),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return _shape_tuple_to_text(value) if all(isinstance(v, int) for v in value) else _float_tuple_to_text(value)
    if isinstance(value, list):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.9g}"
        return value
    return value


def _write_cases_csv(path: Path, rows: list[dict[str, Any]], target_spacings_mm: tuple[float, ...]) -> None:
    base_fields = [
        "split",
        "split_index",
        "name",
        "finding_count",
        "ct_exists",
        "ct_shape_xyz",
        "ct_spacing_xyz_mm",
        "ct_physical_x_mm",
        "ct_physical_y_mm",
        "ct_physical_z_mm",
        "ct_voxel_volume_mm3",
        "ct_voxels",
        "ct_axcodes",
        "ct_dtype",
        "ct_slope",
        "ct_intercept",
        "ct_materialized_hu_header_passed",
        "ct_inplane_spacing_equal",
        "ct_isotropic_spacing",
        "metadata_shape_matches_ct",
        "label_status",
        "label_shape_fxyz",
        "expected_label_shape_fxyz",
        "label_shape_matches_ct",
        "label_dtype",
        "label_zooms",
        "label_axcodes",
        "label_affine_matches_ct",
        "label_spacing_source_policy",
    ]
    iso_fields: list[str] = []
    for target in target_spacings_mm:
        label = _spacing_label(target)
        iso_fields.extend(
            [
                f"iso_{label}_shape_xyz",
                f"iso_{label}_voxels",
                f"iso_{label}_float32_image_gib",
                f"iso_{label}_dense_uint8_targets_gib",
            ]
        )
    fields = base_fields + iso_fields + ["ct_path", "seg_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _table_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def _format_summary_cell(summary: dict[str, Any], key: str) -> str:
    value = summary[key]
    if value is None:
        return ""
    return str(value)


def _format_range(summary: dict[str, Any]) -> str:
    return (
        f"{_format_summary_cell(summary, 'min')} / "
        f"{_format_summary_cell(summary, 'median')} / "
        f"{_format_summary_cell(summary, 'max')}"
    )


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    full = summary["max_cases_per_split"] is None
    lines: list[str] = [
        "# Native Resolution Audit For Isotropic VoxTell Planning",
        "",
        f"Created: `{summary['created_at_utc']}`",
        "",
        "This audit reads NIfTI headers only. It does not resample CTs or labels, "
        "build caches, write predictions, or launch finetuning.",
        "",
        "## Scope",
        "",
        _table_row(["Field", "Value"]),
        _table_row(["---", "---"]),
        _table_row(["Splits requested", ", ".join(summary["splits_requested"])]),
        _table_row(["Max cases per split", summary["max_cases_per_split"] or "none"]),
        _table_row(["Cases audited", summary["case_count"]]),
        _table_row(["CT missing", summary["ct_missing_count"]]),
        _table_row(
            [
                "CT HU header failures",
                summary["ct_materialized_hu_header_failures"],
            ]
        ),
        _table_row(["Metadata shape mismatches", summary["metadata_shape_mismatch_count"]]),
        "",
    ]
    if not full:
        lines.extend(
            [
                "> This is a limited audit because `--max-cases` was used. Run without "
                "`--max-cases` for the authoritative full split summary.",
                "",
            ]
        )

    lines.extend(
        [
            "## CT Geometry",
            "",
            _table_row(
                [
                    "Split/group",
                    "N",
                    "Spacing X min/median/max",
                    "Spacing Y min/median/max",
                    "Spacing Z min/median/max",
                    "Shape Z min/median/max",
                    "In-plane equal",
                    "Isotropic",
                ]
            ),
            _table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
        ]
    )
    for group in ["train", "val", "test", "train+val", "all"]:
        ct = summary["ct_summaries"][group]
        lines.append(
            _table_row(
                [
                    group,
                    ct["ct_loaded"],
                    _format_range(ct["spacing_x_mm"]),
                    _format_range(ct["spacing_y_mm"]),
                    _format_range(ct["spacing_z_mm"]),
                    _format_range(ct["shape_z"]),
                    ct["inplane_equal_count_tol_1e-4"],
                    ct["isotropic_count_tol_1e-4"],
                ]
            )
        )

    lines.extend(["", "## Common CT Spacings", ""])
    for group in ["train", "val", "test"]:
        ct = summary["ct_summaries"][group]
        values = ", ".join(
            f"`{item['value']}`: {item['count']}" for item in ct["top_spacing_xyz_mm"][:8]
        )
        lines.append(f"- `{group}` top `(X, Y, Z)` spacing tuples: {values}")
        bins = ", ".join(
            f"`{key}`: {value}" for key, value in ct["z_spacing_bins_rounded_0.1mm"].items()
        )
        lines.append(f"- `{group}` Z-spacing bins rounded to 0.1 mm: {bins}")

    label = summary["label_audit"]
    lines.extend(
        [
            "",
            "## Label Audit",
            "",
            _table_row(["Field", "Value"]),
            _table_row(["---", "---:"]),
            _table_row(["Train/val cases", label["train_val_cases"]]),
            _table_row(["Train/val label files present", label["train_val_label_files_present"]]),
            _table_row(["Train/val label files missing", label["train_val_label_files_missing"]]),
            _table_row(["Test cases", label["test_cases"]]),
            _table_row(
                [
                    "Test labels missing as expected",
                    label["test_label_files_missing_as_expected"],
                ]
            ),
            _table_row(["Label shape mismatches", label["label_shape_mismatch_count"]]),
            _table_row(["Label affine mismatches", label["label_affine_mismatch_count"]]),
            "",
            "Released labels are grid-aligned to CT by shape and finding count. "
            "Use matched CT headers as the physical spacing source; label NIfTI "
            "affine and zoom metadata are not used as physical-resolution truth.",
            "",
            "## Isotropic Candidate Impact",
            "",
            _table_row(
                [
                    "Target",
                    "Group",
                    "N",
                    "Shape Z min/median/max",
                    "Voxels/case min/median/max",
                    "Image GiB",
                    "Dense target GiB",
                ]
            ),
            _table_row(["---:", "---", "---:", "---:", "---:", "---:", "---:"]),
        ]
    )
    for target in summary["target_spacings_mm"]:
        key = f"isotropic_{_spacing_label(target)}mm"
        for group in ["train", "val", "test", "train+val", "all"]:
            ct = summary["ct_summaries"][group]
            iso = ct[key]
            lines.append(
                _table_row(
                    [
                        f"{target:g} mm",
                        group,
                        ct["ct_loaded"],
                        _format_range(iso["shape_z"]),
                        _format_range(iso["voxels_per_case"]),
                        iso["float32_image_gib_total"],
                        iso["dense_uint8_targets_gib_total"],
                    ]
                )
            )

    lines.extend(
        [
            "",
            "## Storage Caveat",
            "",
            summary["storage_warning"],
            "",
            "## Artifacts",
            "",
            "- `native_resolution_summary.json`: machine-readable aggregate summary.",
            "- `native_resolution_cases.csv`: one row per audited CT case.",
            "- `report.md`: this human-readable report.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", choices=SPLIT_ORDER, default=list(SPLIT_ORDER))
    parser.add_argument(
        "--target-spacing-mm",
        type=float,
        action="append",
        default=None,
        help="Candidate isotropic spacing in mm. Repeat to audit multiple targets.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional smoke-test limit applied per selected split.",
    )
    args = parser.parse_args()

    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")
    target_spacings_mm = tuple(args.target_spacing_mm or DEFAULT_TARGET_SPACINGS_MM)
    if not target_spacings_mm or any(value <= 0 for value in target_spacings_mm):
        raise ValueError("All target spacings must be positive")

    entries = _select_entries(args.metadata, args.splits, args.max_cases)
    if not entries:
        raise ValueError("No entries selected")

    rows: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []
    for entry in entries:
        record, label_record = _case_record(
            entry,
            ct_root=args.ct_root,
            seg_dir=args.seg_dir,
            target_spacings_mm=target_spacings_mm,
        )
        rows.append(record)
        if label_record is not None:
            label_records.append(label_record)

    summary = _build_summary(rows, args, target_spacings_mm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "native_resolution_summary.json", summary)
    _write_cases_csv(args.output_dir / "native_resolution_cases.csv", rows, target_spacings_mm)
    _write_report(args.output_dir / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
