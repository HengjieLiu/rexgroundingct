#!/usr/bin/env python3
"""Report diffuse/focal ReXGroundingCT lesion sizes from verified JSON counts."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from common import CT_ROOT, EXP_ROOT, REX_METADATA, REX_SEG_DIR, ct_rate_abs_path, load_split_entries


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

VOXEL_BINS = [
    ("<100", 0.0, 100.0),
    ("100-1k", 100.0, 1_000.0),
    ("1k-10k", 1_000.0, 10_000.0),
    ("10k-100k", 10_000.0, 100_000.0),
    (">=100k", 100_000.0, None),
]
MM3_BINS = VOXEL_BINS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EXP_ROOT / "diffuse_focal_lesion_size_audit",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--qc-sample-cases", type=int, default=320)
    parser.add_argument("--qc-seed", type=int, default=20260726)
    return parser.parse_args()


def sorted_finding_keys(entry: dict[str, Any]) -> list[str]:
    return sorted((entry.get("findings") or {}).keys(), key=lambda value: int(value))


def focality_for_category(category: str) -> str:
    if category in CATEGORY_INFO:
        return CATEGORY_INFO[category][0]
    if category.startswith("1"):
        return "diffuse"
    if category.startswith("2"):
        return "focal"
    return "unknown"


def category_name(category: str) -> str:
    return CATEGORY_INFO.get(category, ("unknown", "Unknown"))[1]


def mask_channels(mask_path: Path, expected_channels: int) -> np.ndarray:
    image = nib.load(str(mask_path))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 3 and expected_channels == 1:
        return data[None]
    if data.ndim == 4:
        if data.shape[0] == expected_channels:
            return data
        if data.shape[-1] == expected_channels:
            return np.moveaxis(data, -1, 0)
    raise ValueError(
        f"{mask_path}: expected {expected_channels} mask channel(s), got shape {tuple(data.shape)}"
    )


def voxel_volume_mm3(ct_path: Path) -> float:
    image = nib.load(str(ct_path))
    volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    if not np.isfinite(volume) or volume <= 0:
        volume = float(np.prod(image.header.get_zooms()[:3]))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"{ct_path}: invalid CT voxel volume {volume}")
    return volume


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_pixel_counts(
    entries: list[dict[str, Any]],
    seg_dir: Path,
    sample_cases: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    sample = rng.sample(entries, min(sample_cases, len(entries)))
    qc_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for entry in sample:
        split = str(entry["_split"])
        case = str(entry["name"])
        mask_path = seg_dir / case
        finding_keys = sorted_finding_keys(entry)
        if not mask_path.is_file():
            row = {
                "split": split,
                "case": case,
                "status": "missing_mask",
                "mask_path": str(mask_path),
            }
            qc_rows.append(row)
            failures.append(row)
            continue

        try:
            masks = mask_channels(mask_path, len(finding_keys))
        except Exception as exc:
            row = {
                "split": split,
                "case": case,
                "status": "mask_shape_mismatch",
                "error": str(exc),
                "mask_path": str(mask_path),
            }
            qc_rows.append(row)
            failures.append(row)
            continue

        for channel, key in enumerate(finding_keys):
            category = str((entry.get("categories") or {}).get(key, "unknown"))
            json_pixels = int((entry.get("pixels") or {}).get(key, -1))
            mask_voxels = int(np.count_nonzero(masks[channel]))
            delta = mask_voxels - json_pixels
            status = "ok" if delta == 0 else "pixel_mismatch"
            row = {
                "split": split,
                "case": case,
                "finding_idx": int(key),
                "channel": channel,
                "category": category,
                "focality": focality_for_category(category),
                "json_pixels": json_pixels,
                "mask_voxels": mask_voxels,
                "delta": delta,
                "status": status,
                "mask_path": str(mask_path),
            }
            qc_rows.append(row)
            if status != "ok":
                failures.append(row)

    return qc_rows, failures


def finding_rows(entries: list[dict[str, Any]], seg_dir: Path, ct_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    voxel_cache: dict[str, float] = {}

    for entry in entries:
        split = str(entry["_split"])
        case = str(entry["name"])
        ct_path = ct_rate_abs_path(case, ct_root)
        if not ct_path.is_file():
            raise FileNotFoundError(f"Missing CT for {case}: {ct_path}")
        voxel_mm3 = voxel_cache.get(case)
        if voxel_mm3 is None:
            voxel_mm3 = voxel_volume_mm3(ct_path)
            voxel_cache[case] = voxel_mm3

        findings = entry.get("findings") or {}
        categories = entry.get("categories") or {}
        pixels = entry.get("pixels") or {}
        entity_counts = entry.get("entity_counts") or {}
        for key in sorted_finding_keys(entry):
            if key not in categories:
                raise KeyError(f"{case} finding {key}: missing category")
            if key not in pixels:
                raise KeyError(f"{case} finding {key}: missing pixel count")
            category = str(categories.get(key, "unknown"))
            focality = focality_for_category(category)
            if focality == "unknown":
                raise ValueError(f"{case} finding {key}: unknown focality for category {category!r}")
            count = int(pixels[key])
            volume = float(count) * float(voxel_mm3)
            rows.append(
                {
                    "split": split,
                    "case": case,
                    "finding_idx": int(key),
                    "finding_ordinal": int(key) + 1,
                    "category": category,
                    "category_name": category_name(category),
                    "focality": focality,
                    "entity_count": entity_counts.get(key, ""),
                    "pixels": count,
                    "voxel_mm3": voxel_mm3,
                    "volume_mm3": volume,
                    "volume_ml": volume / 1000.0,
                    "prompt": findings.get(key, ""),
                    "segmentation_path": str(seg_dir / case),
                    "ct_path": str(ct_path),
                }
            )
    return rows


def augmented_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows + [{**row, "split": "train+val"} for row in rows]


def stats_for(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    p10, p25, p50, p75, p90 = np.percentile(array, [10, 25, 50, 75, 90])
    return {
        "mean": float(array.mean()),
        "median": float(p50),
        "p10": float(p10),
        "q1": float(p25),
        "q3": float(p75),
        "p90": float(p90),
        "min": float(array.min()),
        "max": float(array.max()),
        "total": float(array.sum()),
    }


def summary_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in augmented_records(rows):
        groups[tuple(row[field] for field in group_fields)].append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        record = {field: value for field, value in zip(group_fields, key)}
        record["n_findings"] = len(group)
        record["n_cases"] = len({row["case"] for row in group})
        for field in ["pixels", "volume_mm3", "volume_ml"]:
            stats = stats_for([float(row[field]) for row in group])
            for stat_name, value in stats.items():
                record[f"{field}_{stat_name}"] = value
        output.append(record)
    return output


def bin_label(value: float, bins: list[tuple[str, float, float | None]]) -> str:
    for label, lower, upper in bins:
        if value >= lower and (upper is None or value < upper):
            return label
    raise ValueError(f"Value {value} did not match any bin")


def size_bin_rows(
    rows: list[dict[str, Any]],
    value_field: str,
    bins: list[tuple[str, float, float | None]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in augmented_records(rows):
        groups[(str(row["split"]), str(row["focality"]))].append(row)

    output: list[dict[str, Any]] = []
    for (split, focality), group in sorted(groups.items()):
        counts = {label: 0 for label, _, _ in bins}
        for row in group:
            counts[bin_label(float(row[value_field]), bins)] += 1
        total = len(group)
        for label, _, _ in bins:
            count = counts[label]
            output.append(
                {
                    "split": split,
                    "focality": focality,
                    "bin": label,
                    "n_findings": count,
                    "fraction": count / total if total else 0.0,
                }
            )
    return output


def fmt_float(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    focality_summary: list[dict[str, Any]],
    category_summary: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    qc_failures: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    by_split_focality = {
        (row["split"], row["focality"]): row
        for row in focality_summary
        if row["split"] in {"train", "val", "train+val"}
    }
    lines = [
        "# Diffuse/Focal Lesion Size Audit",
        "",
        "## Inputs",
        "",
        f"- Metadata: `{args.metadata}`",
        f"- Segmentations for QC: `{args.seg_dir}`",
        f"- CT root for voxel spacing: `{args.ct_root}`",
        f"- QC seed: `{args.qc_seed}`",
        f"- QC sampled cases: `{min(args.qc_sample_cases, len(load_split_entries(args.metadata, args.splits)))}`",
        "",
        "## QC",
        "",
        f"- QC findings checked: `{len([row for row in qc_rows if row.get('finding_idx') != ''])}`",
        f"- QC failures: `{len(qc_failures)}`",
        "",
        "## Summary By Split And Focality",
        "",
        "| Split | Focality | Findings | Cases | Median voxels | Median mm3 | Mean mm3 | P90 mm3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["train", "val", "train+val"]:
        for focality in ["diffuse", "focal", "unknown"]:
            row = by_split_focality.get((split, focality))
            if row is None:
                continue
            lines.append(
                f"| {split} | {focality} | {row['n_findings']} | {row['n_cases']} | "
                f"{fmt_float(row['pixels_median'])} | {fmt_float(row['volume_mm3_median'])} | "
                f"{fmt_float(row['volume_mm3_mean'])} | {fmt_float(row['volume_mm3_p90'])} |"
            )

    lines.extend(
        [
            "",
            "## Category Highlights",
            "",
            "| Split | Category | Focality | Findings | Median voxels | Median mm3 | P90 mm3 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in category_summary:
        if row["split"] != "train+val":
            continue
        lines.append(
            f"| {row['split']} | {row['category']} | {row['focality']} | {row['n_findings']} | "
            f"{fmt_float(row['pixels_median'])} | {fmt_float(row['volume_mm3_median'])} | "
            f"{fmt_float(row['volume_mm3_p90'])} |"
        )

    largest = sorted(rows, key=lambda item: float(item["volume_mm3"]), reverse=True)[:10]
    lines.extend(
        [
            "",
            "## Largest Findings",
            "",
            "| Split | Case | Finding | Category | Focality | Voxels | Volume mm3 | Prompt |",
            "| --- | --- | ---: | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in largest:
        prompt = str(row["prompt"]).replace("|", "\\|")
        lines.append(
            f"| {row['split']} | {row['case']} | {row['finding_idx']} | {row['category']} | "
            f"{row['focality']} | {row['pixels']} | {fmt_float(row['volume_mm3'])} | {prompt} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `finding_sizes.csv`",
            "- `summary_by_split_focality.csv`",
            "- `summary_by_category.csv`",
            "- `size_bin_counts_voxels.csv`",
            "- `size_bin_counts_mm3.csv`",
            "- `qc_pixel_count_sample.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    entries = load_split_entries(args.metadata, args.splits)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qc_rows, qc_failures = validate_pixel_counts(
        entries,
        args.seg_dir,
        args.qc_sample_cases,
        args.qc_seed,
    )
    qc_fields = [
        "split",
        "case",
        "finding_idx",
        "channel",
        "category",
        "focality",
        "json_pixels",
        "mask_voxels",
        "delta",
        "status",
        "error",
        "mask_path",
    ]
    write_csv(args.out_dir / "qc_pixel_count_sample.csv", qc_rows, qc_fields)
    if qc_failures:
        raise RuntimeError(
            f"JSON pixel-count QC failed for {len(qc_failures)} row(s). "
            f"See {args.out_dir / 'qc_pixel_count_sample.csv'}"
        )

    rows = finding_rows(entries, args.seg_dir, args.ct_root)
    finding_fields = [
        "split",
        "case",
        "finding_idx",
        "finding_ordinal",
        "category",
        "category_name",
        "focality",
        "entity_count",
        "pixels",
        "voxel_mm3",
        "volume_mm3",
        "volume_ml",
        "prompt",
        "segmentation_path",
        "ct_path",
    ]
    write_csv(args.out_dir / "finding_sizes.csv", rows, finding_fields)

    focality_summary = summary_rows(rows, ["split", "focality"])
    category_summary = summary_rows(rows, ["split", "category", "category_name", "focality"])
    summary_fields = list(focality_summary[0].keys())
    category_fields = list(category_summary[0].keys())
    write_csv(args.out_dir / "summary_by_split_focality.csv", focality_summary, summary_fields)
    write_csv(args.out_dir / "summary_by_category.csv", category_summary, category_fields)
    write_csv(
        args.out_dir / "size_bin_counts_voxels.csv",
        size_bin_rows(rows, "pixels", VOXEL_BINS),
        ["split", "focality", "bin", "n_findings", "fraction"],
    )
    write_csv(
        args.out_dir / "size_bin_counts_mm3.csv",
        size_bin_rows(rows, "volume_mm3", MM3_BINS),
        ["split", "focality", "bin", "n_findings", "fraction"],
    )
    write_csv(args.out_dir / "largest_findings.csv", sorted(rows, key=lambda row: row["volume_mm3"], reverse=True)[:100], finding_fields)
    write_csv(args.out_dir / "smallest_findings.csv", sorted(rows, key=lambda row: row["volume_mm3"])[:100], finding_fields)
    write_report(
        args.out_dir / "report.md",
        rows,
        focality_summary,
        category_summary,
        qc_rows,
        qc_failures,
        args,
    )
    print(f"Wrote {len(rows)} finding rows to {args.out_dir}")
    print(f"QC checked {len(qc_rows)} sampled finding row(s) with {len(qc_failures)} failure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
