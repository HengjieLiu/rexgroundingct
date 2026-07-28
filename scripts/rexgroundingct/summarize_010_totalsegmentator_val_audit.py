#!/usr/bin/env python3
"""Summarize and visualize the exp010 TotalSegmentator val200 audit."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "010_voxtell_public_anatomy_prior_fusion/config/"
    "val200_totalsegmentator_case_plan.jsonl"
)
DEFAULT_CACHE_ROOT = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/"
    "totalsegmentator_total_fast_3mm_v2_16_0"
)
DEFAULT_EXP_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "010_voxtell_public_anatomy_prior_fusion"
)
DEFAULT_SEG_DIR = Path("/data/hengjie/datasets/rexgroundingct/segmentations")

LUNG_LABELS = {10, 11, 12, 13, 14}
TRACHEA_LABELS = {16}
CARDIOVASCULAR_LABELS = set(range(51, 69))
THORACIC_BONE_LABELS = set(range(32, 45)) | set(range(92, 118))

ATOMIC_THORACIC_LABELS = {
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "esophagus",
    "trachea",
    "heart",
    "aorta",
    "pulmonary_vein",
    "superior_vena_cava",
}

GROUP_DEFINITIONS = {
    "whole_lung": LUNG_LABELS,
    "left_lung": {10, 11},
    "right_lung": {12, 13, 14},
    "trachea": TRACHEA_LABELS,
    "central_cardiovascular": CARDIOVASCULAR_LABELS,
    "thoracic_spine": set(range(32, 45)),
    "left_ribs": set(range(92, 104)),
    "right_ribs": set(range(104, 116)),
    "anterior_chest_wall": {116, 117},
}

RECOMMENDED_GROUP_CHANNELS = {
    "other_central_vessels": set(range(54, 62)),
    "thoracic_spine": set(range(32, 45)),
    "left_ribs": set(range(92, 104)),
    "right_ribs": set(range(104, 116)),
    "anterior_chest_wall": {116, 117},
}

GROUP_COLORS = {
    "lung_lobes": (0.10, 0.75, 0.95),
    "trachea": (1.00, 0.80, 0.05),
    "cardiovascular": (0.92, 0.20, 0.25),
    "thoracic_bones": (0.70, 0.90, 0.35),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def query_class_map(task: str = "total") -> dict[int, str]:
    output = subprocess.check_output(
        ["totalseg_info", "--classes", "-ta", task, "--json"],
        text=True,
    )
    return {int(key): value for key, value in json.loads(output).items()}


def case_status(plan: list[dict[str, Any]], cache_root: Path) -> dict[str, Any]:
    complete: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    locked: list[str] = []
    for row in plan:
        case_dir = cache_root / "cases" / row["case_key"]
        if (case_dir / ".complete").is_file():
            complete.append(row["name"])
        elif (case_dir / "failure.json").is_file():
            failed.append(row["name"])
        else:
            missing.append(row["name"])
        if (case_dir / ".case.lock").is_dir():
            locked.append(row["name"])
    return {
        "checked_at_utc": utc_now(),
        "expected": len(plan),
        "complete": len(complete),
        "failed": len(failed),
        "missing": len(missing),
        "locked": len(locked),
        "complete_cases": complete,
        "failed_cases": failed,
        "first_missing_cases": missing[:20],
        "locked_cases": locked,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selection_class(label_name: str) -> str:
    if label_name in ATOMIC_THORACIC_LABELS:
        return "retain_atomic"
    if (
        label_name.startswith("rib_")
        or label_name.startswith("vertebrae_T")
        or label_name in {
            "sternum",
            "costal_cartilages",
            "spinal_cord",
            "scapula_left",
            "scapula_right",
            "clavicula_left",
            "clavicula_right",
        }
    ):
        return "retain_grouped"
    if label_name in {
        "brachiocephalic_trunk",
        "brachiocephalic_vein_left",
        "brachiocephalic_vein_right",
        "subclavian_artery_left",
        "subclavian_artery_right",
        "common_carotid_artery_left",
        "common_carotid_artery_right",
        "atrial_appendage_left",
    }:
        return "candidate_grouped_vessel"
    return "exclude_v1"


def collect_label_rows(
    plan: list[dict[str, Any]],
    cache_root: Path,
    class_map: dict[int, str],
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    case_count = len(plan)
    presence = np.zeros((case_count, len(class_map)), dtype=np.uint8)
    voxels = np.zeros((case_count, len(class_map)), dtype=np.int64)
    volumes = np.zeros((case_count, len(class_map)), dtype=np.float64)
    runtimes: list[dict[str, Any]] = []
    for case_index, row in enumerate(plan):
        metadata_path = cache_root / "cases" / row["case_key"] / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        geometry = metadata["geometry"]
        for label_text, count in geometry["class_voxels"].items():
            label_id = int(label_text)
            if label_id not in class_map:
                continue
            column = label_id - 1
            presence[case_index, column] = 1
            voxels[case_index, column] = int(count)
            volumes[case_index, column] = float(geometry["class_volume_mm3"][label_text])
        runtimes.append(
            {
                "val_order": row["val_order"],
                "case": row["name"],
                "runtime_seconds": float(metadata["runtime_seconds"]),
                "gpu_memory_baseline_used_mib": int(
                    metadata["gpu_memory_baseline_used_mib"]
                ),
                "gpu_memory_peak_total_used_mib": int(
                    metadata["gpu_memory_peak_total_used_mib"]
                ),
                "gpu_memory_peak_delta_mib": int(metadata["gpu_memory_peak_delta_mib"]),
                "observed_class_count": int(geometry["observed_class_count"]),
            }
        )

    label_rows: list[dict[str, Any]] = []
    for label_id, label_name in class_map.items():
        column = label_id - 1
        nonzero_volumes = volumes[:, column][volumes[:, column] > 0]
        cases_present = int(presence[:, column].sum())
        label_rows.append(
            {
                "label_id": label_id,
                "label_name": label_name,
                "selection": selection_class(label_name),
                "cases_present": cases_present,
                "case_fraction": cases_present / case_count,
                "total_voxels": int(voxels[:, column].sum()),
                "total_volume_mm3": float(volumes[:, column].sum()),
                "median_present_volume_mm3": (
                    float(np.median(nonzero_volumes)) if nonzero_volumes.size else 0.0
                ),
                "min_present_volume_mm3": (
                    float(nonzero_volumes.min()) if nonzero_volumes.size else 0.0
                ),
                "max_present_volume_mm3": (
                    float(nonzero_volumes.max()) if nonzero_volumes.size else 0.0
                ),
            }
        )
    return label_rows, presence, runtimes


def target_overlap_rows(
    plan: list[dict[str, Any]],
    cache_root: Path,
    seg_dir: Path,
    class_map: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in plan:
        anatomy_img = nib.load(
            str(cache_root / "cases" / case["case_key"] / "total_labels.nii.gz")
        )
        anatomy = np.asarray(anatomy_img.dataobj, dtype=np.uint8)
        target_path = seg_dir / case["name"]
        target_img = nib.load(str(target_path))
        targets = np.asarray(target_img.dataobj, dtype=np.uint8)
        if targets.ndim != 4 or tuple(targets.shape[1:]) != tuple(anatomy.shape):
            raise ValueError(
                f"{case['name']}: target/anatomy shape mismatch "
                f"{targets.shape} versus {anatomy.shape}"
            )
        finding_ids = sorted(case["findings"], key=lambda value: int(value))
        if len(finding_ids) != targets.shape[0]:
            raise ValueError(
                f"{case['name']}: {len(finding_ids)} prompts but {targets.shape[0]} targets"
            )
        for channel, finding_id in enumerate(finding_ids):
            target = targets[channel] > 0
            target_voxels = int(target.sum())
            if target_voxels == 0:
                raise ValueError(f"{case['name']} target {finding_id} is empty")
            target_anatomy = anatomy[target]
            labels_at_target, counts = np.unique(target_anatomy, return_counts=True)
            atomic = [
                (int(label), int(count))
                for label, count in zip(labels_at_target, counts, strict=True)
                if int(label) != 0
            ]
            atomic.sort(key=lambda item: item[1], reverse=True)
            best_label_id = atomic[0][0] if atomic else 0
            best_label_voxels = atomic[0][1] if atomic else 0
            row: dict[str, Any] = {
                "val_order": case["val_order"],
                "case": case["name"],
                "finding_id": finding_id,
                "category": case.get("categories", {}).get(finding_id, ""),
                "finding": case["findings"][finding_id],
                "target_voxels": target_voxels,
                "best_overlapping_label_id": best_label_id,
                "best_overlapping_label": class_map.get(best_label_id, "background"),
                "best_label_target_fraction": best_label_voxels / target_voxels,
                "background_target_fraction": float((target_anatomy == 0).mean()),
            }
            for group_name, label_ids in GROUP_DEFINITIONS.items():
                row[f"{group_name}_target_fraction"] = float(
                    np.isin(target_anatomy, list(label_ids)).mean()
                )
            rows.append(row)
    return rows


def window_ct(data: np.ndarray, center: float = -600.0, width: float = 1500.0) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    return np.clip((data.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def colored_overlay(seg_slice: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*seg_slice.shape, 4), dtype=np.float32)
    groups = [
        ("lung_lobes", LUNG_LABELS),
        ("trachea", TRACHEA_LABELS),
        ("cardiovascular", CARDIOVASCULAR_LABELS),
        ("thoracic_bones", THORACIC_BONE_LABELS),
    ]
    for group_name, labels in groups:
        mask = np.isin(seg_slice, list(labels))
        color = GROUP_COLORS[group_name]
        overlay[mask, :3] = color
        overlay[mask, 3] = 0.38
    return overlay


def best_slice(mask: np.ndarray, axis: int) -> int:
    other_axes = tuple(index for index in range(3) if index != axis)
    areas = mask.sum(axis=other_axes)
    return int(np.argmax(areas)) if np.any(areas) else mask.shape[axis] // 2


def oriented_slice(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return np.rot90(data[index, :, :])
    if axis == 1:
        return np.rot90(data[:, index, :])
    return np.rot90(data[:, :, index])


def render_case_qc(
    row: dict[str, Any],
    cache_root: Path,
    output_dir: Path,
) -> Path:
    ct = nib.as_closest_canonical(nib.load(row["ct_path"]))
    segmentation = nib.as_closest_canonical(
        nib.load(str(cache_root / "cases" / row["case_key"] / "total_labels.nii.gz"))
    )
    ct_data = np.asarray(ct.dataobj, dtype=np.float32)
    seg_data = np.asarray(segmentation.dataobj, dtype=np.uint8)
    if ct_data.shape != seg_data.shape:
        raise ValueError(f"{row['name']}: canonical CT/anatomy shape mismatch")
    lung = np.isin(seg_data, list(LUNG_LABELS))
    indices = [best_slice(lung, axis) for axis in range(3)]
    titles = ["Sagittal", "Coronal", "Axial"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (panel, index, title) in enumerate(zip(axes, indices, titles, strict=True)):
        ct_slice = oriented_slice(window_ct(ct_data), axis, index)
        seg_slice = oriented_slice(seg_data, axis, index)
        panel.imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
        panel.imshow(colored_overlay(seg_slice))
        panel.set_title(f"{title} index {index}", fontsize=9)
        panel.set_axis_off()
    labels_present = int(np.unique(seg_data).size - int(np.any(seg_data == 0)))
    fig.suptitle(
        f"val[{row['val_order']:03d}] {row['name']} | "
        f"{labels_present} anatomy labels | {len(row['findings'])} findings",
        fontsize=10,
    )
    handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=name.replace("_", " "))
        for name, color in GROUP_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row['val_order']:03d}_{row['case_key']}.png"
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_contact_sheets(
    figure_paths: list[Path],
    output_dir: Path,
    cases_per_sheet: int = 20,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[Path] = []
    for sheet_index, start in enumerate(range(0, len(figure_paths), cases_per_sheet)):
        selected = figure_paths[start : start + cases_per_sheet]
        columns = 2
        rows = math.ceil(len(selected) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(18, 4.3 * rows))
        panels = np.asarray(axes).reshape(-1)
        for panel, path in zip(panels, selected, strict=False):
            panel.imshow(plt.imread(path))
            panel.set_axis_off()
        for panel in panels[len(selected) :]:
            panel.set_axis_off()
        fig.suptitle(
            f"Exp010 TotalSegmentator val200 QC "
            f"cases {start}-{start + len(selected) - 1}",
            fontsize=14,
        )
        fig.tight_layout()
        output_path = output_dir / f"contact_sheet_{sheet_index:02d}.png"
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        sheets.append(output_path)
    return sheets


def render_aggregate_plots(
    label_rows: list[dict[str, Any]],
    presence: np.ndarray,
    plan: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    observed = [row for row in label_rows if row["cases_present"] > 0]
    top = sorted(observed, key=lambda row: row["cases_present"], reverse=True)[:40]
    fig, axis = plt.subplots(figsize=(10, 12))
    axis.barh(
        [row["label_name"] for row in reversed(top)],
        [row["cases_present"] for row in reversed(top)],
        color="#2878b5",
    )
    axis.set_xlabel("Validation cases with nonempty predicted label")
    axis.set_title(f"Top 40 observed TotalSegmentator labels (n={len(plan)} cases)")
    axis.grid(axis="x", alpha=0.2)
    path = output_dir / "label_case_frequency_top40.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(18, 10))
    axis.imshow(presence.T, aspect="auto", interpolation="nearest", cmap="Greys")
    axis.set_xlabel("Fixed val200 order")
    axis.set_ylabel("TotalSegmentator label ID")
    axis.set_title("Case-by-label presence (117 labels x 200 cases)")
    axis.set_yticks(np.arange(0, presence.shape[1], 5))
    axis.set_yticklabels(np.arange(1, presence.shape[1] + 1, 5))
    path = output_dir / "case_by_label_presence_heatmap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def markdown_report(
    summary: dict[str, Any],
    label_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> str:
    observed = [row for row in label_rows if row["cases_present"] > 0]
    atomic = [row for row in label_rows if row["selection"] == "retain_atomic"]
    group_count = len(RECOMMENDED_GROUP_CHANNELS)
    proposed_channels = len(atomic) + group_count
    target_count = len(target_rows)
    target_in_lung_50 = sum(
        row["whole_lung_target_fraction"] >= 0.5 for row in target_rows
    )
    target_any_lung = sum(row["whole_lung_target_fraction"] > 0 for row in target_rows)

    lines = [
        "# Exp010 TotalSegmentator Val200 Anatomy Audit",
        "",
        f"Generated: `{summary['completed_at_utc']}`",
        "",
        "## Completion",
        "",
        f"- Cases: `{summary['cases_complete']}/{summary['cases_expected']}`.",
        f"- Configured labels: `{summary['configured_labels']}`.",
        f"- Labels observed in at least one case: `{len(observed)}`.",
        f"- Median runtime per case: `{summary['runtime_seconds']['median']:.1f} s`.",
        f"- Maximum recorded TotalSegmentator GPU delta: "
        f"`{summary['gpu_memory_peak_delta_mib']['max']} MiB`.",
        "",
        "## Initial Ontology Decision",
        "",
        "Do not feed all 117 classes independently. For the first 1.5 mm",
        "AnatomyBridge cache, retain the following atomic structures:",
        "",
    ]
    lines.extend(f"- `{row['label_name']}`" for row in atomic)
    lines.extend(
        [
            "",
            "Merge repetitive landmarks into derived channels:",
            "",
        ]
    )
    lines.extend(
        f"- `{name}` from label IDs `{sorted(label_ids)}`"
        for name, label_ids in RECOMMENDED_GROUP_CHANNELS.items()
    )
    lines.extend(
        [
            "",
            f"This phase-A proposal has `{len(atomic)}` atomic channels plus",
            f"`{group_count}` derived/grouped channels, or `{proposed_channels}`",
            "channels before adding the separate open `lung_vessels` task.",
            "The later airway/vessel audit should add at most four channels:",
            "`lung_arteries`, `lung_veins`, `lung_airways`, and",
            "`lung_airways_wall`. The anticipated final range is therefore",
            f"`{proposed_channels}-{proposed_channels + 4}` channels, followed",
            "by a learned compact mask encoder rather than direct float32",
            "concatenation.",
            "",
            "Ribs and thoracic vertebrae are retained only as merged coordinate",
            "families. Separate per-rib and per-vertebra channels would spend",
            "dozens of channels on distinctions rarely requested by ReX text.",
            "",
            "## Target Overlap",
            "",
            f"- ReX findings audited: `{target_count}`.",
            f"- Findings with any target voxel inside a predicted lung lobe:",
            f"`{target_any_lung}/{target_count}`.",
            f"- Findings with at least 50% of target inside predicted lung lobes:",
            f"`{target_in_lung_50}/{target_count}`.",
            "",
            "This overlap is descriptive, not a hard-gating criterion. Pleural",
            "effusion, pleural thickening, and pneumothorax can correctly sit",
            "outside lung-lobe masks, and severe disease can degrade the public",
            "anatomy prediction. Anatomy must remain a soft residual input.",
            "",
            "## Observed Label Table",
            "",
            "| ID | Label | Cases | Fraction | Selection |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in label_rows:
        lines.append(
            f"| {row['label_id']} | `{row['label_name']}` | "
            f"{row['cases_present']} | {row['case_fraction']:.3f} | "
            f"`{row['selection']}` |"
        )
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "Run the selected thoracic ontology at 1.5 mm on the fixed first 20",
            "validation cases and compare it with these 3 mm masks for lobe",
            "survival, target overlap, boundary agreement, runtime, and memory.",
            "Only then generate train/val high-resolution anatomy caches.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(args: argparse.Namespace, plan: list[dict[str, Any]]) -> int:
    status = case_status(plan, args.cache_root)
    write_json_atomic(args.exp_dir / "reports/val200_progress.json", status)
    if status["complete"] != len(plan):
        print(json.dumps(status, indent=2, sort_keys=True))
        return 2

    class_map = query_class_map()
    label_rows, presence, runtimes = collect_label_rows(plan, args.cache_root, class_map)
    target_rows = target_overlap_rows(plan, args.cache_root, args.seg_dir, class_map)

    reports_dir = args.exp_dir / "reports"
    visuals_dir = args.exp_dir / "visualizations/totalsegmentator_total_fast_3mm_val200"
    write_csv(
        reports_dir / "label_presence.csv",
        label_rows,
        list(label_rows[0]),
    )
    write_csv(
        reports_dir / "case_runtime_memory.csv",
        runtimes,
        list(runtimes[0]),
    )
    write_csv(
        reports_dir / "target_anatomy_overlap.csv",
        target_rows,
        list(target_rows[0]),
    )

    figure_paths: list[Path] = []
    if not args.skip_visuals:
        per_case_dir = visuals_dir / "cases"
        for index, row in enumerate(plan, start=1):
            expected = per_case_dir / f"{row['val_order']:03d}_{row['case_key']}.png"
            if not expected.is_file() or args.overwrite_visuals:
                render_case_qc(row, args.cache_root, per_case_dir)
            figure_paths.append(expected)
            if index % 20 == 0:
                print(f"Rendered/verified {index}/{len(plan)} case figures", flush=True)
        render_contact_sheets(figure_paths, visuals_dir / "contact_sheets")
        render_aggregate_plots(label_rows, presence, plan, visuals_dir)

    runtime_values = np.asarray([row["runtime_seconds"] for row in runtimes])
    gpu_deltas = np.asarray([row["gpu_memory_peak_delta_mib"] for row in runtimes])
    summary = {
        "completed_at_utc": utc_now(),
        "cases_expected": len(plan),
        "cases_complete": status["complete"],
        "configured_labels": len(class_map),
        "observed_labels": int(sum(row["cases_present"] > 0 for row in label_rows)),
        "runtime_seconds": {
            "total": float(runtime_values.sum()),
            "mean": float(runtime_values.mean()),
            "median": float(np.median(runtime_values)),
            "min": float(runtime_values.min()),
            "max": float(runtime_values.max()),
        },
        "gpu_memory_peak_delta_mib": {
            "median": int(np.median(gpu_deltas)),
            "max": int(gpu_deltas.max()),
        },
        "findings": len(target_rows),
        "val_plan_sha256": sha256_file(args.plan),
        "label_presence_csv": str(reports_dir / "label_presence.csv"),
        "target_overlap_csv": str(reports_dir / "target_anatomy_overlap.csv"),
        "visualizations": str(visuals_dir),
    }
    report_text = markdown_report(summary, label_rows, target_rows)
    report_path = reports_dir / "totalsegmentator_val200_anatomy_audit.md"
    report_path.write_text(report_text)
    write_json_atomic(reports_dir / "totalsegmentator_val200_anatomy_audit.json", summary)

    cache_manifest = {
        **summary,
        "cache_root": str(args.cache_root),
        "task": "total",
        "fast": True,
        "model_spacing_mm": 3.0,
        "save_lowres": False,
        "output_geometry": "source CT shape and affine",
        "class_map": {str(key): value for key, value in class_map.items()},
        "per_case_contract": [
            "total_labels.nii.gz",
            "metadata.json",
            "statistics.json",
            "run_report.json",
            "inference.log",
            ".complete",
        ],
    }
    write_json_atomic(args.cache_root / "manifest.json", cache_manifest)
    write_json_atomic(
        args.cache_root / ".complete",
        {
            "completed_at_utc": summary["completed_at_utc"],
            "manifest_sha256": sha256_file(args.cache_root / "manifest.json"),
            "cases": len(plan),
        },
    )
    print(report_text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--seg-dir", type=Path, default=DEFAULT_SEG_DIR)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--skip-visuals", action="store_true")
    parser.add_argument("--overwrite-visuals", action="store_true")
    args = parser.parse_args()

    plan = read_jsonl(args.plan)
    while True:
        result = summarize(args, plan)
        if result != 2 or not args.watch:
            return result
        print(f"Waiting {args.interval}s for val200 anatomy workers...", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
