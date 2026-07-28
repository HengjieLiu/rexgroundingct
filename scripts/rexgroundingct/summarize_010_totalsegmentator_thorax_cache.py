#!/usr/bin/env python3
"""Summarize the exp010 informative TotalSegmentator thorax val200 cache."""

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


DEFAULT_PLAN = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "010_voxtell_public_anatomy_prior_fusion/config/"
    "val200_totalsegmentator_case_plan.jsonl"
)
DEFAULT_CACHE_ROOT = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/"
    "totalsegmentator_thorax_v1"
)
DEFAULT_PHASE_A_CACHE_ROOT = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/anatomy/"
    "totalsegmentator_total_fast_3mm_v2_16_0"
)
DEFAULT_EXP_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "010_voxtell_public_anatomy_prior_fusion"
)
DEFAULT_SEG_DIR = Path("/data/hengjie/datasets/rexgroundingct/segmentations")

TOTAL_ROI_NAMES = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "trachea",
    "esophagus",
    "heart",
    "aorta",
    "pulmonary_vein",
    "brachiocephalic_trunk",
    "subclavian_artery_right",
    "subclavian_artery_left",
    "common_carotid_artery_right",
    "common_carotid_artery_left",
    "brachiocephalic_vein_left",
    "brachiocephalic_vein_right",
    "atrial_appendage_left",
    "superior_vena_cava",
    "inferior_vena_cava",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
    "vertebrae_C7",
    "spinal_cord",
    "rib_left_1",
    "rib_left_2",
    "rib_left_3",
    "rib_left_4",
    "rib_left_5",
    "rib_left_6",
    "rib_left_7",
    "rib_left_8",
    "rib_left_9",
    "rib_left_10",
    "rib_left_11",
    "rib_left_12",
    "rib_right_1",
    "rib_right_2",
    "rib_right_3",
    "rib_right_4",
    "rib_right_5",
    "rib_right_6",
    "rib_right_7",
    "rib_right_8",
    "rib_right_9",
    "rib_right_10",
    "rib_right_11",
    "rib_right_12",
    "sternum",
    "costal_cartilages",
    "clavicula_left",
    "clavicula_right",
    "scapula_left",
    "scapula_right",
    "liver",
    "spleen",
    "stomach",
    "adrenal_gland_right",
    "adrenal_gland_left",
]

TASK_SPECS = {
    "total_roi": {
        "task": "total",
        "output": "total_roi_labels.nii.gz",
        "metadata": "total_roi_metadata.json",
        "complete": ".complete_total_roi",
        "failure": "total_roi_failure.json",
        "lock": ".case_total_roi.lock",
    },
    "lung_vessels": {
        "task": "lung_vessels",
        "output": "lung_vessels_labels.nii.gz",
        "metadata": "lung_vessels_metadata.json",
        "complete": ".complete_lung_vessels",
        "failure": "lung_vessels_failure.json",
        "lock": ".case_lung_vessels.lock",
    },
    "trunk_cavities": {
        "task": "trunk_cavities",
        "output": "trunk_cavities_labels.nii.gz",
        "metadata": "trunk_cavities_metadata.json",
        "complete": ".complete_trunk_cavities",
        "failure": "trunk_cavities_failure.json",
        "lock": ".case_trunk_cavities.lock",
    },
    "body": {
        "task": "body",
        "output": "body_labels.nii.gz",
        "metadata": "body_metadata.json",
        "complete": ".complete_body",
        "failure": "body_failure.json",
        "lock": ".case_body.lock",
    },
}

TOTAL_GROUPS = {
    "total_whole_lung": {10, 11, 12, 13, 14},
    "total_left_lung": {10, 11},
    "total_right_lung": {12, 13, 14},
    "total_trachea": {16},
    "total_cardiomediastinal": set(range(51, 64)),
    "total_thoracic_spine": set(range(32, 45)),
    "total_left_ribs": set(range(92, 104)),
    "total_right_ribs": set(range(104, 116)),
    "total_anterior_chest_wall": {116, 117},
    "total_inferior_anchors": {1, 5, 6, 8, 9},
}
LUNG_VESSEL_LABELS = {
    "lung_airways": 1,
    "lung_airways_wall": 2,
    "lung_arteries": 3,
    "lung_veins": 4,
}
TRUNK_CAVITY_LABELS = {
    "abdominal_cavity": 1,
    "thoracic_cavity": 2,
    "pericardium": 3,
    "mediastinum": 4,
}
BODY_LABELS = {"body_trunc": 1, "body_extremities": 2}


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def query_class_map(task: str) -> dict[int, str]:
    output = subprocess.check_output(
        ["totalseg_info", "--classes", "-ta", task, "--json"],
        text=True,
    )
    return {int(key): value for key, value in json.loads(output).items()}


def task_class_map(task_key: str) -> dict[int, str]:
    task = TASK_SPECS[task_key]["task"]
    class_map = query_class_map(task)
    if task_key != "total_roi":
        return class_map
    retained = set(TOTAL_ROI_NAMES)
    return {label_id: name for label_id, name in class_map.items() if name in retained}


def task_status(plan: list[dict[str, Any]], cache_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checked_at_utc": utc_now(),
        "cases_expected": len(plan),
        "tasks": {},
        "all_complete": False,
    }
    all_complete = True
    for task_key, spec in TASK_SPECS.items():
        complete: list[str] = []
        failed: list[str] = []
        missing: list[str] = []
        locked: list[str] = []
        for row in plan:
            case_dir = cache_root / "cases" / row["case_key"]
            if (case_dir / spec["complete"]).is_file():
                complete.append(row["name"])
            elif (case_dir / spec["failure"]).is_file():
                failed.append(row["name"])
            else:
                missing.append(row["name"])
            if (case_dir / spec["lock"]).is_dir():
                locked.append(row["name"])
        payload["tasks"][task_key] = {
            "complete": len(complete),
            "failed": len(failed),
            "missing": len(missing),
            "locked": len(locked),
            "first_missing_cases": missing[:20],
            "failed_cases": failed,
            "locked_cases": locked,
        }
        all_complete = all_complete and len(complete) == len(plan) and not failed
    payload["all_complete"] = all_complete
    return payload


def collect_label_and_runtime_rows(
    plan: list[dict[str, Any]],
    cache_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for task_key, spec in TASK_SPECS.items():
        class_map = task_class_map(task_key)
        present_counts = {label_id: 0 for label_id in class_map}
        total_voxels = {label_id: 0 for label_id in class_map}
        present_volumes = {label_id: [] for label_id in class_map}
        for row in plan:
            metadata_path = cache_root / "cases" / row["case_key"] / spec["metadata"]
            metadata = json.loads(metadata_path.read_text())
            geometry = metadata["geometry"]
            observed = {
                int(label): int(count)
                for label, count in geometry["class_voxels"].items()
            }
            volume_by_label = {
                int(label): float(volume)
                for label, volume in geometry["class_volume_mm3"].items()
            }
            for label_id in class_map:
                voxels = observed.get(label_id, 0)
                if voxels > 0:
                    present_counts[label_id] += 1
                    total_voxels[label_id] += voxels
                    present_volumes[label_id].append(volume_by_label.get(label_id, 0.0))
            runtime_rows.append(
                {
                    "task_key": task_key,
                    "task": spec["task"],
                    "val_order": row["val_order"],
                    "case": row["name"],
                    "runtime_seconds": float(metadata["runtime_seconds"]),
                    "gpu_memory_baseline_used_mib": int(
                        metadata["gpu_memory_baseline_used_mib"]
                    ),
                    "gpu_memory_peak_total_used_mib": int(
                        metadata["gpu_memory_peak_total_used_mib"]
                    ),
                    "gpu_memory_peak_delta_mib": int(
                        metadata["gpu_memory_peak_delta_mib"]
                    ),
                    "observed_class_count": int(geometry["observed_class_count"]),
                    "output_sha256": metadata["output_sha256"],
                }
            )
        for label_id, label_name in class_map.items():
            volumes = np.asarray(present_volumes[label_id], dtype=np.float64)
            label_rows.append(
                {
                    "task_key": task_key,
                    "task": spec["task"],
                    "label_id": label_id,
                    "label_name": label_name,
                    "cases_present": present_counts[label_id],
                    "case_fraction": present_counts[label_id] / len(plan),
                    "total_voxels": total_voxels[label_id],
                    "median_present_volume_mm3": (
                        float(np.median(volumes)) if volumes.size else 0.0
                    ),
                    "min_present_volume_mm3": (
                        float(volumes.min()) if volumes.size else 0.0
                    ),
                    "max_present_volume_mm3": (
                        float(volumes.max()) if volumes.size else 0.0
                    ),
                }
            )
    return label_rows, runtime_rows


def load_task_array(cache_root: Path, row: dict[str, Any], task_key: str) -> np.ndarray:
    path = cache_root / "cases" / row["case_key"] / TASK_SPECS[task_key]["output"]
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.uint8)


def target_overlap_rows(
    plan: list[dict[str, Any]],
    cache_root: Path,
    phase_a_cache_root: Path,
    seg_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_map = task_class_map("total_roi")
    phase_a_available = phase_a_cache_root.is_dir()
    for case in plan:
        total = load_task_array(cache_root, case, "total_roi")
        lung_vessels = load_task_array(cache_root, case, "lung_vessels")
        trunk = load_task_array(cache_root, case, "trunk_cavities")
        body = load_task_array(cache_root, case, "body")
        phase_a = None
        phase_a_path = phase_a_cache_root / "cases" / case["case_key"] / "total_labels.nii.gz"
        if phase_a_available and phase_a_path.is_file():
            phase_a = np.asarray(nib.load(str(phase_a_path)).dataobj, dtype=np.uint8)
        target_img = nib.load(str(seg_dir / case["name"]))
        targets = np.asarray(target_img.dataobj, dtype=np.uint8)
        if targets.ndim != 4 or tuple(targets.shape[1:]) != tuple(total.shape):
            raise ValueError(
                f"{case['name']}: target/anatomy shape mismatch "
                f"{targets.shape} versus {total.shape}"
            )
        for task_key, array in {
            "lung_vessels": lung_vessels,
            "trunk_cavities": trunk,
            "body": body,
        }.items():
            if array.shape != total.shape:
                raise ValueError(
                    f"{case['name']}: {task_key} shape {array.shape} "
                    f"does not match total ROI shape {total.shape}"
                )
        finding_ids = sorted(case["findings"], key=lambda value: int(value))
        for channel, finding_id in enumerate(finding_ids):
            target = targets[channel] > 0
            target_voxels = int(target.sum())
            if target_voxels == 0:
                raise ValueError(f"{case['name']} target {finding_id} is empty")
            total_at_target = total[target]
            labels_at_target, counts = np.unique(total_at_target, return_counts=True)
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
                "total_best_label_id": best_label_id,
                "total_best_label": total_map.get(best_label_id, "background"),
                "total_best_label_target_fraction": best_label_voxels / target_voxels,
                "total_background_target_fraction": float((total_at_target == 0).mean()),
            }
            for group_name, label_ids in TOTAL_GROUPS.items():
                row[f"{group_name}_target_fraction"] = float(
                    np.isin(total_at_target, list(label_ids)).mean()
                )
            lung_vessels_at_target = lung_vessels[target]
            for name, label_id in LUNG_VESSEL_LABELS.items():
                row[f"lung_vessels_{name}_target_fraction"] = float(
                    (lung_vessels_at_target == label_id).mean()
                )
            row["lung_vessels_any_target_fraction"] = float(
                (lung_vessels_at_target > 0).mean()
            )
            trunk_at_target = trunk[target]
            for name, label_id in TRUNK_CAVITY_LABELS.items():
                row[f"trunk_{name}_target_fraction"] = float(
                    (trunk_at_target == label_id).mean()
                )
            body_at_target = body[target]
            for name, label_id in BODY_LABELS.items():
                row[f"body_{name}_target_fraction"] = float(
                    (body_at_target == label_id).mean()
                )
            if phase_a is not None:
                phase_a_at_target = phase_a[target]
                row["phase_a_fast_whole_lung_target_fraction"] = float(
                    np.isin(phase_a_at_target, list(TOTAL_GROUPS["total_whole_lung"])).mean()
                )
                row["phase_a_fast_background_target_fraction"] = float(
                    (phase_a_at_target == 0).mean()
                )
            rows.append(row)
    return rows


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    count_a = int(mask_a.sum())
    count_b = int(mask_b.sum())
    if count_a == 0 and count_b == 0:
        return 1.0
    if count_a == 0 or count_b == 0:
        return 0.0
    return 2.0 * float(np.logical_and(mask_a, mask_b).sum()) / float(count_a + count_b)


def comparison_rows(
    plan: list[dict[str, Any]],
    cache_root: Path,
    phase_a_cache_root: Path,
) -> list[dict[str, Any]]:
    if not phase_a_cache_root.is_dir():
        return []
    class_map = task_class_map("total_roi")
    rows: list[dict[str, Any]] = []
    for case in plan:
        phase_a_path = phase_a_cache_root / "cases" / case["case_key"] / "total_labels.nii.gz"
        if not phase_a_path.is_file():
            continue
        total = load_task_array(cache_root, case, "total_roi")
        phase_a = np.asarray(nib.load(str(phase_a_path)).dataobj, dtype=np.uint8)
        if total.shape != phase_a.shape:
            raise ValueError(
                f"{case['name']}: 1.5mm ROI/3mm fast shape mismatch "
                f"{total.shape} versus {phase_a.shape}"
            )
        for label_id, label_name in class_map.items():
            rows.append(
                {
                    "val_order": case["val_order"],
                    "case": case["name"],
                    "label_id": label_id,
                    "label_name": label_name,
                    "roi_1p5_voxels": int((total == label_id).sum()),
                    "fast_3mm_voxels": int((phase_a == label_id).sum()),
                    "dice": dice(total == label_id, phase_a == label_id),
                }
            )
    return rows


def window_ct(data: np.ndarray, center: float = -600.0, width: float = 1500.0) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    return np.clip((data.astype(np.float32) - low) / (high - low), 0.0, 1.0)


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


def overlay(mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.35) -> np.ndarray:
    out = np.zeros((*mask.shape, 4), dtype=np.float32)
    out[mask, :3] = color
    out[mask, 3] = alpha
    return out


def render_case_qc(row: dict[str, Any], cache_root: Path, output_dir: Path) -> Path:
    ct = nib.as_closest_canonical(nib.load(row["ct_path"]))
    total = nib.as_closest_canonical(
        nib.load(str(cache_root / "cases" / row["case_key"] / "total_roi_labels.nii.gz"))
    )
    lung_vessels = nib.as_closest_canonical(
        nib.load(str(cache_root / "cases" / row["case_key"] / "lung_vessels_labels.nii.gz"))
    )
    trunk = nib.as_closest_canonical(
        nib.load(str(cache_root / "cases" / row["case_key"] / "trunk_cavities_labels.nii.gz"))
    )
    body = nib.as_closest_canonical(
        nib.load(str(cache_root / "cases" / row["case_key"] / "body_labels.nii.gz"))
    )
    ct_data = np.asarray(ct.dataobj, dtype=np.float32)
    total_data = np.asarray(total.dataobj, dtype=np.uint8)
    lung_vessel_data = np.asarray(lung_vessels.dataobj, dtype=np.uint8)
    trunk_data = np.asarray(trunk.dataobj, dtype=np.uint8)
    body_data = np.asarray(body.dataobj, dtype=np.uint8)
    lung = np.isin(total_data, list(TOTAL_GROUPS["total_whole_lung"]))
    axial_index = best_slice(lung, 2)
    ct_slice = oriented_slice(window_ct(ct_data), 2, axial_index)

    panels = [
        (
            "lobes and thoracic bones",
            [
                overlay(
                    np.isin(oriented_slice(total_data, 2, axial_index), list(TOTAL_GROUPS["total_whole_lung"])),
                    (0.1, 0.7, 0.95),
                ),
                overlay(
                    np.isin(oriented_slice(total_data, 2, axial_index), list(TOTAL_GROUPS["total_thoracic_spine"] | TOTAL_GROUPS["total_left_ribs"] | TOTAL_GROUPS["total_right_ribs"])),
                    (0.8, 0.85, 0.25),
                    0.25,
                ),
            ],
        ),
        (
            "airway and vessels",
            [
                overlay(oriented_slice(lung_vessel_data, 2, axial_index) == 1, (1.0, 0.85, 0.05), 0.45),
                overlay(oriented_slice(lung_vessel_data, 2, axial_index) == 2, (1.0, 0.55, 0.05), 0.45),
                overlay(oriented_slice(lung_vessel_data, 2, axial_index) == 3, (0.95, 0.2, 0.2), 0.35),
                overlay(oriented_slice(lung_vessel_data, 2, axial_index) == 4, (0.25, 0.35, 1.0), 0.35),
            ],
        ),
        (
            "body and cavities",
            [
                overlay(oriented_slice(trunk_data, 2, axial_index) == 2, (0.25, 0.9, 0.35), 0.25),
                overlay(oriented_slice(trunk_data, 2, axial_index) == 4, (0.95, 0.35, 0.75), 0.35),
                overlay(oriented_slice(body_data, 2, axial_index) == 1, (0.75, 0.75, 0.75), 0.15),
            ],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (title, overlays) in zip(axes, panels, strict=True):
        axis.imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
        for layer in overlays:
            axis.imshow(layer)
        axis.set_title(title, fontsize=9)
        axis.set_axis_off()
    fig.suptitle(
        f"val[{row['val_order']:03d}] {row['name']} | axial index {axial_index}",
        fontsize=10,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row['val_order']:03d}_{row['case_key']}.png"
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_contact_sheet(figure_paths: list[Path], output_path: Path) -> None:
    columns = 2
    rows = math.ceil(len(figure_paths) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(18, 4.3 * rows))
    panels = np.asarray(axes).reshape(-1)
    for panel, path in zip(panels, figure_paths, strict=False):
        panel.imshow(plt.imread(path))
        panel.set_axis_off()
    for panel in panels[len(figure_paths) :]:
        panel.set_axis_off()
    fig.suptitle("Exp010 TotalSegmentator thorax_v1 representative val200 QC", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def scalar_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "total": float(array.sum()),
    }


def markdown_report(
    summary: dict[str, Any],
    label_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
) -> str:
    lines = [
        "# Exp010 TotalSegmentator Thorax Val200 Cache",
        "",
        f"Generated: `{summary['completed_at_utc']}`",
        "",
        "## Completion",
        "",
    ]
    for task_key in TASK_SPECS:
        task_summary = summary["tasks"][task_key]
        lines.append(
            f"- `{task_key}`: `{task_summary['complete']}/{summary['cases_expected']}` "
            f"cases, median runtime `{task_summary['runtime_seconds']['median']:.1f}s`, "
            f"max GPU delta `{task_summary['gpu_memory_peak_delta_mib']['max']} MiB`."
        )
    lines.extend(
        [
            "",
            "## Label Presence",
            "",
            "| Task | Labels | Observed Labels | Always Present |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for task_key in TASK_SPECS:
        rows = [row for row in label_rows if row["task_key"] == task_key]
        observed = sum(row["cases_present"] > 0 for row in rows)
        always = sum(row["cases_present"] == summary["cases_expected"] for row in rows)
        lines.append(f"| `{task_key}` | {len(rows)} | {observed} | {always} |")

    target_count = len(target_rows)
    whole_lung_50 = sum(
        row["total_whole_lung_target_fraction"] >= 0.5 for row in target_rows
    )
    whole_lung_95 = sum(
        row["total_whole_lung_target_fraction"] >= 0.95 for row in target_rows
    )
    airway_any = sum(row["lung_vessels_any_target_fraction"] > 0 for row in target_rows)
    mediastinum_any = sum(row["trunk_mediastinum_target_fraction"] > 0 for row in target_rows)
    thoracic_cavity_50 = sum(
        row["trunk_thoracic_cavity_target_fraction"] >= 0.5 for row in target_rows
    )
    lines.extend(
        [
            "",
            "## Target Overlap",
            "",
            f"- Findings audited: `{target_count}`.",
            f"- >=50% inside 1.5 mm ROI lung lobes: `{whole_lung_50}/{target_count}`.",
            f"- >=95% inside 1.5 mm ROI lung lobes: `{whole_lung_95}/{target_count}`.",
            f"- Any overlap with airway/vessel task: `{airway_any}/{target_count}`.",
            f"- Any overlap with mediastinum: `{mediastinum_any}/{target_count}`.",
            f"- >=50% inside thoracic cavity: `{thoracic_cavity_50}/{target_count}`.",
            "",
            "Anatomy is still a soft prior. These numbers are containment diagnostics,",
            "not a proposal to clip challenge predictions.",
        ]
    )
    if comparison:
        lung_labels = set(TOTAL_GROUPS["total_whole_lung"])
        lung_dice = [
            row["dice"] for row in comparison if int(row["label_id"]) in lung_labels
        ]
        all_dice = [row["dice"] for row in comparison]
        lines.extend(
            [
                "",
                "## 3 mm Fast Comparison",
                "",
                f"- Mean Dice over ROI labels versus phase-A 3 mm cache: `{np.mean(all_dice):.4f}`.",
                f"- Mean Dice over lung lobes versus phase-A 3 mm cache: `{np.mean(lung_dice):.4f}`.",
                "",
                "This measures anatomy-cache agreement only. It is not a lesion Dice metric.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Cache root: `{summary['cache_root']}`",
            f"- Label CSV: `{summary['label_presence_csv']}`",
            f"- Target-overlap CSV: `{summary['target_overlap_csv']}`",
            f"- Runtime CSV: `{summary['runtime_csv']}`",
            f"- Visual QC: `{summary['visualizations']}`",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(args: argparse.Namespace, plan: list[dict[str, Any]]) -> int:
    status = task_status(plan, args.cache_root)
    reports_dir = args.exp_dir / "reports"
    visuals_dir = args.exp_dir / "visualizations/totalsegmentator_thorax_v1_val200"
    write_json_atomic(reports_dir / "thorax_val200_progress.json", status)
    if not status["all_complete"]:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 2

    label_rows, runtime_rows = collect_label_and_runtime_rows(plan, args.cache_root)
    target_rows = target_overlap_rows(
        plan,
        args.cache_root,
        args.phase_a_cache_root,
        args.seg_dir,
    )
    comparison = comparison_rows(plan, args.cache_root, args.phase_a_cache_root)

    write_csv(reports_dir / "thorax_label_presence.csv", label_rows, list(label_rows[0]))
    write_csv(reports_dir / "thorax_case_runtime_memory.csv", runtime_rows, list(runtime_rows[0]))
    write_csv(
        reports_dir / "thorax_target_anatomy_overlap.csv",
        target_rows,
        list(target_rows[0]),
    )
    if comparison:
        write_csv(
            reports_dir / "thorax_total_roi_vs_fast_label_dice.csv",
            comparison,
            list(comparison[0]),
        )

    if not args.skip_visuals:
        case_figures: list[Path] = []
        per_case_dir = visuals_dir / "cases"
        for row in plan[: args.visual_cases]:
            expected = per_case_dir / f"{row['val_order']:03d}_{row['case_key']}.png"
            if not expected.is_file() or args.overwrite_visuals:
                render_case_qc(row, args.cache_root, per_case_dir)
            case_figures.append(expected)
        render_contact_sheet(case_figures, visuals_dir / "contact_sheet_first20.png")

    task_summaries: dict[str, Any] = {}
    for task_key in TASK_SPECS:
        task_runtime = [
            row for row in runtime_rows if row["task_key"] == task_key
        ]
        task_summaries[task_key] = {
            **status["tasks"][task_key],
            "runtime_seconds": scalar_stats(
                [row["runtime_seconds"] for row in task_runtime]
            ),
            "gpu_memory_peak_delta_mib": {
                key: int(value)
                for key, value in scalar_stats(
                    [row["gpu_memory_peak_delta_mib"] for row in task_runtime]
                ).items()
            },
        }
    summary = {
        "completed_at_utc": utc_now(),
        "cache_root": str(args.cache_root),
        "cases_expected": len(plan),
        "tasks": task_summaries,
        "task_specs": TASK_SPECS,
        "total_roi_subset": TOTAL_ROI_NAMES,
        "val_plan_sha256": sha256_file(args.plan),
        "phase_a_cache_root": str(args.phase_a_cache_root),
        "label_presence_csv": str(reports_dir / "thorax_label_presence.csv"),
        "target_overlap_csv": str(reports_dir / "thorax_target_anatomy_overlap.csv"),
        "runtime_csv": str(reports_dir / "thorax_case_runtime_memory.csv"),
        "comparison_csv": (
            str(reports_dir / "thorax_total_roi_vs_fast_label_dice.csv")
            if comparison
            else None
        ),
        "visualizations": str(visuals_dir),
    }
    report_text = markdown_report(summary, label_rows, target_rows, comparison)
    (reports_dir / "totalsegmentator_thorax_val200_cache.md").write_text(report_text)
    write_json_atomic(reports_dir / "totalsegmentator_thorax_val200_cache.json", summary)
    write_json_atomic(args.cache_root / "manifest.json", summary)
    write_json_atomic(
        args.cache_root / ".complete",
        {
            "completed_at_utc": summary["completed_at_utc"],
            "cases": len(plan),
            "tasks": list(TASK_SPECS),
            "manifest_sha256": sha256_file(args.cache_root / "manifest.json"),
        },
    )
    print(report_text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--phase-a-cache-root", type=Path, default=DEFAULT_PHASE_A_CACHE_ROOT)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--seg-dir", type=Path, default=DEFAULT_SEG_DIR)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--skip-visuals", action="store_true")
    parser.add_argument("--overwrite-visuals", action="store_true")
    parser.add_argument("--visual-cases", type=int, default=20)
    args = parser.parse_args()

    plan = read_jsonl(args.plan)
    while True:
        result = summarize(args, plan)
        if result != 2 or not args.watch:
            return result
        print(f"Waiting {args.interval}s for thorax val200 workers...", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
