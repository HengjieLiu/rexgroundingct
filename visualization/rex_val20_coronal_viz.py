#!/usr/bin/env python3
"""Coronal validation-set visualization for ReXGroundingCT predictions."""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAL_JSON = REPO_ROOT / "configs" / "evaluation" / "rexgroundingct_val200_seed20260723.json"
DEFAULT_METADATA_JSON = Path("/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json")
DEFAULT_SEG_DIR = Path("/data/hengjie/datasets/rexgroundingct/segmentations")
DEFAULT_CT_ROOT = Path("/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct")
DEFAULT_PRED_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/"
    "v123_opt_poscrop_emptyloss/full_100ep/eval_epoch100_val200/predictions"
)
DEFAULT_PRETRAINED_PRED_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "001_voxtell_v1_1_miccai200_val_eval/predictions"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "003_voxtell_rex_ft_rescue_ablation/runs/exp003_full_20260723T075256Z/"
    "v123_opt_poscrop_emptyloss/full_100ep/visualizations/"
    "2026-07-23_val20_coronal_gt_pred_v123_epoch100"
)
DEFAULT_VAL200_CATEGORY_OUTPUT_DIR = Path(
    "/data/hengjie/datasets/rexgroundingct/visualizations/"
    "2026-07-24_val200_category_gt_v123_pretrained_coronal"
)

GLOBAL_HIT_THRESHOLD = 0.1
LUNG_WINDOW_CENTER = -600.0
LUNG_WINDOW_WIDTH = 1500.0

CATEGORY_NAMES = {
    "1a": "Bronchial wall thickening",
    "1b": "Bronchiectasis",
    "1c": "Emphysema (including Centrilobular, Paraseptal, Bullous)",
    "1d": "Septal thickening (including Interlobular, Reticulation)",
    "1e": "Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic)",
    "1f": "Other",
    "2a": "Linear (including subsegmental atelectasis, scarring, fibrosis)",
    "2b": "Atelectasis, consolidation",
    "2c": "Groundglass opacity",
    "2d": "Pulmonary nodules/masses",
    "2e": "Pleural effusion or thickening",
    "2f": "Honeycombing",
    "2g": "Pneumothorax",
    "2h": "Other",
}


@dataclass(frozen=True)
class PredictionSpec:
    key: str
    label: str
    column_title: str
    pred_dir: Path
    contour_color: str


@dataclass
class VisualizationResult:
    output_dir: Path
    summary_df: pd.DataFrame
    component_df: pd.DataFrame
    figure_paths: list[Path]
    manifest_path: Path


@dataclass
class CaseArrays:
    ct_ras: np.ndarray
    gt_ras: np.ndarray
    predictions_ras: dict[str, np.ndarray]
    spacing_ras: tuple[float, float, float]
    voxel_volume_cm3: float
    ct_axcodes: tuple[str, str, str]
    orientation_transform: list[list[float]]


def default_prediction_specs(pred_dir: Path = DEFAULT_PRED_DIR) -> list[PredictionSpec]:
    return [
        PredictionSpec(
            key="v123",
            label="v123",
            column_title="v123 epoch-100 pred contour",
            pred_dir=pred_dir,
            contour_color="#ffcc00",
        )
    ]


def val200_category_prediction_specs(
    v123_pred_dir: Path = DEFAULT_PRED_DIR,
    pretrained_pred_dir: Path = DEFAULT_PRETRAINED_PRED_DIR,
) -> list[PredictionSpec]:
    return [
        PredictionSpec(
            key="v123",
            label="v123",
            column_title="v123 epoch-100 pred contour",
            pred_dir=v123_pred_dir,
            contour_color="#ffcc00",
        ),
        PredictionSpec(
            key="pretrained",
            label="pretrained",
            column_title="pretrained VoxTell v1.1 pred contour",
            pred_dir=pretrained_pred_dir,
            contour_color="#00b7ff",
        ),
    ]


def ct_rate_path(filename: str, ct_root: Path = DEFAULT_CT_ROOT) -> Path:
    if not filename.endswith(".nii.gz"):
        raise ValueError(f"Expected .nii.gz filename, got {filename!r}")
    stem = filename[: -len(".nii.gz")]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected CT-RATE filename format: {filename!r}")
    split = parts[0]
    patient = "_".join(parts[:2])
    scan = "_".join(parts[:3])
    return ct_root / "dataset" / f"{split}_fixed" / patient / scan / filename


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_reshuffled_entries(
    val_json: Path = DEFAULT_VAL_JSON,
    limit: int | None = 20,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    data = load_json(val_json)
    entries = data.get("test")
    if not isinstance(entries, list):
        raise ValueError(f"{val_json} must contain a list under key 'test'")
    if start_index < 0 or start_index >= len(entries):
        raise ValueError(f"start_index must be in [0, {len(entries) - 1}], got {start_index}")
    end_index = None if limit is None else start_index + limit
    selected = entries[start_index:end_index]
    limited = []
    for order, entry in enumerate(selected, start=start_index):
        copied = dict(entry)
        copied["reshuffled_val_index"] = order
        limited.append(copied)
    return limited


def load_metadata_index(metadata_json: Path = DEFAULT_METADATA_JSON) -> dict[str, dict[str, Any]]:
    data = load_json(metadata_json)
    indexed: dict[str, dict[str, Any]] = {}
    for split, entries in data.items():
        if not isinstance(entries, list):
            continue
        for index, entry in enumerate(entries):
            copied = dict(entry)
            copied["_split"] = split
            copied["_index"] = index
            indexed[copied["name"]] = copied
    return indexed


def sorted_finding_ids(entry: dict[str, Any]) -> list[str]:
    return sorted(entry.get("findings", {}), key=lambda value: int(value))


def category_label(code: str) -> str:
    if not code:
        return "unknown-Unknown category"
    return f"{code}-{CATEGORY_NAMES.get(code, 'Unknown category')}"


def dice_score(gt_mask: np.ndarray, pred_mask: np.ndarray, eps: float = 1e-6) -> float:
    gt = gt_mask > 0
    pred = pred_mask > 0
    denom = int(gt.sum()) + int(pred.sum())
    if denom == 0:
        return 1.0
    intersection = int((gt & pred).sum())
    return float((2.0 * intersection + eps) / (denom + eps))


def window_ct(ct_slice: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(ct_slice.astype(np.float32, copy=False), low, high)
    return (clipped - low) / (high - low)


def display_coronal_slice(volume_ras: np.ndarray, y_index: int) -> np.ndarray:
    """Return coronal display slice with left=patient right and top=superior.

    Input arrays must already be in RAS voxel orientation. RAS axis 0 increases
    toward patient right and axis 2 increases toward superior. Display columns
    therefore descend axis 0, and display rows descend axis 2.
    """
    return volume_ras[::-1, y_index, ::-1].T


def display_coronal_xz_slice(xz_slice_ras: np.ndarray) -> np.ndarray:
    """Display an RAS-space X/Z slice with left=patient right and top=superior."""
    return xz_slice_ras[::-1, ::-1].T


def orientation_as_list(ornt: np.ndarray) -> list[list[float]]:
    return [[float(axis), float(direction)] for axis, direction in ornt.tolist()]


def reorient_channel_first_masks(masks: np.ndarray, to_ras: np.ndarray) -> np.ndarray:
    return np.stack([apply_orientation(masks[index], to_ras) for index in range(masks.shape[0])], axis=0)


def load_prediction_ras(
    name: str,
    spec: PredictionSpec,
    gt_shape: tuple[int, ...],
    to_ras: np.ndarray,
) -> np.ndarray:
    pred_path = spec.pred_dir / name
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing {spec.label} prediction for {name}: {pred_path}")
    pred_img = nib.load(str(pred_path))
    pred = np.asarray(pred_img.dataobj, dtype=np.uint8)
    if tuple(pred.shape) != gt_shape:
        raise ValueError(f"{name}: {spec.label} shape mismatch: {pred.shape} vs GT {gt_shape}")
    return np.ascontiguousarray(reorient_channel_first_masks(pred, to_ras))


def load_case_arrays(
    name: str,
    seg_dir: Path = DEFAULT_SEG_DIR,
    pred_dir: Path = DEFAULT_PRED_DIR,
    ct_root: Path = DEFAULT_CT_ROOT,
    prediction_specs: list[PredictionSpec] | None = None,
    load_ct_data: bool = True,
) -> CaseArrays:
    prediction_specs = prediction_specs or default_prediction_specs(pred_dir)
    ct_path = ct_rate_path(name, ct_root)
    gt_path = seg_dir / name
    missing = [str(path) for path in (ct_path, gt_path) if not path.is_file()]
    missing.extend(str(spec.pred_dir / name) for spec in prediction_specs if not (spec.pred_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing input file(s) for {name}: {missing}")

    ct_img = nib.load(str(ct_path))
    gt_img = nib.load(str(gt_path))
    gt = np.asarray(gt_img.dataobj, dtype=np.uint8)

    if gt.ndim != 4:
        raise ValueError(f"{name}: expected 4D GT array, got {gt.shape}")
    if tuple(gt.shape[1:]) != tuple(ct_img.shape):
        raise ValueError(f"{name}: mask spatial shape {gt.shape[1:]} does not match CT shape {ct_img.shape}")

    ct_ornt = io_orientation(ct_img.affine)
    ras_ornt = axcodes2ornt(("R", "A", "S"))
    to_ras = ornt_transform(ct_ornt, ras_ornt)
    if load_ct_data:
        ct = np.asarray(ct_img.dataobj, dtype=np.float32)
        ct_ras = apply_orientation(ct, to_ras)
    else:
        ct_ras = np.empty((0, 0, 0), dtype=np.float32)
    gt_ras = reorient_channel_first_masks(gt, to_ras)
    predictions_ras = {
        spec.key: load_prediction_ras(name, spec, tuple(gt.shape), to_ras) for spec in prediction_specs
    }

    zooms = tuple(float(value) for value in ct_img.header.get_zooms()[:3])
    spacing_ras = tuple(zooms[int(axis)] for axis, _direction in to_ras)
    voxel_volume_cm3 = float(np.prod(zooms) / 1000.0)

    return CaseArrays(
        ct_ras=np.ascontiguousarray(ct_ras),
        gt_ras=np.ascontiguousarray(gt_ras),
        predictions_ras=predictions_ras,
        spacing_ras=spacing_ras,
        voxel_volume_cm3=voxel_volume_cm3,
        ct_axcodes=tuple(nib.aff2axcodes(ct_img.affine)),
        orientation_transform=orientation_as_list(to_ras),
    )


def component_structure() -> np.ndarray:
    return np.ones((3, 3, 3), dtype=bool)


def prediction_metric_fields(
    gt_mask: np.ndarray,
    arrays: CaseArrays,
    finding_index: int,
    prediction_specs: list[PredictionSpec],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for spec in prediction_specs:
        pred_mask = arrays.predictions_ras[spec.key][finding_index] > 0
        global_dice = dice_score(gt_mask, pred_mask)
        fields[f"{spec.key}_global_dice"] = round(global_dice, 6)
        fields[f"{spec.key}_hit"] = bool(global_dice >= GLOBAL_HIT_THRESHOLD)
    first = prediction_specs[0]
    fields["prediction_global_dice"] = fields[f"{first.key}_global_dice"]
    fields["prediction_hit"] = fields[f"{first.key}_hit"]
    return fields


def gt_component_infos(
    gt_mask: np.ndarray,
    voxel_volume_cm3: float,
) -> list[dict[str, Any]]:
    """Measure 3D GT components while storing only the selected display slice."""
    if not np.any(gt_mask):
        return []

    coords = np.where(gt_mask)
    x0, x1 = int(coords[0].min()), int(coords[0].max()) + 1
    y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
    z0, z1 = int(coords[2].min()), int(coords[2].max()) + 1
    cropped = gt_mask[x0:x1, y0:y1, z0:z1]
    labeled, component_count = ndimage.label(cropped, structure=component_structure())
    infos: list[dict[str, Any]] = []

    for component_id in range(1, int(component_count) + 1):
        local_mask = labeled == component_id
        voxel_count = int(local_mask.sum())
        volume_cm3 = float(voxel_count * voxel_volume_cm3)
        local_com = tuple(float(value) for value in ndimage.center_of_mass(local_mask))
        com = (local_com[0] + x0, local_com[1] + y0, local_com[2] + z0)
        y_index = int(np.clip(round(com[1]), 0, gt_mask.shape[1] - 1))
        xz_slice = np.zeros((gt_mask.shape[0], gt_mask.shape[2]), dtype=bool)
        if y0 <= y_index < y1:
            xz_slice[x0:x1, z0:z1] = local_mask[:, y_index - y0, :]
        infos.append(
            {
                "component_id": component_id,
                "voxel_count": voxel_count,
                "volume_cm3": volume_cm3,
                "com": com,
                "coronal_y_index_ras": y_index,
                "component_display_slice": display_coronal_xz_slice(xz_slice),
            }
        )
    return infos


def collect_case_records(
    entry: dict[str, Any],
    metadata_entry: dict[str, Any],
    arrays: CaseArrays,
    prediction_specs: list[PredictionSpec],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    summary_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    finding_components: dict[str, dict[str, Any]] = {}
    findings = metadata_entry.get("findings", entry.get("findings", {}))
    categories = metadata_entry.get("categories", entry.get("categories", {}))

    for finding_id in sorted_finding_ids({"findings": findings}):
        index = int(finding_id)
        gt_mask = arrays.gt_ras[index] > 0
        metric_fields = prediction_metric_fields(gt_mask, arrays, index, prediction_specs)
        component_infos = gt_component_infos(gt_mask, arrays.voxel_volume_cm3)
        component_count = len(component_infos)
        component_summaries: list[dict[str, Any]] = []
        component_volumes_cm3: list[float] = []
        component_coms: list[tuple[float, float, float]] = []

        for info in component_infos:
            component_id = int(info["component_id"])
            voxel_count = int(info["voxel_count"])
            volume_cm3 = float(info["volume_cm3"])
            com = tuple(float(value) for value in info["com"])
            y_index = int(info["coronal_y_index_ras"])
            component_volumes_cm3.append(volume_cm3)
            component_coms.append(com)
            component_record = {
                "reshuffled_val_index": entry["reshuffled_val_index"],
                "case_name": entry["name"],
                "finding_id": finding_id,
                "component_id": component_id,
                "gt_component_count": int(component_count),
                "category": categories.get(finding_id, ""),
                "category_label": category_label(categories.get(finding_id, "")),
                "finding": findings[finding_id],
                "gt_component_voxels": voxel_count,
                "gt_component_volume_cm3": round(volume_cm3, 6),
                "gt_component_com_ras_x": round(com[0], 3),
                "gt_component_com_ras_y": round(com[1], 3),
                "gt_component_com_ras_z": round(com[2], 3),
                "coronal_y_index_ras": y_index,
                **metric_fields,
            }
            component_records.append(component_record)
            component_summaries.append({**component_record, "component_display_slice": info["component_display_slice"]})

        summary_records.append(
            {
                "reshuffled_val_index": entry["reshuffled_val_index"],
                "case_name": entry["name"],
                "finding_id": finding_id,
                "category": categories.get(finding_id, ""),
                "category_label": category_label(categories.get(finding_id, "")),
                "finding": findings[finding_id],
                "gt_component_count": int(component_count),
                "gt_total_voxels": int(gt_mask.sum()),
                "gt_total_volume_cm3": round(float(gt_mask.sum() * arrays.voxel_volume_cm3), 6),
                "gt_component_volumes_cm3": "; ".join(f"{value:.6f}" for value in component_volumes_cm3),
                "gt_component_com_ras_xyz": "; ".join(
                    f"({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f})" for com in component_coms
                ),
                **metric_fields,
            }
        )
        finding_components[finding_id] = {
            "component_count": int(component_count),
            "components": component_summaries,
            "category": categories.get(finding_id, ""),
            "category_label": category_label(categories.get(finding_id, "")),
            "finding": findings[finding_id],
            **metric_fields,
        }

    return summary_records, component_records, finding_components


def category_groups(finding_components: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for finding_id in sorted(finding_components, key=lambda value: int(value)):
        category = finding_components[finding_id]["category"]
        groups.setdefault(category, []).append(finding_id)
    return dict(sorted(groups.items()))


def draw_contour(ax: plt.Axes, mask_slice: np.ndarray, color: str, empty_label: str) -> None:
    if np.any(mask_slice):
        ax.contour(mask_slice.astype(float), levels=[0.5], colors=color, linewidths=1.5)
    else:
        ax.text(
            0.98,
            0.03,
            empty_label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=color,
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2},
        )


def wrapped_label(text: str, width: int = 58, max_lines: int = 5) -> str:
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][: max(0, width - 3)] + "..."]
    return "\n".join(lines)


def metric_label(component: dict[str, Any], prediction_specs: list[PredictionSpec]) -> str:
    lines = []
    for spec in prediction_specs:
        dice_value = component[f"{spec.key}_global_dice"]
        hit_value = component[f"{spec.key}_hit"]
        lines.append(f"{spec.label}: Dice {dice_value:.3f} | Hit {'yes' if hit_value else 'no'}")
    return "\n".join(lines)


def render_case_figure(
    entry: dict[str, Any],
    arrays: CaseArrays,
    finding_components: dict[str, dict[str, Any]],
    figure_dir: Path,
    prediction_specs: list[PredictionSpec],
    selected_finding_ids: list[str] | None = None,
    category_code: str | None = None,
    window_center: float = LUNG_WINDOW_CENTER,
    window_width: float = LUNG_WINDOW_WIDTH,
    dpi: int = 140,
) -> Path:
    rows: list[dict[str, Any]] = []
    selected = selected_finding_ids or sorted(finding_components, key=lambda value: int(value))
    for finding_id in selected:
        rows.extend(finding_components[finding_id]["components"])
    if not rows:
        raise ValueError(f"{entry['name']}: no GT connected components to plot")

    row_count = len(rows)
    column_count = 2 + len(prediction_specs)
    min_height = 4.8 if row_count == 1 else 3.6
    fig_height = max(min_height, 2.55 * row_count)
    fig_width = 8.2 + 3.9 * column_count
    fig, axes = plt.subplots(row_count, column_count, figsize=(fig_width, fig_height), squeeze=False)
    spacing_x, _spacing_y, spacing_z = arrays.spacing_ras
    aspect = spacing_z / spacing_x if spacing_x else "auto"

    column_titles = ["Raw CT", "GT contour", *[spec.column_title for spec in prediction_specs]]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10, pad=8)

    for row_index, component in enumerate(rows):
        y_index = int(component["coronal_y_index_ras"])
        ct_slice = window_ct(display_coronal_slice(arrays.ct_ras, y_index), window_center, window_width)
        gt_slice = component["component_display_slice"]
        row_label = (
            f"val {component['reshuffled_val_index']:03d} | F{component['finding_id']} "
            f"C{component['component_id']}/{component['gt_component_count']}\n"
            f"{component['category_label']}\n"
            f"{metric_label(component, prediction_specs)}\n"
            f"GT comp {component['gt_component_volume_cm3']:.3f} cm3\n"
            f"{wrapped_label(component['finding'])}"
        )

        for col in range(column_count):
            ax = axes[row_index, col]
            ax.imshow(ct_slice, cmap="gray", origin="upper", aspect=aspect, vmin=0.0, vmax=1.0)
            ax.set_axis_off()
        draw_contour(axes[row_index, 1], gt_slice, "#30d158", "no GT on slice")
        for spec_index, spec in enumerate(prediction_specs, start=2):
            pred_slice = display_coronal_slice(
                arrays.predictions_ras[spec.key][int(component["finding_id"])] > 0,
                y_index,
            )
            draw_contour(axes[row_index, spec_index], pred_slice, spec.contour_color, f"no {spec.label} pred")
        axes[row_index, 0].text(
            -0.05,
            0.5,
            row_label,
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7.2,
            clip_on=False,
        )

    category_title = f" | {category_label(category_code)}" if category_code else ""
    fig.suptitle(
        (
            f"{entry['name']} | reshuffled val index {entry['reshuffled_val_index']}{category_title} | "
            f"CT axcodes {''.join(arrays.ct_axcodes)} | lung window C={window_center:g}, W={window_width:g}\n"
            "Coronal radiology display: top=S, bottom=I, screen left=patient R, screen right=patient L"
        ),
        fontsize=11,
        y=0.985,
    )
    top = 0.74 if row_count == 1 else 0.91
    fig.subplots_adjust(left=0.30, right=0.99, top=top, bottom=0.01, wspace=0.03, hspace=0.07)
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / figure_filename(entry, category_code)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def figure_filename(entry: dict[str, Any], category_code: str | None = None) -> str:
    stem = entry["name"].replace(".nii.gz", "")
    if category_code:
        return f"{category_code}_{entry['reshuffled_val_index']:03d}_{stem}.png"
    return f"{entry['reshuffled_val_index']:03d}_{stem}.png"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    summary_df: pd.DataFrame,
    component_df: pd.DataFrame,
    figure_paths: list[Path],
    manifest: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    component_df.to_csv(output_dir / "component_table.csv", index=False)
    (output_dir / "summary_table.md").write_text(dataframe_to_markdown(summary_df))
    (output_dir / "component_table.md").write_text(dataframe_to_markdown(component_df))
    manifest = dict(manifest)
    manifest["figure_paths"] = [str(path) for path in figure_paths]
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def run_visualization(
    limit: int | None = 20,
    start_index: int = 0,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    val_json: Path = DEFAULT_VAL_JSON,
    metadata_json: Path = DEFAULT_METADATA_JSON,
    seg_dir: Path = DEFAULT_SEG_DIR,
    pred_dir: Path = DEFAULT_PRED_DIR,
    ct_root: Path = DEFAULT_CT_ROOT,
    prediction_specs: list[PredictionSpec] | None = None,
    category_grouped: bool = False,
    render: bool = True,
    progress: bool = True,
    skip_existing: bool = False,
    window_center: float = LUNG_WINDOW_CENTER,
    window_width: float = LUNG_WINDOW_WIDTH,
    dpi: int = 140,
) -> VisualizationResult:
    prediction_specs = prediction_specs or default_prediction_specs(pred_dir)
    entries = load_reshuffled_entries(val_json, limit=limit, start_index=start_index)
    metadata_index = load_metadata_index(metadata_json)
    summary_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    group_records: list[dict[str, Any]] = []
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_number, entry in enumerate(entries, start=1):
        name = entry["name"]
        metadata_entry = metadata_index.get(name, entry)
        arrays = load_case_arrays(
            name,
            seg_dir=seg_dir,
            pred_dir=pred_dir,
            ct_root=ct_root,
            prediction_specs=prediction_specs,
            load_ct_data=render,
        )
        case_summary, case_components, finding_components = collect_case_records(
            entry,
            metadata_entry,
            arrays,
            prediction_specs,
        )
        summary_records.extend(case_summary)
        component_records.extend(case_components)
        groups = category_groups(finding_components) if category_grouped else {"": sorted_finding_ids(metadata_entry)}
        for category_code, finding_ids in groups.items():
            group_records.append(
                {
                    "reshuffled_val_index": entry["reshuffled_val_index"],
                    "case_name": name,
                    "category": category_code,
                    "category_label": category_label(category_code) if category_code else "",
                    "finding_ids": "; ".join(finding_ids),
                    "finding_count": len(finding_ids),
                }
            )
            if render:
                figure_category = category_code if category_grouped else None
                existing_path = figure_dir / figure_filename(entry, figure_category)
                if skip_existing and existing_path.is_file():
                    figure_paths.append(existing_path)
                else:
                    figure_paths.append(
                        render_case_figure(
                            entry,
                            arrays,
                            finding_components,
                            figure_dir=figure_dir,
                            prediction_specs=prediction_specs,
                            selected_finding_ids=finding_ids,
                            category_code=figure_category,
                            window_center=window_center,
                            window_width=window_width,
                            dpi=dpi,
                        )
                    )
        if progress:
            print(
                f"[{case_number:03d}/{len(entries):03d}] {name} "
                f"findings={len(case_summary)} components={len(case_components)} groups={len(groups)} "
                f"figures={len(figure_paths)}",
                flush=True,
            )

    summary_df = pd.DataFrame.from_records(summary_records)
    component_df = pd.DataFrame.from_records(component_records)
    group_df = pd.DataFrame.from_records(group_records)
    if not group_df.empty:
        group_df.to_csv(output_dir / "category_group_table.csv", index=False)
        (output_dir / "category_group_table.md").write_text(dataframe_to_markdown(group_df))
    manifest = {
        "description": "Reshuffled validation coronal visualization with GT and prediction contours.",
        "val_json": str(val_json),
        "metadata_json": str(metadata_json),
        "ct_root": str(ct_root),
        "seg_dir": str(seg_dir),
        "prediction_specs": [
            {
                "key": spec.key,
                "label": spec.label,
                "column_title": spec.column_title,
                "pred_dir": str(spec.pred_dir),
                "contour_color": spec.contour_color,
            }
            for spec in prediction_specs
        ],
        "output_dir": str(output_dir),
        "start_index": start_index,
        "case_count": len(entries),
        "finding_count": int(len(summary_df)),
        "component_count": int(len(component_df)),
        "category_grouped": category_grouped,
        "category_group_count": int(len(group_df)),
        "figure_count": len(figure_paths),
        "skip_existing": skip_existing,
        "global_hit_threshold": GLOBAL_HIT_THRESHOLD,
        "window_center": window_center,
        "window_width": window_width,
        "dpi": dpi,
        "category_name_source": "challenge_info/rexgroundingct.md ReXrank category table",
        "orientation": "RAS reorientation from CT affine; coronal display top=S, screen-left=patient R.",
    }
    manifest_path = write_outputs(output_dir, summary_df, component_df, figure_paths, manifest)
    return VisualizationResult(output_dir, summary_df, component_df, figure_paths, manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="Number of reshuffled val cases to render. Use 0 for all.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-json", type=Path, default=DEFAULT_VAL_JSON)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_JSON)
    parser.add_argument("--seg-dir", type=Path, default=DEFAULT_SEG_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR, help="Primary v123 prediction directory.")
    parser.add_argument("--pretrained-pred-dir", type=Path, default=None)
    parser.add_argument("--ct-root", type=Path, default=DEFAULT_CT_ROOT)
    parser.add_argument("--category-grouped", action="store_true")
    parser.add_argument("--window-center", type=float, default=LUNG_WINDOW_CENTER)
    parser.add_argument("--window-width", type=float, default=LUNG_WINDOW_WIDTH)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--no-render", action="store_true", help="Write tables only, without PNG figures.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-case progress output.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing PNG files instead of overwriting them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_specs = default_prediction_specs(args.pred_dir)
    if args.pretrained_pred_dir is not None:
        prediction_specs = val200_category_prediction_specs(args.pred_dir, args.pretrained_pred_dir)
    result = run_visualization(
        limit=None if args.limit == 0 else args.limit,
        start_index=args.start_index,
        output_dir=args.output_dir,
        val_json=args.val_json,
        metadata_json=args.metadata_json,
        seg_dir=args.seg_dir,
        pred_dir=args.pred_dir,
        ct_root=args.ct_root,
        prediction_specs=prediction_specs,
        category_grouped=args.category_grouped,
        render=not args.no_render,
        progress=not args.quiet,
        skip_existing=args.skip_existing,
        window_center=args.window_center,
        window_width=args.window_width,
        dpi=args.dpi,
    )
    print(f"wrote output_dir={result.output_dir}")
    print(f"findings={len(result.summary_df)} components={len(result.component_df)} figures={len(result.figure_paths)}")
    print(f"manifest={result.manifest_path}")


if __name__ == "__main__":
    main()
