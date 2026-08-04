#!/usr/bin/env python3
"""Render representative coronal CT projections with four VoxTell masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_ROOT = SCRIPT_DIR.parent
REPO_ROOT = VISUALIZATION_ROOT.parent
if str(VISUALIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_ROOT))

from rex_val20_coronal_viz import (  # noqa: E402
    CATEGORY_NAMES,
    DEFAULT_CT_ROOT,
    DEFAULT_METADATA_JSON,
    DEFAULT_SEG_DIR,
    DEFAULT_VAL_JSON,
    GLOBAL_HIT_THRESHOLD,
    LUNG_WINDOW_CENTER,
    LUNG_WINDOW_WIDTH,
    PredictionSpec,
    category_label,
    dice_score,
    display_coronal_xz_slice,
    load_case_arrays,
    load_json,
    load_metadata_index,
    load_reshuffled_entries,
    sorted_finding_ids,
)


DEFAULT_PILOT_CONFIG = SCRIPT_DIR / "pilot_cases.json"
DOCKER_VISUALIZATION_ROOT = Path("/database/datasets/rexgroundingct/visualizations")
HOST_VISUALIZATION_ROOT = Path("/data/hengjie/datasets/rexgroundingct/visualizations")
PILOT_MODE = "pilot"
VAL200_CATEGORY_MODE = "val200-by-category"
OFFICIAL_CATEGORY_CODES = tuple(CATEGORY_NAMES)
EMPTY_CATEGORY_CODE = "2f"


def visualization_dataset_root() -> Path:
    override = os.environ.get("REXGROUNDINGCT_VISUALIZATION_ROOT")
    if override:
        return Path(override)
    elif DOCKER_VISUALIZATION_ROOT.parent.is_dir():
        return DOCKER_VISUALIZATION_ROOT
    return HOST_VISUALIZATION_ROOT


def default_output_dir(mode: str = PILOT_MODE) -> Path:
    leaf = "pilot" if mode == PILOT_MODE else "full_val200_mean_by_category"
    return visualization_dataset_root() / "2026-07-31_visualdev_coronal_projection" / leaf


DEFAULT_OUTPUT_DIR = default_output_dir(PILOT_MODE)
DEFAULT_VAL200_CATEGORY_OUTPUT_DIR = default_output_dir(VAL200_CATEGORY_MODE)

PUBLIC_PRED_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "001_voxtell_v1_1_miccai200_val_eval/predictions"
)
EXP009_ATTENTION_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "009_voxtell_s3_attention_coupling_ablation/runs/"
    "exp009_s3_attention_20260727T081747Z/s3v1_fixedrho_suppress_half_quarter/"
    "eval_epoch100_val200/predictions"
)
NODDP_BEST_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "009_voxtell_s3_attention_coupling_ablation/runs/"
    "exp009_s3_attention_20260727T081747Z/baseline_cont100/"
    "eval_epoch100_val200/predictions"
)
DDP_BEST_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/"
    "exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/"
    "eval_epoch050_val200/predictions"
)

EXP009_ATTENTION_CHECKPOINT = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "009_voxtell_s3_attention_coupling_ablation/runs/"
    "exp009_s3_attention_20260727T081747Z/s3v1_fixedrho_suppress_half_quarter/"
    "model_epoch100/fold_0/checkpoint_final.pth"
)
NODDP_BEST_CHECKPOINT = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "009_voxtell_s3_attention_coupling_ablation/runs/"
    "exp009_s3_attention_20260727T081747Z/baseline_cont100/"
    "model_epoch100/fold_0/checkpoint_final.pth"
)
DDP_BEST_CHECKPOINT = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/"
    "exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/"
    "checkpoints/checkpoint_update_005000.pth"
)

GT_COLOR = np.array([0.188, 0.820, 0.345, 0.58], dtype=np.float32)
TP_COLOR = np.array([0.188, 0.820, 0.345, 0.58], dtype=np.float32)
FP_COLOR = np.array([1.000, 0.271, 0.227, 0.62], dtype=np.float32)
FN_COLOR = np.array([0.039, 0.518, 1.000, 0.62], dtype=np.float32)
DEPTH_MISMATCH_COLOR = np.array([0.749, 0.353, 0.949, 0.62], dtype=np.float32)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    short_label: str
    column_title: str
    pred_dir: Path
    checkpoint_path: Path | None
    checkpoint_reference: str
    overall_dice: float
    overall_hit_rate: float
    lineage: str

    def prediction_spec(self) -> PredictionSpec:
        return PredictionSpec(
            key=self.key,
            label=self.short_label,
            column_title=self.column_title,
            pred_dir=self.pred_dir,
            contour_color="#ffffff",
        )


MODEL_SPECS = [
    ModelSpec(
        key="public_v11",
        short_label="Public v1.1",
        column_title="Public VoxTell v1.1\nOriginal pretrained | no project epoch",
        pred_dir=PUBLIC_PRED_DIR,
        checkpoint_path=None,
        checkpoint_reference="Hugging Face mrokuss/VoxTell model voxtell_v1.1",
        overall_dice=0.22522803991156426,
        overall_hit_rate=0.5354330708661418,
        lineage="Public VoxTell v1.1 pretrained checkpoint; corrected val200 export.",
    ),
    ModelSpec(
        key="exp009_s3v1_e100",
        short_label="100+100 attention",
        column_title="100+100 attention\nExp009 s3v1 e100 | update 10,000",
        pred_dir=EXP009_ATTENTION_DIR,
        checkpoint_path=EXP009_ATTENTION_CHECKPOINT,
        checkpoint_reference=str(EXP009_ATTENTION_CHECKPOINT),
        overall_dice=0.33470946473151214,
        overall_hit_rate=0.7480314960629921,
        lineage=(
            "Exp009 s3v1_fixedrho_suppress_half_quarter attention continuation for "
            "100 epochs / 10,000 updates from Exp006 e5d4 epoch 100."
        ),
    ),
    ModelSpec(
        key="noddp_best",
        short_label="100+100 plain",
        column_title="100+100 plain | Best no-DDP\nExp009 e100 | update 10,000",
        pred_dir=NODDP_BEST_DIR,
        checkpoint_path=NODDP_BEST_CHECKPOINT,
        checkpoint_reference=str(NODDP_BEST_CHECKPOINT),
        overall_dice=0.3398392601031359,
        overall_hit_rate=0.7559055118110236,
        lineage=(
            "Exp009 baseline_cont100 plain single-GPU continuation for 100 epochs / "
            "10,000 updates from Exp006 e5d4 epoch 100."
        ),
    ),
    ModelSpec(
        key="ddp_best",
        short_label="Best DDP",
        column_title="Best DDP\nrelative e50 | absolute e150 | update 5,000",
        pred_dir=DDP_BEST_DIR,
        checkpoint_path=DDP_BEST_CHECKPOINT,
        checkpoint_reference=str(DDP_BEST_CHECKPOINT),
        overall_dice=0.3460234857111323,
        overall_hit_rate=0.7769028871391076,
        lineage=(
            "Exp007 weights-only continuation update 5,000 / relative epoch 50 from the "
            "original Exp007 update-10,000 weights."
        ),
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def host_path_alias(path: Path) -> Path:
    """Translate the Docker dataset mount to its equivalent host path."""
    try:
        relative = path.relative_to(DOCKER_VISUALIZATION_ROOT)
    except ValueError:
        return path
    return HOST_VISUALIZATION_ROOT / relative


def category_directory_name(category_code: str) -> str:
    if category_code not in OFFICIAL_CATEGORY_CODES:
        raise ValueError(f"Unknown ReX category code: {category_code!r}")
    return "2f_empty" if category_code == EMPTY_CATEGORY_CODE else category_code


def case_stem(entry: dict[str, Any]) -> str:
    return str(entry["name"]).removesuffix(".nii.gz")


def pilot_figure_path(
    output_dir: Path,
    projection_method: str,
    entry: dict[str, Any],
) -> Path:
    return output_dir / projection_method / f"{entry['reshuffled_val_index']:03d}_{case_stem(entry)}.png"


def category_figure_path(
    output_dir: Path,
    category_code: str,
    entry: dict[str, Any],
) -> Path:
    filename = f"val{entry['reshuffled_val_index']:03d}_{case_stem(entry)}.png"
    return output_dir / category_directory_name(category_code) / filename


def finding_ids_by_category(metadata_entry: dict[str, Any]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    categories = metadata_entry.get("categories", {})
    for finding_id in sorted_finding_ids(metadata_entry):
        if finding_id not in categories:
            raise ValueError(f"{metadata_entry.get('name', '<unknown>')}: finding {finding_id} has no category")
        category_code = categories[finding_id]
        if category_code not in OFFICIAL_CATEGORY_CODES:
            raise ValueError(
                f"{metadata_entry.get('name', '<unknown>')}: finding {finding_id} has unknown "
                f"category {category_code!r}"
            )
        groups.setdefault(category_code, []).append(finding_id)
    return groups


def resolve_projection_methods(mode: str, requested: list[str] | None) -> tuple[str, ...]:
    if mode == VAL200_CATEGORY_MODE:
        if requested is not None and tuple(requested) != ("mean",):
            raise ValueError("val200-by-category mode supports only '--projection-methods mean'")
        return ("mean",)
    return tuple(requested or ("p75", "mean"))


def normalized_lung_window(
    ct_ras: np.ndarray,
    center: float = LUNG_WINDOW_CENTER,
    width: float = LUNG_WINDOW_WIDTH,
) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(ct_ras, low, high).astype(np.float32, copy=False)
    clipped -= low
    clipped /= high - low
    return clipped


def coronal_ct_projection_xz(
    ct_ras: np.ndarray,
    method: str,
    percentile: float = 75.0,
    center: float = LUNG_WINDOW_CENTER,
    width: float = LUNG_WINDOW_WIDTH,
) -> np.ndarray:
    """Collapse RAS anterior-posterior axis, returning an X/Z projection."""
    windowed = normalized_lung_window(ct_ras, center=center, width=width)
    if method == "p75":
        projected = np.percentile(windowed, percentile, axis=1, overwrite_input=True)
    elif method == "mean":
        projected = np.mean(windowed, axis=1)
    else:
        raise ValueError(f"Unsupported CT projection method: {method!r}")
    return np.asarray(projected, dtype=np.float32)


def coronal_mask_projection_xz(mask_ras: np.ndarray) -> np.ndarray:
    """Return the binary MIP of a 3D RAS mask along anterior-posterior."""
    if mask_ras.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask_ras.shape}")
    return np.any(mask_ras > 0, axis=1)


def radiology_coronal_display(xz_projection: np.ndarray) -> np.ndarray:
    """Display X/Z data with superior up and patient right on screen left."""
    if xz_projection.ndim != 2:
        raise ValueError(f"Expected a 2D X/Z projection, got shape {xz_projection.shape}")
    return display_coronal_xz_slice(xz_projection)


def rgba_mask(mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask] = color
    return rgba


def coronal_projected_error_masks_xz(
    gt_mask_ras: np.ndarray,
    pred_mask_ras: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project voxelwise error classes along AP, preserving true TP priority."""
    if gt_mask_ras.ndim != 3 or pred_mask_ras.ndim != 3:
        raise ValueError(f"Expected two 3D masks, got {gt_mask_ras.shape} and {pred_mask_ras.shape}")
    if gt_mask_ras.shape != pred_mask_ras.shape:
        raise ValueError(f"Mask shapes must match, got {gt_mask_ras.shape} and {pred_mask_ras.shape}")

    gt = gt_mask_ras > 0
    pred = pred_mask_ras > 0
    true_positive = np.any(gt & pred, axis=1)
    false_positive = np.any(~gt & pred, axis=1)
    false_negative = np.any(gt & ~pred, axis=1)

    depth_mismatch = ~true_positive & false_positive & false_negative
    projected_fp = ~true_positive & false_positive & ~false_negative
    projected_fn = ~true_positive & ~false_positive & false_negative
    return true_positive, projected_fp, projected_fn, depth_mismatch


def projected_error_rgba(gt_mask_ras: np.ndarray, pred_mask_ras: np.ndarray) -> np.ndarray:
    true_positive, false_positive, false_negative, depth_mismatch = (
        coronal_projected_error_masks_xz(gt_mask_ras, pred_mask_ras)
    )
    true_positive = radiology_coronal_display(true_positive)
    false_positive = radiology_coronal_display(false_positive)
    false_negative = radiology_coronal_display(false_negative)
    depth_mismatch = radiology_coronal_display(depth_mismatch)

    rgba = np.zeros((*true_positive.shape, 4), dtype=np.float32)
    rgba[true_positive] = TP_COLOR
    rgba[false_positive] = FP_COLOR
    rgba[false_negative] = FN_COLOR
    rgba[depth_mismatch] = DEPTH_MISMATCH_COLOR
    return rgba


def add_orientation_markers(ax: plt.Axes) -> None:
    style = {
        "color": "white",
        "fontsize": 7,
        "fontweight": "bold",
        "bbox": {"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.2},
    }
    ax.text(0.5, 0.99, "S", transform=ax.transAxes, ha="center", va="top", **style)
    ax.text(0.5, 0.01, "I", transform=ax.transAxes, ha="center", va="bottom", **style)
    ax.text(0.01, 0.5, "R", transform=ax.transAxes, ha="left", va="center", **style)
    ax.text(0.99, 0.5, "L", transform=ax.transAxes, ha="right", va="center", **style)


def wrap_finding(text: str, width: int = 50, max_lines: int = 4) -> str:
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, width - 3)] + "..."
    return "\n".join(lines)


def load_pilot_cases(pilot_config: Path, val_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_json(pilot_config)
    requested = config.get("cases")
    if not isinstance(requested, list) or not requested:
        raise ValueError(f"{pilot_config} must contain a non-empty 'cases' list")

    all_entries = load_reshuffled_entries(val_json=val_json, limit=None)
    selected: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for item in requested:
        index = int(item["reshuffled_val_index"])
        if index in seen_indices:
            raise ValueError(f"Duplicate pilot val index: {index}")
        if index < 0 or index >= len(all_entries):
            raise ValueError(f"Pilot val index {index} is outside [0, {len(all_entries) - 1}]")
        entry = dict(all_entries[index])
        expected_name = item["case_name"]
        if entry["name"] != expected_name:
            raise ValueError(
                f"Pilot config drift at val index {index}: expected {expected_name}, found {entry['name']}"
            )
        entry["selection_reason"] = item.get("reason", "")
        selected.append(entry)
        seen_indices.add(index)
    return config, selected


def load_full_val_cases(val_json: Path) -> list[dict[str, Any]]:
    entries = load_reshuffled_entries(val_json=val_json, limit=None)
    selected: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        item["selection_reason"] = ""
        selected.append(item)
    return selected


def build_category_case_index(
    entries: list[dict[str, Any]],
    metadata_index: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    category_rank = {code: rank for rank, code in enumerate(OFFICIAL_CATEGORY_CODES)}
    records: list[dict[str, Any]] = []
    for entry in entries:
        name = entry["name"]
        metadata_entry = metadata_index[name]
        patient_id = "_".join(case_stem(entry).split("_")[:2])
        for category_code, finding_ids in finding_ids_by_category(metadata_entry).items():
            output_path = category_figure_path(output_dir, category_code, entry)
            records.append(
                {
                    "category": category_code,
                    "category_label": category_label(category_code),
                    "reshuffled_val_index": int(entry["reshuffled_val_index"]),
                    "patient_id": patient_id,
                    "case_name": name,
                    "finding_ids": ";".join(finding_ids),
                    "finding_count": len(finding_ids),
                    "figure_path": str(output_path),
                    "figure_path_host_alias": str(host_path_alias(output_path)),
                }
            )
    return sorted(
        records,
        key=lambda record: (
            category_rank[record["category"]],
            record["reshuffled_val_index"],
            record["case_name"],
        ),
    )


def validate_inputs(
    entries: list[dict[str, Any]],
    metadata_index: dict[str, dict[str, Any]],
    ct_root: Path,
    seg_dir: Path,
    model_specs: list[ModelSpec],
) -> list[str]:
    errors: list[str] = []
    for model in model_specs:
        if not model.pred_dir.is_dir():
            errors.append(f"Missing prediction directory: {model.pred_dir}")
    for entry in entries:
        name = entry["name"]
        if name not in metadata_index:
            errors.append(f"Missing metadata entry: {name}")
        gt_path = seg_dir / name
        if not gt_path.is_file():
            errors.append(f"Missing GT: {gt_path}")
        # ct_rate_path validation is performed by load_case_arrays; dry-run keeps
        # its checks lightweight by avoiding NIfTI data loading.
        for model in model_specs:
            pred_path = model.pred_dir / name
            if not pred_path.is_file():
                errors.append(f"Missing {model.key} prediction: {pred_path}")
    if not ct_root.is_dir():
        errors.append(f"Missing CT root: {ct_root}")
    return errors


def render_case(
    entry: dict[str, Any],
    metadata_entry: dict[str, Any],
    arrays: Any,
    output_path: Path,
    projection_method: str,
    percentile: float,
    center: float,
    width: float,
    dpi: int,
    model_specs: list[ModelSpec],
    finding_ids: list[str] | None = None,
    category_code: str | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    finding_ids = list(finding_ids or sorted_finding_ids(metadata_entry))
    if not finding_ids:
        raise ValueError(f"{entry['name']} has no findings")
    if category_code is not None:
        mismatched = [
            finding_id
            for finding_id in finding_ids
            if metadata_entry.get("categories", {}).get(finding_id) != category_code
        ]
        if mismatched:
            raise ValueError(
                f"{entry['name']}: category {category_code} received mismatched findings {mismatched}"
            )

    ct_xz = coronal_ct_projection_xz(
        arrays.ct_ras,
        method=projection_method,
        percentile=percentile,
        center=center,
        width=width,
    )
    ct_display = radiology_coronal_display(ct_xz)
    row_count = len(finding_ids)
    column_count = 2 + len(model_specs)
    fig_width = 3.25 * column_count + 2.5
    fig_height = max(4.8, 2.75 * row_count + 1.2)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    spacing_x, _spacing_y, spacing_z = arrays.spacing_ras
    aspect = spacing_z / spacing_x if spacing_x else "auto"

    method_title = f"CT P{percentile:g}" if projection_method == "p75" else "CT mean"
    titles = [method_title, "GT mask MIP"]
    titles.extend(
        f"{model.column_title}\nval200 Dice {model.overall_dice:.4f} | hit {model.overall_hit_rate:.3f}"
        for model in model_specs
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=8.5, pad=7)

    metric_records: list[dict[str, Any]] = []
    findings = metadata_entry["findings"]
    categories = metadata_entry.get("categories", {})
    for row, finding_id in enumerate(finding_ids):
        finding_index = int(finding_id)
        gt_mask = arrays.gt_ras[finding_index] > 0
        gt_display = radiology_coronal_display(coronal_mask_projection_xz(gt_mask))

        for column in range(column_count):
            ax = axes[row, column]
            ax.imshow(ct_display, cmap="gray", origin="upper", aspect=aspect, vmin=0.0, vmax=1.0)
            ax.set_axis_off()
            add_orientation_markers(ax)

        axes[row, 1].imshow(rgba_mask(gt_display, GT_COLOR), origin="upper", aspect=aspect)

        finding_category_code = categories.get(finding_id, "")
        row_label = (
            f"val {entry['reshuffled_val_index']:03d} | F{finding_id}\n"
            f"{category_label(finding_category_code)}\n"
            f"GT {int(gt_mask.sum()):,} voxels\n"
            f"{wrap_finding(findings[finding_id])}"
        )
        axes[row, 0].text(
            -0.04,
            0.5,
            row_label,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7.0,
            clip_on=False,
        )

        base_record = {
            "reshuffled_val_index": entry["reshuffled_val_index"],
            "case_name": entry["name"],
            "selection_reason": entry["selection_reason"],
            "finding_id": finding_id,
            "category": finding_category_code,
            "category_label": category_label(finding_category_code),
            "finding": findings[finding_id],
            "gt_voxels": int(gt_mask.sum()),
        }
        record = dict(base_record)
        for model_column, model in enumerate(model_specs, start=2):
            pred_mask = arrays.predictions_ras[model.key][finding_index] > 0
            axes[row, model_column].imshow(
                projected_error_rgba(gt_mask, pred_mask),
                origin="upper",
                aspect=aspect,
            )
            dice = dice_score(gt_mask, pred_mask)
            hit = bool(dice >= GLOBAL_HIT_THRESHOLD)
            axes[row, model_column].text(
                0.02,
                0.02,
                f"3D Dice {dice:.3f} | hit {'yes' if hit else 'no'}",
                transform=axes[row, model_column].transAxes,
                ha="left",
                va="bottom",
                fontsize=7.0,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 1.8},
            )
            record[f"{model.key}_dice"] = round(float(dice), 6)
            record[f"{model.key}_hit"] = hit
            record[f"{model.key}_pred_voxels"] = int(pred_mask.sum())
        metric_records.append(record)

    legend = [
        Patch(facecolor=TP_COLOR, label="projected TP"),
        Patch(facecolor=FP_COLOR, label="projected FP"),
        Patch(facecolor=FN_COLOR, label="projected FN"),
        Patch(facecolor=DEPTH_MISMATCH_COLOR, label="depth-disjoint FN+FP"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8, frameon=False)
    category_title = f" | {category_label(category_code)}" if category_code else ""
    fig.suptitle(
        (
            f"{entry['name']} | fixed val index {entry['reshuffled_val_index']:03d} | "
            f"{projection_method} AP projection{category_title} | "
            f"CT axcodes {''.join(arrays.ct_axcodes)}\n"
            "Radiology coronal display: top=S, bottom=I, screen left=patient R, screen right=patient L"
        ),
        fontsize=10.5,
        y=0.992,
    )
    fig.subplots_adjust(left=0.20, right=0.995, top=0.90, bottom=0.06, wspace=0.025, hspace=0.08)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path, metric_records


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(PILOT_MODE, VAL200_CATEGORY_MODE), default=PILOT_MODE)
    parser.add_argument("--pilot-config", type=Path, default=DEFAULT_PILOT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--val-json", type=Path, default=DEFAULT_VAL_JSON)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_JSON)
    parser.add_argument("--seg-dir", type=Path, default=DEFAULT_SEG_DIR)
    parser.add_argument("--ct-root", type=Path, default=DEFAULT_CT_ROOT)
    parser.add_argument("--public-pred-dir", type=Path, default=PUBLIC_PRED_DIR)
    parser.add_argument("--attention-pred-dir", type=Path, default=EXP009_ATTENTION_DIR)
    parser.add_argument("--noddp-pred-dir", type=Path, default=NODDP_BEST_DIR)
    parser.add_argument("--ddp-pred-dir", type=Path, default=DDP_BEST_DIR)
    parser.add_argument("--projection-methods", nargs="+", choices=("p75", "mean"))
    parser.add_argument("--percentile", type=float, default=75.0)
    parser.add_argument("--window-center", type=float, default=LUNG_WINDOW_CENTER)
    parser.add_argument("--window-width", type=float, default=LUNG_WINDOW_WIDTH)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.projection_methods = resolve_projection_methods(args.mode, args.projection_methods)
    except ValueError as error:
        parser.error(str(error))
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.mode)
    return args


def model_specs_from_args(args: argparse.Namespace) -> list[ModelSpec]:
    return [
        replace(MODEL_SPECS[0], pred_dir=args.public_pred_dir),
        replace(MODEL_SPECS[1], pred_dir=args.attention_pred_dir),
        replace(MODEL_SPECS[2], pred_dir=args.noddp_pred_dir),
        replace(MODEL_SPECS[3], pred_dir=args.ddp_pred_dir),
    ]


def orientation_record(entry: dict[str, Any], arrays: Any) -> dict[str, Any]:
    return {
        "reshuffled_val_index": entry["reshuffled_val_index"],
        "case_name": entry["name"],
        "ct_axcodes": list(arrays.ct_axcodes),
        "orientation_transform_to_ras": arrays.orientation_transform,
        "spacing_ras": list(arrays.spacing_ras),
    }


def model_manifest_records(model_specs: list[ModelSpec]) -> list[dict[str, Any]]:
    return [
        {
            "key": model.key,
            "short_label": model.short_label,
            "column_title": model.column_title,
            "prediction_dir": str(model.pred_dir),
            "checkpoint_reference": model.checkpoint_reference,
            "checkpoint_path": str(model.checkpoint_path) if model.checkpoint_path else None,
            "checkpoint_exists": model.checkpoint_path.is_file() if model.checkpoint_path else None,
            "overall_val200_dice": model.overall_dice,
            "overall_val200_hit_rate": model.overall_hit_rate,
            "lineage": model.lineage,
        }
        for model in model_specs
    ]


def write_dataframe_artifacts(df: pd.DataFrame, csv_path: Path, markdown_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    markdown_path.write_text(dataframe_to_markdown(df))


def main() -> None:
    args = parse_args()
    model_specs = model_specs_from_args(args)
    pilot_config: dict[str, Any] | None = None
    if args.mode == PILOT_MODE:
        pilot_config, entries = load_pilot_cases(args.pilot_config, args.val_json)
    else:
        entries = load_full_val_cases(args.val_json)

    metadata_index = load_metadata_index(args.metadata_json)
    input_errors = validate_inputs(entries, metadata_index, args.ct_root, args.seg_dir, model_specs)
    if input_errors:
        raise FileNotFoundError("\n".join(input_errors))

    finding_count = sum(len(metadata_index[entry["name"]]["findings"]) for entry in entries)
    category_case_index: list[dict[str, Any]] = []
    if args.mode == VAL200_CATEGORY_MODE:
        category_case_index = build_category_case_index(entries, metadata_index, args.output_dir)
        print(
            f"validated {len(entries)} val200 cases, {finding_count} findings, "
            f"{len(category_case_index)} category figures, and {len(model_specs)} prediction sets"
        )
        for category_code in OFFICIAL_CATEGORY_CODES:
            records = [record for record in category_case_index if record["category"] == category_code]
            print(
                f"{category_directory_name(category_code)} | cases={len(records)} | "
                f"findings={sum(record['finding_count'] for record in records)}"
            )
    else:
        print(f"validated {len(entries)} pilot cases and {len(model_specs)} prediction sets")
        for entry in entries:
            print(
                f"val {entry['reshuffled_val_index']:03d} {entry['name']} | "
                f"findings={len(metadata_index[entry['name']]['findings'])} | "
                f"{entry['selection_reason']}"
            )
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == VAL200_CATEGORY_MODE:
        for category_code in OFFICIAL_CATEGORY_CODES:
            (args.output_dir / category_directory_name(category_code)).mkdir(parents=True, exist_ok=True)

    prediction_specs = [model.prediction_spec() for model in model_specs]
    all_records: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    case_orientations: list[dict[str, Any]] = []
    category_records_by_case: dict[str, list[dict[str, Any]]] = {}
    for record in category_case_index:
        category_records_by_case.setdefault(record["case_name"], []).append(record)

    for case_number, entry in enumerate(entries, start=1):
        name = entry["name"]
        arrays = load_case_arrays(
            name,
            seg_dir=args.seg_dir,
            ct_root=args.ct_root,
            prediction_specs=prediction_specs,
            load_ct_data=True,
        )
        metadata_entry = metadata_index[name]
        if args.mode == VAL200_CATEGORY_MODE:
            case_category_records = category_records_by_case[name]
            for category_record in case_category_records:
                finding_ids = category_record["finding_ids"].split(";")
                figure_path, records = render_case(
                    entry,
                    metadata_entry,
                    arrays,
                    output_path=Path(category_record["figure_path"]),
                    projection_method="mean",
                    percentile=args.percentile,
                    center=args.window_center,
                    width=args.window_width,
                    dpi=args.dpi,
                    model_specs=model_specs,
                    finding_ids=finding_ids,
                    category_code=category_record["category"],
                )
                figure_paths.append(figure_path)
                all_records.extend(records)
            progress_width = 3
            figure_message = (
                f"category_figures={len(case_category_records)} cumulative={len(figure_paths)}"
            )
        else:
            case_records: list[dict[str, Any]] | None = None
            for method in args.projection_methods:
                figure_path, records = render_case(
                    entry,
                    metadata_entry,
                    arrays,
                    output_path=pilot_figure_path(args.output_dir, method, entry),
                    projection_method=method,
                    percentile=args.percentile,
                    center=args.window_center,
                    width=args.window_width,
                    dpi=args.dpi,
                    model_specs=model_specs,
                )
                figure_paths.append(figure_path)
                if case_records is None:
                    case_records = records
            all_records.extend(case_records or [])
            progress_width = 2
            figure_message = f"figures={len(args.projection_methods)}"

        case_orientations.append(orientation_record(entry, arrays))
        print(
            f"[{case_number:0{progress_width}d}/{len(entries):0{progress_width}d}] "
            f"val {entry['reshuffled_val_index']:03d} {name} "
            f"findings={len(metadata_entry['findings'])} {figure_message}",
            flush=True,
        )

    metrics_df = pd.DataFrame.from_records(all_records)
    if args.mode == VAL200_CATEGORY_MODE:
        category_rank = {code: rank for rank, code in enumerate(OFFICIAL_CATEGORY_CODES)}
        metrics_df["_category_rank"] = metrics_df["category"].map(category_rank)
        metrics_df["_finding_rank"] = metrics_df["finding_id"].astype(int)
        metrics_df = metrics_df.sort_values(
            ["_category_rank", "reshuffled_val_index", "_finding_rank", "case_name"]
        ).drop(columns=["_category_rank", "_finding_rank"])
        metrics_csv = args.output_dir / "val200_finding_metrics.csv"
        metrics_markdown = args.output_dir / "val200_finding_metrics.md"
        category_index_df = pd.DataFrame.from_records(category_case_index)
        write_dataframe_artifacts(
            category_index_df,
            args.output_dir / "category_case_index.csv",
            args.output_dir / "category_case_index.md",
        )
        figure_paths = [Path(record["figure_path"]) for record in category_case_index]
    else:
        metrics_csv = args.output_dir / "pilot_finding_metrics.csv"
        metrics_markdown = args.output_dir / "pilot_finding_metrics.md"
    write_dataframe_artifacts(metrics_df, metrics_csv, metrics_markdown)

    category_case_counts = {
        category_directory_name(code): sum(
            record["category"] == code for record in category_case_index
        )
        for code in OFFICIAL_CATEGORY_CODES
    }
    category_finding_counts = {
        category_directory_name(code): sum(
            record["finding_count"]
            for record in category_case_index
            if record["category"] == code
        )
        for code in OFFICIAL_CATEGORY_CODES
    }

    manifest = {
        "description": (
            "Full val200 mean coronal projection visualization organized by ReX category."
            if args.mode == VAL200_CATEGORY_MODE
            else "Representative coronal projection visualization pilot with four VoxTell prediction sets."
        ),
        "mode": args.mode,
        "repo_commit": repo_commit(),
        "script": str(Path(__file__).resolve()),
        "val_json": str(args.val_json.resolve()),
        "metadata_json": str(args.metadata_json),
        "seg_dir": str(args.seg_dir),
        "ct_root": str(args.ct_root),
        "output_dir": str(args.output_dir),
        "output_dir_host_alias": str(host_path_alias(args.output_dir)),
        "docker_visualization_root": str(DOCKER_VISUALIZATION_ROOT),
        "host_visualization_root": str(HOST_VISUALIZATION_ROOT),
        "case_count": len(entries),
        "finding_count": len(metrics_df),
        "figure_count": len(figure_paths),
        "projection_methods": list(args.projection_methods),
        "ct_projection": {
            "axis": "RAS anterior-posterior axis 1",
            "percentile": args.percentile,
            "window_center": args.window_center,
            "window_width": args.window_width,
        },
        "mask_projection": "binary maximum/any along RAS anterior-posterior axis 1",
        "display_orientation": {
            "top": "patient superior",
            "bottom": "patient inferior",
            "screen_left": "patient right",
            "screen_right": "patient left",
        },
        "models": model_manifest_records(model_specs),
        "selected_cases": entries,
        "case_orientations": case_orientations,
        "figure_paths": [str(path) for path in figure_paths],
        "figure_paths_host_alias": [str(host_path_alias(path)) for path in figure_paths],
        "metric_table": str(metrics_csv),
        "projected_metric_caveat": (
            "Colors are projected after voxelwise 3D TP/FP/FN classification. Purple marks rays "
            "with FN and FP at different AP depths but no voxelwise TP. All displayed Dice values "
            "are computed in 3D."
        ),
        "projected_error_colors": {
            "green": "ray contains any real voxelwise TP",
            "red": "ray contains FP only and no voxelwise TP",
            "blue": "ray contains FN only and no voxelwise TP",
            "purple": "ray contains both FN and FP at different AP depths but no voxelwise TP",
        },
    }
    if pilot_config is not None:
        manifest["pilot_config"] = str(args.pilot_config.resolve())
        manifest["pilot_config_sha256"] = sha256_file(args.pilot_config)
    if args.mode == VAL200_CATEGORY_MODE:
        manifest.update(
            {
                "organization": "official ReX category, then fixed validation index",
                "category_order": [category_directory_name(code) for code in OFFICIAL_CATEGORY_CODES],
                "category_case_counts": category_case_counts,
                "category_finding_counts": category_finding_counts,
                "category_case_index": str(args.output_dir / "category_case_index.csv"),
                "empty_category_directory": "2f_empty",
                "filename_pattern": "valNNN_<case-stem>.png",
            }
        )
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(figure_paths)} figures to {args.output_dir}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
