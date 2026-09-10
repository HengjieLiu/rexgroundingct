#!/usr/bin/env python3
"""Quantify whether VoxTell false positives are inside or outside lung priors."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from tqdm import tqdm


DIFFUSE_RE = re.compile(
    r"\b(ground[- ]?glass|ggo|consolidation|opacity|opacities|infiltrate|infiltrates|"
    r"atelectasis|atelectatic|fibrotic|fibrosis|reticulation|reticular|honeycombing|"
    r"emphysema|mosaic attenuation|bronchiectasis|peribronchial thickening|"
    r"pleural effusion|pneumothorax|pleural thickening)\b",
    re.IGNORECASE,
)
FOCAL_RE = re.compile(r"\b(nodule|nodules|mass|lesion|lymph node|calcification|granuloma)\b", re.IGNORECASE)
RIGHT_RE = re.compile(r"\b(right|right lung|right upper|right middle|right lower|rul|rml|rll)\b", re.IGNORECASE)
LEFT_RE = re.compile(r"\b(left|left lung|left upper|left lower|lingula|lingular|lul|lll)\b", re.IGNORECASE)
BILATERAL_RE = re.compile(
    r"\b(bilateral|both lungs|both lung|both lower lobes|both upper lobes|diffuse bilateral)\b",
    re.IGNORECASE,
)

LOBE_FILES = {
    "RUL": "lung_upper_lobe_right",
    "RML": "lung_middle_lobe_right",
    "RLL": "lung_lower_lobe_right",
    "LUL": "lung_upper_lobe_left",
    "LLL": "lung_lower_lobe_left",
}
LEFT_LOBES = ["LUL", "LLL"]
RIGHT_LOBES = ["RUL", "RML", "RLL"]
GT_POSITIVE_LOBE_MIN_FRAC = 0.05
GT_POSITIVE_LOBE_MIN_VOXELS = 50
GT_POSITIVE_LUNG_MIN_FRAC = 0.05
GT_POSITIVE_LUNG_MIN_VOXELS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--ct_dir", type=Path, default=None, help="Optional, currently only used for failure context.")
    parser.add_argument("--prior_dir", type=Path, required=True)
    parser.add_argument("--findings_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=None, help="Optional split names to keep if findings CSV has a split column.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dilation_mm", type=int, choices=[0, 5, 10, 15], default=10)
    parser.add_argument("--prediction_is_probability", action="store_true")
    parser.add_argument("--scan_id_column", default=None)
    parser.add_argument("--finding_id_column", default=None)
    parser.add_argument("--description_column", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N finding rows after split filtering.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel worker processes for per-finding metrics.")
    return parser.parse_args()


def case_id_from_name(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def infer_column(df: pd.DataFrame, requested: str | None, candidates: list[str], label: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"{label} column {requested!r} not found. Columns: {list(df.columns)}")
        return requested
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer {label} column. Columns: {list(df.columns)}")


def classify_finding_type(description: str) -> str:
    text = str(description or "")
    diffuse = bool(DIFFUSE_RE.search(text))
    focal = bool(FOCAL_RE.search(text))
    if diffuse and focal:
        return "mixed"
    if diffuse:
        return "diffuse"
    if focal:
        return "focal"
    return "unknown"


def parse_laterality(description: str) -> str:
    text = str(description or "").lower()
    bilateral = bool(BILATERAL_RE.search(text))
    right = bool(RIGHT_RE.search(text))
    left = bool(LEFT_RE.search(text))
    if bilateral or (right and left):
        return "bilateral"
    if right:
        return "right"
    if left:
        return "left"
    return "unspecified"


def parse_lobes(description: str) -> list[str]:
    text = str(description or "").lower()
    lobes: set[str] = set()
    patterns = {
        "RUL": [r"\bright upper lobe\b", r"\bright upper\b", r"\brul\b"],
        "RML": [r"\bright middle lobe\b", r"\bright middle\b", r"\brml\b"],
        "RLL": [r"\bright lower lobe\b", r"\bright lower\b", r"\brll\b"],
        "LUL": [r"\bleft upper lobe\b", r"\bleft upper\b", r"\blul\b", r"\blingula\b", r"\blingular\b"],
        "LLL": [r"\bleft lower lobe\b", r"\bleft lower\b", r"\blll\b"],
    }
    for lobe, pats in patterns.items():
        if any(re.search(p, text, re.IGNORECASE) for p in pats):
            lobes.add(lobe)
    if re.search(r"\bboth upper lobes\b", text, re.IGNORECASE):
        lobes.update(["RUL", "LUL"])
    if re.search(r"\bboth lower lobes\b", text, re.IGNORECASE):
        lobes.update(["RLL", "LLL"])
    return [lobe for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"] if lobe in lobes]


def load_channel(path: Path, finding_idx: int, threshold: float, is_probability: bool) -> tuple[np.ndarray, nib.Nifti1Image]:
    if not path.exists():
        raise FileNotFoundError(path)
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 3:
        arr = data
    elif data.ndim == 4:
        if data.shape[0] <= 64:
            if finding_idx >= data.shape[0]:
                raise IndexError(f"finding_idx {finding_idx} out of bounds for first-axis channels {data.shape}")
            arr = data[finding_idx]
        elif data.shape[-1] <= 64:
            if finding_idx >= data.shape[-1]:
                raise IndexError(f"finding_idx {finding_idx} out of bounds for last-axis channels {data.shape}")
            arr = data[..., finding_idx]
        else:
            raise ValueError(f"Cannot infer finding/channel axis for {path}: shape={data.shape}")
    else:
        raise ValueError(f"Expected 3D or 4D NIfTI for {path}; got shape {data.shape}")
    mask = arr > threshold if is_probability else arr > 0
    ref = nib.Nifti1Image(np.zeros(mask.shape, dtype=np.uint8), img.affine, img.header)
    return mask.astype(bool), ref


def load_prior(prior_dir: Path, case_name: str, mask_name: str, target: nib.Nifti1Image) -> np.ndarray:
    scan_id = case_id_from_name(case_name)
    path = prior_dir / scan_id / f"{mask_name}.nii.gz"
    if not path.exists():
        fallback = prior_dir / scan_id / "totalseg" / f"{mask_name}.nii.gz"
        if fallback.exists():
            path = fallback
    if not path.exists():
        raise FileNotFoundError(path)
    prior = nib.load(str(path))
    # ReX GT/prediction masks in this workspace often have an identity affine
    # even though their voxel arrays are already in CT index space. If the shape
    # matches, keep index-space alignment instead of resampling by a misleading
    # mask affine.
    if prior.shape == target.shape:
        return np.asanyarray(prior.dataobj) > 0
    if prior.shape != target.shape or not np.allclose(prior.affine, target.affine, atol=1e-3):
        prior = resample_from_to(prior, target, order=0)
    return np.asanyarray(prior.dataobj) > 0


def geometry_reference(case_name: str, mask_ref: nib.Nifti1Image, ct_dir: Path | None) -> nib.Nifti1Image:
    if ct_dir is None:
        return mask_ref
    ct_path = ct_dir / case_name
    if not ct_path.exists():
        return mask_ref
    ct = nib.load(str(ct_path))
    if ct.shape == mask_ref.shape:
        return ct
    return mask_ref


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("Cannot union an empty mask list")
    return np.logical_or.reduce(masks)


def voxel_volume_mm3(ref: nib.Nifti1Image) -> float:
    return float(abs(np.linalg.det(ref.affine[:3, :3])))


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = int(pred.sum()) + int(gt.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / denom)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def is_positive_region(overlap_voxels: int, total_gt_voxels: int, min_frac: float, min_voxels: int) -> bool:
    return safe_div(float(overlap_voxels), float(total_gt_voxels)) >= min_frac or overlap_voxels >= min_voxels


def sorted_lobes(lobes: set[str]) -> list[str]:
    return [lobe for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"] if lobe in lobes]


def lobe_overlap_json(mask: np.ndarray, lobe_masks: dict[str, np.ndarray]) -> dict[str, int]:
    return {lobe: int((mask & lobe_masks[lobe]).sum()) for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"]}


def positive_lobes_from_gt(gt: np.ndarray, lobe_masks: dict[str, np.ndarray]) -> tuple[list[str], dict[str, int], dict[str, float]]:
    total_gt_voxels = int(gt.sum())
    overlap_voxels = lobe_overlap_json(gt, lobe_masks)
    overlap_frac = {lobe: safe_div(voxels, total_gt_voxels) for lobe, voxels in overlap_voxels.items()}
    positives = sorted_lobes(
        {
            lobe
            for lobe, voxels in overlap_voxels.items()
            if is_positive_region(voxels, total_gt_voxels, GT_POSITIVE_LOBE_MIN_FRAC, GT_POSITIVE_LOBE_MIN_VOXELS)
        }
    )
    return positives, overlap_voxels, overlap_frac


def positive_laterality_from_gt(gt: np.ndarray, left_lung: np.ndarray, right_lung: np.ndarray) -> tuple[str, set[str]]:
    total_gt_voxels = int(gt.sum())
    left_voxels = int((gt & left_lung).sum())
    right_voxels = int((gt & right_lung).sum())
    sides: set[str] = set()
    if is_positive_region(left_voxels, total_gt_voxels, GT_POSITIVE_LUNG_MIN_FRAC, GT_POSITIVE_LUNG_MIN_VOXELS):
        sides.add("left")
    if is_positive_region(right_voxels, total_gt_voxels, GT_POSITIVE_LUNG_MIN_FRAC, GT_POSITIVE_LUNG_MIN_VOXELS):
        sides.add("right")
    if sides == {"left", "right"}:
        return "bilateral", sides
    if sides == {"left"}:
        return "left", sides
    if sides == {"right"}:
        return "right", sides
    return "none", sides


def blank_like(mask: np.ndarray) -> np.ndarray:
    return np.zeros(mask.shape, dtype=bool)


def union_or_blank(masks: list[np.ndarray], reference: np.ndarray) -> np.ndarray:
    return union_masks(masks) if masks else blank_like(reference)


def gt_lung_side_mask(gt_positive_sides: set[str], left_lung: np.ndarray, right_lung: np.ndarray, reference: np.ndarray) -> np.ndarray:
    masks = []
    if "left" in gt_positive_sides:
        masks.append(left_lung)
    if "right" in gt_positive_sides:
        masks.append(right_lung)
    return union_or_blank(masks, reference)


def non_gt_lobes_same_side_mask(
    gt_positive_lobes: list[str],
    gt_positive_sides: set[str],
    lobe_masks: dict[str, np.ndarray],
    reference: np.ndarray,
) -> np.ndarray:
    lobes: list[str] = []
    if "left" in gt_positive_sides:
        lobes.extend([lobe for lobe in LEFT_LOBES if lobe not in gt_positive_lobes])
    if "right" in gt_positive_sides:
        lobes.extend([lobe for lobe in RIGHT_LOBES if lobe not in gt_positive_lobes])
    return union_or_blank([lobe_masks[lobe] for lobe in lobes], reference)


def contralateral_lung_mask(
    gt_positive_laterality: str,
    left_lung: np.ndarray,
    right_lung: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    if gt_positive_laterality == "right":
        return left_lung
    if gt_positive_laterality == "left":
        return right_lung
    return blank_like(reference)


def mask_counts(mask: np.ndarray, vv: float, fp_total_voxels: int) -> dict[str, float]:
    voxels = int(mask.sum())
    return {
        "voxels": voxels,
        "mm3": float(voxels * vv),
        "fraction": safe_div(float(voxels), float(fp_total_voxels)),
    }


def compute_metrics(pred: np.ndarray, gt: np.ndarray, ref: nib.Nifti1Image) -> dict[str, float]:
    vv = voxel_volume_mm3(ref)
    tp = pred & gt
    fp = pred & ~gt
    fn = gt & ~pred
    gt_vol = float(gt.sum() * vv)
    pred_vol = float(pred.sum() * vv)
    fp_vol = float(fp.sum() * vv)
    return {
        "dice": dice(pred, gt),
        "gt_volume_mm3": gt_vol,
        "pred_volume_mm3": pred_vol,
        "pred_gt_volume_ratio": safe_div(pred_vol, gt_vol),
        "tp_volume_mm3": float(tp.sum() * vv),
        "fp_volume_mm3": fp_vol,
        "fn_volume_mm3": float(fn.sum() * vv),
        "precision": safe_div(float(tp.sum()), float(pred.sum())),
        "recall": safe_div(float(tp.sum()), float(gt.sum())),
    }


def row_for_finding(row: pd.Series, args: argparse.Namespace, cols: dict[str, str]) -> dict:
    case = str(row[cols["case"]])
    finding_idx = int(row[cols["finding_idx"]])
    desc = str(row[cols["description"]])
    pred, ref = load_channel(args.pred_dir / case, finding_idx, args.threshold, args.prediction_is_probability)
    gt, gt_ref = load_channel(args.gt_dir / case, finding_idx, 0.5, False)
    geom_ref = geometry_reference(case, ref, args.ct_dir)
    if gt.shape != pred.shape or not np.allclose(gt_ref.affine, ref.affine, atol=1e-3):
        if gt.shape != pred.shape:
            gt_img = nib.Nifti1Image(gt.astype(np.uint8), gt_ref.affine, gt_ref.header)
            gt = np.asanyarray(resample_from_to(gt_img, ref, order=0).dataobj) > 0

    whole = load_prior(args.prior_dir, case, "whole_lung", geom_ref)
    left_lung = load_prior(args.prior_dir, case, "left_lung", geom_ref)
    right_lung = load_prior(args.prior_dir, case, "right_lung", geom_ref)
    lobe_masks: dict[str, np.ndarray] = {}
    missing_lobe_masks: list[str] = []
    for lobe, mask_name in LOBE_FILES.items():
        try:
            lobe_masks[lobe] = load_prior(args.prior_dir, case, mask_name, geom_ref)
        except FileNotFoundError:
            lobe_masks[lobe] = blank_like(whole)
            missing_lobe_masks.append(lobe)
    dilated_name = "whole_lung" if args.dilation_mm == 0 else f"lung_dilated_{args.dilation_mm}mm"
    dilated = whole if args.dilation_mm == 0 else load_prior(args.prior_dir, case, dilated_name, geom_ref)
    laterality = parse_laterality(desc)
    mentioned_lobes = parse_lobes(desc)
    if laterality == "right":
        valid_laterality = right_lung
    elif laterality == "left":
        valid_laterality = left_lung
    else:
        valid_laterality = whole
    valid_lobe = union_masks([lobe_masks[lobe] for lobe in mentioned_lobes]) if mentioned_lobes else valid_laterality

    base = compute_metrics(pred, gt, geom_ref)
    vv = voxel_volume_mm3(geom_ref)
    fp = pred & ~gt
    fp_total_voxels = int(fp.sum())
    fp_total = float(fp.sum() * vv)
    pred_total = base["pred_volume_mm3"]
    fp_out_whole_mask = fp & ~whole
    fp_in_whole_mask = fp & whole
    fp_out_whole_voxels = int(fp_out_whole_mask.sum())
    fp_in_whole_voxels = int(fp_in_whole_mask.sum())
    fp_out_whole = float(fp_out_whole_voxels * vv)
    fp_in_whole = float(fp_in_whole_voxels * vv)
    fp_out_dil = float((fp & ~dilated).sum() * vv)
    fp_in_dil = float((fp & dilated).sum() * vv)
    pred_out_laterality = float((pred & whole & ~valid_laterality).sum() * vv)
    fp_out_laterality = float((fp & whole & ~valid_laterality).sum() * vv)
    fp_in_laterality = float((fp & valid_laterality).sum() * vv)
    pred_out_lobe = float((pred & whole & ~valid_lobe).sum() * vv)
    fp_out_lobe = float((fp & whole & ~valid_lobe).sum() * vv)
    fp_in_lobe = float((fp & valid_lobe).sum() * vv)

    gt_positive_lobes, gt_lobe_overlap_voxels, gt_lobe_overlap_fraction = positive_lobes_from_gt(gt, lobe_masks)
    gt_positive_laterality, gt_positive_sides = positive_laterality_from_gt(gt, left_lung, right_lung)
    pred_lobe_overlap_voxels = lobe_overlap_json(pred, lobe_masks)
    fp_lobe_overlap_voxels = lobe_overlap_json(fp, lobe_masks)

    gt_positive_lobe_mask = union_or_blank([lobe_masks[lobe] for lobe in gt_positive_lobes], whole)
    same_side_non_gt_lobe_mask = non_gt_lobes_same_side_mask(gt_positive_lobes, gt_positive_sides, lobe_masks, whole)
    contra_mask = contralateral_lung_mask(gt_positive_laterality, left_lung, right_lung, whole)

    assigned = blank_like(fp)
    bucket_outside = fp_out_whole_mask
    assigned |= bucket_outside
    bucket_inside_gt_lobes = fp & whole & gt_positive_lobe_mask & ~assigned
    assigned |= bucket_inside_gt_lobes
    bucket_contralateral = fp & whole & contra_mask & ~assigned
    assigned |= bucket_contralateral
    bucket_same_lung_diff_lobe = fp & whole & same_side_non_gt_lobe_mask & ~assigned
    assigned |= bucket_same_lung_diff_lobe
    bucket_unknown_inside = fp & whole & ~assigned
    assigned |= bucket_unknown_inside

    inside_gt_lobes = mask_counts(bucket_inside_gt_lobes, vv, fp_total_voxels)
    same_lung_diff_lobe = mask_counts(bucket_same_lung_diff_lobe, vv, fp_total_voxels)
    contralateral = mask_counts(bucket_contralateral, vv, fp_total_voxels)
    unknown_inside = mask_counts(bucket_unknown_inside, vv, fp_total_voxels)
    bucket_sum_voxels = int(assigned.sum())
    bucket_residual_voxels = fp_total_voxels - bucket_sum_voxels
    bucket_sum_fraction = safe_div(float(bucket_sum_voxels), float(fp_total_voxels))
    bucket_residual_fraction = safe_div(float(bucket_residual_voxels), float(fp_total_voxels))
    if abs(bucket_residual_voxels) > 1:
        warnings.warn(
            f"FP bucket residual for {case} finding {finding_idx}: "
            f"{bucket_residual_voxels} voxel(s) of {fp_total_voxels}",
            RuntimeWarning,
        )

    constrained = pred & dilated
    constrained_metrics = compute_metrics(constrained, gt, geom_ref)
    out = {
        "case": case,
        "scan_id": case_id_from_name(case),
        "finding_idx": finding_idx,
        "description": desc,
        "finding_type": classify_finding_type(desc),
        "laterality_label": laterality,
        "mentioned_lobes": ";".join(mentioned_lobes),
        "has_mentioned_lobes": bool(mentioned_lobes),
        "gt_positive_lobes": ";".join(gt_positive_lobes),
        "number_of_gt_positive_lobes": len(gt_positive_lobes),
        "gt_positive_laterality": gt_positive_laterality,
        "missing_lobe_masks": ";".join(missing_lobe_masks),
        "gt_lobe_overlap_voxels_json": json.dumps(gt_lobe_overlap_voxels, sort_keys=True),
        "gt_lobe_overlap_fraction_json": json.dumps(gt_lobe_overlap_fraction, sort_keys=True),
        "pred_lobe_overlap_voxels_json": json.dumps(pred_lobe_overlap_voxels, sort_keys=True),
        "fp_lobe_overlap_voxels_json": json.dumps(fp_lobe_overlap_voxels, sort_keys=True),
        "dilation_mm": args.dilation_mm,
        **base,
        "fp_outside_whole_lung_voxels": fp_out_whole_voxels,
        "fp_outside_whole_lung_mm3": fp_out_whole,
        "fp_inside_whole_lung_voxels": fp_in_whole_voxels,
        "fp_inside_whole_lung_mm3": fp_in_whole,
        "fp_outside_dilated_lung_mm3": fp_out_dil,
        "fp_inside_dilated_lung_mm3": fp_in_dil,
        "fp_outside_whole_lung_fraction": safe_div(fp_out_whole, fp_total),
        "fp_inside_whole_lung_fraction": safe_div(fp_in_whole, fp_total),
        "fp_outside_dilated_lung_fraction": safe_div(fp_out_dil, fp_total),
        "fp_inside_dilated_lung_fraction": safe_div(fp_in_dil, fp_total),
        "pred_outside_whole_lung_fraction": safe_div(float((pred & ~whole).sum() * vv), pred_total),
        "pred_outside_dilated_lung_fraction": safe_div(float((pred & ~dilated).sum() * vv), pred_total),
        "pred_outside_laterality_mm3": pred_out_laterality,
        "pred_outside_laterality_fraction": safe_div(pred_out_laterality, pred_total),
        "fp_outside_laterality_mm3": fp_out_laterality,
        "fp_outside_laterality_fraction": safe_div(fp_out_laterality, fp_total),
        "fp_inside_laterality_mm3": fp_in_laterality,
        "pred_outside_mentioned_lobes_mm3": pred_out_lobe,
        "pred_outside_mentioned_lobes_fraction": safe_div(pred_out_lobe, pred_total),
        "fp_outside_mentioned_lobes_mm3": fp_out_lobe,
        "fp_outside_mentioned_lobes_fraction": safe_div(fp_out_lobe, fp_total),
        "fp_inside_mentioned_lobes_mm3": fp_in_lobe,
        "pred_inside_whole_lung_but_outside_mentioned_lobe_mm3": pred_out_lobe,
        "fp_inside_whole_lung_but_outside_mentioned_lobe_mm3": fp_out_lobe,
        "fp_inside_gt_positive_lobes_voxels": inside_gt_lobes["voxels"],
        "fp_inside_gt_positive_lobes_mm3": inside_gt_lobes["mm3"],
        "fp_inside_gt_positive_lobes_fraction": inside_gt_lobes["fraction"],
        "fp_same_lung_different_lobe_voxels": same_lung_diff_lobe["voxels"],
        "fp_same_lung_different_lobe_mm3": same_lung_diff_lobe["mm3"],
        "fp_same_lung_different_lobe_fraction": same_lung_diff_lobe["fraction"],
        "fp_contralateral_lung_voxels": contralateral["voxels"],
        "fp_contralateral_lung_mm3": contralateral["mm3"],
        "fp_contralateral_lung_fraction": contralateral["fraction"],
        "fp_unknown_inside_lung_voxels": unknown_inside["voxels"],
        "fp_unknown_inside_lung_mm3": unknown_inside["mm3"],
        "fp_unknown_inside_lung_fraction": unknown_inside["fraction"],
        "fp_bucket_sum_voxels": bucket_sum_voxels,
        "fp_bucket_sum_fraction": bucket_sum_fraction,
        "fp_bucket_residual_voxels": bucket_residual_voxels,
        "fp_bucket_residual_fraction": bucket_residual_fraction,
        "dice_constrained": constrained_metrics["dice"],
        "pred_volume_constrained_mm3": constrained_metrics["pred_volume_mm3"],
        "pred_gt_volume_ratio_constrained": constrained_metrics["pred_gt_volume_ratio"],
        "fp_volume_constrained_mm3": constrained_metrics["fp_volume_mm3"],
        "fn_volume_constrained_mm3": constrained_metrics["fn_volume_mm3"],
    }
    out["delta_dice"] = out["dice_constrained"] - out["dice"]
    out["delta_fp_volume"] = out["fp_volume_constrained_mm3"] - out["fp_volume_mm3"]
    out["delta_volume_ratio"] = out["pred_gt_volume_ratio_constrained"] - out["pred_gt_volume_ratio"]
    for col in ["split", "category", "category_code", "focality", "category_name", "finding_ordinal"]:
        if col in row.index:
            out[col] = row[col]
    return out


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    metrics = [
        "dice",
        "precision",
        "recall",
        "dice_constrained",
        "fp_volume_mm3",
        "fp_outside_whole_lung_fraction",
        "fp_inside_whole_lung_fraction",
        "fp_inside_gt_positive_lobes_fraction",
        "fp_same_lung_different_lobe_fraction",
        "fp_contralateral_lung_fraction",
        "fp_unknown_inside_lung_fraction",
        "pred_gt_volume_ratio",
        "pred_gt_volume_ratio_constrained",
        "delta_dice",
        "delta_fp_volume",
        "fp_outside_laterality_fraction",
        "fp_outside_mentioned_lobes_fraction",
        "fp_inside_mentioned_lobes_mm3",
        "pred_outside_laterality_fraction",
        "pred_outside_mentioned_lobes_fraction",
    ]
    rows = []
    for key, sub in df.groupby(by, dropna=False):
        rec = {by: key, "n": len(sub)}
        for m in metrics:
            rec[f"{m}_mean"] = float(sub[m].mean())
            rec[f"{m}_median"] = float(sub[m].median())
        rows.append(rec)
    return pd.DataFrame(rows)


def add_gt_extent_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    labels = ["small", "medium", "large"]
    try:
        out["gt_extent_bin"] = pd.qcut(out["gt_volume_mm3"], q=3, labels=labels, duplicates="drop")
        if out["gt_extent_bin"].isna().all():
            raise ValueError("empty qcut")
        out["gt_extent_bin"] = out["gt_extent_bin"].astype(str)
    except Exception:
        q1 = out["gt_volume_mm3"].quantile(1 / 3)
        q2 = out["gt_volume_mm3"].quantile(2 / 3)
        out["gt_extent_bin"] = np.select(
            [out["gt_volume_mm3"] <= q1, out["gt_volume_mm3"] <= q2],
            ["small", "medium"],
            default="large",
        )
    return out


def summarize_fp_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = add_gt_extent_bins(df)
    value_cols = [
        "dice",
        "precision",
        "recall",
        "pred_gt_volume_ratio",
        "fp_outside_whole_lung_fraction",
        "fp_inside_gt_positive_lobes_fraction",
        "fp_same_lung_different_lobe_fraction",
        "fp_contralateral_lung_fraction",
        "fp_unknown_inside_lung_fraction",
    ]
    group_specs = [("all", None), ("finding_type", "finding_type"), ("laterality_label", "laterality_label")]
    if "category_code" in df.columns:
        group_specs.append(("category_code", "category_code"))
    group_specs.extend(
        [
            ("number_of_gt_positive_lobes", "number_of_gt_positive_lobes"),
            ("gt_positive_laterality", "gt_positive_laterality"),
            ("gt_extent_bin", "gt_extent_bin"),
        ]
    )
    rows: list[dict] = []
    for group_name, col in group_specs:
        grouped = [("all", df)] if col is None else df.groupby(col, dropna=False)
        for group_value, sub in grouped:
            rec = {"group": group_name, "value": str(group_value), "count": int(len(sub))}
            for metric in value_cols:
                rec[f"{metric}_mean"] = float(sub[metric].mean())
                rec[f"{metric}_median"] = float(sub[metric].median())
                rec[f"{metric}_std"] = float(sub[metric].std(ddof=0))
            rows.append(rec)
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = ["diffuse", "focal", "mixed", "unknown"]
    plot_df = df.copy()
    plot_df["finding_type"] = pd.Categorical(plot_df["finding_type"], categories=groups, ordered=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df.boxplot(column="fp_outside_whole_lung_fraction", by="finding_type", ax=ax, grid=False)
    ax.set_title("FP Outside Whole Lung Fraction")
    ax.set_xlabel("")
    ax.set_ylabel("fraction")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_fp_outside_fraction_by_group.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [plot_df.loc[plot_df["finding_type"] == g, "pred_gt_volume_ratio"].dropna() for g in groups]
    data2 = [plot_df.loc[plot_df["finding_type"] == g, "pred_gt_volume_ratio_constrained"].dropna() for g in groups]
    pos = np.arange(len(groups))
    ax.boxplot(data, positions=pos - 0.18, widths=0.28, showfliers=False)
    ax.boxplot(data2, positions=pos + 0.18, widths=0.28, showfliers=False)
    ax.set_xticks(pos, groups)
    ax.set_ylabel("pred / GT volume ratio")
    ax.legend(["before", "after"], loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_volume_ratio_before_after_by_group.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for g, sub in plot_df.groupby("finding_type", observed=False):
        ax.scatter(sub["fp_inside_whole_lung_mm3"], sub["fp_outside_whole_lung_mm3"], s=14, alpha=0.65, label=str(g))
    ax.set_xlabel("FP inside whole lung (mm3)")
    ax.set_ylabel("FP outside whole lung (mm3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_fp_inside_vs_outside.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(df["fp_outside_whole_lung_fraction"], df["delta_dice"], s=14, alpha=0.65)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("FP outside whole lung fraction")
    ax.set_ylabel("Delta Dice after lung constraint")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_dice_delta_vs_outside_fp_fraction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df.boxplot(column="fp_outside_laterality_fraction", by="finding_type", ax=ax, grid=False)
    ax.set_title("FP Outside Prompt Laterality Fraction")
    ax.set_xlabel("")
    ax.set_ylabel("fraction")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_fp_outside_laterality_fraction_by_group.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df.boxplot(column="fp_outside_mentioned_lobes_fraction", by="finding_type", ax=ax, grid=False)
    ax.set_title("FP Outside Mentioned Lobe Region Fraction")
    ax.set_xlabel("")
    ax.set_ylabel("fraction")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_fp_outside_lobe_fraction_by_group.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for g, sub in plot_df.groupby("finding_type", observed=False):
        ax.scatter(
            sub["fp_inside_whole_lung_mm3"],
            sub["fp_inside_whole_lung_but_outside_mentioned_lobe_mm3"],
            s=14,
            alpha=0.65,
            label=str(g),
        )
    ax.set_xlabel("FP inside whole lung (mm3)")
    ax.set_ylabel("FP inside lung but outside mentioned lobe/side (mm3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_inside_lung_fp_vs_lobe_violation.png", dpi=180)
    plt.close(fig)

    bucket_cols = [
        "fp_outside_whole_lung_fraction",
        "fp_inside_gt_positive_lobes_fraction",
        "fp_same_lung_different_lobe_fraction",
        "fp_contralateral_lung_fraction",
        "fp_unknown_inside_lung_fraction",
    ]
    bucket_labels = ["outside lung", "GT+ lobe", "same lung diff lobe", "contralateral", "unknown inside"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([df[col].dropna() for col in bucket_cols], showfliers=False)
    ax.set_xticklabels(bucket_labels, rotation=20, ha="right")
    ax.set_ylabel("fraction of FP")
    ax.set_title("GT-Derived FP Bucket Fractions")
    fig.tight_layout()
    fig.savefig(out_dir / "fp_gtderived_bucket_fraction_boxplot.png", dpi=180)
    plt.close(fig)

    means = plot_df.groupby("finding_type", observed=False)[bucket_cols].mean().reindex(groups)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(groups))
    width = 0.16
    for idx, col in enumerate(bucket_cols):
        ax.bar(x + (idx - 2) * width, means[col].fillna(0), width=width, label=bucket_labels[idx])
    ax.set_xticks(x, groups)
    ax.set_ylabel("mean fraction of FP")
    ax.set_title("GT-Derived FP Buckets By Finding Type")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fp_gtderived_bucket_fraction_by_finding_type.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(df["fp_inside_gt_positive_lobes_fraction"], df["dice"], s=14, alpha=0.65)
    ax.set_xlabel("FP inside GT-positive lobes fraction")
    ax.set_ylabel("Dice")
    fig.tight_layout()
    fig.savefig(out_dir / "dice_vs_fp_inside_gt_positive_lobes_fraction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(df["fp_inside_gt_positive_lobes_fraction"], df["pred_gt_volume_ratio"], s=14, alpha=0.65)
    ax.set_xlabel("FP inside GT-positive lobes fraction")
    ax.set_ylabel("pred / GT volume ratio")
    ax.set_yscale("symlog", linthresh=1)
    fig.tight_layout()
    fig.savefig(out_dir / "pred_gt_volume_ratio_vs_fp_inside_gt_positive_lobes_fraction.png", dpi=180)
    plt.close(fig)


def write_failures(path: Path, failures: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "finding_idx", "error"])
        writer.writeheader()
        writer.writerows(failures)


def print_recommendation(summary: pd.DataFrame) -> None:
    diffuse = summary[summary["finding_type"] == "diffuse"]
    if diffuse.empty:
        print("No diffuse findings in summary; cannot make diffuse-specific recommendation.")
        return
    row = diffuse.iloc[0]
    outside = row["fp_outside_whole_lung_fraction_median"]
    lobe_outside = row.get("fp_outside_mentioned_lobes_fraction_median", 0.0)
    inside_lobe = row.get("fp_inside_mentioned_lobes_mm3_median", 0.0)
    print("\nRecommendation:")
    if outside > 0.25:
        print("- Over-segmentation often leaves the lung; broad anatomical prior is useful.")
    elif outside <= 0.25 and lobe_outside > 0.25:
        print(
            "- Over-segmentation is mostly intra-lung but spreads beyond the described side/lobe; "
            "lobe-level or text-derived anatomical prior may be useful."
        )
    elif inside_lobe > 0:
        print(
            "- Over-segmentation is mostly inside the correct anatomical region; whole-lung/lobe priors alone may not solve it. "
            "Low-res coarse-to-fine finding-specific extent prediction should be prioritized."
        )
    else:
        print("- No dominant diffuse anatomy-prior pattern by these thresholds; inspect worst-case visualizations.")


def dominant_bucket(row: pd.Series) -> str:
    buckets = {
        "outside whole lung": row["fp_outside_whole_lung_fraction_median"],
        "inside GT-positive lobes": row["fp_inside_gt_positive_lobes_fraction_median"],
        "same lung different lobe": row["fp_same_lung_different_lobe_fraction_median"],
        "contralateral lung": row["fp_contralateral_lung_fraction_median"],
        "unknown inside lung": row["fp_unknown_inside_lung_fraction_median"],
    }
    return max(buckets, key=buckets.get)


def interpretation_from_bucket_summary(summary: pd.DataFrame) -> str:
    row = summary[(summary["group"] == "all") & (summary["value"] == "all")].iloc[0]
    dom = dominant_bucket(row)
    if dom == "inside GT-positive lobes":
        return (
            "FP is dominated by voxels inside GT-positive lobes. A lobe prior alone will probably not solve "
            "over-segmentation; prioritize same-lobe extent control, diffuse boundary ambiguity, loss, or prompt strategy."
        )
    if dom == "same lung different lobe":
        return "FP is substantially in the same lung but different lobes. A lobe-level prior may help."
    if dom == "contralateral lung":
        return "FP is substantially contralateral. A laterality prior may help."
    if dom == "outside whole lung":
        return "FP is substantially outside whole lung. A whole-lung hard or soft mask may help."
    return "FP is substantially in unknown inside-lung regions. Inspect lobe mask coverage and hilum/airway regions."


def format_bucket_lines(summary: pd.DataFrame, group: str, value: str) -> list[str]:
    sub = summary[(summary["group"] == group) & (summary["value"] == value)]
    if sub.empty:
        return []
    row = sub.iloc[0]
    return [
        f"- {value}:",
        f"  - median FP outside whole lung: {row['fp_outside_whole_lung_fraction_median']:.4f}",
        f"  - median FP inside GT-positive lobes: {row['fp_inside_gt_positive_lobes_fraction_median']:.4f}",
        f"  - median FP same lung different lobe: {row['fp_same_lung_different_lobe_fraction_median']:.4f}",
        f"  - median FP contralateral lung: {row['fp_contralateral_lung_fraction_median']:.4f}",
        f"  - median FP unknown inside lung: {row['fp_unknown_inside_lung_fraction_median']:.4f}",
    ]


def write_report(metrics: pd.DataFrame, bucket_summary: pd.DataFrame, out_path: Path) -> None:
    missing_lobe_rows = metrics[metrics["missing_lobe_masks"].fillna("").astype(str).str.len() > 0]
    no_gt_lobe = metrics[metrics["number_of_gt_positive_lobes"] == 0]
    no_text_lobe = metrics[~metrics["has_mentioned_lobes"].astype(bool)]
    lines = [
        "# Anatomy Prior Diagnostic Report",
        "",
        "## GT-derived FP anatomical decomposition",
        "",
        "FP buckets are mutually exclusive and ordered as outside whole lung, inside GT-positive lobes, "
        "contralateral lung, same lung different lobe, then unknown inside lung.",
        "",
        "### Overall",
        *format_bucket_lines(bucket_summary, "all", "all"),
        "",
        "### Diffuse vs Focal",
    ]
    for value in ["diffuse", "focal", "mixed", "unknown"]:
        lines.extend(format_bucket_lines(bucket_summary, "finding_type", value))
    lines.extend(
        [
            "",
            "### QA Counts",
            f"- finding rows: {len(metrics)}",
            f"- unique cases: {metrics['case'].nunique()}",
            f"- rows with missing lobe masks: {len(missing_lobe_rows)}",
            f"- cases with missing lobe masks: {missing_lobe_rows['case'].nunique()}",
            f"- rows where no GT-positive lobe could be determined: {len(no_gt_lobe)}",
            f"- cases where no GT-positive lobe could be determined: {no_gt_lobe['case'].nunique()}",
            f"- rows where text-mentioned lobe was empty: {len(no_text_lobe)}",
            f"- cases where text-mentioned lobe was empty: {no_text_lobe['case'].nunique()}",
            f"- max FP bucket residual fraction: {metrics['fp_bucket_residual_fraction'].abs().max():.6f}",
            "",
            "### Interpretation",
            interpretation_from_bucket_summary(bucket_summary),
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def sanity_checks() -> None:
    a = np.array([1, 1, 0], dtype=bool)
    b = np.array([1, 0, 1], dtype=bool)
    assert abs(dice(a, b) - 0.5) < 1e-8
    affine = np.diag([2, 3, 4, 1])
    ref = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine)
    assert abs(voxel_volume_mm3(ref) - 24.0) < 1e-8


def process_finding_item(item: tuple[int, dict], args: argparse.Namespace, cols: dict[str, str]) -> tuple[int, dict | None, dict | None]:
    idx, row_dict = item
    row = pd.Series(row_dict)
    try:
        return idx, row_for_finding(row, args, cols), None
    except Exception as exc:
        case = str(row[cols["case"]]) if cols["case"] in row.index else ""
        finding_idx = row[cols["finding_idx"]] if cols["finding_idx"] in row.index else ""
        return idx, None, {"case": case, "finding_idx": finding_idx, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        gc.collect()


def main() -> int:
    sanity_checks()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.findings_csv)
    if args.splits is not None and "split" in df.columns:
        df = df[df["split"].astype(str).isin(set(args.splits))].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()
    cols = {
        "case": infer_column(df, args.scan_id_column, ["case", "scan_id", "name", "file", "filename"], "scan id"),
        "finding_idx": infer_column(df, args.finding_id_column, ["finding_idx", "finding_id", "channel", "label_idx"], "finding id"),
        "description": infer_column(df, args.description_column, ["finding", "prompt", "description", "text"], "description"),
    }

    rows_with_idx: list[tuple[int, dict]] = []
    failures: list[dict] = []
    items = list(enumerate(df.to_dict(orient="records")))
    if args.num_workers <= 1:
        for item in tqdm(items, total=len(items), desc="Anatomy diagnostic", unit="finding"):
            idx, result, failure = process_finding_item(item, args, cols)
            if result is not None:
                rows_with_idx.append((idx, result))
            if failure is not None:
                failures.append(failure)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_finding_item, item, args, cols) for item in items]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Anatomy diagnostic", unit="finding"):
                idx, result, failure = future.result()
                if result is not None:
                    rows_with_idx.append((idx, result))
                if failure is not None:
                    failures.append(failure)

    rows = [row for _, row in sorted(rows_with_idx, key=lambda x: x[0])]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "per_finding_metrics.csv", index=False)
    write_failures(args.out_dir / "failed_cases.csv", failures)
    if metrics.empty:
        print(f"No successful findings. Failures written to {args.out_dir / 'failed_cases.csv'}")
        return 1

    summary = summarize(metrics, "finding_type")
    summary.to_csv(args.out_dir / "summary_by_group.csv", index=False)
    bucket_summary = summarize_fp_buckets(metrics)
    bucket_summary.to_csv(args.out_dir / "summary_fp_bucket_by_group.csv", index=False)
    summarize(metrics, "laterality_label").to_csv(args.out_dir / "summary_by_laterality.csv", index=False)
    lobe_summary = metrics.copy()
    lobe_summary["lobe_mentioned_label"] = np.where(
        lobe_summary["mentioned_lobes"].astype(str).str.len() > 0,
        lobe_summary["mentioned_lobes"].astype(str),
        "no_lobe_mentioned",
    )
    summarize(lobe_summary, "lobe_mentioned_label").to_csv(args.out_dir / "summary_by_lobe_mentioned.csv", index=False)
    cat_col = "category_code" if "category_code" in metrics.columns else ("category" if "category" in metrics.columns else None)
    if cat_col:
        summarize(metrics, cat_col).to_csv(args.out_dir / "summary_by_category_if_available.csv", index=False)

    metrics.sort_values(
        ["fp_outside_whole_lung_fraction", "fp_outside_dilated_lung_fraction"], ascending=False
    ).head(50).to_csv(args.out_dir / "worst_cases_outside_lung.csv", index=False)
    metrics.sort_values(["fp_inside_whole_lung_mm3", "pred_gt_volume_ratio"], ascending=False).head(50).to_csv(
        args.out_dir / "worst_cases_inside_lung.csv", index=False
    )
    metrics.sort_values(["fp_outside_laterality_fraction", "fp_outside_laterality_mm3"], ascending=False).head(50).to_csv(
        args.out_dir / "worst_cases_laterality_violation.csv", index=False
    )
    metrics.sort_values(["fp_outside_mentioned_lobes_fraction", "fp_outside_mentioned_lobes_mm3"], ascending=False).head(50).to_csv(
        args.out_dir / "worst_cases_lobe_violation.csv", index=False
    )
    make_plots(metrics, args.out_dir / "plots")
    write_report(metrics, bucket_summary, args.out_dir / "report.md")
    print(f"Saved per-finding metrics: {args.out_dir / 'per_finding_metrics.csv'}")
    print(f"Saved GT-derived FP bucket summary: {args.out_dir / 'summary_fp_bucket_by_group.csv'}")
    print(f"Saved report: {args.out_dir / 'report.md'}")
    print(f"Failures: {len(failures)} -> {args.out_dir / 'failed_cases.csv'}")
    print(summary.to_string(index=False))
    print_recommendation(summary)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
