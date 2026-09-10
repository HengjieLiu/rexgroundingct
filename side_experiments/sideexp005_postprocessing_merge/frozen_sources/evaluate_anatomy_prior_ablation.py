#!/usr/bin/env python3
"""Evaluate prompt-derived anatomical prior masking for ReX predictions.

This is an ablation script: it does not rerun VoxTell. It loads existing
full-volume predictions, applies several anatomical masks, and recomputes
per-finding metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from evaluate_anatomy_prior_diagnostic import (
    LOBE_FILES,
    geometry_reference,
    infer_column,
    load_prior,
    parse_laterality,
    positive_lobes_from_gt,
)


LOBE_ORDER = ["RUL", "RML", "RLL", "LUL", "LLL"]
LEFT_LOBES = ["LUL", "LLL"]
RIGHT_LOBES = ["RUL", "RML", "RLL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--ct_dir", type=Path, default=None)
    parser.add_argument("--prior_dir", type=Path, required=True)
    parser.add_argument("--findings_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prediction_is_probability", action="store_true")
    parser.add_argument("--dilations_mm", nargs="+", type=int, default=[0, 5, 10, 15])
    parser.add_argument("--scan_id_column", default=None)
    parser.add_argument("--finding_id_column", default=None)
    parser.add_argument("--description_column", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--whole-lung-only",
        action="store_true",
        help=(
            "Evaluate only raw and whole-lung clipping. For 10 mm, reuse the "
            "precomputed lung_dilated_10mm mask instead of recomputing a distance transform."
        ),
    )
    return parser.parse_args()


def case_id_from_name(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def channel_first_mask(path: Path, threshold: float, is_probability: bool) -> tuple[np.ndarray, nib.Nifti1Image]:
    if not path.exists():
        raise FileNotFoundError(path)
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 3:
        arr = data[None]
    elif data.ndim == 4:
        if data.shape[0] <= 64:
            arr = data
        elif data.shape[-1] <= 64:
            arr = np.moveaxis(data, -1, 0)
        else:
            raise ValueError(f"Cannot infer channel axis for {path}: shape={data.shape}")
    else:
        raise ValueError(f"Expected 3D or 4D NIfTI for {path}; got shape={data.shape}")
    mask = arr > threshold if is_probability else arr > 0
    ref = nib.Nifti1Image(np.zeros(mask.shape[1:], dtype=np.uint8), img.affine, img.header)
    return mask.astype(bool, copy=False), ref


def open_mask_image(path: Path) -> tuple[nib.Nifti1Image, nib.Nifti1Image, int, tuple[int, int, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    img = nib.load(str(path))
    shape = tuple(int(x) for x in img.shape)
    if len(shape) == 3:
        n_channels = 1
        spatial_shape = shape
    elif len(shape) == 4:
        if shape[0] <= 64:
            n_channels = shape[0]
            spatial_shape = shape[1:]
        elif shape[-1] <= 64:
            n_channels = shape[-1]
            spatial_shape = shape[:3]
        else:
            raise ValueError(f"Cannot infer channel axis for {path}: shape={shape}")
    else:
        raise ValueError(f"Expected 3D or 4D NIfTI for {path}; got shape={shape}")
    ref = nib.Nifti1Image(np.zeros(spatial_shape, dtype=np.uint8), img.affine, img.header)
    return img, ref, n_channels, spatial_shape


def read_channel_mask(img: nib.Nifti1Image, finding_idx: int, threshold: float, is_probability: bool) -> np.ndarray:
    shape = tuple(int(x) for x in img.shape)
    data = img.dataobj
    if len(shape) == 3:
        if finding_idx != 0:
            raise IndexError(f"finding_idx {finding_idx} out of bounds for 3D image")
        arr = np.asanyarray(data)
    elif len(shape) == 4 and shape[0] <= 64:
        if finding_idx >= shape[0]:
            raise IndexError(f"finding_idx {finding_idx} out of bounds for first-axis channels {shape}")
        arr = np.asanyarray(data[finding_idx, ...])
    elif len(shape) == 4 and shape[-1] <= 64:
        if finding_idx >= shape[-1]:
            raise IndexError(f"finding_idx {finding_idx} out of bounds for last-axis channels {shape}")
        arr = np.asanyarray(data[..., finding_idx])
    else:
        raise ValueError(f"Cannot infer channel axis for image shape={shape}")
    return (arr > threshold if is_probability else arr > 0).astype(bool, copy=False)


def improved_parse_lobes(description: str) -> list[str]:
    """Parse lobe mentions with common compressed radiology phrasing."""
    text = str(description or "").lower()
    lobes: set[str] = set()

    direct_patterns = {
        "RUL": [r"\bright upper lobe\b", r"\bright upper\b", r"\brul\b"],
        "RML": [r"\bright middle lobe\b", r"\bright middle\b", r"\brml\b", r"\bmiddle lobe of the right lung\b"],
        "RLL": [r"\bright lower lobe\b", r"\bright lower\b", r"\brll\b", r"\blower lobe of the right lung\b"],
        "LUL": [r"\bleft upper lobe\b", r"\bleft upper\b", r"\blul\b", r"\blingula\b", r"\blingular\b"],
        "LLL": [r"\bleft lower lobe\b", r"\bleft lower\b", r"\blll\b", r"\blower lobe of the left lung\b"],
    }
    for lobe, patterns in direct_patterns.items():
        if any(re.search(pattern, text) for pattern in patterns):
            lobes.add(lobe)

    if re.search(r"\bboth upper lobes\b|\bupper lobes of both lungs\b|\bbilateral upper lobes\b", text):
        lobes.update(["RUL", "LUL"])
    if re.search(r"\bboth lower lobes\b|\blower lobes of both lungs\b|\bbilateral lower lobes\b", text):
        lobes.update(["RLL", "LLL"])
    if re.search(r"\ball lobes\b|\ball lung lobes\b", text):
        lobes.update(LOBE_ORDER)

    # Phrases such as "right middle and lower lobes" or "left upper and lower lobes".
    if re.search(r"\bright\b.{0,35}\bmiddle and lower lobes\b", text):
        lobes.update(["RML", "RLL"])
    if re.search(r"\bright\b.{0,35}\bupper and lower lobes\b", text):
        lobes.update(["RUL", "RLL"])
    if re.search(r"\bright\b.{0,35}\bupper,? middle,? and lower lobes\b", text):
        lobes.update(RIGHT_LOBES)
    if re.search(r"\bleft\b.{0,35}\bupper and lower lobes\b", text):
        lobes.update(["LUL", "LLL"])

    # Segment/location hints only when laterality is explicit.
    laterality = parse_laterality(description)
    has_lower_hint = re.search(r"\b(basal|posterobasal|anterobasal|laterobasal|lower zone|lung base|basilar)\b", text)
    has_upper_hint = re.search(r"\b(apex|apical|upper zone)\b", text)
    if laterality == "right":
        if has_lower_hint:
            lobes.add("RLL")
        if has_upper_hint:
            lobes.add("RUL")
    elif laterality == "left":
        if has_lower_hint:
            lobes.add("LLL")
        if has_upper_hint:
            lobes.add("LUL")

    return [lobe for lobe in LOBE_ORDER if lobe in lobes]


def union_masks(masks: list[np.ndarray], reference: np.ndarray) -> np.ndarray:
    if not masks:
        return np.zeros(reference.shape, dtype=bool)
    return np.logical_or.reduce(masks)


def voxel_volume_mm3_fast(ref: nib.Nifti1Image) -> float:
    zooms = nib.affines.voxel_sizes(ref.affine)
    if len(zooms) >= 3 and np.all(np.isfinite(zooms[:3])) and np.all(np.asarray(zooms[:3]) > 0):
        return float(np.prod(zooms[:3]))
    return 1.0


def compute_metrics_fast(pred: np.ndarray, gt: np.ndarray, ref: nib.Nifti1Image) -> dict[str, float]:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    tp = int(np.logical_and(pred, gt).sum())
    fp = pred_sum - tp
    fn = gt_sum - tp
    denom = pred_sum + gt_sum
    vv = voxel_volume_mm3_fast(ref)
    return {
        "dice": 1.0 if denom == 0 else float(2 * tp / denom),
        "precision": 0.0 if pred_sum == 0 else float(tp / pred_sum),
        "recall": 0.0 if gt_sum == 0 else float(tp / gt_sum),
        "gt_volume_mm3": float(gt_sum * vv),
        "pred_volume_mm3": float(pred_sum * vv),
        "fp_volume_mm3": float(fp * vv),
        "fn_volume_mm3": float(fn * vv),
        "pred_gt_volume_ratio": 0.0 if gt_sum == 0 else float(pred_sum / gt_sum),
    }


def metrics_from_counts(pred_sum: int, gt_sum: int, tp: int, vv: float) -> dict[str, float]:
    fp = pred_sum - tp
    fn = gt_sum - tp
    denom = pred_sum + gt_sum
    return {
        "dice": 1.0 if denom == 0 else float(2 * tp / denom),
        "precision": 0.0 if pred_sum == 0 else float(tp / pred_sum),
        "recall": 0.0 if gt_sum == 0 else float(tp / gt_sum),
        "gt_volume_mm3": float(gt_sum * vv),
        "pred_volume_mm3": float(pred_sum * vv),
        "fp_volume_mm3": float(fp * vv),
        "fn_volume_mm3": float(fn * vv),
        "pred_gt_volume_ratio": 0.0 if gt_sum == 0 else float(pred_sum / gt_sum),
    }


def count_context(pred: np.ndarray, gt: np.ndarray, ref: nib.Nifti1Image) -> dict[str, Any]:
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    pred_idx = np.flatnonzero(pred_flat)
    gt_idx = np.flatnonzero(gt_flat)
    tp_idx = pred_idx[gt_flat[pred_idx]] if pred_idx.size else pred_idx
    vv = voxel_volume_mm3_fast(ref)
    pred_sum = int(pred_idx.size)
    gt_sum = int(gt_idx.size)
    tp = int(tp_idx.size)
    return {
        "pred_idx": pred_idx,
        "gt_idx": gt_idx,
        "tp_idx": tp_idx,
        "pred_sum": pred_sum,
        "gt_sum": gt_sum,
        "tp": tp,
        "voxel_volume_mm3": vv,
        "baseline_metrics": metrics_from_counts(pred_sum, gt_sum, tp, vv),
    }


def dilate_mask(mask: np.ndarray, ref: nib.Nifti1Image, dilation_mm: int) -> np.ndarray:
    if dilation_mm <= 0 or not mask.any():
        return mask
    zooms = nib.affines.voxel_sizes(ref.affine)
    if len(zooms) < 3 or not np.all(np.isfinite(zooms[:3])) or np.any(np.asarray(zooms[:3]) <= 0):
        sampling = None
    else:
        sampling = tuple(float(x) for x in zooms[:3])
    distance_to_mask = ndimage.distance_transform_edt(~mask, sampling=sampling)
    return mask | (distance_to_mask <= float(dilation_mm))


def prior_masks_for_case(case: str, prior_dir: Path, ref: nib.Nifti1Image, ct_dir: Path | None) -> dict[str, np.ndarray]:
    geom_ref = geometry_reference(case, ref, ct_dir)
    masks = {
        "whole": load_prior(prior_dir, case, "whole_lung", geom_ref),
        "left": load_prior(prior_dir, case, "left_lung", geom_ref),
        "right": load_prior(prior_dir, case, "right_lung", geom_ref),
    }
    for lobe, name in LOBE_FILES.items():
        masks[lobe] = load_prior(prior_dir, case, name, geom_ref)
    return masks


def text_laterality_mask(description: str, priors: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    laterality = parse_laterality(description)
    if laterality == "right":
        return priors["right"], "right"
    if laterality == "left":
        return priors["left"], "left"
    return priors["whole"], laterality


def text_lobe_mask(description: str, priors: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    lobes = improved_parse_lobes(description)
    if lobes:
        return union_masks([priors[lobe] for lobe in lobes], priors["whole"]), ";".join(lobes)
    laterality_mask, laterality = text_laterality_mask(description, priors)
    return laterality_mask, f"fallback_{laterality}"


def gt_positive_lobe_mask(gt: np.ndarray, priors: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    lobe_masks = {lobe: priors[lobe] for lobe in LOBE_ORDER}
    lobes, _, _ = positive_lobes_from_gt(gt, lobe_masks)
    if lobes:
        return union_masks([priors[lobe] for lobe in lobes], priors["whole"]), ";".join(lobes)
    return priors["whole"], "fallback_whole"


def metric_row(
    base: dict[str, Any],
    variant: str,
    dilation_mm: int,
    counts: dict[str, Any],
    prior: np.ndarray,
    prior_voxels: int,
    prior_label: str,
) -> dict[str, Any]:
    prior_flat = prior.ravel()
    pred_idx = counts["pred_idx"]
    gt_idx = counts["gt_idx"]
    tp_idx = counts["tp_idx"]
    pred_prior = int(prior_flat[pred_idx].sum()) if pred_idx.size else 0
    gt_prior = int(prior_flat[gt_idx].sum()) if gt_idx.size else 0
    tp_prior = int(prior_flat[tp_idx].sum()) if tp_idx.size else 0
    metrics = metrics_from_counts(pred_prior, counts["gt_sum"], tp_prior, counts["voxel_volume_mm3"])
    out = {
        **base,
        "variant": variant,
        "dilation_mm": dilation_mm,
        "prior_label": prior_label,
        "prior_voxels": prior_voxels,
        "dice": metrics["dice"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "pred_volume_mm3": metrics["pred_volume_mm3"],
        "fp_volume_mm3": metrics["fp_volume_mm3"],
        "fn_volume_mm3": metrics["fn_volume_mm3"],
        "pred_gt_volume_ratio": metrics["pred_gt_volume_ratio"],
        "delta_dice": metrics["dice"] - base["baseline_dice"],
        "delta_precision": metrics["precision"] - base["baseline_precision"],
        "delta_recall": metrics["recall"] - base["baseline_recall"],
        "removed_pred_fraction": float((counts["pred_sum"] - pred_prior) / max(counts["pred_sum"], 1)),
        "removed_gt_fraction": float((counts["gt_sum"] - gt_prior) / max(counts["gt_sum"], 1)),
        "voxel_volume_mm3": counts["voxel_volume_mm3"],
    }
    return out


def process_case_group(task: tuple[str, pd.DataFrame, dict[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    case, case_df, cols, config = task
    case_rows: list[dict[str, Any]] = []
    try:
        pred_dir = Path(config["pred_dir"])
        gt_dir = Path(config["gt_dir"])
        prior_dir = Path(config["prior_dir"])
        ct_dir = Path(config["ct_dir"]) if config.get("ct_dir") else None
        threshold = float(config["threshold"])
        prediction_is_probability = bool(config["prediction_is_probability"])
        dilations_mm = [int(x) for x in config["dilations_mm"]]

        pred_img, pred_ref, pred_channels, pred_spatial_shape = open_mask_image(pred_dir / case)
        gt_img, _gt_ref, gt_channels, gt_spatial_shape = open_mask_image(gt_dir / case)
        if pred_spatial_shape != gt_spatial_shape:
            raise ValueError(f"pred spatial shape {pred_spatial_shape} != gt spatial shape {gt_spatial_shape}")
        ref = geometry_reference(case, pred_ref, ct_dir)
        if config.get("whole_lung_only"):
            if dilations_mm != [10]:
                raise ValueError(
                    f"--whole-lung-only currently requires --dilations_mm 10, got {dilations_mm}"
                )
            whole_lung_10mm = load_prior(
                prior_dir, case, "lung_dilated_10mm", ref
            )
            priors = None
            all_voxels = np.ones_like(whole_lung_10mm, dtype=bool)
        else:
            priors = prior_masks_for_case(case, prior_dir, pred_ref, ct_dir)
            whole_lung_10mm = None
            all_voxels = np.ones_like(priors["whole"], dtype=bool)
        dilated_prior_cache: dict[tuple[str, str, int], tuple[np.ndarray, int]] = {}

        def cached_prior(variant: str, label: str, prior: np.ndarray, dilation: int) -> tuple[np.ndarray, int]:
            key = (variant, label, int(dilation))
            if key not in dilated_prior_cache:
                cached = prior if dilation == 0 else dilate_mask(prior, ref, dilation)
                dilated_prior_cache[key] = (cached, int(np.count_nonzero(cached)))
            return dilated_prior_cache[key]

        for _, row in case_df.iterrows():
            idx = int(row[cols["finding_idx"]])
            if idx >= pred_channels or idx >= gt_channels:
                raise IndexError(f"{case} finding {idx} out of bounds pred_channels={pred_channels} gt_channels={gt_channels}")
            pred = read_channel_mask(pred_img, idx, threshold, prediction_is_probability)
            gt = read_channel_mask(gt_img, idx, 0.5, False)
            desc = str(row[cols["description"]])
            counts = count_context(pred, gt, ref)
            baseline_metrics = counts["baseline_metrics"]
            base = {
                "case": case,
                "scan_id": case_id_from_name(case),
                "finding_idx": idx,
                "description": desc,
                "split": row.get("split", ""),
                "category_code": row.get("category_code", ""),
                "focality": row.get("focality", ""),
                "baseline_dice": baseline_metrics["dice"],
                "baseline_precision": baseline_metrics["precision"],
                "baseline_recall": baseline_metrics["recall"],
                "text_laterality": parse_laterality(desc),
                "text_lobes": ";".join(improved_parse_lobes(desc)),
            }

            if whole_lung_10mm is not None:
                case_rows.append(
                    metric_row(
                        base,
                        "none",
                        0,
                        counts,
                        all_voxels,
                        int(all_voxels.size),
                        "all_voxels",
                    )
                )
                case_rows.append(
                    metric_row(
                        base,
                        "whole_lung",
                        10,
                        counts,
                        whole_lung_10mm,
                        int(np.count_nonzero(whole_lung_10mm)),
                        "lung_dilated_10mm",
                    )
                )
                continue

            assert priors is not None
            prior_specs = [
                ("none", all_voxels, "all_voxels", [0]),
                ("whole_lung", priors["whole"], "whole_lung", dilations_mm),
            ]
            laterality_prior, laterality_label = text_laterality_mask(desc, priors)
            lobe_prior, lobe_label = text_lobe_mask(desc, priors)
            oracle_prior, oracle_label = gt_positive_lobe_mask(gt, priors)
            prior_specs.extend(
                [
                    ("text_laterality", laterality_prior, laterality_label, dilations_mm),
                    ("text_lobe_or_laterality", lobe_prior, lobe_label, dilations_mm),
                    ("oracle_gt_positive_lobe", oracle_prior, oracle_label, dilations_mm),
                ]
            )
            for variant, prior, label, dilations in prior_specs:
                for dilation in dilations:
                    prior_for_metric, prior_voxels = cached_prior(variant, label, prior, dilation)
                    case_rows.append(metric_row(base, variant, dilation, counts, prior_for_metric, prior_voxels, label))
        return case_rows, None
    except Exception as exc:
        return [], {"case": case, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    findings = pd.read_csv(args.findings_csv)
    if args.splits and "split" in findings.columns:
        findings = findings[findings["split"].astype(str).isin(set(args.splits))].copy()
    if args.limit:
        findings = findings.head(args.limit).copy()
    cols = {
        "case": infer_column(args.findings_csv and findings, args.scan_id_column, ["case", "scan_id", "name", "file", "filename"], "scan id"),
        "finding_idx": infer_column(findings, args.finding_id_column, ["finding_idx", "finding_id", "index", "idx"], "finding id"),
        "description": infer_column(findings, args.description_column, ["finding", "description", "prompt", "text"], "description"),
    }

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    grouped = list(findings.groupby(cols["case"], sort=True))
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
    if args.num_shards > 1:
        grouped = [item for i, item in enumerate(grouped) if i % args.num_shards == args.shard_index]
    config = {
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir),
        "ct_dir": str(args.ct_dir) if args.ct_dir else "",
        "prior_dir": str(args.prior_dir),
        "threshold": args.threshold,
        "prediction_is_probability": args.prediction_is_probability,
        "dilations_mm": args.dilations_mm,
        "whole_lung_only": args.whole_lung_only,
    }
    tasks = [(str(case), case_df.copy(), cols, config) for case, case_df in grouped]
    workers = max(1, int(args.num_workers))
    if workers == 1:
        iterator = (process_case_group(task) for task in tasks)
        for case_rows, failure in tqdm(iterator, total=len(tasks), desc="anatomy prior ablation", unit="case"):
            rows.extend(case_rows)
            if failure:
                failures.append(failure)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_case_group, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="anatomy prior ablation", unit="case"):
                case_rows, failure = fut.result()
                rows.extend(case_rows)
                if failure:
                    failures.append(failure)

    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "per_finding_prior_ablation.csv", index=False)
    pd.DataFrame(failures).to_csv(args.out_dir / "failed_cases.csv", index=False)

    if not result.empty:
        summary = (
            result.groupby(["variant", "dilation_mm"], dropna=False)
            .agg(
                n=("dice", "size"),
                dice_mean=("dice", "mean"),
                dice_median=("dice", "median"),
                precision_mean=("precision", "mean"),
                recall_mean=("recall", "mean"),
                delta_dice_mean=("delta_dice", "mean"),
                delta_dice_median=("delta_dice", "median"),
                removed_pred_fraction_mean=("removed_pred_fraction", "mean"),
                removed_gt_fraction_mean=("removed_gt_fraction", "mean"),
            )
            .reset_index()
            .sort_values(["dice_mean", "variant", "dilation_mm"], ascending=[False, True, True])
        )
        summary.to_csv(args.out_dir / "summary_by_variant.csv", index=False)

        group_summary = (
            result.groupby(["variant", "dilation_mm", "focality"], dropna=False)
            .agg(
                n=("dice", "size"),
                dice_mean=("dice", "mean"),
                delta_dice_mean=("delta_dice", "mean"),
                removed_pred_fraction_mean=("removed_pred_fraction", "mean"),
                removed_gt_fraction_mean=("removed_gt_fraction", "mean"),
            )
            .reset_index()
        )
        group_summary.to_csv(args.out_dir / "summary_by_variant_focality.csv", index=False)

        best_rows = summary.head(20).to_dict(orient="records")
        report = {
            "pred_dir": str(args.pred_dir),
            "prior_dir": str(args.prior_dir),
            "findings_csv": str(args.findings_csv),
            "splits": args.splits,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "n_rows": int(len(result)),
            "n_failures": int(len(failures)),
            "best_by_dice_mean": best_rows,
        }
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        lines = [
            "# Anatomy Prior Ablation",
            "",
            f"- Prediction dir: `{args.pred_dir}`",
            f"- Prior dir: `{args.prior_dir}`",
            f"- Findings: `{args.findings_csv}`",
            f"- Failures: {len(failures)}",
            "",
            "## Summary By Variant",
            "",
            summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
            "",
        ]
        (args.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
