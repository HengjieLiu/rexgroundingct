#!/usr/bin/env python3
"""Shardable CPU audit of deterministic fine spatial-modifier proxy ROIs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from evaluate_anatomy_prior_ablation import channel_first_mask
from evaluate_anatomy_prior_diagnostic import LOBE_FILES, geometry_reference, load_prior


LOBE_ORDER = ("LUL", "LLL", "RUL", "RML", "RLL")
LEFT_LOBES = ("LUL", "LLL")
RIGHT_LOBES = ("RUL", "RML", "RLL")
HIT_THRESHOLD = 0.1
POLICIES = ("raw", "whole10", "semantic_v1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--ct-dir", type=Path, required=True)
    p.add_argument("--prior-dir", type=Path, required=True)
    p.add_argument("--modifier-assignments", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    return p.parse_args()


def split_lobes(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [item for item in str(value).replace(",", ";").split(";") if item in LOBE_ORDER]


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pn, gn = int(np.count_nonzero(pred)), int(np.count_nonzero(gt))
    return 1.0 if pn + gn == 0 else 2.0 * int(np.count_nonzero(pred & gt)) / (pn + gn)


def dilate_mm(mask: np.ndarray, sampling: tuple[float, float, float], mm: float) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    return ndimage.distance_transform_edt(~mask, sampling=sampling) <= float(mm)


def world_axis_info(affine: np.ndarray, shape: tuple[int, int, int]) -> dict[int, tuple[int, np.ndarray, float]]:
    """Map RAS world axes to voxel axes; reject unverified oblique geometry."""
    result: dict[int, tuple[int, np.ndarray, float]] = {}
    for world_axis in range(3):
        coefficients = np.asarray(affine[world_axis, :3], dtype=float)
        voxel_axis = int(np.argmax(np.abs(coefficients)))
        dominant = abs(float(coefficients[voxel_axis]))
        off_axis = float(np.max(np.abs(np.delete(coefficients, voxel_axis))))
        ratio = off_axis / dominant if dominant > 0 else float("inf")
        if dominant == 0 or ratio > 1e-4:
            raise ValueError(
                f"Oblique/unresolved affine world axis {world_axis}: coefficients={coefficients.tolist()} ratio={ratio}"
            )
        values = affine[world_axis, voxel_axis] * np.arange(shape[voxel_axis], dtype=np.float32) + affine[world_axis, 3]
        result[world_axis] = (voxel_axis, values.astype(np.float32), ratio)
    if len({value[0] for value in result.values()}) != 3:
        raise ValueError(f"Affine does not provide a one-to-one RAS/voxel axis map: {result}")
    return result


def axis_condition(shape: tuple[int, int, int], voxel_axis: int, values: np.ndarray, threshold: float, high: bool) -> np.ndarray:
    view_shape = [1, 1, 1]
    view_shape[voxel_axis] = shape[voxel_axis]
    line = values.reshape(view_shape)
    return np.broadcast_to(line >= threshold if high else line <= threshold, shape)


def axis_extent(mask: np.ndarray, voxel_axis: int, values: np.ndarray) -> tuple[float, float]:
    reduce_axes = tuple(axis for axis in range(3) if axis != voxel_axis)
    present = np.any(mask, axis=reduce_axes)
    selected = values[present]
    if selected.size == 0:
        return float("nan"), float("nan")
    return float(selected.min()), float(selected.max())


def fraction_region(
    anatomical_base: np.ndarray,
    container: np.ndarray,
    axis: tuple[int, np.ndarray, float],
    high: bool,
    fraction: float,
    margin_mm: float,
) -> np.ndarray:
    voxel_axis, values, _ = axis
    low, upper = axis_extent(anatomical_base, voxel_axis, values)
    if not np.isfinite(low) or upper <= low:
        return np.zeros_like(container)
    if high:
        threshold = upper - fraction * (upper - low) - margin_mm
    else:
        threshold = low + fraction * (upper - low) + margin_mm
    return container & axis_condition(container.shape, voxel_axis, values, threshold, high)


def relevant_sides(side: str | None, lobes: list[str]) -> list[str]:
    if side in {"left", "right"}:
        return [side]
    inferred = sorted({"left" if lobe in LEFT_LOBES else "right" for lobe in lobes})
    return inferred or ["left", "right"]


def medial_lateral_region(
    priors: dict[str, np.ndarray],
    container: np.ndarray,
    x_axis: tuple[int, np.ndarray, float],
    sides: list[str],
    medial: bool,
    fraction: float = 0.5,
    margin_mm: float = 15.0,
) -> np.ndarray:
    voxel_axis, values, _ = x_axis
    abs_values = np.abs(values)
    output = np.zeros_like(container)
    for side in sides:
        base = priors[side]
        low, high = axis_extent(base, voxel_axis, abs_values)
        if not np.isfinite(low) or high <= low:
            continue
        if medial:
            threshold = low + fraction * (high - low) + margin_mm
            condition = axis_condition(container.shape, voxel_axis, abs_values, threshold, False)
        else:
            threshold = high - fraction * (high - low) - margin_mm
            condition = axis_condition(container.shape, voxel_axis, abs_values, threshold, True)
        output |= container & base & condition
    return output


def build_fissure_proxy(priors: dict[str, np.ndarray], sampling: tuple[float, float, float]) -> np.ndarray:
    proximity_count = np.zeros_like(priors["whole"], dtype=np.uint8)
    for lobe in LOBE_ORDER:
        proximity_count += (ndimage.distance_transform_edt(~priors[lobe], sampling=sampling) <= 10.0).astype(np.uint8)
    return (proximity_count >= 2) & priors["whole10"]


def build_proxy(
    cluster: str,
    semantic_roi: np.ndarray,
    strict_base: np.ndarray,
    priors: dict[str, np.ndarray],
    axes: dict[int, tuple[int, np.ndarray, float]],
    sampling: tuple[float, float, float],
    sides: list[str],
    case_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    if cluster in {"paraspinal_or_costovertebral", "retrocardiac"}:
        return None, "none", {}
    if cluster in {"apical", "superior_within_base"}:
        return fraction_region(strict_base, semantic_roi, axes[2], True, 0.50, 15.0), "superior_half_plus_15mm", {"fraction": 0.50, "margin_mm": 15}
    if cluster in {"basal", "inferior_within_base"}:
        return fraction_region(strict_base, semantic_roi, axes[2], False, 0.50, 15.0), "inferior_half_plus_15mm", {"fraction": 0.50, "margin_mm": 15}
    if cluster == "diaphragmatic":
        return fraction_region(strict_base, semantic_roi, axes[2], False, 0.35, 15.0), "inferior_35pct_plus_15mm", {"fraction": 0.35, "margin_mm": 15}
    if cluster == "anterior":
        return fraction_region(strict_base, semantic_roi, axes[1], True, 0.50, 15.0), "anterior_half_plus_15mm", {"fraction": 0.50, "margin_mm": 15}
    if cluster in {"posterior", "dependent"}:
        name = "posterior_half_plus_15mm_supine" if cluster == "dependent" else "posterior_half_plus_15mm"
        return fraction_region(strict_base, semantic_roi, axes[1], False, 0.50, 15.0), name, {"fraction": 0.50, "margin_mm": 15, "position_assumption": "supine" if cluster == "dependent" else "none"}
    if cluster in {"medial", "paramediastinal"}:
        fraction = 0.35 if cluster == "paramediastinal" else 0.50
        margin = 10.0 if cluster == "paramediastinal" else 15.0
        return medial_lateral_region(priors, semantic_roi, axes[0], sides, True, fraction, margin), f"medial_{int(fraction*100)}pct_plus_{int(margin)}mm", {"fraction": fraction, "margin_mm": margin}
    if cluster == "lateral":
        return medial_lateral_region(priors, semantic_roi, axes[0], sides, False), "lateral_half_plus_15mm", {"fraction": 0.50, "margin_mm": 15}
    if cluster == "subpleural_peripheral":
        if "boundary_shell" not in case_cache:
            inside_distance = ndimage.distance_transform_edt(priors["whole"], sampling=sampling)
            outside_distance = ndimage.distance_transform_edt(~priors["whole"], sampling=sampling)
            case_cache["boundary_shell"] = (
                (priors["whole"] & (inside_distance <= 30.0))
                | (~priors["whole"] & (outside_distance <= 10.0))
            )
        return semantic_roi & case_cache["boundary_shell"], "lung_boundary_inside30_outside10mm", {"inside_shell_mm": 30, "outside_margin_mm": 10}
    if cluster == "central":
        if "central_core" not in case_cache:
            case_cache["central_core"] = priors["whole"] & (ndimage.distance_transform_edt(priors["whole"], sampling=sampling) > 20.0)
        return semantic_roi & case_cache["central_core"], "lung_interior_over_20mm", {"minimum_boundary_distance_mm": 20}
    if cluster == "fissural":
        if "fissure_proxy" not in case_cache:
            case_cache["fissure_proxy"] = build_fissure_proxy(priors, sampling)
        return semantic_roi & case_cache["fissure_proxy"], "within10mm_of_two_lobes", {"lobe_proximity_mm": 10, "minimum_lobes": 2}
    if cluster == "lingular":
        if "lingula_proxy" not in case_cache:
            lul10 = dilate_mm(priors["LUL"], sampling, 10.0)
            inferior = fraction_region(priors["LUL"], lul10, axes[2], False, 0.65, 10.0)
            anterior = fraction_region(priors["LUL"], lul10, axes[1], True, 0.65, 10.0)
            case_cache["lingula_proxy"] = inferior & anterior
        return semantic_roi & case_cache["lingula_proxy"], "inferior_anterior_65pct_LUL_plus10mm", {"inferior_fraction": 0.65, "anterior_fraction": 0.65, "margin_mm": 10}
    raise KeyError(cluster)


def process_case(case: str, findings: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pred, pred_ref = channel_first_mask(args.pred_dir / case, 0.5, False)
    gt, _ = channel_first_mask(args.gt_dir / case, 0.5, False)
    if pred.shape != gt.shape:
        raise ValueError(f"{case}: pred {pred.shape} != GT {gt.shape}")
    ref = geometry_reference(case, pred_ref, args.ct_dir)
    spatial_shape = tuple(int(x) for x in pred.shape[1:])
    sampling = tuple(float(x) for x in nib.affines.voxel_sizes(ref.affine)[:3])
    voxel_volume = float(abs(np.linalg.det(ref.affine[:3, :3])))
    axes = world_axis_info(ref.affine, spatial_shape)
    priors = {
        "whole": load_prior(args.prior_dir, case, "whole_lung", ref),
        "whole10": load_prior(args.prior_dir, case, "lung_dilated_10mm", ref),
    }
    for lobe, filename in LOBE_FILES.items():
        priors[lobe] = load_prior(args.prior_dir, case, filename, ref)
    priors["left"] = priors["LUL"] | priors["LLL"]
    priors["right"] = priors["RUL"] | priors["RML"] | priors["RLL"]
    for name, mask in priors.items():
        if mask.shape != spatial_shape:
            raise ValueError(f"{case}: {name} shape {mask.shape} != {spatial_shape}")

    roi_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray, str, str, int]] = {}
    proxy_cache: dict[tuple[str, tuple[str, ...], str, str], tuple[np.ndarray | None, str, dict[str, Any]]] = {}
    case_cache: dict[str, np.ndarray] = {}
    gt_rows, fp_rows = [], []
    for finding in findings:
        finding_id = int(finding["finding_id"])
        if finding_id >= pred.shape[0]:
            raise IndexError(f"{case}: finding {finding_id} outside prediction channels")
        group = str(finding["parser_group"])
        side = None if pd.isna(finding.get("parsed_side")) else str(finding.get("parsed_side"))
        lobes = split_lobes(finding.get("parsed_lobes"))
        roi_key = (group, side or "", *sorted(lobes))
        if roi_key not in roi_cache:
            exact_base = np.logical_or.reduce([priors[lobe] for lobe in lobes]) if lobes else None
            if group == "high_conf_exact_lobe" and exact_base is not None:
                strict_base = exact_base
                semantic_roi = dilate_mm(strict_base, sampling, 15.0)
                kind, label, dilation = "exact_lobe", ";".join(lobes), 15
            elif group in {"high_conf_exact_lobe", "ambiguous_lobe", "laterality_only"} and side in {"left", "right"}:
                strict_base = priors[side]
                semantic_roi = dilate_mm(strict_base, sampling, 15.0)
                kind, label, dilation = "side_lung", side, 15
            else:
                strict_base = priors["whole"]
                semantic_roi = priors["whole10"]
                kind, label, dilation = "whole_lung", "whole", 10
            roi_cache[roi_key] = (strict_base, semantic_roi, kind, label, dilation)
        strict_base, semantic_roi, base_kind, base_label, base_dilation = roi_cache[roi_key]
        cluster = str(finding["modifier_cluster"])
        sides = relevant_sides(side, lobes)
        proxy_key = (cluster, roi_key, ";".join(sides), base_kind)
        if proxy_key not in proxy_cache:
            proxy_cache[proxy_key] = build_proxy(
                cluster, semantic_roi, strict_base, priors, axes, sampling, sides, case_cache
            )
        proxy, proxy_name, parameters = proxy_cache[proxy_key]
        if proxy is None:
            continue
        gt_mask = gt[finding_id]
        raw = pred[finding_id]
        gt_n = int(np.count_nonzero(gt_mask))
        proxy_n = int(np.count_nonzero(proxy))
        base_n = int(np.count_nonzero(semantic_roi))
        gt_inside_n = int(np.count_nonzero(gt_mask & proxy))
        if gt_n:
            center = ndimage.center_of_mass(gt_mask)
            center_index = tuple(min(max(int(round(x)), 0), spatial_shape[i] - 1) for i, x in enumerate(center))
            centroid_inside = bool(proxy[center_index])
        else:
            centroid_inside = False
        common = {
            "case_id": case, "finding_id": finding_id,
            "finding_text": finding["finding_text"], "parser_group": group,
            "parsed_side": side, "parsed_lobes": ";".join(lobes),
            "modifier_cluster": cluster, "canonical_name": finding["canonical_name"],
            "matched_raw_phrases": finding["matched_raw_phrases"],
            "base_roi_type": base_kind, "base_roi_label": base_label,
            "base_roi_dilation_mm": base_dilation, "proxy_roi_name": proxy_name,
            "proxy_parameters": json.dumps(parameters, sort_keys=True),
            "roi_selection_uses_gt": False, "voxel_volume_mm3": voxel_volume,
        }
        gt_rows.append({
            **common,
            "gt_volume_total": gt_n * voxel_volume,
            "gt_volume_inside_proxy_roi": gt_inside_n * voxel_volume,
            "gt_retention_inside_proxy_roi": safe_ratio(gt_inside_n, gt_n),
            "gt_centroid_inside_proxy_roi": centroid_inside if gt_n else float("nan"),
            "gt_any_overlap_with_proxy_roi": gt_inside_n > 0,
            "gt_majority_inside_proxy_roi": safe_ratio(gt_inside_n, gt_n) > 0.5 if gt_n else float("nan"),
            "proxy_roi_volume": proxy_n * voxel_volume,
            "base_roi_volume": base_n * voxel_volume,
            "proxy_roi_fraction_of_base_roi": safe_ratio(proxy_n, base_n),
            "no_gt": gt_n == 0,
        })
        policies = {"raw": raw, "whole10": raw & priors["whole10"], "semantic_v1": raw & semantic_roi}
        for policy, policy_pred in policies.items():
            fp = policy_pred & ~gt_mask
            tp = policy_pred & gt_mask
            clipped = policy_pred & proxy
            pred_n = int(np.count_nonzero(policy_pred))
            fp_n = int(np.count_nonzero(fp))
            tp_n = int(np.count_nonzero(tp))
            outside_pred_n = int(np.count_nonzero(policy_pred & ~proxy))
            outside_fp_n = int(np.count_nonzero(fp & ~proxy))
            tp_inside_n = int(np.count_nonzero(tp & proxy))
            clipped_n = int(np.count_nonzero(clipped))
            before_dice, after_dice = dice_score(policy_pred, gt_mask), dice_score(clipped, gt_mask)
            fp_rows.append({
                **common, "policy": policy,
                "prediction_volume": pred_n * voxel_volume,
                "fp_volume_total": fp_n * voxel_volume,
                "tp_volume_total": tp_n * voxel_volume,
                "fp_volume_outside_proxy_roi": outside_fp_n * voxel_volume,
                "prediction_volume_outside_proxy_roi": outside_pred_n * voxel_volume,
                "fp_outside_proxy_roi_ratio": safe_ratio(outside_fp_n, fp_n),
                "prediction_outside_proxy_roi_ratio": safe_ratio(outside_pred_n, pred_n),
                "tp_retention_inside_proxy_roi": safe_ratio(tp_inside_n, tp_n),
                "gt_retention_inside_proxy_roi": safe_ratio(gt_inside_n, gt_n),
                "dice_before_clipping": before_dice,
                "dice_after_clipping": after_dice,
                "dice_delta": after_dice - before_dice,
                "hit_before": before_dice >= HIT_THRESHOLD,
                "hit_after": after_dice >= HIT_THRESHOLD,
                "new_hit": before_dice < HIT_THRESHOLD <= after_dice,
                "lost_hit": before_dice >= HIT_THRESHOLD > after_dice,
                "clipped_prediction_volume": clipped_n * voxel_volume,
                "volume_ratio_after_before": safe_ratio(clipped_n, pred_n),
                "no_prediction": pred_n == 0, "no_fp": fp_n == 0, "no_gt": gt_n == 0,
            })
    quality = {
        "case_id": case, "spatial_shape": list(spatial_shape),
        "affine_axis_codes": "".join(str(x) for x in nib.aff2axcodes(ref.affine)),
        "max_off_axis_ratio": max(value[2] for value in axes.values()),
        "mask_alignment_pass": True, "gt_used_for_roi_selection": False,
        "modifier_rows": len(gt_rows), "policy_rows": len(fp_rows),
    }
    return gt_rows, fp_rows, quality


def main() -> int:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    assignments = pd.read_csv(args.modifier_assignments)
    assignments = assignments[assignments["proxy_defined"].astype(bool)].copy()
    cases = sorted(assignments["case_id"].unique())
    selected = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_index]
    gt_rows, fp_rows, quality_rows = [], [], []
    for case in tqdm(selected, desc=f"fine modifier shard {args.shard_index}"):
        case_findings = assignments[assignments["case_id"] == case].to_dict(orient="records")
        case_gt, case_fp, quality = process_case(case, case_findings, args)
        gt_rows.extend(case_gt)
        fp_rows.extend(case_fp)
        quality_rows.append(quality)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gt_rows).sort_values(["case_id", "finding_id", "modifier_cluster"]).to_csv(
        args.output_dir / "modifier_gt_consistency_per_finding.csv", index=False
    )
    pd.DataFrame(fp_rows).sort_values(["case_id", "finding_id", "modifier_cluster", "policy"]).to_csv(
        args.output_dir / "modifier_fp_leakage_per_finding.csv", index=False
    )
    payload = {
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "cases": len(selected), "modifier_rows": len(gt_rows), "policy_rows": len(fp_rows),
        "mask_alignment_failures": sum(not row["mask_alignment_pass"] for row in quality_rows),
        "gt_used_for_roi_selection": False, "cases_detail": quality_rows,
    }
    (args.output_dir / "quality_checks.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "cases_detail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
