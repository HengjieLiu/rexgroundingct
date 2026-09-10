#!/usr/bin/env python3
"""Apply the frozen semantic exact15 / side15 / whole10 inference policy.

GT is optional and is used only after ROI selection for metric reproduction.
The deployable path requires only predictions, finding text, and TotalSegmentator
lung/lobe masks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage
from tqdm import tqdm

from evaluate_anatomy_prior_ablation import channel_first_mask
from evaluate_anatomy_prior_diagnostic import LOBE_FILES, geometry_reference, load_prior
from evaluate_semantic_aware_lobe_constraint_part1 import semantic_parse


HIT_THRESHOLD = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("outputs/semantic_aware_lobe_constraint_groupB_refine_allcat1over1_step6000/"
                     "frozen_best_policy_config.yaml"),
    )
    p.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("outputs/zoom_out_256_matched/fullvol_evaluation/"
                     "s3_allcategory_1over1_step6000"),
    )
    p.add_argument(
        "--findings-csv",
        type=Path,
        default=Path("outputs/s3_attention_candidate_diagnostic_all_categories_step6000/findings.csv"),
    )
    p.add_argument(
        "--prior-dir",
        type=Path,
        default=Path("outputs/anatomy_prior_diagnostic_val_hardneg_step1000_thr0p7/lung_priors"),
    )
    p.add_argument("--ct-dir", type=Path, default=Path("data/ct_rate_volumes_fixed"))
    p.add_argument("--gt-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None, help="Optional output NIfTI directory.")
    p.add_argument("--metrics-out", type=Path, default=None)
    p.add_argument("--per-finding-out", type=Path, default=None)
    p.add_argument("--prediction-is-probability", action="store_true")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "high_conf_exact_lobe": ("exact_lobe", 15),
        "ambiguous_lobe": ("side_lung", 15),
        "laterality_only": ("side_lung", 15),
        "bilateral_multifocal": ("whole_lung", 10),
        "nonlocalizable": ("whole_lung", 10),
    }
    for group, (kind, dilation) in expected.items():
        value = config["groups"][group]
        if value["roi"] != kind or int(value["dilation_mm"]) != dilation:
            raise ValueError(f"Frozen policy mismatch for {group}: {value}")
    return config


def load_priors(case: str, prior_dir: Path, ref: nib.Nifti1Image) -> dict[str, np.ndarray]:
    priors = {
        "whole": load_prior(prior_dir, case, "whole_lung", ref),
        "left": load_prior(prior_dir, case, "left_lung", ref),
        "right": load_prior(prior_dir, case, "right_lung", ref),
    }
    for lobe, filename in LOBE_FILES.items():
        priors[lobe] = load_prior(prior_dir, case, filename, ref)
    return priors


def process_case(task: tuple[str, list[dict[str, Any]], dict[str, Any]]) -> tuple[list[dict], str | None]:
    case, findings, settings = task
    try:
        pred_path = Path(settings["pred_dir"]) / case
        pred, pred_ref = channel_first_mask(
            pred_path, float(settings["threshold"]), bool(settings["prediction_is_probability"])
        )
        ref = geometry_reference(case, pred_ref, Path(settings["ct_dir"]))
        priors = load_priors(case, Path(settings["prior_dir"]), ref)
        sampling = tuple(float(x) for x in nib.affines.voxel_sizes(ref.affine)[:3])
        base_cache: dict[str, np.ndarray] = {
            "whole": priors["whole"],
            "left": priors["left"],
            "right": priors["right"],
        }
        distance_cache: dict[str, np.ndarray] = {}
        roi_cache: dict[tuple[str, int], np.ndarray] = {}

        def base(key: str) -> np.ndarray:
            if key not in base_cache:
                labels = key.removeprefix("lobes:").split("+")
                base_cache[key] = np.logical_or.reduce([priors[label] for label in labels])
            return base_cache[key]

        def dilated(key: str, mm: int) -> np.ndarray:
            cache_key = (key, mm)
            if cache_key not in roi_cache:
                mask = base(key)
                if key not in distance_cache:
                    distance_cache[key] = ndimage.distance_transform_edt(~mask, sampling=sampling)
                roi_cache[cache_key] = mask | (distance_cache[key] <= float(mm))
            return roi_cache[cache_key]

        def select_roi(parsed: dict[str, Any]) -> tuple[np.ndarray, str, str, int]:
            group = parsed["group"]
            if group == "high_conf_exact_lobe" and parsed["lobes"]:
                key = "lobes:" + "+".join(sorted(parsed["lobes"]))
                return dilated(key, 15), "exact_lobe", "+".join(parsed["lobes"]), 15
            if group in {"high_conf_exact_lobe", "ambiguous_lobe", "laterality_only"}:
                side = parsed["side"]
                if side in {"left", "right"}:
                    return dilated(side, 15), "side_lung", side, 15
            return dilated("whole", 10), "whole_lung", "whole", 10

        gt = None
        if settings.get("gt_dir"):
            gt, _ = channel_first_mask(Path(settings["gt_dir"]) / case, 0.5, False)
            if gt.shape != pred.shape:
                raise ValueError(f"GT {gt.shape} != prediction {pred.shape}")
        constrained = np.zeros_like(pred, dtype=np.uint8)
        vv = float(np.prod(nib.affines.voxel_sizes(ref.affine)[:3]))
        rows: list[dict[str, Any]] = []
        for finding in findings:
            idx = int(finding["finding_idx"])
            parsed = semantic_parse(str(finding["prompt"]))
            roi, roi_kind, roi_label, dilation = select_roi(parsed)
            output = pred[idx] & roi
            constrained[idx] = output
            row = {
                "case_id": case,
                "finding_id": idx,
                "finding_text": finding["prompt"],
                "group": parsed["group"],
                "side": parsed["side"],
                "lobes": ";".join(parsed["lobes"]),
                "selected_roi_kind": roi_kind,
                "selected_roi_label": roi_label,
                "dilation_mm": dilation,
                "pred_volume_mm3": int(output.sum()) * vv,
            }
            if gt is not None:
                gt_mask = gt[idx]
                denom = int(output.sum()) + int(gt_mask.sum())
                dice = 1.0 if denom == 0 else 2.0 * int((output & gt_mask).sum()) / denom
                row.update({"dice": dice, "hit": dice >= HIT_THRESHOLD})
            rows.append(row)

        output_dir = settings.get("output_dir")
        if output_dir:
            output_path = Path(output_dir) / case
            output_path.parent.mkdir(parents=True, exist_ok=True)
            original = nib.load(str(pred_path))
            data: np.ndarray
            if len(original.shape) == 3:
                data = constrained[0]
            elif original.shape[0] <= 64:
                data = constrained
            else:
                data = np.moveaxis(constrained, 0, -1)
            header = original.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(data.astype(np.uint8), original.affine, header), str(output_path))
        return rows, None
    except Exception as exc:
        return [], f"{case}: {type(exc).__name__}: {exc}"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    findings = pd.read_csv(args.findings_csv)
    if args.limit is not None:
        findings = findings.head(args.limit).copy()
    settings = {
        "pred_dir": str(args.pred_dir),
        "prior_dir": str(args.prior_dir),
        "ct_dir": str(args.ct_dir),
        "gt_dir": str(args.gt_dir) if args.gt_dir else "",
        "output_dir": str(args.output_dir) if args.output_dir else "",
        "prediction_is_probability": args.prediction_is_probability,
        "threshold": float(config["prediction_threshold"]),
    }
    tasks = [
        (str(case), frame.to_dict(orient="records"), settings)
        for case, frame in findings.groupby("case_id", sort=True)
    ]
    rows: list[dict] = []
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
        futures = [pool.submit(process_case, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="frozen semantic policy"):
            case_rows, failure = future.result()
            rows.extend(case_rows)
            if failure:
                failures.append(failure)
    if failures:
        raise RuntimeError(f"{len(failures)} failures; first: {failures[0]}")
    frame = pd.DataFrame(rows).sort_values(["case_id", "finding_id"])
    if args.per_finding_out:
        args.per_finding_out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.per_finding_out, index=False)
    if args.metrics_out:
        if "dice" not in frame:
            raise ValueError("--metrics-out requires --gt-dir")
        summary = pd.DataFrame(
            [
                {
                    "policy": config["policy_name"],
                    "n_findings": len(frame),
                    "mean_dice": frame["dice"].mean(),
                    "hit_count": int(frame["hit"].sum()),
                    "predicted_volume_sum_mm3": frame["pred_volume_mm3"].sum(),
                }
            ]
        )
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.metrics_out, index=False)
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
