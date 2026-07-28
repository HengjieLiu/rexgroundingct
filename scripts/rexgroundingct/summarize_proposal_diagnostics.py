#!/usr/bin/env python3
"""Summarize proposal-mask inclusion and burden for fixed ReX validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from common import REX_SEG_DIR, write_json
from run_voxtell_val_inference import parse_thresholds, threshold_label


def load_entries(path: Path) -> list[dict]:
    entries = json.loads(path.read_text()).get("test")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list under 'test'")
    return entries


def load_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.asanyarray(nib.load(str(path)).dataobj)
    if value.ndim == 3:
        value = value[None]
    if value.ndim != 4:
        raise ValueError(f"{path}: expected a 4D mask, got {value.shape}")
    return value > 0


def summarize_threshold(
    entries: list[dict],
    seg_dir: Path,
    prediction_dir: Path,
) -> dict:
    findings = []
    for entry in entries:
        name = entry["name"]
        ground_truth = load_mask(seg_dir / name)
        proposal = load_mask(prediction_dir / name)
        if ground_truth.shape != proposal.shape:
            raise ValueError(
                f"{name}: GT shape {ground_truth.shape} != proposal shape {proposal.shape}"
            )
        if ground_truth.shape[0] != len(entry.get("findings", {})):
            raise ValueError(
                f"{name}: mask channels {ground_truth.shape[0]} != finding count "
                f"{len(entry.get('findings', {}))}"
            )
        for finding_index in range(ground_truth.shape[0]):
            gt = ground_truth[finding_index]
            pred = proposal[finding_index]
            gt_voxels = int(gt.sum())
            pred_voxels = int(pred.sum())
            intersection = int(np.logical_and(gt, pred).sum())
            findings.append(
                {
                    "name": name,
                    "finding_index": finding_index,
                    "gt_voxels": gt_voxels,
                    "proposal_voxels": pred_voxels,
                    "intersection_voxels": intersection,
                    "gt_voxel_coverage": intersection / gt_voxels,
                    "proposal_to_gt_volume_ratio": pred_voxels / gt_voxels,
                    "has_overlap": bool(intersection > 0),
                }
            )

    coverage = np.asarray(
        [finding["gt_voxel_coverage"] for finding in findings],
        dtype=np.float64,
    )
    ratios = np.asarray(
        [finding["proposal_to_gt_volume_ratio"] for finding in findings],
        dtype=np.float64,
    )
    total_gt = sum(int(finding["gt_voxels"]) for finding in findings)
    total_intersection = sum(
        int(finding["intersection_voxels"]) for finding in findings
    )
    total_proposal = sum(int(finding["proposal_voxels"]) for finding in findings)
    return {
        "cases": len(entries),
        "findings": len(findings),
        "findings_with_overlap": sum(
            int(finding["has_overlap"]) for finding in findings
        ),
        "finding_overlap_rate": float(
            np.mean([finding["has_overlap"] for finding in findings])
        ),
        "mean_gt_voxel_coverage": float(coverage.mean()),
        "median_gt_voxel_coverage": float(np.median(coverage)),
        "global_gt_voxel_coverage": total_intersection / total_gt,
        "mean_proposal_to_gt_volume_ratio": float(ratios.mean()),
        "median_proposal_to_gt_volume_ratio": float(np.median(ratios)),
        "global_proposal_to_gt_volume_ratio": total_proposal / total_gt,
        "per_finding": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--proposal-output-root", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--thresholds", default="0.1,0.3,0.5")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    entries = load_entries(args.dataset_json)
    results = {
        "dataset_json": str(args.dataset_json),
        "proposal_output_root": str(args.proposal_output_root),
        "thresholds": {},
    }
    for threshold in parse_thresholds(args.thresholds):
        label = threshold_label(threshold)
        summary = summarize_threshold(
            entries,
            args.seg_dir,
            args.proposal_output_root / label / "predictions",
        )
        eval_summary_path = (
            args.proposal_output_root
            / label
            / "reports"
            / "val_quick_global_eval_summary.json"
        )
        if eval_summary_path.is_file():
            eval_summary = json.loads(eval_summary_path.read_text())
            summary["mean_global_dice_per_finding"] = eval_summary.get(
                "mean_global_dice_per_finding"
            )
            summary["hit_rate"] = eval_summary.get("hit_rate")
            summary["hits"] = eval_summary.get("total_hits")
        results["thresholds"][label] = {
            "threshold": threshold,
            **summary,
        }

    write_json(args.output_json, results)
    lines = [
        "# Proposal Diagnostics",
        "",
        "| Threshold | Dice | Hit rate | GT coverage | Finding overlap | Volume / GT |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, result in results["thresholds"].items():
        dice = result.get("mean_global_dice_per_finding")
        hit_rate = result.get("hit_rate")
        lines.append(
            "| "
            f"{result['threshold']:.2f} | "
            f"{dice:.4f} | " if dice is not None else f"| {result['threshold']:.2f} | n/a | "
        )
        prefix = lines.pop()
        lines.append(
            prefix
            + (f"{hit_rate:.4f} | " if hit_rate is not None else "n/a | ")
            + f"{result['mean_gt_voxel_coverage']:.4f} | "
            + f"{result['finding_overlap_rate']:.4f} | "
            + f"{result['mean_proposal_to_gt_volume_ratio']:.3f} |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
