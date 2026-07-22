#!/usr/bin/env python3
"""Summarize a single-case orientation-fix diagnostic for experiment 001."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def case_from_eval(eval_json: Path, case_name: str) -> dict[str, Any]:
    data = read_json(eval_json)
    for case in data.get("cases", []):
        if case.get("file") == case_name:
            return case
    raise KeyError(f"{case_name} not found in {eval_json}")


def case_metrics(case: dict[str, Any]) -> dict[str, Any]:
    stats = case["case_stats"]
    findings = case.get("findings", {})
    dice_values = stats.get("finding_global_dice_list", [])
    return {
        "total_cases": 1,
        "total_findings": len(findings),
        "mean_global_dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "hit_rate": stats.get("hit_rate"),
        "instance_precision": stats.get("instance_precision"),
        "instance_recall": stats.get("instance_recall"),
        "instance_f1": stats.get("instance_f1"),
        "gt_instances": stats.get("gt_instances"),
        "pred_instances": stats.get("pred_instances"),
        "tp": stats.get("tp"),
        "fp": stats.get("fp"),
        "fn": stats.get("fn"),
        "finding_global_dice_list": dice_values,
    }


def lightweight_global_metrics(gt_path: Path, pred_path: Path) -> dict[str, Any]:
    gt = np.asanyarray(nib.load(str(gt_path)).dataobj) > 0
    pred = np.asanyarray(nib.load(str(pred_path)).dataobj) > 0
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch for lightweight metrics: {gt.shape} vs {pred.shape}")
    dices: list[float] = []
    hits = 0
    for index in range(gt.shape[0]):
        g = gt[index]
        p = pred[index]
        denom = int(g.sum() + p.sum())
        dice = float((2 * (g & p).sum() / denom) if denom else 1.0)
        dices.append(dice)
        hits += int(dice >= 0.1)
    return {
        "total_findings": int(gt.shape[0]),
        "mean_global_dice": float(np.mean(dices)) if dices else 0.0,
        "hit_rate": float(hits / len(dices)) if dices else 0.0,
        "finding_global_dice_list": dices,
    }


def posthoc_expected_prediction(gt_path: Path, baseline_pred_path: Path) -> dict[str, Any]:
    gt_img = nib.load(str(gt_path))
    pred = np.asanyarray(nib.load(str(baseline_pred_path)).dataobj) > 0
    expected = np.swapaxes(pred, -3, -2)[..., ::-1, ::-1, :]
    if expected.shape != gt_img.shape:
        raise ValueError(f"Post-hoc expected shape {expected.shape} != GT shape {gt_img.shape}")
    tmp = baseline_pred_path.parent / f".posthoc_expected_{baseline_pred_path.name}"
    nib.save(
        nib.Nifti1Image(expected.astype(np.uint8, copy=False), gt_img.affine, gt_img.header),
        str(tmp),
    )
    try:
        return lightweight_global_metrics(gt_path, tmp)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def status_metadata(status_json: Path) -> dict[str, Any]:
    data = read_json(status_json)
    cases = data.get("cases", [])
    if len(cases) != 1:
        raise ValueError(f"Expected one status case in {status_json}, found {len(cases)}")
    return cases[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--baseline-eval-json", type=Path, required=True)
    parser.add_argument("--after-eval-json", type=Path, required=True)
    parser.add_argument("--after-status-json", type=Path, required=True)
    parser.add_argument("--gt-path", type=Path, required=True)
    parser.add_argument("--baseline-pred-path", type=Path, required=True)
    parser.add_argument("--after-pred-path", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fail-unless-improved", action="store_true")
    args = parser.parse_args()

    baseline = case_metrics(case_from_eval(args.baseline_eval_json, args.case_name))
    after = case_metrics(case_from_eval(args.after_eval_json, args.case_name))
    after_lightweight = lightweight_global_metrics(args.gt_path, args.after_pred_path)
    posthoc = posthoc_expected_prediction(args.gt_path, args.baseline_pred_path)
    status = status_metadata(args.after_status_json)
    orientation = status.get("orientation", {})
    dice_delta = after["mean_global_dice"] - baseline["mean_global_dice"]
    hit_delta = after["hit_rate"] - baseline["hit_rate"]
    passed = (
        status.get("status") == "written"
        and orientation.get("raw_prediction_shape") == [2, 253, 512, 512]
        and orientation.get("final_prediction_shape") == [2, 512, 512, 253]
        and dice_delta > 0.05
        and hit_delta > 0
        and status.get("runtime", {}).get("cuda_available") is True
    )

    result = {
        "case_name": args.case_name,
        "passed": passed,
        "dice_delta": dice_delta,
        "hit_delta": hit_delta,
        "baseline_official": baseline,
        "after_official": after,
        "after_lightweight_global": after_lightweight,
        "posthoc_expected_lightweight_global": posthoc,
        "status": status,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    rows = [
        ("Baseline official full eval", baseline),
        ("Corrected official one-case eval", after),
        ("Corrected lightweight global", after_lightweight),
        ("Post-hoc expected lightweight global", posthoc),
    ]
    lines = [
        "# Experiment 001 Single-Case Orientation Diagnostic",
        "",
        f"- Case: `{args.case_name}`",
        f"- Passed gate: `{passed}`",
        f"- Official Dice delta: `{fmt(dice_delta)}`",
        f"- Official hit-rate delta: `{fmt(hit_delta)}`",
        f"- Inference device: `{status.get('runtime', {}).get('device')}`",
        f"- CUDA device: `{status.get('runtime', {}).get('cuda_device_name')}`",
        "",
        "## Metrics",
        "",
        "| Run | Findings | Dice | Hit | Precision | Recall | F1 | GT Inst | Pred Inst | TP/FP/FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in rows:
        lines.append(
            f"| {label} | {metrics.get('total_findings')} | {fmt(metrics.get('mean_global_dice'))} | "
            f"{fmt(metrics.get('hit_rate'))} | {fmt(metrics.get('instance_precision'))} | "
            f"{fmt(metrics.get('instance_recall'))} | {fmt(metrics.get('instance_f1'))} | "
            f"{fmt(metrics.get('gt_instances'))} | {fmt(metrics.get('pred_instances'))} | "
            f"{fmt(metrics.get('tp'))}/{fmt(metrics.get('fp'))}/{fmt(metrics.get('fn'))} |"
        )
    lines.extend(
        [
            "",
            "## Orientation Metadata",
            "",
            f"- Raw prediction shape: `{orientation.get('raw_prediction_shape')}`",
            f"- Final prediction shape: `{orientation.get('final_prediction_shape')}`",
            f"- CT original axcodes: `{orientation.get('ct_original_axcodes')}`",
            f"- CT reoriented axcodes: `{orientation.get('ct_reoriented_axcodes')}`",
            f"- Applied transform: `{orientation.get('applied_transform')}`",
            "",
            "## Files",
            "",
            f"- Baseline eval JSON: `{args.baseline_eval_json}`",
            f"- Corrected eval JSON: `{args.after_eval_json}`",
            f"- Status JSON: `{args.after_status_json}`",
            f"- Corrected prediction: `{args.after_pred_path}`",
            f"- Machine-readable summary: `{args.output_json}`",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines))
    print(f"Wrote report: {args.output_md}")
    print(f"Wrote summary: {args.output_json}")
    if args.fail_unless_improved and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
