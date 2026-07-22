#!/usr/bin/env python3
"""Summarize official ReXrank evaluator JSON and write a Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REFERENCE_ROWS = [
    ("VoxTell paper Figure 4 fine-tuned val", 0.282, 0.678, None),
    ("ReXrank main benchmark VoxTell", 0.285, 0.615, 0.227),
    ("MICCAI public leaderboard ThoraxTell", 0.296, 0.689, 0.163),
]


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--label", default="VoxTell v1.1 MICCAI val")
    args = parser.parse_args()

    data = json.loads(args.eval_json.read_text())
    summary = data["summary"]
    dice = summary.get("mean_global_dice_per_finding")
    hit = summary.get("hit_rate")
    f1 = summary.get("overall_instance_f1")
    precision = summary.get("overall_instance_precision")
    recall = summary.get("overall_instance_recall")
    total_findings = summary.get("total_findings")
    model_prompts = total_findings
    total_gt_instances = summary.get("total_gt_instances")
    total_pred_instances = summary.get("total_pred_instances")
    total_tp = summary.get("total_tp")
    total_fp = summary.get("total_fp")
    total_fn = summary.get("total_fn")

    lines = [
        f"# {args.label}",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary.get('total_cases')}`",
        f"- Findings: `{total_findings}`",
        f"- Mean global Dice per finding: `{fmt(dice)}`",
        f"- Hit rate: `{fmt(hit)}`",
        f"- Instance precision: `{fmt(precision)}`",
        f"- Instance recall: `{fmt(recall)}`",
        f"- Instance F1: `{fmt(f1)}`",
        "",
        "## Finding and Instance Counts",
        "",
        f"- Total GT findings: `{total_findings}`",
        f"- Model findings/prompts evaluated: `{model_prompts}`",
        "- The model was prompted once per GT finding, so this count matches total findings.",
        "- Predicted connected components:",
        f"  - GT instances: `{total_gt_instances}`",
        f"  - Model predicted instances: `{total_pred_instances}`",
        f"  - TP/FP/FN: `{total_tp}` / `{total_fp}` / `{total_fn}`",
        "",
        "## Comparison",
        "",
        "| Source | Dice | Hit | Instance F1 |",
        "| --- | ---: | ---: | ---: |",
        f"| {args.label} | {fmt(dice)} | {fmt(hit)} | {fmt(f1)} |",
    ]
    for source, ref_dice, ref_hit, ref_f1 in REFERENCE_ROWS:
        lines.append(f"| {source} | {fmt(ref_dice)} | {fmt(ref_hit)} | {fmt(ref_f1)} |")
    lines.extend(
        [
            "",
            "Notes:",
            "- VoxTell paper Figure 4 reports fine-tuned validation metrics with HIT5, not the official HIT@0.1.",
            "- MICCAI leaderboard rows are contextual because the public leaderboard split differs from local MICCAI validation.",
            "- Public `voxtell_v1.1` is not guaranteed to match a ReXGroundingCT-fine-tuned checkpoint.",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines))
    print(f"Wrote report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
