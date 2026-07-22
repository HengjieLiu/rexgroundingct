#!/usr/bin/env python3
"""Summarize a global-only ReXrank evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--label", default="VoxTell v1.1 MICCAI val corrected orientation quick eval")
    args = parser.parse_args()

    data = json.loads(args.eval_json.read_text())
    summary = data["summary"]
    compact = {
        "label": args.label,
        "total_cases": summary.get("total_cases"),
        "total_findings": summary.get("total_findings"),
        "total_hits": summary.get("total_hits"),
        "total_misses": summary.get("total_misses"),
        "mean_global_dice_per_finding": summary.get("mean_global_dice_per_finding"),
        "mean_global_dice_per_case": summary.get("mean_global_dice_per_case"),
        "hit_rate": summary.get("hit_rate"),
        "mean_findings_per_case": summary.get("mean_findings_per_case"),
        "params": summary.get("params", {}),
        "eval_json": str(args.eval_json),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# {args.label}",
        "",
        "This is a global-only quick evaluation. It reports Dice and hit rate, but",
        "does not compute connected components or instance precision/recall/F1.",
        "",
        "The challenge ranking metric is **Mean global Dice per finding**. Mean",
        "global Dice per case is a secondary analysis metric.",
        "",
        "## Metrics",
        "",
        f"- Cases: `{fmt(compact['total_cases'])}`",
        f"- Findings/prompts: `{fmt(compact['total_findings'])}`",
        f"- Hits: `{fmt(compact['total_hits'])}`",
        f"- Misses: `{fmt(compact['total_misses'])}`",
        f"- Challenge metric, mean global Dice per finding: `{fmt(compact['mean_global_dice_per_finding'])}`",
        f"- Secondary mean global Dice per case: `{fmt(compact['mean_global_dice_per_case'])}`",
        f"- Hit rate: `{fmt(compact['hit_rate'])}`",
        "",
        "## Files",
        "",
        f"- Evaluation JSON: `{args.eval_json}`",
        f"- Machine-readable summary: `{args.output_json}`",
        "",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines))
    print(f"Wrote report: {args.output_md}")
    print(f"Wrote summary: {args.output_json}")
    print(
        "dice={dice} hit={hit} cases={cases} findings={findings}".format(
            dice=fmt(compact["mean_global_dice_per_finding"]),
            hit=fmt(compact["hit_rate"]),
            cases=fmt(compact["total_cases"]),
            findings=fmt(compact["total_findings"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
