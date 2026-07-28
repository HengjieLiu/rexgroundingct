#!/usr/bin/env python3
"""Summarize Exp008 epoch-80 fixed-val200 evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import utc_now_iso, write_json


VARIANTS = (
    "v1_sharedfusion_softguide",
    "v1_dualfusion_softguide",
    "v2_dualfusion_precision",
    "v3_dualfusion_softguide_joint",
)
SOURCE = {
    "label": "Exp006 v123_cached_e5_d4 epoch100 (Exp008 starting point)",
    "dice": 0.32413407768565705,
    "hit_rate": 0.7611548556430446,
    "hits": 290,
    "findings": 381,
}


def load_metric(path: Path) -> dict:
    summary = json.loads(path.read_text())
    return {
        "dice": float(summary["mean_global_dice_per_finding"]),
        "hit_rate": float(summary["hit_rate"]),
        "hits": int(summary["total_hits"]),
        "findings": int(summary["total_findings"]),
        "cases": int(summary["total_cases"]),
        "summary_path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    records: dict[str, dict] = {}
    for variant in VARIANTS:
        summary_path = (
            args.group_dir
            / variant
            / "eval_epoch080_val200"
            / "reports"
            / "val_quick_global_eval_summary.json"
        )
        if not summary_path.is_file():
            raise SystemExit(f"Missing summary: {summary_path}")
        metric = load_metric(summary_path)
        if metric["cases"] != 200 or metric["findings"] != SOURCE["findings"]:
            raise SystemExit(
                f"{variant}: expected 200 cases/{SOURCE['findings']} findings, "
                f"got {metric['cases']}/{metric['findings']}"
            )
        metric["dice_delta_from_source"] = metric["dice"] - SOURCE["dice"]
        metric["hit_rate_delta_from_source"] = (
            metric["hit_rate"] - SOURCE["hit_rate"]
        )
        records[variant] = metric

    result = {
        "created_at_utc": utc_now_iso(),
        "experiment": "008_voxtell_dual_branch_proposal_refinement_ablation",
        "checkpoint_epoch": 80,
        "dataset": "rexgroundingct_val200_seed20260723.json",
        "threshold": 0.5,
        "group_dir": str(args.group_dir),
        "source_reference": SOURCE,
        "variants": records,
    }
    write_json(args.output_json, result)

    lines = [
        "# Experiment 008 Epoch-80 Val200",
        "",
        "Fixed seeded val200, final-branch masks at threshold `0.5`.",
        "",
        "| Model | Dice | Hit rate | Hits / findings | Delta Dice | Delta hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `{SOURCE['label']}` | {SOURCE['dice']:.4f} | "
            f"{SOURCE['hit_rate']:.4f} | {SOURCE['hits']} / "
            f"{SOURCE['findings']} | - | - |"
        ),
    ]
    for variant, metric in records.items():
        lines.append(
            f"| `{variant}` | {metric['dice']:.4f} | "
            f"{metric['hit_rate']:.4f} | {metric['hits']} / "
            f"{metric['findings']} | {metric['dice_delta_from_source']:+.4f} | "
            f"{metric['hit_rate_delta_from_source']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "The source row is the Exp008 initialization checkpoint evaluated "
            "previously on the same fixed val200 set.",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
