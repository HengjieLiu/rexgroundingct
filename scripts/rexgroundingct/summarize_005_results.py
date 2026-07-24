#!/usr/bin/env python3
"""Collect experiment 005 proposal and cascade results into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    variants = ["strict_inclusion", "inclusion_margin16mm"]
    records = {}
    for variant in variants:
        root = args.group_dir / variant
        proposal_path = root / "reports" / "val200_step_05000.json"
        cascade_summary = (
            root
            / "cascade_step05000_val200"
            / "reports"
            / "val_quick_global_eval_summary.json"
        )
        if not proposal_path.is_file() or not cascade_summary.is_file():
            continue
        proposal = json.loads(proposal_path.read_text())
        cascade = json.loads(cascade_summary.read_text())
        records[variant] = {
            "proposal_json": str(proposal_path),
            "proposal": proposal["summary"],
            "proposal_by_category": proposal.get("by_category", {}),
            "cascade_summary_json": str(cascade_summary),
            "cascade": cascade,
        }
    payload = {"group_dir": str(args.group_dir), "variants": records}
    write_json(args.output_json, payload)
    lines = [
        "# Experiment 005 Final Comparison",
        "",
        "| Variant | Proposal hit@3 | Full inclusion@3 | Coverage@3 | Cascade Dice | Cascade hit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, result in records.items():
        proposal = result["proposal"]
        cascade = result["cascade"]
        lines.append(
            f"| {variant} | {proposal['hit_at_3']:.4f} | "
            f"{proposal['full_inclusion_at_3']:.4f} | "
            f"{proposal['target_coverage_at_3']:.4f} | "
            f"{cascade['mean_global_dice_per_finding']:.4f} | "
            f"{cascade['hit_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Stage 1 is selected by full inclusion and target coverage; end-to-end selection",
            "also considers cascade Dice and hit rate at the fixed threshold 0.5.",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
