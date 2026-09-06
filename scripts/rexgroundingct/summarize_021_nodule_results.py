#!/usr/bin/env python3
"""Render the Exp021 full-val200 and nodule-specific milestone report."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import read_json, sha256_file, utc_now_iso, write_json


EXPERIMENT_ID = "021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100"
MILESTONES = (25, 50, 75, 100)
ALL_CATEGORIES = (
    "1a",
    "1b",
    "1c",
    "1d",
    "1e",
    "1f",
    "2a",
    "2b",
    "2c",
    "2d",
    "2e",
    "2f",
    "2g",
    "2h",
)
NODULE_CATEGORY = "2d"
NODULE_HIT_THRESHOLD = 0.1


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def metric_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    values = list(records)
    if not values:
        return None
    dice = sum(float(record["dice"]) for record in values) / len(values)
    hits = sum(int(float(record["dice"]) >= NODULE_HIT_THRESHOLD) for record in values)
    return {
        "mean_global_dice_per_finding": dice,
        "hit_rate": hits / len(values),
        "hits": hits,
        "findings": len(values),
        "cases": len({record["case_name"] for record in values}),
    }


def evaluation_records(eval_json: Path, dataset_json: Path) -> list[dict[str, Any]]:
    dataset = read_json(dataset_json).get("test")
    evaluation = read_json(eval_json)
    cases = evaluation.get("cases")
    if not isinstance(dataset, list) or not isinstance(cases, list):
        raise ValueError(f"Invalid evaluation or dataset JSON: {eval_json}")
    dataset_by_name = {str(entry["name"]): entry for entry in dataset}
    cases_by_name = {str(entry["file"]): entry for entry in cases}
    if set(dataset_by_name) != set(cases_by_name):
        raise ValueError(f"Evaluation case set does not match {dataset_json}")

    records: list[dict[str, Any]] = []
    for case_name, entry in dataset_by_name.items():
        metrics = cases_by_name[case_name].get("findings", {})
        expected_keys = set(str(key) for key in entry.get("findings", {}))
        observed_keys = {str(key).removeprefix("finding_") for key in metrics}
        if expected_keys != observed_keys:
            raise ValueError(f"{case_name}: evaluator finding keys do not match dataset")
        for metric_key, metric in metrics.items():
            finding_index = str(metric_key).removeprefix("finding_")
            records.append(
                {
                    "case_name": case_name,
                    "finding_index": finding_index,
                    "category": str(entry["categories"][finding_index]),
                    "dice": float(metric["global_dice"]),
                }
            )
    return records


def summarize_evaluation(eval_json: Path, dataset_json: Path) -> dict[str, Any]:
    records = evaluation_records(eval_json, dataset_json)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
    overall = metric_summary(records)
    nodule = metric_summary(by_category[NODULE_CATEGORY])
    if overall is None or nodule is None:
        raise ValueError(f"Evaluation is missing overall or category-2d records: {eval_json}")
    return {
        "evaluation_json": str(eval_json),
        "evaluation_json_sha256": sha256_file(eval_json),
        "dataset_json": str(dataset_json),
        "dataset_json_sha256": sha256_file(dataset_json),
        "overall": overall,
        "nodule": nodule,
        "categories": {
            category: metric_summary(by_category[category])
            for category in ALL_CATEGORIES
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def metric_cell(metric: dict[str, Any] | None) -> str:
    if metric is None:
        return "—"
    return (
        f"{fmt(metric['mean_global_dice_per_finding'])} / "
        f"{int(metric['hits'])}/{int(metric['findings'])} "
        f"({fmt(metric['hit_rate'] * 100, 1)}%)"
    )


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    source = None
    if args.source_evaluation_json.is_file():
        source = summarize_evaluation(args.source_evaluation_json, args.val200_json)
        source["label"] = "Initial reference: Exp017 phase-1 epoch 100"

    milestones: dict[str, Any] = {}
    for epoch in MILESTONES:
        eval_json = (
            args.group_dir
            / "ddp_bs4"
            / f"eval_epoch{epoch:03d}_val200"
            / "eval"
            / "val_quick_global_eval.json"
        )
        milestones[str(epoch)] = {
            "epoch": epoch,
            "update": epoch * args.steps_per_epoch,
            "evaluation": (
                summarize_evaluation(eval_json, args.val200_json)
                if eval_json.is_file()
                else None
            ),
        }

    available = [
        item["evaluation"]
        for item in milestones.values()
        if item["evaluation"] is not None
    ]
    best = None
    if available:
        best = max(
            available,
            key=lambda item: (
                float(item["nodule"]["mean_global_dice_per_finding"]),
                float(item["nodule"]["hit_rate"]),
            ),
        )
    return {
        "experiment": EXPERIMENT_ID,
        "status": "complete" if len(available) == len(MILESTONES) else "running",
        "updated_at_utc": utc_now_iso(),
        "evaluation_contract": {
            "milestone_epochs": list(MILESTONES),
            "dataset": str(args.val200_json),
            "cases": 200,
            "findings": 381,
            "nodule_category": NODULE_CATEGORY,
            "nodule_cases": 119,
            "nodule_findings": 132,
            "nodule_hit_threshold_dice": NODULE_HIT_THRESHOLD,
        },
        "source_reference": source,
        "milestones": milestones,
        "best_nodule_milestone": best,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Experiment 021: 0.7 mm Nodule-Only Positive Specialist",
        "",
        f"Status: `{summary['status']}`; report updated `{summary['updated_at_utc']}`.",
        "",
        "Training uses only official category `2d` pulmonary nodules/masses as positive targets. Other findings and background are zero-target nodule examples.",
        "",
        "## Evaluation contract",
        "",
        "Every milestone below is a synchronous full evaluation of the fixed 200-case / 381-finding validation set at threshold 0.5. Nodule metrics are recomposed from the 132 category-2d findings in 119 cases; a hit is Dice ≥ 0.1.",
        "",
        "## Initial nodule reference",
        "",
        "| Reference | Overall Dice / hits | Nodule Dice / hits |",
        "| --- | ---: | ---: |",
    ]
    source = summary.get("source_reference")
    if source is None:
        lines.append("| Exp017 phase-1 epoch 100 | — | — |")
    else:
        lines.append(
            f"| Exp017 phase-1 epoch 100 | {metric_cell(source['overall'])} | {metric_cell(source['nodule'])} |"
        )

    lines.extend(
        [
            "",
            "## Milestone summary",
            "",
            "Cells show `Dice / hits/findings (hit rate)`.",
            "",
            "| Epoch | Update | Overall | Nodule category 2d |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for epoch in MILESTONES:
        item = summary["milestones"][str(epoch)]
        evaluation = item["evaluation"]
        lines.append(
            f"| {epoch} | {item['update']} | "
            f"{metric_cell(evaluation['overall']) if evaluation else '—'} | "
            f"{metric_cell(evaluation['nodule']) if evaluation else '—'} |"
        )

    lines.extend(["", "## Full val200 per-category performance", ""])
    for epoch in MILESTONES:
        item = summary["milestones"][str(epoch)]
        evaluation = item["evaluation"]
        lines.extend(
            [
                f"### Epoch {epoch}",
                "",
                "| Category | Findings | Dice | Hits | Hit rate |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for category in ALL_CATEGORIES:
            metric = evaluation["categories"].get(category) if evaluation else None
            if metric is None:
                lines.append(f"| {category} | 0 | — | — | — |")
            else:
                lines.append(
                    f"| {category} | {metric['findings']} | "
                    f"{fmt(metric['mean_global_dice_per_finding'])} | "
                    f"{metric['hits']}/{metric['findings']} | "
                    f"{fmt(metric['hit_rate'] * 100, 1)}% |"
                )
        lines.append("")

    best = summary.get("best_nodule_milestone")
    if best is not None:
        best_epoch = next(
            epoch
            for epoch, item in summary["milestones"].items()
            if item["evaluation"] is best
        )
        lines.extend(
            [
                "## Current readout",
                "",
                f"Best available nodule Dice is epoch `{best_epoch}`: **{fmt(best['nodule']['mean_global_dice_per_finding'])}**, with `{best['nodule']['hits']}/{best['nodule']['findings']}` hits.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--val200-json", type=Path, required=True)
    parser.add_argument("--source-evaluation-json", type=Path, required=True)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args)
    write_json(args.output_json, summary)
    atomic_write_text(args.output_md, render_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
