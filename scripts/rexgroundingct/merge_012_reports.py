#!/usr/bin/env python3
"""Merge final Exp012 Run-1 and Run-2 JSON reports into one Markdown report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MILESTONES = (0, 5, 20, 40, 60, 80, 100)
VAL80_MILESTONES = (0, 20, 40, 60, 80, 100)
ARM_ORDER = (
    ("r01", "category_1alldiffuse_replay50", "1a-1f D50 (R1)"),
    ("r02", "category_1alldiffuse_replay25", "1a-1f D25 (R2)"),
    ("r02", "category_1alldiffuse_replay10", "1a-1f D10 (R2)"),
    ("r02", "category_1alldiffuse_replay00", "1a-1f D00 (R2)"),
    ("r01", "category_2a_replay50", "2a50 (R1)"),
    ("r01", "category_2b_replay50", "2b50 (R1)"),
    ("r01", "category_2c_replay50", "2c50 (R1)"),
    ("r02", "category_2d_replay50", "2d50 (R2)"),
)


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("status") != "selection_ready":
        raise ValueError(f"Report is not selection_ready: {path}")
    return report


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(int(record["hit"]) for record in records)
    return {
        "mean_global_dice_per_finding": sum(float(record["dice"]) for record in records)
        / len(records),
        "hit_rate": hits / len(records),
        "hits": hits,
        "findings": len(records),
        "cases": len({record["case_name"] for record in records}),
    }


def load_baseline_evaluation(
    evaluation_path: Path, val200_path: Path
) -> dict[str, Any]:
    evaluation = json.loads(evaluation_path.read_text())
    dataset = json.loads(val200_path.read_text())["test"]
    dataset_by_name = {entry["name"]: entry for entry in dataset}
    cases_by_name = {entry["file"]: entry for entry in evaluation["cases"]}
    if set(dataset_by_name) != set(cases_by_name):
        raise ValueError("Initial-checkpoint evaluation does not cover the fixed val200 cases")

    records: list[dict[str, Any]] = []
    for case_name, entry in dataset_by_name.items():
        findings = cases_by_name[case_name]["findings"]
        expected = set(entry["findings"])
        observed = {key.removeprefix("finding_") for key in findings}
        if expected != observed:
            raise ValueError(f"Initial-checkpoint finding mismatch for {case_name}")
        for key, metric in findings.items():
            finding_index = key.removeprefix("finding_")
            records.append(
                {
                    "case_name": case_name,
                    "category": str(entry["categories"][finding_index]),
                    "dice": float(metric["global_dice"]),
                    "hit": bool(metric["global_hit"]),
                }
            )

    categories = sorted({record["category"] for record in records}, key=category_sort_key)
    return {
        "overall": metric_summary(records),
        "categories": {
            category: metric_summary(
                [record for record in records if record["category"] == category]
            )
            for category in categories
        },
    }


def metric_cell(metric: dict[str, Any] | None, digits: int = 4) -> str:
    if not metric:
        return "—"
    return (
        f"{float(metric['mean_global_dice_per_finding']):.{digits}f} / "
        f"{float(metric['hit_rate']):.3f} "
        f"({int(metric['hits'])}/{int(metric['findings'])})"
    )


def milestone_metric(arm: dict[str, Any], epoch: int, key: str) -> dict[str, Any] | None:
    evaluation = arm["milestones"][str(epoch)].get("evaluation")
    return evaluation.get(key) if evaluation else None


def category_sort_key(category: str) -> tuple[int, str]:
    prefix = "".join(character for character in category if character.isdigit())
    return (int(prefix) if prefix else 999, category)


def bold_if_best(value: float, best: float, digits: int = 6) -> str:
    formatted = f"{value:.{digits}f}"
    return f"**{formatted}**" if abs(value - best) <= 1e-12 else formatted


def render(
    run1_path: Path,
    run2_path: Path,
    baseline_evaluation_path: Path,
    val200_path: Path,
) -> str:
    reports = {"r01": load_report(run1_path), "r02": load_report(run2_path)}
    baseline = load_baseline_evaluation(baseline_evaluation_path, val200_path)
    arms: list[tuple[str, str, str, dict[str, Any]]] = []
    for run, arm_name, alias in ARM_ORDER:
        arm = reports[run].get("arms", {}).get(arm_name)
        if arm is None:
            raise ValueError(f"{run} report is missing {arm_name}")
        arms.append((run, arm_name, alias, arm))

    lines = [
        "# Exp012 Combined Run 1 and Run 2 Report",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Run 1: `{reports['r01']['run_group']}` — `{reports['r01']['status']}`",
        f"- Run 2: `{reports['r02']['run_group']}` — `{reports['r02']['status']}`",
        "- Checkpoint selection uses target Dice only; val80, non-target, and overall metrics are diagnostic.",
        "- Run-1 diffuse replay50 and the Run-2 diffuse ratios use independent schedules and are not sample-paired.",
        "",
        "## Arm Key",
        "",
        "| Alias | Run | Arm | Target categories |",
        "| --- | --- | --- | --- |",
    ]
    for run, arm_name, alias, arm in arms:
        lines.append(
            f"| `{alias}` | {run.upper()} | `{arm_name}` | "
            f"{', '.join(arm['target_categories'])} |"
        )

    lines.extend(
        [
            "",
            "## Target Milestones",
            "",
            "Cells show `Dice / hit rate (hits/findings)`.",
            "",
            "| Arm | e0 | e5 | e20 | e40 | e60 | e80 | e100 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _run, _arm_name, alias, arm in arms:
        cells = [metric_cell(milestone_metric(arm, epoch, "target")) for epoch in MILESTONES]
        lines.append(f"| `{alias}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Val80 Total — Cross-arm Comparable",
            "",
            "Every cell uses the same fixed 80 cases and 195 findings.",
            "",
            "| Arm | e0 | e20 | e40 | e60 | e80 | e100 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _run, _arm_name, alias, arm in arms:
        cells = [
            metric_cell(milestone_metric(arm, epoch, "val80_total"))
            for epoch in VAL80_MILESTONES
        ]
        lines.append(f"| `{alias}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Val80 Non-target — Within-arm Forgetting",
            "",
            "Targets are excluded separately for each arm, so compare a row across epochs rather than comparing denominators across rows.",
            "",
            "| Arm | e0 | e20 | e40 | e60 | e80 | e100 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _run, _arm_name, alias, arm in arms:
        cells = [
            metric_cell(milestone_metric(arm, epoch, "sentinel_non_target"))
            for epoch in VAL80_MILESTONES
        ]
        lines.append(f"| `{alias}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Epoch-100 Full Val200",
            "",
            "The initial row is the shared Exp009 checkpoint used to initialize every arm. Bold marks the highest displayed value in each metric column.",
            "Non-target metrics are arm-specific exclusions, so there is no single baseline non-target value.",
            "",
            "| Arm | Overall Dice | Overall hit | Non-target Dice | Non-target hit |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    overall_rows = [baseline["overall"]] + [
        arm["milestones"]["100"]["evaluation"]["overall"]
        for _run, _arm_name, _alias, arm in arms
    ]
    non_target_rows = [
        arm["milestones"]["100"]["evaluation"]["non_target"]
        for _run, _arm_name, _alias, arm in arms
    ]
    best_overall_dice = max(
        float(metric["mean_global_dice_per_finding"]) for metric in overall_rows
    )
    best_overall_hit = max(float(metric["hit_rate"]) for metric in overall_rows)
    best_non_target_dice = max(
        float(metric["mean_global_dice_per_finding"]) for metric in non_target_rows
    )
    best_non_target_hit = max(float(metric["hit_rate"]) for metric in non_target_rows)
    baseline_overall = baseline["overall"]
    lines.append(
        "| `Initial checkpoint (Exp009)` | "
        f"{bold_if_best(float(baseline_overall['mean_global_dice_per_finding']), best_overall_dice)} "
        f"| {bold_if_best(float(baseline_overall['hit_rate']), best_overall_hit)} "
        "| — | — |"
    )
    for _run, _arm_name, alias, arm in arms:
        evaluation = arm["milestones"]["100"]["evaluation"]
        overall = evaluation["overall"]
        non_target = evaluation["non_target"]
        lines.append(
            f"| `{alias}` | "
            f"{bold_if_best(float(overall['mean_global_dice_per_finding']), best_overall_dice)} "
            f"| {bold_if_best(float(overall['hit_rate']), best_overall_hit)} "
            f"| {bold_if_best(float(non_target['mean_global_dice_per_finding']), best_non_target_dice)} "
            f"| {bold_if_best(float(non_target['hit_rate']), best_non_target_hit)} |"
        )

    lines.extend(
        [
            "",
            "## Target-only Checkpoint Recommendations",
            "",
            "| Arm | Selected epoch | Target Dice | Target hit | Delta from e0 | Checkpoint |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _run, _arm_name, alias, arm in arms:
        selection = arm["selection"]
        target = selection["target"]
        lines.append(
            f"| `{alias}` | {int(selection['epoch'])} "
            f"| {float(target['mean_global_dice_per_finding']):.6f} "
            f"| {float(target['hit_rate']):.6f} "
            f"| {float(selection['target_dice_delta_from_epoch0']):+.6f} "
            f"| `{selection['checkpoint']}` |"
        )

    categories = sorted(
        {
            category
            for _run, _arm_name, _alias, arm in arms
            for category in arm["milestones"]["100"]["evaluation"]["categories"]
        }
        | set(baseline["categories"]),
        key=category_sort_key,
    )
    lines.extend(
        [
            "",
            "## Epoch-100 Per-category Metrics — Combined",
            "",
            "The initial checkpoint and all arms are shown in one table. Cells show `Dice / hit rate`; findings are fixed by val200. Bold marks the highest Dice in each category row.",
            "",
            "| Category | Findings | `Initial (Exp009)` | "
            + " | ".join(f"`{alias}`" for _, _, alias, _ in arms)
            + " |",
            "| --- | ---: | --- | " + " | ".join("---" for _ in arms) + " |",
        ]
    )
    for category in categories:
        baseline_metric = baseline["categories"].get(category)
        arm_metrics = [
            arm["milestones"]["100"]["evaluation"]["categories"].get(category)
            for _run, _arm_name, _alias, arm in arms
        ]
        metrics = [baseline_metric] + arm_metrics
        findings = {int(metric["findings"]) for metric in metrics if metric is not None}
        if len(findings) != 1:
            raise ValueError(f"Inconsistent val200 finding count for category {category}")
        best_dice = max(
            float(metric["mean_global_dice_per_finding"])
            for metric in metrics
            if metric is not None
        )
        cells = []
        for metric in metrics:
            if metric is None:
                cells.append("—")
                continue
            dice = float(metric["mean_global_dice_per_finding"])
            cell = f"{dice:.6f} / {float(metric['hit_rate']):.3f}"
            cells.append(f"**{cell}**" if abs(dice - best_dice) <= 1e-12 else cell)
        lines.append(
            f"| {category} | {next(iter(findings))} | " + " | ".join(cells) + " |"
        )

    lines.extend(
        [
            "",
            "## Source Reports",
            "",
            f"- Run 1 JSON: `{run1_path}`",
            f"- Run 2 JSON: `{run2_path}`",
            f"- Initial-checkpoint evaluation: `{baseline_evaluation_path}`",
            f"- Fixed val200: `{val200_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run1-json", type=Path, required=True)
    parser.add_argument("--run2-json", type=Path, required=True)
    parser.add_argument("--baseline-eval-json", type=Path, required=True)
    parser.add_argument("--val200-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    atomic_write(
        args.output_md,
        render(
            args.run1_json,
            args.run2_json,
            args.baseline_eval_json,
            args.val200_json,
        ),
    )
    print(f"combined_report={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
