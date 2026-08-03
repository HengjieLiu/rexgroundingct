#!/usr/bin/env python3
"""Build the atomic live report and target-only selection for Exp013."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import read_json, sha256_file, utc_now_iso
from prepare_013_category_only_specialists import ARM_TARGETS


EXPERIMENT_ID = "013_voxtell_public_category_only_specialists"
MILESTONES = (0, 20, 40, 60, 80, 100)
EVAL_SCOPE = {0: "target", 20: "target", 40: "target", 60: "target", 80: "target", 100: "val200"}
ALL_DATA_BASELINE = {
    "method": "all_data_e5d4_exp006_reference",
    "label": "All-data e5d4 Exp006 reference",
    "checkpoint": (
        "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
        "006_voxtell_cached_native_v123_lr_ablation/runs/"
        "exp006_cached_native_lr_20260725T050001Z/v123_cached_e5_d4/"
        "model_epoch100/fold_0/checkpoint_final.pth"
    ),
    "evaluation_json": (
        "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
        "006_voxtell_cached_native_v123_lr_ablation/runs/"
        "exp006_cached_native_lr_20260725T050001Z/v123_cached_e5_d4/"
        "eval_epoch100_val200/eval/val_quick_global_eval.json"
    ),
}


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(value)
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def metric_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    values = list(records)
    if not values:
        return None
    dice = sum(float(record["dice"]) for record in values) / len(values)
    hits = sum(int(bool(record["hit"])) for record in values)
    return {
        "mean_global_dice_per_finding": dice,
        "hit_rate": hits / len(values),
        "hits": hits,
        "findings": len(values),
        "cases": len({record["case_name"] for record in values}),
    }


def evaluation_records(eval_json: Path, dataset_json: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = read_json(dataset_json).get("test")
    evaluation = read_json(eval_json)
    cases = evaluation.get("cases")
    if not isinstance(dataset, list) or not isinstance(cases, list):
        raise ValueError(f"Invalid evaluation or dataset JSON: {eval_json} {dataset_json}")
    dataset_by_name = {entry["name"]: entry for entry in dataset}
    cases_by_name = {entry["file"]: entry for entry in cases}
    if len(dataset_by_name) != len(dataset) or len(cases_by_name) != len(cases):
        raise ValueError("Duplicate case names in evaluation inputs")
    if set(dataset_by_name) != set(cases_by_name):
        raise ValueError(f"Evaluation case set does not match {dataset_json}")

    records: list[dict[str, Any]] = []
    for case_name, dataset_entry in dataset_by_name.items():
        metrics = cases_by_name[case_name].get("findings", {})
        expected_keys = set(dataset_entry.get("findings", {}))
        observed_keys = {key.removeprefix("finding_") for key in metrics}
        if expected_keys != observed_keys:
            raise ValueError(f"{case_name}: evaluator finding keys do not match dataset")
        for metric_key, metric in metrics.items():
            finding_index = metric_key.removeprefix("finding_")
            records.append(
                {
                    "case_name": case_name,
                    "finding_index": finding_index,
                    "category": str(dataset_entry["categories"][finding_index]),
                    "dice": float(metric["global_dice"]),
                    "hit": bool(metric["global_hit"]),
                }
            )
    return records, evaluation.get("summary", {})


def summarize_evaluation(
    eval_json: Path,
    dataset_json: Path,
    target_categories: set[str],
) -> dict[str, Any]:
    records, raw_summary = evaluation_records(eval_json, dataset_json)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
    target = [record for record in records if record["category"] in target_categories]
    non_target = [record for record in records if record["category"] not in target_categories]
    return {
        "evaluation_json": str(eval_json),
        "evaluation_json_sha256": sha256_file(eval_json),
        "dataset_json": str(dataset_json),
        "dataset_json_sha256": sha256_file(dataset_json),
        "overall": {
            "mean_global_dice_per_finding": raw_summary.get("mean_global_dice_per_finding"),
            "hit_rate": raw_summary.get("hit_rate"),
            "hits": raw_summary.get("total_hits"),
            "findings": raw_summary.get("total_findings"),
            "cases": raw_summary.get("total_cases"),
        },
        "target": metric_summary(target),
        "non_target": metric_summary(non_target),
        "categories": {
            category: metric_summary(category_records)
            for category, category_records in sorted(by_category.items())
        },
    }


def training_summary(group_dir: Path, arm: str, epoch: int) -> dict[str, Any] | None:
    if epoch == 0:
        return None
    path = group_dir / arm / "reports" / f"training_metrics_segment_epoch{epoch:03d}.json"
    if not path.is_file():
        return None
    data = read_json(path)
    return {
        "path": str(path),
        "last_loss": data.get("last_loss"),
        "mean_loss": data.get("mean_loss"),
        "lrs_final": data.get("lrs_final"),
        "memory": data.get("memory"),
        "completed_updates": data.get("completed_updates"),
        "status": data.get("status"),
    }


def checkpoint_path(group_dir: Path, source_checkpoint: Path, arm: str, epoch: int) -> Path:
    if epoch == 0:
        return source_checkpoint
    return group_dir / arm / "checkpoints" / f"checkpoint_update_{epoch * 100:06d}.pth"


def select_checkpoint(
    group_dir: Path,
    source_checkpoint: Path,
    arm: str,
    milestones: dict[str, Any],
) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for epoch in MILESTONES:
        evaluation = milestones.get(str(epoch), {}).get("evaluation") or {}
        metric = evaluation.get("target")
        if metric is not None:
            candidates.append((epoch, metric))
    if not candidates:
        return None

    best_epoch, best_metric = candidates[0]
    for epoch, metric in candidates[1:]:
        dice = float(metric["mean_global_dice_per_finding"])
        best_dice = float(best_metric["mean_global_dice_per_finding"])
        if dice > best_dice + 1e-6:
            best_epoch, best_metric = epoch, metric
            continue
        if abs(dice - best_dice) <= 1e-6:
            hit = float(metric["hit_rate"])
            best_hit = float(best_metric["hit_rate"])
            if hit > best_hit + 1e-12 or (abs(hit - best_hit) <= 1e-12 and epoch < best_epoch):
                best_epoch, best_metric = epoch, metric

    baseline = (milestones.get("0", {}).get("evaluation") or {}).get("target")
    return {
        "epoch": best_epoch,
        "checkpoint": str(checkpoint_path(group_dir, source_checkpoint, arm, best_epoch)),
        "target": best_metric,
        "target_dice_delta_from_epoch0": (
            None
            if baseline is None
            else float(best_metric["mean_global_dice_per_finding"])
            - float(baseline["mean_global_dice_per_finding"])
        ),
        "policy": "target Dice; hit rate then earlier epoch for Dice ties within 1e-6",
    }


def resolve_dataset(
    epoch: int, arm: str, subset_manifest: dict[str, Any], val200_json: Path
) -> Path:
    if EVAL_SCOPE[epoch] == "val200":
        return val200_json
    return Path(subset_manifest["arms"][arm]["target"]["path"])


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    subset_manifest = read_json(args.subset_manifest)
    arms: dict[str, Any] = {}
    complete = True
    for arm, target_tuple in ARM_TARGETS.items():
        arm_milestones: dict[str, Any] = {}
        for epoch in MILESTONES:
            scope = EVAL_SCOPE[epoch]
            eval_json = (
                args.group_dir
                / arm
                / f"eval_epoch{epoch:03d}_{scope}"
                / "eval"
                / "val_quick_global_eval.json"
            )
            item: dict[str, Any] = {
                "epoch": epoch,
                "scope": scope,
                "checkpoint": str(checkpoint_path(args.group_dir, args.source_checkpoint, arm, epoch)),
                "training": training_summary(args.group_dir, arm, epoch),
                "evaluation": None,
            }
            if eval_json.is_file():
                dataset = resolve_dataset(epoch, arm, subset_manifest, args.val200_json)
                item["evaluation"] = summarize_evaluation(eval_json, dataset, set(target_tuple))
            else:
                complete = False
            arm_milestones[str(epoch)] = item
        failures = sorted(
            str(path.relative_to(args.group_dir))
            for path in (args.group_dir / arm).glob(".*_failed")
        )
        arms[arm] = {
            "target_categories": list(target_tuple),
            "milestones": arm_milestones,
            "selection": select_checkpoint(
                args.group_dir, args.source_checkpoint, arm, arm_milestones
            ),
            "failures": failures,
        }

    status = "selection_ready" if complete else "active"
    if args.failure:
        status = "failed"
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "run_group": args.group_dir.name,
        "group_dir": str(args.group_dir),
        "updated_at_utc": utc_now_iso(),
        "status": status,
        "current_stage": args.current_stage,
        "selection_uses_forgetting_guard": False,
        "failure": args.failure,
        "all_data_baseline": baseline_summary(args.val200_json),
        "arms": arms,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def bold(value: str, enabled: bool) -> str:
    return f"**{value}**" if enabled and value != "pending" else value


def metric_value(metric: dict[str, Any] | None, key: str) -> float | None:
    if metric is None or metric.get(key) is None:
        return None
    return float(metric[key])


def best_methods(
    rows: list[tuple[str, dict[str, Any] | None]],
    key: str = "mean_global_dice_per_finding",
) -> set[str]:
    values = [
        (method, metric_value(metric, key))
        for method, metric in rows
        if metric_value(metric, key) is not None
    ]
    if not values:
        return set()
    best = max(value for _method, value in values if value is not None)
    return {method for method, value in values if value is not None and abs(value - best) <= 1e-12}


def metric_cell(metric: dict[str, Any] | None) -> str:
    if metric is None:
        return "pending"
    return (
        f"{metric['mean_global_dice_per_finding']:.4f} / "
        f"{metric['hit_rate']:.3f} ({metric['hits']}/{metric['findings']})"
    )


def render_markdown(summary: dict[str, Any]) -> str:
    methods = list(summary["arms"])
    baseline = summary.get("all_data_baseline")
    if baseline is not None:
        methods.append(baseline["method"])

    lines = [
        "# Exp013 Public-Start Category-Only Progress",
        "",
        f"- Status: `{summary['status']}`",
        f"- Run group: `{summary['run_group']}`",
        f"- Current stage: `{summary['current_stage']}`",
        f"- Updated: `{summary['updated_at_utc']}`",
        "- Selection: target Dice only; full val200 is run only at epoch 100.",
        "",
        "## Target Milestones",
        "",
        "Cells show `Dice / hit rate (hits/findings)`.",
        "",
        "| Arm | e0 | e20 | e40 | e60 | e80 | e100 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, arm_data in summary["arms"].items():
        cells = []
        for epoch in MILESTONES:
            evaluation = arm_data["milestones"][str(epoch)].get("evaluation")
            cells.append(metric_cell(evaluation.get("target") if evaluation else None))
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Epoch-100 Full Val200 (overall)",
            "",
            "| Method | Overall Dice | Overall hit | Hits / Findings |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    overall_rows: list[tuple[str, dict[str, Any] | None]] = []
    for arm, arm_data in summary["arms"].items():
        evaluation = arm_data["milestones"]["100"].get("evaluation")
        overall = evaluation.get("overall") if evaluation else None
        overall_rows.append((arm, overall))
    if baseline is not None:
        overall_rows.append((baseline["method"], baseline["evaluation"]["overall"]))
    best_overall_dice = best_methods(overall_rows, "mean_global_dice_per_finding")
    best_overall_hit = best_methods(overall_rows, "hit_rate")
    for method, overall in overall_rows:
        hit_count = f"{overall['hits']} / {overall['findings']}" if overall else "pending"
        lines.append(
            f"| `{method}` | "
            f"{bold(fmt(overall.get('mean_global_dice_per_finding') if overall else None), method in best_overall_dice)} "
            f"| {bold(fmt(overall.get('hit_rate') if overall else None, 3), method in best_overall_hit)} "
            f"| {bold(hit_count, method in best_overall_hit)} |"
        )

    lines.extend(
        [
            "",
            "## Checkpoint Recommendation",
            "",
            "| Arm | Selected epoch | Target Dice | Target hit | Delta from e0 | Checkpoint |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for arm, arm_data in summary["arms"].items():
        selection = arm_data.get("selection")
        if selection is None:
            lines.append(f"| `{arm}` | pending | pending | pending | pending | pending |")
            continue
        target = selection["target"]
        lines.append(
            f"| `{arm}` | {selection['epoch']} | "
            f"{target['mean_global_dice_per_finding']:.6f} | "
            f"{target['hit_rate']:.6f} | "
            f"{fmt(selection['target_dice_delta_from_epoch0'], 6)} | "
            f"`{selection['checkpoint']}` |"
        )

    lines.extend(["", "## Epoch-100 Per-category Metrics", ""])
    lines.append("Cells show `Dice / hit rate (hits/findings)`; bold marks the best Dice for that category.")
    lines.append("")
    lines.append("| Category | Findings | " + " | ".join(f"`{method}`" for method in methods) + " |")
    lines.append("| --- | ---: | " + " | ".join("---:" for _method in methods) + " |")
    categories = sorted(
        {
            category
            for arm_data in summary["arms"].values()
            for category in (
                (
                    arm_data["milestones"]["100"].get("evaluation") or {}
                ).get("categories", {})
            )
        }
        | set((baseline or {}).get("evaluation", {}).get("categories", {}))
    )
    for category in categories:
        category_rows: list[tuple[str, dict[str, Any] | None]] = []
        for arm, arm_data in summary["arms"].items():
            evaluation = arm_data["milestones"]["100"].get("evaluation")
            metric = (evaluation.get("categories", {}) if evaluation else {}).get(category)
            category_rows.append((arm, metric))
        if baseline is not None:
            category_rows.append(
                (baseline["method"], baseline["evaluation"]["categories"].get(category))
            )
        best_category_dice = best_methods(category_rows, "mean_global_dice_per_finding")
        findings = next(
            (metric["findings"] for _method, metric in category_rows if metric is not None),
            "pending",
        )
        cells = []
        for method, metric in category_rows:
            cell = metric_cell(metric)
            cells.append(bold(cell, method in best_category_dice))
        lines.append(f"| {category} | {findings} | " + " | ".join(cells) + " |")
    lines.append("")

    if summary.get("failure"):
        lines.extend(["## Failure", "", f"`{summary['failure']}`", ""])
    if summary["status"] == "selection_ready":
        lines.extend(
            [
                "## Pause Point",
                "",
                "Training and epoch-100 val200 are complete. The pipeline has stopped",
                "with target-only checkpoint recommendations ready for review.",
                "",
            ]
        )
    return "\n".join(lines)


def baseline_summary(val200_json: Path) -> dict[str, Any] | None:
    eval_json = Path(ALL_DATA_BASELINE["evaluation_json"])
    checkpoint = Path(ALL_DATA_BASELINE["checkpoint"])
    if not eval_json.is_file():
        return None
    evaluation = summarize_evaluation(eval_json, val200_json, set())
    return {
        "method": ALL_DATA_BASELINE["method"],
        "label": ALL_DATA_BASELINE["label"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "evaluation": evaluation,
        "note": "All-data public-start e5d4 reference; target/non-target is intentionally undefined.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--val200-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--current-stage", default="unknown")
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--failure", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    markdown = render_markdown(summary)
    atomic_write_json(args.output_json, summary)
    atomic_write_text(args.output_md, markdown)
    if args.snapshot_dir is not None:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.snapshot_dir / "report.json", summary)
        atomic_write_text(args.snapshot_dir / "report.md", markdown)
    print(f"status={summary['status']} report={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
