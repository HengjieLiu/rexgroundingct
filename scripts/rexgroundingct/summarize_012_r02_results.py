#!/usr/bin/env python3
"""Build the atomic live report and target-only selection for Exp012 Run 2."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import read_json, sha256_file, utc_now_iso
from prepare_012_r02_replay_ratio import ARM_TARGETS, PROFILE_ID


EXPERIMENT_ID = "012_voxtell_category_specialists_replay50_cont100"
MILESTONES = (0, 5, 20, 40, 60, 80, 100)
EVAL_SCOPE = {
    0: "union",
    5: "target",
    20: "union",
    40: "union",
    60: "union",
    80: "union",
    100: "val200",
}
EXPECTED_BASELINES = {
    "category_2d_replay50": {"dice": 0.37841685089239674, "hits": 113, "findings": 132},
    "category_1alldiffuse_replay25": {
        "dice": 0.16273172502327543,
        "hits": 19,
        "findings": 52,
    },
    "category_1alldiffuse_replay10": {
        "dice": 0.16273172502327543,
        "hits": 19,
        "findings": 52,
    },
    "category_1alldiffuse_replay00": {
        "dice": 0.16273172502327543,
        "hits": 19,
        "findings": 52,
    },
}
EXPECTED_VAL80_TOTAL = {
    "dice": 0.30867434411682876,
    "hits": 133,
    "findings": 195,
    "cases": 80,
}
EXPECTED_VAL80_NON_TARGET_FINDINGS = {
    "category_2d_replay50": 155,
    "category_1alldiffuse_replay25": 143,
    "category_1alldiffuse_replay10": 143,
    "category_1alldiffuse_replay00": 143,
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


def evaluation_records(
    eval_json: Path, dataset_json: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = read_json(dataset_json).get("test")
    evaluation = read_json(eval_json)
    cases = evaluation.get("cases")
    if not isinstance(dataset, list) or not isinstance(cases, list):
        raise ValueError(
            f"Invalid evaluation or dataset JSON: {eval_json} {dataset_json}"
        )
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
            raise ValueError(
                f"{case_name}: evaluator finding keys do not match dataset"
            )
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
    val80_finding_keys: set[tuple[str, str]],
    *,
    include_val80_metrics: bool = True,
) -> dict[str, Any]:
    records, raw_summary = evaluation_records(eval_json, dataset_json)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
    target = [record for record in records if record["category"] in target_categories]
    non_target = [
        record for record in records if record["category"] not in target_categories
    ]
    val80_records: list[dict[str, Any]] = []
    sentinel: list[dict[str, Any]] = []
    if include_val80_metrics:
        records_by_key = {
            (record["case_name"], record["finding_index"]): record for record in records
        }
        missing = val80_finding_keys - set(records_by_key)
        if missing:
            examples = ", ".join(
                f"{case}:{index}" for case, index in sorted(missing)[:5]
            )
            raise ValueError(
                f"Evaluation is missing {len(missing)} fixed-val80 findings; examples: {examples}"
            )
        val80_records = [records_by_key[key] for key in sorted(val80_finding_keys)]
        sentinel = [
            record
            for record in val80_records
            if record["category"] not in target_categories
        ]
    return {
        "evaluation_json": str(eval_json),
        "evaluation_json_sha256": sha256_file(eval_json),
        "dataset_json": str(dataset_json),
        "dataset_json_sha256": sha256_file(dataset_json),
        "overall": {
            "mean_global_dice_per_finding": raw_summary.get(
                "mean_global_dice_per_finding"
            ),
            "hit_rate": raw_summary.get("hit_rate"),
            "hits": raw_summary.get("total_hits"),
            "findings": raw_summary.get("total_findings"),
            "cases": raw_summary.get("total_cases"),
        },
        "target": metric_summary(target),
        "non_target": metric_summary(non_target),
        "val80_total": metric_summary(val80_records) if include_val80_metrics else None,
        "sentinel_non_target": metric_summary(sentinel)
        if include_val80_metrics
        else None,
        "categories": {
            category: metric_summary(category_records)
            for category, category_records in sorted(by_category.items())
        },
    }


def training_summary(group_dir: Path, arm: str, epoch: int) -> dict[str, Any] | None:
    if epoch == 0:
        return None
    path = (
        group_dir / arm / "reports" / f"training_metrics_segment_epoch{epoch:03d}.json"
    )
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


def checkpoint_path(
    group_dir: Path, source_checkpoint: Path, arm: str, epoch: int
) -> Path:
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
            if hit > best_hit + 1e-12 or (
                abs(hit - best_hit) <= 1e-12 and epoch < best_epoch
            ):
                best_epoch, best_metric = epoch, metric

    baseline_evaluation = milestones.get("0", {}).get("evaluation") or {}
    baseline = baseline_evaluation.get("target")
    return {
        "epoch": best_epoch,
        "checkpoint": str(
            checkpoint_path(group_dir, source_checkpoint, arm, best_epoch)
        ),
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
    scope = EVAL_SCOPE[epoch]
    if scope == "val200":
        return val200_json
    return Path(subset_manifest["arms"][arm][scope]["path"])


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    subset_manifest = read_json(args.subset_manifest)
    val80 = read_json(args.val80_json)["test"]
    val80_finding_keys = {
        (entry["name"], str(finding_index))
        for entry in val80
        for finding_index in entry.get("findings", {})
    }
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
                "checkpoint": str(
                    checkpoint_path(args.group_dir, args.source_checkpoint, arm, epoch)
                ),
                "training": training_summary(args.group_dir, arm, epoch),
                "evaluation": None,
            }
            if eval_json.is_file():
                dataset = resolve_dataset(epoch, arm, subset_manifest, args.val200_json)
                item["evaluation"] = summarize_evaluation(
                    eval_json,
                    dataset,
                    set(target_tuple),
                    val80_finding_keys,
                    include_val80_metrics=scope != "target",
                )
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

    historical_references: dict[str, Any] = {}
    if args.run1_reference_json is not None:
        reference_report = read_json(args.run1_reference_json)
        reference_arm = reference_report.get("arms", {}).get(
            "category_1alldiffuse_replay50"
        )
        if reference_arm is None:
            raise ValueError("Run-1 reference report lacks diffuse replay50 arm")
        historical_references["category_1alldiffuse_replay50"] = {
            "source_report": str(args.run1_reference_json),
            "source_report_sha256": sha256_file(args.run1_reference_json),
            "source_run_group": reference_report.get("run_group"),
            "relationship": "independent historical schedule; not sample-paired",
            "milestones": reference_arm.get("milestones", {}),
            "selection": reference_arm.get("selection"),
        }

    status = "selection_ready" if complete else "active"
    if args.failure:
        status = "failed"
    return {
        "schema_version": 2,
        "experiment": EXPERIMENT_ID,
        "profile": PROFILE_ID,
        "experiment_status": "active",
        "run_group": args.group_dir.name,
        "group_dir": str(args.group_dir),
        "updated_at_utc": utc_now_iso(),
        "status": status,
        "current_stage": args.current_stage,
        "selection_uses_forgetting_guard": False,
        "sentinel_is_diagnostic_only": True,
        "failure": args.failure,
        "arms": arms,
        "historical_references": historical_references,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def target_cell(item: dict[str, Any]) -> str:
    evaluation = item.get("evaluation")
    if not evaluation or not evaluation.get("target"):
        return "pending"
    metric = evaluation["target"]
    return (
        f"{metric['mean_global_dice_per_finding']:.4f} / "
        f"{metric['hit_rate']:.3f} ({metric['hits']}/{metric['findings']})"
    )


def metric_cell(metric: dict[str, Any] | None) -> str:
    if metric is None:
        return "pending"
    return (
        f"{metric['mean_global_dice_per_finding']:.4f} / "
        f"{metric['hit_rate']:.3f} ({metric['hits']}/{metric['findings']})"
    )


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Exp012 Run 2 — 2d and Diffuse Replay-Ratio Progress",
        "",
        f"- Status: `{summary['status']}`",
        f"- Run group: `{summary['run_group']}`",
        f"- Profile: `{summary['profile']}`",
        "- Exp012 umbrella status: `active` (additional runs are expected).",
        f"- Current stage: `{summary['current_stage']}`",
        f"- Updated: `{summary['updated_at_utc']}`",
        "- Selection: target Dice only; sentinel and non-target results are diagnostic.",
        "",
        "## Target Milestones",
        "",
        "Cells show `Dice / hit rate (hits/findings)`.",
        "",
        "| Arm | e0 | e5 | e20 | e40 | e60 | e80 | e100 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, arm_data in summary["arms"].items():
        cells = [
            target_cell(arm_data["milestones"][str(epoch)]) for epoch in MILESTONES
        ]
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    reference = summary.get("historical_references", {}).get(
        "category_1alldiffuse_replay50"
    )
    lines.extend(
        [
            "",
            "## Diffuse Replay-ratio Comparison",
            "",
            "Run-1 replay50 is a frozen historical reference produced with an independent",
            "schedule. It is not sample-paired with the three Run-2 diffuse arms.",
            "",
            "| Arm | Schedule | e0 | e5 | e20 | e40 | e60 | e80 | e100 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if reference:
        cells = [
            target_cell(reference.get("milestones", {}).get(str(epoch), {}))
            for epoch in MILESTONES
        ]
        lines.append(
            "| `category_1alldiffuse_replay50` | historical independent | "
            + " | ".join(cells)
            + " |"
        )
    else:
        lines.append(
            "| `category_1alldiffuse_replay50` | historical independent | "
            + " | ".join(["pending"] * len(MILESTONES))
            + " |"
        )
    for arm in (
        "category_1alldiffuse_replay25",
        "category_1alldiffuse_replay10",
        "category_1alldiffuse_replay00",
    ):
        cells = [
            target_cell(summary["arms"][arm]["milestones"][str(epoch)])
            for epoch in MILESTONES
        ]
        lines.append(f"| `{arm}` | Run 2 independent | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Val80 Total — Cross-arm Comparable",
            "",
            "The same fixed 80 cases and 195 findings are used for every arm.",
            "",
            "| Arm | e0 | e20 | e40 | e60 | e80 | e100 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for arm, arm_data in summary["arms"].items():
        cells = []
        for epoch in (0, 20, 40, 60, 80, 100):
            evaluation = arm_data["milestones"][str(epoch)].get("evaluation")
            metric = evaluation.get("val80_total") if evaluation else None
            cells.append(metric_cell(metric))
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Val80 Non-target — Within-arm Forgetting",
            "",
            "Each arm excludes its own target categories, so denominators differ and rows",
            "should be compared across epochs within an arm, not across arms.",
            "",
            "| Arm | e0 | e20 | e40 | e60 | e80 | e100 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for arm, arm_data in summary["arms"].items():
        cells = []
        for epoch in (0, 20, 40, 60, 80, 100):
            evaluation = arm_data["milestones"][str(epoch)].get("evaluation")
            metric = evaluation.get("sentinel_non_target") if evaluation else None
            cells.append(metric_cell(metric))
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Epoch-100 Full Val200",
            "",
            "| Arm | Overall Dice | Overall hit | Non-target Dice | Non-target hit |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm, arm_data in summary["arms"].items():
        evaluation = arm_data["milestones"]["100"].get("evaluation")
        overall = evaluation.get("overall") if evaluation else None
        non_target = evaluation.get("non_target") if evaluation else None
        lines.append(
            f"| `{arm}` | {fmt(overall.get('mean_global_dice_per_finding') if overall else None)} "
            f"| {fmt(overall.get('hit_rate') if overall else None, 3)} "
            f"| {fmt(non_target.get('mean_global_dice_per_finding') if non_target else None)} "
            f"| {fmt(non_target.get('hit_rate') if non_target else None, 3)} |"
        )

    lines.extend(
        [
            "",
            "## Checkpoint Recommendation",
            "",
            "| Arm | Selected epoch | Target Dice | Delta from e0 | Checkpoint |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for arm, arm_data in summary["arms"].items():
        selection = arm_data.get("selection")
        if selection is None:
            lines.append(f"| `{arm}` | pending | pending | pending | pending |")
            continue
        lines.append(
            f"| `{arm}` | {selection['epoch']} | "
            f"{selection['target']['mean_global_dice_per_finding']:.6f} | "
            f"{fmt(selection['target_dice_delta_from_epoch0'], 6)} | "
            f"`{selection['checkpoint']}` |"
        )

    lines.extend(["", "## Epoch-100 Per-category Metrics", ""])
    for arm, arm_data in summary["arms"].items():
        lines.extend(
            [
                f"### `{arm}`",
                "",
                "| Category | Findings | Dice | Hit rate |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        evaluation = arm_data["milestones"]["100"].get("evaluation")
        categories = evaluation.get("categories", {}) if evaluation else {}
        if not categories:
            lines.append("| pending | pending | pending | pending |")
        else:
            for category, metric in categories.items():
                lines.append(
                    f"| {category} | {metric['findings']} | "
                    f"{metric['mean_global_dice_per_finding']:.6f} | {metric['hit_rate']:.6f} |"
                )
        lines.append("")

    if summary.get("failure"):
        lines.extend(["## Failure", "", f"`{summary['failure']}`", ""])
    if summary["status"] == "selection_ready":
        lines.extend(
            [
                "## Pause Point",
                "",
                "Training and epoch-100 val200 are complete. The pipeline has intentionally",
                "stopped before evaluating any selected pre-100 checkpoint on val200.",
                "",
            ]
        )
    return "\n".join(lines)


def verify_epoch0(summary: dict[str, Any]) -> None:
    errors = []
    val80_dice = []
    for arm, expected in EXPECTED_BASELINES.items():
        evaluation = summary["arms"][arm]["milestones"]["0"].get("evaluation")
        target = evaluation.get("target") if evaluation else None
        if target is None:
            errors.append(f"{arm}: missing epoch-0 target metrics")
            continue
        if abs(float(target["mean_global_dice_per_finding"]) - expected["dice"]) > 1e-4:
            errors.append(f"{arm}: epoch-0 Dice mismatch")
        if (
            int(target["hits"]) != expected["hits"]
            or int(target["findings"]) != expected["findings"]
        ):
            errors.append(f"{arm}: epoch-0 hit/findings mismatch")
        val80_total = evaluation.get("val80_total") if evaluation else None
        if val80_total is None:
            errors.append(f"{arm}: missing epoch-0 total-val80 metrics")
            continue
        val80_dice.append(float(val80_total["mean_global_dice_per_finding"]))
        if (
            int(val80_total["hits"]) != EXPECTED_VAL80_TOTAL["hits"]
            or int(val80_total["findings"]) != EXPECTED_VAL80_TOTAL["findings"]
            or int(val80_total["cases"]) != EXPECTED_VAL80_TOTAL["cases"]
        ):
            errors.append(f"{arm}: epoch-0 total-val80 completeness mismatch")
        if (
            abs(
                float(val80_total["mean_global_dice_per_finding"])
                - EXPECTED_VAL80_TOTAL["dice"]
            )
            > 1e-4
        ):
            errors.append(f"{arm}: epoch-0 total-val80 Dice mismatch")
        sentinel = evaluation.get("sentinel_non_target") if evaluation else None
        if (
            sentinel is None
            or int(sentinel["findings"]) != EXPECTED_VAL80_NON_TARGET_FINDINGS[arm]
        ):
            errors.append(f"{arm}: epoch-0 non-target-val80 denominator mismatch")
    if val80_dice and max(val80_dice) - min(val80_dice) > 1e-4:
        errors.append("epoch-0 total-val80 Dice differs across arms by more than 1e-4")
    if errors:
        raise ValueError("; ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--val80-json", type=Path, required=True)
    parser.add_argument("--val200-json", type=Path, required=True)
    parser.add_argument("--run1-reference-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--current-stage", default="unknown")
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--verify-epoch0", action="store_true")
    parser.add_argument("--failure", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    if args.verify_epoch0:
        verify_epoch0(summary)
    markdown = render_markdown(summary)
    atomic_write_json(args.output_json, summary)
    atomic_write_text(args.output_md, markdown)
    if args.snapshot_dir is not None:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if not (args.snapshot_dir / "report.complete").is_file():
            atomic_write_json(args.snapshot_dir / "report.json", summary)
            atomic_write_text(args.snapshot_dir / "report.md", markdown)
    print(f"status={summary['status']} report={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
