#!/usr/bin/env python3
"""Summarize experiment 006 LR-ablation training and validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import write_json


ARMS = [
    "v123_cached_e7_d6",
    "v123_cached_e7_d4",
    "v123_cached_e6_d4",
    "v123_cached_e5_d4",
]
VAL20_EPOCHS = [5, 20, 40, 60, 80, 100]


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def metric_row(label: str, summary_path: Path) -> dict[str, Any]:
    summary = read_json_if_exists(summary_path)
    if summary is None:
        return {
            "label": label,
            "summary_path": str(summary_path),
            "status": "missing",
        }
    return {
        "label": label,
        "summary_path": str(summary_path),
        "status": "complete",
        "dice": summary.get("mean_global_dice_per_finding"),
        "hit_rate": summary.get("hit_rate"),
        "hits": summary.get("total_hits"),
        "findings": summary.get("total_findings"),
        "cases": summary.get("total_cases"),
    }


def training_row(arm_dir: Path) -> dict[str, Any]:
    metrics = read_json_if_exists(arm_dir / "reports" / "training_metrics.json")
    if metrics is None:
        return {"status": "missing"}
    updates = metrics.get("updates", [])
    losses = [float(item["loss"]) for item in updates if item.get("loss") is not None]
    first_window = losses[:100]
    last_window = losses[-100:]
    return {
        "status": metrics.get("status"),
        "completed_updates": metrics.get("completed_updates", metrics.get("global_update")),
        "mean_loss": metrics.get("mean_loss"),
        "last_loss": metrics.get("last_loss"),
        "min_loss": metrics.get("min_loss"),
        "max_loss": metrics.get("max_loss"),
        "mean_update_seconds": metrics.get("mean_update_seconds"),
        "updates_per_second": metrics.get("updates_per_second"),
        "first100_mean_loss": float(np.mean(first_window)) if first_window else None,
        "last100_mean_loss": float(np.mean(last_window)) if last_window else None,
        "memory": metrics.get("memory", {}),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--reference", action="append", default=[], help="label=/path/to/summary.json")
    args = parser.parse_args()

    arms: dict[str, Any] = {}
    for arm in ARMS:
        arm_dir = args.group_dir / arm
        val20 = {
            str(epoch): metric_row(
                f"{arm} epoch {epoch} val20",
                arm_dir
                / f"eval_epoch{epoch:03d}_val20"
                / "reports"
                / "val_quick_global_eval_summary.json",
            )
            for epoch in VAL20_EPOCHS
        }
        val200 = metric_row(
            f"{arm} epoch 100 val200",
            arm_dir
            / "eval_epoch100_val200"
            / "reports"
            / "val_quick_global_eval_summary.json",
        )
        arms[arm] = {
            "training": training_row(arm_dir),
            "val20": val20,
            "val200": val200,
        }

    references = {}
    for item in args.reference:
        label, path = item.split("=", 1)
        references[label] = metric_row(label, Path(path))

    payload = {
        "experiment": "006_voxtell_cached_native_v123_lr_ablation",
        "exp_dir": str(args.exp_dir),
        "group_dir": str(args.group_dir),
        "arms": arms,
        "references": references,
    }
    write_json(args.output_json, payload)

    lines = [
        "# Experiment 006 LR Ablation Report",
        "",
        f"Run group: `{args.group_dir.name}`",
        "",
        "## Val200 Epoch 100",
        "",
        "| Model | Dice | Hit rate | Hits / findings | Train status | Last100 loss | Sec/update |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for arm in ARMS:
        result = arms[arm]
        metric = result["val200"]
        training = result["training"]
        lines.append(
            f"| {arm} | {fmt(metric.get('dice'))} | {fmt(metric.get('hit_rate'))} | "
            f"{fmt(metric.get('hits'), 0)} / {fmt(metric.get('findings'), 0)} | "
            f"{fmt(training.get('status'))} | {fmt(training.get('last100_mean_loss'))} | "
            f"{fmt(training.get('mean_update_seconds'), 3)} |"
        )

    if references:
        lines.extend(["", "## References", "", "| Model | Dice | Hit rate | Hits / findings |", "| --- | ---: | ---: | ---: |"])
        for label, metric in references.items():
            lines.append(
                f"| {label} | {fmt(metric.get('dice'))} | {fmt(metric.get('hit_rate'))} | "
                f"{fmt(metric.get('hits'), 0)} / {fmt(metric.get('findings'), 0)} |"
            )

    lines.extend(["", "## Val20 Progress", ""])
    for arm in ARMS:
        lines.extend(
            [
                f"### {arm}",
                "",
                "| Epoch | Dice | Hit rate | Hits / findings |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for epoch in VAL20_EPOCHS:
            metric = arms[arm]["val20"][str(epoch)]
            lines.append(
                f"| {epoch} | {fmt(metric.get('dice'))} | {fmt(metric.get('hit_rate'))} | "
                f"{fmt(metric.get('hits'), 0)} / {fmt(metric.get('findings'), 0)} |"
            )
        lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
