#!/usr/bin/env python3
"""Summarize experiment 007 DDP batch4 continuation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import sha256_file, write_json
from summarize_007_results import (
    EPOCHS,
    arm_dir_from_training_metrics,
    fmt,
    metric_row,
    training_row,
)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def source_checkpoint_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--absolute-epoch-offset", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--reference", action="append", default=[], help="label=/path/to/summary.json")
    parser.add_argument(
        "--reference-training",
        action="append",
        default=[],
        help="label=/path/to/reports/training_metrics.json",
    )
    args = parser.parse_args()

    arm_dir = args.group_dir / "ddp_bs4"
    val200 = {
        str(epoch): metric_row(
            f"cont relative epoch {epoch} val200",
            arm_dir
            / f"eval_epoch{epoch:03d}_val200"
            / "reports"
            / "val_quick_global_eval_summary.json",
        )
        for epoch in EPOCHS
    }
    references = {}
    for item in args.reference:
        label, path = item.split("=", 1)
        references[label] = metric_row(label, Path(path))
    reference_trainings = {}
    for item in args.reference_training:
        label, path = item.split("=", 1)
        reference_trainings[label] = training_row(arm_dir_from_training_metrics(Path(path)))

    payload = {
        "experiment": "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched",
        "run_type": "continue100_from_ddp_epoch100",
        "exp_dir": str(args.exp_dir),
        "group_dir": str(args.group_dir),
        "source_run_dir": str(args.source_run_dir),
        "source_checkpoint": source_checkpoint_record(args.source_checkpoint),
        "absolute_epoch_offset": args.absolute_epoch_offset,
        "continuation_policy": "weights_only_init_checkpoint_fresh_optimizer_scaler_warmup_poly_10000_updates",
        "training": training_row(arm_dir),
        "reference_trainings": reference_trainings,
        "val200": val200,
        "references": references,
    }
    payload["training"].pop("_cumulative_seconds_by_update", None)
    for training_payload in payload["reference_trainings"].values():
        training_payload.pop("_cumulative_seconds_by_update", None)
    write_json(args.output_json, payload)

    lines = [
        "# Experiment 007 DDP Batch4 Continuation Report",
        "",
        f"Run group: `{args.group_dir.name}`",
        f"Source run: `{args.source_run_dir}`",
        f"Source checkpoint: `{args.source_checkpoint}`",
        "",
        "Continuation policy: load source network weights only, then reset optimizer, AMP scaler, LR schedule, and update counter.",
        "",
        "## Val200 Progress",
        "",
        "| Relative epoch | Absolute epoch | Dice | Hit rate | Hits / findings |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for epoch in EPOCHS:
        metric = val200[str(epoch)]
        lines.append(
            f"| {epoch} | {args.absolute_epoch_offset + epoch} | "
            f"{fmt(metric.get('dice'))} | {fmt(metric.get('hit_rate'))} | "
            f"{fmt(metric.get('hits'), 0)} / {fmt(metric.get('findings'), 0)} |"
        )

    if references:
        lines.extend(
            [
                "",
                "## References",
                "",
                "| Model | Dice | Hit rate | Hits / findings |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for label, metric in references.items():
            lines.append(
                f"| {label} | {fmt(metric.get('dice'))} | {fmt(metric.get('hit_rate'))} | "
                f"{fmt(metric.get('hits'), 0)} / {fmt(metric.get('findings'), 0)} |"
            )

    training = payload["training"]
    lines.extend(
        [
            "",
            "## Training",
            "",
            f"- Status: `{training.get('status')}`",
            f"- Completed continuation updates: `{training.get('completed_updates')}` / `{training.get('total_updates')}`",
            f"- World size: `{training.get('world_size')}`",
            f"- Per-GPU batch size: `{training.get('batch_size')}`",
            f"- Effective global batch size: `{training.get('global_batch_size')}`",
            f"- Gradient accumulation: `{training.get('grad_accum')}`",
            f"- Last loss: `{fmt(training.get('last_loss'))}`",
            f"- Training time: `{training.get('training_duration', 'n/a')}`",
            f"- Seconds/update: `{fmt(training.get('seconds_per_update'), 3)}`",
            f"- Peak allocated GiB: `{fmt(training.get('peak_allocated_gib'), 2)}`",
        ]
    )
    if training.get("segments"):
        lines.extend(
            [
                "",
                "| Relative segment end epoch | Absolute epoch | Completed updates | Last100 loss | Sec/update |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for epoch in EPOCHS:
            segment = training["segments"].get(str(epoch), {})
            lines.append(
                f"| {epoch} | {args.absolute_epoch_offset + epoch} | "
                f"{fmt(segment.get('completed_updates'), 0)} | "
                f"{fmt(segment.get('last100_mean_loss'))} | "
                f"{fmt(segment.get('mean_update_seconds'), 3)} |"
            )

    if reference_trainings:
        lines.extend(
            [
                "",
                "## Training Time References",
                "",
                "| Model | Global batch | Updates | Training time | Sec/update | Peak allocated GiB |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, item in reference_trainings.items():
            lines.append(
                f"| {label} | {fmt(item.get('global_batch_size'), 0)} | "
                f"{fmt(item.get('completed_updates'), 0)} | "
                f"{item.get('training_duration', 'n/a')} | "
                f"{fmt(item.get('seconds_per_update'), 3)} | "
                f"{fmt(item.get('peak_allocated_gib'), 2)} |"
            )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
