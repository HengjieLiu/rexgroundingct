#!/usr/bin/env python3
"""Summarize experiment 007 DDP batch-size results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from common import write_json


EPOCHS = [25, 50, 75, 100]
CHECKPOINT_RE = re.compile(r"checkpoint_update_(\d+)\.pth$")


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_duration(seconds: Any) -> str:
    if seconds is None:
        return "n/a"
    total = int(round(float(seconds)))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def local_mtime(path: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(path.stat().st_mtime))


def metric_files_for_arm(arm_dir: Path) -> list[Path]:
    reports = arm_dir / "reports"
    paths = sorted(reports.glob("training_metrics_segment_epoch*.json"))
    final = reports / "training_metrics.json"
    if final.is_file():
        paths.append(final)
    return paths


def collect_training_details(arm_dir: Path) -> dict[str, Any]:
    metric_paths = metric_files_for_arm(arm_dir)
    if not metric_paths:
        return {"status": "missing", "arm_dir": str(arm_dir)}

    final_metrics = read_json_if_exists(arm_dir / "reports" / "training_metrics.json")
    summary_metrics = final_metrics or read_json_if_exists(metric_paths[-1]) or {}
    updates_by_global: dict[int, dict[str, Any]] = {}
    metric_file_summaries = []
    for path in metric_paths:
        metrics = read_json_if_exists(path)
        if metrics is None:
            continue
        metric_file_summaries.append(
            {
                "path": str(path),
                "status": metrics.get("status"),
                "start_global_update": metrics.get("start_global_update"),
                "target_global_update": metrics.get("target_global_update"),
                "completed_updates": metrics.get("completed_updates"),
                "segment_updates_completed": metrics.get("segment_updates_completed"),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
                "mean_update_seconds": metrics.get("mean_update_seconds"),
                "created_at_utc": metrics.get("created_at_utc"),
            }
        )
        for item in metrics.get("updates", []):
            global_update = item.get("global_update")
            if global_update is None:
                continue
            updates_by_global.setdefault(int(global_update), item)

    cumulative_seconds_by_update: dict[int, float] = {}
    cumulative = 0.0
    for update in sorted(updates_by_global):
        cumulative += float(updates_by_global[update].get("update_seconds") or 0.0)
        cumulative_seconds_by_update[update] = cumulative

    completed_updates = summary_metrics.get(
        "completed_updates", summary_metrics.get("global_update")
    )
    if completed_updates is None and updates_by_global:
        completed_updates = max(updates_by_global)

    if completed_updates is not None and int(completed_updates) in cumulative_seconds_by_update:
        update_timer_training_seconds = cumulative_seconds_by_update[int(completed_updates)]
    elif len(metric_paths) == 1:
        update_timer_training_seconds = summary_metrics.get("elapsed_seconds")
    else:
        update_timer_training_seconds = sum(
            float(item.get("elapsed_seconds") or 0.0) for item in metric_file_summaries
        )

    segment_elapsed_seconds = [
        float(item.get("elapsed_seconds") or 0.0)
        for item in metric_file_summaries
        if "training_metrics_segment_epoch" in Path(item["path"]).name
    ]
    if segment_elapsed_seconds:
        total_training_seconds = sum(segment_elapsed_seconds)
        training_timer_source = "sum_segment_elapsed_seconds"
    elif summary_metrics.get("elapsed_seconds") is not None:
        total_training_seconds = float(summary_metrics["elapsed_seconds"])
        training_timer_source = "training_metrics_elapsed_seconds"
    else:
        total_training_seconds = update_timer_training_seconds
        training_timer_source = "sum_update_seconds"

    world_size = int(summary_metrics.get("world_size") or 1)
    batch_size = int(summary_metrics.get("batch_size") or 1)
    grad_accum = int(summary_metrics.get("grad_accum") or 1)
    global_batch_size = world_size * batch_size * grad_accum
    completed_int = int(completed_updates or 0)
    effective_samples = completed_int * global_batch_size
    samples_per_second = (
        effective_samples / total_training_seconds
        if total_training_seconds and effective_samples
        else None
    )
    seconds_per_update = (
        total_training_seconds / completed_int
        if total_training_seconds and completed_int
        else summary_metrics.get("mean_update_seconds")
    )

    memories = []
    for path in metric_paths:
        metrics = read_json_if_exists(path) or {}
        memory = metrics.get("memory", {})
        if isinstance(memory, dict) and memory.get("max_allocated_gib") is not None:
            memories.append(float(memory["max_allocated_gib"]))
    peak_allocated_gib = max(memories) if memories else None

    return {
        "status": summary_metrics.get("status"),
        "arm_dir": str(arm_dir),
        "completed_updates": completed_updates,
        "total_updates": summary_metrics.get("total_updates"),
        "world_size": world_size,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "global_batch_size": global_batch_size,
        "effective_samples": effective_samples,
        "training_seconds": total_training_seconds,
        "training_duration": fmt_duration(total_training_seconds),
        "training_timer_source": training_timer_source,
        "update_timer_training_seconds": update_timer_training_seconds,
        "update_timer_training_duration": fmt_duration(update_timer_training_seconds),
        "seconds_per_update": seconds_per_update,
        "samples_per_second": samples_per_second,
        "peak_allocated_gib": peak_allocated_gib,
        "mean_loss": summary_metrics.get("mean_loss"),
        "last_loss": summary_metrics.get("last_loss"),
        "memory": summary_metrics.get("memory", {}),
        "metric_files": metric_file_summaries,
        "updates_observed": len(updates_by_global),
        "_cumulative_seconds_by_update": cumulative_seconds_by_update,
    }


def checkpoint_timing_rows(arm_dir: Path, details: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoints_dir = arm_dir / "checkpoints"
    cumulative_seconds_by_update = details.get("_cumulative_seconds_by_update", {})
    steps_per_epoch = None
    final_metrics = read_json_if_exists(arm_dir / "reports" / "training_metrics.json")
    if final_metrics is not None:
        steps_per_epoch = final_metrics.get("steps_per_epoch")
    if steps_per_epoch is None:
        for path in metric_files_for_arm(arm_dir):
            metrics = read_json_if_exists(path) or {}
            if metrics.get("steps_per_epoch") is not None:
                steps_per_epoch = metrics["steps_per_epoch"]
                break
    steps_per_epoch = int(steps_per_epoch or 100)

    rows = []
    for path in sorted(checkpoints_dir.glob("checkpoint_update_*.pth")):
        match = CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        update = int(match.group(1))
        training_seconds = cumulative_seconds_by_update.get(update)
        rows.append(
            {
                "epoch": update / steps_per_epoch,
                "update": update,
                "cumulative_update_timer_seconds": training_seconds,
                "cumulative_update_timer_duration": fmt_duration(training_seconds),
                "seconds_per_update_cumulative": (
                    training_seconds / update if training_seconds else None
                ),
                "checkpoint_mtime_local": local_mtime(path),
                "checkpoint_path": str(path),
            }
        )
    return rows


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
    details = collect_training_details(arm_dir)
    if details.get("status") == "missing":
        return details
    metrics = read_json_if_exists(arm_dir / "reports" / "training_metrics.json")
    if metrics is None:
        return {"status": "missing"}
    segment_metrics = {}
    for epoch in EPOCHS:
        path = arm_dir / "reports" / f"training_metrics_segment_epoch{epoch:03d}.json"
        segment = read_json_if_exists(path)
        if segment is not None:
            losses = [
                float(item["loss"])
                for item in segment.get("updates", [])
                if item.get("loss") is not None
            ]
            segment_metrics[str(epoch)] = {
                "path": str(path),
                "completed_updates": segment.get("completed_updates"),
                "segment_updates_completed": segment.get("segment_updates_completed"),
                "mean_update_seconds": segment.get("mean_update_seconds"),
                "last_loss": segment.get("last_loss"),
                "first100_mean_loss": float(np.mean(losses[:100])) if losses else None,
                "last100_mean_loss": float(np.mean(losses[-100:])) if losses else None,
            }
    details.update(
        {
            "status": metrics.get("status"),
            "completed_updates": metrics.get("completed_updates", metrics.get("global_update")),
            "total_updates": metrics.get("total_updates"),
            "world_size": metrics.get("world_size"),
            "batch_size": metrics.get("batch_size"),
            "grad_accum": metrics.get("grad_accum"),
            "mean_update_seconds": metrics.get("mean_update_seconds"),
            "updates_per_second": metrics.get("updates_per_second"),
            "mean_loss": metrics.get("mean_loss"),
            "last_loss": metrics.get("last_loss"),
            "memory": metrics.get("memory", {}),
            "segments": segment_metrics,
            "checkpoint_timings": checkpoint_timing_rows(arm_dir, details),
        }
    )
    return details


def arm_dir_from_training_metrics(path: Path) -> Path:
    if path.name.startswith("training_metrics") and path.parent.name == "reports":
        return path.parent.parent
    raise ValueError(f"reference training path must be .../reports/training_metrics*.json: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--group-dir", type=Path, required=True)
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
            f"ddp_bs4 epoch {epoch} val200",
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
        "exp_dir": str(args.exp_dir),
        "group_dir": str(args.group_dir),
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
        "# Experiment 007 DDP Batch4 Update-Matched Report",
        "",
        f"Run group: `{args.group_dir.name}`",
        "",
        "## Val200 Progress",
        "",
        "| Checkpoint | Comparison role | Dice | Hit rate | Hits / findings |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    roles = {
        25: "sample-matched to exp006 epoch100",
        50: "intermediate",
        75: "intermediate",
        100: "update-matched to exp006 epoch100",
    }
    for epoch in EPOCHS:
        metric = val200[str(epoch)]
        lines.append(
            f"| {epoch} | {roles[epoch]} | {fmt(metric.get('dice'))} | "
            f"{fmt(metric.get('hit_rate'))} | {fmt(metric.get('hits'), 0)} / "
            f"{fmt(metric.get('findings'), 0)} |"
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
            f"- Completed updates: `{training.get('completed_updates')}` / `{training.get('total_updates')}`",
            f"- World size: `{training.get('world_size')}`",
            f"- Per-GPU batch size: `{training.get('batch_size')}`",
            f"- Gradient accumulation: `{training.get('grad_accum')}`",
            f"- Last loss: `{fmt(training.get('last_loss'))}`",
            f"- Final segment seconds/update: `{fmt(training.get('mean_update_seconds'), 3)}`",
        ]
    )

    training_models = {"exp007_ddp_bs4_e5_d4": training}
    training_models.update(payload["reference_trainings"])
    if training_models:
        lines.extend(
            [
                "",
                "## Training Time",
                "",
                "Training time excludes validation pauses between exp007 segments.",
                "Model-level training time uses training-script elapsed timers. Checkpoint rows use cumulative update timers plus checkpoint mtimes, because exp006 did not emit separate elapsed metrics at every checkpoint.",
                "",
                "| Model | Global batch | Updates | Effective samples | Training time | Sec/update | Samples/sec | Peak allocated GiB |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, item in training_models.items():
            lines.append(
                f"| {label} | {fmt(item.get('global_batch_size'), 0)} | "
                f"{fmt(item.get('completed_updates'), 0)} | "
                f"{fmt(item.get('effective_samples'), 0)} | "
                f"{item.get('training_duration', 'n/a')} | "
                f"{fmt(item.get('seconds_per_update'), 3)} | "
                f"{fmt(item.get('samples_per_second'), 3)} | "
                f"{fmt(item.get('peak_allocated_gib'), 2)} |"
            )

        lines.extend(
            [
                "",
                "## Checkpoint Training Time",
                "",
                "| Model | Epoch | Update | Cumulative update-timer time | Cumulative sec/update | Checkpoint saved |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for label, item in training_models.items():
            for row in item.get("checkpoint_timings", []):
                lines.append(
                    f"| {label} | {fmt(row.get('epoch'), 1)} | "
                    f"{fmt(row.get('update'), 0)} | "
                    f"{row.get('cumulative_update_timer_duration', 'n/a')} | "
                    f"{fmt(row.get('seconds_per_update_cumulative'), 3)} | "
                    f"{row.get('checkpoint_mtime_local', 'n/a')} |"
                )

    if training.get("segments"):
        lines.extend(
            [
                "",
                "| Segment end epoch | Completed updates | Last100 loss | Sec/update |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for epoch in EPOCHS:
            segment = training["segments"].get(str(epoch), {})
            lines.append(
                f"| {epoch} | {fmt(segment.get('completed_updates'), 0)} | "
                f"{fmt(segment.get('last100_mean_loss'))} | "
                f"{fmt(segment.get('mean_update_seconds'), 3)} |"
            )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
