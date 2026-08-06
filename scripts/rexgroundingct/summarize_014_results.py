#!/usr/bin/env python3
"""Summarize experiment 014 DDP batch16 milestone results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    number = maybe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def parse_reference(values: list[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in values:
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if path.is_file():
            output[label] = read_json(path)
    return output


def parse_reference_training(values: list[str]) -> dict[str, dict[str, Any]]:
    return parse_reference(values)


def eval_summary_path(run_dir: Path, epoch: int) -> Path:
    return (
        run_dir
        / f"eval_epoch{epoch:03d}_val200"
        / "reports"
        / "val_quick_global_eval_summary.json"
    )


def training_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "reports" / "training_metrics.json"
    return read_json(path) if path.is_file() else {}


def summarize_training(metrics: dict[str, Any]) -> dict[str, Any]:
    world_size = int(metrics.get("world_size") or 1)
    batch_size = int(metrics.get("batch_size") or 1)
    grad_accum = int(metrics.get("grad_accum") or 1)
    completed = int(metrics.get("completed_updates") or 0)
    global_batch = world_size * batch_size * grad_accum
    elapsed = maybe_float(metrics.get("elapsed_seconds"))
    effective_samples = completed * global_batch
    samples_per_second = (
        effective_samples / elapsed
        if elapsed is not None and elapsed > 0 and effective_samples
        else None
    )
    return {
        "world_size": world_size,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "global_batch_size": global_batch,
        "completed_updates": completed,
        "effective_samples": effective_samples,
        "elapsed_seconds": elapsed,
        "mean_update_seconds": metrics.get("mean_update_seconds"),
        "samples_per_second": samples_per_second,
        "memory": metrics.get("memory", {}),
        "last_loss": metrics.get("last_loss"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--reference-training", action="append", default=[])
    args = parser.parse_args()

    run_dir = args.group_dir / "ddp_bs16"
    references = parse_reference(args.reference)
    reference_training = parse_reference_training(args.reference_training)
    milestones: list[dict[str, Any]] = []
    for epoch in (25, 50, 75, 100):
        path = eval_summary_path(run_dir, epoch)
        if not path.is_file():
            continue
        data = read_json(path)
        milestones.append(
            {
                "epoch": epoch,
                "update": epoch * 100,
                "summary_path": str(path),
                "mean_global_dice_per_finding": data.get("mean_global_dice_per_finding"),
                "hit_rate": data.get("hit_rate"),
                "hits": data.get("hits"),
                "total_findings": data.get("total_findings"),
                "total_cases": data.get("total_cases"),
            }
        )

    training = summarize_training(training_summary(run_dir))
    summary = {
        "experiment": "014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched",
        "exp_dir": str(args.exp_dir),
        "group_dir": str(args.group_dir),
        "run_dir": str(run_dir),
        "training": training,
        "milestones": milestones,
        "references": references,
        "reference_training": {
            label: summarize_training(data)
            for label, data in reference_training.items()
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Experiment 014 DDP Batch16 Update-Matched Report",
        "",
        f"Run group: `{args.group_dir.name}`",
        "",
        "## Val200 Progress",
        "",
        "| Checkpoint | Comparison role | Dice | Hit rate | Hits / findings |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    roles = {
        25: "sample-matched to exp007 epoch100",
        50: "intermediate",
        75: "intermediate",
        100: "update-matched to exp007 epoch100",
    }
    for item in milestones:
        hits = item.get("hits")
        total = item.get("total_findings")
        hit_text = f"{hits} / {total}" if hits is not None and total is not None else "n/a"
        lines.append(
            "| {epoch} | {role} | {dice} | {hit_rate} | {hits} |".format(
                epoch=item["epoch"],
                role=roles.get(int(item["epoch"]), "milestone"),
                dice=fmt(item.get("mean_global_dice_per_finding")),
                hit_rate=fmt(item.get("hit_rate")),
                hits=hit_text,
            )
        )
    if not milestones:
        lines.append("| n/a | no val200 milestones complete yet | n/a | n/a | n/a |")

    lines.extend(["", "## References", ""])
    if references:
        lines.extend(
            [
                "| Model | Dice | Hit rate | Hits / findings |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for label, data in references.items():
            hits = data.get("hits")
            total = data.get("total_findings")
            hit_text = f"{hits} / {total}" if hits is not None and total is not None else "n/a"
            lines.append(
                "| {label} | {dice} | {hit_rate} | {hits} |".format(
                    label=label,
                    dice=fmt(data.get("mean_global_dice_per_finding")),
                    hit_rate=fmt(data.get("hit_rate")),
                    hits=hit_text,
                )
            )
    else:
        lines.append("No reference summaries were available.")

    lines.extend(
        [
            "",
            "## Training",
            "",
            f"- Completed updates: `{training.get('completed_updates')}`",
            f"- World size: `{training.get('world_size')}`",
            f"- Per-GPU batch size: `{training.get('batch_size')}`",
            f"- Gradient accumulation: `{training.get('grad_accum')}`",
            f"- Effective global batch size: `{training.get('global_batch_size')}`",
            f"- Last loss: `{fmt(training.get('last_loss'))}`",
            f"- Mean seconds/update: `{fmt(training.get('mean_update_seconds'), 3)}`",
            f"- Samples/second: `{fmt(training.get('samples_per_second'), 3)}`",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
