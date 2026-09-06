#!/usr/bin/env python3
"""Render the live Exp007 phase-3 training and fixed-val200 report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import read_json, sha256_file, utc_now_iso


EXPERIMENT_ID = "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched"
MILESTONES = tuple(range(10, 101, 10))
ALL_CATEGORIES = (
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h",
)
HIT_THRESHOLD = 0.1
EXPECTED_CASES = 200
EXPECTED_FINDINGS = 381


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def metric_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    values = list(records)
    if not values:
        return None
    hits = sum(int(float(item["dice"]) >= HIT_THRESHOLD) for item in values)
    return {
        "mean_global_dice_per_finding": sum(float(item["dice"]) for item in values) / len(values),
        "hit_rate": hits / len(values),
        "hits": hits,
        "findings": len(values),
        "cases": len({item["case_name"] for item in values}),
    }


def evaluation_records(eval_json: Path, dataset_json: Path) -> list[dict[str, Any]]:
    dataset = read_json(dataset_json).get("test")
    evaluation = read_json(eval_json)
    cases = evaluation.get("cases")
    if not isinstance(dataset, list) or not isinstance(cases, list):
        raise ValueError(f"Invalid evaluation or dataset JSON: {eval_json}")
    if len(dataset) != EXPECTED_CASES or len(cases) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} evaluation cases: {eval_json}")

    dataset_by_name = {str(entry["name"]): entry for entry in dataset}
    cases_by_name = {str(entry["file"]): entry for entry in cases}
    if set(dataset_by_name) != set(cases_by_name):
        raise ValueError(f"Evaluation case set does not match {dataset_json}")

    records: list[dict[str, Any]] = []
    for case_name, entry in dataset_by_name.items():
        metrics = cases_by_name[case_name].get("findings", {})
        expected_keys = {str(key) for key in entry.get("findings", {})}
        observed_keys = {str(key).removeprefix("finding_") for key in metrics}
        if expected_keys != observed_keys:
            raise ValueError(f"{case_name}: evaluator finding keys do not match dataset")
        for metric_key, metric in metrics.items():
            finding_index = str(metric_key).removeprefix("finding_")
            records.append({
                "case_name": case_name,
                "finding_index": finding_index,
                "category": str(entry["categories"][finding_index]),
                "dice": float(metric["global_dice"]),
            })

    if len(records) != EXPECTED_FINDINGS:
        raise ValueError(f"Expected {EXPECTED_FINDINGS} findings, observed {len(records)}: {eval_json}")
    return records


def summarize_evaluation(eval_json: Path, dataset_json: Path) -> dict[str, Any]:
    records = evaluation_records(eval_json, dataset_json)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
    return {
        "evaluation_json": str(eval_json),
        "evaluation_json_sha256": sha256_file(eval_json),
        "overall": metric_summary(records),
        "categories": {
            category: metric_summary(by_category[category])
            for category in ALL_CATEGORIES
        },
    }


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def metric_cell(metric: dict[str, Any] | None) -> str:
    if metric is None:
        return "—"
    return (
        f"{fmt(metric['mean_global_dice_per_finding'])} / "
        f"{int(metric['hits'])}/{int(metric['findings'])} "
        f"({fmt(100 * metric['hit_rate'], 1)}%)"
    )


def training_summary(arm_dir: Path) -> dict[str, Any]:
    path = arm_dir / "reports" / "training_metrics.json"
    if not path.is_file():
        return {"status": "pending", "arm_dir": str(arm_dir), "completed_updates": 0}
    payload = read_json(path)
    completed = int(payload.get("completed_updates") or payload.get("global_update") or 0)
    total = int(payload.get("total_updates") or 10000)
    seconds_per_update = payload.get("mean_update_seconds")
    if seconds_per_update is not None:
        seconds_per_update = float(seconds_per_update)
    remaining = max(0, total - completed)
    return {
        "status": payload.get("status", "running"),
        "metrics_json": str(path),
        "completed_updates": completed,
        "total_updates": total,
        "world_size": payload.get("world_size"),
        "batch_size": payload.get("batch_size"),
        "grad_accum": payload.get("grad_accum"),
        "effective_global_batch_size": (
            int(payload["world_size"]) * int(payload["batch_size"]) * int(payload.get("grad_accum") or 1)
            if payload.get("world_size") is not None and payload.get("batch_size") is not None
            else 4
        ),
        "mean_update_seconds": seconds_per_update,
        "eta_seconds": remaining * seconds_per_update if seconds_per_update else None,
        "eta": duration(remaining * seconds_per_update) if seconds_per_update else "n/a",
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "last_loss": payload.get("last_loss"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(path.stat().st_mtime)),
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    arm_dir = args.group_dir / "ddp_bs4"
    state_path = args.state_json or (args.group_dir / "phase3_state.json")
    state = read_json(state_path) if state_path.is_file() else None

    source = None
    if args.source_evaluation_json.is_file():
        source = summarize_evaluation(args.source_evaluation_json, args.val200_json)
        source["label"] = "Exp007 phase-2 absolute epoch 200"

    milestones: dict[str, Any] = {}
    for epoch in MILESTONES:
        eval_json = arm_dir / f"eval_epoch{epoch:03d}_val200" / "eval" / "val_quick_global_eval.json"
        evaluation = (
            summarize_evaluation(eval_json, args.val200_json)
            if eval_json.is_file()
            else None
        )
        milestones[str(epoch)] = {
            "relative_epoch": epoch,
            "absolute_epoch": args.absolute_epoch_offset + epoch,
            "update": epoch * args.steps_per_epoch,
            "status": "complete" if evaluation is not None else "pending",
            "evaluation": evaluation,
        }

    completed_epochs = [epoch for epoch in MILESTONES if milestones[str(epoch)]["evaluation"] is not None]
    training = training_summary(arm_dir)
    if (arm_dir / ".experiment_complete").is_file() or (args.group_dir / ".experiment_complete").is_file():
        status = "complete"
    elif (arm_dir / ".train_failed").is_file() or (args.group_dir / ".train_failed").is_file():
        status = "failed"
    elif state is not None:
        status = str(state.get("status", "running"))
    elif completed_epochs or training.get("completed_updates", 0):
        status = "running"
    else:
        status = "pending"

    if status == "complete" and len(completed_epochs) != len(MILESTONES):
        status = "running"

    return {
        "experiment": EXPERIMENT_ID,
        "phase": "phase3_from_abs_e200",
        "status": status,
        "updated_at_utc": utc_now_iso(),
        "group_dir": str(args.group_dir),
        "run_dir": str(arm_dir),
        "relative_epoch_offset": args.absolute_epoch_offset,
        "source_run_dir": str(args.source_run_dir),
        "source_checkpoint": file_record(args.source_checkpoint),
        "source_evaluation": source,
        "schedule_manifest": file_record(args.schedule_manifest),
        "state": state,
        "training": training,
        "latest_completed_milestone": max(completed_epochs, default=0),
        "milestones": milestones,
        "evaluation_contract": {
            "dataset_json": str(args.val200_json),
            "dataset_json_sha256": sha256_file(args.val200_json),
            "cases": EXPECTED_CASES,
            "findings": EXPECTED_FINDINGS,
            "evaluation_threshold": 0.5,
            "hit_threshold_dice": HIT_THRESHOLD,
            "categories": list(ALL_CATEGORIES),
            "milestone_relative_epochs": list(MILESTONES),
        },
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Exp007 Phase 3: Continuation from Absolute Epoch 200",
        "",
        f"Status: `{summary['status']}`; updated `{summary['updated_at_utc']}`.",
        "",
        "This run loads the Exp007 phase-2 absolute-epoch-200 checkpoint as network weights only. The optimizer, AMP scaler, update counter, warmup, and 100-epoch polynomial LR schedule are reset.",
        "",
        "## Configuration",
        "",
        f"- Run group: `{summary['group_dir']}`",
        f"- Source run: `{summary['source_run_dir']}`",
        f"- Source checkpoint: `{summary['source_checkpoint']['path']}`",
        f"- Source checkpoint SHA256: `{summary['source_checkpoint'].get('sha256', 'missing')}`",
        f"- Schedule manifest: `{summary['schedule_manifest']['path']}`",
        "- Training: DDP world size 4, per-GPU batch size 1, effective batch size 4, 100 epochs, 10,000 updates.",
        "- Evaluation: all 200 fixed validation cases / 381 findings every 10 relative epochs.",
        "",
        "## Current progress",
        "",
        f"- Current state: `{(summary.get('state') or {}).get('action', 'n/a')}`",
        f"- Latest completed milestone: relative epoch `{summary['latest_completed_milestone']}` / absolute epoch `{200 + summary['latest_completed_milestone']}`",
        f"- Completed updates: `{summary['training'].get('completed_updates', 0)}` / `{summary['training'].get('total_updates', 10000)}`",
        f"- ETA from latest training rate: `{summary['training'].get('eta', 'n/a')}`",
        "",
        "## Initial reference",
        "",
        "| Model | Overall Dice / hits | Nodule 2d Dice / hits |",
        "| --- | ---: | ---: |",
    ]
    source = summary.get("source_evaluation")
    lines.append(
        f"| Exp007 phase-2 absolute epoch 200 | {metric_cell(source['overall']) if source else '—'} | "
        f"{metric_cell(source['categories'].get('2d')) if source else '—'} |"
    )

    lines.extend([
        "",
        "## Val200 progress",
        "",
        "Cells show `Dice / hits/findings (hit rate)`; nodule metrics are official category `2d`.",
        "",
        "| Relative epoch | Absolute epoch | Update | Overall | Nodule 2d | Status |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for epoch in MILESTONES:
        item = summary["milestones"][str(epoch)]
        evaluation = item["evaluation"]
        nodule = evaluation["categories"].get("2d") if evaluation else None
        lines.append(
            f"| {epoch} | {item['absolute_epoch']} | {item['update']} | "
            f"{metric_cell(evaluation['overall']) if evaluation else '—'} | "
            f"{metric_cell(nodule)} | {item['status']} |"
        )

    for epoch in MILESTONES:
        item = summary["milestones"][str(epoch)]
        evaluation = item["evaluation"]
        lines.extend([
            "",
            f"## Category performance — relative epoch {epoch} / absolute epoch {item['absolute_epoch']}",
            "",
            "| Category | Findings | Dice | Hits | Hit rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for category in ALL_CATEGORIES:
            metric = evaluation["categories"].get(category) if evaluation else None
            if metric is None:
                lines.append(f"| {category} | 0 | — | — | — |")
            else:
                lines.append(
                    f"| {category} | {metric['findings']} | {fmt(metric['mean_global_dice_per_finding'])} | "
                    f"{metric['hits']}/{metric['findings']} | {fmt(100 * metric['hit_rate'], 1)}% |"
                )

    training = summary["training"]
    lines.extend([
        "",
        "## Training status",
        "",
        f"- Status: `{training.get('status', 'n/a')}`",
        f"- World size / per-GPU batch / accumulation: `{training.get('world_size', 4)} / {training.get('batch_size', 1)} / {training.get('grad_accum', 1)}`",
        f"- Effective global batch size: `{training.get('effective_global_batch_size', 4)}`",
        f"- Mean update time: `{fmt(training.get('mean_update_seconds'), 3)}` seconds",
        f"- Last loss: `{fmt(training.get('last_loss'))}`",
        "",
        "## Artifact paths",
        "",
        f"- Machine-readable report: `{summary['group_dir'].rsplit('/runs/', 1)[0]}/reports/phase3_status.json`",
        f"- Per-run logs and checkpoints: `{summary['run_dir']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-evaluation-json", type=Path, required=True)
    parser.add_argument("--val200-json", type=Path, default=Path("/workspace/configs/evaluation/rexgroundingct_val200_seed20260723.json"))
    parser.add_argument("--absolute-epoch-offset", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--state-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args)
    atomic_write_json(args.output_json, summary)
    atomic_write(args.output_md, render(summary))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
