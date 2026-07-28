#!/usr/bin/env python3
"""Summarize Exp009 S3 attention coupling ablation milestones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import utc_now_iso, write_json


VARIANTS = (
    "baseline_cont100",
    "s3v1_fixedrho_suppress_half_quarter",
    "s3v2_balanced_feature_half_quarter",
    "s3v3_logit_residual_half_quarter",
)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def metric_record(eval_dir: Path) -> dict[str, Any] | None:
    summary = read_json_if_exists(eval_dir / "reports" / "val_quick_global_eval_summary.json")
    if summary is None:
        return None
    return {
        "mean_global_dice_per_finding": summary.get("mean_global_dice_per_finding"),
        "hit_rate": summary.get("hit_rate"),
        "total_hits": summary.get("total_hits"),
        "total_findings": summary.get("total_findings"),
        "eval_dir": str(eval_dir),
    }


def training_record(arm_dir: Path, epoch: int) -> dict[str, Any] | None:
    path = arm_dir / "reports" / f"training_metrics_segment_epoch{epoch:03d}.json"
    if not path.is_file():
        path = arm_dir / "reports" / "training_metrics.json"
    metrics = read_json_if_exists(path)
    if metrics is None:
        return None
    updates = list(metrics.get("updates") or [])
    last_update = updates[-1] if updates else {}
    return {
        "path": str(path),
        "completed_updates": metrics.get("completed_updates"),
        "target_global_update": metrics.get("target_global_update"),
        "mean_loss": metrics.get("mean_loss"),
        "last_loss": metrics.get("last_loss"),
        "mean_update_seconds": metrics.get("mean_update_seconds"),
        "memory": metrics.get("memory"),
        "last_update": {
            key: value
            for key, value in last_update.items()
            if key.startswith("loss_")
            or key.startswith("s3_")
            or key in {"global_update", "epoch", "loss", "grad_norm", "update_seconds"}
        },
    }


def build_summary(group_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "group_dir": str(group_dir),
        "variants": {},
    }
    for variant in VARIANTS:
        arm_dir = group_dir / variant
        failure = read_json_if_exists(arm_dir / "failure_status.json")
        variant_record: dict[str, Any] = {
            "arm_dir": str(arm_dir),
            "status": (
                "success" if (arm_dir / ".arm_success").exists()
                else "failed" if (arm_dir / ".arm_failed").exists()
                else "running" if arm_dir.exists()
                else "missing"
            ),
            "failure": failure,
            "val20": {},
            "val200": {},
            "training": {},
        }
        for epoch in (0, 5, 20, 40, 60, 80, 100):
            record = metric_record(arm_dir / f"eval_epoch{epoch:03d}_val20")
            if record is not None:
                variant_record["val20"][str(epoch)] = record
            training = training_record(arm_dir, epoch)
            if training is not None:
                variant_record["training"][str(epoch)] = training
        val200 = metric_record(arm_dir / "eval_epoch100_val200")
        if val200 is not None:
            variant_record["val200"]["100"] = val200
        summary["variants"][variant] = variant_record
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Experiment 009 S3 Attention Coupling Ablation",
        "",
        f"Run group: `{Path(summary['group_dir']).name}`",
        "",
        "## Val20 Milestones",
        "",
        "| Variant | Epoch | Dice | Hit rate | Hits / findings |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, record in summary["variants"].items():
        val20 = record.get("val20", {})
        for epoch in ("0", "5", "20", "40", "60", "80", "100"):
            metric = val20.get(epoch)
            if metric is None:
                continue
            lines.append(
                "| {variant} | {epoch} | {dice:.4f} | {hit:.4f} | {hits} / {findings} |".format(
                    variant=variant,
                    epoch=epoch,
                    dice=float(metric["mean_global_dice_per_finding"]),
                    hit=float(metric["hit_rate"]),
                    hits=int(metric["total_hits"]),
                    findings=int(metric["total_findings"]),
                )
            )
    lines.extend(
        [
            "",
            "## Val200 Epoch 100",
            "",
            "| Variant | Dice | Hit rate | Hits / findings |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for variant, record in summary["variants"].items():
        metric = record.get("val200", {}).get("100")
        if metric is None:
            continue
        lines.append(
            "| {variant} | {dice:.4f} | {hit:.4f} | {hits} / {findings} |".format(
                variant=variant,
                dice=float(metric["mean_global_dice_per_finding"]),
                hit=float(metric["hit_rate"]),
                hits=int(metric["total_hits"]),
                findings=int(metric["total_findings"]),
            )
        )
    lines.extend(["", "## Status", ""])
    for variant, record in summary["variants"].items():
        failure = record.get("failure")
        if failure:
            lines.append(
                "- `{variant}`: `{status}` ({phase}, {reason}; log: `{log}`)".format(
                    variant=variant,
                    status=record["status"],
                    phase=failure.get("phase"),
                    reason=failure.get("reason"),
                    log=failure.get("log_path"),
                )
            )
        else:
            lines.append(f"- `{variant}`: `{record['status']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args.group_dir)
    write_json(args.output_json, summary)
    write_markdown(summary, args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
