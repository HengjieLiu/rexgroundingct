#!/usr/bin/env python3
"""Summarize Exp026 fixed-LR training and five-epoch val200 barriers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import write_json


EPOCHS = list(range(5, 101, 5))


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def eval_summary(path: Path, epoch: int) -> dict[str, Any]:
    data = read_json(path)
    if data is None:
        return {"epoch": epoch, "status": "missing", "path": str(path)}
    return {
        "epoch": epoch,
        "status": "complete",
        "path": str(path),
        "dice": data.get("mean_global_dice_per_finding"),
        "hit_rate": data.get("hit_rate"),
        "hits": data.get("total_hits"),
        "findings": data.get("total_findings"),
        "cases": data.get("total_cases"),
    }


def collect_training(run_dir: Path) -> dict[str, Any]:
    reports = run_dir / "reports"
    paths = sorted(reports.glob("training_metrics_segment_epoch*.json"))
    final = reports / "training_metrics.json"
    if final.is_file() and final not in paths:
        paths.append(final)

    updates: dict[int, dict[str, Any]] = {}
    segments: list[dict[str, Any]] = []
    for path in paths:
        metrics = read_json(path) or {}
        segments.append(
            {
                "path": str(path),
                "status": metrics.get("status"),
                "completed_updates": metrics.get("completed_updates"),
                "target_global_update": metrics.get("target_global_update"),
                "mean_loss": metrics.get("mean_loss"),
                "last_loss": metrics.get("last_loss"),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
            }
        )
        for item in metrics.get("updates", []):
            if item.get("global_update") is not None:
                updates[int(item["global_update"])] = item

    first = updates[min(updates)] if updates else {}
    last = updates[max(updates)] if updates else {}
    return {
        "status": "complete" if (run_dir / ".train_complete").exists() else "in_progress",
        "completed_updates": max(updates) if updates else 0,
        "total_updates": 10000,
        "updates_observed": len(updates),
        "first_update_lrs": first.get("lrs"),
        "last_update_lrs": last.get("lrs"),
        "segments": segments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.group_dir / "ddp_bs4"
    val200 = {}
    checkpoints = {}
    for epoch in EPOCHS:
        checkpoint = run_dir / "checkpoints" / f"checkpoint_update_{epoch * 100:06d}.pth"
        summary = run_dir / f"eval_epoch{epoch:03d}_val200" / "reports" / "val_quick_global_eval_summary.json"
        val200[str(epoch)] = eval_summary(summary, epoch)
        checkpoints[str(epoch)] = {
            "update": epoch * 100,
            "path": str(checkpoint),
            "exists": checkpoint.is_file(),
            "bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
        }

    provenance = read_json(args.exp_dir / "config" / "public_voxtell_v1_1_provenance.json")
    payload = {
        "experiment": "026_voxtell_cached_native_v123_e5_d4_ddp_bs4_fixed_lr",
        "group_dir": str(args.group_dir),
        "training": collect_training(run_dir),
        "learning_rate_policy": {
            "encoder_lr": 1e-5,
            "decoder_lr": 1e-4,
            "schedule": "fixed",
            "warmup_updates": 0,
        },
        "checkpoints": checkpoints,
        "val200": val200,
        "public_model_provenance": provenance,
    }
    write_json(args.output_json, payload)

    lines = [
        "# Exp026 Fixed-LR DDP Batch4 Report",
        "",
        f"Run group: `{args.group_dir.name}`",
        "",
        "## Training",
        "",
        "- Initialization: public VoxTell v1.1",
        "- DDP: world size 4, local batch 1, effective global batch 4",
        "- Learning rates: encoder `1e-5`, decoder `1e-4`, fixed, warmup `0`",
        "- Target: 10,000 optimizer updates",
        "",
        "## Val200 Progress",
        "",
        "| Epoch | Update | Status | Dice | Hit rate | Hits / findings |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for epoch in EPOCHS:
        row = val200[str(epoch)]
        dice = row.get("dice")
        hit_rate = row.get("hit_rate")
        hits = row.get("hits")
        findings = row.get("findings")
        lines.append(
            f"| {epoch} | {epoch * 100} | {row['status']} | "
            f"{dice:.4f} | {hit_rate:.4f} | {hits} / {findings} |"
            if isinstance(dice, (int, float)) and isinstance(hit_rate, (int, float))
            else f"| {epoch} | {epoch * 100} | {row['status']} | n/a | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "## Checkpoint Completeness",
            "",
            f"Immutable milestone checkpoints present: {sum(1 for row in checkpoints.values() if row['exists'])} / {len(checkpoints)}.",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
