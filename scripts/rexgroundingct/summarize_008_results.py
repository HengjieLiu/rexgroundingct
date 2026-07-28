#!/usr/bin/env python3
"""Build a partial or final Exp008 dual-branch comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import utc_now_iso, write_json


VARIANTS = (
    "v1_sharedfusion_softguide",
    "v1_dualfusion_softguide",
    "v2_dualfusion_precision",
    "v3_dualfusion_softguide_joint",
)
EPOCHS = (0, 20, 40, 60, 80, 100)
VAL200_EPOCHS = (80, 100)


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def metric_row(summary: dict | None) -> dict | None:
    if summary is None:
        return None
    return {
        "dice": summary.get("mean_global_dice_per_finding"),
        "hit_rate": summary.get("hit_rate"),
        "hits": summary.get("total_hits"),
        "findings": summary.get("total_findings"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "created_at_utc": utc_now_iso(),
        "group_dir": str(args.group_dir),
        "source": {
            "model": "exp006 v123_cached_e5_d4 epoch100",
            "val20": {
                "dice": 0.3907023703162601,
                "hit_rate": 0.7419354838709677,
                "hits": 23,
                "findings": 31,
            },
            "val200": {
                "dice": 0.32413407768565705,
                "hit_rate": 0.7611548556430446,
                "hits": 290,
                "findings": 381,
            },
        },
        "variants": {},
        "interpretation_limit": (
            "No same-source single-branch continuation control is present; "
            "improvement over epoch 0 cannot be attributed entirely to the "
            "dual-branch architecture."
        ),
    }

    for variant in VARIANTS:
        arm_dir = args.group_dir / variant
        arm = {
            "val20": {},
            "val200": {},
            "val200_epoch100": None,
            "proposal": {},
            "training_segments": [],
        }
        for epoch in EPOCHS:
            eval_dir = arm_dir / f"eval_epoch{epoch:03d}_val20"
            summary = read_json(
                eval_dir / "reports" / "val_quick_global_eval_summary.json"
            )
            arm["val20"][str(epoch)] = metric_row(summary)
            proposal = read_json(
                eval_dir / "reports" / "proposal_diagnostics.json"
            )
            if proposal is not None:
                arm["proposal"][str(epoch)] = {
                    label: {
                        key: value
                        for key, value in threshold.items()
                        if key
                        in {
                            "threshold",
                            "mean_global_dice_per_finding",
                            "hit_rate",
                            "hits",
                            "mean_gt_voxel_coverage",
                            "finding_overlap_rate",
                            "mean_proposal_to_gt_volume_ratio",
                        }
                    }
                    for label, threshold in proposal["thresholds"].items()
                }
        for epoch in VAL200_EPOCHS:
            arm["val200"][str(epoch)] = metric_row(
                read_json(
                    arm_dir
                    / f"eval_epoch{epoch:03d}_val200"
                    / "reports"
                    / "val_quick_global_eval_summary.json"
                )
            )
        arm["val200_epoch100"] = arm["val200"]["100"]
        reports_dir = arm_dir / "reports"
        for path in sorted(reports_dir.glob("training_metrics_segment_epoch*.json")):
            metrics = read_json(path)
            if metrics is not None:
                arm["training_segments"].append(
                    {
                        "path": str(path),
                        "start_global_update": metrics.get("start_global_update"),
                        "completed_updates": metrics.get("completed_updates"),
                        "elapsed_seconds": metrics.get("elapsed_seconds"),
                        "last_loss": metrics.get("last_loss"),
                        "mean_update_seconds": metrics.get("mean_update_seconds"),
                        "memory": metrics.get("memory"),
                    }
                )
        result["variants"][variant] = arm

    write_json(args.output_json, result)
    lines = [
        "# Experiment 008 Dual-Branch Ablation",
        "",
        "## Final Segmentation Val20",
        "",
        "| Variant | e0 | e20 | e40 | e60 | e80 | e100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, arm in result["variants"].items():
        values = []
        for epoch in EPOCHS:
            metric = arm["val20"].get(str(epoch))
            values.append(
                f"{metric['dice']:.4f}/{metric['hit_rate']:.4f}"
                if metric is not None
                else "-"
            )
        lines.append(f"| `{variant}` | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Cells are `Dice / hit rate`.",
            "",
            "## Val200",
            "",
            (
                "Starting-point reference: Exp006 `v123_cached_e5_d4` epoch 100, "
                "Dice `0.3241`, hit rate `0.7612` (`290/381`)."
            ),
            "",
            "| Variant | e80 | e100 |",
            "| --- | ---: | ---: |",
        ]
    )
    for variant, arm in result["variants"].items():
        values = []
        for epoch in VAL200_EPOCHS:
            metric = arm["val200"].get(str(epoch))
            values.append(
                (
                    f"{metric['dice']:.4f}/{metric['hit_rate']:.4f} "
                    f"({metric['hits']}/{metric['findings']})"
                )
                if metric is not None
                else "-"
            )
        lines.append(f"| `{variant}` | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Cells are `Dice / hit rate (hits/findings)`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation Limit",
            "",
            result["interpretation_limit"],
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
