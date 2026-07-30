#!/usr/bin/env python3
"""Summarize the combined exp011 e4d4/e5d4 CT-normalization results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


VAL20_EPOCHS = (5, 20, 40, 60, 80, 100)
CLIP_GRAD_NORM = 12.0
SCHEDULE_SHA256 = "f246927486c69e00a872990dbc6e7e8566f49f3b41dbb1bb05c5ce5a033f4776"
VAL20_SHA256 = "31557d624c47ee8c06799bf89cceb862471b34cfd47065001d6092e32d0dab5d"
VAL200_SHA256 = "7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
DEFAULT_EXP006_E5_D4_DIR = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "006_voxtell_cached_native_v123_lr_ablation/runs/latest/"
    "v123_cached_e5_d4"
)

ARM_SPECS = (
    {
        "key": "e5_zscore",
        "arm": "v123_e5d4_zscore",
        "profile": "e5d4",
        "normalization": "Native z-score",
        "label": "Exp011 native z-score e5d4",
    },
    {
        "key": "e5_clip_zscore",
        "arm": "v123_e5d4_clip1024_zscore",
        "profile": "e5d4",
        "normalization": "Clip1024 + z-score",
        "label": "Exp011 clip1024 + z-score e5d4",
    },
    {
        "key": "e5_linear",
        "arm": "v123_e5d4_clip1024_linear",
        "profile": "e5d4",
        "normalization": "Clip1024 linear HU",
        "label": "Exp011 clip1024 linear HU e5d4",
    },
    {
        "key": "e4_zscore",
        "arm": "v123_e4d4_zscore",
        "profile": "e4d4",
        "normalization": "Native z-score",
        "label": "Exp011 native z-score e4d4",
    },
    {
        "key": "e4_clip_zscore",
        "arm": "v123_e4d4_clip1024_zscore",
        "profile": "e4d4",
        "normalization": "Clip1024 + z-score",
        "label": "Exp011 clip1024 + z-score e4d4",
    },
    {
        "key": "e4_linear",
        "arm": "v123_e4d4_clip1024_linear",
        "profile": "e4d4",
        "normalization": "Clip1024 linear HU",
        "label": "Exp011 clip1024 linear HU e4d4",
    },
)

PAIR_KEYS = (
    ("Native z-score", "e5_zscore", "e4_zscore"),
    ("Clip1024 + z-score", "e5_clip_zscore", "e4_clip_zscore"),
    ("Clip1024 linear HU", "e5_linear", "e4_linear"),
)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def metric_row(label: str, summary_path: Path) -> dict[str, Any]:
    summary = read_json_if_exists(summary_path)
    if summary is None:
        return {
            "label": label,
            "summary_path": str(summary_path),
            "status": "pending",
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
    reports_dir = arm_dir / "reports"
    metric_paths = sorted(reports_dir.glob("training_metrics_segment_epoch*.json"))
    metric_paths.extend(
        sorted(reports_dir.glob("training_metrics_recovery_epoch*.json"))
    )
    if metric_paths:
        segment_metrics = [
            value
            for path in metric_paths
            if (value := read_json_if_exists(path)) is not None
        ]
    else:
        metrics = read_json_if_exists(reports_dir / "training_metrics.json")
        segment_metrics = [metrics] if metrics is not None else []
    if not segment_metrics:
        return {"status": "pending", "completed_updates": 0}

    updates_by_index: dict[int, dict[str, Any]] = {}
    for metrics in segment_metrics:
        for item in metrics.get("updates", []):
            if item.get("global_update") is not None:
                updates_by_index[int(item["global_update"])] = item
    updates = [updates_by_index[index] for index in sorted(updates_by_index)]
    losses = [float(item["loss"]) for item in updates if item.get("loss") is not None]
    observed_grad_norms = [
        float(item["grad_norm"])
        for item in updates
        if item.get("grad_norm") is not None
    ]
    grad_norms = [value for value in observed_grad_norms if np.isfinite(value)]
    nonfinite_grad_norms = len(observed_grad_norms) - len(grad_norms)
    elapsed_seconds = sum(
        float(metrics.get("elapsed_seconds") or 0.0)
        for metrics in segment_metrics
    )
    completed_updates = max(
        int(metrics.get("completed_updates", metrics.get("global_update", 0)) or 0)
        for metrics in segment_metrics
    )
    memory: dict[str, float] = {}
    for metrics in segment_metrics:
        for key, value in (metrics.get("memory") or {}).items():
            if isinstance(value, (int, float)):
                memory[key] = max(float(value), memory.get(key, float("-inf")))
    return {
        "status": "completed" if completed_updates >= 10000 else "running",
        "completed_updates": completed_updates,
        "elapsed_seconds": elapsed_seconds,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "last100_mean_loss": float(np.mean(losses[-100:])) if losses else None,
        "mean_update_seconds": (
            elapsed_seconds / len(updates) if updates and elapsed_seconds else None
        ),
        "mean_preclip_grad_norm": float(np.mean(grad_norms)) if grad_norms else None,
        "max_preclip_grad_norm": float(np.max(grad_norms)) if grad_norms else None,
        "nonfinite_grad_norm_updates": nonfinite_grad_norms,
        "fraction_updates_above_clip_norm": (
            float(np.mean(np.asarray(grad_norms) > CLIP_GRAD_NORM))
            if grad_norms
            else None
        ),
        "recorded_updates": len(updates),
        "memory": memory,
    }


def collect_model(label: str, arm_dir: Path) -> dict[str, Any]:
    return {
        "label": label,
        "arm_dir": str(arm_dir),
        "training": training_row(arm_dir),
        "val20": {
            str(epoch): metric_row(
                f"{label} epoch {epoch} val20",
                arm_dir
                / f"eval_epoch{epoch:03d}_val20"
                / "reports"
                / "val_quick_global_eval_summary.json",
            )
            for epoch in VAL20_EPOCHS
        },
        "val200": metric_row(
            f"{label} epoch 100 val200",
            arm_dir
            / "eval_epoch100_val200"
            / "reports"
            / "val_quick_global_eval_summary.json",
        ),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "pending"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_cell(metric: dict[str, Any]) -> str:
    if metric.get("status") != "complete":
        return "pending"
    return f"{fmt(metric.get('dice'))} / {fmt(metric.get('hit_rate'))}"


def delta_cell(candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
    if candidate.get("status") != "complete" or baseline.get("status") != "complete":
        return "pending"
    return (
        f"{float(candidate['dice']) - float(baseline['dice']):+.4f} / "
        f"{float(candidate['hit_rate']) - float(baseline['hit_rate']):+.4f}"
    )


def resolve_group(
    explicit: Path | None,
    fallback: Path,
) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if fallback.exists() or fallback.is_symlink():
        return fallback.resolve()
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--e4-group-dir", type=Path)
    parser.add_argument("--e5-group-dir", type=Path)
    parser.add_argument(
        "--group-dir",
        type=Path,
        help="Legacy alias for --e4-group-dir",
    )
    parser.add_argument(
        "--exp006-run-dir",
        type=Path,
        default=DEFAULT_EXP006_E5_D4_DIR,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    e4_explicit = args.e4_group_dir or args.group_dir
    e4_group = resolve_group(e4_explicit, args.exp_dir / "runs" / "latest_e4d4")
    e5_group = resolve_group(args.e5_group_dir, args.exp_dir / "runs" / "latest_e5d4")

    models: dict[str, dict[str, Any]] = {}
    for spec in ARM_SPECS:
        group = e5_group if spec["profile"] == "e5d4" else e4_group
        models[spec["key"]] = collect_model(
            str(spec["label"]),
            group / str(spec["arm"]),
        )
        models[spec["key"]]["profile"] = spec["profile"]
        models[spec["key"]]["normalization"] = spec["normalization"]

    models["exp006_e5_zscore"] = collect_model(
        "Exp006 native z-score e5d4",
        args.exp006_run_dir.resolve(),
    )
    models["exp006_e5_zscore"]["profile"] = "exp006_e5d4"
    models["exp006_e5_zscore"]["normalization"] = "Native z-score"

    payload = {
        "experiment": "011_voxtell_v123_e4d4_ct_normalization_ablation",
        "run_groups": {
            "e4d4": str(e4_group),
            "e5d4": str(e5_group),
            "exp006_e5d4_reference": str(args.exp006_run_dir.resolve()),
        },
        "provenance": {
            "schedule_sha256": SCHEDULE_SHA256,
            "val20_json_sha256": VAL20_SHA256,
            "val200_json_sha256": VAL200_SHA256,
            "val20_cases": 20,
            "val20_findings": 31,
            "val200_cases": 200,
            "val200_findings": 381,
        },
        "models": models,
    }
    atomic_write_json(args.output_json, payload)

    lines = [
        "# Experiment 011 CT Normalization And Encoder-LR Ablation",
        "",
        f"- e4d4 run: `{e4_group.name}`",
        f"- e5d4 run: `{e5_group.name}`",
        "- Initialization: public VoxTell v1.1 for every exp011 arm",
        f"- Shared schedule SHA256: `{SCHEDULE_SHA256}`",
        "",
        "## Val200 Epoch 100",
        "",
        "| Model | Dice | Hit rate | Hits / findings | Last100 loss | "
        "Training hours | Sec/update | Mean grad norm | Updates clipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    val200_order = (
        "e5_zscore",
        "e5_clip_zscore",
        "e5_linear",
        "exp006_e5_zscore",
        "e4_zscore",
        "e4_clip_zscore",
        "e4_linear",
    )
    for key in val200_order:
        model = models[key]
        metric = model["val200"]
        training = model["training"]
        clipped = training.get("fraction_updates_above_clip_norm")
        clipped_text = "pending" if clipped is None else f"{100.0 * clipped:.2f}%"
        elapsed = training.get("elapsed_seconds")
        training_hours = None if elapsed is None else float(elapsed) / 3600.0
        hits = (
            "pending"
            if metric.get("status") != "complete"
            else f"{fmt(metric.get('hits'), 0)} / {fmt(metric.get('findings'), 0)}"
        )
        lines.append(
            f"| {model['label']} | {fmt(metric.get('dice'))} | "
            f"{fmt(metric.get('hit_rate'))} | {hits} | "
            f"{fmt(training.get('last100_mean_loss'))} | "
            f"{fmt(training_hours, 2)} | "
            f"{fmt(training.get('mean_update_seconds'), 3)} | "
            f"{fmt(training.get('mean_preclip_grad_norm'), 3)} | "
            f"{clipped_text} |"
        )

    lines.extend(
        [
            "",
            "## Val20 Progress",
            "",
            "Each checkpoint cell is `Dice / hit rate` on the fixed seeded val20 "
            "probe.",
            "",
            "| Model | "
            + " | ".join(f"Epoch {epoch}" for epoch in VAL20_EPOCHS)
            + " |",
            "| --- | " + " | ".join("---:" for _ in VAL20_EPOCHS) + " |",
        ]
    )
    for key in val200_order:
        model = models[key]
        cells = [
            metric_cell(model["val20"][str(epoch)])
            for epoch in VAL20_EPOCHS
        ]
        lines.append(f"| {model['label']} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## E5d4 Minus E4d4",
            "",
            "Each cell is `Dice delta / hit-rate delta`; positive values favor "
            "e5d4.",
            "",
            "| Normalization | "
            + " | ".join(f"Epoch {epoch}" for epoch in VAL20_EPOCHS)
            + " | Val200 |",
            "| --- | "
            + " | ".join("---:" for _ in range(len(VAL20_EPOCHS) + 1))
            + " |",
        ]
    )
    for normalization, e5_key, e4_key in PAIR_KEYS:
        cells = [
            delta_cell(
                models[e5_key]["val20"][str(epoch)],
                models[e4_key]["val20"][str(epoch)],
            )
            for epoch in VAL20_EPOCHS
        ]
        cells.append(delta_cell(models[e5_key]["val200"], models[e4_key]["val200"]))
        lines.append(f"| {normalization} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Native E5d4 Replication",
            "",
            "Each cell is `Exp011 minus Exp006` for `Dice / hit rate`.",
            "",
            "| Comparison | "
            + " | ".join(f"Epoch {epoch}" for epoch in VAL20_EPOCHS)
            + " | Val200 |",
            "| --- | "
            + " | ".join("---:" for _ in range(len(VAL20_EPOCHS) + 1))
            + " |",
        ]
    )
    replication_cells = [
        delta_cell(
            models["e5_zscore"]["val20"][str(epoch)],
            models["exp006_e5_zscore"]["val20"][str(epoch)],
        )
        for epoch in VAL20_EPOCHS
    ]
    replication_cells.append(
        delta_cell(
            models["e5_zscore"]["val200"],
            models["exp006_e5_zscore"]["val200"],
        )
    )
    lines.append(
        "| Exp011 native e5d4 - Exp006 native e5d4 | "
        + " | ".join(replication_cells)
        + " |"
    )

    lines.extend(
        [
            "",
            "## Fixed Evaluation Provenance",
            "",
            f"- val20 JSON SHA256: `{VAL20_SHA256}` (`20` cases, `31` findings)",
            f"- val200 JSON SHA256: `{VAL200_SHA256}` (`200` cases, `381` findings)",
            "- Primary threshold: `0.5`",
            "- GPU policy: exp011 uses host GPUs `0,1,2` only",
            "",
        ]
    )
    atomic_write_text(args.output_md, "\n".join(lines))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
