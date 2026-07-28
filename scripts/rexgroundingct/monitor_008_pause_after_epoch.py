#!/usr/bin/env python3
"""Monitor Exp008, pause after epoch 80, and write a resumable result report."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


VARIANTS = (
    "v1_sharedfusion_softguide",
    "v1_dualfusion_softguide",
    "v2_dualfusion_precision",
    "v3_dualfusion_softguide_joint",
)
REPORT_EPOCHS = (0, 20, 40, 60, 80)
PROGRESS_PATTERN = re.compile(
    r"(\d+)/2000 \[([^<\]]+)<([^,\]]+),\s*([0-9.]+)s/it"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_duration(value: str) -> float:
    fields = [float(field) for field in value.strip().split(":")]
    seconds = 0.0
    for field in fields:
        seconds = seconds * 60.0 + field
    return seconds


def latest_training_progress(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="ignore")
    matches = list(PROGRESS_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    completed = int(match.group(1))
    elapsed = parse_duration(match.group(2))
    return {
        "completed_segment_updates": completed,
        "segment_updates": 2000,
        "elapsed_seconds": elapsed,
        "remaining_seconds": parse_duration(match.group(3)),
        "seconds_per_update": float(match.group(4)),
        "cumulative_seconds_per_update": (
            elapsed / completed if completed > 0 else None
        ),
        "cumulative_projected_remaining_seconds": (
            (2000 - completed) * elapsed / completed
            if completed > 0
            else None
        ),
    }


def valid_eval(eval_dir: Path) -> bool:
    summary = read_json(eval_dir / "reports" / "val_quick_global_eval_summary.json")
    diagnostics = read_json(eval_dir / "reports" / "proposal_diagnostics.json")
    if (
        summary is None
        or diagnostics is None
        or int(summary.get("total_cases", -1)) != 20
        or int(summary.get("total_findings", -1)) != 31
    ):
        return False
    predictions = list((eval_dir / "predictions").glob("*.nii.gz"))
    if len(predictions) != 20:
        return False
    for label in ("thr010", "thr030", "thr050"):
        proposal_summary = read_json(
            eval_dir
            / "proposal"
            / label
            / "reports"
            / "val_quick_global_eval_summary.json"
        )
        if (
            proposal_summary is None
            or int(proposal_summary.get("total_cases", -1)) != 20
            or int(proposal_summary.get("total_findings", -1)) != 31
        ):
            return False
    return (eval_dir / ".complete").is_file()


def pause_paths(group_dir: Path, next_target_update: int) -> tuple[Path, dict[str, Path]]:
    group_marker = group_dir / "control" / "pause_after_epoch080"
    arm_markers = {
        variant: (
            group_dir
            / variant
            / "control"
            / f"pause_before_update_{next_target_update:06d}"
        )
        for variant in VARIANTS
    }
    return group_marker, arm_markers


def install_pause(group_dir: Path, next_target_update: int) -> dict[str, Any]:
    group_marker, arm_markers = pause_paths(group_dir, next_target_update)
    group_marker.parent.mkdir(parents=True, exist_ok=True)
    group_marker.write_text(
        "Pause Exp008 after epoch-80 val20. Remove through the monitor --resume command.\n"
    )
    for marker in arm_markers.values():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            f"Pause before target global update {next_target_update}; no new events consumed.\n"
        )
    policy = {
        "created_at_utc": utc_now_iso(),
        "group_dir": str(group_dir),
        "pause_after_epoch": 80,
        "completed_update_boundary": 8000,
        "next_target_update": next_target_update,
        "group_marker": str(group_marker),
        "arm_markers": {key: str(value) for key, value in arm_markers.items()},
        "resume_command": (
            "python /workspace/scripts/rexgroundingct/"
            "monitor_008_pause_after_epoch.py "
            f"--group-dir {group_dir} --resume"
        ),
    }
    write_json(group_dir / "control" / "pause_policy.json", policy)
    return policy


def release_pause(group_dir: Path, next_target_update: int) -> list[str]:
    group_marker, arm_markers = pause_paths(group_dir, next_target_update)
    removed = []
    for path in (group_marker, *arm_markers.values()):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def arm_is_paused(arm_dir: Path, next_target_update: int) -> bool:
    launcher_status = arm_dir / ".paused_after_epoch080"
    if launcher_status.is_file():
        return True
    status = read_json(arm_dir / "control" / "pause_status.json")
    return bool(
        status
        and status.get("state") == "paused"
        and int(status.get("target_global_update", -1)) == next_target_update
        and int(status.get("updates_consumed_in_segment", -1)) == 0
    )


def inspect_arm(
    group_dir: Path,
    variant: str,
    evaluation_seconds: float,
    next_target_update: int,
) -> dict[str, Any]:
    arm_dir = group_dir / variant
    checkpoint = arm_dir / "checkpoints" / "checkpoint_update_008000.pth"
    eval_dir = arm_dir / "eval_epoch080_val20"
    prediction_dir = eval_dir / "predictions"
    prediction_count = len(list(prediction_dir.glob("*.nii.gz"))) if prediction_dir.is_dir() else 0
    eval_complete = valid_eval(eval_dir)
    paused = arm_is_paused(arm_dir, next_target_update)
    progress = latest_training_progress(
        arm_dir / "logs" / "train_segment_to_epoch080.log"
    )

    if eval_complete and paused:
        stage = "paused_after_epoch080"
        remaining_seconds = 0.0
    elif eval_complete:
        stage = "epoch080_eval_complete_awaiting_pause"
        remaining_seconds = 60.0
    elif checkpoint.is_file():
        stage = "epoch080_evaluating"
        if prediction_count > 0:
            remaining_seconds = max(
                120.0,
                (20 - prediction_count) * (evaluation_seconds - 120.0) / 20.0
                + 120.0,
            )
        else:
            remaining_seconds = evaluation_seconds
    else:
        stage = "epoch080_training"
        train_remaining = (
            float(progress["cumulative_projected_remaining_seconds"])
            if progress is not None
            else 3600.0
        )
        remaining_seconds = train_remaining + evaluation_seconds

    return {
        "variant": variant,
        "stage": stage,
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "eval_dir": str(eval_dir),
        "eval_complete": eval_complete,
        "prediction_count": prediction_count,
        "paused": paused,
        "training_progress": progress,
        "estimated_remaining_seconds": remaining_seconds,
    }


def metric(eval_dir: Path) -> dict[str, Any] | None:
    summary = read_json(eval_dir / "reports" / "val_quick_global_eval_summary.json")
    if summary is None:
        return None
    return {
        "dice": float(summary["mean_global_dice_per_finding"]),
        "hit_rate": float(summary["hit_rate"]),
        "hits": int(summary["total_hits"]),
        "findings": int(summary["total_findings"]),
    }


def audit_checkpoint(path: Path, expected_update: int) -> dict[str, Any]:
    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "global_update": None,
        "has_network_weights": False,
        "has_optimizer": False,
        "has_grad_scaler": False,
        "resumable": False,
        "error": None,
    }
    if not path.is_file():
        result["error"] = "missing checkpoint"
        return result
    try:
        import torch

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        result["global_update"] = int(checkpoint.get("global_update", -1))
        result["has_network_weights"] = "network_weights" in checkpoint
        result["has_optimizer"] = "optimizer" in checkpoint
        result["has_grad_scaler"] = "grad_scaler" in checkpoint
        result["resumable"] = bool(
            result["global_update"] == expected_update
            and result["has_network_weights"]
            and result["has_optimizer"]
            and result["has_grad_scaler"]
        )
        del checkpoint
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def refresh_main_summary(group_dir: Path, exp_dir: Path) -> None:
    script = Path(__file__).with_name("summarize_008_results.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--group-dir",
            str(group_dir),
            "--output-json",
            str(exp_dir / "reports" / "dual_branch_ablation_summary.json"),
            "--output-md",
            str(exp_dir / "reports" / "dual_branch_ablation_report.md"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def write_pause_report(
    group_dir: Path,
    exp_dir: Path,
    status: dict[str, Any],
    output_json: Path,
    output_md: Path,
) -> None:
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        arm_dir = group_dir / variant
        trajectory = {
            str(epoch): metric(arm_dir / f"eval_epoch{epoch:03d}_val20")
            for epoch in REPORT_EPOCHS
        }
        proposal = read_json(
            arm_dir
            / "eval_epoch080_val20"
            / "reports"
            / "proposal_diagnostics.json"
        )
        checkpoint = arm_dir / "checkpoints" / "checkpoint_update_008000.pth"
        materialized = (
            arm_dir / "model_epoch080" / "fold_0" / "checkpoint_final.pth"
        )
        variants[variant] = {
            "val20": trajectory,
            "epoch080_proposal": (
                proposal.get("thresholds", {}) if proposal is not None else None
            ),
            "checkpoint_audit": audit_checkpoint(checkpoint, 8000),
            "materialized_model_path": str(materialized),
            "materialized_model_exists": materialized.is_file(),
            "materialized_matches_checkpoint": bool(
                materialized.is_file()
                and checkpoint.is_file()
                and os.path.samefile(materialized, checkpoint)
            ),
            "paused_before_update10000": arm_is_paused(arm_dir, 10000),
        }

    result = {
        "created_at_utc": utc_now_iso(),
        "experiment": "008_voxtell_dual_branch_proposal_refinement_ablation",
        "run_group": group_dir.name,
        "state": "paused_after_epoch080_val20",
        "completed_optimizer_updates": 8000,
        "remaining_optimizer_updates_if_resumed": 2000,
        "source_baseline": {
            "dice": 0.3907023703162601,
            "hit_rate": 0.7419354838709677,
            "hits": 23,
            "findings": 31,
        },
        "monitor_status": status,
        "variants": variants,
        "resume_command": (
            "python /workspace/scripts/rexgroundingct/"
            "monitor_008_pause_after_epoch.py "
            f"--group-dir {group_dir} --resume"
        ),
        "interpretation_limit": (
            "No same-source single-branch continuation control is present. "
            "Val20 contains 31 findings and is an interim probe, not final "
            "challenge evidence."
        ),
    }
    write_json(output_json, result)

    lines = [
        "# Experiment 008 Pause After Epoch 80",
        "",
        f"- Run group: `{group_dir.name}`",
        "- State: paused after all epoch-80 val20 evaluations",
        "- Completed updates: `8000 / 10000`",
        "- Epoch-100 events consumed after pause: `0`",
        "- Resume state: network, optimizer, AMP scaler, LR horizon, and update "
        "counter preserved in each update-8000 checkpoint",
        "",
        "## Final Val20 Trajectory Before Pause",
        "",
        "| Variant | e0 | e20 | e40 | e60 | e80 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        values = []
        for epoch in REPORT_EPOCHS:
            value = variants[variant]["val20"][str(epoch)]
            values.append(
                (
                    f"{value['dice']:.4f} / {value['hit_rate']:.4f} "
                    f"({value['hits']}/{value['findings']})"
                )
                if value is not None
                else "-"
            )
        lines.append(f"| `{variant}` | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Cells are `Dice / hit rate (hits/findings)`.",
            "",
            "## Resume Audit",
            "",
            "| Variant | Update | Network | Optimizer | Scaler | Model link | Resumable |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in VARIANTS:
        value = variants[variant]
        audit = value["checkpoint_audit"]
        lines.append(
            f"| `{variant}` | {audit['global_update']} | "
            f"{audit['has_network_weights']} | {audit['has_optimizer']} | "
            f"{audit['has_grad_scaler']} | "
            f"{value['materialized_matches_checkpoint']} | "
            f"{audit['resumable']} |"
        )
    lines.extend(
        [
            "",
            "## Resume",
            "",
            "Remove the installed pause markers with:",
            "",
            "```bash",
            result["resume_command"],
            "```",
            "",
            "The existing supervisors then continue from update 8000 to update "
            "10000 without resetting optimizer, AMP scaler, LR horizon, schedule "
            "cursor, or data order.",
            "",
            "## Interpretation Limit",
            "",
            result["interpretation_limit"],
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")
    refresh_main_summary(group_dir, exp_dir)


def format_status(status: dict[str, Any]) -> str:
    lines = [
        (
            f"[{status['observed_at_local']}] overall={status['overall_stage']} "
            f"eta={status['estimated_completion_at_local']}"
        )
    ]
    for variant, arm in status["arms"].items():
        progress = arm["training_progress"]
        progress_text = (
            f"{progress['completed_segment_updates']}/2000"
            if progress is not None
            else "-"
        )
        lines.append(
            f"  {variant}: stage={arm['stage']} train={progress_text} "
            f"predictions={arm['prediction_count']}/20 paused={arm['paused']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=600.0)
    parser.add_argument("--evaluation-seconds", type=float, default=900.0)
    parser.add_argument("--next-target-update", type=int, default=10000)
    parser.add_argument(
        "--timezone",
        default=os.environ.get("TZ", "US/Pacific"),
    )
    parser.add_argument("--status-json", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    group_dir = args.group_dir.resolve()
    exp_dir = group_dir.parent.parent
    status_json = args.status_json or (
        group_dir / "control" / "epoch080_pause_monitor_status.json"
    )
    log_file = args.log_file or (
        exp_dir / "logs" / f"{group_dir.name}_epoch080_pause_monitor.log"
    )
    output_json = args.output_json or (
        exp_dir / "reports" / "dual_branch_pause_epoch080_summary.json"
    )
    output_md = args.output_md or (
        exp_dir / "reports" / "dual_branch_pause_epoch080_report.md"
    )

    if args.resume:
        removed = release_pause(group_dir, args.next_target_update)
        print(json.dumps({"removed": removed}, indent=2))
        return 0

    policy = install_pause(group_dir, args.next_target_update)
    if args.install_only:
        print(json.dumps(policy, indent=2, sort_keys=True))
        return 0

    log_file.parent.mkdir(parents=True, exist_ok=True)
    while True:
        arms = {
            variant: inspect_arm(
                group_dir,
                variant,
                args.evaluation_seconds,
                args.next_target_update,
            )
            for variant in VARIANTS
        }
        remaining = max(
            float(value["estimated_remaining_seconds"]) for value in arms.values()
        )
        now_local = datetime.now(ZoneInfo(args.timezone))
        completion = now_local + timedelta(seconds=remaining)
        all_eval_complete = all(value["eval_complete"] for value in arms.values())
        all_paused = all(value["paused"] for value in arms.values())
        overall_stage = (
            "paused_after_epoch080"
            if all_eval_complete and all_paused
            else "awaiting_pause"
            if all_eval_complete
            else "epoch080_in_progress"
        )
        status = {
            "observed_at_utc": utc_now_iso(),
            "observed_at_local": now_local.isoformat(),
            "overall_stage": overall_stage,
            "estimated_remaining_seconds": remaining,
            "estimated_completion_at_local": completion.isoformat(),
            "poll_seconds": args.poll_seconds,
            "pause_policy": policy,
            "arms": arms,
        }
        write_json(status_json, status)
        message = format_status(status)
        with log_file.open("a") as handle:
            handle.write(message + "\n")
        print(message, flush=True)

        if all_eval_complete and all_paused:
            write_pause_report(
                group_dir,
                exp_dir,
                status,
                output_json,
                output_md,
            )
            (group_dir / "control" / ".epoch080_pause_complete").touch()
            print(f"Wrote epoch-80 pause report: {output_md}", flush=True)
            return 0
        if args.once:
            return 0
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
