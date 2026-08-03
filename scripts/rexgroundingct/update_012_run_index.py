#!/usr/bin/env python3
"""Atomically refresh the experiment-level Exp012 multi-run index."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def infer_profile(group: Path, manifest: dict[str, Any]) -> str:
    if manifest.get("profile"):
        return str(manifest["profile"])
    if "r02_replay_ratio" in group.name:
        return "r02_2d_diffuse_replay_ratio"
    return "r01_core_replay50"


def find_report(group: Path) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    candidates = (
        (
            group / "reports" / "exp012_r02_progress.json",
            group / "reports" / "exp012_r02_progress.md",
        ),
        (
            group / "reports" / "exp012_progress.json",
            group / "reports" / "exp012_progress.md",
        ),
    )
    for json_path, markdown_path in candidates:
        if json_path.is_file():
            return (
                json_path,
                markdown_path if markdown_path.is_file() else None,
                read_json(json_path),
            )
    return None, None, None


def group_status(group: Path, report: dict[str, Any] | None) -> str:
    if (group / ".selection_ready").is_file():
        return "selection_ready"
    if (group / ".orchestrator_failed").is_file():
        return "failed"
    if (group / ".orchestrator_running").is_file():
        return "running"
    if report and report.get("status"):
        return str(report["status"])
    return "created"


def build_index(experiment_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for group in sorted((experiment_dir / "runs").glob("exp012_category_specialists*")):
        if not group.is_dir():
            continue
        manifest_path = group / "run_group_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        json_report, markdown_report, report = find_report(group)
        records.append(
            {
                "profile": infer_profile(group, manifest),
                "run_group": group.name,
                "group_dir": str(group),
                "status": group_status(group, report),
                "report_json": str(json_report) if json_report else None,
                "report_markdown": str(markdown_report) if markdown_report else None,
                "selection_ready_marker": str(group / ".selection_ready"),
                "selection_ready": (group / ".selection_ready").is_file(),
                "updated_at_utc": report.get("updated_at_utc")
                if report
                else manifest.get("updated_at_utc"),
            }
        )
    return {
        "schema_version": 1,
        "experiment": "012_voxtell_category_specialists_replay50_cont100",
        "experiment_status": "active",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": records,
    }


def render_markdown(index: dict[str, Any]) -> str:
    lines = [
        "# Exp012 Run Index",
        "",
        "- Experiment status: `active`",
        f"- Updated: `{index['updated_at_utc']}`",
        "",
        "| Profile | Run group | Status | Selection ready | Report |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for record in index["runs"]:
        report = record["report_markdown"] or record["report_json"] or "pending"
        lines.append(
            f"| `{record['profile']}` | `{record['run_group']}` | `{record['status']}` "
            f"| {'yes' if record['selection_ready'] else 'no'} | `{report}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    index = build_index(args.experiment_dir)
    reports = args.experiment_dir / "reports"
    atomic_write(
        reports / "exp012_run_index.json",
        json.dumps(index, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(reports / "exp012_run_index.md", render_markdown(index))
    print(f"indexed_runs={len(index['runs'])} experiment_status=active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
