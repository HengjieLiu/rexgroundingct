#!/usr/bin/env python3
"""Sync a small repo-local experiment index from runtime experiment folders."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from common import (
    EXPERIMENTS,
    EXP_ROOT,
    REPO_ROOT,
    REPO_EXPERIMENT_ROOT,
    VOXTELL_SUBMODULE,
    canonical_config_path,
    git_commit,
    read_json,
    repo_relative,
    sha256_file,
    utc_now_iso,
    write_json,
)


def rel_to_repo(path: Path) -> str:
    return repo_relative(path)


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def copy_small_artifact(source: Path, destination: Path, *, max_bytes: int) -> dict[str, Any]:
    record = artifact_record(source)
    record["repo_path"] = rel_to_repo(destination)
    record["copied"] = False
    if not source.exists():
        return record
    if source.stat().st_size > max_bytes:
        record["skipped_reason"] = f"source exceeds max copy size {max_bytes} bytes"
        return record
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    record["copied"] = True
    record["repo_sha256"] = sha256_file(destination)
    return record


def ensure_runtime_symlink(repo_dir: Path, runtime_dir: Path, *, create: bool) -> str:
    link = repo_dir / "runtime"
    if not create:
        return "disabled"
    if link.is_symlink():
        current = Path(os.readlink(link))
        if current == runtime_dir:
            return "exists"
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"Refusing to replace non-symlink path: {link}")
    link.symlink_to(runtime_dir)
    return "created"


def summarize_eval(eval_json: Path) -> dict[str, Any] | None:
    if not eval_json.exists():
        return None
    data = read_json(eval_json)
    summary = data.get("summary", {})
    keys = [
        "total_cases",
        "total_findings",
        "mean_global_dice_per_finding",
        "mean_global_dice_per_case",
        "hit_rate",
        "overall_instance_precision",
        "overall_instance_recall",
        "overall_instance_f1",
        "total_gt_instances",
        "total_pred_instances",
        "total_tp",
        "total_fp",
        "total_fn",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def status_for(runtime_dir: Path, report: Path, eval_json: Path) -> str:
    checkpoints = runtime_dir / "checkpoints"
    if report.exists() and eval_json.exists():
        return "evaluation_complete"
    if checkpoints.exists() and any(checkpoints.iterdir()):
        return "training_artifacts_present"
    if runtime_dir.exists():
        return "runtime_initialized"
    return "planned"


def render_experiment_readme(record: dict[str, Any]) -> str:
    report_path = record["report"].get("repo_path") if record["report"].get("copied") else "not available"
    lines = [
        "---",
        "created: 2026-07-22",
        "updated: 2026-07-22",
        "status: active",
        "---",
        "",
        f"# {record['title']}",
        "",
        "This is a repo-local index for a runtime experiment. Edit the canonical",
        "config in the repo, and keep large runtime artifacts under `/mnt/shengdata1`.",
        "",
        "## Folder Map",
        "",
        "- `README.md`: this experiment's status and ownership rules.",
        "- `metrics_summary.json`: synced metrics and provenance summary.",
        "- `sync_manifest.json`: hashes, paths, and sync provenance.",
        "- `report.md`: small copied runtime report when available.",
        "- `runtime`: ignored symlink to heavyweight runtime outputs.",
        "",
        "## Status",
        "",
        f"- Status: `{record['status']}`",
        f"- Canonical config: `{record['canonical_config']['repo_path']}`",
        f"- Runtime directory: `{record['runtime_dir']}`",
        "- Runtime link: `runtime` is ignored by git and points to the runtime directory.",
        "",
        "## Synced Small Artifacts",
        "",
        f"- Report snapshot: `{report_path}`",
        f"- Metrics summary: `metrics_summary.json`",
        f"- Sync manifest: `sync_manifest.json`",
        "",
        "## Ownership Rule",
        "",
        "- Configs are canonical in the repo.",
        "- Runtime config files are snapshots and should not be edited by hand.",
        "- Logs, predictions, checkpoints, and raw evaluator outputs stay on `/mnt/shengdata1`.",
        "- Re-run the sync script after an evaluation or training run writes a new report.",
        "",
    ]
    return "\n".join(lines)


def yaml_quote(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(str(value))


def render_registry(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Generated by scripts/rexgroundingct/sync_experiment_index.py",
        "# Repo configs are canonical; runtime files under /mnt are provenance/output.",
        "experiments:",
    ]
    for record in records:
        metrics = record.get("metrics") or {}
        repo_report = record["report"].get("repo_path") if record["report"].get("copied") else None
        lines.extend(
            [
                f"  - id: {yaml_quote(record['id'])}",
                f"    title: {yaml_quote(record['title'])}",
                f"    status: {yaml_quote(record['status'])}",
                f"    canonical_config: {yaml_quote(record['canonical_config']['repo_path'])}",
                f"    runtime_dir: {yaml_quote(record['runtime_dir'])}",
                f"    runtime_config_snapshot: {yaml_quote(record['runtime_config_snapshot']['path'])}",
                f"    runtime_config_matches_canonical: {yaml_quote(record['runtime_config_matches_canonical'])}",
                f"    repo_report: {yaml_quote(repo_report)}",
                f"    runtime_report: {yaml_quote(record['report'].get('path'))}",
                f"    repo_metrics: {yaml_quote(record['repo_metrics'])}",
                f"    mean_global_dice_per_finding: {yaml_quote(metrics.get('mean_global_dice_per_finding'))}",
                f"    hit_rate: {yaml_quote(metrics.get('hit_rate'))}",
                f"    overall_instance_f1: {yaml_quote(metrics.get('overall_instance_f1'))}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_top_readme(records: list[dict[str, Any]]) -> str:
    lines = [
        "---",
        "created: 2026-07-22",
        "updated: 2026-07-22",
        "status: active",
        "---",
        "",
        "# ReXGroundingCT Experiments",
        "",
        "This directory is a lightweight repo-local index. It keeps configs and",
        "small summaries easy to inspect while large outputs remain on `/mnt/shengdata1`.",
        "",
        "## Folder Map",
        "",
        "- `README.md`: experiment index overview and drift policy.",
        "- `registry.yaml`: generated machine-readable experiment summary.",
        "- `<experiment-id>/README.md`: one experiment's status and ownership rules.",
        "- `<experiment-id>/metrics_summary.json`: small synced metrics/provenance summary.",
        "- `<experiment-id>/sync_manifest.json`: synced hash and path manifest.",
        "- `<experiment-id>/report.md`: small copied runtime report when available.",
        "- `<experiment-id>/runtime`: ignored symlink to heavyweight runtime outputs.",
        "",
        "## Drift Policy",
        "",
        "- Edit canonical configs under `configs/experiments/`.",
        "- Runtime configs are hashed snapshots copied at run start.",
        "- `experiments/*/runtime` symlinks are ignored by git.",
        "- Use `scripts/rexgroundingct/check_experiment_consistency.py` before committing.",
        "",
        "## Experiments",
        "",
        "| ID | Status | Report | Runtime |",
        "| --- | --- | --- | --- |",
    ]
    for record in records:
        report = record["report"].get("repo_path") if record["report"].get("copied") else ""
        lines.append(
            f"| `{record['id']}` | `{record['status']}` | `{report}` | `{record['runtime_dir']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def sync_one(
    experiment: str,
    runtime_root: Path,
    repo_root: Path,
    create_symlinks: bool,
    max_copy_mb: int,
) -> dict[str, Any]:
    definition = EXPERIMENTS[experiment]
    repo_dir = repo_root / experiment
    runtime_dir = runtime_root / experiment
    repo_dir.mkdir(parents=True, exist_ok=True)

    config = canonical_config_path(experiment)
    runtime_config = runtime_dir / "config" / config.name
    report = runtime_dir / definition["primary_report"]
    eval_json = runtime_dir / definition["primary_eval_json"]

    runtime_link_status = ensure_runtime_symlink(repo_dir, runtime_dir, create=create_symlinks)
    report_record = copy_small_artifact(
        report,
        repo_dir / "report.md",
        max_bytes=max_copy_mb * 1024 * 1024,
    )
    config_record = artifact_record(config)
    runtime_config_record = artifact_record(runtime_config)
    runtime_config_matches = (
        bool(config_record["sha256"])
        and config_record["sha256"] == runtime_config_record["sha256"]
    )
    eval_record = artifact_record(eval_json)
    metrics = summarize_eval(eval_json)

    record: dict[str, Any] = {
        "id": experiment,
        "title": definition["title"],
        "synced_at_utc": utc_now_iso(),
        "status": status_for(runtime_dir, report, eval_json),
        "repo_dir": rel_to_repo(repo_dir),
        "runtime_dir": str(runtime_dir),
        "runtime_link_status": runtime_link_status,
        "canonical_config": {
            **config_record,
            "repo_path": rel_to_repo(config),
        },
        "runtime_config_snapshot": runtime_config_record,
        "runtime_config_matches_canonical": runtime_config_matches,
        "report": report_record,
        "eval_json": eval_record,
        "repo_metrics": rel_to_repo(repo_dir / "metrics_summary.json"),
        "metrics": metrics,
        "repo_commit": git_commit(REPO_ROOT),
        "voxtell_commit": git_commit(VOXTELL_SUBMODULE),
    }

    write_json(repo_dir / "metrics_summary.json", record)
    write_json(repo_dir / "sync_manifest.json", record)
    (repo_dir / "README.md").write_text(render_experiment_readme(record))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=EXP_ROOT)
    parser.add_argument("--repo-experiment-root", type=Path, default=REPO_EXPERIMENT_ROOT)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--no-symlinks", action="store_true")
    parser.add_argument("--max-copy-mb", type=int, default=20)
    args = parser.parse_args()

    records = [
        sync_one(
            experiment,
            runtime_root=args.runtime_root,
            repo_root=args.repo_experiment_root,
            create_symlinks=not args.no_symlinks,
            max_copy_mb=args.max_copy_mb,
        )
        for experiment in args.experiments
    ]
    args.repo_experiment_root.mkdir(parents=True, exist_ok=True)
    (args.repo_experiment_root / "README.md").write_text(render_top_readme(records))
    (args.repo_experiment_root / "registry.yaml").write_text(render_registry(records))
    print(f"Synced {len(records)} experiment index entries under {args.repo_experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
