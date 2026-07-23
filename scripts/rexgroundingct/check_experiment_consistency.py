#!/usr/bin/env python3
"""Check repo/runtime experiment consistency before committing or launching."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from common import (
    EXPERIMENTS,
    EXP_ROOT,
    REPO_EXPERIMENT_ROOT,
    REPO_ROOT,
    canonical_config_path,
    read_json,
    sha256_file,
)


BLOCKED_GIT_PATTERNS = [
    "experiments/*/runtime",
    "experiments/**/logs/**",
    "experiments/**/predictions/**",
    "experiments/**/checkpoints/**",
    "experiments/**/eval/**",
    "experiments/**/*.log",
    "experiments/**/*.nii",
    "experiments/**/*.nii.gz",
    "experiments/**/*.pt",
    "experiments/**/*.pth",
    "experiments/**/*.ckpt",
    "experiments/**/*.safetensors",
]


def is_python_cache_path(path: str) -> bool:
    posix = PurePosixPath(path)
    return "__pycache__" in posix.parts or posix.suffix in {".pyc", ".pyo", ".pyd"}


def matches_any(path: str, patterns: list[str]) -> bool:
    posix = PurePosixPath(path)
    return any(posix.match(pattern) for pattern in patterns)


def git_paths(args: list[str]) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return [line for line in output.splitlines() if line]


def check_git_artifacts(errors: list[str], warnings: list[str]) -> None:
    try:
        tracked = set(git_paths(["ls-files"]))
        staged = set(git_paths(["diff", "--cached", "--name-only", "--diff-filter=ACMR"]))
        untracked = set(git_paths(["ls-files", "--others", "--exclude-standard"]))
    except Exception as exc:
        warnings.append(f"Could not inspect git paths: {exc}")
        return

    for label, paths in [
        ("tracked", tracked),
        ("staged", staged),
        ("untracked", untracked),
    ]:
        bad = sorted(path for path in paths if matches_any(path, BLOCKED_GIT_PATTERNS))
        for path in bad:
            errors.append(f"Blocked {label} experiment artifact in git view: {path}")

    for label, paths in [
        ("tracked", tracked),
        ("staged", staged),
    ]:
        bad = sorted(path for path in paths if is_python_cache_path(path))
        for path in bad:
            errors.append(f"Blocked {label} Python cache artifact in git view: {path}")


def compare_hash(
    label: str,
    expected_path: Path,
    observed_path: Path,
    errors: list[str],
    warnings: list[str],
    *,
    required: bool,
) -> None:
    if not expected_path.exists():
        message = f"Missing expected {label} source: {expected_path}"
        (errors if required else warnings).append(message)
        return
    if not observed_path.exists():
        message = f"Missing observed {label}: {observed_path}"
        (errors if required else warnings).append(message)
        return
    expected = sha256_file(expected_path)
    observed = sha256_file(observed_path)
    if expected != observed:
        errors.append(
            f"{label} drift: {observed_path} sha256={observed} "
            f"does not match {expected_path} sha256={expected}"
        )


def check_sync_manifest(
    experiment: str,
    repo_dir: Path,
    runtime_dir: Path,
    errors: list[str],
    warnings: list[str],
    *,
    runtime_required: bool,
) -> None:
    manifest = repo_dir / "sync_manifest.json"
    if not manifest.exists():
        warnings.append(f"No repo sync manifest yet for {experiment}: {manifest}")
        return

    data: dict[str, Any] = read_json(manifest)
    config = canonical_config_path(experiment)
    recorded_config_sha = data.get("canonical_config", {}).get("sha256")
    if recorded_config_sha and config.exists() and recorded_config_sha != sha256_file(config):
        errors.append(
            f"{experiment} sync manifest has stale canonical config hash; run sync_experiment_index.py"
        )

    report_record = data.get("report", {})
    repo_report_path = report_record.get("repo_path")
    runtime_report_path = report_record.get("path")
    if repo_report_path and runtime_report_path:
        repo_report = REPO_ROOT / repo_report_path
        runtime_report = Path(runtime_report_path)
        if runtime_report.exists():
            compare_hash(
                f"{experiment} report snapshot",
                runtime_report,
                repo_report,
                errors,
                warnings,
                required=False,
            )
        elif runtime_required and report_record.get("exists"):
            errors.append(f"{experiment} recorded runtime report is missing: {runtime_report}")

    runtime_config_record = data.get("runtime_config_snapshot", {})
    runtime_config_path = runtime_config_record.get("path")
    runtime_config_sha = runtime_config_record.get("sha256")
    if runtime_config_path and runtime_config_sha:
        runtime_config = Path(runtime_config_path)
        if runtime_config.exists() and sha256_file(runtime_config) != runtime_config_sha:
            errors.append(
                f"{experiment} runtime config changed since repo sync: {runtime_config}"
            )
        elif runtime_required and not runtime_config.exists():
            errors.append(f"{experiment} recorded runtime config is missing: {runtime_config}")


def check_one(
    experiment: str,
    runtime_root: Path,
    repo_root: Path,
    errors: list[str],
    warnings: list[str],
    *,
    strict_runtime: bool,
) -> None:
    definition = EXPERIMENTS[experiment]
    repo_dir = repo_root / experiment
    runtime_dir = runtime_root / experiment
    config = canonical_config_path(experiment)
    runtime_config = runtime_dir / "config" / config.name
    report = runtime_dir / definition["primary_report"]
    repo_report = repo_dir / "report.md"

    if not config.exists():
        errors.append(f"Missing canonical config for {experiment}: {config}")
        return

    link = repo_dir / "runtime"
    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            errors.append(f"{experiment} runtime link exists but is not a symlink: {link}")
        else:
            target = Path(os.readlink(link))
            if target != runtime_dir:
                errors.append(
                    f"{experiment} runtime symlink points to {target}, expected {runtime_dir}"
                )

    runtime_available = runtime_root.exists() and runtime_dir.exists()
    if not runtime_available:
        message = f"Runtime directory unavailable for {experiment}: {runtime_dir}"
        (errors if strict_runtime else warnings).append(message)
        check_sync_manifest(
            experiment,
            repo_dir,
            runtime_dir,
            errors,
            warnings,
            runtime_required=False,
        )
        return

    compare_hash(
        f"{experiment} config snapshot",
        config,
        runtime_config,
        errors,
        warnings,
        required=strict_runtime,
    )
    if repo_report.exists() and report.exists():
        compare_hash(
            f"{experiment} report snapshot",
            report,
            repo_report,
            errors,
            warnings,
            required=False,
        )
    check_sync_manifest(
        experiment,
        repo_dir,
        runtime_dir,
        errors,
        warnings,
        runtime_required=strict_runtime,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=EXP_ROOT)
    parser.add_argument("--repo-experiment-root", type=Path, default=REPO_EXPERIMENT_ROOT)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--strict-runtime", action="store_true")
    parser.add_argument("--skip-git", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    if not args.skip_git:
        check_git_artifacts(errors, warnings)
    for experiment in args.experiments:
        check_one(
            experiment,
            runtime_root=args.runtime_root,
            repo_root=args.repo_experiment_root,
            errors=errors,
            warnings=warnings,
            strict_runtime=args.strict_runtime,
        )

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"Experiment consistency check failed with {len(errors)} error(s).")
        return 1
    print(f"Experiment consistency check passed with {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
