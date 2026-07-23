#!/usr/bin/env python3
"""Run the canonical lightweight repo workflow checks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    print("+ " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    return completed.returncode


def run_collect(cmd: list[str]) -> tuple[int, str]:
    print("+ " + " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout.strip()
    if output:
        print(output)
    return completed.returncode, output


def main() -> int:
    python = sys.executable
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    checks: list[tuple[str, int]] = []
    checks.append(
        (
            "challenge_info unit tests",
            run(
                [python, "-m", "unittest", "challenge_info.tests.test_update_challenge_info"],
                env=env,
            ),
        )
    )
    checks.append(
        (
            "experiment consistency",
            run([python, "scripts/rexgroundingct/check_experiment_consistency.py"]),
        )
    )

    compile_targets = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "scripts" / "rexgroundingct").glob("*.py")
    )
    compile_targets.extend(
        [
            "challenge_info/update_challenge_info.py",
            "dataset/download_ct_rate_challenge_subset.py",
        ]
    )
    checks.append(("python compile", run([python, "-m", "py_compile", *compile_targets])))
    checks.append(("git diff whitespace", run(["git", "diff", "--check"])))
    checks.append(("git cached whitespace", run(["git", "diff", "--cached", "--check"])))

    cache_status, cache_output = run_collect(["git", "ls-files", "*__pycache__*", "*.pyc"])
    checks.append(("tracked Python caches", 1 if cache_output else cache_status))

    failed = [name for name, status in checks if status != 0]
    if failed:
        print("Workflow check failed: " + ", ".join(failed))
        return 1
    print("Workflow check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
