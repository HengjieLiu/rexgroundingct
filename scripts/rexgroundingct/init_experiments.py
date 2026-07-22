#!/usr/bin/env python3
"""Create numbered experiment directories and baseline manifests."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

from common import (
    EXPERIMENTS,
    EXP_ROOT,
    REPO_ROOT,
    VOXTELL_SUBMODULE,
    command_string,
    env_snapshot,
    git_commit,
    snapshot_experiment_config,
    update_run_manifest,
    utc_now_iso,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", type=Path, default=EXP_ROOT)
    parser.add_argument("--overwrite-manifest", action="store_true")
    parser.add_argument("--overwrite-config-snapshot", action="store_true")
    args = parser.parse_args()

    for name in EXPERIMENTS:
        exp_dir = args.exp_root / name
        for child in ["config", "logs", "predictions", "eval", "reports", "checkpoints"]:
            (exp_dir / child).mkdir(parents=True, exist_ok=True)
        config_snapshot = snapshot_experiment_config(
            name,
            exp_dir=exp_dir,
            overwrite=args.overwrite_config_snapshot,
        )
        manifest = exp_dir / "run_manifest.json"
        base = {}
        if not manifest.exists() or args.overwrite_manifest:
            base = {
                "created_at_utc": utc_now_iso(),
                "command": command_string(),
                "env": env_snapshot(),
                "experiment": name,
                "python": platform.python_version(),
                "repo_commit": git_commit(REPO_ROOT),
                "repo_root": str(REPO_ROOT),
                "voxtell_commit": git_commit(VOXTELL_SUBMODULE),
                "voxtell_submodule": str(VOXTELL_SUBMODULE),
            }
        update_run_manifest(
            exp_dir,
            {
                **base,
                "config_snapshot": config_snapshot,
            },
        )
        print(f"ready: {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
