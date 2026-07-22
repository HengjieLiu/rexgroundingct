#!/usr/bin/env python3
"""Create or verify a runtime snapshot of a repo-canonical experiment config."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    EXP_ROOT,
    command_string,
    env_snapshot,
    git_commit,
    snapshot_experiment_config,
    update_run_manifest,
    REPO_ROOT,
    VOXTELL_SUBMODULE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--exp-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    exp_dir = args.exp_dir or (EXP_ROOT / args.experiment)
    record = snapshot_experiment_config(
        args.experiment,
        exp_dir=exp_dir,
        source_config=args.config,
        overwrite=args.overwrite,
    )
    update_run_manifest(
        exp_dir,
        {
            "experiment": args.experiment,
            "repo_root": str(REPO_ROOT),
            "repo_commit": git_commit(REPO_ROOT),
            "voxtell_submodule": str(VOXTELL_SUBMODULE),
            "voxtell_commit": git_commit(VOXTELL_SUBMODULE),
            "command": command_string(),
            "env": env_snapshot(),
            "config_snapshot": record,
        },
    )
    print(
        "Config snapshot "
        f"{record['status']}: {record['runtime_config_snapshot_path']} "
        f"sha256={record['runtime_config_snapshot_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
