#!/usr/bin/env python3
"""Report whether the CT-RATE subset needed by ReXGroundingCT is ready."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import CT_ROOT, REX_METADATA, ct_rate_abs_path, load_split_entries, write_json


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def snapshot(metadata: Path, ct_root: Path, splits: list[str]) -> dict:
    entries = load_split_entries(metadata, splits)
    expected = [ct_rate_abs_path(entry["name"], ct_root) for entry in entries]
    present = [path for path in expected if path.is_file()]
    missing = [path for path in expected if not path.is_file()]
    incomplete = sorted(ct_root.rglob("*.incomplete")) if ct_root.exists() else []
    total_bytes = sum(path.stat().st_size for path in present)
    return {
        "metadata": str(metadata),
        "ct_root": str(ct_root),
        "splits": splits,
        "expected_files": len(expected),
        "present_files": len(present),
        "missing_files": len(missing),
        "incomplete_files": len(incomplete),
        "present_bytes": total_bytes,
        "present_human": human_size(total_bytes),
        "ready": len(missing) == 0 and len(incomplete) == 0,
        "first_missing": [str(path) for path in missing[:20]],
        "first_incomplete": [str(path) for path in incomplete[:20]],
    }


def print_report(report: dict) -> None:
    print(f"CT root: {report['ct_root']}")
    print(f"Splits: {' '.join(report['splits'])}")
    print(
        "Files: "
        f"{report['present_files']}/{report['expected_files']} present, "
        f"{report['missing_files']} missing, "
        f"{report['incomplete_files']} incomplete"
    )
    print(f"Present size: {report['present_human']}")
    print(f"Ready: {report['ready']}")
    if report["first_missing"]:
        print("First missing:")
        for path in report["first_missing"]:
            print(f"  {path}")
    if report["first_incomplete"]:
        print("First incomplete:")
        for path in report["first_incomplete"]:
            print(f"  {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val"])
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write readiness JSON.")
    parser.add_argument("--watch", action="store_true", help="Keep polling until ready.")
    parser.add_argument("--interval", type=int, default=300, help="Polling interval in seconds.")
    args = parser.parse_args()

    while True:
        report = snapshot(args.metadata, args.ct_root, args.splits)
        print_report(report)
        if args.json:
            write_json(args.json, report)
        if report["ready"]:
            return 0
        if not args.watch:
            return 1
        print(f"Sleeping {args.interval}s...\n", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
