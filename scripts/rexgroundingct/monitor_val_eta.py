#!/usr/bin/env python3
"""Poll VoxTell validation progress and print an ETA."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    return str(timedelta(seconds=int(seconds)))


def count_statuses(exp_dir: Path) -> tuple[int, int]:
    written = 0
    failed = 0
    for path in sorted((exp_dir / "logs").glob("val_shard_*_status.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for case in data.get("cases", []):
            status = case.get("status")
            if status in {"written", "skipped_existing"}:
                written += 1
            elif status in {"missing_ct", "missing_gt", "error"}:
                failed += 1
    return written, failed


def progress_counts(exp_dir: Path) -> tuple[int, int, int]:
    pred_count = len(list((exp_dir / "predictions").glob("*.nii.gz")))
    status_written, failed = count_statuses(exp_dir)
    done = max(pred_count, status_written)
    return done, failed, pred_count


def snapshot(
    exp_dir: Path,
    total: int,
    start_count: int,
    start_time: float,
    last_count: int,
    last_time: float,
) -> tuple[int, float]:
    done, failed, pred_count = progress_counts(exp_dir)
    status_written, _failed_from_status = count_statuses(exp_dir)
    now = time.time()
    elapsed = now - start_time
    average_delta = done - start_count
    average_rate = average_delta / elapsed if elapsed > 0 and average_delta > 0 else 0
    recent_delta = done - last_count
    recent_elapsed = now - last_time
    recent_rate = recent_delta / recent_elapsed if recent_elapsed > 0 and recent_delta > 0 else 0
    rate = recent_rate or average_rate
    remaining = max(total - done, 0)
    eta_seconds = remaining / rate if rate > 0 else None
    eta_time = datetime.fromtimestamp(now + eta_seconds).isoformat(timespec="seconds") if eta_seconds is not None else "n/a"
    report = exp_dir / "reports" / "val_evaluation_report.md"
    eval_json = exp_dir / "eval" / "val_official_eval.json"
    print(
        f"{datetime.now().isoformat(timespec='seconds')} "
        f"done={done}/{total} failed={failed} "
        f"pred_files={pred_count} status_written={status_written} "
        f"avg_rate={average_rate:.3f}/s recent_rate={recent_rate:.3f}/s "
        f"eta={fmt_duration(eta_seconds)} eta_at={eta_time} "
        f"report_exists={report.exists()} eval_exists={eval_json.exists()}",
        flush=True,
    )
    return done, now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct/001_voxtell_v1_1_miccai200_val_eval"),
    )
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--total", type=int, default=200)
    args = parser.parse_args()

    start_count, _failed, _pred_count = progress_counts(args.exp_dir)
    start_time = time.time()
    last_count = start_count
    last_time = start_time
    while True:
        last_count, last_time = snapshot(args.exp_dir, args.total, start_count, start_time, last_count, last_time)
        if (args.exp_dir / "reports" / "val_evaluation_report.md").exists():
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
