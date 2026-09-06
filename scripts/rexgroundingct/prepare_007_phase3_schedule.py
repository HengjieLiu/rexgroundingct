#!/usr/bin/env python3
"""Verify Exp007 schedule continuity and materialize the phase-3 slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def verify_prefix(prefix: Path, full: Path, events: int) -> None:
    if line_count(prefix) != events:
        raise ValueError(f"{prefix} has {line_count(prefix)} lines, expected {events}")
    with prefix.open("rb") as lhs, full.open("rb") as rhs:
        for index, expected in enumerate(lhs):
            if rhs.readline() != expected:
                raise ValueError(f"Schedule prefix mismatch at line {index}: {prefix} vs {full}")


def materialize_slice(
    original_schedule: Path,
    prior_extended_schedule: Path,
    extended_schedule: Path,
    output_schedule: Path,
    manifest_path: Path,
    start_event: int = 80000,
    events: int = 40000,
    original_events: int = 40000,
    prior_events: int = 80000,
) -> dict:
    if start_event < 0 or events < 1:
        raise ValueError("start_event must be >= 0 and events must be positive")
    if line_count(original_schedule) != original_events:
        raise ValueError(f"Original schedule line count mismatch: {original_schedule}")
    verify_prefix(original_schedule, prior_extended_schedule, original_events)
    verify_prefix(prior_extended_schedule, extended_schedule, prior_events)
    extended_count = line_count(extended_schedule)
    if extended_count < start_event + events:
        raise ValueError(
            f"Extended schedule has {extended_count} lines, but slice ends at {start_event + events}"
        )

    stats: Counter[str] = Counter()
    output_lines: list[str] = []
    with extended_schedule.open() as source:
        for source_index, line in enumerate(source):
            if source_index < start_event:
                continue
            if source_index >= start_event + events:
                break
            event = json.loads(line)
            if int(event.get("event_index", -1)) != source_index:
                raise ValueError(f"Extended event index mismatch at line {source_index}")
            event["source_event_index"] = source_index
            event["event_index"] = source_index - start_event
            stats.update(event.get("stats", {}))
            output_lines.append(json.dumps(event, sort_keys=True))

    if len(output_lines) != events:
        raise ValueError(f"Output schedule has {len(output_lines)} events, expected {events}")
    output_schedule.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_schedule.with_name(f".{output_schedule.name}.tmp.{os.getpid()}")
    tmp.write_text("\n".join(output_lines) + "\n")
    os.replace(tmp, output_schedule)
    payload = {
        "schedule_schema_version": 2,
        "purpose": f"exp007 phase3 events {start_event}..{start_event + events - 1} rewritten to zero-based line order",
        "source_original_schedule": str(original_schedule),
        "source_original_schedule_sha256": sha256(original_schedule),
        "source_prior_extended_schedule": str(prior_extended_schedule),
        "source_prior_extended_schedule_sha256": sha256(prior_extended_schedule),
        "source_extended_schedule": str(extended_schedule),
        "source_extended_schedule_sha256": sha256(extended_schedule),
        "output_jsonl": str(output_schedule),
        "output_jsonl_sha256": sha256(output_schedule),
        "events": events,
        "extended_events": extended_count,
        "source_event_start_inclusive": start_event,
        "source_event_stop_exclusive": start_event + events,
        "event_index_policy": "source event_index preserved as source_event_index; local event_index rewritten to 0..events-1",
        "ddp_world_size": 4,
        "effective_global_batch_size": 4,
        "event_consumption_rule": "event_index = update * 4 + rank",
        "stats": dict(stats),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    manifest_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-schedule", type=Path, required=True)
    parser.add_argument("--prior-extended-schedule", type=Path, required=True)
    parser.add_argument("--extended-schedule", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--start-event", type=int, default=80000)
    parser.add_argument("--events", type=int, default=40000)
    parser.add_argument("--original-events", type=int, default=40000)
    parser.add_argument("--prior-events", type=int, default=80000)
    args = parser.parse_args()
    payload = materialize_slice(
        args.original_schedule,
        args.prior_extended_schedule,
        args.extended_schedule,
        args.output_jsonl,
        args.manifest_json,
        args.start_event,
        args.events,
        args.original_events,
        args.prior_events,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
