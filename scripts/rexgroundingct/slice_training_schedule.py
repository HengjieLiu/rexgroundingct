#!/usr/bin/env python3
"""Materialize a reindexed deterministic slice from a training schedule."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from common import sha256_file, utc_now_iso, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--start-event", type=int, required=True)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    args = parser.parse_args()

    if args.start_event < 0 or args.events < 1:
        raise ValueError("--start-event must be >= 0 and --events must be >= 1")
    source_lines = args.source_jsonl.read_text().splitlines()
    stop_event = args.start_event + args.events
    if stop_event > len(source_lines):
        raise ValueError(
            f"Requested source events [{args.start_event}, {stop_event}) but "
            f"{args.source_jsonl} has {len(source_lines)} lines"
        )

    stats: Counter = Counter()
    output_lines = []
    for local_index, source_index in enumerate(
        range(args.start_event, stop_event)
    ):
        event = json.loads(source_lines[source_index])
        observed_index = int(event.get("event_index", source_index))
        if observed_index != source_index:
            raise ValueError(
                f"Source line {source_index} has event_index {observed_index}"
            )
        event["source_event_index"] = source_index
        event["event_index"] = local_index
        slots = event.get("prompt_slots", [])
        stats["events"] += 1
        stats["positive_slots"] += sum(
            int(bool(slot.get("is_positive"))) for slot in slots
        )
        stats["negative_slots"] += sum(
            int(not bool(slot.get("is_positive"))) for slot in slots
        )
        stats["events_with_patch_starts"] += int(
            isinstance(event.get("patch_starts"), list)
        )
        output_lines.append(json.dumps(event, sort_keys=True))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_jsonl.with_name(
        f".{args.output_jsonl.name}.tmp.{os.getpid()}"
    )
    tmp.write_text("\n".join(output_lines) + "\n")
    os.replace(tmp, args.output_jsonl)
    manifest = {
        "created_at_utc": utc_now_iso(),
        "source_jsonl": str(args.source_jsonl),
        "source_jsonl_sha256": sha256_file(args.source_jsonl),
        "source_event_start_inclusive": args.start_event,
        "source_event_stop_exclusive": stop_event,
        "events": args.events,
        "event_reindexing": (
            "local event_index is 0..events-1; source_event_index preserves "
            "the original schedule line"
        ),
        "output_jsonl": str(args.output_jsonl),
        "output_jsonl_sha256": sha256_file(args.output_jsonl),
        "stats": dict(stats),
    }
    write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
