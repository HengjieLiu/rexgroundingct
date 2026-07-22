#!/usr/bin/env python3
"""Create an evaluator-compatible JSON for a selected ReXGroundingCT split."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import REX_METADATA, load_split_entries, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    entries = load_split_entries(args.metadata, [args.split])
    if args.limit is not None:
        entries = entries[: args.limit]
    output = {
        "test": [
            {
                "name": entry["name"],
                "seg_path": entry["name"],
                "findings": entry.get("findings", {}),
                "categories": entry.get("categories", {}),
            }
            for entry in entries
        ]
    }
    write_json(args.output_json, output)
    print(f"Wrote {len(entries)} {args.split} entries to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
