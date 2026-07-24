#!/usr/bin/env python3
"""Materialize a deterministic fixed-size ReXGroundingCT probe subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import REX_METADATA, command_string, load_split_entries, sha256_file, utc_now_iso, write_json


def evaluator_entry(entry: dict) -> dict:
    return {
        "name": entry["name"],
        "seg_path": entry["name"],
        "findings": entry.get("findings", {}),
        "categories": entry.get("categories", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, default=None)
    args = parser.parse_args()

    if args.size < 1:
        raise ValueError("--size must be >= 1")

    entries = load_split_entries(args.metadata, [args.split])
    if args.size > len(entries):
        raise ValueError(f"--size {args.size} exceeds {args.split} split size {len(entries)}")

    rng = np.random.default_rng(args.seed)
    selected_indices = [int(index) for index in rng.permutation(len(entries))[: args.size]]
    selected_entries = [entries[index] for index in selected_indices]

    output = {"test": [evaluator_entry(entry) for entry in selected_entries]}
    write_json(args.output_json, output)

    manifest_path = args.manifest_json or args.output_json.with_suffix(".manifest.json")
    manifest = {
        "created_at_utc": utc_now_iso(),
        "command": command_string(),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "source_split": args.split,
        "source_split_size": len(entries),
        "selection_seed": args.seed,
        "selection_algorithm": "np.random.default_rng(seed).permutation(split_size)[:size]",
        "size": args.size,
        "selected_source_indices": selected_indices,
        "selected_case_names": [entry["name"] for entry in selected_entries],
        "output_json": str(args.output_json),
        "output_json_sha256": sha256_file(args.output_json),
    }
    write_json(manifest_path, manifest)
    print(f"Wrote {args.size} fixed {args.split} probe cases to {args.output_json}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
