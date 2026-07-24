#!/usr/bin/env python3
"""Create paired native/2 mm continuation schedules from a verified 20k stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import REX_METADATA, command_string, sha256_file, utc_now_iso, write_json
from train_text_conditioned_voxtell import PromptSlot
from voxtell_2mm import cache_case_paths


def read_jsonl(path: Path) -> tuple[list[str], list[dict]]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return lines, [json.loads(line) for line in lines]


def write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def paired_signature(event: dict) -> dict:
    return {
        "case_name": event["case_name"],
        "prompt_slots": event["prompt_slots"],
        "patch_seed": event["patch_seed"],
    }


def signature_hash(events: list[dict]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(json.dumps(paired_signature(event), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def starts_containing_points(
    points: np.ndarray,
    spatial_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    rng: np.random.Generator,
) -> list[int] | None:
    starts = []
    for axis, (dim, patch) in enumerate(zip(spatial_shape, patch_size)):
        if dim <= patch:
            starts.append(0)
            continue
        low = max(0, int(points[:, axis].max()) - patch + 1)
        high = min(int(points[:, axis].min()), dim - patch)
        if low > high:
            return None
        starts.append(int(rng.integers(low, high + 1)))
    return starts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--source-10k", type=Path, required=True)
    parser.add_argument("--extended-20k", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--iso2mm-output", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--prefix-events", type=int, default=10000)
    parser.add_argument("--continuation-events", type=int, default=10000)
    args = parser.parse_args()

    source_lines, source_events = read_jsonl(args.source_10k)
    extended_lines, extended_events = read_jsonl(args.extended_20k)
    if len(source_events) != args.prefix_events:
        raise ValueError(f"Source schedule has {len(source_events)} events, expected {args.prefix_events}")
    expected_extended = args.prefix_events + args.continuation_events
    if len(extended_events) != expected_extended:
        raise ValueError(f"Extended schedule has {len(extended_events)} events, expected {expected_extended}")
    mismatch = next(
        (index for index in range(args.prefix_events) if source_lines[index] != extended_lines[index]),
        None,
    )
    if mismatch is not None:
        raise RuntimeError(f"Extended schedule does not reproduce source at event {mismatch}")

    continuation = extended_events[args.prefix_events:]
    native_events = []
    for new_index, source in enumerate(continuation):
        event = dict(source)
        event["source_event_index"] = int(source["event_index"])
        event["event_index"] = new_index
        event["continuation_phase"] = "exp004_events_10000_19999"
        native_events.append(event)

    iso_events = []
    stats: Counter = Counter()
    for native in tqdm(native_events, desc="Recomputing 2 mm positive crops"):
        metadata_path = cache_case_paths(args.cache_root, native["case_name"])["metadata"]
        metadata = json.loads(metadata_path.read_text())
        slots = [
            PromptSlot(
                prompt=str(slot["prompt"]),
                target_index=None if slot.get("target_index") is None else int(slot["target_index"]),
                is_positive=bool(slot["is_positive"]),
            )
            for slot in native["prompt_slots"]
        ]
        event_rng = np.random.default_rng(int(native["patch_seed"]))
        spatial_shape = tuple(int(value) for value in metadata["resampled_shape_zyx"])
        candidate_lists = [
            np.asarray(
                metadata["resampled_target_positive_point_candidates"][int(slot.target_index)],
                dtype=np.int64,
            )
            for slot in slots
            if slot.is_positive and slot.target_index is not None
        ]
        starts = None
        chosen_points = None
        if len(candidate_lists) == 1:
            for point in candidate_lists[0][event_rng.permutation(len(candidate_lists[0]))]:
                starts = starts_containing_points(point[None], spatial_shape, (192, 192, 192), event_rng)
                if starts is not None:
                    chosen_points = point[None]
                    break
        elif len(candidate_lists) == 2:
            lefts = candidate_lists[0][event_rng.permutation(len(candidate_lists[0]))]
            rights = candidate_lists[1][event_rng.permutation(len(candidate_lists[1]))]
            for left in lefts:
                for right in rights:
                    points = np.stack([left, right])
                    starts = starts_containing_points(points, spatial_shape, (192, 192, 192), event_rng)
                    if starts is not None:
                        chosen_points = points
                        break
                if starts is not None:
                    break
        if starts is None:
            raise RuntimeError(
                f"{native['case_name']}: event {native['source_event_index']} has no shared "
                "positive 192^3 crop after 2 mm resampling"
            )
        iso = dict(native)
        event_stats: Counter = Counter(positive_crop_requested=1, positive_crop_success=1)
        iso["spatial_shape"] = [int(value) for value in spatial_shape]
        iso["patch_starts"] = [int(value) for value in starts]
        iso["positive_crop_points"] = [
            {
                "target_index": int(slot.target_index),
                "point": [int(value) for value in point],
            }
            for slot, point in zip(
                [slot for slot in slots if slot.is_positive and slot.target_index is not None],
                chosen_points,
            )
        ]
        iso["preprocessing_space"] = "crop_zscore_then_2mm_isotropic_FZYX"
        iso["stats"] = dict(Counter(native.get("stats", {})) + event_stats)
        iso_events.append(iso)
        stats.update(event_stats)

    if signature_hash(native_events) != signature_hash(iso_events):
        raise RuntimeError("Paired schedule signature mismatch")
    write_jsonl(args.native_output, native_events)
    write_jsonl(args.iso2mm_output, iso_events)
    manifest = {
        "created_at_utc": utc_now_iso(),
        "command": command_string(),
        "selection_seed": args.seed,
        "source_10k": str(args.source_10k),
        "source_10k_sha256": sha256_file(args.source_10k),
        "extended_20k": str(args.extended_20k),
        "extended_20k_sha256": sha256_file(args.extended_20k),
        "prefix_events_verified_exact": args.prefix_events,
        "continuation_source_event_range": [
            args.prefix_events,
            args.prefix_events + args.continuation_events - 1,
        ],
        "continuation_events": args.continuation_events,
        "native_schedule": str(args.native_output),
        "native_schedule_sha256": sha256_file(args.native_output),
        "iso2mm_schedule": str(args.iso2mm_output),
        "iso2mm_schedule_sha256": sha256_file(args.iso2mm_output),
        "paired_case_prompt_patch_seed_sha256": signature_hash(native_events),
        "pairing_rule": (
            "case_name, prompt slots, positive/negative selections, and patch_seed are identical; "
            "native starts are inherited and 2 mm starts are recomputed in cached FZYX geometry"
        ),
        "iso2mm_positive_crop_stats": dict(stats),
    }
    write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
