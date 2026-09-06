#!/usr/bin/env python3
"""Prepare the deterministic Exp021 nodule-only-positive DDP schedule."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common import (
    REX_METADATA,
    command_string,
    load_split_entries,
    read_json,
    sha256_file,
    sorted_prompts,
    utc_now_iso,
    write_json,
)
from prepare_training_schedule import starts_containing_points


EXPERIMENT_ID = "021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100"
TARGET_CATEGORY = "2d"
PATCH_SIZE = (192, 192, 192)
EXPECTED_TRAIN_CASES = 2992
EXPECTED_NODULE_CASES = 1305
EXPECTED_NODULE_FINDINGS = 1743
EXPECTED_NON_NODULE_CASES = 1687


def cache_case_root(cache_root: Path, name: str) -> Path:
    if not name.endswith(".nii.gz"):
        raise ValueError(f"Expected .nii.gz case name, got {name!r}")
    return cache_root / "cases" / name[: -len(".nii.gz")]


def entry_categories(entry: dict[str, Any]) -> dict[int, str]:
    return {
        int(index): str(category)
        for index, category in entry.get("categories", {}).items()
    }


def load_case_geometry(entry: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    root = cache_case_root(cache_root, str(entry["name"]))
    metadata_path = root / "metadata.json"
    if not (root / ".complete").is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete cached case: {root}")
    metadata = read_json(metadata_path)
    shape = metadata.get("resampled_shape_zyx")
    candidates = metadata.get("resampled_target_positive_point_candidates")
    voxels = metadata.get("resampled_target_voxels") or metadata.get("source_target_voxels")
    prompts = sorted_prompts(entry)
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError(f"{entry['name']}: invalid resampled_shape_zyx")
    if not isinstance(candidates, list) or len(candidates) < len(prompts):
        raise ValueError(f"{entry['name']}: missing positive-point candidates")
    if not isinstance(voxels, list) or len(voxels) < len(prompts):
        raise ValueError(f"{entry['name']}: missing target voxel counts")

    categories = entry_categories(entry)
    points: dict[int, list[int]] = {}
    for index, category in categories.items():
        if category != TARGET_CATEGORY:
            continue
        if int(voxels[index]) <= 0 or not candidates[index]:
            raise ValueError(f"{entry['name']}: empty category-2d target {index}")
        points[index] = [int(value) for value in candidates[index][0]]

    non_target_prompts = sorted(
        {
            prompt.lower()
            for index, prompt in enumerate(prompts)
            if categories.get(index) != TARGET_CATEGORY
        }
    )
    return {
        "spatial_shape": [int(value) for value in shape],
        "points": points,
        "non_target_prompts": non_target_prompts,
    }


def random_patch_starts(
    spatial_shape: Iterable[int],
    patch_size: tuple[int, int, int],
    rng: np.random.Generator,
) -> list[int]:
    starts: list[int] = []
    for dimension, patch in zip(spatial_shape, patch_size):
        max_start = max(0, int(dimension) - int(patch))
        starts.append(int(rng.integers(0, max_start + 1)))
    return starts


def choose_negative_prompts(
    preferred: Iterable[str],
    fallback: list[str],
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    candidates = sorted(set(preferred)) or fallback
    if not candidates:
        raise ValueError("No negative prompt candidates are available")
    return [
        str(value)
        for value in rng.choice(candidates, size=count, replace=len(candidates) < count).tolist()
    ]


def make_positive_event(
    event_index: int,
    entry: dict[str, Any],
    geometry: dict[str, Any],
    target_index: int,
    global_non_target_prompts: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    if target_index not in geometry["points"]:
        raise ValueError(f"{entry['name']}: requested category-2d target {target_index} is unavailable")
    prompts = sorted_prompts(entry)
    negative_prompts = choose_negative_prompts(
        geometry["non_target_prompts"], global_non_target_prompts, 2, rng
    )
    slots = [
        {
            "prompt": prompts[target_index],
            "target_index": target_index,
            "is_positive": True,
        }
    ] + [
        {"prompt": prompt, "target_index": None, "is_positive": False}
        for prompt in negative_prompts
    ]
    order = rng.permutation(3).tolist()
    slots = [slots[int(index)] for index in order]
    anchor_slot_index = next(
        index for index, slot in enumerate(slots) if slot["target_index"] == target_index
    )
    point = geometry["points"][target_index]
    starts = starts_containing_points(
        np.asarray([point], dtype=np.int64),
        tuple(int(value) for value in geometry["spatial_shape"]),
        PATCH_SIZE,
        rng,
    )
    if starts is None:
        raise ValueError(f"{entry['name']}: category-2d target cannot fit in patch")
    return {
        "schema_version": 2,
        "event_index": event_index,
        "case_name": entry["name"],
        "source_split": entry.get("_split", "train"),
        "source_split_index": entry.get("_index"),
        "patch_size": list(PATCH_SIZE),
        "patch_seed": int(rng.integers(0, 2**63 - 1)),
        "patch_starts": [int(value) for value in starts],
        "spatial_shape": geometry["spatial_shape"],
        "prompt_slots": slots,
        "positive_crop_points": [
            {
                "slot_index": anchor_slot_index,
                "target_index": target_index,
                "point": [int(value) for value in point],
            }
        ],
        "foreground_requested": True,
        "foreground_forced": True,
        "require_positive_crop": True,
        "anchor_slot_index": anchor_slot_index,
        "anchor_target_index": target_index,
        "sampling_source": "category_2d_positive",
        "requested_target_category": TARGET_CATEGORY,
        "requested_target_index": target_index,
        "stats": {
            "samples": 1,
            "category_2d_positive_events": 1,
            "positive_slots": 1,
            "negative_slots": 2,
            "positive_crop_success": 1,
        },
    }


def make_negative_event(
    event_index: int,
    entry: dict[str, Any],
    geometry: dict[str, Any],
    nodule_prompt: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    other_prompts = choose_negative_prompts(
        geometry["non_target_prompts"], [nodule_prompt], 2, rng
    )
    prompts = [nodule_prompt, *other_prompts]
    rng.shuffle(prompts)
    starts = random_patch_starts(geometry["spatial_shape"], PATCH_SIZE, rng)
    slots = [
        {"prompt": prompt, "target_index": None, "is_positive": False}
        for prompt in prompts
    ]
    return {
        "schema_version": 2,
        "event_index": event_index,
        "case_name": entry["name"],
        "source_split": entry.get("_split", "train"),
        "source_split_index": entry.get("_index"),
        "patch_size": list(PATCH_SIZE),
        "patch_seed": int(rng.integers(0, 2**63 - 1)),
        "patch_starts": starts,
        "spatial_shape": geometry["spatial_shape"],
        "prompt_slots": slots,
        "foreground_requested": False,
        "foreground_forced": False,
        "require_positive_crop": False,
        "anchor_slot_index": None,
        "anchor_target_index": None,
        "sampling_source": "nodule_negative_case",
        "requested_target_category": None,
        "requested_target_index": None,
        "stats": {
            "samples": 1,
            "nodule_negative_case_events": 1,
            "positive_slots": 0,
            "negative_slots": 3,
        },
    }


def cycling_choices(
    pool: list[Any], count: int, rng: np.random.Generator
) -> list[Any]:
    if not pool:
        raise ValueError("Cannot cycle an empty pool")
    order = rng.permutation(len(pool)).tolist()
    cursor = 0
    result: list[Any] = []
    for _ in range(count):
        if cursor == len(order):
            order = rng.permutation(len(pool)).tolist()
            cursor = 0
        result.append(pool[int(order[cursor])])
        cursor += 1
    return result


def audit_schedule(
    events: list[dict[str, Any]],
    positive_events_per_block: int,
    events_per_block: int,
    nodule_case_names: set[str],
) -> dict[str, Any]:
    if len(events) % events_per_block:
        raise ValueError("Schedule length must be divisible by one DDP epoch block")
    counts: Counter[str] = Counter()
    target_counts: Counter[int] = Counter()
    case_counts: Counter[str] = Counter()
    for index, event in enumerate(events):
        if int(event.get("event_index", -1)) != index:
            raise ValueError(f"Event index mismatch at line {index}")
        slots = event.get("prompt_slots", [])
        if len(slots) != 3:
            raise ValueError(f"Event {index} does not contain exactly three slots")
        source = str(event.get("sampling_source"))
        counts[source] += 1
        case_counts[str(event["case_name"])] += 1
        positives = [slot for slot in slots if slot.get("is_positive")]
        if source == "category_2d_positive":
            if len(positives) != 1 or event.get("requested_target_category") != TARGET_CATEGORY:
                raise ValueError(f"Event {index} has invalid nodule-positive slots")
            target_index = int(event["requested_target_index"])
            if positives[0].get("target_index") != target_index:
                raise ValueError(f"Event {index} positive slot does not match requested target")
            points = event.get("positive_crop_points", [])
            if len(points) != 1 or int(points[0]["target_index"]) != target_index:
                raise ValueError(f"Event {index} lacks its positive crop anchor")
            start = event["patch_starts"]
            point = points[0]["point"]
            if any(not (int(start[a]) <= int(point[a]) < int(start[a]) + PATCH_SIZE[a]) for a in range(3)):
                raise ValueError(f"Event {index} positive point is outside patch")
            target_counts[target_index] += 1
        elif source == "nodule_negative_case":
            if str(event["case_name"]) not in nodule_case_names:
                raise ValueError(f"Event {index} negative case has a category-2d annotation")
            if positives or any(slot.get("target_index") is not None for slot in slots):
                raise ValueError(f"Event {index} negative event has a positive target")
        else:
            raise ValueError(f"Unknown sampling source {source!r} at event {index}")

    block_counts: list[dict[str, int]] = []
    for start in range(0, len(events), events_per_block):
        block = Counter(event["sampling_source"] for event in events[start : start + events_per_block])
        observed = {
            "category_2d_positive": int(block["category_2d_positive"]),
            "nodule_negative_case": int(block["nodule_negative_case"]),
        }
        expected = {
            "category_2d_positive": positive_events_per_block,
            "nodule_negative_case": events_per_block - positive_events_per_block,
        }
        if observed != expected:
            raise ValueError(f"Block {start // events_per_block} counts {observed}, expected {expected}")
        block_counts.append(observed)
    return {
        "events": len(events),
        "sampling_source_counts": dict(counts),
        "unique_positive_targets": len(target_counts),
        "positive_target_event_count_min": min(target_counts.values()),
        "positive_target_event_count_max": max(target_counts.values()),
        "unique_negative_cases": len(
            {name for name in case_counts if name in nodule_case_names}
        ),
        "per_ddp_epoch_block": block_counts,
    }


def build_schedule(args: argparse.Namespace) -> dict[str, Any]:
    entries = load_split_entries(args.metadata, ["train"])
    if len(entries) != EXPECTED_TRAIN_CASES:
        raise ValueError(f"Expected {EXPECTED_TRAIN_CASES} train cases, got {len(entries)}")

    geometries = {
        str(entry["name"]): load_case_geometry(entry, args.preprocessed_cache_dir)
        for entry in entries
    }
    positive_targets: list[tuple[dict[str, Any], int]] = []
    positive_case_names: set[str] = set()
    negative_entries: list[dict[str, Any]] = []
    non_target_prompts: set[str] = set()
    nodule_prompts: set[str] = set()
    for entry in entries:
        name = str(entry["name"])
        geometry = geometries[name]
        categories = entry_categories(entry)
        for index, category in categories.items():
            if category == TARGET_CATEGORY:
                positive_targets.append((entry, index))
                positive_case_names.add(name)
                nodule_prompts.add(sorted_prompts(entry)[index].lower())
            else:
                non_target_prompts.add(sorted_prompts(entry)[index].lower())
        if not geometry["points"]:
            negative_entries.append(entry)

    if len(positive_case_names) != EXPECTED_NODULE_CASES:
        raise ValueError(f"Expected {EXPECTED_NODULE_CASES} nodule-positive cases, got {len(positive_case_names)}")
    if len(positive_targets) != EXPECTED_NODULE_FINDINGS:
        raise ValueError(f"Expected {EXPECTED_NODULE_FINDINGS} nodule findings, got {len(positive_targets)}")
    if len(negative_entries) != EXPECTED_NON_NODULE_CASES:
        raise ValueError(f"Expected {EXPECTED_NON_NODULE_CASES} nodule-negative cases, got {len(negative_entries)}")
    if not nodule_prompts or not non_target_prompts:
        raise ValueError("Prompt pools are unexpectedly empty")

    events_per_block = args.steps_per_epoch * args.world_size * args.batch_size * args.grad_accum
    total_events = args.epochs * events_per_block
    positive_events_per_block = int(args.positive_events_per_block)
    if not 0 < positive_events_per_block < events_per_block:
        raise ValueError("Positive events per block must be between zero and the block size")
    positive_count = args.epochs * positive_events_per_block
    negative_count = total_events - positive_count
    rng = np.random.default_rng(args.seed)
    target_sequence = cycling_choices(positive_targets, positive_count, rng)
    negative_sequence = cycling_choices(negative_entries, negative_count, rng)
    target_cursor = 0
    negative_cursor = 0
    events: list[dict[str, Any]] = []
    for _epoch in range(args.epochs):
        sources = ["category_2d_positive"] * positive_events_per_block
        sources.extend(["nodule_negative_case"] * (events_per_block - positive_events_per_block))
        rng.shuffle(sources)
        for source in sources:
            event_index = len(events)
            if source == "category_2d_positive":
                entry, _target_index = target_sequence[target_cursor]
                target_cursor += 1
                event = make_positive_event(
                    event_index,
                    entry,
                    geometries[str(entry["name"])],
                    _target_index,
                    sorted(non_target_prompts),
                    rng,
                )
            else:
                entry = negative_sequence[negative_cursor]
                negative_cursor += 1
                event = make_negative_event(
                    event_index,
                    entry,
                    geometries[str(entry["name"])],
                    sorted(nodule_prompts)[int(rng.integers(0, len(nodule_prompts)))],
                    rng,
                )
            events.append(event)

    audit = audit_schedule(
        events,
        positive_events_per_block,
        events_per_block,
        {str(entry["name"]) for entry in negative_entries},
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    manifest = {
        "created_at_utc": utc_now_iso(),
        "experiment": EXPERIMENT_ID,
        "command": command_string(),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "preprocessed_cache_dir": str(args.preprocessed_cache_dir),
        "cache_manifest_sha256": sha256_file(args.preprocessed_cache_dir / "manifest.json"),
        "seed": args.seed,
        "schedule_schema_version": 2,
        "patch_size": list(PATCH_SIZE),
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "world_size": args.world_size,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "events_per_ddp_epoch": events_per_block,
        "positive_events_per_ddp_epoch": positive_events_per_block,
        "events": len(events),
        "positive_events": positive_count,
        "negative_events": negative_count,
        "positive_category": TARGET_CATEGORY,
        "positive_case_count": len(positive_case_names),
        "positive_finding_count": len(positive_targets),
        "negative_case_count": len(negative_entries),
        "negative_prompt_pool_size": len(non_target_prompts),
        "nodule_prompt_pool_size": len(nodule_prompts),
        "sampling_policy": "finding-uniform cycling shuffled category-2d target pool plus cycling shuffled no-category-2d case pool",
        "event_consumption_rule": "event_index = update * world_size + rank for local batch=1 and grad_accum=1",
        "audit": audit,
        "output_jsonl": str(args.output_jsonl),
        "output_jsonl_sha256": sha256_file(args.output_jsonl),
    }
    manifest_path = args.manifest_json or args.output_jsonl.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(f"Wrote {len(events)} events to {args.output_jsonl}")
    print(f"Wrote manifest to {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--preprocessed-cache-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--positive-events-per-block", type=int, default=300)
    args = parser.parse_args()
    if args.epochs <= 0 or args.steps_per_epoch <= 0 or args.world_size <= 0:
        raise ValueError("epochs, steps-per-epoch, and world-size must be positive")
    build_schedule(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
