#!/usr/bin/env python3
"""Prepare Exp012 Run-2 independent replay-ratio schedules and validation subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

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
from prepare_training_schedule import (
    build_negative_prompt_pool,
    starts_containing_points,
)


EXPERIMENT_ID = "012_voxtell_category_specialists_replay50_cont100"
PROFILE_ID = "r02_2d_diffuse_replay_ratio"
DEFAULT_CACHE = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/"
    "crop_zscore_native_v1"
)
DEFAULT_REPLAY_SCHEDULE = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/config/"
    "train_schedule_v123_ddp_bs4_seed20260723_100ep_100steps_gb4.jsonl"
)
DEFAULT_VAL200 = Path("configs/evaluation/rexgroundingct_val200_seed20260723.json")
DEFAULT_VAL80 = Path(
    "side_experiments/sideexp001_validation_probe_design/outputs/"
    "rexgroundingct_val80_sideexp001_validation_probe_design.json"
)
PATCH_SIZE = (192, 192, 192)
EPOCHS = 100
EVENTS_PER_EPOCH = 100
TOTAL_EVENTS = EPOCHS * EVENTS_PER_EPOCH
MASTER_SEED = 20260802

ARM_SPECS: dict[str, dict[str, Any]] = {
    "category_2d_replay50": {
        "categories": ("2d",),
        "target_per_epoch": 50,
        "replay_per_epoch": 50,
        "replay_range": (25_000, 30_000),
        "target_seed": 20261802,
        "placement_seed": 20262802,
    },
    "category_1alldiffuse_replay25": {
        "categories": ("1a", "1b", "1c", "1d", "1e", "1f"),
        "target_per_epoch": 75,
        "replay_per_epoch": 25,
        "replay_range": (30_000, 32_500),
        "target_seed": 20261803,
        "placement_seed": 20262803,
    },
    "category_1alldiffuse_replay10": {
        "categories": ("1a", "1b", "1c", "1d", "1e", "1f"),
        "target_per_epoch": 90,
        "replay_per_epoch": 10,
        "replay_range": (32_500, 33_500),
        "target_seed": 20261804,
        "placement_seed": 20262804,
    },
    "category_1alldiffuse_replay00": {
        "categories": ("1a", "1b", "1c", "1d", "1e", "1f"),
        "target_per_epoch": 100,
        "replay_per_epoch": 0,
        "replay_range": None,
        "target_seed": 20261805,
        "placement_seed": 20262805,
    },
}
ARM_TARGETS: dict[str, tuple[str, ...]] = {
    arm: tuple(spec["categories"]) for arm, spec in ARM_SPECS.items()
}
DIFFUSE_TARGET_COUNTS_BY_ARM = {
    "category_1alldiffuse_replay25": {
        "1a": 1186,
        "1b": 1296,
        "1c": 1630,
        "1d": 1075,
        "1e": 1368,
        "1f": 945,
    },
    "category_1alldiffuse_replay10": {
        "1a": 1423,
        "1b": 1556,
        "1c": 1956,
        "1d": 1290,
        "1e": 1641,
        "1f": 1134,
    },
    "category_1alldiffuse_replay00": {
        "1a": 1581,
        "1b": 1728,
        "1c": 2173,
        "1d": 1433,
        "1e": 1824,
        "1f": 1261,
    },
}
EXPECTED_TRAIN_FINDINGS = {
    "1a": 236,
    "1b": 282,
    "1c": 446,
    "1d": 194,
    "1e": 314,
    "1f": 150,
    "2d": 1743,
}
EVALUATION_NAMES = {
    "category_2d_replay50": {
        "target": "rexgroundingct_val119_category_2d_target_census.json",
        "union": "rexgroundingct_val160_category_2d_milestone_union.json",
    },
    "category_1alldiffuse_replay25": {
        "target": "rexgroundingct_val42_category_1alldiffuse_target_census.json",
        "union": "rexgroundingct_val80_category_1alldiffuse_milestone_union.json",
    },
    "category_1alldiffuse_replay10": {
        "target": "rexgroundingct_val42_category_1alldiffuse_target_census.json",
        "union": "rexgroundingct_val80_category_1alldiffuse_milestone_union.json",
    },
    "category_1alldiffuse_replay00": {
        "target": "rexgroundingct_val42_category_1alldiffuse_target_census.json",
        "union": "rexgroundingct_val80_category_1alldiffuse_milestone_union.json",
    },
}


def entry_categories(entry: dict[str, Any]) -> dict[int, str]:
    return {
        int(index): str(category)
        for index, category in entry.get("categories", {}).items()
    }


def evaluator_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": entry["name"],
        "seg_path": entry.get("seg_path", entry["name"]),
        "findings": entry.get("findings", {}),
        "categories": entry.get("categories", {}),
    }


def load_target_options(
    entries: list[dict[str, Any]], cache_root: Path
) -> dict[str, dict[str, Any]]:
    """Load target points for every finding, including unpairable multi-finding cases."""
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = str(entry["name"])
        key = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem
        root = cache_root / "cases" / key
        paths = {
            "root": root,
            "metadata": root / "metadata.json",
            "complete": root / ".complete",
        }
        if not paths["complete"].is_file() or not paths["metadata"].is_file():
            raise FileNotFoundError(f"Incomplete cached case: {paths['root']}")
        metadata = read_json(paths["metadata"])
        shape = metadata.get("resampled_shape_zyx")
        candidates = metadata.get("resampled_target_positive_point_candidates")
        voxels = metadata.get("resampled_target_voxels") or metadata.get(
            "source_target_voxels"
        )
        prompts = sorted_prompts(entry)
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f"{entry['name']}: invalid cached spatial shape")
        if not isinstance(candidates, list) or len(candidates) < len(prompts):
            raise ValueError(
                f"{entry['name']}: missing target positive-point candidates"
            )
        if not isinstance(voxels, list) or len(voxels) < len(prompts):
            raise ValueError(f"{entry['name']}: missing target voxel counts")

        points: dict[int, list[int]] = {}
        for index in range(len(prompts)):
            if int(voxels[index]) <= 0 or not candidates[index]:
                continue
            points[index] = [int(value) for value in candidates[index][0]]

        compatible: dict[int, list[int]] = {index: [] for index in points}
        point_indices = sorted(points)
        for left_offset, left in enumerate(point_indices):
            for right in point_indices[left_offset + 1 :]:
                pair_points = np.asarray([points[left], points[right]], dtype=np.int64)
                if (
                    starts_containing_points(
                        pair_points, tuple(shape), PATCH_SIZE, rng=None
                    )
                    is not None
                ):
                    compatible[left].append(right)
                    compatible[right].append(left)
        result[entry["name"]] = {
            "spatial_shape": [int(value) for value in shape],
            "points": points,
            "compatible": compatible,
        }
    return result


def target_finding_pools(
    entries: list[dict[str, Any]], options: dict[str, dict[str, Any]]
) -> dict[str, list[tuple[dict[str, Any], int]]]:
    pools: dict[str, list[tuple[dict[str, Any], int]]] = {
        category: [] for category in EXPECTED_TRAIN_FINDINGS
    }
    for entry in entries:
        available = options[entry["name"]]["points"]
        for target_index, category in entry_categories(entry).items():
            if category in pools and target_index in available:
                pools[category].append((entry, target_index))
    observed = {category: len(values) for category, values in pools.items()}
    if observed != EXPECTED_TRAIN_FINDINGS:
        raise ValueError(
            "Eligible target finding counts differ from the audited train split: "
            f"expected={EXPECTED_TRAIN_FINDINGS} observed={observed}"
        )
    return pools


def choose_negative_prompts(
    entry: dict[str, Any],
    negative_pool: list[str],
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    case_prompts = {prompt.lower() for prompt in sorted_prompts(entry)}
    candidates = [prompt for prompt in negative_pool if prompt not in case_prompts]
    if not candidates:
        candidates = negative_pool
    return [
        str(value)
        for value in rng.choice(
            candidates, size=count, replace=len(candidates) < count
        ).tolist()
    ]


def make_target_event(
    entry: dict[str, Any],
    target_index: int,
    target_category: str,
    options: dict[str, Any],
    negative_pool: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    prompts = sorted_prompts(entry)
    points: dict[int, list[int]] = options["points"]
    compatible: dict[int, list[int]] = options["compatible"]
    if target_index not in points:
        raise ValueError(
            f"{entry['name']}: target {target_index} has no positive point"
        )

    partners = compatible.get(target_index, [])
    if partners:
        partner = int(partners[int(rng.integers(0, len(partners)))])
        positive_indices = [target_index, partner]
        fallback = False
    else:
        positive_indices = [target_index]
        fallback = len(prompts) > 1

    slots: list[dict[str, Any]] = [
        {"prompt": prompts[index], "target_index": index, "is_positive": True}
        for index in positive_indices
    ]
    slots.extend(
        {"prompt": prompt, "target_index": None, "is_positive": False}
        for prompt in choose_negative_prompts(entry, negative_pool, 3 - len(slots), rng)
    )
    order = rng.permutation(len(slots))
    slots = [slots[int(index)] for index in order]

    anchor_slot_index = next(
        index
        for index, slot in enumerate(slots)
        if slot.get("target_index") == target_index
    )
    ordered_points: list[list[int]] = []
    positive_crop_points: list[dict[str, Any]] = []
    for slot_index, slot in enumerate(slots):
        if not slot["is_positive"]:
            continue
        point = points[int(slot["target_index"])]
        ordered_points.append(point)
        positive_crop_points.append(
            {
                "slot_index": slot_index,
                "target_index": int(slot["target_index"]),
                "point": point,
            }
        )
    starts = starts_containing_points(
        np.asarray(ordered_points, dtype=np.int64),
        tuple(int(value) for value in options["spatial_shape"]),
        PATCH_SIZE,
        rng,
    )
    if starts is None:
        raise ValueError(
            f"{entry['name']}: selected targeted positives cannot share a crop"
        )

    stats = Counter(
        samples=1,
        targeted_events=1,
        foreground_oversample_requested=1,
        positive_crop_forced_foreground_anchor=1,
        positive_crop_success=1,
        positive_slots=len(positive_indices),
        negative_slots=3 - len(positive_indices),
    )
    if fallback:
        stats["targeted_multifinding_single_positive_fallback"] += 1
    elif len(positive_indices) == 2:
        stats["targeted_compatible_pair_samples"] += 1

    return {
        "schema_version": 2,
        "event_index": -1,
        "case_name": entry["name"],
        "source_split": entry.get("_split", "train"),
        "source_split_index": entry.get("_index"),
        "patch_size": list(PATCH_SIZE),
        "patch_seed": int(rng.integers(0, 2**63 - 1)),
        "patch_starts": [int(value) for value in starts],
        "spatial_shape": options["spatial_shape"],
        "prompt_slots": slots,
        "positive_crop_points": positive_crop_points,
        "foreground_requested": True,
        "foreground_forced": True,
        "require_positive_crop": True,
        "anchor_slot_index": anchor_slot_index,
        "anchor_target_index": target_index,
        "sampling_source": "targeted",
        "requested_target_category": target_category,
        "requested_target_index": target_index,
        "stats": dict(stats),
    }


def read_replay_slice(path: Path, start: int, stop: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_index, line in enumerate(handle):
            if line_index >= stop:
                break
            if line_index < start:
                continue
            event = json.loads(line)
            if int(event.get("event_index", -1)) != line_index:
                raise ValueError(
                    f"Replay schedule event index mismatch at line {line_index}"
                )
            event["replay_source_event_index"] = line_index
            event["sampling_source"] = "replay"
            selected.append(event)
    if len(selected) != stop - start:
        raise ValueError(
            f"Expected {stop - start} replay events, found {len(selected)}"
        )
    return selected


def target_category_sequence(arm: str, rng: np.random.Generator) -> list[str]:
    target_events = EPOCHS * int(ARM_SPECS[arm]["target_per_epoch"])
    if arm in DIFFUSE_TARGET_COUNTS_BY_ARM:
        allocation = DIFFUSE_TARGET_COUNTS_BY_ARM[arm]
        sequence = [
            category for category, count in allocation.items() for _ in range(count)
        ]
    else:
        category = ARM_TARGETS[arm][0]
        sequence = [category] * target_events
    if len(sequence) != target_events:
        raise ValueError(
            f"{arm}: expected {target_events} target categories, got {len(sequence)}"
        )
    rng.shuffle(sequence)
    return sequence


def replay_fingerprint(events: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for event in events:
        payload = {
            "source_event_index": event["replay_source_event_index"],
            "case_name": event["case_name"],
            "prompt_slots": event["prompt_slots"],
            "patch_starts": event.get("patch_starts"),
        }
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def audit_schedule(events: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    if len(events) != TOTAL_EVENTS:
        raise ValueError(f"{arm}: schedule has {len(events)} events")
    target_counts: Counter = Counter()
    per_epoch: list[dict[str, int]] = []
    replay_events: list[dict[str, Any]] = []
    anchor_failures: list[int] = []
    for index, event in enumerate(events):
        if int(event["event_index"]) != index:
            raise ValueError(f"{arm}: nonsequential event index at {index}")
        if event["sampling_source"] == "replay":
            replay_events.append(event)
        else:
            category = str(event["requested_target_category"])
            target_counts[category] += 1
            target_index = int(event["requested_target_index"])
            target_slots = [
                slot
                for slot in event["prompt_slots"]
                if slot["is_positive"] and slot.get("target_index") == target_index
            ]
            crop_targets = {
                int(item["target_index"])
                for item in event.get("positive_crop_points", [])
            }
            if (
                len(target_slots) != 1
                or int(event.get("anchor_target_index", -1)) != target_index
                or target_index not in crop_targets
            ):
                anchor_failures.append(index)
    spec = ARM_SPECS[arm]
    target_per_epoch = int(spec["target_per_epoch"])
    replay_per_epoch = int(spec["replay_per_epoch"])
    target_events = EPOCHS * target_per_epoch
    replay_count = EPOCHS * replay_per_epoch
    for epoch in range(EPOCHS):
        block = events[epoch * EVENTS_PER_EPOCH : (epoch + 1) * EVENTS_PER_EPOCH]
        counts = Counter(event["sampling_source"] for event in block)
        observed = {"targeted": counts["targeted"], "replay": counts["replay"]}
        if observed != {"targeted": target_per_epoch, "replay": replay_per_epoch}:
            raise ValueError(f"{arm}: epoch {epoch + 1} source counts are {observed}")
        per_epoch.append(observed)
    if anchor_failures:
        raise ValueError(
            f"{arm}: {len(anchor_failures)} targeted anchor audit failures"
        )
    expected_categories = (
        DIFFUSE_TARGET_COUNTS_BY_ARM[arm]
        if arm in DIFFUSE_TARGET_COUNTS_BY_ARM
        else {ARM_TARGETS[arm][0]: target_events}
    )
    if dict(target_counts) != expected_categories:
        raise ValueError(
            f"{arm}: target counts differ: expected={expected_categories} observed={dict(target_counts)}"
        )
    return {
        "events": len(events),
        "sampling_source_counts": {"targeted": target_events, "replay": replay_count},
        "target_category_counts": dict(target_counts),
        "target_anchor_failures": 0,
        "per_epoch": per_epoch,
        "replay_fingerprint": replay_fingerprint(replay_events),
    }


def prepare_schedules(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != MASTER_SEED:
        raise ValueError(f"{PROFILE_ID} requires master seed {MASTER_SEED}")
    train_entries = load_split_entries(args.metadata, ["train"])
    negative_pool = build_negative_prompt_pool(train_entries)
    options = load_target_options(train_entries, args.preprocessed_cache_dir)
    pools = target_finding_pools(train_entries, options)
    args.schedule_output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Any] = {}
    replay_ranges: list[tuple[int, int]] = []
    for arm, spec in ARM_SPECS.items():
        target_rng = np.random.default_rng(int(spec["target_seed"]))
        placement_rng = np.random.default_rng(int(spec["placement_seed"]))
        target_per_epoch = int(spec["target_per_epoch"])
        replay_per_epoch = int(spec["replay_per_epoch"])
        replay_range = spec["replay_range"]
        if replay_range is None:
            replay: list[dict[str, Any]] = []
        else:
            start, stop = (int(value) for value in replay_range)
            if stop - start != EPOCHS * replay_per_epoch:
                raise ValueError(f"{arm}: replay range does not match requested ratio")
            replay = read_replay_slice(args.replay_schedule, start, stop)
            replay_ranges.append((start, stop))
        target_positions_by_epoch = [
            set(
                int(value)
                for value in placement_rng.permutation(EVENTS_PER_EPOCH)[
                    :target_per_epoch
                ]
            )
            for _ in range(EPOCHS)
        ]
        categories = target_category_sequence(arm, target_rng)
        targeted: list[dict[str, Any]] = []
        for category in categories:
            pool = pools[category]
            entry, target_index = pool[int(target_rng.integers(0, len(pool)))]
            targeted.append(
                make_target_event(
                    entry,
                    target_index,
                    category,
                    options[entry["name"]],
                    negative_pool,
                    target_rng,
                )
            )

        mixed: list[dict[str, Any]] = []
        target_cursor = 0
        replay_cursor = 0
        for epoch in range(EPOCHS):
            positions = target_positions_by_epoch[epoch]
            for position in range(EVENTS_PER_EPOCH):
                if position in positions:
                    event = dict(targeted[target_cursor])
                    target_cursor += 1
                else:
                    event = dict(replay[replay_cursor])
                    replay_cursor += 1
                event["event_index"] = len(mixed)
                mixed.append(event)
        if (
            target_cursor != EPOCHS * target_per_epoch
            or replay_cursor != EPOCHS * replay_per_epoch
        ):
            raise AssertionError("Exp012 schedule cursors did not consume every event")

        schedule_path = args.schedule_output_dir / f"{arm}.jsonl"
        with schedule_path.open("w") as handle:
            for event in mixed:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        audit = audit_schedule(mixed, arm)
        manifest = {
            "created_at_utc": utc_now_iso(),
            "experiment": EXPERIMENT_ID,
            "profile": PROFILE_ID,
            "arm": arm,
            "target_categories": list(ARM_TARGETS[arm]),
            "metadata": str(args.metadata),
            "metadata_sha256": sha256_file(args.metadata),
            "preprocessed_cache_dir": str(args.preprocessed_cache_dir),
            "source_replay_schedule": str(args.replay_schedule),
            "source_replay_schedule_sha256": sha256_file(args.replay_schedule),
            "source_replay_event_range": list(replay_range) if replay_range else None,
            "seed": args.seed,
            "target_seed": int(spec["target_seed"]),
            "placement_seed": int(spec["placement_seed"]),
            "target_events_per_epoch": target_per_epoch,
            "replay_events_per_epoch": replay_per_epoch,
            "schedule_schema_version": 2,
            "patch_size": list(PATCH_SIZE),
            "output_jsonl": str(schedule_path),
            "output_jsonl_sha256": sha256_file(schedule_path),
            "audit": audit,
        }
        manifest_path = schedule_path.with_suffix(".manifest.json")
        write_json(manifest_path, manifest)
        manifests[arm] = manifest

    combined = {
        "created_at_utc": utc_now_iso(),
        "experiment": EXPERIMENT_ID,
        "profile": PROFILE_ID,
        "command": command_string(),
        "arms": manifests,
        "independent_schedules": True,
        "replay_ranges_disjoint": all(
            left_stop <= right_start or right_stop <= left_start
            for index, (left_start, left_stop) in enumerate(replay_ranges)
            for right_start, right_stop in replay_ranges[index + 1 :]
        ),
    }
    if not combined["replay_ranges_disjoint"]:
        raise ValueError("Run-2 replay source ranges overlap")
    write_json(
        args.schedule_output_dir / "exp012_r02_schedules.manifest.json", combined
    )
    return combined


def category_count(entries: list[dict[str, Any]], categories: set[str]) -> int:
    return sum(
        1
        for entry in entries
        for category in entry.get("categories", {}).values()
        if category in categories
    )


def prepare_evaluation_subsets(args: argparse.Namespace) -> dict[str, Any]:
    val200 = read_json(args.val200_json)["test"]
    val80 = read_json(args.val80_json)["test"]
    val80_names = {entry["name"] for entry in val80}
    val200_names = {entry["name"] for entry in val200}
    if not val80_names <= val200_names:
        raise ValueError("Accepted val80 is not a subset of fixed val200")
    train_names = {
        entry["name"] for entry in load_split_entries(args.metadata, ["train"])
    }
    if train_names & val200_names:
        raise ValueError("Train/validation case-name leakage detected")

    args.evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for arm, target_tuple in ARM_TARGETS.items():
        target_categories = set(target_tuple)
        target_entries = [
            entry
            for entry in val200
            if target_categories & set(entry.get("categories", {}).values())
        ]
        target_names = {entry["name"] for entry in target_entries}
        union_names = target_names | val80_names
        union_entries = [entry for entry in val200 if entry["name"] in union_names]
        target_path = args.evaluation_output_dir / EVALUATION_NAMES[arm]["target"]
        union_path = args.evaluation_output_dir / EVALUATION_NAMES[arm]["union"]
        write_json(
            target_path, {"test": [evaluator_entry(entry) for entry in target_entries]}
        )
        write_json(
            union_path, {"test": [evaluator_entry(entry) for entry in union_entries]}
        )
        target_findings = category_count(target_entries, target_categories)
        record = {
            "target_categories": list(target_tuple),
            "target": {
                "path": str(target_path),
                "sha256": sha256_file(target_path),
                "cases": len(target_entries),
                "all_findings": sum(
                    len(entry.get("findings", {})) for entry in target_entries
                ),
                "target_findings": target_findings,
            },
            "union": {
                "path": str(union_path),
                "sha256": sha256_file(union_path),
                "cases": len(union_entries),
                "all_findings": sum(
                    len(entry.get("findings", {})) for entry in union_entries
                ),
                "target_findings": target_findings,
                "val80_cases": len(val80_names),
            },
        }
        records[arm] = record

    expected = {
        "category_2d_replay50": (119, 132, 160),
        "category_1alldiffuse_replay25": (42, 52, 80),
        "category_1alldiffuse_replay10": (42, 52, 80),
        "category_1alldiffuse_replay00": (42, 52, 80),
    }
    for arm, (cases, findings, union_cases) in expected.items():
        record = records[arm]
        observed = (
            record["target"]["cases"],
            record["target"]["target_findings"],
            record["union"]["cases"],
        )
        if observed != (cases, findings, union_cases):
            raise ValueError(
                f"{arm}: expected {(cases, findings, union_cases)}, observed {observed}"
            )

    manifest = {
        "created_at_utc": utc_now_iso(),
        "experiment": EXPERIMENT_ID,
        "profile": PROFILE_ID,
        "command": command_string(),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "val200_json": str(args.val200_json),
        "val200_json_sha256": sha256_file(args.val200_json),
        "val80_json": str(args.val80_json),
        "val80_json_sha256": sha256_file(args.val80_json),
        "val80_cases": len(val80_names),
        "zero_train_validation_case_overlap": True,
        "arms": records,
    }
    manifest_path = (
        args.evaluation_output_dir
        / "rexgroundingct_exp012_r02_evaluation_subsets.manifest.json"
    )
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["all", "schedules", "evaluation"], default="all"
    )
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--preprocessed-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--replay-schedule", type=Path, default=DEFAULT_REPLAY_SCHEDULE)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument(
        "--schedule-output-dir", type=Path, default=Path("exp012_r02_schedules")
    )
    parser.add_argument("--val200-json", type=Path, default=DEFAULT_VAL200)
    parser.add_argument("--val80-json", type=Path, default=DEFAULT_VAL80)
    parser.add_argument(
        "--evaluation-output-dir", type=Path, default=Path("configs/evaluation")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode in {"all", "schedules"}:
        result = prepare_schedules(args)
        print(
            f"Prepared {len(result['arms'])} Exp012 {PROFILE_ID} schedules under "
            f"{args.schedule_output_dir}"
        )
    if args.mode in {"all", "evaluation"}:
        result = prepare_evaluation_subsets(args)
        print(
            f"Prepared {len(result['arms'])} Exp012 {PROFILE_ID} evaluation subset pairs under "
            f"{args.evaluation_output_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
