#!/usr/bin/env python3
"""Prepare deterministic Exp012 replay schedules and validation subsets."""

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
from prepare_training_schedule import build_negative_prompt_pool, starts_containing_points


EXPERIMENT_ID = "012_voxtell_category_specialists_replay50_cont100"
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
TARGET_EVENTS_PER_EPOCH = 50
REPLAY_EVENTS_PER_EPOCH = 50
TARGET_EVENTS = EPOCHS * TARGET_EVENTS_PER_EPOCH
REPLAY_EVENTS = EPOCHS * REPLAY_EVENTS_PER_EPOCH
TOTAL_EVENTS = EPOCHS * EVENTS_PER_EPOCH
REPLAY_START = 20_000
REPLAY_STOP = 25_000
MASTER_SEED = 20260801

ARM_TARGETS: dict[str, tuple[str, ...]] = {
    "category_1alldiffuse_replay50": ("1a", "1b", "1c", "1d", "1e", "1f"),
    "category_2a_replay50": ("2a",),
    "category_2b_replay50": ("2b",),
    "category_2c_replay50": ("2c",),
}
DIFFUSE_TARGET_COUNTS = {
    "1a": 791,
    "1b": 864,
    "1c": 1087,
    "1d": 716,
    "1e": 912,
    "1f": 630,
}
EXPECTED_TRAIN_FINDINGS = {
    "1a": 236,
    "1b": 282,
    "1c": 446,
    "1d": 194,
    "1e": 314,
    "1f": 150,
    "2a": 1120,
    "2b": 1367,
    "2c": 1507,
}
EVALUATION_NAMES = {
    "category_1alldiffuse_replay50": {
        "target": "rexgroundingct_val42_category_1alldiffuse_target_census.json",
        "union": "rexgroundingct_val80_category_1alldiffuse_milestone_union.json",
    },
    "category_2a_replay50": {
        "target": "rexgroundingct_val63_category_2a_target_census.json",
        "union": "rexgroundingct_val114_category_2a_milestone_union.json",
    },
    "category_2b_replay50": {
        "target": "rexgroundingct_val40_category_2b_target_census.json",
        "union": "rexgroundingct_val98_category_2b_milestone_union.json",
    },
    "category_2c_replay50": {
        "target": "rexgroundingct_val53_category_2c_target_census.json",
        "union": "rexgroundingct_val110_category_2c_milestone_union.json",
    },
}


def entry_categories(entry: dict[str, Any]) -> dict[int, str]:
    return {int(index): str(category) for index, category in entry.get("categories", {}).items()}


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
        voxels = metadata.get("resampled_target_voxels") or metadata.get("source_target_voxels")
        prompts = sorted_prompts(entry)
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f"{entry['name']}: invalid cached spatial shape")
        if not isinstance(candidates, list) or len(candidates) < len(prompts):
            raise ValueError(f"{entry['name']}: missing target positive-point candidates")
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
                if starts_containing_points(pair_points, tuple(shape), PATCH_SIZE, rng=None) is not None:
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
    entry: dict[str, Any], negative_pool: list[str], count: int, rng: np.random.Generator
) -> list[str]:
    case_prompts = {prompt.lower() for prompt in sorted_prompts(entry)}
    candidates = [prompt for prompt in negative_pool if prompt not in case_prompts]
    if not candidates:
        candidates = negative_pool
    return [
        str(value)
        for value in rng.choice(candidates, size=count, replace=len(candidates) < count).tolist()
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
        raise ValueError(f"{entry['name']}: target {target_index} has no positive point")

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
        index for index, slot in enumerate(slots) if slot.get("target_index") == target_index
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
        raise ValueError(f"{entry['name']}: selected targeted positives cannot share a crop")

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
                raise ValueError(f"Replay schedule event index mismatch at line {line_index}")
            event["replay_source_event_index"] = line_index
            event["sampling_source"] = "replay"
            selected.append(event)
    if len(selected) != stop - start:
        raise ValueError(f"Expected {stop - start} replay events, found {len(selected)}")
    return selected


def target_category_sequence(arm: str, rng: np.random.Generator) -> list[str]:
    if arm == "category_1alldiffuse_replay50":
        sequence = [
            category
            for category, count in DIFFUSE_TARGET_COUNTS.items()
            for _ in range(count)
        ]
    else:
        category = ARM_TARGETS[arm][0]
        sequence = [category] * TARGET_EVENTS
    if len(sequence) != TARGET_EVENTS:
        raise ValueError(f"{arm}: expected {TARGET_EVENTS} target categories, got {len(sequence)}")
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
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
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
                int(item["target_index"]) for item in event.get("positive_crop_points", [])
            }
            if (
                len(target_slots) != 1
                or int(event.get("anchor_target_index", -1)) != target_index
                or target_index not in crop_targets
            ):
                anchor_failures.append(index)
    for epoch in range(EPOCHS):
        block = events[epoch * EVENTS_PER_EPOCH : (epoch + 1) * EVENTS_PER_EPOCH]
        counts = Counter(event["sampling_source"] for event in block)
        observed = {"targeted": counts["targeted"], "replay": counts["replay"]}
        if observed != {"targeted": TARGET_EVENTS_PER_EPOCH, "replay": REPLAY_EVENTS_PER_EPOCH}:
            raise ValueError(f"{arm}: epoch {epoch + 1} source counts are {observed}")
        per_epoch.append(observed)
    if anchor_failures:
        raise ValueError(f"{arm}: {len(anchor_failures)} targeted anchor audit failures")
    expected_categories = (
        DIFFUSE_TARGET_COUNTS
        if arm == "category_1alldiffuse_replay50"
        else {ARM_TARGETS[arm][0]: TARGET_EVENTS}
    )
    if dict(target_counts) != expected_categories:
        raise ValueError(
            f"{arm}: target counts differ: expected={expected_categories} observed={dict(target_counts)}"
        )
    return {
        "events": len(events),
        "sampling_source_counts": {"targeted": TARGET_EVENTS, "replay": REPLAY_EVENTS},
        "target_category_counts": dict(target_counts),
        "target_anchor_failures": 0,
        "per_epoch": per_epoch,
        "shared_replay_fingerprint": replay_fingerprint(replay_events),
    }


def prepare_schedules(args: argparse.Namespace) -> dict[str, Any]:
    if args.replay_stop - args.replay_start != REPLAY_EVENTS:
        raise ValueError("Exp012 requires exactly 5,000 source replay events")
    train_entries = load_split_entries(args.metadata, ["train"])
    negative_pool = build_negative_prompt_pool(train_entries)
    options = load_target_options(train_entries, args.preprocessed_cache_dir)
    pools = target_finding_pools(train_entries, options)
    replay = read_replay_slice(args.replay_schedule, args.replay_start, args.replay_stop)

    placement_rng = np.random.default_rng(args.seed)
    target_positions_by_epoch = []
    for _ in range(EPOCHS):
        target_positions_by_epoch.append(
            set(int(value) for value in placement_rng.permutation(EVENTS_PER_EPOCH)[:50])
        )

    args.schedule_output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Any] = {}
    expected_replay_fingerprint: str | None = None
    for arm_index, arm in enumerate(ARM_TARGETS):
        target_rng = np.random.default_rng(args.seed + 1000 + arm_index)
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
        if target_cursor != TARGET_EVENTS or replay_cursor != REPLAY_EVENTS:
            raise AssertionError("Exp012 schedule cursors did not consume every event")

        schedule_path = args.schedule_output_dir / f"{arm}.jsonl"
        with schedule_path.open("w") as handle:
            for event in mixed:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        audit = audit_schedule(mixed, arm)
        fingerprint = audit["shared_replay_fingerprint"]
        if expected_replay_fingerprint is None:
            expected_replay_fingerprint = fingerprint
        elif fingerprint != expected_replay_fingerprint:
            raise ValueError(f"{arm}: replay fingerprint differs from the first arm")

        manifest = {
            "created_at_utc": utc_now_iso(),
            "experiment": EXPERIMENT_ID,
            "arm": arm,
            "target_categories": list(ARM_TARGETS[arm]),
            "metadata": str(args.metadata),
            "metadata_sha256": sha256_file(args.metadata),
            "preprocessed_cache_dir": str(args.preprocessed_cache_dir),
            "source_replay_schedule": str(args.replay_schedule),
            "source_replay_schedule_sha256": sha256_file(args.replay_schedule),
            "source_replay_event_range": [args.replay_start, args.replay_stop],
            "seed": args.seed,
            "target_seed": args.seed + 1000 + arm_index,
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
        "command": command_string(),
        "arms": manifests,
        "shared_replay_fingerprint": expected_replay_fingerprint,
    }
    write_json(args.schedule_output_dir / "exp012_schedules.manifest.json", combined)
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
    train_names = {entry["name"] for entry in load_split_entries(args.metadata, ["train"])}
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
        write_json(target_path, {"test": [evaluator_entry(entry) for entry in target_entries]})
        write_json(union_path, {"test": [evaluator_entry(entry) for entry in union_entries]})
        target_findings = category_count(target_entries, target_categories)
        record = {
            "target_categories": list(target_tuple),
            "target": {
                "path": str(target_path),
                "sha256": sha256_file(target_path),
                "cases": len(target_entries),
                "all_findings": sum(len(entry.get("findings", {})) for entry in target_entries),
                "target_findings": target_findings,
            },
            "union": {
                "path": str(union_path),
                "sha256": sha256_file(union_path),
                "cases": len(union_entries),
                "all_findings": sum(len(entry.get("findings", {})) for entry in union_entries),
                "target_findings": target_findings,
                "val80_cases": len(val80_names),
            },
        }
        records[arm] = record

    expected = {
        "category_1alldiffuse_replay50": (42, 52, 80),
        "category_2a_replay50": (63, 69, 114),
        "category_2b_replay50": (40, 49, 98),
        "category_2c_replay50": (53, 60, 110),
    }
    for arm, (cases, findings, union_cases) in expected.items():
        record = records[arm]
        observed = (
            record["target"]["cases"],
            record["target"]["target_findings"],
            record["union"]["cases"],
        )
        if observed != (cases, findings, union_cases):
            raise ValueError(f"{arm}: expected {(cases, findings, union_cases)}, observed {observed}")

    manifest = {
        "created_at_utc": utc_now_iso(),
        "experiment": EXPERIMENT_ID,
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
        / "rexgroundingct_exp012_category_evaluation_subsets.manifest.json"
    )
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "schedules", "evaluation"], default="all")
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--preprocessed-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--replay-schedule", type=Path, default=DEFAULT_REPLAY_SCHEDULE)
    parser.add_argument("--replay-start", type=int, default=REPLAY_START)
    parser.add_argument("--replay-stop", type=int, default=REPLAY_STOP)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--schedule-output-dir", type=Path, default=Path("exp012_schedules"))
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
            f"Prepared {len(result['arms'])} Exp012 schedules under "
            f"{args.schedule_output_dir}"
        )
    if args.mode in {"all", "evaluation"}:
        result = prepare_evaluation_subsets(args)
        print(
            f"Prepared {len(result['arms'])} Exp012 evaluation subset pairs under "
            f"{args.evaluation_output_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
