#!/usr/bin/env python3
"""Prepare deterministic Exp013 target-only schedules and target censuses."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from math import floor, sqrt
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


EXPERIMENT_ID = "013_voxtell_public_category_only_specialists"
DEFAULT_CACHE = Path(
    "/mnt/shengdata1/hengjie/datasets/rexgroundingct/preprocessed/voxtell/"
    "crop_zscore_native_v1"
)
DEFAULT_VAL200 = Path("configs/evaluation/rexgroundingct_val200_seed20260723.json")
PATCH_SIZE = (192, 192, 192)
EPOCHS = 100
EVENTS_PER_EPOCH = 100
TOTAL_EVENTS = EPOCHS * EVENTS_PER_EPOCH
MASTER_SEED = 20260803

ARM_TARGETS: dict[str, tuple[str, ...]] = {
    "category_1all_diffuse_target100": ("1a", "1b", "1c", "1d", "1e", "1f"),
    "category_2all_focal_target100": ("2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h"),
    "category_2bc_target100": ("2b", "2c"),
    "category_2d_target100": ("2d",),
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
    "2d": 1743,
    "2e": 237,
    "2f": 16,
    "2g": 18,
    "2h": 57,
}
EVALUATION_NAMES = {
    "category_1all_diffuse_target100": "rexgroundingct_val42_category_1all_diffuse_target_census.json",
    "category_2all_focal_target100": "rexgroundingct_val196_category_2all_focal_target_census.json",
    "category_2bc_target100": "rexgroundingct_val76_category_2bc_target_census.json",
    "category_2d_target100": "rexgroundingct_val119_category_2d_target_census.json",
}
EXPECTED_EVALUATION = {
    "category_1all_diffuse_target100": (42, 52),
    "category_2all_focal_target100": (196, 329),
    "category_2bc_target100": (76, 109),
    "category_2d_target100": (119, 132),
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


def allocate_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    raw = {key: value / sum(weights.values()) * total for key, value in weights.items()}
    counts = {key: floor(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    for key, _value in sorted(
        raw.items(), key=lambda item: (item[1] - floor(item[1]), item[0]), reverse=True
    )[:remainder]:
        counts[key] += 1
    if sum(counts.values()) != total:
        raise AssertionError("category allocation failed to sum to requested total")
    return counts


def target_event_counts(arm: str) -> dict[str, int]:
    categories = ARM_TARGETS[arm]
    if arm == "category_1all_diffuse_target100":
        return allocate_counts(
            {category: sqrt(EXPECTED_TRAIN_FINDINGS[category]) for category in categories},
            TOTAL_EVENTS,
        )
    if arm in {"category_2all_focal_target100", "category_2bc_target100"}:
        return allocate_counts(
            {category: float(EXPECTED_TRAIN_FINDINGS[category]) for category in categories},
            TOTAL_EVENTS,
        )
    return {categories[0]: TOTAL_EVENTS}


def load_target_options(
    entries: list[dict[str, Any]], cache_root: Path
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = str(entry["name"])
        key = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem
        root = cache_root / "cases" / key
        metadata_path = root / "metadata.json"
        if not (root / ".complete").is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete cached case: {root}")
        metadata = read_json(metadata_path)
        shape = metadata.get("resampled_shape_zyx")
        candidates = metadata.get("resampled_target_positive_point_candidates")
        voxels = metadata.get("resampled_target_voxels") or metadata.get("source_target_voxels")
        prompts = sorted_prompts(entry)
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f"{entry['name']}: invalid cached spatial shape")
        if not isinstance(candidates, list) or len(candidates) < len(prompts):
            raise ValueError(f"{entry['name']}: missing positive-point candidates")
        if not isinstance(voxels, list) or len(voxels) < len(prompts):
            raise ValueError(f"{entry['name']}: missing target voxel counts")

        points: dict[int, list[int]] = {}
        for index in range(len(prompts)):
            if int(voxels[index]) > 0 and candidates[index]:
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
            "Eligible target finding counts differ from audited train split: "
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
        raise ValueError(f"{entry['name']}: selected positives cannot share a crop")

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


def audit_schedule(events: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    target_counts: Counter[str] = Counter()
    per_epoch: list[dict[str, int]] = []
    anchor_failures: list[int] = []
    for index, event in enumerate(events):
        if int(event["event_index"]) != index:
            raise ValueError(f"{arm}: nonsequential event index at {index}")
        if event["sampling_source"] != "targeted":
            raise ValueError(f"{arm}: non-target event at {index}")
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
        if observed != {"targeted": 100, "replay": 0}:
            raise ValueError(f"{arm}: epoch {epoch + 1} source counts are {observed}")
        per_epoch.append(observed)
    expected_categories = target_event_counts(arm)
    if dict(target_counts) != expected_categories:
        raise ValueError(
            f"{arm}: target counts differ: expected={expected_categories} observed={dict(target_counts)}"
        )
    if anchor_failures:
        raise ValueError(f"{arm}: {len(anchor_failures)} targeted anchor audit failures")
    digest = hashlib.sha256()
    for event in events:
        payload = {
            "case_name": event["case_name"],
            "target_category": event["requested_target_category"],
            "target_index": event["requested_target_index"],
            "prompt_slots": event["prompt_slots"],
            "patch_starts": event["patch_starts"],
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return {
        "events": len(events),
        "sampling_source_counts": {"targeted": TOTAL_EVENTS, "replay": 0},
        "target_category_counts": dict(target_counts),
        "target_anchor_failures": 0,
        "per_epoch": per_epoch,
        "schedule_fingerprint": digest.hexdigest(),
    }


def prepare_schedules(args: argparse.Namespace) -> dict[str, Any]:
    train_entries = load_split_entries(args.metadata, ["train"])
    negative_pool = build_negative_prompt_pool(train_entries)
    options = load_target_options(train_entries, args.preprocessed_cache_dir)
    pools = target_finding_pools(train_entries, options)
    placement_rng = np.random.default_rng(args.seed)

    args.schedule_output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARM_TARGETS):
        target_rng = np.random.default_rng(args.seed + 1000 + arm_index)
        category_sequence = [
            category
            for category, count in target_event_counts(arm).items()
            for _ in range(count)
        ]
        target_rng.shuffle(category_sequence)
        epoch_orders = [
            placement_rng.permutation(EVENTS_PER_EPOCH).tolist() for _ in range(EPOCHS)
        ]
        events: list[dict[str, Any]] = []
        for category in category_sequence:
            pool = pools[category]
            entry, target_index = pool[int(target_rng.integers(0, len(pool)))]
            events.append(
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
        cursor = 0
        for epoch in range(EPOCHS):
            epoch_events = events[cursor : cursor + EVENTS_PER_EPOCH]
            cursor += EVENTS_PER_EPOCH
            for position in epoch_orders[epoch]:
                event = dict(epoch_events[int(position)])
                event["event_index"] = len(mixed)
                mixed.append(event)
        if cursor != TOTAL_EVENTS:
            raise AssertionError("schedule cursor did not consume every event")

        schedule_path = args.schedule_output_dir / f"{arm}.jsonl"
        with schedule_path.open("w") as handle:
            for event in mixed:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        audit = audit_schedule(mixed, arm)
        manifest = {
            "created_at_utc": utc_now_iso(),
            "experiment": EXPERIMENT_ID,
            "arm": arm,
            "target_categories": list(ARM_TARGETS[arm]),
            "metadata": str(args.metadata),
            "metadata_sha256": sha256_file(args.metadata),
            "preprocessed_cache_dir": str(args.preprocessed_cache_dir),
            "seed": args.seed,
            "target_seed": args.seed + 1000 + arm_index,
            "placement_seed": args.seed,
            "schedule_schema_version": 2,
            "patch_size": list(PATCH_SIZE),
            "output_jsonl": str(schedule_path),
            "output_jsonl_sha256": sha256_file(schedule_path),
            "audit": audit,
        }
        write_json(schedule_path.with_suffix(".manifest.json"), manifest)
        manifests[arm] = manifest

    combined = {
        "created_at_utc": utc_now_iso(),
        "experiment": EXPERIMENT_ID,
        "command": command_string(),
        "arms": manifests,
    }
    write_json(args.schedule_output_dir / "exp013_schedules.manifest.json", combined)
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
    train_names = {entry["name"] for entry in load_split_entries(args.metadata, ["train"])}
    val_names = {entry["name"] for entry in val200}
    if train_names & val_names:
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
        target_path = args.evaluation_output_dir / EVALUATION_NAMES[arm]
        write_json(target_path, {"test": [evaluator_entry(entry) for entry in target_entries]})
        target_findings = category_count(target_entries, target_categories)
        expected_cases, expected_findings = EXPECTED_EVALUATION[arm]
        observed = (len(target_entries), target_findings)
        if observed != (expected_cases, expected_findings):
            raise ValueError(f"{arm}: expected {(expected_cases, expected_findings)}, observed {observed}")
        records[arm] = {
            "target_categories": list(target_tuple),
            "target": {
                "path": str(target_path),
                "sha256": sha256_file(target_path),
                "cases": len(target_entries),
                "all_findings": sum(len(entry.get("findings", {})) for entry in target_entries),
                "target_findings": target_findings,
            },
        }

    manifest = {
        "created_at_utc": utc_now_iso(),
        "experiment": EXPERIMENT_ID,
        "command": command_string(),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "val200_json": str(args.val200_json),
        "val200_json_sha256": sha256_file(args.val200_json),
        "zero_train_validation_case_overlap": True,
        "arms": records,
    }
    manifest_path = args.evaluation_output_dir / "rexgroundingct_exp013_category_only_evaluation_subsets.manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "schedules", "evaluation"], default="all")
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--preprocessed-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--schedule-output-dir", type=Path, default=Path("exp013_schedules"))
    parser.add_argument("--val200-json", type=Path, default=DEFAULT_VAL200)
    parser.add_argument(
        "--evaluation-output-dir", type=Path, default=Path("configs/evaluation")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode in {"all", "schedules"}:
        result = prepare_schedules(args)
        print(f"Prepared {len(result['arms'])} Exp013 schedules under {args.schedule_output_dir}")
    if args.mode in {"all", "evaluation"}:
        result = prepare_evaluation_subsets(args)
        print(f"Prepared {len(result['arms'])} Exp013 evaluation subsets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
