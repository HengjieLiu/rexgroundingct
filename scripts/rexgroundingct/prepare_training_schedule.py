#!/usr/bin/env python3
"""Generate a deterministic JSONL patch-event schedule for ReXGroundingCT training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    CT_ROOT,
    REX_METADATA,
    REX_SEG_DIR,
    command_string,
    ct_rate_abs_path,
    load_split_entries,
    sha256_file,
    sorted_prompts,
    utc_now_iso,
    write_json,
)


DEFAULT_PATCH_SIZE = (192, 192, 192)


def parse_patch_size(values: list[int]) -> tuple[int, int, int]:
    patch_size = tuple(int(value) for value in values)
    if len(patch_size) != 3:
        raise ValueError("--patch-size must contain exactly 3 integers")
    if any(value <= 0 for value in patch_size):
        raise ValueError("--patch-size values must be positive")
    return patch_size


def build_negative_prompt_pool(entries: list[dict]) -> list[str]:
    return sorted({prompt.lower() for entry in entries for prompt in sorted_prompts(entry)})


def starts_containing_points(
    points: np.ndarray,
    spatial_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    rng: np.random.Generator | None = None,
) -> list[int] | None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    starts: list[int] = []
    for axis, (dim, patch) in enumerate(zip(spatial_shape, patch_size)):
        if dim <= patch:
            starts.append(0)
            continue
        low = max(0, int(points[:, axis].max()) - patch + 1)
        high = min(int(points[:, axis].min()), dim - patch)
        if low > high:
            return None
        if rng is None:
            starts.append(int((low + high) // 2))
        else:
            starts.append(int(rng.integers(low, high + 1)))
    return starts


def representative_positive_point(coords: np.ndarray) -> list[int]:
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    center = (bbox_min.astype(np.float32) + bbox_max.astype(np.float32)) / 2.0
    distances = np.sum((coords.astype(np.float32, copy=False) - center) ** 2, axis=1)
    return [int(value) for value in coords[int(np.argmin(distances))].tolist()]


def load_preprocessed_targets(
    entry: dict,
    ct_root: Path,
    seg_dir: Path,
) -> np.ndarray:
    import nibabel as nib
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero

    from train_text_conditioned_voxtell import (
        crop_target,
        reorient_target_to_voxtell_layout,
    )

    name = entry["name"]
    ct_path = ct_rate_abs_path(name, ct_root)
    gt_path = seg_dir / name
    if not ct_path.is_file():
        raise FileNotFoundError(f"Missing CT: {ct_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing segmentation: {gt_path}")

    reader = NibabelIOWithReorient()
    image, ct_properties = reader.read_images([str(ct_path)])
    image = image.astype(np.float32, copy=False)

    gt_img = nib.load(str(gt_path))
    target_fxyz = np.asanyarray(gt_img.dataobj).astype(np.float32, copy=False)
    target_fzyx, _ = reorient_target_to_voxtell_layout(target_fxyz, ct_properties, name)
    if tuple(image.shape[1:]) != tuple(target_fzyx.shape[1:]):
        raise ValueError(
            f"{name}: image shape {image.shape[1:]} != orientation-fixed target shape {target_fzyx.shape[1:]}"
        )

    _, _, bbox = crop_to_nonzero(image, None)
    bbox_list = [[int(v) for v in axis] for axis in bbox]
    return crop_target(target_fzyx, bbox_list)


def load_cached_preprocessed_targets(entry: dict, cache_root: Path) -> np.ndarray:
    from voxtell_preprocessed_cache import cache_case_paths

    paths = cache_case_paths(cache_root, entry["name"])
    if not paths["complete"].is_file():
        raise FileNotFoundError(f"Incomplete cached case: {paths['root']}")
    if not paths["targets"].is_file():
        raise FileNotFoundError(f"Cached case has no targets: {paths['targets']}")
    with np.load(paths["targets"], allow_pickle=False) as data:
        targets = data["targets"]
    if targets.ndim != 4:
        raise ValueError(f"{entry['name']}: cached targets must be FZYX, got {targets.shape}")
    return np.asarray(targets, dtype=np.uint8)


def positive_crop_options_from_cached_metadata(
    entry: dict,
    cache_root: Path,
    patch_size: tuple[int, int, int],
) -> tuple[dict | None, Counter]:
    from voxtell_preprocessed_cache import cache_case_paths

    paths = cache_case_paths(cache_root, entry["name"])
    if not paths["complete"].is_file():
        raise FileNotFoundError(f"Incomplete cached case: {paths['root']}")
    if not paths["metadata"].is_file():
        raise FileNotFoundError(f"Cached case has no metadata: {paths['metadata']}")
    metadata = json.loads(paths["metadata"].read_text())
    prompts = sorted_prompts(entry)
    stats: Counter = Counter()
    shape = metadata.get("resampled_shape_zyx")
    positive_candidates = metadata.get("resampled_target_positive_point_candidates")
    target_voxels = metadata.get("resampled_target_voxels") or metadata.get("source_target_voxels")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError(f"{entry['name']}: cached metadata has invalid resampled_shape_zyx")
    if not isinstance(positive_candidates, list) or len(positive_candidates) < len(prompts):
        raise ValueError(f"{entry['name']}: cached metadata lacks positive-point candidates")
    if not isinstance(target_voxels, list) or len(target_voxels) < len(prompts):
        raise ValueError(f"{entry['name']}: cached metadata lacks target voxel counts")

    spatial_shape = tuple(int(dim) for dim in shape)
    representative_points: dict[int, list[int]] = {}
    nonempty: list[int] = []
    for index in range(len(prompts)):
        candidates = positive_candidates[index]
        if int(target_voxels[index]) <= 0 or not candidates:
            continue
        nonempty.append(index)
        representative_points[index] = [int(value) for value in candidates[0]]

    compatible_pairs: list[dict] = []
    for left, right in combinations(nonempty, 2):
        points = np.asarray(
            [representative_points[left], representative_points[right]],
            dtype=np.int64,
        )
        if starts_containing_points(points, spatial_shape, patch_size, rng=None) is not None:
            compatible_pairs.append(
                {
                    "indices": [int(left), int(right)],
                    "points": [
                        [int(value) for value in representative_points[left]],
                        [int(value) for value in representative_points[right]],
                    ],
                }
            )
    if len(prompts) >= 2:
        if compatible_pairs:
            stats["eligible_multi_finding_cases"] += 1
            stats["compatible_positive_pairs"] += len(compatible_pairs)
            return (
                {
                    "positive_pairs": compatible_pairs,
                    "single_positive_indices": nonempty,
                    "single_positive_points": {
                        str(index): [int(value) for value in representative_points[index]]
                        for index in nonempty
                    },
                    "spatial_shape": [int(dim) for dim in spatial_shape],
                },
                stats,
            )
        stats["ineligible_multi_finding_cases"] += 1
        return None, stats
    if nonempty:
        stats["eligible_one_finding_cases"] += 1
        return (
            {
                "positive_pairs": [],
                "single_positive_indices": nonempty,
                "single_positive_points": {
                    str(index): [int(value) for value in representative_points[index]]
                    for index in nonempty
                },
                "spatial_shape": [int(dim) for dim in spatial_shape],
            },
            stats,
        )
    stats["ineligible_one_finding_cases"] += 1
    return None, stats


def build_positive_crop_options(
    entries: list[dict],
    ct_root: Path,
    seg_dir: Path,
    patch_size: tuple[int, int, int],
    num_workers: int,
    preprocessed_cache_dir: Path | None = None,
) -> tuple[dict[str, dict], Counter]:
    options_by_case: dict[str, dict] = {}
    stats: Counter = Counter()

    jobs = [
        (
            entry,
            str(ct_root),
            str(seg_dir),
            patch_size,
            str(preprocessed_cache_dir) if preprocessed_cache_dir is not None else None,
        )
        for entry in entries
    ]
    if num_workers <= 1:
        iterator = map(positive_crop_options_for_entry, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=num_workers)
        iterator = pool.map(positive_crop_options_for_entry, jobs)
    try:
        for name, options, case_stats in tqdm(
            iterator,
            total=len(jobs),
            desc="Indexing positive-crop training geometry",
        ):
            stats.update(case_stats)
            if options is not None:
                options_by_case[name] = options
    finally:
        if num_workers > 1:
            pool.shutdown()
    return options_by_case, stats


def positive_crop_options_for_entry(
    job: tuple[dict, str, str, tuple[int, int, int], str | None],
) -> tuple[str, dict | None, dict]:
    entry, ct_root, seg_dir, patch_size, preprocessed_cache_dir = job
    prompts = sorted_prompts(entry)
    stats: Counter = Counter()
    if preprocessed_cache_dir is not None:
        options, cached_stats = positive_crop_options_from_cached_metadata(
            entry,
            Path(preprocessed_cache_dir),
            patch_size,
        )
        return entry["name"], options, dict(cached_stats)

    targets = load_preprocessed_targets(entry, Path(ct_root), Path(seg_dir))
    if targets.shape[0] < len(prompts):
        raise ValueError(
            f"{entry['name']}: {len(prompts)} prompts but only {targets.shape[0]} mask channels"
        )
    spatial_shape = tuple(int(dim) for dim in targets.shape[1:])
    representative_points: dict[int, list[int]] = {}
    nonempty: list[int] = []
    for index in range(len(prompts)):
        coords = np.argwhere(targets[index] > 0)
        if coords.size == 0:
            continue
        nonempty.append(index)
        representative_points[index] = representative_positive_point(coords)

    compatible_pairs: list[dict] = []
    for left, right in combinations(nonempty, 2):
        points = np.asarray(
            [representative_points[left], representative_points[right]],
            dtype=np.int64,
        )
        if starts_containing_points(points, spatial_shape, patch_size, rng=None) is not None:
            compatible_pairs.append(
                {
                    "indices": [int(left), int(right)],
                    "points": [
                        [int(value) for value in representative_points[left]],
                        [int(value) for value in representative_points[right]],
                    ],
                }
            )
    if len(prompts) >= 2:
        if compatible_pairs:
            stats["eligible_multi_finding_cases"] += 1
            stats["compatible_positive_pairs"] += len(compatible_pairs)
            return (
                entry["name"],
                {
                    "positive_pairs": compatible_pairs,
                    "single_positive_indices": nonempty,
                    "single_positive_points": {
                        str(index): [int(value) for value in representative_points[index]]
                        for index in nonempty
                    },
                    "spatial_shape": [int(dim) for dim in spatial_shape],
                },
                dict(stats),
            )
        stats["ineligible_multi_finding_cases"] += 1
        return entry["name"], None, dict(stats)
    if nonempty:
        stats["eligible_one_finding_cases"] += 1
        return (
            entry["name"],
            {
                "positive_pairs": [],
                "single_positive_indices": nonempty,
                "single_positive_points": {
                    str(index): [int(value) for value in representative_points[index]]
                    for index in nonempty
                },
                "spatial_shape": [int(dim) for dim in spatial_shape],
            },
            dict(stats),
        )
    stats["ineligible_one_finding_cases"] += 1
    return entry["name"], None, dict(stats)


def sample_event(
    entries: list[dict],
    negative_prompt_pool: list[str],
    rng,
    foreground_oversample_prob: float,
    patch_size: tuple[int, int, int],
    event_index: int,
    require_positive_crop: bool,
    positive_crop_options: dict[str, dict] | None,
) -> dict:
    entry = entries[int(rng.integers(0, len(entries)))]
    prompts = sorted_prompts(entry)
    stats: Counter = Counter(samples=1)
    n_findings = len(prompts)
    if n_findings < 1:
        raise ValueError(f"{entry['name']}: no findings available")

    case_options = positive_crop_options.get(entry["name"]) if positive_crop_options else None
    if n_findings >= 2:
        if require_positive_crop:
            if not case_options or not case_options["positive_pairs"]:
                raise ValueError(f"{entry['name']}: no compatible positive-crop pair")
            pair = case_options["positive_pairs"][int(rng.integers(0, len(case_options["positive_pairs"])))]
            positive_indices = [int(pair["indices"][0]), int(pair["indices"][1])]
            positive_points_by_target = {
                positive_indices[0]: [int(value) for value in pair["points"][0]],
                positive_indices[1]: [int(value) for value in pair["points"][1]],
            }
            stats["positive_crop_compatible_pair_samples"] += 1
        else:
            positive_indices = rng.choice(n_findings, size=2, replace=False).tolist()
            positive_points_by_target = {}
        n_negative = 1
        stats["multi_finding_samples"] += 1
    else:
        if require_positive_crop and (not case_options or not case_options["single_positive_indices"]):
            raise ValueError(f"{entry['name']}: no non-empty one-finding positive-crop target")
        positive_indices = [0]
        positive_points_by_target = {}
        if require_positive_crop and case_options:
            positive_points_by_target = {
                0: [
                    int(value)
                    for value in case_options["single_positive_points"][str(positive_indices[0])]
                ]
            }
        n_negative = 2
        stats["one_finding_fallback_samples"] += 1

    positive_slots = [
        {
            "prompt": prompts[index],
            "target_index": int(index),
            "is_positive": True,
        }
        for index in positive_indices
    ]
    case_prompt_keys = {prompt.lower() for prompt in prompts}
    candidates = [prompt for prompt in negative_prompt_pool if prompt not in case_prompt_keys]
    if not candidates:
        candidates = negative_prompt_pool
    replace = len(candidates) < n_negative
    negative_prompts = rng.choice(candidates, size=n_negative, replace=replace).tolist()
    negative_slots = [
        {
            "prompt": str(prompt),
            "target_index": None,
            "is_positive": False,
        }
        for prompt in negative_prompts
    ]

    slots = positive_slots + negative_slots
    order = rng.permutation(len(slots))
    slots = [slots[int(index)] for index in order]
    stats["positive_slots"] += len(positive_slots)
    stats["negative_slots"] += len(negative_slots)

    positive_slot_indices = [
        slot_index for slot_index, slot in enumerate(slots)
        if slot["is_positive"]
    ]
    one_finding_foreground_forced = bool(stats.get("one_finding_fallback_samples", 0))
    foreground_forced = one_finding_foreground_forced or require_positive_crop
    foreground_requested = foreground_forced or bool(rng.random() < foreground_oversample_prob)
    anchor_slot_index = None
    anchor_target_index = None
    if foreground_requested and positive_slot_indices:
        if one_finding_foreground_forced:
            anchor_slot_index = positive_slot_indices[0]
            stats["one_finding_forced_foreground_anchor"] += 1
        elif require_positive_crop:
            anchor_slot_index = int(positive_slot_indices[int(rng.integers(0, len(positive_slot_indices)))])
            stats["positive_crop_forced_foreground_anchor"] += 1
        else:
            anchor_slot_index = int(positive_slot_indices[int(rng.integers(0, len(positive_slot_indices)))])
        anchor_target_index = int(slots[anchor_slot_index]["target_index"])
        stats["foreground_oversample_requested"] += 1

    event = {
        "schema_version": 2,
        "event_index": event_index,
        "case_name": entry["name"],
        "source_split": entry.get("_split"),
        "source_split_index": entry.get("_index"),
        "patch_size": [int(value) for value in patch_size],
        "patch_seed": int(rng.integers(0, 2**63 - 1)),
        "prompt_slots": slots,
        "foreground_requested": bool(foreground_requested),
        "foreground_forced": bool(foreground_forced),
        "require_positive_crop": bool(require_positive_crop),
        "anchor_slot_index": anchor_slot_index,
        "anchor_target_index": anchor_target_index,
        "stats": dict(stats),
    }
    if require_positive_crop:
        if not case_options:
            raise ValueError(f"{entry['name']}: missing positive-crop options")
        positive_points = []
        ordered_points = []
        for slot_index, slot in enumerate(slots):
            if not slot["is_positive"]:
                continue
            target_index = int(slot["target_index"])
            point = positive_points_by_target.get(target_index)
            if point is None:
                point = [
                    int(value)
                    for value in case_options["single_positive_points"][str(target_index)]
                ]
            positive_points.append(
                {
                    "slot_index": int(slot_index),
                    "target_index": target_index,
                    "point": [int(value) for value in point],
                }
            )
            ordered_points.append(point)
        spatial_shape = tuple(int(dim) for dim in case_options["spatial_shape"])
        starts = starts_containing_points(
            np.asarray(ordered_points, dtype=np.int64),
            spatial_shape,
            patch_size,
            rng,
        )
        if starts is None:
            raise ValueError(
                f"{entry['name']}: compatible positive-crop points cannot share patch: {positive_points}"
            )
        event["spatial_shape"] = [int(dim) for dim in spatial_shape]
        event["patch_starts"] = [int(value) for value in starts]
        event["positive_crop_points"] = positive_points
        stats["positive_crop_success"] += 1
        event["stats"] = dict(stats)
    return event


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--patch-size", nargs=3, type=int, default=list(DEFAULT_PATCH_SIZE))
    parser.add_argument("--foreground-oversample-prob", type=float, default=0.85)
    parser.add_argument("--require-positive-crop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--preprocessed-cache-dir",
        type=Path,
        default=None,
        help="Optional VoxTell preprocessed cache for positive-crop geometry.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-train-cases", type=int, default=None)
    args = parser.parse_args()

    if args.events < 1:
        raise ValueError("--events must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    patch_size = parse_patch_size(args.patch_size)

    train_entries = load_split_entries(args.metadata, ["train"])
    if args.max_train_cases is not None:
        if args.max_train_cases < 1:
            raise ValueError("--max-train-cases must be >= 1 when set")
        train_entries = train_entries[: args.max_train_cases]
    original_train_entry_count = len(train_entries)
    negative_pool = build_negative_prompt_pool(train_entries)
    positive_crop_options = None
    positive_crop_stats: Counter = Counter()
    if args.require_positive_crop:
        positive_crop_options, positive_crop_stats = build_positive_crop_options(
            train_entries,
            args.ct_root,
            args.seg_dir,
            patch_size,
            args.num_workers,
            args.preprocessed_cache_dir,
        )
        train_entries = [entry for entry in train_entries if entry["name"] in positive_crop_options]
        if not train_entries:
            raise RuntimeError("No train entries are eligible for --require-positive-crop")
    rng = np.random.default_rng(args.seed)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    with args.output_jsonl.open("w") as handle:
        for event_index in tqdm(range(args.events), desc="Generating patch schedule"):
            event = sample_event(
                entries=train_entries,
                negative_prompt_pool=negative_pool,
                rng=rng,
                foreground_oversample_prob=args.foreground_oversample_prob,
                patch_size=patch_size,
                event_index=event_index,
                require_positive_crop=args.require_positive_crop,
                positive_crop_options=positive_crop_options,
            )
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            stats.update(event.get("stats", {}))

    manifest_path = args.manifest_json or args.output_jsonl.with_suffix(".manifest.json")
    manifest = {
        "created_at_utc": utc_now_iso(),
        "command": command_string(),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "source_split": "train",
        "source_split_size": original_train_entry_count,
        "ct_root": str(args.ct_root),
        "seg_dir": str(args.seg_dir),
        "negative_prompt_pool_size": len(negative_pool),
        "selection_seed": args.seed,
        "schedule_schema_version": 2,
        "patch_start_policy": (
            "materialized_patch_starts_for_positive_crop_events"
            if args.require_positive_crop
            else "computed_at_training_time_from_patch_seed_after_case_preprocessing"
        ),
        "events": args.events,
        "patch_size": list(patch_size),
        "foreground_oversample_probability": args.foreground_oversample_prob,
        "require_positive_crop": bool(args.require_positive_crop),
        "positive_crop_compatibility": {
            "space": "VoxTell training FZYX after CT reorientation and crop_to_nonzero",
            "preprocessed_cache_dir": (
                str(args.preprocessed_cache_dir) if args.preprocessed_cache_dir is not None else None
            ),
            "eligible_source_split_size": len(train_entries),
            "stats": dict(positive_crop_stats),
        },
        "one_finding_fallback": "1_positive_2_negative_forced_foreground_anchor",
        "stats": dict(stats),
        "output_jsonl": str(args.output_jsonl),
        "output_jsonl_sha256": sha256_file(args.output_jsonl),
    }
    write_json(manifest_path, manifest)
    print(f"Wrote {args.events} scheduled patch events to {args.output_jsonl}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
