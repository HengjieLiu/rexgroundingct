#!/usr/bin/env python3
"""Summarize evaluator output for an accepted SideExp001 probe."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from analyze_validation_probe import (
    SIDE_EXPERIMENT_ID,
    canonical_json_bytes,
    read_json,
    sha256_file,
)


def summarize(
    probe: dict[str, Any],
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    if manifest.get("side_experiment_id") != SIDE_EXPERIMENT_ID:
        raise ValueError("probe manifest belongs to a different side experiment")
    if manifest.get("status") != "accepted":
        raise ValueError("probe manifest is not accepted")

    probe_entries = probe.get("test")
    evaluation_cases = evaluation.get("cases")
    if not isinstance(probe_entries, list) or not isinstance(evaluation_cases, list):
        raise ValueError("expected probe['test'] and evaluation['cases'] lists")
    probe_by_case = {entry["name"]: entry for entry in probe_entries}
    evaluation_by_case = {entry["file"]: entry for entry in evaluation_cases}
    if len(probe_by_case) != len(probe_entries):
        raise ValueError("probe contains duplicate case names")
    if len(evaluation_by_case) != len(evaluation_cases):
        raise ValueError("evaluation contains duplicate case names")
    if set(probe_by_case) != set(evaluation_by_case):
        raise ValueError("evaluation case set does not exactly match the probe")

    weights = manifest["weights"]
    categories = sorted(weights["validation"])
    dice_sum = {category: 0.0 for category in categories}
    hit_sum = {category: 0 for category in categories}
    counts = {category: 0 for category in categories}
    for case_name in sorted(probe_by_case):
        probe_entry = probe_by_case[case_name]
        evaluation_entry = evaluation_by_case[case_name]
        expected_keys = set(probe_entry["findings"])
        observed_keys = {
            key.removeprefix("finding_")
            for key in evaluation_entry.get("findings", {})
        }
        if expected_keys != observed_keys:
            raise ValueError(
                f"{case_name}: evaluation finding keys do not match the probe"
            )
        for metric_key, metric in evaluation_entry["findings"].items():
            finding_key = metric_key.removeprefix("finding_")
            category = probe_entry["categories"][finding_key]
            if category not in counts:
                raise ValueError(f"{case_name}: unexpected category {category!r}")
            counts[category] += 1
            dice_sum[category] += float(metric["global_dice"])
            hit_sum[category] += int(bool(metric["global_hit"]))

    if counts != {key: int(value) for key, value in manifest["selection"]["category_counts"].items()}:
        raise ValueError("observed category counts do not match the probe manifest")
    category_metrics = {
        category: {
            "findings": counts[category],
            "mean_global_dice": dice_sum[category] / counts[category],
            "hit_rate": hit_sum[category] / counts[category],
            "hits": hit_sum[category],
        }
        for category in categories
    }
    finding_count = sum(counts.values())
    raw_dice = sum(dice_sum.values()) / finding_count
    raw_hit = sum(hit_sum.values()) / finding_count

    def weighted(metric: str, weight_name: str) -> float:
        return sum(
            float(weights[weight_name][category])
            * float(category_metrics[category][metric])
            for category in categories
        )

    return {
        "side_experiment_id": SIDE_EXPERIMENT_ID,
        "selection_sha256_pre_holdout": manifest[
            "selection_sha256_pre_holdout"
        ],
        "case_count": len(probe_entries),
        "finding_count": finding_count,
        "category_metrics": category_metrics,
        "metrics": {
            "raw_evaluator": {
                "mean_global_dice_per_finding": raw_dice,
                "hit_rate": raw_hit,
            },
            "validation_post_stratified": {
                "label": "estimate for the full 200-case validation distribution",
                "mean_global_dice_per_finding": weighted(
                    "mean_global_dice", "validation"
                ),
                "hit_rate": weighted("hit_rate", "validation"),
            },
            "full_test_reweighted_extrapolation": {
                "label": (
                    "extrapolation from validation outcomes; "
                    "not measured test performance"
                ),
                "mean_global_dice_per_finding": weighted(
                    "mean_global_dice", "full_test"
                ),
                "hit_rate": weighted("hit_rate", "full_test"),
            },
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-json", required=True, type=Path)
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--eval-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = read_json(args.probe_manifest)
    expected_probe_hash = manifest.get("probe_json_sha256")
    actual_probe_hash = sha256_file(args.probe_json)
    if expected_probe_hash != actual_probe_hash:
        raise ValueError(
            "probe JSON SHA256 does not match the accepted provenance manifest"
        )
    result = summarize(
        read_json(args.probe_json),
        manifest,
        read_json(args.eval_json),
    )
    result["inputs"] = {
        "probe_json": str(args.probe_json.resolve()),
        "probe_json_sha256": actual_probe_hash,
        "probe_manifest": str(args.probe_manifest.resolve()),
        "probe_manifest_sha256": sha256_file(args.probe_manifest),
        "evaluation_json": str(args.eval_json.resolve()),
        "evaluation_json_sha256": sha256_file(args.eval_json),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_bytes(canonical_json_bytes(result))
    print(
        f"cases={result['case_count']} findings={result['finding_count']} "
        f"raw_dice={result['metrics']['raw_evaluator']['mean_global_dice_per_finding']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
