#!/usr/bin/env python3
"""Compare paired collaborator attention variants with the 20-model envelope."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

from analyze_candidates import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
DEFAULT_COLLABORATOR_MANIFEST = HERE / "collaborator_attention_manifest.json"
DEFAULT_BASELINE_SUMMARY = (
    HERE / "outputs" / "baseline_per_category_performance.json"
)
DEFAULT_OUTPUT_JSON = (
    HERE / "outputs" / "collaborator_attention_vs_20model_variance.json"
)
DEFAULT_OUTPUT_MD = (
    HERE / "outputs" / "collaborator_attention_vs_20model_variance.md"
)

CATEGORY_ORDER = (
    "1a",
    "1b",
    "1c",
    "1d",
    "1e",
    "1f",
    "2a",
    "2b",
    "2c",
    "2d",
    "2e",
    "2f",
    "2g",
    "2h",
)
REPRESENTED_CATEGORY_ORDER = tuple(
    code for code in CATEGORY_ORDER if code != "2f"
)
VARIANT_ORDER = ("baseline", "all_category", "strict_2b2c")
ATTENTION_VARIANTS = ("all_category", "strict_2b2c")
VARIANT_LABELS = {
    "baseline": "Baseline",
    "all_category": "All-category",
    "strict_2b2c": "Strict-2b2c",
}
EXPECTED_MODELS = 20
EXPECTED_FINDINGS = 381
EXPECTED_CASES = 200
EXPECTED_TRIPLETS = 1140
EXPECTED_PAIRS = 190
HIGH_SUPPORT_MINIMUM = 20
FLOAT_TOLERANCE = 1e-12
FOUR_DECIMAL_PATTERN = re.compile(r"^0\.\d{4}$")
SIGNED_FOUR_DECIMAL_PATTERN = re.compile(r"^[+-]0\.\d{4}$")

EXP009_BASELINE_ID = "exp009_baseline_e100"
EXP009_ATTENTION_IDS = (
    "exp009_s3v1_e100",
    "exp009_s3v2_e100",
    "exp009_s3v3_e100",
)
EXP007_SNAPSHOT_IDS = (
    "exp007_ddp_e050",
    "exp007_ddp_e075",
    "exp007_ddp_e100",
)
EXP008_SNAPSHOT_ARMS = {
    "shared": ("exp008_shared_e080", "exp008_shared_e100"),
    "dual": ("exp008_dual_e080", "exp008_dual_e100"),
    "precision": ("exp008_precision_e080", "exp008_precision_e100"),
    "joint": ("exp008_joint_e080", "exp008_joint_e100"),
}


def quantile_linear(values: Sequence[float], probability: float) -> float:
    """Return an R-7/NumPy-style linearly interpolated quantile."""
    if not values:
        raise ValueError("cannot calculate a quantile from no values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"invalid quantile probability {probability}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe_values(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("cannot describe no values")
    median = statistics.median(data)
    q1 = quantile_linear(data, 0.25)
    q3 = quantile_linear(data, 0.75)
    minimum = min(data)
    maximum = max(data)
    sample_variance = statistics.variance(data) if len(data) > 1 else None
    sample_sd = statistics.stdev(data) if len(data) > 1 else None
    return {
        "count": len(data),
        "mean": math.fsum(data) / len(data),
        "sample_variance": sample_variance,
        "sample_sd": sample_sd,
        "minimum": minimum,
        "q1": q1,
        "median": median,
        "q3": q3,
        "maximum": maximum,
        "range": maximum - minimum,
        "iqr": q3 - q1,
        "mad": statistics.median(abs(value - median) for value in data),
    }


def empirical_midrank_percentile(
    value: float,
    reference: Sequence[float],
    *,
    tolerance: float = FLOAT_TOLERANCE,
) -> dict[str, Any]:
    """Place a value in a reference distribution using a tie-aware midrank."""
    if not reference:
        raise ValueError("cannot rank against an empty reference distribution")
    less = sum(candidate < value - tolerance for candidate in reference)
    equal = sum(abs(candidate - value) <= tolerance for candidate in reference)
    greater = len(reference) - less - equal
    return {
        "value": float(value),
        "reference_count": len(reference),
        "less": less,
        "equal": equal,
        "greater": greater,
        "percentile": 100.0 * (less + 0.5 * equal) / len(reference),
        "method": "100 * (count_less + 0.5 * count_equal) / reference_count",
    }


def pearson_correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Pearson vectors must have equal length of at least two")
    first_mean = math.fsum(first) / len(first)
    second_mean = math.fsum(second) / len(second)
    first_centered = [value - first_mean for value in first]
    second_centered = [value - second_mean for value in second]
    numerator = math.fsum(
        left * right for left, right in zip(first_centered, second_centered)
    )
    denominator = math.sqrt(
        math.fsum(value * value for value in first_centered)
        * math.fsum(value * value for value in second_centered)
    )
    if denominator == 0.0:
        return None
    return numerator / denominator


def sign(value: float) -> int:
    if value > FLOAT_TOLERANCE:
        return 1
    if value < -FLOAT_TOLERANCE:
        return -1
    return 0


def sign_comparison(
    first: Sequence[float],
    second: Sequence[float],
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("sign vectors must be aligned")
    pairs = [(sign(left), sign(right)) for left, right in zip(first, second)]
    same_nonzero = sum(left == right and left != 0 for left, right in pairs)
    opposite = sum(left * right < 0 for left, right in pairs)
    any_zero = len(pairs) - same_nonzero - opposite
    return {
        "categories": len(pairs),
        "same_nonzero_direction": same_nonzero,
        "opposite_direction": opposite,
        "at_least_one_zero": any_zero,
        "same_direction_fraction": same_nonzero / len(pairs),
        "pearson_correlation": pearson_correlation(first, second),
    }


def weighted_rms(deltas: dict[str, float], supports: dict[str, int]) -> float:
    denominator = sum(supports[code] for code in REPRESENTED_CATEGORY_ORDER)
    return math.sqrt(
        math.fsum(
            supports[code] * deltas[code] * deltas[code]
            for code in REPRESENTED_CATEGORY_ORDER
        )
        / denominator
    )


def macro_rms(deltas: dict[str, float]) -> float:
    return math.sqrt(
        math.fsum(
            deltas[code] * deltas[code]
            for code in REPRESENTED_CATEGORY_ORDER
        )
        / len(REPRESENTED_CATEGORY_ORDER)
    )


def validate_baseline_summary(
    baseline: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if baseline.get("counts", {}).get("models") != EXPECTED_MODELS:
        raise ValueError("baseline summary must contain exactly 20 models")
    if baseline.get("counts", {}).get("cases_per_model") != EXPECTED_CASES:
        raise ValueError("baseline summary must contain 200 cases per model")
    if baseline.get("counts", {}).get("findings_per_model") != EXPECTED_FINDINGS:
        raise ValueError("baseline summary must contain 381 findings per model")
    if float(baseline.get("definitions", {}).get("mask_threshold", math.nan)) != 0.5:
        raise ValueError("baseline summary mask threshold must be 0.5")
    categories = baseline.get("categories")
    counts = baseline.get("category_counts")
    models = baseline.get("models")
    if not isinstance(categories, dict) or tuple(categories) != CATEGORY_ORDER:
        raise ValueError("baseline summary category order/coverage is invalid")
    if not isinstance(counts, dict) or tuple(counts) != CATEGORY_ORDER:
        raise ValueError("baseline summary category supports are invalid")
    supports = {code: int(counts[code]) for code in CATEGORY_ORDER}
    if sum(supports.values()) != EXPECTED_FINDINGS or supports["2f"] != 0:
        raise ValueError("baseline category supports must sum to 381 with 2f zero")
    if not isinstance(models, list) or len(models) != EXPECTED_MODELS:
        raise ValueError("baseline summary model records are invalid")
    model_ids = [model.get("candidate_id") for model in models]
    if len(set(model_ids)) != EXPECTED_MODELS:
        raise ValueError("baseline summary has duplicate candidate IDs")
    for model in models:
        model_categories = model.get("categories")
        if not isinstance(model_categories, dict) or tuple(model_categories) != CATEGORY_ORDER:
            raise ValueError(f"{model.get('candidate_id')}: incomplete categories")
        recomposed = math.fsum(
            float(model_categories[code]["dice"]) * supports[code]
            for code in REPRESENTED_CATEGORY_ORDER
        ) / EXPECTED_FINDINGS
        if not math.isclose(
            recomposed,
            float(model["overall"]["dice"]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                f"{model['candidate_id']}: category Dice does not recompose"
            )
    return sorted(models, key=lambda model: str(model["candidate_id"])), supports


def load_and_validate_inputs(
    collaborator_manifest_path: Path,
    baseline_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    manifest = read_json(collaborator_manifest_path)
    baseline_sha256 = sha256_file(baseline_summary_path)
    expected_sha256 = manifest.get("baseline_per_category_sha256")
    if baseline_sha256 != expected_sha256:
        raise ValueError(
            f"{baseline_summary_path}: baseline summary SHA256 drifted; expected "
            f"{expected_sha256}, got {baseline_sha256}"
        )
    baseline = read_json(baseline_summary_path)
    models, supports = validate_baseline_summary(baseline)

    if manifest.get("schema_version") != 1:
        raise ValueError("collaborator manifest schema_version must be 1")
    contract = manifest.get("evaluation_contract", {})
    if (
        contract.get("cases") != EXPECTED_CASES
        or contract.get("findings") != EXPECTED_FINDINGS
        or float(contract.get("mask_threshold", math.nan)) != 0.5
        or contract.get("dice_aggregation")
        != "unweighted mean across findings"
    ):
        raise ValueError("collaborator evaluation contract is invalid")
    paired = manifest.get("paired_training_declaration", {})
    if not paired.get("attention_configuration_is_only_intended_intervention"):
        raise ValueError("attention must be declared as the only intended change")
    if any(
        paired.get(field) != "paired"
        for field in (
            "initialization",
            "seed",
            "data_order",
            "schedule_and_settings",
            "evaluation",
        )
    ):
        raise ValueError("collaborator training/evaluation declaration is not paired")
    variant_ids = [variant.get("id") for variant in manifest.get("variants", [])]
    if tuple(variant_ids) != VARIANT_ORDER:
        raise ValueError("collaborator variants are missing or out of order")
    rows = manifest.get("categories")
    if not isinstance(rows, list):
        raise ValueError("collaborator categories must be a list")
    if tuple(row.get("code") for row in rows) != REPRESENTED_CATEGORY_ORDER:
        raise ValueError("collaborator manifest must contain the exact 13 categories")
    unavailable = manifest.get("unavailable_categories")
    if (
        not isinstance(unavailable, list)
        or len(unavailable) != 1
        or unavailable[0].get("code") != "2f"
        or int(unavailable[0].get("support", -1)) != 0
    ):
        raise ValueError("collaborator manifest must explicitly mark 2f unavailable")
    if sum(int(row.get("support", -1)) for row in rows) != EXPECTED_FINDINGS:
        raise ValueError("collaborator supports must sum to 381")
    for row in rows:
        code = row["code"]
        if int(row.get("support", -1)) != supports[code]:
            raise ValueError(f"collaborator/baseline support mismatch for {code}")
        if row.get("official_label") != baseline["categories"][code]["label"]:
            raise ValueError(f"collaborator/baseline label mismatch for {code}")
        scores = row.get("scores", {})
        deltas = row.get("reported_deltas", {})
        if tuple(scores) != ("all_category", "baseline", "strict_2b2c"):
            raise ValueError(f"{code}: score variants are invalid")
        for variant in VARIANT_ORDER:
            if not FOUR_DECIMAL_PATTERN.fullmatch(str(scores.get(variant, ""))):
                raise ValueError(f"{code} {variant}: score must preserve four decimals")
        for variant in ATTENTION_VARIANTS:
            if not SIGNED_FOUR_DECIMAL_PATTERN.fullmatch(
                str(deltas.get(variant, ""))
            ):
                raise ValueError(
                    f"{code} {variant}: delta must preserve sign and four decimals"
                )
    return manifest, baseline, models, supports


def transcription_delta_checks(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    tolerance = Decimal(
        str(manifest["rounding_contract"]["delta_discrepancy_tolerance"])
    )
    checks: dict[str, dict[str, Any]] = {}
    flagged: list[dict[str, Any]] = []
    for row in manifest["categories"]:
        code = row["code"]
        baseline = Decimal(row["scores"]["baseline"])
        checks[code] = {}
        for variant in ATTENTION_VARIANTS:
            recomputed = Decimal(row["scores"][variant]) - baseline
            reported = Decimal(row["reported_deltas"][variant])
            discrepancy = recomputed - reported
            within = abs(discrepancy) <= tolerance
            result = {
                "reported_delta": float(reported),
                "recomputed_from_displayed_scores": float(recomputed),
                "discrepancy": float(discrepancy),
                "within_rounding_tolerance": within,
            }
            checks[code][variant] = result
            if not within:
                flagged.append({"category": code, "variant": variant, **result})
    return checks, flagged


def model_lookup(models: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(model["candidate_id"]): model for model in models}


def evidence_label(
    support: int,
    first_delta: float,
    second_delta: float,
    triplet_percentile: float,
    first_pair_percentile: float,
    second_pair_percentile: float,
) -> str:
    if support < HIGH_SUPPORT_MINIMUM:
        return "low_support_no_attribution"
    same_direction = sign(first_delta) == sign(second_delta) != 0
    is_outlier = (
        same_direction
        and triplet_percentile >= 95.0
        and max(first_pair_percentile, second_pair_percentile) >= 95.0
    )
    if is_outlier:
        return "attention_associated_outlier"
    if same_direction and first_delta < 0.0:
        return "possible_attention_tradeoff_not_outlier"
    if same_direction and first_delta > 0.0:
        return "consistent_positive_not_outlier"
    return "mixed_direction_not_outlier"


def build_category_comparison(
    manifest: dict[str, Any],
    baseline: dict[str, Any],
    models: Sequence[dict[str, Any]],
    supports: dict[str, int],
) -> dict[str, dict[str, Any]]:
    manifest_by_code = {row["code"]: row for row in manifest["categories"]}
    results: dict[str, dict[str, Any]] = {}
    for code in CATEGORY_ORDER:
        if code == "2f":
            results[code] = {
                "available": False,
                "label": baseline["categories"][code]["label"],
                "support": 0,
                "evidence_label": "unavailable",
            }
            continue
        row = manifest_by_code[code]
        collaborator_scores = {
            variant: float(row["scores"][variant]) for variant in VARIANT_ORDER
        }
        collaborator_deltas = {
            variant: collaborator_scores[variant] - collaborator_scores["baseline"]
            for variant in ATTENTION_VARIANTS
        }
        collaborator_statistics = describe_values(collaborator_scores.values())
        best_variant = min(
            VARIANT_ORDER,
            key=lambda variant: (-collaborator_scores[variant], VARIANT_ORDER.index(variant)),
        )
        worst_variant = min(
            VARIANT_ORDER,
            key=lambda variant: (collaborator_scores[variant], VARIANT_ORDER.index(variant)),
        )
        collaborator_statistics.update(
            {
                "best_variant": best_variant,
                "best_score": collaborator_scores[best_variant],
                "worst_variant": worst_variant,
                "worst_score": collaborator_scores[worst_variant],
            }
        )

        twenty_values = [float(model["categories"][code]["dice"]) for model in models]
        twenty_statistics = describe_values(twenty_values)
        triplet_spreads = [
            max(values) - min(values)
            for values in itertools.combinations(twenty_values, 3)
        ]
        pairwise_differences = [
            abs(first - second)
            for first, second in itertools.combinations(twenty_values, 2)
        ]
        if len(triplet_spreads) != EXPECTED_TRIPLETS:
            raise AssertionError(f"{code}: expected 1,140 triplet spreads")
        if len(pairwise_differences) != EXPECTED_PAIRS:
            raise AssertionError(f"{code}: expected 190 pairwise differences")
        triplet_position = empirical_midrank_percentile(
            collaborator_statistics["range"], triplet_spreads
        )
        delta_positions = {
            variant: empirical_midrank_percentile(
                abs(collaborator_deltas[variant]), pairwise_differences
            )
            for variant in ATTENTION_VARIANTS
        }
        label = evidence_label(
            supports[code],
            collaborator_deltas["all_category"],
            collaborator_deltas["strict_2b2c"],
            triplet_position["percentile"],
            delta_positions["all_category"]["percentile"],
            delta_positions["strict_2b2c"]["percentile"],
        )
        results[code] = {
            "available": True,
            "label": baseline["categories"][code]["label"],
            "support": supports[code],
            "support_group": (
                "high_support" if supports[code] >= HIGH_SUPPORT_MINIMUM else "low_support"
            ),
            "collaborator": {
                "scores": collaborator_scores,
                "deltas_from_baseline": collaborator_deltas,
                "reported_deltas": {
                    variant: float(row["reported_deltas"][variant])
                    for variant in ATTENTION_VARIANTS
                },
                "displayed_best": row["displayed_best"],
                "statistics": collaborator_statistics,
            },
            "twenty_model_envelope": {
                "statistics": twenty_statistics,
                "values": [
                    {
                        "candidate_id": model["candidate_id"],
                        "dice": float(model["categories"][code]["dice"]),
                    }
                    for model in models
                ],
            },
            "range_ratio": (
                collaborator_statistics["range"] / twenty_statistics["range"]
                if twenty_statistics["range"] > 0.0
                else None
            ),
            "matched_triplet_spread_reference": {
                "distribution": describe_values(triplet_spreads),
                "position": triplet_position,
            },
            "pairwise_absolute_delta_reference": {
                "distribution": describe_values(pairwise_differences),
                "positions": delta_positions,
            },
            "same_attention_direction": (
                sign(collaborator_deltas["all_category"])
                == sign(collaborator_deltas["strict_2b2c"])
                != 0
            ),
            "evidence_label": label,
        }
    return results


def build_exp009_context(
    models: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    lookup = model_lookup(models)
    required = (EXP009_BASELINE_ID, *EXP009_ATTENTION_IDS)
    missing = sorted(set(required) - set(lookup))
    if missing:
        raise ValueError(f"missing Exp009 context models: {missing}")
    categories: dict[str, Any] = {}
    for code in REPRESENTED_CATEGORY_ORDER:
        baseline_score = float(lookup[EXP009_BASELINE_ID]["categories"][code]["dice"])
        variants = {}
        values = [baseline_score]
        for candidate_id in EXP009_ATTENTION_IDS:
            score = float(lookup[candidate_id]["categories"][code]["dice"])
            values.append(score)
            variants[candidate_id] = {
                "score": score,
                "delta_from_exp009_baseline": score - baseline_score,
            }
        categories[code] = {
            "baseline_score": baseline_score,
            "attention_variants": variants,
            "four_model_range": max(values) - min(values),
        }
    baseline_overall = float(lookup[EXP009_BASELINE_ID]["overall"]["dice"])
    return {
        "interpretation": (
            "Method-ablation context only; these arms do not estimate random-seed variance."
        ),
        "baseline_candidate_id": EXP009_BASELINE_ID,
        "attention_candidate_ids": list(EXP009_ATTENTION_IDS),
        "overall": {
            "baseline_score": baseline_overall,
            "attention_variants": {
                candidate_id: {
                    "score": float(lookup[candidate_id]["overall"]["dice"]),
                    "delta_from_exp009_baseline": (
                        float(lookup[candidate_id]["overall"]["dice"])
                        - baseline_overall
                    ),
                }
                for candidate_id in EXP009_ATTENTION_IDS
            },
        },
        "categories": categories,
    }


def build_trajectory_context(
    models: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    lookup = model_lookup(models)
    required = set(EXP007_SNAPSHOT_IDS)
    for pair in EXP008_SNAPSHOT_ARMS.values():
        required.update(pair)
    missing = sorted(required - set(lookup))
    if missing:
        raise ValueError(f"missing trajectory context models: {missing}")
    categories: dict[str, Any] = {}
    for code in REPRESENTED_CATEGORY_ORDER:
        exp007_scores = {
            candidate_id: float(lookup[candidate_id]["categories"][code]["dice"])
            for candidate_id in EXP007_SNAPSHOT_IDS
        }
        exp008_arms = {}
        for arm, (epoch80_id, epoch100_id) in EXP008_SNAPSHOT_ARMS.items():
            epoch80 = float(lookup[epoch80_id]["categories"][code]["dice"])
            epoch100 = float(lookup[epoch100_id]["categories"][code]["dice"])
            exp008_arms[arm] = {
                "epoch80_candidate_id": epoch80_id,
                "epoch100_candidate_id": epoch100_id,
                "epoch80_score": epoch80,
                "epoch100_score": epoch100,
                "delta_epoch100_minus_epoch80": epoch100 - epoch80,
                "absolute_delta": abs(epoch100 - epoch80),
            }
        abs_deltas = [value["absolute_delta"] for value in exp008_arms.values()]
        categories[code] = {
            "exp007": {
                "scores": exp007_scores,
                "trajectory_range": max(exp007_scores.values()) - min(exp007_scores.values()),
                "delta_e100_minus_e050": (
                    exp007_scores["exp007_ddp_e100"]
                    - exp007_scores["exp007_ddp_e050"]
                ),
            },
            "exp008": {
                "arms": exp008_arms,
                "absolute_snapshot_delta_summary": describe_values(abs_deltas),
            },
        }
    return {
        "interpretation": (
            "Within-lineage optimization-trajectory context; repeated snapshots are "
            "dependent and do not estimate random-seed variance."
        ),
        "exp007_candidate_ids": list(EXP007_SNAPSHOT_IDS),
        "exp008_arms": {
            arm: list(pair) for arm, pair in EXP008_SNAPSHOT_ARMS.items()
        },
        "categories": categories,
    }


def build_cross_category_analysis(
    category_results: dict[str, dict[str, Any]],
    models: Sequence[dict[str, Any]],
    supports: dict[str, int],
) -> dict[str, Any]:
    collaborator_deltas = {
        variant: {
            code: float(
                category_results[code]["collaborator"]["deltas_from_baseline"][variant]
            )
            for code in REPRESENTED_CATEGORY_ORDER
        }
        for variant in ATTENTION_VARIANTS
    }
    pair_vectors = []
    for first, second in itertools.combinations(models, 2):
        deltas = {
            code: float(second["categories"][code]["dice"])
            - float(first["categories"][code]["dice"])
            for code in REPRESENTED_CATEGORY_ORDER
        }
        pair_vectors.append(
            {
                "first_candidate_id": first["candidate_id"],
                "second_candidate_id": second["candidate_id"],
                "support_weighted_rms": weighted_rms(deltas, supports),
                "macro_rms": macro_rms(deltas),
                "absolute_overall_dice_difference": abs(
                    float(second["overall"]["dice"])
                    - float(first["overall"]["dice"])
                ),
            }
        )
    if len(pair_vectors) != EXPECTED_PAIRS:
        raise AssertionError("expected 190 cross-category model-pair vectors")
    reference_values = {
        metric: [float(pair[metric]) for pair in pair_vectors]
        for metric in (
            "support_weighted_rms",
            "macro_rms",
            "absolute_overall_dice_difference",
        )
    }
    variant_results: dict[str, Any] = {}
    baseline_overall = math.fsum(
        supports[code]
        * float(category_results[code]["collaborator"]["scores"]["baseline"])
        for code in REPRESENTED_CATEGORY_ORDER
    ) / EXPECTED_FINDINGS
    for variant in ATTENTION_VARIANTS:
        deltas = collaborator_deltas[variant]
        weighted_value = weighted_rms(deltas, supports)
        macro_value = macro_rms(deltas)
        variant_overall = math.fsum(
            supports[code]
            * float(category_results[code]["collaborator"]["scores"][variant])
            for code in REPRESENTED_CATEGORY_ORDER
        ) / EXPECTED_FINDINGS
        overall_delta = variant_overall - baseline_overall
        variant_results[variant] = {
            "support_weighted_rms": {
                "value": weighted_value,
                "position": empirical_midrank_percentile(
                    weighted_value, reference_values["support_weighted_rms"]
                ),
            },
            "macro_rms": {
                "value": macro_value,
                "position": empirical_midrank_percentile(
                    macro_value, reference_values["macro_rms"]
                ),
            },
            "recomposed_overall_delta": {
                "value": overall_delta,
                "absolute_value": abs(overall_delta),
                "position": empirical_midrank_percentile(
                    abs(overall_delta),
                    reference_values["absolute_overall_dice_difference"],
                ),
            },
        }
    all_codes = list(REPRESENTED_CATEGORY_ORDER)
    high_codes = [code for code in all_codes if supports[code] >= HIGH_SUPPORT_MINIMUM]
    first_all = [collaborator_deltas["all_category"][code] for code in all_codes]
    second_all = [collaborator_deltas["strict_2b2c"][code] for code in all_codes]
    first_high = [collaborator_deltas["all_category"][code] for code in high_codes]
    second_high = [collaborator_deltas["strict_2b2c"][code] for code in high_codes]
    return {
        "model_pair_count": len(pair_vectors),
        "reference_distributions": {
            metric: describe_values(values) for metric, values in reference_values.items()
        },
        "variants": variant_results,
        "attention_variant_agreement": {
            "all_represented_categories": sign_comparison(first_all, second_all),
            "high_support_categories": {
                "codes": high_codes,
                **sign_comparison(first_high, second_high),
            },
        },
    }


def recompose_collaborator_overall(
    category_results: dict[str, dict[str, Any]],
    supports: dict[str, int],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for variant in VARIANT_ORDER:
        weighted = math.fsum(
            supports[code]
            * float(category_results[code]["collaborator"]["scores"][variant])
            for code in REPRESENTED_CATEGORY_ORDER
        ) / EXPECTED_FINDINGS
        macro = math.fsum(
            float(category_results[code]["collaborator"]["scores"][variant])
            for code in REPRESENTED_CATEGORY_ORDER
        ) / len(REPRESENTED_CATEGORY_ORDER)
        results[variant] = {
            "support_weighted_dice": weighted,
            "macro_category_dice": macro,
        }
    baseline_weighted = results["baseline"]["support_weighted_dice"]
    baseline_macro = results["baseline"]["macro_category_dice"]
    for variant in VARIANT_ORDER:
        results[variant]["support_weighted_delta_from_baseline"] = (
            results[variant]["support_weighted_dice"] - baseline_weighted
        )
        results[variant]["macro_delta_from_baseline"] = (
            results[variant]["macro_category_dice"] - baseline_macro
        )
    return results


def build_evidence_assessment(
    categories: dict[str, dict[str, Any]],
    cross_category: dict[str, Any],
) -> dict[str, Any]:
    attention_outliers = [
        code
        for code in REPRESENTED_CATEGORY_ORDER
        if categories[code]["evidence_label"] == "attention_associated_outlier"
    ]
    tradeoffs = [
        code
        for code in REPRESENTED_CATEGORY_ORDER
        if categories[code]["evidence_label"]
        == "possible_attention_tradeoff_not_outlier"
    ]
    overall_percentiles = {
        variant: cross_category["variants"][variant][
            "recomposed_overall_delta"
        ]["position"]["percentile"]
        for variant in ATTENTION_VARIANTS
    }
    return {
        "attention_associated_outlier_categories": attention_outliers,
        "possible_tradeoff_categories": tradeoffs,
        "broad_attention_benefit_supported": False,
        "central_interpretation": [
            (
                "Category 2b is the strongest targeted attention-associated signal: "
                "both paired attention variants improve it and its size-matched spread "
                "is unusually high relative to the 20-model envelope."
            ),
            (
                "Category 2d moves downward for both attention variants, but its "
                "magnitude is not exceptional; it is a possible attention tradeoff, "
                "not strong outlier evidence."
            ),
            (
                "Most other category changes are mixed or remain inside the empirical "
                "20-model variation envelope."
            ),
            (
                "The recomposed overall shifts are ordinary relative to the 190 "
                "20-model pairs, so these results do not show a broad attention benefit."
            ),
            (
                "Low-support swings, especially 1f, 2g, and 2h, cannot support "
                "attention attribution."
            ),
        ],
        "overall_absolute_delta_percentiles": overall_percentiles,
    }


def canonical_result_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_summary(
    collaborator_manifest_path: Path,
    baseline_summary_path: Path,
) -> dict[str, Any]:
    manifest, baseline, models, supports = load_and_validate_inputs(
        collaborator_manifest_path,
        baseline_summary_path,
    )
    delta_checks, flagged_deltas = transcription_delta_checks(manifest)
    categories = build_category_comparison(
        manifest,
        baseline,
        models,
        supports,
    )
    collaborator_overall = recompose_collaborator_overall(categories, supports)
    cross_category = build_cross_category_analysis(categories, models, supports)
    exp009_context = build_exp009_context(models)
    trajectory_context = build_trajectory_context(models)
    evidence = build_evidence_assessment(categories, cross_category)
    warnings = [
        (
            "The 20-model pool is a heterogeneous empirical envelope, not a "
            "random-seed variance estimate."
        ),
        (
            "Matched-triplet and model-pair percentiles are descriptive reference "
            "positions, not p-values or confidence intervals."
        ),
        (
            "No statistical-causality claim is supported without repeated matched "
            "seeds or collaborator per-finding outputs."
        ),
        (
            "Low-support categories are unstable; 2g has one finding, and 2f is "
            "unavailable."
        ),
    ]
    if flagged_deltas:
        warnings.append(
            "One or more reported deltas differ from displayed-score subtraction "
            "beyond the declared rounding tolerance; see transcription checks."
        )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "collaborator_attention_variants_vs_20model_variation",
        "definitions": {
            "collaborator_variance": "sample variance across the three paired variants",
            "twenty_model_variance": "sample variance across 20 heterogeneous models",
            "iqr": "R-7 linear-interpolation Q3 minus Q1",
            "mad": "median absolute deviation from the median",
            "range": "maximum minus minimum",
            "matched_triplet_reference": (
                "all C(20,3)=1,140 max-minus-min spreads per category"
            ),
            "pairwise_delta_reference": (
                "all C(20,2)=190 absolute Dice differences per category"
            ),
            "percentile": (
                "tie-aware empirical midrank: 100*(less + 0.5*equal)/count"
            ),
            "support_weighted_delta_magnitude": (
                "root mean square category delta weighted by finding support"
            ),
            "macro_delta_magnitude": "unweighted root mean square category delta",
            "high_support_minimum": HIGH_SUPPORT_MINIMUM,
            "attention_associated_outlier_rule": (
                "N>=20; both attention variants move in the same nonzero direction; "
                "matched-triplet spread percentile>=95; and at least one absolute "
                "baseline-delta percentile>=95"
            ),
        },
        "sources": {
            "collaborator_manifest": str(collaborator_manifest_path),
            "collaborator_manifest_sha256": sha256_file(collaborator_manifest_path),
            "baseline_per_category_summary": str(baseline_summary_path),
            "baseline_per_category_summary_sha256": sha256_file(baseline_summary_path),
            "baseline_candidate_manifest": baseline["sources"]["candidate_manifest"],
            "baseline_candidate_manifest_sha256": baseline["sources"][
                "candidate_manifest_sha256"
            ],
            "dataset_json": baseline["sources"]["dataset_json"],
            "dataset_sha256": baseline["sources"]["dataset_sha256"],
        },
        "evaluation_contract": manifest["evaluation_contract"],
        "paired_training_declaration": manifest["paired_training_declaration"],
        "counts": {
            "collaborator_variants": len(VARIANT_ORDER),
            "twenty_model_candidates": len(models),
            "cases": EXPECTED_CASES,
            "findings": EXPECTED_FINDINGS,
            "official_categories": len(CATEGORY_ORDER),
            "represented_categories": len(REPRESENTED_CATEGORY_ORDER),
            "matched_triplets_per_category": EXPECTED_TRIPLETS,
            "model_pairs": EXPECTED_PAIRS,
        },
        "category_supports": supports,
        "transcription": {
            "variants": manifest["variants"],
            "rows": manifest["categories"],
            "unavailable_categories": manifest["unavailable_categories"],
            "delta_checks": delta_checks,
            "flagged_delta_discrepancies": flagged_deltas,
        },
        "collaborator_recomposed_overall": collaborator_overall,
        "categories": categories,
        "cross_category": cross_category,
        "secondary_context": {
            "exp009_attention_ablation": exp009_context,
            "within_lineage_trajectory": trajectory_context,
        },
        "evidence_assessment": evidence,
        "validation": {
            "exact_13_represented_categories": (
                tuple(row["code"] for row in manifest["categories"])
                == REPRESENTED_CATEGORY_ORDER
            ),
            "category_supports_sum_to_381": (
                sum(supports.values()) == EXPECTED_FINDINGS
            ),
            "category_2f_explicitly_unavailable": (
                supports["2f"] == 0 and not categories["2f"]["available"]
            ),
            "baseline_model_count_is_20": len(models) == EXPECTED_MODELS,
            "triplet_distribution_count_is_1140": all(
                categories[code]["matched_triplet_spread_reference"]["distribution"][
                    "count"
                ]
                == EXPECTED_TRIPLETS
                for code in REPRESENTED_CATEGORY_ORDER
            ),
            "pair_distribution_count_is_190": all(
                categories[code]["pairwise_absolute_delta_reference"]["distribution"][
                    "count"
                ]
                == EXPECTED_PAIRS
                for code in REPRESENTED_CATEGORY_ORDER
            ),
            "all_reported_deltas_within_rounding_tolerance": not flagged_deltas,
        },
        "warnings": warnings,
    }
    summary["result_sha256"] = canonical_result_hash(summary)
    return summary


def format_signed(value: float, decimals: int = 6) -> str:
    return f"{value:+.{decimals}f}"


def format_optional(value: float | None, decimals: int = 6) -> str:
    return "—" if value is None else f"{value:.{decimals}f}"


def human_evidence_label(value: str) -> str:
    return {
        "attention_associated_outlier": "Attention-associated outlier",
        "possible_attention_tradeoff_not_outlier": "Possible tradeoff; not outlier",
        "consistent_positive_not_outlier": "Consistent positive; not outlier",
        "mixed_direction_not_outlier": "Mixed direction; not outlier",
        "low_support_no_attribution": "Low support; no attribution",
        "unavailable": "Unavailable",
    }[value]


def build_markdown(summary: dict[str, Any]) -> str:
    categories = summary["categories"]
    lines = [
        "# Collaborator Attention Variants Versus 20-Model Variation",
        "",
        "The collaborator's three runs are a paired comparison on the exact same "
        "val200 evaluation: initialization, seed, data order, schedule/settings, "
        "and evaluation are paired, with attention configuration as the only "
        "intended intervention. The 20-model reference is a heterogeneous empirical "
        "variation envelope, **not** a random-seed variance estimate.",
        "",
        "> Percentiles below are descriptive reference positions, not p-values or "
        "confidence intervals. Without repeated matched seeds or collaborator "
        "per-finding outputs, the analysis cannot establish statistical causality.",
        "",
        "## Take-Home Message",
        "",
        "Attention appears to **redistribute category performance rather than "
        "uniformly improve the model**. The strongest credible signal is a targeted "
        "improvement for `2b` (atelectasis/consolidation): both paired attention "
        "variants improve it, and the effect is unusually large relative to the "
        "20-model envelope. `2d` (pulmonary nodules/masses) moves consistently "
        "downward, suggesting a possible tradeoff, while most other changes are "
        "compatible with ordinary model variation. Rare-category swings are too "
        "unstable to attribute. Repeated matched seeds are still required for a "
        "causal claim.",
        "",
        "## Concise Evidence Assessment",
        "",
    ]
    for statement in summary["evidence_assessment"]["central_interpretation"]:
        lines.append(f"- {statement}")
    category_2b = categories["2b"]
    category_2d = categories["2d"]
    lines.extend(
        [
            "",
            f"The clearest result is `2b`: collaborator spread "
            f"`{category_2b['collaborator']['statistics']['range']:.6f}` is at the "
            f"`{category_2b['matched_triplet_spread_reference']['position']['percentile']:.1f}`th "
            "matched-triplet percentile, and both attention variants improve the "
            "baseline. In contrast, `2d` moves down in both variants but its spread "
            f"is only at the `{category_2d['matched_triplet_spread_reference']['position']['percentile']:.1f}`th "
            "percentile.",
            "",
            "## Evaluation and Category Support",
            "",
            "- Split: `rexgroundingct_val200_seed20260723` (200 cases, 381 findings).",
            "- Mask threshold: `0.5`; category/global Dice is the unweighted mean "
            "across findings.",
            f"- High-support interpretation threshold: `N >= {HIGH_SUPPORT_MINIMUM}`.",
            "- `2f` honeycombing is unavailable because val200 has no findings.",
            "",
            "| Code | Official category | N | Interpretation tier |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for code in CATEGORY_ORDER:
        category = categories[code]
        if not category["available"]:
            tier = "Unavailable"
        elif category["support"] >= HIGH_SUPPORT_MINIMUM:
            tier = "High support"
        else:
            tier = "Low support; no attribution"
        lines.append(
            f"| {code} | {category['label']} | {category['support']} | {tier} |"
        )

    lines.extend(
        [
            "",
            "## Collaborator Table Transcription",
            "",
            "Scores and reported deltas preserve the screenshot's four-decimal "
            "display. A recomputed delta may differ by `0.0001` because the source "
            "scores were rounded independently; the declared tolerance is `0.0002`.",
            "",
            "| Code | Category (N) | Baseline | All-category (reported / recomputed Δ) | "
            "Strict-2b2c (reported / recomputed Δ) | Displayed best | Check |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    transcription_rows = {
        row["code"]: row for row in summary["transcription"]["rows"]
    }
    for code in REPRESENTED_CATEGORY_ORDER:
        row = transcription_rows[code]
        checks = summary["transcription"]["delta_checks"][code]
        all_ok = all(
            checks[variant]["within_rounding_tolerance"]
            for variant in ATTENTION_VARIANTS
        )
        lines.append(
            f"| {code} | {row['display_label']} ({row['support']}) | "
            f"{row['scores']['baseline']} | {row['scores']['all_category']} "
            f"({row['reported_deltas']['all_category']} / "
            f"{checks['all_category']['recomputed_from_displayed_scores']:+.4f}) | "
            f"{row['scores']['strict_2b2c']} "
            f"({row['reported_deltas']['strict_2b2c']} / "
            f"{checks['strict_2b2c']['recomputed_from_displayed_scores']:+.4f}) | "
            f"{row['displayed_best']} | {'Pass' if all_ok else '**Review**'} |"
        )
    lines.append("| 2f | Honeycombing (0) | — | — | — | Unavailable | Pass |")

    lines.extend(
        [
            "",
            "## Recomposed Overall Scores",
            "",
            "These scores are recomposed from category means and supports; no "
            "separate collaborator global result was supplied.",
            "",
            "| Variant | Support-weighted Dice | Δ vs baseline | Macro category Dice | "
            "Macro Δ |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in VARIANT_ORDER:
        record = summary["collaborator_recomposed_overall"][variant]
        lines.append(
            f"| {VARIANT_LABELS[variant]} | "
            f"{record['support_weighted_dice']:.6f} | "
            f"{format_signed(record['support_weighted_delta_from_baseline'])} | "
            f"{record['macro_category_dice']:.6f} | "
            f"{format_signed(record['macro_delta_from_baseline'])} |"
        )

    lines.extend(
        [
            "",
            "## Full Category Variance Comparison",
            "",
            "`Triplet pct` compares the three collaborator scores' range with all "
            "1,140 three-model ranges from our pool. `A pct` and `S pct` compare "
            "each absolute attention-baseline delta with all 190 within-category "
            "20-model differences. Percentiles use tie-aware empirical midranks.",
            "",
            "| Code | N | B / A / S Dice | Collaborator mean ± sample SD | "
            "Collaborator range | Best / worst variant | Sample variance C / 20 | "
            "20-model mean ± sample SD | 20-model IQR / MAD | 20-model range | "
            "Range ratio | Triplet pct | A pct | S pct | Assessment |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for code in CATEGORY_ORDER:
        category = categories[code]
        if not category["available"]:
            lines.append(
                f"| {code} | 0 | — | — | — | — | — | — | — | — | — | — | — | "
                "— | Unavailable |"
            )
            continue
        collaborator = category["collaborator"]
        collab_stats = collaborator["statistics"]
        twenty_stats = category["twenty_model_envelope"]["statistics"]
        positions = category["pairwise_absolute_delta_reference"]["positions"]
        lines.append(
            f"| {code} | {category['support']} | "
            f"{collaborator['scores']['baseline']:.4f} / "
            f"{collaborator['scores']['all_category']:.4f} / "
            f"{collaborator['scores']['strict_2b2c']:.4f} | "
            f"{collab_stats['mean']:.6f} ± {collab_stats['sample_sd']:.6f} | "
            f"{collab_stats['range']:.6f} | "
            f"{VARIANT_LABELS[collab_stats['best_variant']]} / "
            f"{VARIANT_LABELS[collab_stats['worst_variant']]} | "
            f"{collab_stats['sample_variance']:.8f} / "
            f"{twenty_stats['sample_variance']:.8f} | "
            f"{twenty_stats['mean']:.6f} ± {twenty_stats['sample_sd']:.6f} | "
            f"{twenty_stats['iqr']:.6f} / {twenty_stats['mad']:.6f} | "
            f"{twenty_stats['range']:.6f} | "
            f"{format_optional(category['range_ratio'], 3)} | "
            f"{category['matched_triplet_spread_reference']['position']['percentile']:.1f} | "
            f"{positions['all_category']['percentile']:.1f} | "
            f"{positions['strict_2b2c']['percentile']:.1f} | "
            f"{human_evidence_label(category['evidence_label'])} |"
        )

    lines.extend(
        [
            "",
            "## High-Support Categories (N >= 20)",
            "",
        ]
    )
    for code in REPRESENTED_CATEGORY_ORDER:
        category = categories[code]
        if category["support"] < HIGH_SUPPORT_MINIMUM:
            continue
        deltas = category["collaborator"]["deltas_from_baseline"]
        triplet_pct = category["matched_triplet_spread_reference"]["position"][
            "percentile"
        ]
        lines.append(
            f"- `{code}` ({category['label']}, N={category['support']}): "
            f"All-category {deltas['all_category']:+.4f}, Strict-2b2c "
            f"{deltas['strict_2b2c']:+.4f}; matched-triplet percentile "
            f"{triplet_pct:.1f}. **{human_evidence_label(category['evidence_label'])}.**"
        )

    lines.extend(
        [
            "",
            "## Low-Support Categories",
            "",
            "No attention-attribution label is assigned below 20 findings. In "
            "particular, `2g` has one finding; `1f` has four; and `2h` has seven. "
            "Their large-looking swings can be dominated by one or a few examples.",
            "",
            "| Code | N | All-category Δ | Strict-2b2c Δ | Collaborator range | "
            "Matched-triplet pct |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for code in REPRESENTED_CATEGORY_ORDER:
        category = categories[code]
        if category["support"] >= HIGH_SUPPORT_MINIMUM:
            continue
        deltas = category["collaborator"]["deltas_from_baseline"]
        lines.append(
            f"| {code} | {category['support']} | {deltas['all_category']:+.6f} | "
            f"{deltas['strict_2b2c']:+.6f} | "
            f"{category['collaborator']['statistics']['range']:.6f} | "
            f"{category['matched_triplet_spread_reference']['position']['percentile']:.1f} |"
        )

    cross = summary["cross_category"]
    agreement_all = cross["attention_variant_agreement"]["all_represented_categories"]
    agreement_high = cross["attention_variant_agreement"]["high_support_categories"]
    lines.extend(
        [
            "",
            "## Cross-Category Direction and Magnitude",
            "",
            f"Across all 13 represented categories, the variants move in the same "
            f"nonzero direction for `{agreement_all['same_nonzero_direction']}/13` "
            f"categories (Pearson delta correlation "
            f"`{agreement_all['pearson_correlation']:.6f}`). Across high-support "
            f"categories `{', '.join(agreement_high['codes'])}`, they agree for "
            f"`{agreement_high['same_nonzero_direction']}/{agreement_high['categories']}` "
            f"(correlation `{agreement_high['pearson_correlation']:.6f}`).",
            "",
            "| Variant | Support-weighted RMS Δ | Pair percentile | Macro RMS Δ | "
            "Pair percentile | Absolute recomposed overall Δ | Pair percentile |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in ATTENTION_VARIANTS:
        record = cross["variants"][variant]
        lines.append(
            f"| {VARIANT_LABELS[variant]} | "
            f"{record['support_weighted_rms']['value']:.6f} | "
            f"{record['support_weighted_rms']['position']['percentile']:.1f} | "
            f"{record['macro_rms']['value']:.6f} | "
            f"{record['macro_rms']['position']['percentile']:.1f} | "
            f"{record['recomposed_overall_delta']['absolute_value']:.6f} | "
            f"{record['recomposed_overall_delta']['position']['percentile']:.1f} |"
        )

    exp009 = summary["secondary_context"]["exp009_attention_ablation"]
    lines.extend(
        [
            "",
            "## Secondary Context: Exp009 Attention Ablation",
            "",
            f"> {exp009['interpretation']}",
            "",
            "Values are category Dice changes from `exp009_baseline_e100` for the "
            "three S3 attention arms.",
            "",
            "| Code | Exp009 baseline | S3v1 Δ | S3v2 Δ | S3v3 Δ | Four-arm range |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for code in REPRESENTED_CATEGORY_ORDER:
        record = exp009["categories"][code]
        variants = record["attention_variants"]
        lines.append(
            f"| {code} | {record['baseline_score']:.6f} | "
            f"{variants['exp009_s3v1_e100']['delta_from_exp009_baseline']:+.6f} | "
            f"{variants['exp009_s3v2_e100']['delta_from_exp009_baseline']:+.6f} | "
            f"{variants['exp009_s3v3_e100']['delta_from_exp009_baseline']:+.6f} | "
            f"{record['four_model_range']:.6f} |"
        )

    trajectory = summary["secondary_context"]["within_lineage_trajectory"]
    lines.extend(
        [
            "",
            "## Secondary Context: Optimization Trajectories",
            "",
            f"> {trajectory['interpretation']}",
            "",
            "Exp007 reports the range over epochs 50/75/100. Exp008 cells are "
            "signed epoch-100 minus epoch-80 category Dice changes.",
            "",
            "| Code | Exp007 e50/e75/e100 range | Exp007 e100−e50 | "
            "Exp008 shared Δ | dual Δ | precision Δ | joint Δ |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for code in REPRESENTED_CATEGORY_ORDER:
        record = trajectory["categories"][code]
        exp007 = record["exp007"]
        arms = record["exp008"]["arms"]
        lines.append(
            f"| {code} | {exp007['trajectory_range']:.6f} | "
            f"{exp007['delta_e100_minus_e050']:+.6f} | "
            f"{arms['shared']['delta_epoch100_minus_epoch80']:+.6f} | "
            f"{arms['dual']['delta_epoch100_minus_epoch80']:+.6f} | "
            f"{arms['precision']['delta_epoch100_minus_epoch80']:+.6f} | "
            f"{arms['joint']['delta_epoch100_minus_epoch80']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- The paired collaborator design makes attention the primary intended "
            "intervention, so directionally consistent, unusually large effects are "
            "more suggestive than ordinary mixed changes.",
            "- The 20-model pool mixes experiments, architectures, normalization, "
            "and trajectory snapshots. It is an empirical plausibility envelope, "
            "not an estimate of independent-seed noise.",
            "- Exp009 is method-ablation context; Exp007/Exp008 snapshots are "
            "dependent optimization-trajectory context. Neither is random-seed "
            "variance.",
            "- Formal attribution would require repeated matched seeds, ideally "
            "with aligned per-finding collaborator outputs.",
            "",
            "## Provenance and Reproduction",
            "",
            f"- Collaborator manifest SHA256: `{summary['sources']['collaborator_manifest_sha256']}`",
            f"- 20-model baseline summary SHA256: `{summary['sources']['baseline_per_category_summary_sha256']}`",
            f"- Dataset SHA256: `{summary['sources']['dataset_sha256']}`",
            f"- Deterministic result SHA256: `{summary['result_sha256']}`",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 python \\",
            "  side_experiments/sideexp002_multimodel_ensemble_selection/compare_collaborator_attention_variance.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    collaborator_manifest_path: Path,
    baseline_summary_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    summary = build_summary(collaborator_manifest_path, baseline_summary_path)
    atomic_write_json(output_json, summary)
    atomic_write_text(output_markdown, build_markdown(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collaborator-manifest",
        type=Path,
        default=DEFAULT_COLLABORATOR_MANIFEST,
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY,
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()
    summary = run(
        args.collaborator_manifest,
        args.baseline_summary,
        args.output_json,
        args.output_markdown,
    )
    print(
        "Compared 3 paired collaborator variants with "
        f"{summary['counts']['twenty_model_candidates']} models across "
        f"{summary['counts']['represented_categories']} represented categories."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
