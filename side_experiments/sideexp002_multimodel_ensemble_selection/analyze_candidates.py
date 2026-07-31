#!/usr/bin/env python3
"""Validate the sideexp002 roster and regenerate hard-mask evidence reports."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MANIFEST_REPO_PREFIX = Path("/home/hengjie/code_sync/rexgroundingct")
DEFAULT_MANIFEST = HERE / "candidate_manifest.json"
DEFAULT_MODEL_REPORT = HERE / "outputs" / "candidate_model_selection.md"
DEFAULT_COMPLEMENTARITY_REPORT = HERE / "outputs" / "hard_mask_complementarity.md"
DEFAULT_COMPLEMENTARITY_JSON = (
    HERE / "outputs" / "hard_mask_complementarity_summary.json"
)
FROZEN_SOURCE_DIR = HERE / "frozen_sources"
EXPECTED_CASES = 200
EXPECTED_FINDINGS = 381
HIT_THRESHOLD = 0.1


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def resolve_source_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists() or not path.is_absolute():
        return path if path.is_absolute() else REPO_ROOT / path
    try:
        relative = path.relative_to(MANIFEST_REPO_PREFIX)
    except ValueError:
        return path
    return REPO_ROOT / relative


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )


def require_candidate_config(candidate: dict[str, Any]) -> Path:
    """Validate the declared config, with an immutable hash-named fallback."""
    expected = candidate["config"]["sha256"]
    declared = resolve_source_path(candidate["config"]["path"])
    if declared.is_file() and sha256_file(declared) == expected:
        return declared
    frozen = FROZEN_SOURCE_DIR / f"{expected}.json"
    if frozen.is_file():
        require_sha256(
            frozen,
            expected,
            f"{candidate['id']} frozen config",
        )
        return frozen
    require_sha256(declared, expected, f"{candidate['id']} config")
    raise AssertionError("unreachable")


def reproduction_decision(
    storage_passed: bool,
    historical_passed: bool,
) -> dict[str, Any]:
    """Make storage integrity authoritative and history comparison diagnostic."""
    warnings = []
    if not historical_passed:
        warnings.append("historical_reproduction_mismatch")
    return {
        "accepted": bool(storage_passed),
        "warnings": warnings,
        "historical_reproduction_is_gate": False,
    }


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 20:
        raise ValueError(f"{path}: expected exactly 20 candidates")
    ids = [candidate.get("id") for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate candidate IDs")
    if float(manifest.get("selection", {}).get("minimum_dice", math.nan)) != 0.32:
        raise ValueError(f"{path}: selection cutoff must be exactly 0.32")
    return manifest


def load_dataset_keys(path: Path) -> set[tuple[str, int]]:
    data = read_json(path)
    cases = data.get("test")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ValueError(f"{path}: expected {EXPECTED_CASES} cases under 'test'")
    keys: set[tuple[str, int]] = set()
    case_names: set[str] = set()
    for case in cases:
        name = case.get("name")
        findings = case.get("findings")
        categories = case.get("categories")
        if not isinstance(name, str) or name in case_names:
            raise ValueError(f"{path}: duplicate or invalid case {name!r}")
        case_names.add(name)
        if (
            not isinstance(findings, dict)
            or not isinstance(categories, dict)
            or set(findings) != set(categories)
        ):
            raise ValueError(f"{path} {name}: finding/category mismatch")
        for finding_index in findings:
            key = (name, int(finding_index))
            if key in keys:
                raise ValueError(f"{path}: duplicate finding {key}")
            keys.add(key)
    if len(keys) != EXPECTED_FINDINGS:
        raise ValueError(
            f"{path}: expected {EXPECTED_FINDINGS} findings, got {len(keys)}"
        )
    return keys


def flatten_eval(
    candidate: dict[str, Any],
    verify_hash: bool = True,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    eval_record = candidate["evaluation"]
    path = Path(eval_record["path"])
    if verify_hash:
        require_sha256(path, eval_record["sha256"], f"{candidate['id']} evaluation")
    data = read_json(path)
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ValueError(
            f"{candidate['id']}: expected {EXPECTED_CASES} cases, got "
            f"{len(cases) if isinstance(cases, list) else type(cases).__name__}"
        )
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    seen_cases: set[str] = set()
    dice_values: list[float] = []
    hits = 0
    for case in cases:
        name = case.get("file")
        if not isinstance(name, str) or name in seen_cases:
            raise ValueError(f"{candidate['id']}: duplicate or invalid case {name!r}")
        seen_cases.add(name)
        findings = case.get("findings")
        if not isinstance(findings, dict):
            raise ValueError(f"{candidate['id']} {name}: missing findings")
        for finding_key, metrics in findings.items():
            if not finding_key.startswith("finding_"):
                raise ValueError(
                    f"{candidate['id']} {name}: invalid finding key {finding_key!r}"
                )
            index = int(finding_key.removeprefix("finding_"))
            key = (name, index)
            if key in rows:
                raise ValueError(f"{candidate['id']}: duplicate finding {key}")
            dice = float(metrics["global_dice"])
            hit = bool(metrics["global_hit"])
            if not math.isfinite(dice) or hit != (dice >= HIT_THRESHOLD):
                raise ValueError(f"{candidate['id']} {key}: invalid Dice/hit pair")
            rows[key] = {"dice": dice, "hit": hit}
            dice_values.append(dice)
            hits += int(hit)
    if len(rows) != EXPECTED_FINDINGS:
        raise ValueError(
            f"{candidate['id']}: expected {EXPECTED_FINDINGS} findings, got {len(rows)}"
        )
    dice = math.fsum(dice_values) / len(dice_values)
    hit_rate = hits / len(dice_values)
    expected = candidate["performance"]
    if not math.isclose(dice, float(expected["dice"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{candidate['id']}: recomposed Dice {dice} != {expected['dice']}"
        )
    if hits != int(expected["hits"]):
        raise ValueError(f"{candidate['id']}: recomposed hits {hits} != {expected['hits']}")
    if not math.isclose(
        hit_rate,
        float(expected["hit_rate"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            f"{candidate['id']}: recomposed hit rate {hit_rate} != "
            f"{expected['hit_rate']}"
        )
    summary = {
        "cases": len(cases),
        "findings": len(rows),
        "dice": dice,
        "hits": hits,
        "hit_rate": hit_rate,
    }
    return rows, summary


def compare_pair(
    first_id: str,
    second_id: str,
    rows_by_id: dict[str, dict[tuple[str, int], dict[str, Any]]],
    summary_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    first = rows_by_id[first_id]
    second = rows_by_id[second_id]
    if first.keys() != second.keys():
        missing_first = sorted(second.keys() - first.keys())
        missing_second = sorted(first.keys() - second.keys())
        raise ValueError(
            f"{first_id}/{second_id}: finding keys differ; "
            f"missing_first={missing_first[:3]}, missing_second={missing_second[:3]}"
        )
    keys = sorted(first)
    first_dice = np.asarray([first[key]["dice"] for key in keys], dtype=np.float64)
    second_dice = np.asarray([second[key]["dice"] for key in keys], dtype=np.float64)
    first_hit = np.asarray([first[key]["hit"] for key in keys], dtype=bool)
    second_hit = np.asarray([second[key]["hit"] for key in keys], dtype=bool)
    oracle_dice = float(np.maximum(first_dice, second_dice).mean())
    return {
        "first": first_id,
        "second": second_id,
        "dice_correlation": float(np.corrcoef(first_dice, second_dice)[0, 1]),
        "hit_disagreements": int(np.count_nonzero(first_hit != second_hit)),
        "first_only_hits": int(np.count_nonzero(first_hit & ~second_hit)),
        "second_only_hits": int(np.count_nonzero(second_hit & ~first_hit)),
        "union_hits": int(np.count_nonzero(first_hit | second_hit)),
        "oracle_dice": oracle_dice,
        "oracle_gain_over_better": oracle_dice
        - max(summary_by_id[first_id]["dice"], summary_by_id[second_id]["dice"]),
    }


def greedy_oracle(
    candidate_ids: list[str],
    rows_by_id: dict[str, dict[tuple[str, int], dict[str, Any]]],
    summary_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted(rows_by_id[candidate_ids[0]])
    arrays = {
        candidate_id: np.asarray(
            [rows_by_id[candidate_id][key]["dice"] for key in keys],
            dtype=np.float64,
        )
        for candidate_id in candidate_ids
    }
    selected = [
        sorted(
            candidate_ids,
            key=lambda candidate_id: (
                -summary_by_id[candidate_id]["dice"],
                candidate_id,
            ),
        )[0]
    ]
    oracle = arrays[selected[0]].copy()
    result = [
        {
            "step": 1,
            "added": selected[0],
            "oracle_dice": float(oracle.mean()),
            "increment": float(oracle.mean()),
        }
    ]
    while len(selected) < len(candidate_ids):
        current = float(oracle.mean())
        choices = []
        for candidate_id in candidate_ids:
            if candidate_id in selected:
                continue
            value = float(np.maximum(oracle, arrays[candidate_id]).mean())
            choices.append((value, candidate_id))
        value, added = sorted(choices, key=lambda item: (-item[0], item[1]))[0]
        oracle = np.maximum(oracle, arrays[added])
        selected.append(added)
        result.append(
            {
                "step": len(selected),
                "added": added,
                "oracle_dice": value,
                "increment": value - current,
            }
        )
    return result


def short_hash(value: str) -> str:
    return value[:12]


def build_model_report(
    manifest: dict[str, Any],
    summary_by_id: dict[str, dict[str, Any]],
) -> str:
    candidates = sorted(
        manifest["candidates"],
        key=lambda item: (-summary_by_id[item["id"]]["dice"], item["id"]),
    )
    lines = [
        "# Side Experiment 002 Candidate Model Selection",
        "",
        "This is the immutable 20-checkpoint roster for multi-model ensemble "
        "selection. Inclusion uses fixed val200 mean global Dice per finding "
        "`>= 0.320000` at mask threshold `0.5`.",
        "",
        "## Ranked Candidates",
        "",
        "| Rank | Candidate | Source | Model / arm | Epoch | Architecture | "
        "Normalization | Dice | Hits | Hit rate |",
        "| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for rank, candidate in enumerate(candidates, 1):
        performance = summary_by_id[candidate["id"]]
        lines.append(
            f"| {rank} | `{candidate['id']}` | Exp{candidate['experiment_number']:03d} | "
            f"`{candidate['model_id']}` | {candidate['epoch']} | "
            f"{candidate['architecture']} | {candidate['normalization']} | "
            f"{performance['dice']:.6f} | {performance['hits']} / "
            f"{performance['findings']} | {performance['hit_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "All candidates pass the inclusive cutoff. Performance is reconstructed "
            "from the per-finding evaluator JSON rather than copied from a report.",
            "",
            "## Experiment Provenance",
            "",
            "| Experiment | Full name | Candidates | Training lineage |",
            "| --- | --- | ---: | --- |",
        ]
    )
    by_experiment: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_experiment.setdefault(candidate["experiment_number"], []).append(candidate)
    for experiment_number in sorted(by_experiment):
        values = by_experiment[experiment_number]
        lineages = "; ".join(
            sorted({value["training_lineage"] for value in values})
        )
        lines.append(
            f"| Exp{experiment_number:03d} | "
            f"`{values[0]['experiment_name']}` | {len(values)} | "
            f"{lineages} |"
        )
    lines.extend(
        [
            "",
            "## Immutable Source Hashes",
            "",
            "| Candidate | Checkpoint SHA256 | Config SHA256 | Evaluation SHA256 | "
            "Cache manifest SHA256 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for candidate in sorted(manifest["candidates"], key=lambda item: item["id"]):
        lines.append(
            f"| `{candidate['id']}` | `{candidate['checkpoint']['sha256']}` | "
            f"`{candidate['config']['sha256']}` | "
            f"`{candidate['evaluation']['sha256']}` | "
            f"`{candidate['cache']['manifest_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Immutable Source Locations",
            "",
            "| Candidate | Checkpoint | Configuration | Evaluation | Cache |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for candidate in sorted(manifest["candidates"], key=lambda item: item["id"]):
        lines.append(
            f"| `{candidate['id']}` | `{candidate['checkpoint']['path']}` | "
            f"`{candidate['config']['path']}` | "
            f"`{candidate['evaluation']['path']}` | "
            f"`{candidate['cache']['root']}` |"
        )
    lines.extend(
        [
            "",
            "Configuration validation checks the declared canonical path first. "
            "If that file later changes, a byte-identical hash-named snapshot "
            "under `frozen_sources/` may satisfy the original frozen hash.",
            "",
        ]
    )
    return "\n".join(lines)


def pair_table(
    rows: list[dict[str, Any]],
    labels: dict[str, str],
    first_only_label: str,
) -> list[str]:
    output = [
        "| Pair | Dice correlation | Hit disagreements | "
        f"{first_only_label} | Second-only hits | Union hits | Oracle Dice | "
        "Oracle gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        output.append(
            f"| {labels[row['first']]} vs {labels[row['second']]} | "
            f"{row['dice_correlation']:.6f} | "
            f"{row['hit_disagreements']} / {EXPECTED_FINDINGS} | "
            f"{row['first_only_hits']} | {row['second_only_hits']} | "
            f"{row['union_hits']} | {row['oracle_dice']:.6f} | "
            f"{row['oracle_gain_over_better']:+.6f} |"
        )
    return output


def build_complementarity_report(
    manifest: dict[str, Any],
    pairwise: dict[str, dict[str, Any]],
    greedy: list[dict[str, Any]],
) -> str:
    labels = {
        candidate["id"]: f"`{candidate['id']}`"
        for candidate in manifest["candidates"]
    }
    exp008_ids = [
        ("exp008_shared_e080", "exp008_shared_e100"),
        ("exp008_dual_e080", "exp008_dual_e100"),
        ("exp008_precision_e080", "exp008_precision_e100"),
        ("exp008_joint_e080", "exp008_joint_e100"),
    ]
    exp011_vs_baseline = [
        ("exp011_native_zscore_e100", "exp009_baseline_e100"),
        ("exp011_clip_zscore_e100", "exp009_baseline_e100"),
        ("exp011_linear_hu_e100", "exp009_baseline_e100"),
    ]
    exp011_pairs = list(
        itertools.combinations(
            [
                "exp011_native_zscore_e100",
                "exp011_clip_zscore_e100",
                "exp011_linear_hu_e100",
            ],
            2,
        )
    )
    cross_pairs = [
        ("exp006_e6d4_e100", "exp008_precision_e080"),
        ("exp007_ddp_e050", "exp008_precision_e080"),
        ("exp009_baseline_e100", "exp008_joint_e080"),
    ]

    def get(rows: list[tuple[str, str]]) -> list[dict[str, Any]]:
        oriented = []
        for first, second in rows:
            source = dict(pairwise["|".join(sorted((first, second)))])
            if source["first"] != first:
                source["first"], source["second"] = (
                    source["second"],
                    source["first"],
                )
                source["first_only_hits"], source["second_only_hits"] = (
                    source["second_only_hits"],
                    source["first_only_hits"],
                )
            oriented.append(source)
        return oriented

    lines = [
        "# Side Experiment 002 Hard-Mask Complementarity",
        "",
        "This report quantifies why individual val200 Dice alone is insufficient "
        "for pruning the ensemble pool. It uses the existing threshold-0.5 masks "
        "for the same 200 cases and 381 findings; no inference is rerun.",
        "",
        "> **Audit correction:** an earlier exploratory comparison aligned rows by "
        "their serialized JSON order. The evaluator writes cases in multiprocessing "
        "completion order, so that comparison overstated disagreement. Every value "
        "below is corrected by joining on `(case name, finding index)`.",
        "",
        "> **Interpretation limit:** union hits and best-of-model oracle Dice use "
        "hindsight to choose a successful or higher-Dice model per finding. They "
        "are upper bounds that demonstrate complementary errors, not scores "
        "achievable by averaging.",
        "",
        "## Definitions",
        "",
        "- Dice correlation: Pearson correlation between aligned per-finding Dice vectors.",
        "- Hit disagreement: one model has Dice at least `0.1` and the other does not.",
        "- Model-only hits: findings hit by that model and missed by its pair.",
        "- Union hits: findings hit by either model.",
        "- Oracle Dice: mean of the larger per-finding Dice from the two models.",
        "",
        "## Exp008 Epoch 80 Versus Epoch 100",
        "",
    ]
    lines.extend(pair_table(get(exp008_ids), labels, "Epoch-80-only hits"))
    lines.extend(
        [
            "",
            "Each epoch-80 snapshot uniquely hits 5-6 findings missed by its "
            "epoch-100 continuation. Correctly aligned correlations are roughly "
            "0.97, so these snapshots are related but not identical.",
            "",
            "## Exp011 Normalization Arms Versus Exp009 Baseline",
            "",
        ]
    )
    lines.extend(
        pair_table(
            get(exp011_vs_baseline),
            labels,
            "Exp011-only hits",
        )
    )
    lines.extend(
        [
            "",
            "The three independently trained Exp011 arms recover 7-9 findings "
            "missed by the highest-Dice individual model.",
            "",
            "## Exp011 Pairwise",
            "",
        ]
    )
    lines.extend(pair_table(get(exp011_pairs), labels, "First-only hits"))
    lines.extend(["", "## Selected Cross-Family Pairs", ""])
    lines.extend(pair_table(get(cross_pairs), labels, "First-only hits"))
    lines.extend(
        [
            "",
            "The Exp007 epoch-50 and Exp008 precision epoch-80 pair disagrees on "
            "27 hit decisions and covers 299 of 381 findings.",
            "",
            "## Greedy Best-of-Model Oracle",
            "",
            "This deliberately optimistic sequence repeatedly adds the model that "
            "maximizes per-finding hindsight Dice.",
            "",
            "| Step | Added model | Oracle Dice | Increment |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for row in greedy:
        lines.append(
            f"| {row['step']} | `{row['added']}` | "
            f"{row['oracle_dice']:.6f} | {row['increment']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "The realizable gain must be measured from saved logits with "
            "cross-validated averaging, weights, and thresholds.",
            "",
        ]
    )
    return "\n".join(lines)


def assert_highlighted_findings(pairwise: dict[str, dict[str, Any]]) -> None:
    def pair(first: str, second: str) -> dict[str, Any]:
        return pairwise["|".join(sorted((first, second)))]

    expected_exp008 = [
        ("exp008_shared_e080", "exp008_shared_e100", 6),
        ("exp008_dual_e080", "exp008_dual_e100", 5),
        ("exp008_precision_e080", "exp008_precision_e100", 6),
        ("exp008_joint_e080", "exp008_joint_e100", 5),
    ]
    for first, second, expected in expected_exp008:
        row = pair(first, second)
        value = (
            row["first_only_hits"]
            if row["first"] == first
            else row["second_only_hits"]
        )
        if value != expected:
            raise AssertionError(f"{first} unique hits: expected {expected}, got {value}")
    for first, expected in [
        ("exp011_native_zscore_e100", 7),
        ("exp011_clip_zscore_e100", 8),
        ("exp011_linear_hu_e100", 9),
    ]:
        row = pair(first, "exp009_baseline_e100")
        value = (
            row["first_only_hits"]
            if row["first"] == first
            else row["second_only_hits"]
        )
        if value != expected:
            raise AssertionError(f"{first} unique hits: expected {expected}, got {value}")
    cross = pair("exp007_ddp_e050", "exp008_precision_e080")
    if cross["union_hits"] != 299:
        raise AssertionError(
            "Exp007 e50 + Exp008 precision e80 union: "
            f"expected 299, got {cross['union_hits']}"
        )


def run(
    manifest_path: Path,
    model_report_path: Path,
    complementarity_report_path: Path,
    complementarity_json_path: Path,
    verify_large_hashes: bool,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset_path = resolve_source_path(manifest["dataset"]["path"])
    require_sha256(
        dataset_path,
        manifest["dataset"]["sha256"],
        "val200 dataset JSON",
    )
    dataset_keys = load_dataset_keys(dataset_path)
    rows_by_id: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    summary_by_id: dict[str, dict[str, Any]] = {}
    common_keys: set[tuple[str, int]] | None = None
    for candidate in manifest["candidates"]:
        require_candidate_config(candidate)
        require_sha256(
            Path(candidate["cache"]["manifest_path"]),
            candidate["cache"]["manifest_sha256"],
            f"{candidate['id']} cache manifest",
        )
        if verify_large_hashes:
            require_sha256(
                Path(candidate["checkpoint"]["path"]),
                candidate["checkpoint"]["sha256"],
                f"{candidate['id']} checkpoint",
            )
        rows, summary = flatten_eval(candidate, verify_hash=True)
        if common_keys is None:
            common_keys = set(rows)
        elif set(rows) != common_keys:
            raise ValueError(f"{candidate['id']}: case/finding set differs from roster")
        if set(rows) != dataset_keys:
            raise ValueError(
                f"{candidate['id']}: case/finding set differs from dataset JSON"
            )
        if summary["dice"] < float(manifest["selection"]["minimum_dice"]):
            raise ValueError(f"{candidate['id']}: does not pass the Dice cutoff")
        rows_by_id[candidate["id"]] = rows
        summary_by_id[candidate["id"]] = summary

    candidate_ids = sorted(rows_by_id)
    pairwise: dict[str, dict[str, Any]] = {}
    for first, second in itertools.combinations(candidate_ids, 2):
        row = compare_pair(first, second, rows_by_id, summary_by_id)
        pairwise[f"{first}|{second}"] = row
    assert_highlighted_findings(pairwise)
    greedy = greedy_oracle(candidate_ids, rows_by_id, summary_by_id)

    result = {
        "schema_version": 1,
        "analysis": "sideexp002_hard_mask_complementarity",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "threshold": 0.5,
        "hit_threshold": HIT_THRESHOLD,
        "cases": EXPECTED_CASES,
        "findings": EXPECTED_FINDINGS,
        "candidate_metrics": summary_by_id,
        "pairwise": pairwise,
        "greedy_best_of_model_oracle": greedy,
        "interpretation_limit": (
            "Union and best-of-model oracle metrics use per-finding hindsight "
            "and are not realizable ensemble scores."
        ),
    }
    atomic_write_text(
        model_report_path,
        build_model_report(manifest, summary_by_id),
    )
    atomic_write_text(
        complementarity_report_path,
        build_complementarity_report(manifest, pairwise, greedy),
    )
    atomic_write_json(complementarity_json_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-report", type=Path, default=DEFAULT_MODEL_REPORT)
    parser.add_argument(
        "--complementarity-report",
        type=Path,
        default=DEFAULT_COMPLEMENTARITY_REPORT,
    )
    parser.add_argument(
        "--complementarity-json",
        type=Path,
        default=DEFAULT_COMPLEMENTARITY_JSON,
    )
    parser.add_argument(
        "--verify-checkpoint-hashes",
        action="store_true",
        help="Also stream and verify all 20 multi-gigabyte checkpoint files.",
    )
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        model_report_path=args.model_report,
        complementarity_report_path=args.complementarity_report,
        complementarity_json_path=args.complementarity_json,
        verify_large_hashes=args.verify_checkpoint_hashes,
    )
    print(
        f"Validated {len(result['candidate_metrics'])} candidates, "
        f"{result['cases']} cases, and {result['findings']} findings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
