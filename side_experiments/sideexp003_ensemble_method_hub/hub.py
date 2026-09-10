#!/usr/bin/env python3
"""SideExp003 catalog, cache, and reproducible ensemble-method entrypoint.

The hub never loads a model itself. GPU inference and CPU-only ensemble workers
run through explicit, frozen job specifications.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fresh_cache  # noqa: E402
import preliminary_ensemble  # noqa: E402
import seeded_caruana  # noqa: E402

CATALOG_PATH = HERE / "checkpoint_catalog.json"
LEADERBOARD_PATH = HERE / "val200_checkpoint_leaderboard.md"
SUBCATEGORY_PATH = HERE / "val200_subcategory_leaderboard.md"
IMPORT_MANIFEST_PATH = HERE / "legacy_import_manifest.json"
OVERRIDES_PATH = HERE / "catalog_overrides.json"

DATASET_PATH = REPO_ROOT / "configs/evaluation/rexgroundingct_val200_seed20260723.json"
EXPECTED_DATASET_SHA256 = "7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
EXPECTED_CASES = 200
EXPECTED_FINDINGS = 381
HIT_THRESHOLD = 0.1
MASK_THRESHOLD = 0.5
CACHE_CONTRACT_VERSION = 1
OFFICIAL_CATEGORY_ORDER = (
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h",
)

EXPERIMENT_RUNTIME_ROOT = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct"
)
SIDEEXP002_TRACKED = REPO_ROOT / "side_experiments/sideexp002_multimodel_ensemble_selection"
SIDEEXP002_RUNTIME = Path(
    "/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/"
    "sideexp002_multimodel_ensemble_selection"
)
RUNTIME_ROOT = Path(
    "/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/"
    "sideexp003_ensemble_method_hub"
)
LEGACY_SOURCE = SIDEEXP002_RUNTIME / "logits"
LEGACY_DESTINATION = RUNTIME_ROOT / "cache/logits/legacy_sideexp002"

EVALUATOR_GLOB = "*/runs/**/eval_epoch*_val200/eval/val_quick_global_eval.json"
EPOCH_RE = re.compile(r"^eval_epoch(?P<epoch>\d+)_val200$")
EXPERIMENT_RE = re.compile(r"^(?P<number>\d{3})_(?P<name>.+)$")
SAFE_SLUG_RE = re.compile(r"[^a-z0-9]+")
EXPECTED_TOP20_CATALOG_SHA256 = (
    "0f70f841cd9f9f88f94bac5b0a8160394af12b6efd832823aa11c9c357d4a6fe"
)
CONTINUATION_ALLOWED_SOURCE_DRIFT = {
    "scripts/rexgroundingct/common.py": (
        "experiment_registry_only; worker imports REX_SEG_DIR and sorted_prompts, "
        "whose committed definitions did not change"
    ),
    "side_experiments/sideexp003_ensemble_method_hub/hub.py": (
        "control_plane_cli_only; no model prediction implementation"
    ),
    "side_experiments/sideexp003_ensemble_method_hub/fresh_cache.py": (
        "continuation contract construction and validation only"
    ),
    "side_experiments/sideexp003_ensemble_method_hub/fresh_orchestrator.py": (
        "continuation rank/wave scheduling and completion reporting only"
    ),
}


class HubError(RuntimeError):
    """Raised when the SideExp003 contract cannot be satisfied."""


class CatalogConflict(HubError):
    """Raised when one checkpoint SHA maps to inconsistent evaluators."""

    def __init__(self, message: str, stub: dict[str, Any]):
        super().__init__(message)
        self.stub = stub


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise HubError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HubError(f"invalid JSON file {path}: {exc}") from exc


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_cache_key(
    *,
    dataset_sha256: str,
    checkpoint_sha256: str,
    inference_contract_sha256: str,
    preprocessing_manifest_sha256: str,
    code_version: str,
    storage_contract: dict[str, Any],
) -> str:
    """Build the content-addressed identity for a new base-model logit cache."""

    fields = {
        "dataset_sha256": dataset_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "inference_contract_sha256": inference_contract_sha256,
        "preprocessing_manifest_sha256": preprocessing_manifest_sha256,
        "code_version": code_version,
    }
    missing = [key for key, value in fields.items() if not isinstance(value, str) or not value]
    if missing:
        raise HubError(f"cache-key contract has empty fields: {missing}")
    if not isinstance(storage_contract, dict) or not storage_contract:
        raise HubError("cache-key storage contract must be a non-empty object")
    contract = {
        "schema_version": CACHE_CONTRACT_VERSION,
        **fields,
        "storage_contract": storage_contract,
    }
    return f"v{CACHE_CONTRACT_VERSION}_{json_sha256(contract)}"


def _finding_index(value: str) -> int:
    match = re.fullmatch(r"finding_(\d+)", value)
    if not match:
        raise HubError(f"invalid evaluator finding key: {value!r}")
    return int(match.group(1))


def load_dataset_contract(path: Path = DATASET_PATH) -> dict[str, Any]:
    data = read_json(path)
    cases = data.get("test") if isinstance(data, dict) else None
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise HubError(f"{path}: expected {EXPECTED_CASES} test cases")

    positions: dict[tuple[str, int], str] = {}
    case_names: set[str] = set()
    category_counts: Counter[str] = Counter()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise HubError(f"{path}: every test case needs a string name")
        name = case["name"]
        if name in case_names:
            raise HubError(f"{path}: duplicate case {name}")
        case_names.add(name)
        findings = case.get("findings")
        categories = case.get("categories")
        if not isinstance(findings, dict) or not isinstance(categories, dict):
            raise HubError(f"{path}: {name} lacks finding/category mappings")
        if set(findings) != set(categories):
            raise HubError(f"{path}: {name} finding/category keys differ")
        for raw_index, category in categories.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise HubError(f"{path}: invalid finding index {raw_index!r}") from exc
            if not isinstance(category, str) or category not in OFFICIAL_CATEGORY_ORDER:
                raise HubError(f"{path}: invalid official category {category!r}")
            key = (name, index)
            if key in positions:
                raise HubError(f"{path}: duplicate finding {key}")
            positions[key] = category
            category_counts[category] += 1
    if len(positions) != EXPECTED_FINDINGS:
        raise HubError(f"{path}: expected {EXPECTED_FINDINGS} findings, got {len(positions)}")
    observed_sha = sha256_file(path)
    if path.resolve() == DATASET_PATH.resolve() and observed_sha != EXPECTED_DATASET_SHA256:
        raise HubError(
            f"fixed val200 SHA drift: expected {EXPECTED_DATASET_SHA256}, got {observed_sha}"
        )
    return {
        "path": str(path),
        "sha256": observed_sha,
        "cases": len(cases),
        "findings": len(positions),
        "positions": positions,
        "category_counts": {
            category: category_counts.get(category, 0)
            for category in OFFICIAL_CATEGORY_ORDER
        },
    }


def recompose_evaluator(
    path: Path, dataset: dict[str, Any]
) -> dict[str, Any]:
    evaluation = read_json(path)
    if not isinstance(evaluation, dict):
        raise HubError(f"{path}: evaluator must be an object")
    summary = evaluation.get("summary")
    cases = evaluation.get("cases")
    if not isinstance(summary, dict) or not isinstance(cases, list):
        raise HubError(f"{path}: evaluator needs summary and cases")
    params = summary.get("params")
    if not isinstance(params, dict) or params.get("global_only") is not True:
        raise HubError(f"{path}: evaluator is not global-only")
    if float(params.get("global_hit_thr", math.nan)) != HIT_THRESHOLD:
        raise HubError(f"{path}: global hit threshold is not {HIT_THRESHOLD}")
    if summary.get("total_cases") != EXPECTED_CASES or len(cases) != EXPECTED_CASES:
        raise HubError(f"{path}: evaluator is not exactly {EXPECTED_CASES} cases")
    if summary.get("total_findings") != EXPECTED_FINDINGS:
        raise HubError(f"{path}: evaluator is not exactly {EXPECTED_FINDINGS} findings")

    expected_positions: dict[tuple[str, int], str] = dataset["positions"]
    observed: dict[tuple[str, int], tuple[float, bool]] = {}
    seen_cases: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("file"), str):
            raise HubError(f"{path}: evaluator case has no string file")
        case_name = case["file"]
        if case_name in seen_cases:
            raise HubError(f"{path}: duplicate evaluator case {case_name}")
        seen_cases.add(case_name)
        findings = case.get("findings")
        if not isinstance(findings, dict):
            raise HubError(f"{path}: {case_name} has no findings object")
        for finding_key, result in findings.items():
            if not isinstance(finding_key, str) or not isinstance(result, dict):
                raise HubError(f"{path}: malformed finding in {case_name}")
            key = (case_name, _finding_index(finding_key))
            if key not in expected_positions:
                raise HubError(f"{path}: finding is absent from fixed val200: {key}")
            if key in observed:
                raise HubError(f"{path}: duplicate evaluator finding {key}")
            dice = result.get("global_dice")
            hit = result.get("global_hit")
            if not isinstance(dice, (int, float)) or not math.isfinite(float(dice)):
                raise HubError(f"{path}: invalid Dice at {key}")
            if not isinstance(hit, bool) or hit != (float(dice) >= HIT_THRESHOLD):
                raise HubError(f"{path}: invalid Dice/hit pair at {key}")
            observed[key] = (float(dice), hit)

    missing = set(expected_positions).difference(observed)
    if missing or len(observed) != EXPECTED_FINDINGS:
        preview = sorted(missing)[:3]
        raise HubError(
            f"{path}: evaluator finding alignment failed; missing={len(missing)} {preview}"
        )

    def aggregate(keys: Iterable[tuple[str, int]]) -> dict[str, Any]:
        records = [observed[key] for key in keys]
        hits = sum(int(hit) for _dice, hit in records)
        return {
            "available": bool(records),
            "dice": math.fsum(dice for dice, _hit in records) / len(records)
            if records
            else None,
            "hits": hits,
            "findings": len(records),
            "hit_rate": hits / len(records) if records else None,
        }

    overall = aggregate(expected_positions)
    categories = {}
    for category in OFFICIAL_CATEGORY_ORDER:
        category_keys = [
            key for key, observed_category in expected_positions.items()
            if observed_category == category
        ]
        categories[category] = aggregate(category_keys)

    reported_dice = float(summary.get("mean_global_dice_per_finding", math.nan))
    reported_hits = int(summary.get("total_hits", -1))
    if not math.isclose(overall["dice"], reported_dice, rel_tol=0.0, abs_tol=1e-12):
        raise HubError(
            f"{path}: recomposed Dice {overall['dice']} != reported {reported_dice}"
        )
    if overall["hits"] != reported_hits:
        raise HubError(
            f"{path}: recomposed hits {overall['hits']} != reported {reported_hits}"
        )
    return {
        "overall": overall,
        "categories": categories,
        "finding_key_sha256": json_sha256(
            [[case, index] for case, index in sorted(observed)]
        ),
    }


def _nested_value(data: Any, dotted_path: str) -> Any:
    value = data
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def threshold_provenance(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.is_file():
        return {
            "status": "missing",
            "value": None,
            "path": str(config_path) if config_path else None,
            "sha256": None,
            "json_path": None,
        }
    config = read_json(config_path)
    for json_path in ("validation.primary_threshold", "validation.final_threshold"):
        value = _nested_value(config, json_path)
        if isinstance(value, (int, float)):
            return {
                "status": "verified" if float(value) == MASK_THRESHOLD else "conflict",
                "value": float(value),
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "json_path": json_path,
            }
    return {
        "status": "missing",
        "value": None,
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "json_path": None,
    }


def _metadata_from_config(
    config_path: Path | None, legacy: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    if legacy:
        architecture = {
            "name": str(legacy.get("architecture", "unknown")),
            "provenance": "SideExp002 candidate_manifest.json",
        }
    else:
        architecture = None

    config = read_json(config_path) if config_path and config_path.is_file() else {}
    raw_arch = config.get("architecture") if isinstance(config, dict) else None
    if architecture is None:
        if isinstance(raw_arch, dict):
            arch_name = raw_arch.get("model_type", raw_arch.get("classification", "unknown"))
        elif isinstance(raw_arch, str):
            arch_name = raw_arch
        else:
            arch_name = "voxtell_standard"
        architecture = {
            "name": str(arch_name),
            "provenance": "experiment config or standard-model default",
        }
    raw_pre = config.get("preprocessing") if isinstance(config, dict) else None
    if isinstance(raw_pre, dict):
        pre_name = raw_pre.get("cache_id", raw_pre.get("classification", "unknown"))
        pre_description = raw_pre.get("normalization", raw_pre.get("classification", "unknown"))
        cache_root = raw_pre.get("cache_root")
        manifest_path = raw_pre.get("manifest_path")
        if manifest_path is None and isinstance(cache_root, str) and cache_root:
            manifest_path = str(Path(cache_root) / "manifest.json")
        preprocessing = {
            "name": str(pre_name),
            "description": str(pre_description),
            "cache_root": cache_root,
            "manifest_path": manifest_path,
            "manifest_sha256": raw_pre.get("manifest_sha256"),
            "provenance": "experiment config preprocessing",
        }
    elif legacy:
        cache = legacy.get("cache") if isinstance(legacy.get("cache"), dict) else {}
        preprocessing = {
            "name": str(cache.get("id", legacy.get("normalization", "unknown"))),
            "description": str(legacy.get("normalization", "unknown")),
            "cache_root": cache.get("root"),
            "manifest_path": cache.get("manifest_path"),
            "manifest_sha256": cache.get("manifest_sha256"),
            "provenance": "SideExp002 candidate_manifest.json fallback",
        }
    else:
        preprocessing = {
            "name": "unknown",
            "description": "not declared in experiment config",
            "cache_root": None,
            "manifest_path": None,
            "manifest_sha256": None,
            "provenance": "missing",
        }
    return architecture, preprocessing


def load_legacy_index() -> dict[str, dict[str, Any]]:
    manifest_path = SIDEEXP002_TRACKED / "candidate_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = read_json(manifest_path)
    candidates = manifest.get("candidates", [])
    failed_ids: set[str] = set()
    finalization = SIDEEXP002_RUNTIME / "finalization_manifest.json"
    if finalization.is_file():
        result = read_json(finalization)
        failed_ids = set(result.get("incomplete_candidate_ids", []))
    index = {}
    for candidate in candidates:
        checkpoint = candidate.get("checkpoint", {})
        checksum = checkpoint.get("sha256")
        if not isinstance(checksum, str):
            continue
        candidate_id = str(candidate.get("id"))
        index[checksum] = {
            **candidate,
            "cache_gate": "analysis_usable" if candidate_id in failed_ids else "strict_gate_passed",
            "legacy_logical_path": str(LEGACY_SOURCE / candidate_id),
            "legacy_physical_path": str(LEGACY_DESTINATION / candidate_id),
        }
    return index


def _parse_evaluator_path(path: Path, runtime_root: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(runtime_root)
    except ValueError as exc:
        raise HubError(f"evaluator is outside runtime root: {path}") from exc
    parts = relative.parts
    if len(parts) < 7 or parts[1] != "runs":
        raise HubError(f"unrecognized evaluator layout: {path}")
    experiment_id = parts[0]
    experiment_match = EXPERIMENT_RE.match(experiment_id)
    if not experiment_match:
        raise HubError(f"invalid experiment directory: {experiment_id}")
    eval_index = next(
        (index for index, part in enumerate(parts) if EPOCH_RE.match(part)), None
    )
    if eval_index is None or eval_index <= 3:
        raise HubError(f"unrecognized evaluator epoch layout: {path}")
    epoch_match = EPOCH_RE.match(parts[eval_index])
    assert epoch_match is not None
    arm_parts = parts[3:eval_index]
    return {
        "experiment_id": experiment_id,
        "experiment_number": int(experiment_match.group("number")),
        "run": parts[2],
        "model": "/".join(arm_parts),
        "arm": "/".join(arm_parts),
        "epoch": int(epoch_match.group("epoch")),
        "arm_root": runtime_root.joinpath(*parts[:eval_index]),
    }


def _config_for_experiment(experiment_id: str) -> Path | None:
    path = REPO_ROOT / "configs/experiments" / f"{experiment_id}.json"
    return path if path.is_file() else None


def _checkpoint_for_source(source: dict[str, Any]) -> Path | None:
    epoch = int(source["epoch"])
    path = source["arm_root"] / f"model_epoch{epoch:03d}/fold_0/checkpoint_final.pth"
    if path.is_file():
        return path
    alternatives = sorted(source["arm_root"].glob(
        f"model_epoch{epoch}*/fold_0/checkpoint_final.pth"
    ))
    return alternatives[0] if len(alternatives) == 1 else None


def _safe_slug(value: str) -> str:
    return SAFE_SLUG_RE.sub("_", value.lower()).strip("_") or "model"


def _candidate_id(source: dict[str, Any], identity_sha: str) -> str:
    return (
        f"exp{source['experiment_number']:03d}_{_safe_slug(source['model'])}_"
        f"e{source['epoch']:03d}_{identity_sha[:8]}"
    )


def _metrics_signature(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": metrics["overall"],
        "categories": metrics["categories"],
        "finding_key_sha256": metrics["finding_key_sha256"],
    }


def _metrics_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return json_sha256(_metrics_signature(first)) == json_sha256(_metrics_signature(second))


def scan_catalog_candidates(
    evaluator_paths: Sequence[Path],
    *,
    dataset_path: Path = DATASET_PATH,
    runtime_root: Path = EXPERIMENT_RUNTIME_ROOT,
    known_checkpoint_hashes: dict[str, str] | None = None,
    evaluator_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset_contract(dataset_path)
    legacy_index = load_legacy_index()
    known_checkpoint_hashes = known_checkpoint_hashes or {}
    evaluator_overrides = evaluator_overrides or {}
    raw: list[dict[str, Any]] = []
    errors: list[str] = []
    for evaluation_path in sorted(set(evaluator_paths), key=str):
        try:
            source = _parse_evaluator_path(evaluation_path, runtime_root)
            checkpoint_path = _checkpoint_for_source(source)
            checkpoint_sha = None
            if checkpoint_path is not None:
                checkpoint_sha = known_checkpoint_hashes.get(str(checkpoint_path))
                if checkpoint_sha is None:
                    checkpoint_sha = sha256_file(checkpoint_path)
            evaluation_sha = sha256_file(evaluation_path)
            metrics = recompose_evaluator(evaluation_path, dataset)
            config_path = _config_for_experiment(source["experiment_id"])
            threshold = threshold_provenance(config_path)
            raw.append(
                {
                    "source": source,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_sha256": checkpoint_sha,
                    "evaluation_path": evaluation_path,
                    "evaluation_sha256": evaluation_sha,
                    "metrics": metrics,
                    "config_path": config_path,
                    "threshold": threshold,
                }
            )
        except (HubError, OSError, ValueError) as exc:
            errors.append(f"{evaluation_path}: {exc}")
    if errors:
        raise HubError("catalog scan rejected evaluator(s):\n- " + "\n- ".join(errors))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw:
        identity = item["checkpoint_sha256"] or f"missing:{item['evaluation_sha256']}"
        grouped[identity].append(item)

    candidates: list[dict[str, Any]] = []
    for identity, items in sorted(grouped.items()):
        items.sort(key=lambda item: str(item["evaluation_path"]))
        canonical = items[0]
        conflicting = [
            item for item in items[1:]
            if not _metrics_equal(canonical["metrics"], item["metrics"])
        ]
        applied_override = None
        if conflicting:
            stub = {
                "schema_version": 1,
                "evaluator_overrides": {
                    identity: {
                        "canonical_evaluator_path": None,
                        "reason": None,
                        "reviewed_by": None,
                        "reviewed_at_utc": None,
                        "candidate_sources": [
                            {
                                "evaluation_path": str(item["evaluation_path"]),
                                "evaluation_sha256": item["evaluation_sha256"],
                                "overall": item["metrics"]["overall"],
                            }
                            for item in items
                        ],
                    }
                },
            }
            applied_override = evaluator_overrides.get(identity)
            selected_path = (
                applied_override.get("canonical_evaluator_path")
                if isinstance(applied_override, dict) else None
            )
            selected = [item for item in items if str(item["evaluation_path"]) == selected_path]
            if (
                len(selected) != 1
                or not isinstance(applied_override.get("reason"), str)
                or not applied_override["reason"].strip()
            ):
                raise CatalogConflict(
                    "same checkpoint SHA has conflicting evaluators; explicit override required",
                    stub,
                )
            canonical = selected[0]
        source = canonical["source"]
        checkpoint_sha = canonical["checkpoint_sha256"]
        legacy = legacy_index.get(checkpoint_sha) if checkpoint_sha else None
        architecture, preprocessing = _metadata_from_config(
            canonical["config_path"], legacy
        )
        blocking_warnings = []
        if checkpoint_sha is None:
            blocking_warnings.append("checkpoint_provenance_missing")
        if canonical["threshold"]["status"] != "verified":
            blocking_warnings.append("mask_threshold_provenance_missing_or_conflicting")
        threshold_signatures = {json_sha256(item["threshold"]) for item in items}
        if len(threshold_signatures) != 1:
            blocking_warnings.append("duplicate_evaluator_threshold_provenance_differs")
        warnings = list(blocking_warnings)
        if applied_override is not None:
            warnings.append("conflicting_duplicate_evaluators_resolved_by_reviewed_override")
        eligibility = "eligible" if not blocking_warnings else "needs_review"
        identity_sha = checkpoint_sha or canonical["evaluation_sha256"]
        config_path = canonical["config_path"]
        config_sha = sha256_file(config_path) if config_path else None
        if legacy:
            cache = {
                "status": legacy["cache_gate"],
                "candidate_id": legacy["id"],
                "logical_path": legacy["legacy_logical_path"],
                "physical_path": legacy["legacy_physical_path"],
                "final_use_requires_strict_reexport": legacy["cache_gate"] != "strict_gate_passed",
            }
        else:
            cache = {
                "status": "missing",
                "candidate_id": None,
                "logical_path": None,
                "physical_path": None,
                "final_use_requires_strict_reexport": False,
            }
        candidates.append(
            {
                "candidate_id": _candidate_id(source, identity_sha),
                "experiment": {
                    "number": source["experiment_number"],
                    "id": source["experiment_id"],
                },
                "run": source["run"],
                "model": source["model"],
                "arm": source["arm"],
                "epoch": source["epoch"],
                "architecture": architecture,
                "preprocessing": preprocessing,
                "checkpoint": {
                    "path": str(canonical["checkpoint_path"])
                    if canonical["checkpoint_path"] else None,
                    "sha256": checkpoint_sha,
                },
                "config": {
                    "path": str(config_path) if config_path else None,
                    "sha256": config_sha,
                },
                "dataset": {
                    "path": str(dataset_path),
                    "sha256": dataset["sha256"],
                },
                "canonical_evaluator": {
                    "path": str(canonical["evaluation_path"]),
                    "sha256": canonical["evaluation_sha256"],
                },
                "evaluator_aliases": [
                    {
                        "path": str(item["evaluation_path"]),
                        "sha256": item["evaluation_sha256"],
                    }
                    for item in items
                ],
                "evaluator_override": applied_override,
                "threshold": canonical["threshold"],
                "metrics": canonical["metrics"],
                "eligibility": eligibility,
                "logit_cache": cache,
                "warnings": warnings,
            }
        )
    candidates.sort(key=lambda row: row["candidate_id"])
    return candidates, dataset


def known_checkpoint_hashes(catalog: dict[str, Any] | None) -> dict[str, str]:
    result = {}
    if not catalog:
        return result
    for candidate in catalog.get("candidates", []):
        checkpoint = candidate.get("checkpoint", {})
        path = checkpoint.get("path")
        checksum = checkpoint.get("sha256")
        if isinstance(path, str) and isinstance(checksum, str) and Path(path).is_file():
            result[path] = checksum
    return result


def make_catalog(
    candidates: list[dict[str, Any]],
    dataset: dict[str, Any],
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing_by_sha = {}
    if existing is not None:
        migrated = fresh_cache.migrate_catalog(existing)
        existing_by_sha = {
            candidate.get("checkpoint", {}).get("sha256"): candidate
            for candidate in migrated.get("candidates", [])
            if candidate.get("checkpoint", {}).get("sha256")
        }
    merged = [
        fresh_cache.merge_candidate_state(
            candidate,
            existing_by_sha.get(candidate.get("checkpoint", {}).get("sha256")),
        )
        for candidate in candidates
    ]
    return {
        "schema_version": fresh_cache.CATALOG_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "artifact_state_updated_at_utc": (
            existing.get("artifact_state_updated_at_utc") if existing else None
        ),
        "source_of_truth": "checkpoint_catalog.json",
        "dataset": {
            "path": dataset["path"],
            "sha256": dataset["sha256"],
            "cases": dataset["cases"],
            "findings": dataset["findings"],
            "hit_dice_threshold": HIT_THRESHOLD,
            "official_category_order": list(OFFICIAL_CATEGORY_ORDER),
            "category_findings": dataset["category_counts"],
        },
        "scan": {
            "runtime_root": str(EXPERIMENT_RUNTIME_ROOT),
            "evaluator_glob": EVALUATOR_GLOB,
            "scope": "single-checkpoint fixed-val200 evaluators only",
        },
        "candidate_count": len(merged),
        "eligible_count": sum(c["eligibility"] == "eligible" for c in merged),
        "needs_review_count": sum(c["eligibility"] != "eligible" for c in merged),
        "candidates": merged,
    }


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric(metric: dict[str, Any]) -> str:
    if not metric.get("available"):
        return "—"
    return f"{float(metric['dice']):.6f} ({int(metric['hits'])}/{int(metric['findings'])})"


def _artifact_cells(row: dict[str, Any]) -> dict[str, str]:
    membership = fresh_cache.job_membership(row)
    job_id = membership.get("job_id") if membership else None
    version = fresh_cache.job_version(row, job_id)
    legacy = fresh_cache.legacy_versions(row)
    test300 = row.get("inference_artifacts", {}).get("test300", {})
    if membership and version:
        rank = str(membership["rank"])
        assignment = f"w{membership['wave']}/g{membership['gpu']}"
        status = str(version.get("status", "queued"))
        cache_path = str(version.get("physical_path") or "—")
        result_path = str(version.get("validation_path") or "—")
    else:
        rank = "—"
        assignment = "—"
        status = "not_requested"
        cache_path = "—"
        result_path = "—"
    if legacy:
        legacy_text = ", ".join(
            f"{value.get('status')}:{value.get('cache_version_id')}" for value in legacy
        )
    else:
        legacy_text = "none"
    return {
        "rank": rank,
        "assignment": assignment,
        "status": status,
        "cache_path": cache_path,
        "result_path": result_path,
        "legacy": legacy_text,
        "test300": (
            f"{test300.get('status', 'deferred')}:"
            f"{test300.get('cache_path') or '—'}"
        ),
    }


def render_checkpoint_leaderboard(catalog: dict[str, Any]) -> str:
    candidates = list(catalog["candidates"])
    by_experiment = sorted(
        candidates,
        key=lambda row: (
            int(row["experiment"]["number"]), str(row["model"]), int(row["epoch"]),
            str(row["candidate_id"]),
        ),
    )
    by_dice = sorted(
        candidates,
        key=lambda row: (
            -float(row["metrics"]["overall"]["dice"]),
            -int(row["metrics"]["overall"]["hits"]),
            str(row["candidate_id"]),
        ),
    )
    lines = [
        "# SideExp003 Val200 Checkpoint Leaderboard",
        "",
        "Generated from `checkpoint_catalog.json`; do not edit ranking rows by hand.",
        f"Snapshot: `{catalog['generated_at_utc']}`. Dataset: `{catalog['dataset']['sha256']}` "
        f"({catalog['dataset']['cases']} cases / {catalog['dataset']['findings']} findings).",
        "",
        "`eligible` means checkpoint provenance and a fixed `0.5` mask-threshold source were verified. "
        "`needs_review` rows stay visible but are excluded from rosters by default.",
        "",
        "## By experiment, model, and epoch",
        "",
        "| Experiment | Run | Model / arm | Epoch | Dice | Hits | Eligibility | Top20 | Wave/GPU | Fresh val200 logits | Fresh cache | Fresh result | Legacy cache | Test300 | Candidate ID |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in by_experiment:
        overall = row["metrics"]["overall"]
        artifact = _artifact_cells(row)
        lines.append(
            f"| {row['experiment']['number']:03d} `{_escape(row['experiment']['id'])}` "
            f"| `{_escape(row['run'])}` | `{_escape(row['model'])}` | {row['epoch']} "
            f"| {overall['dice']:.6f} | {overall['hits']}/{overall['findings']} "
            f"| `{row['eligibility']}` | {artifact['rank']} | `{artifact['assignment']}` "
            f"| `{artifact['status']}` | `{_escape(artifact['cache_path'])}` "
            f"| `{_escape(artifact['result_path'])}` | `{_escape(artifact['legacy'])}` "
            f"| `{artifact['test300']}` | `{row['candidate_id']}` |"
        )
    lines.extend(
        [
            "",
            "## By overall Dice",
            "",
            "| Rank | Dice | Hits | Experiment | Model / arm | Epoch | Eligibility | Frozen Top20 | Wave/GPU | Fresh val200 logits | Fresh cache | Fresh result | Legacy cache | Test300 | Candidate ID |",
            "| ---: | ---: | ---: | --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for rank, row in enumerate(by_dice, start=1):
        overall = row["metrics"]["overall"]
        artifact = _artifact_cells(row)
        lines.append(
            f"| {rank} | {overall['dice']:.6f} | {overall['hits']}/{overall['findings']} "
            f"| {row['experiment']['number']:03d} `{_escape(row['experiment']['id'])}` "
            f"| `{_escape(row['model'])}` | {row['epoch']} | `{row['eligibility']}` "
            f"| {artifact['rank']} | `{artifact['assignment']}` | `{artifact['status']}` "
            f"| `{_escape(artifact['cache_path'])}` | `{_escape(artifact['result_path'])}` "
            f"| `{_escape(artifact['legacy'])}` | `{artifact['test300']}` "
            f"| `{row['candidate_id']}` |"
        )
    return "\n".join(lines) + "\n"


def render_subcategory_leaderboard(catalog: dict[str, Any]) -> str:
    candidates = list(catalog["candidates"])
    category_order = list(catalog["dataset"]["official_category_order"])
    by_experiment = sorted(
        candidates,
        key=lambda row: (
            int(row["experiment"]["number"]), str(row["model"]), int(row["epoch"]),
            str(row["candidate_id"]),
        ),
    )
    lines = [
        "# SideExp003 Val200 Official-Category Leaderboard",
        "",
        "Generated from `checkpoint_catalog.json`; metrics are recomposed after exact "
        "`(case name, finding index)` alignment to the fixed val200 category map.",
        "Each cell is `Dice (hits/findings)`. `2f` is unavailable in this val200 split.",
        "",
        "## Wide table by experiment and model",
        "",
        "| Experiment | Model / arm | Epoch | " + " | ".join(category_order) + " | Candidate ID |",
        "| --- | --- | ---: | " + " | ".join(["---:"] * len(category_order)) + " | --- |",
    ]
    for row in by_experiment:
        category_cells = [
            _metric(row["metrics"]["categories"][category])
            for category in category_order
        ]
        lines.append(
            f"| {row['experiment']['number']:03d} `{_escape(row['experiment']['id'])}` "
            f"| `{_escape(row['model'])}` | {row['epoch']} | "
            + " | ".join(category_cells)
            + f" | `{row['candidate_id']}` |"
        )
    lines.extend(
        [
            "",
            "## Ranked within each official category",
            "",
            "| Category | Category rank | Dice | Hits | Experiment | Model / arm | Epoch | Eligibility | Candidate ID |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |",
        ]
    )
    for category in category_order:
        ranked = [
            row for row in candidates
            if row["metrics"]["categories"][category]["available"]
        ]
        ranked.sort(
            key=lambda row: (
                -float(row["metrics"]["categories"][category]["dice"]),
                -int(row["metrics"]["categories"][category]["hits"]),
                str(row["candidate_id"]),
            )
        )
        if not ranked:
            lines.append(f"| {category} | — | — | — | — | — | — | — | val200 unavailable |")
            continue
        for rank, row in enumerate(ranked, start=1):
            metric = row["metrics"]["categories"][category]
            lines.append(
                f"| {category} | {rank} | {metric['dice']:.6f} "
                f"| {metric['hits']}/{metric['findings']} "
                f"| {row['experiment']['number']:03d} `{_escape(row['experiment']['id'])}` "
                f"| `{_escape(row['model'])}` | {row['epoch']} | `{row['eligibility']}` "
                f"| `{row['candidate_id']}` |"
            )
    return "\n".join(lines) + "\n"


def validate_catalog(catalog: dict[str, Any]) -> list[str]:
    errors = []
    if catalog.get("schema_version") != fresh_cache.CATALOG_SCHEMA_VERSION:
        errors.append(
            f"checkpoint_catalog.json schema_version must be "
            f"{fresh_cache.CATALOG_SCHEMA_VERSION}"
        )
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list):
        return errors + ["checkpoint_catalog.json candidates must be a list"]
    ids = [candidate.get("candidate_id") for candidate in candidates]
    if len(ids) != len(set(ids)):
        errors.append("candidate IDs are not unique")
    checkpoint_hashes = [
        candidate.get("checkpoint", {}).get("sha256")
        for candidate in candidates
        if candidate.get("checkpoint", {}).get("sha256") is not None
    ]
    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        errors.append("checkpoint SHA-256 values are not unique")
    if catalog.get("candidate_count") != len(candidates):
        errors.append("candidate_count does not match candidates")
    eligible_count = sum(candidate.get("eligibility") == "eligible" for candidate in candidates)
    if catalog.get("eligible_count") != eligible_count:
        errors.append("eligible_count does not match candidates")
    if catalog.get("needs_review_count") != len(candidates) - eligible_count:
        errors.append("needs_review_count does not match candidates")
    dataset_record = catalog.get("dataset", {})
    if dataset_record.get("sha256") != EXPECTED_DATASET_SHA256:
        errors.append("catalog does not use the fixed val200 dataset SHA-256")
    if dataset_record.get("cases") != EXPECTED_CASES or dataset_record.get("findings") != EXPECTED_FINDINGS:
        errors.append("catalog dataset shape is not 200 cases / 381 findings")
    all_alias_paths: set[str] = set()
    for candidate in candidates:
        overall = candidate.get("metrics", {}).get("overall", {})
        categories = candidate.get("metrics", {}).get("categories", {})
        if overall.get("findings") != EXPECTED_FINDINGS:
            errors.append(f"{candidate.get('candidate_id')}: not {EXPECTED_FINDINGS} findings")
        if list(categories) != list(OFFICIAL_CATEGORY_ORDER):
            errors.append(f"{candidate.get('candidate_id')}: category order mismatch")
        available = [metric for metric in categories.values() if metric.get("available")]
        category_findings = sum(int(metric["findings"]) for metric in available)
        category_hits = sum(int(metric["hits"]) for metric in available)
        if category_findings != overall.get("findings") or category_hits != overall.get("hits"):
            errors.append(f"{candidate.get('candidate_id')}: category totals do not recompose")
        if category_findings:
            category_dice = math.fsum(
                float(metric["dice"]) * int(metric["findings"]) for metric in available
            ) / category_findings
            if not math.isclose(
                category_dice, float(overall.get("dice", math.nan)),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                errors.append(f"{candidate.get('candidate_id')}: category Dice does not recompose")
        alias_paths = [alias.get("path") for alias in candidate.get("evaluator_aliases", [])]
        if len(alias_paths) != len(set(alias_paths)) or not alias_paths:
            errors.append(f"{candidate.get('candidate_id')}: invalid evaluator aliases")
        for alias_path in alias_paths:
            if alias_path in all_alias_paths:
                errors.append(f"evaluator alias belongs to multiple checkpoints: {alias_path}")
            elif isinstance(alias_path, str):
                all_alias_paths.add(alias_path)
        if candidate.get("canonical_evaluator", {}).get("path") not in alias_paths:
            errors.append(f"{candidate.get('candidate_id')}: canonical evaluator is not an alias")
        if candidate.get("dataset", {}).get("sha256") != dataset_record.get("sha256"):
            errors.append(f"{candidate.get('candidate_id')}: dataset SHA mismatch")
        if (
            candidate.get("eligibility") == "eligible"
            and candidate.get("threshold", {}).get("status") != "verified"
        ):
            errors.append(f"{candidate.get('candidate_id')}: eligible without threshold proof")
        artifacts = candidate.get("inference_artifacts")
        if not isinstance(artifacts, dict):
            errors.append(f"{candidate.get('candidate_id')}: missing inference_artifacts")
            continue
        val200 = artifacts.get("val200")
        test300 = artifacts.get("test300")
        if not isinstance(val200, dict) or not isinstance(val200.get("versions"), list):
            errors.append(f"{candidate.get('candidate_id')}: invalid val200 artifacts")
        else:
            version_ids = [value.get("cache_version_id") for value in val200["versions"]]
            if len(version_ids) != len(set(version_ids)):
                errors.append(f"{candidate.get('candidate_id')}: duplicate cache versions")
            active = val200.get("active_cache_version")
            if active is not None and active not in version_ids:
                errors.append(f"{candidate.get('candidate_id')}: unknown active cache version")
        if (
            not isinstance(test300, dict)
            or test300.get("status") not in fresh_cache.VALID_FRESH_STATUSES | {"deferred"}
            or test300.get("metric_status") != "not_available_withheld_labels"
            or test300.get("dataset", {}).get("sha256") != fresh_cache.TEST300_SHA256
        ):
            errors.append(f"{candidate.get('candidate_id')}: invalid test300 contract")
        elif test300.get("status") != "deferred":
            versions = test300.get("versions", [])
            ids = [v.get("cache_version_id") for v in versions]
            if len(ids) != len(set(ids)) or test300.get("active_cache_version") not in [None, *ids]:
                errors.append(f"{candidate.get('candidate_id')}: invalid test cache versions")
            for version in versions:
                if version.get("metrics_path") is not None or version.get("metric_status") != "not_available_withheld_labels":
                    errors.append(f"{candidate.get('candidate_id')}: test metrics must be unavailable")
                if version.get("strict_eligible") and (version.get("cases"), version.get("findings")) != (300, 582):
                    errors.append(f"{candidate.get('candidate_id')}: incomplete test cache")
    return errors


def _paths_for_experiment(argument: str, runtime_root: Path) -> list[Path]:
    match = re.fullmatch(r"(?:exp)?(\d{1,3})", argument.lower())
    if not match:
        raise HubError("--experiment must be like 027, 27, or exp027")
    number = int(match.group(1))
    experiment_dirs = sorted(runtime_root.glob(f"{number:03d}_*"))
    if len(experiment_dirs) != 1:
        raise HubError(
            f"expected one runtime experiment for {number:03d}, found "
            f"{[str(path) for path in experiment_dirs]}"
        )
    paths = sorted(experiment_dirs[0].glob(
        "runs/**/eval_epoch*_val200/eval/val_quick_global_eval.json"
    ))
    if not paths:
        raise HubError(f"no fixed-val200 evaluator found under {experiment_dirs[0]}")
    return paths


def _catalog_summary(catalog: dict[str, Any], *, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "candidate_count": catalog["candidate_count"],
        "eligible_count": catalog["eligible_count"],
        "needs_review_count": catalog["needs_review_count"],
        "dataset": catalog["dataset"],
        "outputs": [str(CATALOG_PATH), str(LEADERBOARD_PATH), str(SUBCATEGORY_PATH)],
    }


def load_evaluator_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = read_json(path)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise HubError(f"{path}: override schema_version must be 1")
    overrides = data.get("evaluator_overrides")
    if not isinstance(overrides, dict):
        raise HubError(f"{path}: evaluator_overrides must be an object")
    if not all(isinstance(key, str) and isinstance(value, dict) for key, value in overrides.items()):
        raise HubError(f"{path}: malformed evaluator override")
    return overrides


def write_catalog_outputs(catalog: dict[str, Any]) -> None:
    errors = validate_catalog(catalog)
    if errors:
        raise HubError("refusing to write invalid catalog:\n- " + "\n- ".join(errors))
    atomic_write_json(CATALOG_PATH, catalog)
    atomic_write_text(LEADERBOARD_PATH, render_checkpoint_leaderboard(catalog))
    atomic_write_text(SUBCATEGORY_PATH, render_subcategory_leaderboard(catalog))


def catalog_command(args: argparse.Namespace) -> int:
    existing = read_json(CATALOG_PATH) if CATALOG_PATH.is_file() else None
    known = known_checkpoint_hashes(existing)
    overrides = load_evaluator_overrides(Path(args.overrides))
    if args.catalog_action == "rebuild":
        paths = sorted(Path(args.runtime_root).glob(EVALUATOR_GLOB))
        if not paths:
            raise HubError(f"no evaluators matched under {args.runtime_root}")
        candidates, dataset = scan_catalog_candidates(
            paths,
            dataset_path=Path(args.dataset),
            runtime_root=Path(args.runtime_root),
            known_checkpoint_hashes=known,
            evaluator_overrides=overrides,
        )
        catalog = make_catalog(candidates, dataset, existing=existing)
    else:
        paths = _paths_for_experiment(args.experiment, Path(args.runtime_root))
        new_candidates, dataset = scan_catalog_candidates(
            paths,
            dataset_path=Path(args.dataset),
            runtime_root=Path(args.runtime_root),
            # A requested experiment is always re-hashed. Existing catalog
            # hashes only accelerate full rebuilds of untouched sources.
            known_checkpoint_hashes={},
            evaluator_overrides=overrides,
        )
        number = int(re.fullmatch(r"(?:exp)?(\d{1,3})", args.experiment.lower()).group(1))
        retained = [] if existing is None else [
            candidate for candidate in existing.get("candidates", [])
            if int(candidate["experiment"]["number"]) != number
        ]
        by_sha = {
            candidate["checkpoint"]["sha256"]: candidate
            for candidate in retained
            if candidate["checkpoint"]["sha256"]
        }
        for candidate in new_candidates:
            checksum = candidate["checkpoint"]["sha256"]
            duplicate = by_sha.get(checksum) if checksum else None
            if duplicate:
                if not _metrics_equal(duplicate["metrics"], candidate["metrics"]):
                    raise CatalogConflict(
                        "new evaluator conflicts with an existing checkpoint SHA",
                        {
                            "schema_version": 1,
                            "action": "choose_canonical_evaluator",
                            "checkpoint_sha256": checksum,
                            "existing_candidate_id": duplicate["candidate_id"],
                            "new_candidate_id": candidate["candidate_id"],
                            "canonical_evaluator_path": None,
                        },
                    )
                aliases = {
                    alias["path"]: alias
                    for alias in duplicate["evaluator_aliases"] + candidate["evaluator_aliases"]
                }
                duplicate["evaluator_aliases"] = [aliases[key] for key in sorted(aliases)]
            else:
                retained.append(candidate)
                if checksum:
                    by_sha[checksum] = candidate
        catalog = make_catalog(
            sorted(retained, key=lambda row: row["candidate_id"]),
            dataset,
            existing=existing,
        )

    print(json.dumps(_catalog_summary(catalog, mode="dry-run" if args.dry_run else "apply"), indent=2))
    if args.apply:
        write_catalog_outputs(catalog)
    return 0


def inventory_logit_tree(root: Path) -> dict[str, Any]:
    physical = root.resolve(strict=True)
    if not physical.is_dir():
        raise HubError(f"legacy logit root is not a directory: {root}")
    candidate_dirs = sorted(path for path in physical.iterdir() if path.is_dir())
    arrays = sorted(physical.rglob("*.npy"))
    manifests = sorted(physical.glob("*/export_manifest.json"))
    manifest_rows = []
    resolvable_arrays = 0
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        cases = manifest.get("cases", [])
        for case in cases:
            array_path = case.get("array_path") if isinstance(case, dict) else None
            if isinstance(array_path, str) and Path(array_path).is_file():
                resolvable_arrays += 1
        manifest_rows.append(
            {
                "candidate_id": manifest.get("candidate_id", manifest_path.parent.name),
                "relative_path": str(manifest_path.relative_to(physical)),
                "sha256": sha256_file(manifest_path),
                "case_count": manifest.get("case_count"),
                "finding_count": manifest.get("finding_count"),
                "dtype": manifest.get("dtype"),
                "status": manifest.get("status"),
            }
        )
    return {
        "logical_path": str(root),
        "physical_path": str(physical),
        "candidate_count": len(candidate_dirs),
        "candidate_ids": [path.name for path in candidate_dirs],
        "array_count": len(arrays),
        "array_bytes": sum(path.stat().st_size for path in arrays),
        "export_manifest_count": len(manifests),
        "manifest_array_paths_resolvable": resolvable_arrays,
        "export_manifests": manifest_rows,
    }


def _inventory_signature(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_count": inventory["candidate_count"],
        "candidate_ids": inventory["candidate_ids"],
        "array_count": inventory["array_count"],
        "array_bytes": inventory["array_bytes"],
        "export_manifest_count": inventory["export_manifest_count"],
        "manifest_array_paths_resolvable": inventory["manifest_array_paths_resolvable"],
        "export_manifests": inventory["export_manifests"],
    }


def _active_sideexp002_processes() -> list[dict[str, Any]]:
    dangerous = (
        "export_logits.py", "launch_logit_exports.py", "search_ensembles.py",
        "finalize_when_complete.py", "wait_then_launch.py",
    )
    active = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(name in command for name in dangerous):
            active.append({"pid": int(proc.name), "command": command.strip()})
    return sorted(active, key=lambda row: row["pid"])


def _nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise HubError(f"cannot find existing parent for {path}")
        current = current.parent
    return current


def migrate_legacy_cache(
    source: Path = LEGACY_SOURCE,
    destination: Path = LEGACY_DESTINATION,
    *,
    apply: bool = False,
    fail_after_move: bool = False,
) -> dict[str, Any]:
    active = _active_sideexp002_processes()
    if active:
        raise HubError(f"SideExp002 exporter/search process is active: {active}")

    if source.is_symlink() and destination.is_dir():
        if source.resolve() != destination.resolve():
            raise HubError(f"legacy symlink points to the wrong destination: {source}")
        if apply:
            (destination.parent / "by_cache_key").mkdir(parents=True, exist_ok=True)
        inventory = inventory_logit_tree(source)
        return {
            "schema_version": 1,
            "status": "already_migrated",
            "migrated_at_utc": None,
            "source_logical_path": str(source),
            "destination_physical_path": str(destination),
            "compatibility_symlink": str(source),
            "inventory": inventory,
            "verification": {
                "same_mount_atomic_rename": True,
                "old_logical_path_resolves": True,
                "new_physical_path_resolves": True,
                "inventory_match": True,
                "export_manifest_hashes_match": True,
                "manifest_array_paths_resolve": (
                    inventory["manifest_array_paths_resolvable"] == inventory["array_count"]
                ),
            },
        }
    if source.is_symlink():
        raise HubError(f"source is a symlink but destination is unavailable: {source}")
    if not source.is_dir():
        raise HubError(f"legacy source does not exist as a physical directory: {source}")
    if destination.exists() or destination.is_symlink():
        raise HubError(f"migration destination already exists: {destination}")

    old_inventory = inventory_logit_tree(source)
    source_device = source.stat().st_dev
    destination_device = _nearest_existing(destination.parent).stat().st_dev
    if source_device != destination_device:
        raise HubError("source and destination are not on the same filesystem")
    plan = {
        "schema_version": 1,
        "status": "dry_run" if not apply else "migrated",
        "migrated_at_utc": None,
        "source_logical_path": str(source),
        "source_physical_path_before_move": str(source.resolve()),
        "destination_physical_path": str(destination),
        "compatibility_symlink": str(source),
        "inventory": old_inventory,
        "verification": {
            "same_mount_atomic_rename": True,
            "old_logical_path_resolves": False,
            "new_physical_path_resolves": False,
            "inventory_match": False,
            "export_manifest_hashes_match": False,
            "manifest_array_paths_resolve": False,
        },
    }
    if not apply:
        return plan

    destination.parent.mkdir(parents=True, exist_ok=True)
    (destination.parent / "by_cache_key").mkdir(parents=True, exist_ok=True)
    moved = False
    linked = False
    try:
        os.rename(source, destination)
        moved = True
        if fail_after_move:
            raise HubError("injected post-move failure")
        os.symlink(destination, source, target_is_directory=True)
        linked = True
        new_inventory = inventory_logit_tree(destination)
        logical_inventory = inventory_logit_tree(source)
        expected = _inventory_signature(old_inventory)
        new_matches = _inventory_signature(new_inventory) == expected
        logical_matches = _inventory_signature(logical_inventory) == expected
        manifests_resolve = (
            logical_inventory["manifest_array_paths_resolvable"]
            == logical_inventory["array_count"]
        )
        if not new_matches or not logical_matches or not manifests_resolve:
            raise HubError(
                "post-migration inventory or immutable-manifest path verification failed"
            )
        plan["migrated_at_utc"] = utc_now()
        plan["inventory"] = new_inventory
        plan["verification"] = {
            "same_mount_atomic_rename": True,
            "old_logical_path_resolves": source.is_dir() and source.resolve() == destination.resolve(),
            "new_physical_path_resolves": destination.is_dir(),
            "inventory_match": new_matches and logical_matches,
            "export_manifest_hashes_match": True,
            "manifest_array_paths_resolve": manifests_resolve,
        }
        runtime_manifest = destination.parents[2] / "legacy_import_manifest.json"
        atomic_write_json(runtime_manifest, plan)
        return plan
    except Exception:
        if linked and source.is_symlink():
            source.unlink()
        if moved and destination.exists() and not source.exists():
            os.rename(destination, source)
        raise


def _check_import_manifest(manifest: dict[str, Any]) -> list[str]:
    errors = []
    status = manifest.get("status")
    if status in {"pending", "not_started"}:
        return errors
    source = Path(str(manifest.get("source_logical_path")))
    destination = Path(str(manifest.get("destination_physical_path")))
    if not source.is_symlink():
        errors.append(f"legacy compatibility path is not a symlink: {source}")
    elif not destination.is_dir() or source.resolve() != destination.resolve():
        errors.append("legacy compatibility symlink does not resolve to physical cache")
    expected = manifest.get("inventory", {})
    if destination.is_dir():
        observed = inventory_logit_tree(destination)
        if _inventory_signature(observed) != _inventory_signature(expected):
            errors.append("legacy cache inventory differs from import manifest")
    return errors


def check_command(args: argparse.Namespace) -> int:
    catalog = read_json(CATALOG_PATH)
    errors = validate_catalog(catalog)
    load_evaluator_overrides(OVERRIDES_PATH)
    dataset = load_dataset_contract(Path(catalog["dataset"]["path"]))
    if dataset["sha256"] != catalog["dataset"]["sha256"]:
        errors.append("catalog dataset SHA-256 differs from disk")
    expected_leaderboard = render_checkpoint_leaderboard(catalog)
    expected_subcategory = render_subcategory_leaderboard(catalog)
    if not LEADERBOARD_PATH.is_file() or LEADERBOARD_PATH.read_text() != expected_leaderboard:
        errors.append("val200_checkpoint_leaderboard.md is stale")
    if not SUBCATEGORY_PATH.is_file() or SUBCATEGORY_PATH.read_text() != expected_subcategory:
        errors.append("val200_subcategory_leaderboard.md is stale")
    for candidate in catalog["candidates"]:
        for field in ("checkpoint", "config", "canonical_evaluator"):
            path_value = candidate.get(field, {}).get("path")
            if path_value and not Path(path_value).is_file():
                errors.append(f"{candidate['candidate_id']}: missing {field} path {path_value}")
        hash_fields = ["config", "canonical_evaluator"]
        if args.deep:
            hash_fields.append("checkpoint")
        for field in hash_fields:
            source = candidate.get(field, {})
            if source.get("path") and source.get("sha256"):
                observed = sha256_file(Path(source["path"]))
                if observed != source["sha256"]:
                    errors.append(f"{candidate['candidate_id']}: {field} SHA drift")
        for alias in candidate.get("evaluator_aliases", []):
            if alias.get("path") and alias.get("sha256"):
                observed = sha256_file(Path(alias["path"]))
                if observed != alias["sha256"]:
                    errors.append(f"{candidate['candidate_id']}: evaluator alias SHA drift")
        preprocessing = candidate.get("preprocessing", {})
        manifest_path = preprocessing.get("manifest_path")
        manifest_sha = preprocessing.get("manifest_sha256")
        if manifest_path and manifest_sha:
            if not Path(manifest_path).is_file():
                errors.append(f"{candidate['candidate_id']}: preprocessing manifest missing")
            elif sha256_file(Path(manifest_path)) != manifest_sha:
                errors.append(f"{candidate['candidate_id']}: preprocessing manifest SHA drift")
        artifacts = candidate.get("inference_artifacts", {})
        for version in [v for split in ("val200", "test300") for v in artifacts.get(split, {}).get("versions", [])]:
            if version.get("status") == "strict_passed":
                for field in ("physical_path", "export_manifest_path", "validation_path"):
                    value = version.get(field)
                    if not value or not Path(value).exists():
                        errors.append(
                            f"{candidate['candidate_id']}: strict cache missing {field} {value}"
                        )
    for roster_path in sorted((HERE / "rosters").glob("roster_*.json")):
        roster = read_json(roster_path)
        if roster.get("roster_kind") == "merged_fresh_analysis":
            roster_errors = seeded_caruana.validate_merged_roster(roster)
        elif roster.get("source_job") and roster.get("reuse_policy") == "fresh_only":
            roster_errors = preliminary_ensemble.validate_derived_roster(roster)
        else:
            roster_errors = fresh_cache.validate_roster(roster)
        errors.extend(f"{roster_path.name}: {error}" for error in roster_errors)
    for job_path in sorted((HERE / "cache_jobs").glob("*/job_spec.json")):
        job = read_json(job_path)
        job_errors = fresh_cache.validate_job(job)
        if job.get("kind") == "test300_base_logits":
            import test300_cache
            try:
                test300_cache.validate_job(job)
            except fresh_cache.FreshCacheError as exc:
                job_errors.append(str(exc))
        errors.extend(f"{job_path.parent.name}: {error}" for error in job_errors)
    for job_path in sorted((HERE / "analysis_jobs").glob("*/job_spec.json")):
        job_errors = seeded_caruana.validate_launch_spec(read_json(job_path))
        errors.extend(f"{job_path.parent.name}: {error}" for error in job_errors)
    if IMPORT_MANIFEST_PATH.is_file():
        errors.extend(_check_import_manifest(read_json(IMPORT_MANIFEST_PATH)))
    if errors:
        print("SideExp003 check FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"SideExp003 check passed: {len(catalog['candidates'])} unique checkpoints, "
        f"{catalog['eligible_count']} eligible, {catalog['needs_review_count']} needs_review."
    )
    return 0


def _roster_path(value: str) -> Path:
    path = Path(value)
    if path.suffix == ".json" or "/" in value:
        return path if path.is_absolute() else REPO_ROOT / path
    name = value if value.startswith("roster_") else f"roster_{value}"
    return HERE / "rosters" / f"{name}.json"


def _job_path(value: str) -> Path:
    path = Path(value)
    if path.suffix == ".json" or "/" in value:
        return path if path.is_absolute() else REPO_ROOT / path
    return HERE / "cache_jobs" / value / "job_spec.json"


def _docker_image_id(image: str) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise HubError(f"cannot inspect container image {image}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _render_roster(roster: dict[str, Any]) -> str:
    lines = [
        f"# {roster['roster_id']}",
        "",
        f"Frozen from catalog `{roster['catalog']['sha256']}` at `{roster['created_at_utc']}`.",
        f"Roster SHA-256: `{roster['roster_sha256']}`. Reuse policy: `fresh_only`.",
        "",
        "| Rank | Dice | Hits | Experiment | Model | Epoch | Checkpoint SHA | Preprocessing |",
        "| ---: | ---: | ---: | --- | --- | ---: | --- | --- |",
    ]
    for candidate in roster["candidates"]:
        performance = candidate["performance"]
        lines.append(
            f"| {candidate['rank']} | {performance['dice']:.6f} "
            f"| {performance['hits']}/{performance['findings']} "
            f"| `{candidate['experiment_name']}` | `{_escape(candidate['model_id'])}` "
            f"| {candidate['epoch']} | `{candidate['checkpoint']['sha256']}` "
            f"| `{candidate['cache']['id']}` |"
        )
    return "\n".join(lines) + "\n"


def roster_command(args: argparse.Namespace) -> int:
    if args.roster_action != "freeze":
        raise HubError(f"unsupported roster action {args.roster_action}")
    roster_path = _roster_path(args.roster_id)
    if roster_path.is_file():
        roster = read_json(roster_path)
        errors = fresh_cache.validate_roster(roster)
        if errors:
            raise HubError("existing roster is invalid: " + "; ".join(errors))
        print(json.dumps({"status": "already_frozen", "path": str(roster_path), "roster_sha256": roster["roster_sha256"]}, indent=2))
        return 0
    catalog_hash = sha256_file(CATALOG_PATH)
    if args.expected_catalog_sha256 and catalog_hash != args.expected_catalog_sha256:
        raise HubError(
            f"catalog snapshot drift: expected {args.expected_catalog_sha256}, got {catalog_hash}"
        )
    catalog = read_json(CATALOG_PATH)
    image_id = _docker_image_id(args.container_image)
    roster = fresh_cache.build_roster(
        catalog,
        catalog_sha256=catalog_hash,
        roster_id=args.roster_id,
        top=args.top,
        container_image=args.container_image,
        container_image_id=image_id,
    )
    source_errors = fresh_cache.validate_real_sources(
        roster, verify_checkpoints=not args.skip_checkpoint_rehash
    )
    if source_errors:
        raise HubError("roster source preflight failed:\n- " + "\n- ".join(source_errors))
    fresh_cache.atomic_write_json(roster_path, roster)
    fresh_cache.atomic_write_text(roster_path.with_suffix(".md"), _render_roster(roster))
    print(
        json.dumps(
            {
                "status": "frozen",
                "path": str(roster_path),
                "roster_sha256": roster["roster_sha256"],
                "N": roster["N"],
                "reuse_policy": roster["reuse_policy"],
            },
            indent=2,
        )
    )
    return 0


def _render_job(job: dict[str, Any]) -> str:
    lines = [
        f"# {job['job_id']} Fresh {job.get('dataset', {}).get('split', 'val200')} Logit Export",
        "",
        f"Roster: `{job['roster_id']}` (`{job['roster_sha256']}`).",
        f"Job-spec SHA-256: `{job['job_spec_sha256']}`.",
        "Reuse policy: `fresh_only`; legacy caches are neither read nor used as fallback.",
        "",
        "| Wave | GPU | Rank | Candidate | Cache key |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for candidate in job["candidates"]:
        lines.append(
            f"| {candidate['wave']} | {candidate['gpu']} | {candidate['rank']} "
            f"| `{candidate['id']}` | `{candidate['cache_key']}` |"
        )
    lines.extend(
        [
            "",
            "The launcher requires 20,000 MiB free before loading each wave and 4,096 MiB "
            "free after the retained largest-case smoke. It automatically continues only "
            "after every member of the previous wave is `strict_passed`.",
        ]
    )
    continuation = job.get("continuation")
    if isinstance(continuation, dict):
        audit = continuation["source_drift_audit"]
        lines.extend(
            [
                "",
                "## Audited continuation",
                "",
                f"Parent job: `{continuation['parent_job_id']}` "
                f"(`{continuation['parent_job_spec_sha256']}`).",
                f"Parent source bundle: `{continuation['parent_source_bundle_sha256']}`.",
                f"Continuation source bundle: `{job['source_bundle']['sha256']}`.",
                f"Source-drift audit: `{audit['audit_sha256']}`.",
                "Completed parent caches remain immutable; every continuation candidate "
                "uses a new cache key bound to the continuation source bundle.",
            ]
        )
    return "\n".join(lines) + "\n"


def cache_plan_command(args: argparse.Namespace) -> int:
    import test300_cache

    if args.reuse_policy != "fresh-only":
        raise HubError("this job requires --reuse-policy fresh-only")
    roster_path = _roster_path(args.roster)
    roster = read_json(roster_path)
    job_path = _job_path(args.job_id)
    if job_path.is_file():
        job = read_json(job_path)
        errors = fresh_cache.validate_job(job)
        if errors:
            raise HubError("existing job is invalid: " + "; ".join(errors))
        if job["dataset"].get("split", "val200") != args.split or job["roster_sha256"] != roster["roster_sha256"]:
            raise HubError("existing job split/roster differs from requested inputs")
        if args.split == "test300":
            test300_cache.validate_job(job)
            if args.staging_root and Path(job["staging_root"]).parent != args.staging_root:
                raise HubError("existing test job has a different staging root")
            if args.native_cache_root and any(c["cache"]["id"] == "crop_zscore_native_v1" and Path(c["cache"]["root"]) != args.native_cache_root for c in job["candidates"]):
                raise HubError("existing test job has a different native cache root")
    elif args.split == "test300":
        if args.wave_size != 4:
            raise HubError("test300 requires four-GPU waves")
        job = test300_cache.build_job(roster, args.job_id,
            staging_root=args.staging_root or test300_cache.STAGING_ROOT,
            native_root=args.native_cache_root or test300_cache.NATIVE_ROOT)
    else:
        job = fresh_cache.build_job(
            roster, job_id=args.job_id, wave_size=args.wave_size, gpus=range(args.wave_size)
        )
    source_errors = fresh_cache.validate_real_sources(
        roster, verify_checkpoints=not args.skip_checkpoint_rehash
    )
    if source_errors:
        raise HubError("cache-plan source preflight failed:\n- " + "\n- ".join(source_errors))
    for candidate in job["candidates"]:
        cache_root = Path(candidate["cache_root"])
        if not cache_root.exists():
            continue
        manifest_path = cache_root / "export_manifest.json"
        if not manifest_path.is_file():
            raise HubError(f"fresh cache target exists without manifest: {cache_root}")
        manifest = read_json(manifest_path)
        if manifest.get("job_id") != job["job_id"]:
            raise HubError(f"fresh cache target belongs to another job: {cache_root}")
    summary = {
        "status": "dry_run" if args.dry_run else "planned",
        "job_id": job["job_id"],
        "job_path": str(job_path),
        "roster": roster["roster_id"],
        "candidates": len(job["candidates"]),
        "waves": len(job["waves"]),
        "reuse_policy": job["reuse_policy"],
    }
    print(json.dumps(summary, indent=2))
    if args.apply:
        if not job_path.is_file():
            fresh_cache.atomic_write_json(job_path, job)
            fresh_cache.atomic_write_text(job_path.with_name("job_spec.md"), _render_job(job))
        catalog = (test300_cache.sync_catalog(read_json(CATALOG_PATH), job) if args.split == "test300"
                   else fresh_cache.apply_job_to_catalog(read_json(CATALOG_PATH), job))
        write_catalog_outputs(catalog)
    return 0


def cache_sync_command(args: argparse.Namespace) -> int:
    job_path = _job_path(args.job)
    job = read_json(job_path)
    progress = {}
    for candidate in job["candidates"]:
        path = Path(candidate["progress_path"])
        if path.is_file():
            value = read_json(path)
            if value.get("job_id") == job["job_id"]:
                progress[candidate["id"]] = value
    if job.get("kind") == "test300_base_logits":
        import test300_cache
        test300_cache.validate_job(job)
        catalog = test300_cache.sync_catalog(read_json(CATALOG_PATH), job)
    else:
        catalog = fresh_cache.update_catalog_progress(read_json(CATALOG_PATH), job, progress)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "job_id": job["job_id"],
                "progress_records": len(progress),
                "strict_passed": sum(value.get("status") == "strict_passed" for value in progress.values()),
            },
            indent=2,
        )
    )
    if args.apply:
        write_catalog_outputs(catalog)
    return 0


def _continuation_audit(
    parent_job: dict[str, Any], start_rank: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    current_bundle = fresh_cache.source_bundle()
    old_files = {
        value["path"]: value["sha256"]
        for value in parent_job["source_bundle"]["files"]
    }
    new_files = {
        value["path"]: value["sha256"] for value in current_bundle["files"]
    }
    changed_paths = sorted(
        path
        for path in set(old_files) | set(new_files)
        if old_files.get(path) != new_files.get(path)
    )
    unexpected = [
        path for path in changed_paths if path not in CONTINUATION_ALLOWED_SOURCE_DRIFT
    ]
    if unexpected:
        raise HubError(
            "continuation source drift includes unaudited inference files: "
            + ", ".join(unexpected)
        )
    if not changed_paths:
        raise HubError("continuation source bundle did not change")

    progress: dict[str, dict[str, Any]] = {}
    parent_evidence = []
    for candidate in parent_job["candidates"]:
        if int(candidate["rank"]) >= start_rank:
            continue
        progress_path = Path(candidate["progress_path"])
        if not progress_path.is_file():
            raise HubError(f"rank {candidate['rank']}: missing parent progress")
        record = read_json(progress_path)
        if record.get("job_id") != parent_job["job_id"] or record.get("status") != "strict_passed":
            raise HubError(f"rank {candidate['rank']}: parent cache is not strict_passed")
        validation_path = Path(candidate["cache_root"]) / "reproduction_validation.json"
        manifest_path = Path(candidate["cache_root"]) / "export_manifest.json"
        if not validation_path.is_file() or not manifest_path.is_file():
            raise HubError(f"rank {candidate['rank']}: parent cache evidence is incomplete")
        validation = read_json(validation_path)
        if (
            validation.get("status") != "passed"
            or validation.get("cases") != 200
            or validation.get("findings") != 381
            or not validation.get("array_hashes_verified")
        ):
            raise HubError(f"rank {candidate['rank']}: parent validation gate is incomplete")
        progress[candidate["id"]] = record
        parent_evidence.append(
            {
                "rank": candidate["rank"],
                "candidate_id": candidate["id"],
                "cache_key": candidate["cache_key"],
                "dtype": validation.get("dtype"),
                "cases": validation.get("cases"),
                "findings": validation.get("findings"),
                "export_manifest_sha256": sha256_file(manifest_path),
                "validation_sha256": sha256_file(validation_path),
            }
        )
    expected_parent_ranks = list(range(1, start_rank))
    if [int(value["rank"]) for value in parent_evidence] != expected_parent_ranks:
        raise HubError("parent strict-cache prefix is not rank-contiguous")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    audit = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "decision": "approved_by_user_2026-09-08",
        "parent_job_id": parent_job["job_id"],
        "parent_job_spec_sha256": parent_job["job_spec_sha256"],
        "parent_source_bundle_sha256": parent_job["source_bundle"]["sha256"],
        "new_source_bundle_sha256": current_bundle["sha256"],
        "current_repo_commit": commit,
        "changed_files": [
            {
                "path": path,
                "parent_sha256": old_files.get(path),
                "continuation_sha256": new_files.get(path),
                "classification": CONTINUATION_ALLOWED_SOURCE_DRIFT[path],
            }
            for path in changed_paths
        ],
        "protected_worker_sources_unchanged": sorted(
            path for path in old_files if path not in changed_paths
        ),
        "parent_strict_cache_evidence": parent_evidence,
        "policy": (
            "The continuation uses new cache keys bound to the new source bundle; "
            "parent cache paths and manifests remain immutable."
        ),
    }
    audit["audit_sha256"] = json_sha256(audit)
    audit["source_bundle"] = current_bundle
    return audit, progress


def cache_continue_command(args: argparse.Namespace) -> int:
    parent_path = _job_path(args.parent_job)
    parent_job = read_json(parent_path)
    parent_errors = fresh_cache.validate_job(parent_job)
    if parent_errors:
        raise HubError("parent job is invalid: " + "; ".join(parent_errors))
    job_path = _job_path(args.job_id)
    if job_path.is_file():
        job = read_json(job_path)
        errors = fresh_cache.validate_job(job)
        if errors:
            raise HubError("existing continuation job is invalid: " + "; ".join(errors))
        audit = job["continuation"]["source_drift_audit"]
        _unused_audit, parent_progress = _continuation_audit(
            parent_job, int(job["continuation"]["rank_start"])
        )
        if job["source_bundle"]["sha256"] != fresh_cache.source_bundle()["sha256"]:
            raise HubError("existing continuation source bundle has drifted")
    else:
        audit, parent_progress = _continuation_audit(parent_job, args.start_rank)
        job = fresh_cache.build_continuation_job(
            parent_job,
            job_id=args.job_id,
            start_rank=args.start_rank,
            current_source_bundle=audit["source_bundle"],
            source_drift_audit={
                key: value for key, value in audit.items() if key != "source_bundle"
            },
        )

    roster = read_json(_roster_path(parent_job["roster_id"]))
    source_errors = fresh_cache.validate_real_sources(
        roster, verify_checkpoints=not args.skip_checkpoint_rehash
    )
    if source_errors:
        raise HubError(
            "continuation source preflight failed:\n- " + "\n- ".join(source_errors)
        )
    for candidate in job["candidates"]:
        cache_root = Path(candidate["cache_root"])
        if cache_root.exists():
            manifest_path = cache_root / "export_manifest.json"
            if not manifest_path.is_file():
                raise HubError(f"continuation cache target exists without manifest: {cache_root}")
            manifest = read_json(manifest_path)
            if manifest.get("job_id") != job["job_id"]:
                raise HubError(f"continuation cache target belongs to another job: {cache_root}")

    summary = {
        "status": "dry_run" if args.dry_run else "planned",
        "job_id": job["job_id"],
        "parent_job_id": parent_job["job_id"],
        "rank_range": [job["candidates"][0]["rank"], job["candidates"][-1]["rank"]],
        "waves": [value["wave"] for value in job["waves"]],
        "candidates": len(job["candidates"]),
        "parent_strict_passed": len(parent_progress),
        "parent_source_bundle_sha256": parent_job["source_bundle"]["sha256"],
        "new_source_bundle_sha256": job["source_bundle"]["sha256"],
        "changed_files": [value["path"] for value in audit["changed_files"]],
    }
    print(json.dumps(summary, indent=2))
    if args.apply and not job_path.is_file():
        audit_path = job_path.with_name("source_drift_audit.json")
        fresh_cache.atomic_write_json(
            audit_path,
            {key: value for key, value in audit.items() if key != "source_bundle"},
        )
        fresh_cache.atomic_write_json(job_path, job)
        fresh_cache.atomic_write_text(job_path.with_name("job_spec.md"), _render_job(job))
        catalog = fresh_cache.update_catalog_progress(
            read_json(CATALOG_PATH), parent_job, parent_progress
        )
        catalog = fresh_cache.apply_job_to_catalog(catalog, job)
        write_catalog_outputs(catalog)
    return 0


def cache_run_command(args: argparse.Namespace) -> int:
    job_path = _job_path(args.job)
    job = read_json(job_path)
    if args.auto_continue and not job.get("auto_continue"):
        raise HubError("job spec does not authorize auto-continue")
    if job.get("kind") == "test300_base_logits":
        from test300_runner import run_job
    else:
        from fresh_orchestrator import run_job

    return int(run_job(job_path))


def cache_watch_command(args: argparse.Namespace) -> int:
    path = _job_path(args.job)
    if read_json(path).get("kind") == "test300_base_logits":
        from test300_runner import watch
        return int(watch(path, args.interval, args.once))
    from fresh_orchestrator import watch_job

    return int(watch_job(_job_path(args.job), args.interval, args.once))


def ensemble_compare_command(args: argparse.Namespace) -> int:
    from preliminary_ensemble import compare_command

    args.roster = str(_roster_path(args.roster))
    args.job = str(_job_path(args.job))
    return int(compare_command(args))


def ensemble_uniform_command(args: argparse.Namespace) -> int:
    from preliminary_ensemble import uniform_command

    args.roster = str(_roster_path(args.roster))
    args.job = str(_job_path(args.job))
    return int(uniform_command(args))


def ensemble_seeded_plan_command(args: argparse.Namespace) -> int:
    args.roster = str(_roster_path(args.roster))
    args.source_job = [str(_job_path(value)) for value in args.source_job]
    return int(seeded_caruana.plan_command(args))


def ensemble_seeded_run_command(args: argparse.Namespace) -> int:
    return int(
        seeded_caruana.run_waiter(
            seeded_caruana._analysis_job_path(args.job),
            int(args.poll_interval),
            bool(args.auto_start),
        )
    )


def ensemble_seeded_watch_command(args: argparse.Namespace) -> int:
    return int(seeded_caruana.watch_command(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="manage the checkpoint catalog")
    catalog_sub = catalog.add_subparsers(dest="catalog_action", required=True)
    for action in ("add", "rebuild"):
        command = catalog_sub.add_parser(action)
        if action == "add":
            command.add_argument("--experiment", required=True)
        command.add_argument("--dataset", default=str(DATASET_PATH))
        command.add_argument("--runtime-root", default=str(EXPERIMENT_RUNTIME_ROOT))
        command.add_argument("--overrides", default=str(OVERRIDES_PATH))
        mode = command.add_mutually_exclusive_group(required=True)
        mode.add_argument("--dry-run", action="store_true")
        mode.add_argument("--apply", action="store_true")
        command.set_defaults(func=catalog_command)

    roster = subparsers.add_parser("roster", help="freeze immutable checkpoint rosters")
    roster_sub = roster.add_subparsers(dest="roster_action", required=True)
    freeze = roster_sub.add_parser("freeze")
    freeze.add_argument("--top", type=int, required=True)
    freeze.add_argument("--metric", choices=["overall_dice"], required=True)
    freeze.add_argument("--roster-id", required=True)
    freeze.add_argument("--container-image", default=fresh_cache.EXPECTED_CONTAINER_IMAGE)
    freeze.add_argument(
        "--expected-catalog-sha256", default=EXPECTED_TOP20_CATALOG_SHA256
    )
    freeze.add_argument("--skip-checkpoint-rehash", action="store_true")
    freeze.set_defaults(func=roster_command)

    cache = subparsers.add_parser("cache", help="manage shared logit cache")
    cache_sub = cache.add_subparsers(dest="cache_action", required=True)
    migrate = cache_sub.add_parser("migrate-sideexp002")
    migrate.add_argument("--source", default=str(LEGACY_SOURCE))
    migrate.add_argument("--destination", default=str(LEGACY_DESTINATION))
    mode = migrate.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    plan = cache_sub.add_parser("plan", help="create a fresh-only cache job")
    plan.add_argument("--roster", required=True)
    plan.add_argument("--job-id", default="j001_top20_val200_fresh")
    plan.add_argument("--split", choices=["val200", "test300"], required=True)
    plan.add_argument("--wave-size", type=int, default=4)
    plan.add_argument("--reuse-policy", choices=["fresh-only"], required=True)
    plan.add_argument("--skip-checkpoint-rehash", action="store_true")
    plan.add_argument("--staging-root", type=Path, help="test300 local staging base")
    plan.add_argument("--native-cache-root", type=Path, help="test-only native preprocessing root")
    plan_mode = plan.add_mutually_exclusive_group(required=True)
    plan_mode.add_argument("--dry-run", action="store_true")
    plan_mode.add_argument("--apply", action="store_true")
    plan.set_defaults(func=cache_plan_command)

    continuation = cache_sub.add_parser(
        "continue", help="create an audited new-source continuation job"
    )
    continuation.add_argument("--parent-job", required=True)
    continuation.add_argument("--job-id", required=True)
    continuation.add_argument("--start-rank", type=int, required=True)
    continuation.add_argument("--skip-checkpoint-rehash", action="store_true")
    continuation_mode = continuation.add_mutually_exclusive_group(required=True)
    continuation_mode.add_argument("--dry-run", action="store_true")
    continuation_mode.add_argument("--apply", action="store_true")
    continuation.set_defaults(func=cache_continue_command)

    run = cache_sub.add_parser("run", help="run the guarded fresh cache waves")
    run.add_argument("--job", required=True)
    run.add_argument("--auto-continue", action="store_true")
    run.set_defaults(func=cache_run_command)

    watch = cache_sub.add_parser("watch", help="watch fresh cache progress")
    watch.add_argument("--job", required=True)
    watch.add_argument("--interval", type=int, default=60)
    watch.add_argument("--once", action="store_true")
    watch.set_defaults(func=cache_watch_command)

    sync = cache_sub.add_parser("sync", help="reconcile runtime progress into catalog")
    sync.add_argument("--job", required=True)
    sync_mode = sync.add_mutually_exclusive_group(required=True)
    sync_mode.add_argument("--dry-run", action="store_true")
    sync_mode.add_argument("--apply", action="store_true")
    sync.set_defaults(func=cache_sync_command)

    ensemble = subparsers.add_parser("ensemble", help="run frozen ensemble methods")
    ensemble_sub = ensemble.add_subparsers(dest="ensemble_action", required=True)
    compare = ensemble_sub.add_parser(
        "compare", help="compare top-4 uniform and Caruana replacement diagnostics"
    )
    compare.add_argument("--roster", required=True)
    compare.add_argument("--job", required=True)
    compare.add_argument("--run-id", required=True)
    compare_mode = compare.add_mutually_exclusive_group(required=True)
    compare_mode.add_argument("--dry-run", action="store_true")
    compare_mode.add_argument("--apply", action="store_true")
    compare.set_defaults(func=ensemble_compare_command)

    uniform = ensemble_sub.add_parser(
        "uniform", help="evaluate a frozen top-k uniform probability ensemble"
    )
    uniform.add_argument("--roster", required=True)
    uniform.add_argument("--job", required=True)
    uniform.add_argument("--top", type=int, required=True)
    uniform.add_argument("--run-id", required=True)
    uniform_mode = uniform.add_mutually_exclusive_group(required=True)
    uniform_mode.add_argument("--dry-run", action="store_true")
    uniform_mode.add_argument("--apply", action="store_true")
    uniform.set_defaults(func=ensemble_uniform_command)

    seeded_plan = ensemble_sub.add_parser(
        "seeded-plan", help="prepare the wave-gated top-16 seeded scoped diagnostic"
    )
    seeded_plan.add_argument("--roster", required=True)
    seeded_plan.add_argument("--source-job", action="append", required=True)
    seeded_plan.add_argument("--top", type=int, required=True)
    seeded_plan.add_argument("--workers", type=int, required=True)
    seeded_plan.add_argument("--job", required=True)
    seeded_plan_mode = seeded_plan.add_mutually_exclusive_group(required=True)
    seeded_plan_mode.add_argument("--dry-run", action="store_true")
    seeded_plan_mode.add_argument("--apply", action="store_true")
    seeded_plan.set_defaults(func=ensemble_seeded_plan_command)

    seeded_run = ensemble_sub.add_parser(
        "seeded-run", help="wait for ranks 1-16 and run the prepared diagnostic"
    )
    seeded_run.add_argument("--job", required=True)
    seeded_run.add_argument("--poll-interval", type=int, default=600)
    seeded_run.add_argument("--auto-start", action="store_true")
    seeded_run.set_defaults(func=ensemble_seeded_run_command)

    seeded_watch = ensemble_sub.add_parser(
        "seeded-watch", help="watch a prepared seeded diagnostic"
    )
    seeded_watch.add_argument("--job", required=True)
    seeded_watch.add_argument("--interval", type=int, default=600)
    seeded_watch.add_argument("--once", action="store_true")
    seeded_watch.set_defaults(func=ensemble_seeded_watch_command)

    check = subparsers.add_parser("check", help="validate catalog, renders, and migration")
    check.add_argument("--deep", action="store_true", help="rehash all source files")
    check.set_defaults(func=check_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "cache" and args.cache_action == "migrate-sideexp002":
            result = migrate_legacy_cache(
                Path(args.source), Path(args.destination), apply=args.apply
            )
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        return int(args.func(args))
    except CatalogConflict as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            f"OVERRIDE STUB (merge into {OVERRIDES_PATH}, fill review fields, then rerun):",
            file=sys.stderr,
        )
        print(json.dumps(exc.stub, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    except (HubError, seeded_caruana.SeededCaruanaError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
