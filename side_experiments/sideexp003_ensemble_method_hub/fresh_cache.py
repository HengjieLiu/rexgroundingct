#!/usr/bin/env python3
"""Pure helpers for SideExp003 fresh logit rosters, jobs, and catalog state."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNTIME_ROOT = Path(
    "/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/"
    "sideexp003_ensemble_method_hub"
)
VAL200_DATASET = REPO_ROOT / "configs/evaluation/rexgroundingct_val200_seed20260723.json"
VAL200_SHA256 = "7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
TEST300_DATASET = Path("/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json")
TEST300_SHA256 = "3a66087608d5d0177a30c0238845a7e53518fcd82f8f41283a7d14fc02b37e6a"
CATALOG_SCHEMA_VERSION = 2
ROSTER_SCHEMA_VERSION = 1
JOB_SCHEMA_VERSION = 1
CACHE_CONTRACT_VERSION = 2
EXPECTED_CONTAINER_IMAGE = "rexgroundingct-voxtell:cu126"
EXPECTED_CONTAINER_IMAGE_ID = (
    "sha256:8ff421d05fbf6044553ba987d065a4d6fdaaaab8e2913272e620d08c4c286e0f"
)
VALID_FRESH_STATUSES = {
    "queued",
    "waiting_for_gpu",
    "smoke_running",
    "smoke_waiting",
    "exporting",
    "finalizing",
    "validating",
    "strict_passed",
    "failed",
    "blocked",
}

SOURCE_FILES = (
    HERE / "hub.py",
    HERE / "fresh_cache.py",
    HERE / "fresh_export.py",
    HERE / "fresh_orchestrator.py",
    REPO_ROOT / "side_experiments/sideexp002_multimodel_ensemble_selection/export_logits.py",
    REPO_ROOT / "side_experiments/sideexp002_multimodel_ensemble_selection/analyze_candidates.py",
    REPO_ROOT / "side_experiments/sideexp002_multimodel_ensemble_selection/storage_validation.py",
    REPO_ROOT / "scripts/rexgroundingct/common.py",
    REPO_ROOT / "scripts/rexgroundingct/run_voxtell_val_inference.py",
    REPO_ROOT / "scripts/rexgroundingct/voxtell_dual_branch.py",
    REPO_ROOT / "scripts/rexgroundingct/voxtell_preprocessed_cache.py",
    REPO_ROOT / "scripts/rexgroundingct/voxtell_s3_attention.py",
)


class FreshCacheError(RuntimeError):
    """Raised when a fresh-cache contract is invalid."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FreshCacheError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FreshCacheError(f"invalid JSON file {path}: {exc}") from exc


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def catalog_by_dice(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        catalog["candidates"],
        key=lambda row: (
            -float(row["metrics"]["overall"]["dice"]),
            -int(row["metrics"]["overall"]["hits"]),
            str(row["candidate_id"]),
        ),
    )


def _legacy_version(candidate: dict[str, Any]) -> dict[str, Any] | None:
    legacy = candidate.get("logit_cache")
    if not isinstance(legacy, dict) or legacy.get("status") == "missing":
        return None
    legacy_status = str(legacy.get("status"))
    status = "strict_passed" if legacy_status == "strict_gate_passed" else "analysis_usable"
    physical = legacy.get("physical_path")
    logical = legacy.get("logical_path")
    return {
        "cache_version_id": f"legacy_sideexp002_{legacy.get('candidate_id')}",
        "origin": "legacy_sideexp002",
        "status": status,
        "job_id": None,
        "cache_key": None,
        "dtype": None,
        "cases": 200,
        "findings": 381,
        "physical_path": physical,
        "logical_path": logical,
        "export_manifest_path": str(Path(physical) / "export_manifest.json") if physical else None,
        "progress_path": None,
        "validation_path": str(Path(physical) / "reproduction_validation.json") if physical else None,
        "metrics_path": str(Path(physical) / "reproduction_validation.json") if physical else None,
        "gpu": None,
        "wave": None,
        "started_at_utc": None,
        "completed_at_utc": None,
        "elapsed_seconds": None,
        "strict_eligible": status == "strict_passed",
        "warnings": [] if status == "strict_passed" else ["legacy_same_pass_sign_proof_unavailable"],
        "error": None,
    }


def ensure_candidate_artifacts(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a schema-v2 candidate while retaining legacy cache provenance."""

    result = copy.deepcopy(candidate)
    config_path = result.get("config", {}).get("path")
    if isinstance(config_path, str) and Path(config_path).is_file():
        config = read_json(Path(config_path))
        raw_pre = config.get("preprocessing") if isinstance(config, dict) else None
        if isinstance(raw_pre, dict):
            cache_root = raw_pre.get("cache_root")
            manifest_path = raw_pre.get("manifest_path")
            if manifest_path is None and isinstance(cache_root, str) and cache_root:
                manifest_path = str(Path(cache_root) / "manifest.json")
            preprocessing = result.setdefault("preprocessing", {})
            preprocessing.update(
                {
                    "name": str(raw_pre.get("cache_id", preprocessing.get("name", "unknown"))),
                    "description": str(
                        raw_pre.get(
                            "normalization",
                            raw_pre.get("classification", preprocessing.get("description", "unknown")),
                        )
                    ),
                    "cache_root": cache_root,
                    "manifest_path": manifest_path,
                    "manifest_sha256": raw_pre.get("manifest_sha256"),
                    "provenance": "experiment config preprocessing",
                }
            )
    artifacts = result.get("inference_artifacts")
    if not isinstance(artifacts, dict):
        legacy = _legacy_version(result)
        versions = [legacy] if legacy else []
        artifacts = {
            "val200": {
                "status": legacy["status"] if legacy else "not_requested",
                "active_cache_version": (
                    legacy["cache_version_id"]
                    if legacy and legacy["status"] == "strict_passed"
                    else None
                ),
                "versions": versions,
            },
            "test300": {
                "status": "deferred",
                "dataset": {
                    "path": str(TEST300_DATASET),
                    "sha256": TEST300_SHA256,
                    "cases": 300,
                    "findings": 582,
                },
                "metric_status": "not_available_withheld_labels",
                "active_cache_version": None,
                "versions": [],
                "cache_path": None,
                "result_path": None,
            },
        }
    result["inference_artifacts"] = artifacts
    result.setdefault("roster_memberships", [])
    result.pop("logit_cache", None)
    return result


def migrate_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    if catalog.get("schema_version") not in {1, CATALOG_SCHEMA_VERSION}:
        raise FreshCacheError(
            f"unsupported catalog schema_version {catalog.get('schema_version')!r}"
        )
    result = copy.deepcopy(catalog)
    result["schema_version"] = CATALOG_SCHEMA_VERSION
    result["candidates"] = [
        ensure_candidate_artifacts(candidate) for candidate in result.get("candidates", [])
    ]
    result.setdefault("artifact_state_updated_at_utc", None)
    return result


def merge_candidate_state(
    scanned: dict[str, Any], existing: dict[str, Any] | None
) -> dict[str, Any]:
    result = ensure_candidate_artifacts(scanned)
    if existing is None:
        return result
    preserved = ensure_candidate_artifacts(existing)
    result["inference_artifacts"] = copy.deepcopy(preserved["inference_artifacts"])
    result["roster_memberships"] = copy.deepcopy(preserved["roster_memberships"])
    return result


def validate_test300_contract(path: Path = TEST300_DATASET) -> dict[str, Any]:
    data = read_json(path)
    cases = data.get("test") if isinstance(data, dict) else None
    if not isinstance(cases, list) or len(cases) != 300:
        raise FreshCacheError(f"{path}: expected 300 test cases")
    names: set[str] = set()
    findings = 0
    for case in cases:
        name = case.get("name") if isinstance(case, dict) else None
        prompts = case.get("findings") if isinstance(case, dict) else None
        if not isinstance(name, str) or name in names or not isinstance(prompts, dict):
            raise FreshCacheError(f"{path}: invalid or duplicate test case {name!r}")
        names.add(name)
        findings += len(prompts)
    if findings != 582:
        raise FreshCacheError(f"{path}: expected 582 findings, got {findings}")
    observed = sha256_file(path)
    if observed != TEST300_SHA256:
        raise FreshCacheError(
            f"test300 SHA drift: expected {TEST300_SHA256}, got {observed}"
        )
    return {"path": str(path), "sha256": observed, "cases": 300, "findings": 582}


def _relative_source_label(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def source_bundle() -> dict[str, Any]:
    files: list[Path] = list(SOURCE_FILES)
    files.extend(sorted((REPO_ROOT / "external/VoxTell/voxtell").rglob("*.py")))
    unique = sorted(set(files), key=str)
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FreshCacheError(f"source bundle files missing: {missing}")
    entries = [
        {"path": _relative_source_label(path), "sha256": sha256_file(path)}
        for path in unique
    ]
    return {"sha256": json_sha256(entries), "files": entries}


def _resolve_preprocessing(candidate: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(candidate["config"]["path"])
    config = read_json(config_path)
    raw = config.get("preprocessing") if isinstance(config, dict) else None
    if not isinstance(raw, dict):
        raise FreshCacheError(f"{candidate['candidate_id']}: config has no preprocessing")
    cache_root = raw.get("cache_root")
    cache_id = raw.get("cache_id")
    manifest_sha = raw.get("manifest_sha256")
    if not all(isinstance(value, str) and value for value in (cache_root, cache_id, manifest_sha)):
        raise FreshCacheError(
            f"{candidate['candidate_id']}: incomplete preprocessing cache contract"
        )
    manifest_path = Path(cache_root) / "manifest.json"
    if not manifest_path.is_file():
        raise FreshCacheError(
            f"{candidate['candidate_id']}: missing preprocessing manifest {manifest_path}"
        )
    observed = sha256_file(manifest_path)
    if observed != manifest_sha:
        raise FreshCacheError(
            f"{candidate['candidate_id']}: preprocessing manifest SHA mismatch"
        )
    return {
        "id": cache_id,
        "root": cache_root,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "normalization": raw.get("normalization", raw.get("classification", "unspecified")),
    }


def _candidate_contract(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    checkpoint_path = Path(candidate["checkpoint"]["path"])
    model_dir = checkpoint_path.parent.parent
    plans_path = model_dir / "plans.json"
    model_spec_path = model_dir / "model_spec.json"
    if not plans_path.is_file():
        raise FreshCacheError(f"{candidate['candidate_id']}: missing {plans_path}")
    model_spec = read_json(model_spec_path) if model_spec_path.is_file() else None
    model_type = (
        str(model_spec.get("model_type"))
        if isinstance(model_spec, dict)
        else "voxtell_standard"
    )
    architecture_kind = (
        "dual" if model_type == "voxtell_dual_branch_v1"
        else "s3" if model_type == "voxtell_s3_attention_v1"
        else "standard"
    )
    preprocessing = _resolve_preprocessing(candidate)
    inference_contract = {
        "schema_version": 1,
        "model_type": model_type,
        "plans_sha256": sha256_file(plans_path),
        "model_spec_sha256": sha256_file(model_spec_path) if model_spec_path.is_file() else None,
        "prompt_order": "sorted_prompts_numeric_finding_index",
        "prediction": "VoxTell sliding-window preprocessed crop final logits",
        "restore": "cached crop to released-label native evaluator layout",
        "output_layout": "F,X,Y,Z",
        "probability_threshold": 0.5,
        "logit_threshold": 0.0,
        "inference_batch_size": 1,
    }
    return {
        "id": candidate["candidate_id"],
        "rank": rank,
        "experiment_number": int(candidate["experiment"]["number"]),
        "experiment_name": candidate["experiment"]["id"],
        "run": candidate["run"],
        "model_id": candidate["model"],
        "epoch": int(candidate["epoch"]),
        "architecture": candidate["architecture"].get("name", architecture_kind),
        "architecture_kind": architecture_kind,
        "model_dir": str(model_dir),
        "plans": {"path": str(plans_path), "sha256": inference_contract["plans_sha256"]},
        "model_spec": {
            "path": str(model_spec_path) if model_spec_path.is_file() else None,
            "sha256": inference_contract["model_spec_sha256"],
            "model_type": model_type,
        },
        "checkpoint": copy.deepcopy(candidate["checkpoint"]),
        "config": copy.deepcopy(candidate["config"]),
        "cache": preprocessing,
        "evaluation": copy.deepcopy(candidate["canonical_evaluator"]),
        "performance": {
            "dice": float(candidate["metrics"]["overall"]["dice"]),
            "findings": int(candidate["metrics"]["overall"]["findings"]),
            "hits": int(candidate["metrics"]["overall"]["hits"]),
            "hit_rate": float(candidate["metrics"]["overall"]["hit_rate"]),
        },
        "inference_contract": inference_contract,
        "inference_contract_sha256": json_sha256(inference_contract),
    }


def build_roster(
    catalog: dict[str, Any],
    *,
    catalog_sha256: str,
    roster_id: str,
    top: int,
    container_image: str,
    container_image_id: str,
) -> dict[str, Any]:
    if top < 1:
        raise FreshCacheError("--top must be positive")
    if container_image_id != EXPECTED_CONTAINER_IMAGE_ID:
        raise FreshCacheError(
            f"container image ID drift: expected {EXPECTED_CONTAINER_IMAGE_ID}, "
            f"got {container_image_id}"
        )
    migrated = migrate_catalog(catalog)
    ranked = [row for row in catalog_by_dice(migrated) if row.get("eligibility") == "eligible"]
    if len(ranked) < top:
        raise FreshCacheError(f"only {len(ranked)} eligible candidates are available")
    checkpoint_hashes = [row["checkpoint"]["sha256"] for row in ranked[:top]]
    if None in checkpoint_hashes or len(set(checkpoint_hashes)) != top:
        raise FreshCacheError("top roster checkpoint SHA values are missing or duplicated")
    bundle = source_bundle()
    candidates = [
        _candidate_contract(candidate, rank)
        for rank, candidate in enumerate(ranked[:top], start=1)
    ]
    roster = {
        "schema_version": ROSTER_SCHEMA_VERSION,
        "roster_id": roster_id,
        "created_at_utc": utc_now(),
        "selection": {
            "source": "checkpoint_catalog.json",
            "metric": "overall_dice",
            "order": ["dice_desc", "hits_desc", "candidate_id_asc"],
            "top": top,
            "eligibility": "eligible_only",
        },
        "catalog": {"sha256": catalog_sha256, "schema_version_at_freeze": catalog.get("schema_version")},
        "dataset": {
            "split": "val200",
            "path": str(VAL200_DATASET),
            "sha256": VAL200_SHA256,
            "cases": 200,
            "findings": 381,
        },
        "test300": {
            "status": "deferred",
            "dataset": validate_test300_contract(),
            "metric_status": "not_available_withheld_labels",
        },
        "container": {"image": container_image, "image_id": container_image_id},
        "source_bundle": bundle,
        "reuse_policy": "fresh_only",
        "N": top,
        "candidates": candidates,
    }
    roster["roster_sha256"] = json_sha256(roster)
    return roster


def validate_roster(roster: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if roster.get("schema_version") != ROSTER_SCHEMA_VERSION:
        errors.append("roster schema_version mismatch")
    candidates = roster.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != roster.get("N"):
        return errors + ["roster candidates/N mismatch"]
    if roster.get("reuse_policy") != "fresh_only":
        errors.append("roster is not fresh_only")
    if [candidate.get("rank") for candidate in candidates] != list(range(1, len(candidates) + 1)):
        errors.append("roster ranks are not contiguous")
    ids = [candidate.get("id") for candidate in candidates]
    hashes = [candidate.get("checkpoint", {}).get("sha256") for candidate in candidates]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)) or None in hashes:
        errors.append("roster candidates/checkpoints are not unique")
    expected_sha = roster.get("roster_sha256")
    unsigned = copy.deepcopy(roster)
    unsigned.pop("roster_sha256", None)
    if expected_sha != json_sha256(unsigned):
        errors.append("roster content SHA mismatch")
    if roster.get("dataset", {}).get("sha256") != VAL200_SHA256:
        errors.append("roster val200 dataset SHA mismatch")
    if roster.get("container", {}).get("image_id") != EXPECTED_CONTAINER_IMAGE_ID:
        errors.append("roster container image ID mismatch")
    return errors


def make_cache_key(candidate: dict[str, Any], roster: dict[str, Any]) -> str:
    contract = {
        "schema_version": CACHE_CONTRACT_VERSION,
        "dataset_sha256": roster["dataset"]["sha256"],
        "checkpoint_sha256": candidate["checkpoint"]["sha256"],
        "config_sha256": candidate["config"]["sha256"],
        "inference_contract_sha256": candidate["inference_contract_sha256"],
        "preprocessing_manifest_sha256": candidate["cache"]["manifest_sha256"],
        "source_bundle_sha256": roster["source_bundle"]["sha256"],
        "container_image_id": roster["container"]["image_id"],
        "storage_contract": {
            "layout": "F,X,Y,Z",
            "kind": "pre_sigmoid_logits",
            "clip": [-30.0, 30.0],
            "staging_dtype": "float32",
            "publication_policy": "float16_if_exact_threshold_sign_else_float32",
            "threshold_logit": 0.0,
        },
    }
    return f"v{CACHE_CONTRACT_VERSION}_{json_sha256(contract)}"


def build_job(
    roster: dict[str, Any], *, job_id: str, wave_size: int = 4, gpus: Iterable[int] = range(4)
) -> dict[str, Any]:
    errors = validate_roster(roster)
    if errors:
        raise FreshCacheError("invalid roster: " + "; ".join(errors))
    gpu_list = list(gpus)
    if wave_size < 1 or len(gpu_list) != wave_size or len(set(gpu_list)) != wave_size:
        raise FreshCacheError("wave size must equal the unique GPU count")
    candidates = copy.deepcopy(roster["candidates"])
    for index, candidate in enumerate(candidates):
        wave = index // wave_size + 1
        gpu = gpu_list[index % wave_size]
        key = make_cache_key(candidate, roster)
        candidate.update(
            {
                "wave": wave,
                "gpu": gpu,
                "cache_key": key,
                "cache_version_id": f"sideexp003_{key}",
                "cache_root": str(RUNTIME_ROOT / "cache/logits/by_cache_key" / key),
                "progress_path": str(RUNTIME_ROOT / "cache/jobs" / job_id / "progress" / f"{candidate['id']}.json"),
            }
        )
    waves = []
    for wave_number in range(1, math.ceil(len(candidates) / wave_size) + 1):
        members = [candidate for candidate in candidates if candidate["wave"] == wave_number]
        waves.append(
            {
                "wave": wave_number,
                "status": "queued",
                "members": [
                    {"rank": candidate["rank"], "candidate_id": candidate["id"], "gpu": candidate["gpu"]}
                    for candidate in members
                ],
            }
        )
    job = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "created_at_utc": utc_now(),
        "status": "planned",
        "roster_id": roster["roster_id"],
        "roster_sha256": roster["roster_sha256"],
        "reuse_policy": "fresh_only",
        "auto_continue": True,
        "wave_size": wave_size,
        "gpus": gpu_list,
        "gpu_safety": {
            "minimum_start_free_mib": 20_000,
            "minimum_post_smoke_free_mib": 4_096,
            "poll_seconds": 60,
        },
        "runtime_root": str(RUNTIME_ROOT / "cache/jobs" / job_id),
        "container": copy.deepcopy(roster["container"]),
        "dataset": copy.deepcopy(roster["dataset"]),
        "test300": copy.deepcopy(roster["test300"]),
        "source_bundle": copy.deepcopy(roster["source_bundle"]),
        "storage_contract": {
            "layout": "F,X,Y,Z",
            "kind": "pre_sigmoid_logits",
            "clip": [-30.0, 30.0],
            "staging_dtype": "float32",
            "publication_policy": "float16_if_exact_threshold_sign_else_float32",
            "same_pass_probability_threshold": 0.5,
        },
        "initial_eta_hours": {"wave_1": [2.5, 5.5], "all_20": [15.0, 28.0]},
        "waves": waves,
        "candidates": candidates,
    }
    job["job_spec_sha256"] = json_sha256(job)
    return job


def build_continuation_job(
    parent_job: dict[str, Any],
    *,
    job_id: str,
    start_rank: int,
    current_source_bundle: dict[str, Any],
    source_drift_audit: dict[str, Any],
) -> dict[str, Any]:
    """Create a new-key continuation without mutating the frozen parent job."""

    parent_errors = validate_job(parent_job)
    if parent_errors:
        raise FreshCacheError("invalid parent job: " + "; ".join(parent_errors))
    parent_candidates = parent_job["candidates"]
    if start_rank <= 1 or start_rank > len(parent_candidates):
        raise FreshCacheError("continuation start rank must be inside the parent roster")
    expected_parent_ranks = list(range(1, len(parent_candidates) + 1))
    if [candidate["rank"] for candidate in parent_candidates] != expected_parent_ranks:
        raise FreshCacheError("parent job must contain the complete frozen rank order")
    if current_source_bundle.get("sha256") == parent_job["source_bundle"].get("sha256"):
        raise FreshCacheError("continuation requires a changed source bundle")
    if source_drift_audit.get("new_source_bundle_sha256") != current_source_bundle.get("sha256"):
        raise FreshCacheError("source-drift audit does not match the current source bundle")
    if source_drift_audit.get("parent_source_bundle_sha256") != parent_job["source_bundle"].get("sha256"):
        raise FreshCacheError("source-drift audit does not match the parent source bundle")

    wave_size = int(parent_job["wave_size"])
    gpus = list(parent_job["gpus"])
    wave_start = (start_rank - 1) // wave_size + 1
    if start_rank != (wave_start - 1) * wave_size + 1:
        raise FreshCacheError("continuation must start at a wave boundary")
    candidates = copy.deepcopy(parent_candidates[start_rank - 1 :])
    cache_context = {
        "dataset": copy.deepcopy(parent_job["dataset"]),
        "source_bundle": copy.deepcopy(current_source_bundle),
        "container": copy.deepcopy(parent_job["container"]),
    }
    for index, candidate in enumerate(candidates):
        expected_rank = start_rank + index
        if candidate["rank"] != expected_rank:
            raise FreshCacheError("continuation candidates are not rank-contiguous")
        wave = wave_start + index // wave_size
        gpu = gpus[index % wave_size]
        key = make_cache_key(candidate, cache_context)
        candidate.update(
            {
                "wave": wave,
                "gpu": gpu,
                "cache_key": key,
                "cache_version_id": f"sideexp003_{key}",
                "cache_root": str(RUNTIME_ROOT / "cache/logits/by_cache_key" / key),
                "progress_path": str(
                    RUNTIME_ROOT
                    / "cache/jobs"
                    / job_id
                    / "progress"
                    / f"{candidate['id']}.json"
                ),
                "source_bundle_sha256": current_source_bundle["sha256"],
                "parent_cache_key": parent_candidates[expected_rank - 1]["cache_key"],
            }
        )
    waves = []
    max_wave = max(candidate["wave"] for candidate in candidates)
    for wave_number in range(wave_start, max_wave + 1):
        members = [candidate for candidate in candidates if candidate["wave"] == wave_number]
        waves.append(
            {
                "wave": wave_number,
                "status": "queued",
                "members": [
                    {
                        "rank": candidate["rank"],
                        "candidate_id": candidate["id"],
                        "gpu": candidate["gpu"],
                    }
                    for candidate in members
                ],
            }
        )
    audit_copy = copy.deepcopy(source_drift_audit)
    job = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "created_at_utc": utc_now(),
        "status": "planned",
        "roster_id": parent_job["roster_id"],
        "roster_sha256": parent_job["roster_sha256"],
        "reuse_policy": "fresh_only",
        "auto_continue": True,
        "wave_size": wave_size,
        "gpus": gpus,
        "gpu_safety": copy.deepcopy(parent_job["gpu_safety"]),
        "runtime_root": str(RUNTIME_ROOT / "cache/jobs" / job_id),
        "container": copy.deepcopy(parent_job["container"]),
        "dataset": copy.deepcopy(parent_job["dataset"]),
        "test300": copy.deepcopy(parent_job["test300"]),
        "source_bundle": copy.deepcopy(current_source_bundle),
        "storage_contract": copy.deepcopy(parent_job["storage_contract"]),
        "initial_eta_hours": {"remaining": [7.0, 12.0]},
        "continuation": {
            "parent_job_id": parent_job["job_id"],
            "parent_job_spec_sha256": parent_job["job_spec_sha256"],
            "parent_source_bundle_sha256": parent_job["source_bundle"]["sha256"],
            "rank_start": start_rank,
            "wave_start": wave_start,
            "completed_parent_ranks": list(range(1, start_rank)),
            "original_top_n": len(parent_candidates),
            "cache_policy": "new_source_bundle_new_cache_keys_no_parent_overwrite",
            "source_drift_audit": audit_copy,
        },
        "waves": waves,
        "candidates": candidates,
    }
    job["job_spec_sha256"] = json_sha256(job)
    return job


def validate_job(job: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if job.get("schema_version") != JOB_SCHEMA_VERSION:
        errors.append("job schema_version mismatch")
    if job.get("reuse_policy") != "fresh_only":
        errors.append("job does not require fresh exports")
    candidates = job.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return errors + ["job has no candidates"]
    expected = copy.deepcopy(job)
    signature = expected.pop("job_spec_sha256", None)
    if signature != json_sha256(expected):
        errors.append("job spec SHA mismatch")
    continuation = job.get("continuation")
    rank_start = 1
    wave_start = 1
    if continuation is not None:
        if not isinstance(continuation, dict):
            errors.append("job continuation must be an object")
        else:
            rank_start = int(continuation.get("rank_start", 0))
            wave_start = int(continuation.get("wave_start", 0))
            if rank_start <= 1 or wave_start <= 1:
                errors.append("continuation rank/wave start is invalid")
            if continuation.get("parent_source_bundle_sha256") == job.get("source_bundle", {}).get("sha256"):
                errors.append("continuation did not change source bundle")
            audit = continuation.get("source_drift_audit")
            if not isinstance(audit, dict) or audit.get("new_source_bundle_sha256") != job.get("source_bundle", {}).get("sha256"):
                errors.append("continuation source-drift audit mismatch")
    ranks = [candidate.get("rank") for candidate in candidates]
    if ranks != list(range(rank_start, rank_start + len(candidates))):
        errors.append("job ranks are not contiguous")
    for index, candidate in enumerate(candidates):
        expected_wave = index // int(job["wave_size"]) + wave_start
        expected_gpu = job["gpus"][index % int(job["wave_size"])]
        if candidate.get("wave") != expected_wave or candidate.get("gpu") != expected_gpu:
            errors.append(f"{candidate.get('id')}: wave/GPU assignment mismatch")
        if not str(candidate.get("cache_root", "")).endswith(str(candidate.get("cache_key", ""))):
            errors.append(f"{candidate.get('id')}: cache path/key mismatch")
    return errors


def fresh_version_from_job(candidate: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    cache_root = Path(candidate["cache_root"])
    return {
        "cache_version_id": candidate["cache_version_id"],
        "origin": "sideexp003",
        "status": "queued",
        "job_id": job["job_id"],
        "cache_key": candidate["cache_key"],
        "dtype": None,
        "cases": 0,
        "findings": 0,
        "physical_path": str(cache_root),
        "logical_path": str(cache_root),
        "export_manifest_path": str(cache_root / "export_manifest.json"),
        "progress_path": candidate["progress_path"],
        "validation_path": str(cache_root / "reproduction_validation.json"),
        "metrics_path": str(cache_root / "reproduction_validation.json"),
        "gpu": candidate["gpu"],
        "wave": candidate["wave"],
        "started_at_utc": None,
        "completed_at_utc": None,
        "elapsed_seconds": None,
        "strict_eligible": False,
        "warnings": [],
        "error": None,
    }


def apply_job_to_catalog(catalog: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    errors = validate_job(job)
    if errors:
        raise FreshCacheError("invalid job: " + "; ".join(errors))
    result = migrate_catalog(catalog)
    by_id = {candidate["candidate_id"]: candidate for candidate in result["candidates"]}
    for job_candidate in job["candidates"]:
        candidate = by_id.get(job_candidate["id"])
        if candidate is None:
            raise FreshCacheError(f"job candidate absent from catalog: {job_candidate['id']}")
        split = candidate["inference_artifacts"]["val200"]
        existing = [
            version for version in split["versions"]
            if version.get("cache_version_id") == job_candidate["cache_version_id"]
        ]
        if existing:
            if existing[0].get("job_id") != job["job_id"]:
                raise FreshCacheError(f"cache version collision: {job_candidate['cache_version_id']}")
        else:
            split["versions"].append(fresh_version_from_job(job_candidate, job))
        split["status"] = "queued"
        membership = {
            "roster_id": job["roster_id"],
            "job_id": job["job_id"],
            "rank": job_candidate["rank"],
            "wave": job_candidate["wave"],
            "gpu": job_candidate["gpu"],
            "reuse_policy": "fresh_only",
        }
        memberships = [
            value for value in candidate["roster_memberships"]
            if value.get("job_id") != job["job_id"]
        ]
        candidate["roster_memberships"] = memberships + [membership]
    result["artifact_state_updated_at_utc"] = utc_now()
    return result


def update_catalog_progress(
    catalog: dict[str, Any], job: dict[str, Any], progress_records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result = migrate_catalog(catalog)
    by_id = {candidate["candidate_id"]: candidate for candidate in result["candidates"]}
    for job_candidate in job["candidates"]:
        progress = progress_records.get(job_candidate["id"])
        if not isinstance(progress, dict):
            continue
        status = progress.get("status")
        if status not in VALID_FRESH_STATUSES:
            continue
        candidate = by_id[job_candidate["id"]]
        split = candidate["inference_artifacts"]["val200"]
        matches = [
            version for version in split["versions"]
            if version.get("cache_version_id") == job_candidate["cache_version_id"]
        ]
        if len(matches) != 1:
            raise FreshCacheError(
                f"{job_candidate['id']}: fresh catalog version missing or duplicated"
            )
        version = matches[0]
        for field in (
            "status", "dtype", "cases", "findings", "started_at_utc",
            "completed_at_utc", "elapsed_seconds", "warnings", "error",
        ):
            if field in progress:
                version[field] = copy.deepcopy(progress[field])
        version["strict_eligible"] = status == "strict_passed"
        split["status"] = status
        if status == "strict_passed":
            split["active_cache_version"] = version["cache_version_id"]
    result["artifact_state_updated_at_utc"] = utc_now()
    return result


def job_membership(candidate: dict[str, Any], job_id: str | None = None) -> dict[str, Any] | None:
    values = candidate.get("roster_memberships", [])
    if job_id is not None:
        values = [value for value in values if value.get("job_id") == job_id]
    return values[-1] if values else None


def job_version(candidate: dict[str, Any], job_id: str | None = None) -> dict[str, Any] | None:
    versions = candidate.get("inference_artifacts", {}).get("val200", {}).get("versions", [])
    matches = [version for version in versions if version.get("origin") == "sideexp003"]
    if job_id is not None:
        matches = [version for version in matches if version.get("job_id") == job_id]
    return matches[-1] if matches else None


def legacy_versions(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        version
        for version in candidate.get("inference_artifacts", {}).get("val200", {}).get("versions", [])
        if version.get("origin") == "legacy_sideexp002"
    ]


def validate_real_sources(roster: dict[str, Any], verify_checkpoints: bool = True) -> list[str]:
    errors = validate_roster(roster)
    for candidate in roster.get("candidates", []):
        for label, record in (
            ("checkpoint", candidate["checkpoint"]),
            ("config", candidate["config"]),
            ("evaluation", candidate["evaluation"]),
            ("plans", candidate["plans"]),
        ):
            path = Path(record["path"])
            if not path.is_file():
                errors.append(f"{candidate['id']}: missing {label} {path}")
                continue
            if label != "checkpoint" or verify_checkpoints:
                if sha256_file(path) != record["sha256"]:
                    errors.append(f"{candidate['id']}: {label} SHA mismatch")
        model_spec = candidate["model_spec"]
        if model_spec["path"] and sha256_file(Path(model_spec["path"])) != model_spec["sha256"]:
            errors.append(f"{candidate['id']}: model_spec SHA mismatch")
        cache_manifest = Path(candidate["cache"]["manifest_path"])
        if not cache_manifest.is_file() or sha256_file(cache_manifest) != candidate["cache"]["manifest_sha256"]:
            errors.append(f"{candidate['id']}: preprocessing manifest mismatch")
        if candidate["performance"]["findings"] != 381:
            errors.append(f"{candidate['id']}: evaluator is not 381 findings")
    return errors
