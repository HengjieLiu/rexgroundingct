#!/usr/bin/env python3
"""Wave-gated top-16 uniform and seeded scoped Caruana diagnostics."""

from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import preliminary_ensemble as pe


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNTIME_ROOT = pe.RUNTIME_ROOT
SCOPES = ("all", "2a", "2b", "2c", "2d")
SEED_RANKS = {
    "all": (1, 2, 3, 4),
    "2a": (4, 3, 15, 2),
    "2b": (7, 4, 11, 1),
    "2c": (1, 16, 13, 8),
    "2d": (1, 3, 2, 5),
}
EXPECTED_FINDINGS = {"all": 381, "2a": 69, "2b": 49, "2c": 60, "2d": 132}
DEFAULT_UNIFORM_RUN = "r003_top16_fresh_val200"
DEFAULT_CARUANA_RUN = "r001_top16_per_scope_seed4_fullval"
DEFAULT_WORKERS = 5
MINIMUM_AVAILABLE_MEMORY_BYTES = 64 * 1024**3


class SeededCaruanaError(RuntimeError):
    """Raised when the frozen diagnostic contract is violated."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SeededCaruanaError(f"cannot read JSON {path}: {exc}") from exc


def _atomic_json(path: Path, value: Any) -> None:
    pe.atomic_write_json(path, value)


def _identity_sha(value: dict[str, Any], field: str, volatile: Iterable[str] = ()) -> str:
    excluded = {field, *volatile}
    return pe.json_sha256({key: item for key, item in value.items() if key not in excluded})


def _container_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise SeededCaruanaError(f"tracked path is outside repository: {path}") from exc
    return Path("/workspace") / relative


def _available_memory_bytes() -> int:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return values.get("MemAvailable", 0)


def _validate_source_contract(source: dict[str, Any], jobs: list[dict[str, Any]], top: int) -> None:
    if top != 16:
        raise SeededCaruanaError("this approved diagnostic requires top=16")
    if source.get("reuse_policy") != "fresh_only" or int(source.get("N", -1)) < top:
        raise SeededCaruanaError("source roster is not a compatible fresh-only top-20 roster")
    if source.get("roster_sha256") != _identity_sha(source, "roster_sha256"):
        raise SeededCaruanaError("source roster identity SHA mismatch")
    if len(jobs) != 2:
        raise SeededCaruanaError("exactly two source jobs are required")
    for job in jobs:
        if job.get("roster_id") != source.get("roster_id"):
            raise SeededCaruanaError(f"{job.get('job_id')}: roster ID mismatch")
        if job.get("roster_sha256") != source.get("roster_sha256"):
            raise SeededCaruanaError(f"{job.get('job_id')}: roster SHA mismatch")
        dataset = job.get("dataset", {})
        if dataset.get("sha256") != source.get("dataset", {}).get("sha256"):
            raise SeededCaruanaError(f"{job.get('job_id')}: dataset SHA mismatch")
        if int(dataset.get("cases", -1)) != 200 or int(dataset.get("findings", -1)) != 381:
            raise SeededCaruanaError(f"{job.get('job_id')}: dataset coverage mismatch")


def create_launch_spec(
    source_roster_path: Path,
    source_job_paths: list[Path],
    *,
    top: int,
    workers: int,
    job_id: str,
) -> dict[str, Any]:
    source = _read_json(source_roster_path)
    jobs = [_read_json(path) for path in source_job_paths]
    _validate_source_contract(source, jobs, top)
    if workers != 5:
        raise SeededCaruanaError("this run contract requires exactly five worker processes")
    rank_sources = {
        str(rank): jobs[0]["job_id"] if rank <= 8 else jobs[1]["job_id"]
        for rank in range(1, top + 1)
    }
    ids_by_rank = {int(row["rank"]): row["id"] for row in source["candidates"][:top]}
    seeds = {
        scope: [ids_by_rank[rank] for rank in SEED_RANKS[scope]] for scope in SCOPES
    }
    spec = {
        "schema_version": 1,
        "job_id": job_id,
        "created_at_utc": pe.utc_now(),
        "status": "waiting_for_wave4",
        "source_roster": {
            "path": str(source_roster_path.resolve()),
            "sha256": pe.sha256_file(source_roster_path),
            "roster_sha256": source["roster_sha256"],
            "roster_id": source["roster_id"],
        },
        "source_jobs": [
            {
                "path": str(path.resolve()),
                "sha256": pe.sha256_file(path),
                "job_id": job["job_id"],
            }
            for path, job in zip(source_job_paths, jobs)
        ],
        "rank_sources": rank_sources,
        "top": top,
        "workers": workers,
        "poll_interval_seconds": 600,
        "reuse_policy": "fresh_only",
        "dataset": copy.deepcopy(source["dataset"]),
        "scopes": list(SCOPES),
        "seed_ranks": {scope: list(values) for scope, values in SEED_RANKS.items()},
        "seed_candidate_ids": seeds,
        "uniform_run_id": DEFAULT_UNIFORM_RUN,
        "caruana_run_id": DEFAULT_CARUANA_RUN,
        "method": "caruana_replacement_seeded_scoped",
        "analysis_sources": [
            {
                "path": str(Path(__file__).resolve()),
                "sha256": pe.sha256_file(Path(__file__)),
            },
            {
                "path": str((HERE / "preliminary_ensemble.py").resolve()),
                "sha256": pe.sha256_file(HERE / "preliminary_ensemble.py"),
            },
            {
                "path": str((HERE / "methods" / "caruana_replacement_seeded_scoped" / "method_spec.md").resolve()),
                "sha256": pe.sha256_file(HERE / "methods" / "caruana_replacement_seeded_scoped" / "method_spec.md"),
            },
        ],
        "parameters": {
            "averaging_domain": "post_sigmoid_probability",
            "probability_threshold": 0.5,
            "hit_dice_threshold": 0.1,
            "initial_basket_size": 4,
            "maximum_basket_size": 16,
            "with_replacement": True,
            "continue_through_maximum": True,
            "chunk_elements": pe.DEFAULT_CHUNK_ELEMENTS,
            "metrics_only": True,
            "oof": False,
        },
        "tie_break": ["scope_mean_finding_dice_desc", "scope_hits_desc", "candidate_id_asc"],
        "safety": {
            "minimum_free_bytes": pe.MINIMUM_FREE_BYTES,
            "minimum_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
            "wave5_degradation_factor": 2.0,
            "wave5_recovery_factor": 1.5,
            "consecutive_degraded_polls": 2,
        },
        "outputs": {
            "runtime_root": str(RUNTIME_ROOT / "analysis_jobs" / job_id),
            "derived_roster": str(HERE / "rosters" / "roster_top16_val200_fresh_j001_j002.json"),
            "uniform_run": str(HERE / "methods" / "uniform_global" / "runs" / DEFAULT_UNIFORM_RUN),
            "caruana_run": str(HERE / "methods" / "caruana_replacement_seeded_scoped" / "runs" / DEFAULT_CARUANA_RUN),
        },
        "approvals": {
            "roster_confirmed": True,
            "full_val_diagnostic_confirmed": True,
            "per_scope_seeds_confirmed": True,
            "wave5_concurrency_confirmed": True,
            "derived_logits_confirmed": False,
        },
    }
    spec["job_spec_sha256"] = _identity_sha(
        spec, "job_spec_sha256", volatile=("created_at_utc", "status")
    )
    return spec


def validate_launch_spec(spec: dict[str, Any], *, verify_paths: bool = True) -> list[str]:
    errors: list[str] = []
    if spec.get("schema_version") != 1:
        errors.append("unsupported launch schema")
    if spec.get("top") != 16 or spec.get("workers") != 5:
        errors.append("launch top/workers mismatch")
    if tuple(spec.get("scopes", [])) != SCOPES:
        errors.append("scope order mismatch")
    if spec.get("seed_ranks") != {scope: list(value) for scope, value in SEED_RANKS.items()}:
        errors.append("seed ranks mismatch")
    if spec.get("reuse_policy") != "fresh_only":
        errors.append("launch is not fresh_only")
    if spec.get("job_spec_sha256") != _identity_sha(
        spec, "job_spec_sha256", volatile=("created_at_utc", "status")
    ):
        errors.append("launch spec identity SHA mismatch")
    if verify_paths:
        for record in [
            spec.get("source_roster", {}),
            *spec.get("source_jobs", []),
            *spec.get("analysis_sources", []),
        ]:
            path = Path(record.get("path", ""))
            if not path.is_file():
                errors.append(f"missing frozen source: {path}")
            elif pe.sha256_file(path) != record.get("sha256"):
                errors.append(f"frozen source SHA drift: {path}")
    return errors


def _candidate_cache_status(job: dict[str, Any], candidate_id: str, rank: int) -> dict[str, Any]:
    matches = [
        row for row in job.get("candidates", [])
        if row.get("id") == candidate_id and int(row.get("rank", -1)) == rank
    ]
    if len(matches) != 1:
        return {"ready": False, "status": "source_job_candidate_mismatch"}
    candidate = matches[0]
    root = Path(candidate["cache_root"])
    export_path = root / "export_manifest.json"
    validation_path = root / "reproduction_validation.json"
    progress_path = Path(candidate["progress_path"])
    progress = _read_json(progress_path) if progress_path.is_file() else {}
    status = str(progress.get("status", "queued"))
    if not export_path.is_file() or not validation_path.is_file():
        return {
            "ready": False,
            "status": status,
            "cases": int(progress.get("cases", 0)),
            "cache_root": str(root),
        }
    export = _read_json(export_path)
    validation = _read_json(validation_path)
    failures = []
    if export.get("status") != "strict_passed":
        failures.append("export_not_strict_passed")
    if validation.get("status") != "passed":
        failures.append("validation_not_passed")
    if validation.get("storage_reproduction_status") != "passed":
        failures.append("storage_reproduction_not_passed")
    if validation.get("array_hashes_verified") is not True:
        failures.append("array_hashes_not_verified")
    if int(validation.get("same_pass_mask_mismatch_voxels", -1)) != 0:
        failures.append("threshold_sign_mismatch")
    if int(validation.get("cases", -1)) != 200 or int(validation.get("findings", -1)) != 381:
        failures.append("coverage_mismatch")
    case_records = export.get("cases")
    if not isinstance(case_records, list) or len(case_records) != 200:
        failures.append("export_case_manifest_mismatch")
    return {
        "ready": not failures,
        "status": "strict_passed" if not failures else status,
        "cases": int(validation.get("cases", progress.get("cases", 0))),
        "cache_root": str(root),
        "cache_key": candidate["cache_key"],
        "export_manifest_path": str(export_path),
        "export_manifest_sha256": pe.sha256_file(export_path),
        "validation_path": str(validation_path),
        "validation_sha256": pe.sha256_file(validation_path),
        "dtype": validation.get("dtype"),
        "array_bytes": int(validation.get("array_bytes", 0)),
        "same_pass_dice": validation.get("same_pass_mean_global_dice_per_finding"),
        "same_pass_hits": validation.get("same_pass_total_hits"),
        "failures": failures,
    }


def readiness_report(spec: dict[str, Any]) -> dict[str, Any]:
    errors = validate_launch_spec(spec)
    if errors:
        return {"ready": False, "blocked": True, "errors": errors, "candidates": {}}
    source = _read_json(Path(spec["source_roster"]["path"]))
    jobs = {
        record["job_id"]: _read_json(Path(record["path"]))
        for record in spec["source_jobs"]
    }
    candidates: dict[str, Any] = {}
    blocked = False
    for row in source["candidates"][: spec["top"]]:
        rank = int(row["rank"])
        job_id = spec["rank_sources"][str(rank)]
        result = _candidate_cache_status(jobs[job_id], row["id"], rank)
        result.update({"rank": rank, "source_job": job_id})
        candidates[row["id"]] = result
        if result["status"] in {"failed", "blocked"} or (
            result["status"] == "strict_passed" and result.get("failures")
        ):
            blocked = True
    ready_count = sum(bool(value["ready"]) for value in candidates.values())
    return {
        "ready": ready_count == spec["top"] and not blocked,
        "blocked": blocked,
        "ready_count": ready_count,
        "required_count": spec["top"],
        "candidates": candidates,
        "errors": [],
        "checked_at_utc": pe.utc_now(),
    }


def build_merged_roster(spec: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    if not readiness.get("ready"):
        raise SeededCaruanaError("cannot freeze merged roster before ranks 1-16 are strict")
    source = _read_json(Path(spec["source_roster"]["path"]))
    candidates = []
    for row in source["candidates"][: spec["top"]]:
        status = readiness["candidates"][row["id"]]
        candidates.append({
            "rank": int(row["rank"]),
            "id": row["id"],
            "checkpoint": copy.deepcopy(row["checkpoint"]),
            "cache_key": status["cache_key"],
            "cache_root": status["cache_root"],
            "export_manifest_path": status["export_manifest_path"],
            "export_manifest_sha256": status["export_manifest_sha256"],
            "validation_path": status["validation_path"],
            "validation_sha256": status["validation_sha256"],
            "source_job": status["source_job"],
            "dtype": status["dtype"],
            "array_bytes": int(status["array_bytes"]),
            "same_pass_metrics": {
                "dice": status["same_pass_dice"],
                "hits": status["same_pass_hits"],
            },
        })
    roster = {
        "schema_version": 2,
        "roster_kind": "merged_fresh_analysis",
        "roster_id": "top16_val200_fresh_j001_j002",
        "created_at_utc": pe.utc_now(),
        "source_roster": copy.deepcopy(spec["source_roster"]),
        "source_jobs": copy.deepcopy(spec["source_jobs"]),
        "dataset": copy.deepcopy(spec["dataset"]),
        "reuse_policy": "fresh_only",
        "N": 16,
        "candidates": candidates,
    }
    roster["roster_sha256"] = _identity_sha(
        roster, "roster_sha256", volatile=("created_at_utc",)
    )
    return roster


def validate_merged_roster(roster: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if roster.get("schema_version") != 2 or roster.get("roster_kind") != "merged_fresh_analysis":
        errors.append("unexpected merged roster schema/kind")
    if roster.get("roster_id") != "top16_val200_fresh_j001_j002":
        errors.append("unexpected merged roster ID")
    if roster.get("N") != 16 or roster.get("reuse_policy") != "fresh_only":
        errors.append("merged roster N/reuse mismatch")
    if roster.get("roster_sha256") != _identity_sha(
        roster, "roster_sha256", volatile=("created_at_utc",)
    ):
        errors.append("merged roster identity SHA mismatch")
    candidates = roster.get("candidates", [])
    if [row.get("rank") for row in candidates] != list(range(1, 17)):
        errors.append("merged roster ranks are not 1..16")
    if len({row.get("id") for row in candidates}) != 16:
        errors.append("merged roster candidate IDs are not unique")
    if len({row.get("checkpoint", {}).get("sha256") for row in candidates}) != 16:
        errors.append("merged roster checkpoint hashes are not unique")
    for record in [roster.get("source_roster", {}), *roster.get("source_jobs", [])]:
        path = Path(record.get("path", ""))
        if not path.is_file():
            errors.append(f"missing merged-roster source {path}")
        elif pe.sha256_file(path) != record.get("sha256"):
            errors.append(f"merged-roster source SHA drift {path}")
    for candidate in candidates:
        for label in ("export_manifest", "validation"):
            path = Path(candidate.get(f"{label}_path", ""))
            if not path.is_file():
                errors.append(f"{candidate.get('id')}: missing {label}")
            elif pe.sha256_file(path) != candidate.get(f"{label}_sha256"):
                errors.append(f"{candidate.get('id')}: {label} SHA drift")
    return errors


def _write_immutable(path: Path, value: dict[str, Any], *, identity_field: str | None = None) -> None:
    if path.is_file():
        existing = _read_json(path)
        if identity_field:
            if existing.get(identity_field) != value.get(identity_field):
                raise SeededCaruanaError(f"immutable identity differs: {path}")
        elif existing != value:
            raise SeededCaruanaError(f"immutable artifact differs: {path}")
        return
    _atomic_json(path, value)


def _run_spec(
    method: str,
    method_spec: Path,
    roster_path: Path,
    roster: dict[str, Any],
    launch: dict[str, Any],
    run_id: str,
    runtime_dir: Path,
) -> dict[str, Any]:
    candidate_ids = [row["id"] for row in roster["candidates"]]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "diagnostic_pending",
        "method": {
            "id": method,
            "spec_path": str(method_spec),
            "spec_sha256": pe.sha256_file(method_spec),
        },
        "roster": {
            "path": str(roster_path),
            "sha256": pe.sha256_file(roster_path),
            "roster_sha256": roster["roster_sha256"],
            "candidate_ids": candidate_ids,
            "checkpoint_sha256s": [row["checkpoint"]["sha256"] for row in roster["candidates"]],
            "cache_keys": [row["cache_key"] for row in roster["candidates"]],
            "N": 16,
        },
        "data": {
            "dataset_path": roster["dataset"]["path"],
            "dataset_sha256": roster["dataset"]["sha256"],
            "folds_path": None,
            "diagnostic_scope": "optimistic_full_val200_same_set",
        },
        "parameters": copy.deepcopy(launch["parameters"]),
        "seed_candidate_ids": copy.deepcopy(launch["seed_candidate_ids"]),
        "tie_break": copy.deepcopy(launch["tie_break"]),
        "execution": {
            "cpu_only": True,
            "worker_processes": 5,
            "case_sharding": "largest_first_by_logit_elements",
            "nice": 10,
            "shared_scope_reads": True,
        },
        "code": {
            "repo_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                check=False, capture_output=True, text=True,
            ).stdout.strip() or None,
            "seeded_caruana_sha256": pe.sha256_file(Path(__file__)),
            "preliminary_ensemble_sha256": pe.sha256_file(HERE / "preliminary_ensemble.py"),
        },
        "outputs": {
            "tracked_run_dir": str(HERE / "methods" / method / "runs" / run_id),
            "runtime_run_dir": str(runtime_dir),
            "result_json": str(HERE / "methods" / method / "runs" / run_id / "result.json"),
            "report_md": str(HERE / "methods" / method / "runs" / run_id / "report.md"),
            "final_decision_json": str(HERE / "methods" / method / "runs" / run_id / "decision.json"),
            "final_logits_dir": None,
        },
        "approvals": copy.deepcopy(launch["approvals"]),
    }


def _render_run_spec(spec: dict[str, Any]) -> str:
    return "\n".join([
        f"# {spec['run_id']} — {spec['method']['id']}", "",
        "Status: `diagnostic_pending`.", "",
        "This is an optimistic same-val200 metrics-only diagnostic. It does not use OOF folds and cannot become a formal recipe.", "",
        f"Roster SHA-256: `{spec['roster']['roster_sha256']}`.",
        f"Runtime: `{spec['outputs']['runtime_run_dir']}`.",
        "No derived logits are authorized.", "",
    ])


def prepare_execution_artifacts(
    launch_path: Path, launch: dict[str, Any], readiness: dict[str, Any]
) -> Path:
    roster = build_merged_roster(launch, readiness)
    roster_path = Path(launch["outputs"]["derived_roster"])
    _write_immutable(roster_path, roster, identity_field="roster_sha256")
    runtime = Path(launch["outputs"]["runtime_root"])
    uniform_method = HERE / "methods" / "uniform_global" / "method_spec.md"
    caruana_method = HERE / "methods" / "caruana_replacement_seeded_scoped" / "method_spec.md"
    if not uniform_method.is_file() or not caruana_method.is_file():
        raise SeededCaruanaError("method specs must exist before the analysis starts")
    uniform_spec = _run_spec(
        "uniform_global", uniform_method, roster_path, roster, launch,
        launch["uniform_run_id"], runtime / "uniform",
    )
    caruana_spec = _run_spec(
        "caruana_replacement_seeded_scoped", caruana_method, roster_path, roster, launch,
        launch["caruana_run_id"], runtime / "caruana",
    )
    for spec in (uniform_spec, caruana_spec):
        run_dir = Path(spec["outputs"]["tracked_run_dir"])
        _write_immutable(run_dir / "run_spec.json", spec)
        if not (run_dir / "run_spec.md").is_file():
            pe.atomic_write_text(run_dir / "run_spec.md", _render_run_spec(spec))
    execution = {
        "schema_version": 1,
        "launch_job_id": launch["job_id"],
        "launch_job_spec_sha256": launch["job_spec_sha256"],
        "created_at_utc": pe.utc_now(),
        "roster_path": str(_container_path(roster_path)),
        "roster_sha256": roster["roster_sha256"],
        "dataset_path": str(_container_path(Path(roster["dataset"]["path"]))),
        "dataset_sha256": roster["dataset"]["sha256"],
        "workers": 5,
        "chunk_elements": int(launch["parameters"]["chunk_elements"]),
        "scopes": list(SCOPES),
        "seed_candidate_ids": copy.deepcopy(launch["seed_candidate_ids"]),
        "runtime_root": str(runtime),
        "source_job_paths": [record["path"] for record in launch["source_jobs"]],
        "uniform": {
            "run_id": launch["uniform_run_id"],
            "result_json": str(_container_path(Path(uniform_spec["outputs"]["result_json"]))),
            "report_md": str(_container_path(Path(uniform_spec["outputs"]["report_md"]))),
            "decision_json": str(_container_path(Path(uniform_spec["outputs"]["final_decision_json"]))),
        },
        "caruana": {
            "run_id": launch["caruana_run_id"],
            "result_json": str(_container_path(Path(caruana_spec["outputs"]["result_json"]))),
            "report_md": str(_container_path(Path(caruana_spec["outputs"]["report_md"]))),
            "decision_json": str(_container_path(Path(caruana_spec["outputs"]["final_decision_json"]))),
        },
        "code": caruana_spec["code"],
    }
    execution["execution_spec_sha256"] = _identity_sha(
        execution, "execution_spec_sha256", volatile=("created_at_utc",)
    )
    execution_path = Path(launch["outputs"]["caruana_run"]) / "execution_spec.json"
    _write_immutable(execution_path, execution, identity_field="execution_spec_sha256")
    return execution_path


def _metric_row(
    case: str, finding_index: int, category: str,
    gt_voxels: int, pred_voxels: int, intersection: int,
) -> dict[str, Any]:
    return {
        "case": case,
        "finding_index": finding_index,
        "category": category,
        "gt_voxels": gt_voxels,
        "pred_voxels": pred_voxels,
        "intersection_voxels": intersection,
        "dice": pe.dice_from_counts(gt_voxels, pred_voxels, intersection),
    }


def _case_file(pass_root: Path, case_name: str) -> Path:
    digest = hashlib.sha256(case_name.encode()).hexdigest()[:16]
    return pass_root / "cases" / f"{digest}.json"


def _evaluate_case(task: dict[str, Any]) -> dict[str, Any]:
    import nibabel as nib

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    case = task["case"]
    name = case["name"]
    output_path = _case_file(Path(task["pass_root"]), name)
    if output_path.is_file():
        existing = _read_json(output_path)
        if existing.get("fingerprint") != task["fingerprint"]:
            raise SeededCaruanaError(f"incompatible partial case result: {output_path}")
        return existing
    started = time.perf_counter()
    candidate_ids = task["candidate_ids"]
    arrays = {
        candidate_id: np.load(task["paths"][candidate_id], mmap_mode="r", allow_pickle=False)
        for candidate_id in candidate_ids
    }
    segmentation_dir = Path(task.get("segmentation_dir", pe.SEG_DIR))
    ground_truth = np.asanyarray(nib.load(str(segmentation_dir / name)).dataobj)
    shape = tuple(next(iter(arrays.values())).shape)
    if tuple(ground_truth.shape) != shape:
        raise SeededCaruanaError(f"{name}: logit/ground-truth geometry mismatch")
    logical_bytes = sum(array.nbytes for array in arrays.values())
    rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    chunk_elements = int(task["chunk_elements"])
    mode = task["mode"]
    for finding_key in sorted(case["findings"], key=int):
        index = int(finding_key)
        category = str(case["categories"][finding_key])
        truth = np.asarray(ground_truth[index] > 0).reshape(-1)
        gt_voxels = int(np.count_nonzero(truth))
        applicable = ["all"] + ([category] if category in SCOPES[1:] else [])
        if mode == "initial":
            names = ["uniform", *[f"seed:{scope}" for scope in applicable]]
        else:
            names = [f"trial:{scope}:{candidate}" for scope in applicable for candidate in candidate_ids]
        counts = {key: [0, 0] for key in names}
        for start in range(0, truth.size, chunk_elements):
            stop = min(start + chunk_elements, truth.size)
            probabilities = {
                candidate: pe.sigmoid_float32(
                    arrays[candidate][index].reshape(-1)[start:stop]
                )
                for candidate in candidate_ids
            }
            truth_chunk = truth[start:stop]
            if mode == "initial":
                uniform_sum = np.zeros(stop - start, dtype=np.float32)
                for candidate in candidate_ids:
                    uniform_sum += probabilities[candidate]
                predicted = uniform_sum >= np.float32(0.5 * len(candidate_ids))
                counts["uniform"][0] += int(np.count_nonzero(predicted))
                counts["uniform"][1] += int(np.count_nonzero(predicted & truth_chunk))
                for scope in applicable:
                    seed = task["seeds"][scope]
                    seed_sum = np.zeros(stop - start, dtype=np.float32)
                    for candidate in seed:
                        seed_sum += probabilities[candidate]
                    predicted = seed_sum >= np.float32(0.5 * len(seed))
                    key = f"seed:{scope}"
                    counts[key][0] += int(np.count_nonzero(predicted))
                    counts[key][1] += int(np.count_nonzero(predicted & truth_chunk))
            else:
                basket_size = int(task["basket_size"])
                for scope in applicable:
                    base = np.zeros(stop - start, dtype=np.float32)
                    for candidate, multiplicity in task["counts"][scope].items():
                        base += np.float32(multiplicity) * probabilities[candidate]
                    for candidate in candidate_ids:
                        predicted = (base + probabilities[candidate]) >= np.float32(
                            0.5 * (basket_size + 1)
                        )
                        key = f"trial:{scope}:{candidate}"
                        counts[key][0] += int(np.count_nonzero(predicted))
                        counts[key][1] += int(np.count_nonzero(predicted & truth_chunk))
        for key, (pred_voxels, intersection) in counts.items():
            rows[key].append(_metric_row(
                name, index, category, gt_voxels, pred_voxels, intersection
            ))
    output = {
        "schema_version": 1,
        "fingerprint": task["fingerprint"],
        "case": name,
        "mode": mode,
        "rows": dict(rows),
        "logical_source_bytes": logical_bytes,
        "wall_seconds": time.perf_counter() - started,
        "completed_at_utc": pe.utc_now(),
    }
    _atomic_json(output_path, output)
    return output


def _evaluate_shard(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate one immutable, voxel-balanced case shard in a persistent worker."""

    return [_evaluate_case(task) for task in tasks]


def balance_case_shards(
    cases: list[dict[str, Any]], paths: dict[str, dict[str, Path]], workers: int
) -> list[list[dict[str, Any]]]:
    first_id = next(iter(paths))
    weighted = []
    for case in cases:
        array = np.load(paths[first_id][case["name"]], mmap_mode="r", allow_pickle=False)
        weighted.append((int(array.size), case))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for weight, case in sorted(weighted, key=lambda item: (-item[0], item[1]["name"])):
        target = min(range(workers), key=lambda index: (loads[index], index))
        bins[target].append(case)
        loads[target] += weight
    return bins


def _run_pass(
    *, execution: dict[str, Any], roster: dict[str, Any], cases: list[dict[str, Any]],
    paths: dict[str, dict[str, Path]], pass_name: str, mode: str,
    seeds: dict[str, list[str]], counts: dict[str, dict[str, int]] | None = None,
    basket_size: int | None = None,
) -> dict[str, Any]:
    runtime = Path(execution["runtime_root"])
    pass_root = runtime / "passes" / pass_name
    fingerprint = pe.json_sha256({
        "execution_spec_sha256": execution["execution_spec_sha256"],
        "pass": pass_name,
        "mode": mode,
        "seeds": seeds,
        "counts": counts,
        "basket_size": basket_size,
    })
    candidate_ids = [row["id"] for row in roster["candidates"]]
    shards = balance_case_shards(cases, paths, int(execution["workers"]))
    tasks_by_shard: list[list[dict[str, Any]]] = []
    for shard in shards:
        shard_tasks = []
        for case in shard:
            shard_tasks.append({
                "case": case,
                "paths": {candidate: str(paths[candidate][case["name"]]) for candidate in candidate_ids},
                "candidate_ids": candidate_ids,
                "pass_root": str(pass_root),
                "fingerprint": fingerprint,
                "mode": mode,
                "seeds": seeds,
                "counts": counts,
                "basket_size": basket_size,
                "chunk_elements": execution["chunk_elements"],
                "segmentation_dir": execution.get("segmentation_dir", str(pe.SEG_DIR)),
            })
        tasks_by_shard.append(shard_tasks)
    started = time.perf_counter()
    context = mp.get_context("spawn")
    with context.Pool(processes=int(execution["workers"])) as pool:
        shard_outputs = pool.map(_evaluate_shard, tasks_by_shard, chunksize=1)
    outputs = [output for shard_output in shard_outputs for output in shard_output]
    rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for output in sorted(outputs, key=lambda value: value["case"]):
        for key, values in output["rows"].items():
            rows[key].extend(values)
    for values in rows.values():
        values.sort(key=lambda row: (row["case"], row["finding_index"]))
    return {
        "rows": dict(rows),
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "logical_source_bytes": sum(int(value["logical_source_bytes"]) for value in outputs),
            "sum_case_wall_seconds": math.fsum(float(value["wall_seconds"]) for value in outputs),
        },
        "fingerprint": fingerprint,
    }


def _summary_for_scope(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    selected = rows if scope == "all" else [row for row in rows if row["category"] == scope]
    summary = pe.summarize_rows(selected)
    expected = EXPECTED_FINDINGS[scope]
    if summary["findings"] != expected:
        raise SeededCaruanaError(
            f"{scope}: expected {expected} findings, got {summary['findings']}"
        )
    return summary


def _counts(sequence: list[str]) -> dict[str, int]:
    return dict(sorted(collections.Counter(sequence).items()))


def _weights(sequence: list[str]) -> dict[str, float]:
    return {key: value / len(sequence) for key, value in _counts(sequence).items()}


def _deterministic_result_sha(result: dict[str, Any]) -> str:
    ignored = {
        "timing", "pass_timings", "updated_at_utc", "completed_at_utc",
        "deterministic_result_sha256",
    }
    return pe.json_sha256({key: value for key, value in result.items() if key not in ignored})


def _render_uniform(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        "# Top-16 Uniform Global Diagnostic", "",
        "Status: `diagnostic_only`; evaluation uses the same full val200 set.", "",
        "| Overall Dice | Hits | Findings | Hit rate |",
        "| ---: | ---: | ---: | ---: |",
        f"| {metrics['dice']:.6f} | {metrics['hits']} | {metrics['findings']} | {metrics['hit_rate']:.6f} |",
        "", "## Official categories", "",
        "| Category | Dice | Hits | Findings | Hit rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category in pe.OFFICIAL_CATEGORY_ORDER:
        value = metrics["categories"][category]
        dice = "unavailable" if value["dice"] is None else f"{value['dice']:.6f}"
        rate = "unavailable" if value["hit_rate"] is None else f"{value['hit_rate']:.6f}"
        lines.append(f"| {category} | {dice} | {value['hits']} | {value['findings']} | {rate} |")
    lines.extend(["", "No derived ensemble logits were materialized.", ""])
    return "\n".join(lines)


def _render_caruana(result: dict[str, Any]) -> str:
    lines = [
        "# Top-16 Per-Scope Seeded Caruana Diagnostic", "",
        "Status: `diagnostic_only`; selection and evaluation use the same full val200 set and are optimistic.", "",
    ]
    for scope in SCOPES:
        value = result["scopes"][scope]
        lines.extend([
            f"## Scope `{scope}`", "",
            f"Fixed K=4 seed: `{', '.join(value['seed_candidate_ids'])}`.", "",
            "| K | Added | Dice | Hits | Hit rate | Marginal Dice | Best so far | Multiplicities |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ])
        for step in value["curve"]:
            lines.append(
                f"| {step['K']} | `{step['added']}` | {step['metrics']['dice']:.6f} | "
                f"{step['metrics']['hits']} | {step['metrics']['hit_rate']:.6f} | "
                f"{step['marginal_dice']:+.6f} | {'yes' if step['best_so_far'] else 'no'} | "
                f"`{json.dumps(step['counts'], sort_keys=True)}` |"
            )
        peak = value.get("peak")
        if peak:
            lines.extend(["", f"Peak: K={peak['K']}, Dice `{peak['dice']:.6f}`, hits `{peak['hits']}`.", ""])
    lines.extend(["No derived ensemble logits were materialized.", ""])
    return "\n".join(lines)


def _decision() -> dict[str, Any]:
    return {
        "status": "diagnostic_only",
        "formal_recipe": False,
        "oof": False,
        "derived_logits_materialized": False,
        "reason": "selection and evaluation both use full val200",
    }


def run_worker(execution_path: Path) -> int:
    execution = _read_json(execution_path)
    expected = _identity_sha(execution, "execution_spec_sha256", volatile=("created_at_utc",))
    if execution.get("execution_spec_sha256") != expected:
        raise SeededCaruanaError("execution spec identity SHA mismatch")
    if pe.sha256_file(Path(__file__)) != execution["code"]["seeded_caruana_sha256"]:
        raise SeededCaruanaError("seeded Caruana code SHA drift")
    roster = _read_json(Path(execution["roster_path"]))
    if roster.get("roster_sha256") != execution["roster_sha256"]:
        raise SeededCaruanaError("merged roster SHA mismatch")
    cases = pe._load_dataset(Path(execution["dataset_path"]))
    if pe.sha256_file(Path(execution["dataset_path"])) != execution["dataset_sha256"]:
        raise SeededCaruanaError("dataset SHA mismatch")
    paths = pe._case_paths(roster, cases)
    runtime = Path(execution["runtime_root"])
    runtime.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "analysis_state.json"
    started = time.perf_counter()
    _atomic_json(state_path, {
        "status": "running_initial", "current_pass": "initial",
        "started_at_utc": pe.utc_now(), "updated_at_utc": pe.utc_now(),
    })
    initial = _run_pass(
        execution=execution, roster=roster, cases=cases, paths=paths,
        pass_name="initial", mode="initial", seeds=execution["seed_candidate_ids"],
    )
    uniform_metrics = _summary_for_scope(initial["rows"]["uniform"], "all")
    uniform = {
        "schema_version": 1,
        "status": "complete",
        "method": "uniform_global",
        "diagnostic_scope": "optimistic_full_val200_same_set",
        "metrics_only": True,
        "derived_logits_written": False,
        "probability_threshold": 0.5,
        "candidate_ids": [row["id"] for row in roster["candidates"]],
        "weights": {row["id"]: 1.0 / 16 for row in roster["candidates"]},
        "metrics": uniform_metrics,
        "timing": initial["timing"],
    }
    uniform["deterministic_result_sha256"] = _deterministic_result_sha(uniform)
    _atomic_json(Path(execution["uniform"]["result_json"]), uniform)
    pe.atomic_write_text(Path(execution["uniform"]["report_md"]), _render_uniform(uniform))
    _atomic_json(Path(execution["uniform"]["decision_json"]), _decision())

    candidate_ids = [row["id"] for row in roster["candidates"]]
    sequences = {scope: list(execution["seed_candidate_ids"][scope]) for scope in SCOPES}
    scope_results: dict[str, Any] = {}
    for scope in SCOPES:
        metrics = _summary_for_scope(initial["rows"][f"seed:{scope}"], scope)
        scope_results[scope] = {
            "scope": scope,
            "findings": EXPECTED_FINDINGS[scope],
            "seed_candidate_ids": list(sequences[scope]),
            "curve": [{
                "K": 4,
                "added": "fixed_seed",
                "sequence": list(sequences[scope]),
                "counts": _counts(sequences[scope]),
                "weights": _weights(sequences[scope]),
                "metrics": metrics,
                "marginal_dice": 0.0,
                "best_so_far": True,
                "trials": None,
            }],
        }
    result = {
        "schema_version": 1,
        "status": "running",
        "method": "caruana_replacement_seeded_scoped",
        "diagnostic_scope": "optimistic_full_val200_same_set",
        "metrics_only": True,
        "derived_logits_written": False,
        "probability_threshold": 0.5,
        "hit_dice_threshold": 0.1,
        "candidate_ids": candidate_ids,
        "initial_basket_size": 4,
        "maximum_basket_size": 16,
        "with_replacement": True,
        "candidate_evaluations": 5 * 12 * 16,
        "scopes": scope_results,
        "pass_timings": {"initial": initial["timing"]},
        "updated_at_utc": pe.utc_now(),
    }
    result["deterministic_result_sha256"] = _deterministic_result_sha(result)
    _atomic_json(Path(execution["caruana"]["result_json"]), result)
    pe.atomic_write_text(Path(execution["caruana"]["report_md"]), _render_caruana(result))

    for basket_size in range(4, 16):
        next_k = basket_size + 1
        _atomic_json(state_path, {
            "status": "running_caruana", "current_pass": f"K={next_k}",
            "completed_through_k": basket_size, "updated_at_utc": pe.utc_now(),
        })
        round_result = _run_pass(
            execution=execution, roster=roster, cases=cases, paths=paths,
            pass_name=f"k{next_k:02d}", mode="round",
            seeds=execution["seed_candidate_ids"],
            counts={scope: _counts(sequences[scope]) for scope in SCOPES},
            basket_size=basket_size,
        )
        result["pass_timings"][f"K={next_k}"] = round_result["timing"]
        for scope in SCOPES:
            trials: dict[str, Any] = {}
            for candidate in candidate_ids:
                rows = round_result["rows"][f"trial:{scope}:{candidate}"]
                metrics = _summary_for_scope(rows, scope)
                trial_sequence = sequences[scope] + [candidate]
                trials[candidate] = {
                    "candidate_id": candidate,
                    "metrics": metrics,
                    "counts": _counts(trial_sequence),
                    "weights": _weights(trial_sequence),
                }
            selected = pe.select_trial(trials)
            previous = float(scope_results[scope]["curve"][-1]["metrics"]["dice"])
            sequences[scope].append(selected)
            metrics = trials[selected]["metrics"]
            best_before = min(
                scope_results[scope]["curve"],
                key=lambda step: (
                    -float(step["metrics"]["dice"]),
                    -int(step["metrics"]["hits"]),
                    int(step["K"]),
                ),
            )
            is_best = (
                float(metrics["dice"]), int(metrics["hits"])
            ) > (
                float(best_before["metrics"]["dice"]),
                int(best_before["metrics"]["hits"]),
            )
            scope_results[scope]["curve"].append({
                "K": next_k,
                "added": selected,
                "sequence": list(sequences[scope]),
                "counts": _counts(sequences[scope]),
                "weights": _weights(sequences[scope]),
                "metrics": metrics,
                "marginal_dice": float(metrics["dice"]) - previous,
                "best_so_far": is_best,
                "trials": trials,
            })
        result["updated_at_utc"] = pe.utc_now()
        result["deterministic_result_sha256"] = _deterministic_result_sha(result)
        _atomic_json(Path(execution["caruana"]["result_json"]), result)
        pe.atomic_write_text(Path(execution["caruana"]["report_md"]), _render_caruana(result))

    for scope in SCOPES:
        curve = scope_results[scope]["curve"]
        peak = min(
            curve,
            key=lambda step: (-float(step["metrics"]["dice"]), -int(step["metrics"]["hits"]), int(step["K"])),
        )
        scope_results[scope]["peak"] = {
            "K": peak["K"], "dice": peak["metrics"]["dice"],
            "hits": peak["metrics"]["hits"], "counts": peak["counts"],
        }
        scope_results[scope]["final_sequence"] = list(sequences[scope])
        scope_results[scope]["final_counts"] = _counts(sequences[scope])
        scope_results[scope]["final_weights"] = _weights(sequences[scope])
    result["status"] = "complete"
    result["completed_at_utc"] = pe.utc_now()
    result["timing"] = {
        "wall_seconds": time.perf_counter() - started,
        "logical_source_bytes": sum(
            int(value["logical_source_bytes"]) for value in result["pass_timings"].values()
        ),
    }
    result["deterministic_result_sha256"] = _deterministic_result_sha(result)
    _atomic_json(Path(execution["caruana"]["result_json"]), result)
    pe.atomic_write_text(Path(execution["caruana"]["report_md"]), _render_caruana(result))
    _atomic_json(Path(execution["caruana"]["decision_json"]), _decision())
    _atomic_json(state_path, {
        "status": "complete", "completed_through_k": 16,
        "completed_at_utc": pe.utc_now(), "updated_at_utc": pe.utc_now(),
        "uniform_dice": uniform_metrics["dice"],
        "uniform_hits": uniform_metrics["hits"],
    })
    return 0


def _analysis_job_path(job_id: str) -> Path:
    return HERE / "analysis_jobs" / job_id / "job_spec.json"


def _render_launch(spec: dict[str, Any]) -> str:
    lines = [
        f"# {spec['job_id']}", "",
        "Wait for fresh ranks 1–16, then run the top-16 uniform and five-scope seeded Caruana diagnostics.", "",
        f"Poll interval: `{spec['poll_interval_seconds']} seconds`; workers: `{spec['workers']}`.", "",
        "| Scope | Fixed seed ranks | Fixed seed candidates |",
        "| --- | --- | --- |",
    ]
    for scope in SCOPES:
        lines.append(
            f"| {scope} | {', '.join(map(str, spec['seed_ranks'][scope]))} | "
            f"`{', '.join(spec['seed_candidate_ids'][scope])}` |"
        )
    lines.extend(["", "This is an optimistic full-val diagnostic and writes no derived logits.", ""])
    return "\n".join(lines)


def plan_command(args: argparse.Namespace) -> int:
    spec = create_launch_spec(
        Path(args.roster), [Path(value) for value in args.source_job],
        top=int(args.top), workers=int(args.workers), job_id=args.job,
    )
    readiness = readiness_report(spec)
    output = {
        "mode": "apply" if args.apply else "dry-run",
        "job_id": spec["job_id"],
        "job_spec_sha256": spec["job_spec_sha256"],
        "seeds": spec["seed_candidate_ids"],
        "readiness": {
            "ready": readiness["ready"], "blocked": readiness["blocked"],
            "ready_count": readiness.get("ready_count", 0), "required_count": 16,
        },
        "estimated": {
            "full_top16_passes": 13,
            "candidate_evaluations": 960,
        },
    }
    if args.apply:
        path = _analysis_job_path(args.job)
        _write_immutable(path, spec, identity_field="job_spec_sha256")
        if not path.with_suffix(".md").is_file():
            pe.atomic_write_text(path.with_suffix(".md"), _render_launch(spec))
        output["job_spec_path"] = str(path)
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return 0


def _source_failure(spec: dict[str, Any]) -> str | None:
    for record in spec["source_jobs"]:
        job = _read_json(Path(record["path"]))
        for candidate in job.get("candidates", []):
            progress_path = Path(candidate["progress_path"])
            if progress_path.is_file():
                progress = _read_json(progress_path)
                if progress.get("status") in {"failed", "blocked"}:
                    return f"{candidate['id']}={progress.get('status')}"
    return None


def _wave_seconds_per_case(job: dict[str, Any], wave: int) -> list[float]:
    values = []
    for candidate in job.get("candidates", []):
        if int(candidate.get("wave", -1)) != wave:
            continue
        path = Path(candidate["progress_path"])
        if not path.is_file():
            continue
        progress = _read_json(path)
        value = progress.get("seconds_per_case")
        if value is not None and float(value) > 0:
            values.append(float(value))
    return values


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _launch_worker(spec: dict[str, Any], execution_path: Path) -> tuple[subprocess.Popen[Any], str, Any]:
    source_job = _read_json(Path(spec["source_jobs"][1]["path"]))
    image = source_job["container"]["image"]
    name = f"sideexp003-{spec['job_id']}-cpu".replace("_", "-")
    runtime = Path(spec["outputs"]["runtime_root"])
    runtime.mkdir(parents=True, exist_ok=True)
    log_handle = (runtime / "worker.log").open("a")
    command = [
        "docker", "run", "--rm", "--name", name,
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--workdir", "/workspace", "--cpus", "6",
        "-v", f"{REPO_ROOT}:/workspace",
        "-v", "/data/hengjie:/data/hengjie",
        "-v", "/mnt/shengdata1:/mnt/shengdata1",
        "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "OMP_NUM_THREADS=1",
        "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
        image, "python",
        "/workspace/side_experiments/sideexp003_ensemble_method_hub/seeded_caruana.py",
        "worker", "--spec", str(_container_path(execution_path)),
    ]
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
    return process, name, log_handle


def run_waiter(job_path: Path, poll_interval: int, auto_start: bool) -> int:
    spec = _read_json(job_path)
    errors = validate_launch_spec(spec)
    if errors:
        raise SeededCaruanaError("invalid launch spec: " + "; ".join(errors))
    if not auto_start:
        raise SeededCaruanaError("seeded-run requires --auto-start")
    runtime = Path(spec["outputs"]["runtime_root"])
    runtime.mkdir(parents=True, exist_ok=True)
    lock_handle = (runtime / "launch.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SeededCaruanaError("another waiter already owns this analysis job") from exc
    state_path = runtime / "orchestrator_state.json"
    if state_path.is_file() and _read_json(state_path).get("status") == "complete":
        print(json.dumps({"job_id": spec["job_id"], "status": "already_complete"}), flush=True)
        return 0
    while True:
        readiness = readiness_report(spec)
        state = {
            "job_id": spec["job_id"],
            "status": "blocked" if readiness["blocked"] else "waiting_for_wave4",
            "readiness": readiness,
            "updated_at_utc": pe.utc_now(),
        }
        _atomic_json(state_path, state)
        print(json.dumps({
            "time": state["updated_at_utc"], "status": state["status"],
            "ready": readiness.get("ready_count", 0), "required": 16,
        }, sort_keys=True), flush=True)
        if readiness["blocked"]:
            return 2
        if readiness["ready"]:
            break
        time.sleep(poll_interval)
    free = shutil.disk_usage(RUNTIME_ROOT).free
    available = _available_memory_bytes()
    if free < int(spec["safety"]["minimum_free_bytes"]):
        raise SeededCaruanaError(f"disk safety gate failed: {free / 1024**4:.2f} TiB free")
    if available < int(spec["safety"]["minimum_available_memory_bytes"]):
        raise SeededCaruanaError(f"memory safety gate failed: {available / 1024**3:.2f} GiB available")
    execution_path = prepare_execution_artifacts(job_path, spec, readiness)
    second_job = _read_json(Path(spec["source_jobs"][1]["path"]))
    wave4_baseline = _median(_wave_seconds_per_case(second_job, 4))
    process, container_name, log_handle = _launch_worker(spec, execution_path)
    degraded_polls = 0
    paused = False
    _atomic_json(state_path, {
        "job_id": spec["job_id"], "status": "running", "pid": process.pid,
        "container": container_name, "wave4_seconds_per_case_baseline": wave4_baseline,
        "execution_spec": str(execution_path), "started_at_utc": pe.utc_now(),
        "updated_at_utc": pe.utc_now(),
    })
    try:
        while process.poll() is None:
            time.sleep(poll_interval)
            failure = _source_failure(spec)
            free = shutil.disk_usage(RUNTIME_ROOT).free
            if failure or free < int(spec["safety"]["minimum_free_bytes"]):
                subprocess.run(["docker", "stop", container_name], check=False)
                reason = failure or f"disk free fell to {free / 1024**4:.2f} TiB"
                _atomic_json(state_path, {
                    "job_id": spec["job_id"], "status": "stopped_for_safety",
                    "reason": reason, "updated_at_utc": pe.utc_now(),
                })
                return 2
            second_job = _read_json(Path(spec["source_jobs"][1]["path"]))
            wave5 = _wave_seconds_per_case(second_job, 5)
            wave5_median = _median(wave5)
            wave5_complete = all(
                _candidate_cache_status(second_job, row["id"], int(row["rank"]))["ready"]
                for row in second_job.get("candidates", []) if int(row.get("wave", -1)) == 5
            )
            if wave4_baseline and wave5_median and not paused:
                degraded_polls = degraded_polls + 1 if wave5_median > 2.0 * wave4_baseline else 0
                if degraded_polls >= 2:
                    subprocess.run(["docker", "pause", container_name], check=False)
                    paused = True
            elif paused and (wave5_complete or (wave5_median and wave5_median < 1.5 * wave4_baseline)):
                subprocess.run(["docker", "unpause", container_name], check=False)
                paused = False
                degraded_polls = 0
            analysis_state_path = Path(spec["outputs"]["runtime_root"]) / "analysis_state.json"
            analysis_state = _read_json(analysis_state_path) if analysis_state_path.is_file() else None
            _atomic_json(state_path, {
                "job_id": spec["job_id"], "status": "paused_for_wave5" if paused else "running",
                "pid": process.pid, "container": container_name,
                "wave4_seconds_per_case_baseline": wave4_baseline,
                "wave5_seconds_per_case": wave5_median,
                "wave5_complete": wave5_complete,
                "degraded_polls": degraded_polls,
                "analysis": analysis_state,
                "updated_at_utc": pe.utc_now(),
            })
        code = int(process.returncode or 0)
    finally:
        log_handle.close()
    _atomic_json(state_path, {
        "job_id": spec["job_id"], "status": "complete" if code == 0 else "failed",
        "returncode": code, "completed_at_utc": pe.utc_now(), "updated_at_utc": pe.utc_now(),
    })
    return code


def watch_command(args: argparse.Namespace) -> int:
    job_path = _analysis_job_path(args.job)
    while True:
        spec = _read_json(job_path)
        runtime = Path(spec["outputs"]["runtime_root"])
        state_path = runtime / "orchestrator_state.json"
        analysis_path = runtime / "analysis_state.json"
        status = {
            "job_id": args.job,
            "readiness": readiness_report(spec),
            "orchestrator": _read_json(state_path) if state_path.is_file() else None,
            "analysis": _read_json(analysis_path) if analysis_path.is_file() else None,
            "completed_case_partials": len(list((runtime / "passes").glob("*/cases/*.json"))),
        }
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        terminal = (status["orchestrator"] or {}).get("status") in {
            "complete", "failed", "blocked", "stopped_for_safety"
        }
        if args.once or terminal:
            return 0 if not terminal or (status["orchestrator"] or {}).get("status") == "complete" else 2
        time.sleep(args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--spec", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "worker":
            try:
                os.nice(10)
            except OSError:
                pass
            if os.environ.get("NVIDIA_VISIBLE_DEVICES") not in {None, "", "void"}:
                raise SeededCaruanaError("analysis worker must not see GPUs")
            return run_worker(args.spec)
        raise SeededCaruanaError(f"unsupported command {args.command}")
    except (SeededCaruanaError, pe.PreliminaryEnsembleError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
