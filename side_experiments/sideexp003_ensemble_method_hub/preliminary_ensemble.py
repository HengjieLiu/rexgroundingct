#!/usr/bin/env python3
"""CPU-only top-k uniform and Caruana-with-replacement diagnostics."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNTIME_ROOT = Path(
    "/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/"
    "sideexp003_ensemble_method_hub"
)
SEG_DIR = Path("/data/hengjie/datasets/rexgroundingct/segmentations")
DEFAULT_CHUNK_ELEMENTS = 4 * 1024 * 1024
MINIMUM_FREE_BYTES = 20 * 1024**4
HIT_THRESHOLD = 0.1
PROBABILITY_THRESHOLD = 0.5
DICE_EPSILON = 1e-6
OFFICIAL_CATEGORY_ORDER = (
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h",
)


class PreliminaryEnsembleError(RuntimeError):
    """Raised when a diagnostic run violates its frozen contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreliminaryEnsembleError(f"cannot read JSON {path}: {exc}") from exc


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


def code_provenance() -> dict[str, Any]:
    """Freeze the exact diagnostic implementation and repository revision."""

    entrypoint = Path(__file__).resolve()
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip() if completed.returncode == 0 else None
    return {
        "entrypoint": str(entrypoint.relative_to(REPO_ROOT)),
        "entrypoint_sha256": sha256_file(entrypoint),
        "repo_commit": commit,
    }


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity_sha(
    value: dict[str, Any], field: str, volatile: Iterable[str] = ()
) -> str:
    excluded = {field, *volatile}
    payload = {key: item for key, item in value.items() if key not in excluded}
    return json_sha256(payload)


def sigmoid_float32(values: np.ndarray) -> np.ndarray:
    """Return stable sigmoid probabilities without mutating the source."""

    output = np.array(values, dtype=np.float32, copy=True)
    np.clip(output, -30.0, 30.0, out=output)
    np.negative(output, out=output)
    np.exp(output, out=output)
    output += np.float32(1.0)
    np.reciprocal(output, out=output)
    return output


def dice_from_counts(gt_voxels: int, pred_voxels: int, intersection: int) -> float:
    denominator = int(gt_voxels) + int(pred_voxels)
    if denominator == 0:
        return 1.0
    return float(
        (2 * int(intersection) + DICE_EPSILON) / (denominator + DICE_EPSILON)
    )


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    if not selected:
        raise PreliminaryEnsembleError("cannot summarize empty finding metrics")
    dice = [float(row["dice"]) for row in selected]
    hits = sum(value >= HIT_THRESHOLD for value in dice)
    categories: dict[str, list[float]] = collections.defaultdict(list)
    for row in selected:
        categories[str(row["category"])].append(float(row["dice"]))

    def aggregate(values: list[float]) -> dict[str, Any]:
        category_hits = sum(value >= HIT_THRESHOLD for value in values)
        return {
            "findings": len(values),
            "dice": math.fsum(values) / len(values),
            "hits": category_hits,
            "hit_rate": category_hits / len(values),
        }

    return {
        "findings": len(dice),
        "dice": math.fsum(dice) / len(dice),
        "hits": hits,
        "hit_rate": hits / len(dice),
        "categories": {
            category: aggregate(categories[category]) if categories[category] else {
                "findings": 0, "dice": None, "hits": 0, "hit_rate": None
            }
            for category in OFFICIAL_CATEGORY_ORDER
        },
    }


def select_trial(trials: dict[str, dict[str, Any]]) -> str:
    if not trials:
        raise PreliminaryEnsembleError("cannot select from zero trials")
    return min(
        trials,
        key=lambda candidate_id: (
            -float(trials[candidate_id]["metrics"]["dice"]),
            -int(trials[candidate_id]["metrics"]["hits"]),
            candidate_id,
        ),
    )


def basket_weights(sequence: list[str]) -> dict[str, float]:
    counts = collections.Counter(sequence)
    return {
        candidate_id: counts[candidate_id] / len(sequence)
        for candidate_id in sorted(counts)
    }


def generate_probability_ensembles(
    probabilities: dict[str, np.ndarray], recipes: dict[str, dict[str, int]]
) -> dict[str, tuple[np.ndarray, int]]:
    if not probabilities or not recipes:
        raise PreliminaryEnsembleError("probabilities and recipes must be non-empty")
    element_counts = {value.size for value in probabilities.values()}
    if len(element_counts) != 1:
        raise PreliminaryEnsembleError("probability chunks have different sizes")
    size = next(iter(element_counts))
    ensembles = {}
    for recipe_name, recipe in recipes.items():
        ensemble = np.zeros(size, dtype=np.float32)
        basket_size = 0
        for candidate_id, multiplicity in recipe.items():
            if candidate_id not in probabilities or int(multiplicity) <= 0:
                raise PreliminaryEnsembleError(f"invalid recipe {recipe_name}")
            ensemble += np.float32(multiplicity) * probabilities[candidate_id]
            basket_size += int(multiplicity)
        ensembles[recipe_name] = (ensemble, basket_size)
    return ensembles


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    cases = value.get("test") if isinstance(value, dict) else None
    if not isinstance(cases, list) or len(cases) != 200:
        raise PreliminaryEnsembleError(f"{path}: expected 200 val200 cases")
    findings = 0
    seen: set[str] = set()
    for case in cases:
        name = case.get("name")
        if not isinstance(name, str) or name in seen:
            raise PreliminaryEnsembleError(f"{path}: invalid/duplicate case {name!r}")
        seen.add(name)
        raw_findings = case.get("findings")
        categories = case.get("categories")
        if not isinstance(raw_findings, dict) or not isinstance(categories, dict):
            raise PreliminaryEnsembleError(f"{path}: {name} lacks findings/categories")
        if set(raw_findings) != set(categories):
            raise PreliminaryEnsembleError(f"{path}: {name} finding/category mismatch")
        findings += len(raw_findings)
    if findings != 381:
        raise PreliminaryEnsembleError(f"{path}: expected 381 findings, got {findings}")
    return cases


def build_derived_roster(
    source_roster_path: Path, job_path: Path, top: int = 4
) -> dict[str, Any]:
    source = read_json(source_roster_path)
    job = read_json(job_path)
    if top < 1 or int(source.get("N", -1)) < top:
        raise PreliminaryEnsembleError(f"source roster does not contain top={top}")
    if source.get("roster_sha256") != _identity_sha(source, "roster_sha256"):
        raise PreliminaryEnsembleError("source roster identity SHA mismatch")
    by_id = {candidate["id"]: candidate for candidate in job.get("candidates", [])}
    candidates = []
    for source_candidate in source["candidates"][:top]:
        candidate_id = source_candidate["id"]
        job_candidate = by_id.get(candidate_id)
        if job_candidate is None or int(job_candidate.get("rank", -1)) != source_candidate["rank"]:
            raise PreliminaryEnsembleError(f"{candidate_id}: source roster/job mismatch")
        root = Path(job_candidate["cache_root"])
        export_path = root / "export_manifest.json"
        validation_path = root / "reproduction_validation.json"
        export = read_json(export_path)
        validation = read_json(validation_path)
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
            failures.append("incomplete_cache")
        if failures:
            raise PreliminaryEnsembleError(f"{candidate_id}: {', '.join(failures)}")
        case_records = export.get("cases")
        if not isinstance(case_records, list) or len(case_records) != 200:
            raise PreliminaryEnsembleError(f"{candidate_id}: invalid export case manifest")
        candidates.append(
            {
                "rank": int(source_candidate["rank"]),
                "id": candidate_id,
                "checkpoint": source_candidate["checkpoint"],
                "cache_key": job_candidate["cache_key"],
                "cache_root": str(root),
                "export_manifest_path": str(export_path),
                "export_manifest_sha256": sha256_file(export_path),
                "validation_path": str(validation_path),
                "validation_sha256": sha256_file(validation_path),
                "dtype": validation["dtype"],
                "array_bytes": int(validation["array_bytes"]),
                "same_pass_metrics": {
                    "dice": validation["same_pass_mean_global_dice_per_finding"],
                    "hits": validation["same_pass_total_hits"],
                },
            }
        )
    derived = {
        "schema_version": 1,
        "roster_id": f"top{top}_val200_fresh_j001",
        "created_at_utc": utc_now(),
        "source_roster": {
            "path": str(source_roster_path),
            "sha256": sha256_file(source_roster_path),
            "roster_sha256": source["roster_sha256"],
        },
        "source_job": {
            "path": str(job_path),
            "sha256": sha256_file(job_path),
            "job_id": job["job_id"],
        },
        "dataset": source["dataset"],
        "reuse_policy": "fresh_only",
        "N": top,
        "candidates": candidates,
    }
    derived["roster_sha256"] = _identity_sha(
        derived, "roster_sha256", volatile=("created_at_utc",)
    )
    return derived


def validate_derived_roster(roster: dict[str, Any]) -> list[str]:
    """Validate an immutable ensemble view without applying base-export schema rules."""

    errors: list[str] = []
    roster_id = str(roster.get("roster_id", ""))
    prefix, separator, suffix = roster_id.partition("_val200_fresh_j001")
    try:
        expected_n = int(prefix.removeprefix("top")) if separator and not suffix else -1
    except ValueError:
        expected_n = -1
    if expected_n < 1:
        errors.append("unexpected derived roster ID")
    if roster.get("roster_sha256") != _identity_sha(
        roster, "roster_sha256", volatile=("created_at_utc",)
    ):
        errors.append("derived roster content SHA mismatch")
    candidates = roster.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != expected_n:
        errors.append(f"derived roster must contain exactly {expected_n} candidates")
        return errors
    ids = [candidate.get("id") for candidate in candidates]
    if len(set(ids)) != expected_n:
        errors.append("derived roster candidate IDs are not unique")
    if [candidate.get("rank") for candidate in candidates] != list(range(1, expected_n + 1)):
        errors.append(f"derived roster ranks must be 1..{expected_n} in order")
    if roster.get("N") != expected_n or roster.get("reuse_policy") != "fresh_only":
        errors.append("derived roster N/reuse policy mismatch")
    for label in ("source_roster", "source_job"):
        record = roster.get(label, {})
        path = Path(record.get("path", ""))
        if not path.is_file():
            errors.append(f"{label} is missing: {path}")
        elif sha256_file(path) != record.get("sha256"):
            errors.append(f"{label} SHA drift")
    for candidate in candidates:
        for label in ("export_manifest", "validation"):
            path = Path(candidate.get(f"{label}_path", ""))
            if not path.is_file():
                errors.append(f"{candidate.get('id')}: missing {label}")
            elif sha256_file(path) != candidate.get(f"{label}_sha256"):
                errors.append(f"{candidate.get('id')}: {label} SHA drift")
    return errors


def _validate_or_write(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        existing = read_json(path)
        volatile = {
            "created_at_utc",
            "completed_at_utc",
            "result_sha256",
            "status",
        }
        left = {key: item for key, item in existing.items() if key not in volatile}
        right = {key: item for key, item in value.items() if key not in volatile}
        if left != right:
            raise PreliminaryEnsembleError(f"immutable tracked artifact differs: {path}")
        return
    atomic_write_json(path, value)


def _case_paths(roster: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, dict[str, Path]]:
    expected = {case["name"] for case in cases}
    output: dict[str, dict[str, Path]] = {}
    for candidate in roster["candidates"]:
        export = read_json(Path(candidate["export_manifest_path"]))
        if sha256_file(Path(candidate["export_manifest_path"])) != candidate["export_manifest_sha256"]:
            raise PreliminaryEnsembleError(f"{candidate['id']}: export manifest drift")
        if sha256_file(Path(candidate["validation_path"])) != candidate["validation_sha256"]:
            raise PreliminaryEnsembleError(f"{candidate['id']}: validation report drift")
        paths = {record["name"]: Path(record["array_path"]) for record in export["cases"]}
        if set(paths) != expected:
            raise PreliminaryEnsembleError(f"{candidate['id']}: case set mismatch")
        output[candidate["id"]] = paths
    for case in cases:
        name = case["name"]
        shapes = set()
        for candidate in roster["candidates"]:
            path = output[candidate["id"]][name]
            if not path.is_file():
                raise PreliminaryEnsembleError(f"missing source array: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype not in {np.dtype("float16"), np.dtype("float32")}:
                raise PreliminaryEnsembleError(f"{path}: unsupported dtype {array.dtype}")
            shapes.add(tuple(array.shape))
        if len(shapes) != 1:
            raise PreliminaryEnsembleError(f"{name}: source geometry mismatch")
    return output


def _process_io() -> dict[str, int | None]:
    result: dict[str, int | None] = {"read_bytes": None, "rchar": None}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, raw = line.split(":", 1)
            if key in result:
                result[key] = int(raw.strip())
    except (OSError, ValueError):
        pass
    return result


def _health_check(job_path: Path) -> None:
    job = read_json(job_path)
    free = shutil.disk_usage(RUNTIME_ROOT).free
    if free < MINIMUM_FREE_BYTES:
        raise PreliminaryEnsembleError(
            f"disk safety stop: {free / 1024**4:.2f} TiB free < 20 TiB"
        )
    for candidate in job.get("candidates", []):
        progress_path = Path(candidate["progress_path"])
        if not progress_path.is_file():
            continue
        progress = read_json(progress_path)
        if progress.get("status") in {"failed", "blocked"}:
            raise PreliminaryEnsembleError(
                f"source export job unhealthy: {candidate['id']}={progress.get('status')}"
            )


def _recipe_counts(sequence: list[str]) -> dict[str, int]:
    return dict(sorted(collections.Counter(sequence).items()))


def _evaluate_pass(
    *,
    roster: dict[str, Any],
    cases: list[dict[str, Any]],
    paths: dict[str, dict[str, Path]],
    recipes: dict[str, dict[str, int]],
    partial_path: Path,
    job_path: Path,
    chunk_elements: int,
) -> dict[str, Any]:
    import nibabel as nib

    fingerprint = json_sha256(
        {"roster_sha256": roster["roster_sha256"], "recipes": recipes, "chunk": chunk_elements}
    )
    if partial_path.is_file():
        partial = read_json(partial_path)
        if partial.get("fingerprint") != fingerprint:
            raise PreliminaryEnsembleError(f"incompatible partial result: {partial_path}")
    else:
        partial = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "recipes": recipes,
            "completed_cases": [],
            "rows": {name: [] for name in recipes},
            "timing": {
                "source_read_convert_sigmoid_seconds": 0.0,
                "ground_truth_read_seconds": 0.0,
                "ensemble_generation_seconds": 0.0,
                "threshold_dice_evaluation_seconds": 0.0,
                "aggregation_seconds": 0.0,
                "active_wall_seconds": 0.0,
                "active_cpu_seconds": 0.0,
                "logical_source_bytes": 0,
                "process_read_bytes": 0,
                "process_rchar": 0,
            },
        }
    completed = set(partial["completed_cases"])
    candidate_ids = [candidate["id"] for candidate in roster["candidates"]]
    active_wall_started = time.perf_counter()
    active_cpu_started = time.process_time()
    io_started = _process_io()

    def checkpoint_activity() -> None:
        nonlocal active_wall_started, active_cpu_started, io_started
        wall_finished = time.perf_counter()
        cpu_finished = time.process_time()
        io_finished = _process_io()
        partial["timing"]["active_wall_seconds"] += wall_finished - active_wall_started
        partial["timing"]["active_cpu_seconds"] += cpu_finished - active_cpu_started
        for key, output_key in (("read_bytes", "process_read_bytes"), ("rchar", "process_rchar")):
            before, after = io_started[key], io_finished[key]
            if before is not None and after is not None:
                partial["timing"][output_key] += max(0, int(after) - int(before))
        active_wall_started = wall_finished
        active_cpu_started = cpu_finished
        io_started = io_finished

    for case_number, case in enumerate(cases, start=1):
        name = case["name"]
        if name in completed:
            continue
        _health_check(job_path)
        arrays = {
            candidate_id: np.load(paths[candidate_id][name], mmap_mode="r", allow_pickle=False)
            for candidate_id in candidate_ids
        }
        gt_started = time.perf_counter()
        ground_truth = np.asanyarray(nib.load(str(SEG_DIR / name)).dataobj)
        partial["timing"]["ground_truth_read_seconds"] += time.perf_counter() - gt_started
        shape = tuple(next(iter(arrays.values())).shape)
        if tuple(ground_truth.shape) != shape:
            raise PreliminaryEnsembleError(f"{name}: logits {shape} != GT {ground_truth.shape}")
        partial["timing"]["logical_source_bytes"] += sum(array.nbytes for array in arrays.values())
        for finding_key in sorted(case["findings"], key=int):
            finding_index = int(finding_key)
            truth = np.asarray(ground_truth[finding_index] > 0).reshape(-1)
            gt_voxels = int(np.count_nonzero(truth))
            counts = {
                recipe_name: {"pred": 0, "intersection": 0}
                for recipe_name in recipes
            }
            for start in range(0, truth.size, chunk_elements):
                stop = min(start + chunk_elements, truth.size)
                source_started = time.perf_counter()
                probabilities = {
                    candidate_id: sigmoid_float32(
                        arrays[candidate_id][finding_index].reshape(-1)[start:stop]
                    )
                    for candidate_id in candidate_ids
                }
                partial["timing"]["source_read_convert_sigmoid_seconds"] += (
                    time.perf_counter() - source_started
                )
                generation_started = time.perf_counter()
                ensembles = generate_probability_ensembles(probabilities, recipes)
                partial["timing"]["ensemble_generation_seconds"] += (
                    time.perf_counter() - generation_started
                )
                evaluation_started = time.perf_counter()
                truth_chunk = truth[start:stop]
                for recipe_name, (ensemble, basket_size) in ensembles.items():
                    predicted = ensemble >= np.float32(PROBABILITY_THRESHOLD * basket_size)
                    counts[recipe_name]["pred"] += int(np.count_nonzero(predicted))
                    counts[recipe_name]["intersection"] += int(
                        np.count_nonzero(predicted & truth_chunk)
                    )
                partial["timing"]["threshold_dice_evaluation_seconds"] += (
                    time.perf_counter() - evaluation_started
                )
            aggregation_started = time.perf_counter()
            for recipe_name in recipes:
                record = counts[recipe_name]
                partial["rows"][recipe_name].append(
                    {
                        "case": name,
                        "finding_index": finding_index,
                        "category": case["categories"][finding_key],
                        "gt_voxels": gt_voxels,
                        "pred_voxels": record["pred"],
                        "intersection_voxels": record["intersection"],
                        "dice": dice_from_counts(
                            gt_voxels, record["pred"], record["intersection"]
                        ),
                    }
                )
            partial["timing"]["aggregation_seconds"] += time.perf_counter() - aggregation_started
        partial["completed_cases"].append(name)
        completed.add(name)
        partial["updated_at_utc"] = utc_now()
        checkpoint_activity()
        atomic_write_json(partial_path, partial)
        print(
            f"{partial_path.stem}: {len(completed)}/200 cases complete "
            f"({case_number}/200 dataset order)",
            flush=True,
        )
        del arrays, ground_truth
    checkpoint_activity()
    summaries = {name: summarize_rows(rows) for name, rows in partial["rows"].items()}
    if any(summary["findings"] != 381 for summary in summaries.values()):
        raise PreliminaryEnsembleError("pass did not produce 381 findings per recipe")
    partial["status"] = "complete"
    partial["completed_at_utc"] = utc_now()
    partial["summaries"] = summaries
    atomic_write_json(partial_path, partial)
    return partial


def _timing_total(passes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "source_read_convert_sigmoid_seconds",
        "ground_truth_read_seconds",
        "ensemble_generation_seconds",
        "threshold_dice_evaluation_seconds",
        "aggregation_seconds",
        "active_wall_seconds",
        "active_cpu_seconds",
        "logical_source_bytes",
        "process_read_bytes",
        "process_rchar",
    )
    values = list(passes)
    output = {
        key: math.fsum(float(value["timing"][key]) for value in values)
        for key in keys
    }
    for key in ("logical_source_bytes", "process_read_bytes", "process_rchar"):
        output[key] = int(output[key])
    return output


def _render_result(title: str, result: dict[str, Any]) -> str:
    lines = [
        f"# {title}", "",
        "Status: `diagnostic_only`; selection and evaluation both use full val200.", "",
        f"Wall time: `{result['timing']['active_wall_seconds'] / 60:.2f} min`. ",
        "No derived logits were materialized.", "",
    ]
    if result["method"] == "uniform_global":
        metric = result["metrics"]
        lines.extend([
            "| Dice | Hits | Findings |",
            "| ---: | ---: | ---: |",
            f"| {metric['dice']:.6f} | {metric['hits']} | {metric['findings']} |",
        ])
    else:
        lines.extend([
            "| K | Added | Basket | Dice | Hits | Marginal Dice |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ])
        for step in result["curve"]:
            lines.append(
                f"| {step['K']} | `{step['added']}` | `{','.join(step['sequence'])}` "
                f"| {step['metrics']['dice']:.6f} | {step['metrics']['hits']} "
                f"| {step['marginal_dice']:+.6f} |"
            )
        lines.extend(["", f"Final weights: `{json.dumps(result['final_weights'], sort_keys=True)}`."])
        lines.extend([
            "", "## All addition trials", "",
            "| K | Candidate | Dice | Hits | Selected |",
            "| ---: | --- | ---: | ---: | :---: |",
        ])
        for step in result["curve"]:
            for candidate_id, trial in sorted(step.get("trials", {}).items()):
                lines.append(
                    f"| {step['K']} | `{candidate_id}` | {trial['metrics']['dice']:.6f} "
                    f"| {trial['metrics']['hits']} | {'yes' if candidate_id == step['added'] else ''} |"
                )
    metric = result["metrics"] if result["method"] == "uniform_global" else result["curve"][-1]["metrics"]
    lines.extend([
        "", "## Timing", "",
        "| Component | Seconds | Minutes |",
        "| --- | ---: | ---: |",
    ])
    for label, key in (
        ("Source read + float conversion + sigmoid", "source_read_convert_sigmoid_seconds"),
        ("Ensemble arithmetic/generation", "ensemble_generation_seconds"),
        ("Threshold + Dice/hits evaluation", "threshold_dice_evaluation_seconds"),
        ("Ground-truth reads", "ground_truth_read_seconds"),
        ("Aggregation/report preparation", "aggregation_seconds"),
        ("Total active wall", "active_wall_seconds"),
        ("Total active CPU", "active_cpu_seconds"),
    ):
        seconds = result["timing"][key]
        lines.append(f"| {label} | {seconds:.3f} | {seconds / 60:.3f} |")
    lines.extend([
        "", "## Official-category metrics", "",
        "| Category | Findings | Dice | Hits | Hit rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for category in OFFICIAL_CATEGORY_ORDER:
        category_metric = metric["categories"][category]
        dice = "—" if category_metric["dice"] is None else f"{category_metric['dice']:.6f}"
        hit_rate = "—" if category_metric["hit_rate"] is None else f"{category_metric['hit_rate']:.6f}"
        lines.append(
            f"| {category} | {category_metric['findings']} | {dice} "
            f"| {category_metric['hits']} | {hit_rate} |"
        )
    return "\n".join(lines) + "\n"


def _render_comparison(uniform: dict[str, Any], caruana: dict[str, Any]) -> str:
    lines = [
        "# Top-4 Uniform vs. Caruana-Replacement Diagnostic", "",
        "This is an optimistic full-val diagnostic, not an OOF estimate or a formal recipe.", "",
        "| Method | Dice | Hits | Wall minutes | Logical source GiB |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Uniform top-4 | {uniform['metrics']['dice']:.6f} | {uniform['metrics']['hits']} "
        f"| {uniform['timing']['active_wall_seconds']/60:.2f} "
        f"| {uniform['timing']['logical_source_bytes']/1024**3:.2f} |",
    ]
    for step in caruana["curve"]:
        round_timing = caruana.get("round_timings", {}).get(f"K={step['K']}")
        wall = "—" if round_timing is None else f"{round_timing['active_wall_seconds']/60:.2f}"
        logical = "—" if round_timing is None else f"{round_timing['logical_source_bytes']/1024**3:.2f}"
        lines.append(
            f"| Caruana K={step['K']} | {step['metrics']['dice']:.6f} "
            f"| {step['metrics']['hits']} | {wall} | {logical} |"
        )
    lines.extend([
        "", "Caruana's wall time and logical bytes cover all three addition rounds: "
        f"`{caruana['timing']['active_wall_seconds']/60:.2f} min` and "
        f"`{caruana['timing']['logical_source_bytes']/1024**3:.2f} GiB`.",
        "", "No derived ensemble logits were written.", "",
    ])
    return "\n".join(lines)


def _evaluate_uniform_result(
    *,
    roster: dict[str, Any],
    cases: list[dict[str, Any]],
    paths: dict[str, dict[str, Path]],
    runtime: Path,
    job_path: Path,
    chunk_elements: int,
) -> dict[str, Any]:
    ids = [candidate["id"] for candidate in roster["candidates"]]
    uniform_pass = _evaluate_pass(
        roster=roster,
        cases=cases,
        paths=paths,
        recipes={"uniform": {candidate_id: 1 for candidate_id in ids}},
        partial_path=runtime / "uniform_global.partial.json",
        job_path=job_path,
        chunk_elements=chunk_elements,
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "method": "uniform_global",
        "diagnostic_scope": "optimistic_full_val200_same_set",
        "metrics_only": True,
        "derived_logits_written": False,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "candidate_ids": ids,
        "weights": {candidate_id: 1.0 / len(ids) for candidate_id in ids},
        "metrics": uniform_pass["summaries"]["uniform"],
        "timing": _timing_total([uniform_pass]),
    }
    result["deterministic_result_sha256"] = json_sha256(
        {key: value for key, value in result.items() if key not in {"timing", "deterministic_result_sha256"}}
    )
    return result


def _complete_run_spec(method: str, run_id: str, result_path: Path) -> None:
    run_spec_path = HERE / "methods" / method / "runs" / run_id / "run_spec.json"
    run_spec = read_json(run_spec_path)
    run_spec.update(
        {
            "status": "diagnostic_complete",
            "completed_at_utc": utc_now(),
            "result_sha256": sha256_file(result_path),
        }
    )
    atomic_write_json(run_spec_path, run_spec)
    atomic_write_text(run_spec_path.with_suffix(".md"), _render_run_spec(run_spec))


def run_uniform_worker(spec: dict[str, Any]) -> int:
    roster = read_json(Path(spec["roster_path"]))
    if roster.get("roster_sha256") != spec["roster_sha256"]:
        raise PreliminaryEnsembleError("derived roster SHA mismatch")
    cases = _load_dataset(Path(spec["dataset_path"]))
    if sha256_file(Path(spec["dataset_path"])) != spec["dataset_sha256"]:
        raise PreliminaryEnsembleError("dataset SHA mismatch")
    paths = _case_paths(roster, cases)
    job_path = Path(spec["job_path"])
    _health_check(job_path)
    runtime = Path(spec["runtime_root"])
    runtime.mkdir(parents=True, exist_ok=True)
    result = _evaluate_uniform_result(
        roster=roster,
        cases=cases,
        paths=paths,
        runtime=runtime,
        job_path=job_path,
        chunk_elements=int(spec["chunk_elements"]),
    )
    outputs = spec["outputs"]
    result_path = Path(outputs["result_json"])
    atomic_write_json(result_path, result)
    atomic_write_text(Path(outputs["report_md"]), _render_result("Top-k Uniform Global Diagnostic", result))
    atomic_write_json(
        Path(outputs["decision_json"]),
        {
            "status": "diagnostic_only",
            "formal_recipe": False,
            "oof": False,
            "derived_logits_materialized": False,
            "reason": "evaluation uses full val200",
        },
    )
    _complete_run_spec("uniform_global", spec["run_id"], result_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "run_id": spec["run_id"],
                "N": len(roster["candidates"]),
                "dice": result["metrics"]["dice"],
                "hits": result["metrics"]["hits"],
                "wall_seconds": result["timing"]["active_wall_seconds"],
                "derived_logits_written": False,
                "deterministic_result_sha256": result["deterministic_result_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def run_worker(spec_path: Path) -> int:
    spec = read_json(spec_path)
    try:
        os.nice(10)
    except OSError:
        pass
    if os.environ.get("NVIDIA_VISIBLE_DEVICES") not in {None, "", "void"}:
        raise PreliminaryEnsembleError("worker must run without GPU devices")
    if spec.get("task") == "uniform_only":
        return run_uniform_worker(spec)
    roster = read_json(Path(spec["roster_path"]))
    if roster.get("roster_sha256") != spec["roster_sha256"]:
        raise PreliminaryEnsembleError("derived roster SHA mismatch")
    cases = _load_dataset(Path(spec["dataset_path"]))
    if sha256_file(Path(spec["dataset_path"])) != spec["dataset_sha256"]:
        raise PreliminaryEnsembleError("dataset SHA mismatch")
    paths = _case_paths(roster, cases)
    _health_check(Path(spec["job_path"]))
    ids = [candidate["id"] for candidate in roster["candidates"]]
    runtime = Path(spec["runtime_root"])
    runtime.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()

    uniform_pass = _evaluate_pass(
        roster=roster,
        cases=cases,
        paths=paths,
        recipes={"uniform_top4": {candidate_id: 1 for candidate_id in ids}},
        partial_path=runtime / "uniform_global.partial.json",
        job_path=Path(spec["job_path"]),
        chunk_elements=int(spec["chunk_elements"]),
    )
    uniform = {
        "schema_version": 1,
        "status": "complete",
        "method": "uniform_global",
        "diagnostic_scope": "optimistic_full_val200_same_set",
        "metrics_only": True,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "candidate_ids": ids,
        "weights": {candidate_id: 0.25 for candidate_id in ids},
        "metrics": uniform_pass["summaries"]["uniform_top4"],
        "timing": _timing_total([uniform_pass]),
    }

    sequence = [ids[0]]
    curve = []
    caruana_passes = []
    previous_dice = None
    for basket_size in range(2, 5):
        recipes = {
            candidate_id: _recipe_counts(sequence + [candidate_id])
            for candidate_id in ids
        }
        if basket_size == 2:
            recipes["__initial_rank1__"] = _recipe_counts(sequence)
        evaluated = _evaluate_pass(
            roster=roster,
            cases=cases,
            paths=paths,
            recipes=recipes,
            partial_path=runtime / f"caruana_k{basket_size:02d}.partial.json",
            job_path=Path(spec["job_path"]),
            chunk_elements=int(spec["chunk_elements"]),
        )
        caruana_passes.append(evaluated)
        if basket_size == 2:
            initial_metrics = evaluated["summaries"]["__initial_rank1__"]
            previous_dice = float(initial_metrics["dice"])
            curve.append({
                "K": 1,
                "added": ids[0],
                "sequence": list(sequence),
                "counts": _recipe_counts(sequence),
                "weights": basket_weights(sequence),
                "metrics": initial_metrics,
                "marginal_dice": 0.0,
            })
        trials = {
            candidate_id: {
                "candidate_id": candidate_id,
                "sequence": sequence + [candidate_id],
                "counts": recipes[candidate_id],
                "metrics": evaluated["summaries"][candidate_id],
            }
            for candidate_id in ids
        }
        selected = select_trial(trials)
        sequence.append(selected)
        metrics = trials[selected]["metrics"]
        assert previous_dice is not None
        curve.append({
            "K": basket_size,
            "added": selected,
            "sequence": list(sequence),
            "counts": _recipe_counts(sequence),
            "weights": basket_weights(sequence),
            "metrics": metrics,
            "marginal_dice": float(metrics["dice"]) - previous_dice,
            "trials": trials,
        })
        previous_dice = float(metrics["dice"])
    caruana = {
        "schema_version": 1,
        "status": "complete",
        "method": "caruana_replacement_global",
        "diagnostic_scope": "optimistic_full_val200_same_set",
        "metrics_only": True,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "start_candidate": ids[0],
        "candidate_ids": ids,
        "addition_rounds": 3,
        "candidate_evaluations": 12,
        "curve": curve,
        "final_sequence": sequence,
        "final_counts": _recipe_counts(sequence),
        "final_weights": basket_weights(sequence),
        "round_timings": {
            f"K={basket_size}": _timing_total([evaluated])
            for basket_size, evaluated in zip(range(2, 5), caruana_passes)
        },
        "timing": _timing_total(caruana_passes),
    }
    comparison = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic_scope": "optimistic_full_val200_same_set",
        "roster_sha256": roster["roster_sha256"],
        "uniform": {
            "dice": uniform["metrics"]["dice"],
            "hits": uniform["metrics"]["hits"],
            "wall_seconds": uniform["timing"]["active_wall_seconds"],
        },
        "caruana_final": {
            "dice": curve[-1]["metrics"]["dice"],
            "hits": curve[-1]["metrics"]["hits"],
            "sequence": sequence,
            "weights": caruana["final_weights"],
            "wall_seconds": caruana["timing"]["active_wall_seconds"],
        },
        "total_worker_wall_seconds": time.perf_counter() - run_started,
        "derived_logits_written": False,
    }
    comparison["deterministic_result_sha256"] = json_sha256(
        {key: value for key, value in comparison.items() if key not in {"total_worker_wall_seconds", "deterministic_result_sha256"}}
    )
    outputs = spec["outputs"]
    atomic_write_json(Path(outputs["uniform_result_json"]), uniform)
    atomic_write_text(Path(outputs["uniform_report_md"]), _render_result("Top-4 Uniform Global Diagnostic", uniform))
    atomic_write_json(Path(outputs["caruana_result_json"]), caruana)
    atomic_write_text(Path(outputs["caruana_report_md"]), _render_result("Top-4 Caruana Replacement Diagnostic", caruana))
    atomic_write_json(Path(outputs["comparison_json"]), comparison)
    atomic_write_text(Path(outputs["comparison_md"]), _render_comparison(uniform, caruana))
    decision = {
        "status": "diagnostic_only",
        "formal_recipe": False,
        "oof": False,
        "derived_logits_materialized": False,
        "reason": "selection and evaluation both use full val200",
    }
    atomic_write_json(Path(outputs["uniform_decision_json"]), decision)
    atomic_write_json(Path(outputs["caruana_decision_json"]), decision)
    for method, result_path in (
        ("uniform_global", Path(outputs["uniform_result_json"])),
        ("caruana_replacement_global", Path(outputs["caruana_result_json"])),
    ):
        _complete_run_spec(method, spec["run_id"], result_path)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)
    return 0


def _run_spec(method: str, method_spec: Path, roster_path: Path, roster: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = HERE / "methods" / method / "runs" / spec["run_id"]
    runtime_dir = RUNTIME_ROOT / "methods" / method / "runs" / spec["run_id"]
    return {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "status": "diagnostic_running",
        "method": {"id": method, "spec_path": str(method_spec), "spec_sha256": sha256_file(method_spec)},
        "code": spec["code"],
        "roster": {"path": str(roster_path), "sha256": sha256_file(roster_path), "roster_sha256": roster["roster_sha256"], "candidate_ids": [row["id"] for row in roster["candidates"]], "N": len(roster["candidates"])},
        "data": {"dataset_path": roster["dataset"]["path"], "dataset_sha256": spec["dataset_sha256"], "folds_path": None, "diagnostic_scope": "optimistic_full_val200_same_set"},
        "parameters": {"averaging_domain": "post_sigmoid_probability", "threshold": 0.5, "chunk_elements": spec["chunk_elements"], "metrics_only": True, "basket_size": 4 if method.startswith("caruana") else None, "with_replacement": method.startswith("caruana")},
        "tie_break": ["mean_finding_dice_desc", "hits_desc", "candidate_id_asc"],
        "outputs": {"tracked_run_dir": str(run_dir), "runtime_run_dir": str(runtime_dir), "final_logits_dir": None},
        "approvals": {"roster_confirmed": True, "diagnostic_full_val_confirmed": True, "materialization_confirmed": False},
    }


def _render_run_spec(run: dict[str, Any]) -> str:
    return "\n".join([
        f"# {run['run_id']} — {run['method']['id']}", "",
        f"Status: `{run['status']}`.", "",
        "This run is a metrics-only optimistic full-val200 diagnostic. It does not use OOF folds and does not materialize derived logits.", "",
        f"Roster SHA-256: `{run['roster']['roster_sha256']}`.",
        f"Runtime: `{run['outputs']['runtime_run_dir']}`.", "",
    ])


def _container_repo_path(path: Path) -> Path:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise PreliminaryEnsembleError(f"tracked path is outside repository: {path}") from exc
    return Path("/workspace") / relative


def prepare_comparison(
    source_roster_path: Path,
    job_path: Path,
    run_id: str,
    *,
    apply: bool,
) -> tuple[dict[str, Any], Path]:
    roster = build_derived_roster(source_roster_path, job_path)
    roster_path = HERE / "rosters" / "roster_top4_val200_fresh_j001.json"
    uniform_dir = HERE / "methods" / "uniform_global" / "runs" / run_id
    caruana_dir = HERE / "methods" / "caruana_replacement_global" / "runs" / run_id
    runtime = RUNTIME_ROOT / "comparisons" / run_id
    spec_path = caruana_dir / "comparison_spec.json"
    uniform_container_dir = _container_repo_path(uniform_dir)
    caruana_container_dir = _container_repo_path(caruana_dir)
    spec = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "roster_path": str(_container_repo_path(roster_path)),
        "roster_sha256": roster["roster_sha256"],
        "job_path": str(_container_repo_path(job_path)),
        "dataset_path": str(
            _container_repo_path(Path(roster["dataset"]["path"]))
        ),
        "dataset_sha256": roster["dataset"]["sha256"],
        "chunk_elements": DEFAULT_CHUNK_ELEMENTS,
        "runtime_root": str(runtime),
        "execution": {"cpu_only": True, "single_process": True, "nice": 10, "metrics_only": True},
        "code": code_provenance(),
        "outputs": {
            "uniform_result_json": str(uniform_container_dir / "result.json"),
            "uniform_report_md": str(uniform_container_dir / "report.md"),
            "uniform_decision_json": str(uniform_container_dir / "decision.json"),
            "caruana_result_json": str(caruana_container_dir / "result.json"),
            "caruana_report_md": str(caruana_container_dir / "report.md"),
            "caruana_decision_json": str(caruana_container_dir / "decision.json"),
            "comparison_json": str(caruana_container_dir / "comparison.json"),
            "comparison_md": str(caruana_container_dir / "comparison.md"),
        },
    }
    spec["comparison_spec_sha256"] = _identity_sha(
        spec, "comparison_spec_sha256", volatile=("created_at_utc",)
    )
    if apply:
        _validate_or_write(roster_path, roster)
        uniform_method = HERE / "methods" / "uniform_global" / "method_spec.md"
        caruana_method = HERE / "methods" / "caruana_replacement_global" / "method_spec.md"
        if not uniform_method.is_file() or not caruana_method.is_file():
            raise PreliminaryEnsembleError("method specs must exist before creating runs")
        for method, method_path, run_dir in (
            ("uniform_global", uniform_method, uniform_dir),
            ("caruana_replacement_global", caruana_method, caruana_dir),
        ):
            run = _run_spec(method, method_path, roster_path, roster, spec)
            _validate_or_write(run_dir / "run_spec.json", run)
            if not (run_dir / "run_spec.md").is_file():
                atomic_write_text(run_dir / "run_spec.md", _render_run_spec(run))
        _validate_or_write(spec_path, spec)
    return spec, spec_path


def prepare_uniform(
    source_roster_path: Path,
    job_path: Path,
    run_id: str,
    top: int,
    *,
    apply: bool,
) -> tuple[dict[str, Any], Path]:
    roster = build_derived_roster(source_roster_path, job_path, top=top)
    roster_path = HERE / "rosters" / f"roster_top{top}_val200_fresh_j001.json"
    run_dir = HERE / "methods" / "uniform_global" / "runs" / run_id
    runtime = RUNTIME_ROOT / "methods" / "uniform_global" / "runs" / run_id
    spec_path = run_dir / "uniform_spec.json"
    container_run_dir = _container_repo_path(run_dir)
    spec = {
        "schema_version": 1,
        "task": "uniform_only",
        "run_id": run_id,
        "top": top,
        "created_at_utc": utc_now(),
        "roster_path": str(_container_repo_path(roster_path)),
        "roster_sha256": roster["roster_sha256"],
        "job_path": str(_container_repo_path(job_path)),
        "dataset_path": str(_container_repo_path(Path(roster["dataset"]["path"]))),
        "dataset_sha256": roster["dataset"]["sha256"],
        "chunk_elements": DEFAULT_CHUNK_ELEMENTS,
        "runtime_root": str(runtime),
        "execution": {
            "cpu_only": True,
            "single_process": True,
            "nice": 10,
            "metrics_only": True,
        },
        "code": code_provenance(),
        "outputs": {
            "result_json": str(container_run_dir / "result.json"),
            "report_md": str(container_run_dir / "report.md"),
            "decision_json": str(container_run_dir / "decision.json"),
        },
    }
    spec["uniform_spec_sha256"] = _identity_sha(
        spec, "uniform_spec_sha256", volatile=("created_at_utc",)
    )
    if apply:
        _validate_or_write(roster_path, roster)
        method_path = HERE / "methods" / "uniform_global" / "method_spec.md"
        if not method_path.is_file():
            raise PreliminaryEnsembleError("uniform_global method spec must exist before creating a run")
        run = _run_spec("uniform_global", method_path, roster_path, roster, spec)
        _validate_or_write(run_dir / "run_spec.json", run)
        if not (run_dir / "run_spec.md").is_file():
            atomic_write_text(run_dir / "run_spec.md", _render_run_spec(run))
        _validate_or_write(spec_path, spec)
    return spec, spec_path


def uniform_command(args: argparse.Namespace) -> int:
    source_roster_path = Path(args.roster).resolve()
    job_path = Path(args.job).resolve()
    result_path = HERE / "methods" / "uniform_global" / "runs" / args.run_id / "result.json"
    if args.apply and result_path.is_file():
        result = read_json(result_path)
        if result.get("status") == "complete":
            print(json.dumps({"mode": "already-complete", **result}, indent=2, sort_keys=True))
            return 0
    spec, spec_path = prepare_uniform(
        source_roster_path,
        job_path,
        args.run_id,
        int(args.top),
        apply=bool(args.apply),
    )
    roster = build_derived_roster(source_roster_path, job_path, top=int(args.top))
    cases = _load_dataset(Path(roster["dataset"]["path"]))
    paths = _case_paths(roster, cases)
    _health_check(job_path)
    logical_bytes = sum(int(candidate["array_bytes"]) for candidate in roster["candidates"])
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "method": "uniform_global",
                "run_id": args.run_id,
                "N": len(roster["candidates"]),
                "roster_sha256": spec["roster_sha256"],
                "metrics_only": True,
                "cpu_only": True,
                "runtime_root": spec["runtime_root"],
                "preflight": {
                    "cases": len(cases),
                    "findings": sum(len(case["findings"]) for case in cases),
                    "source_arrays": sum(len(value) for value in paths.values()),
                    "logical_source_bytes": logical_bytes,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.apply:
        return 0
    job = read_json(job_path)
    image = job["container"]["image"]
    container_spec = Path("/workspace") / spec_path.relative_to(REPO_ROOT)
    name = f"sideexp003_{args.run_id}_cpu".replace("_", "-")
    command = [
        "docker", "run", "--rm", "--name", name,
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--workdir", "/workspace",
        "-v", f"{REPO_ROOT}:/workspace",
        "-v", "/data/hengjie:/data/hengjie",
        "-v", "/mnt/shengdata1:/mnt/shengdata1",
        "-e", "HOME=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "NVIDIA_VISIBLE_DEVICES=void",
        image,
        "python", "/workspace/side_experiments/sideexp003_ensemble_method_hub/preliminary_ensemble.py",
        "worker", "--spec", str(container_spec),
    ]
    return subprocess.run(command, check=False).returncode


def compare_command(args: argparse.Namespace) -> int:
    source_roster_path = Path(args.roster).resolve()
    job_path = Path(args.job).resolve()
    completed_path = (
        HERE / "methods" / "caruana_replacement_global" / "runs" /
        args.run_id / "comparison.json"
    )
    if args.apply and completed_path.is_file():
        completed = read_json(completed_path)
        if completed.get("status") == "complete":
            print(json.dumps({"mode": "already-complete", **completed}, indent=2, sort_keys=True))
            return 0
    spec, spec_path = prepare_comparison(
        source_roster_path, job_path, args.run_id, apply=bool(args.apply)
    )
    roster = build_derived_roster(source_roster_path, job_path)
    cases = _load_dataset(Path(roster["dataset"]["path"]))
    paths = _case_paths(roster, cases)
    _health_check(job_path)
    logical_bytes = sum(
        int(candidate["array_bytes"]) for candidate in roster["candidates"]
    )
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "run_id": args.run_id,
        "roster_sha256": spec["roster_sha256"],
        "methods": ["uniform_global", "caruana_replacement_global"],
        "candidate_evaluations": {"uniform": 1, "caruana_additions": 12},
        "metrics_only": True,
        "cpu_only": True,
        "runtime_root": spec["runtime_root"],
        "preflight": {
            "cases": len(cases),
            "findings": sum(len(case["findings"]) for case in cases),
            "source_arrays": sum(len(value) for value in paths.values()),
            "uniform_logical_source_bytes": logical_bytes,
            "caruana_logical_source_bytes": logical_bytes * 3,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not args.apply:
        return 0
    job = read_json(job_path)
    image = job["container"]["image"]
    container_spec = Path("/workspace") / spec_path.relative_to(REPO_ROOT)
    name = f"sideexp003_{args.run_id}_cpu".replace("_", "-")
    command = [
        "docker", "run", "--rm", "--name", name,
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--workdir", "/workspace",
        "-v", f"{REPO_ROOT}:/workspace",
        "-v", "/data/hengjie:/data/hengjie",
        "-v", "/mnt/shengdata1:/mnt/shengdata1",
        "-e", "HOME=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "NVIDIA_VISIBLE_DEVICES=void",
        image,
        "python", "/workspace/side_experiments/sideexp003_ensemble_method_hub/preliminary_ensemble.py",
        "worker", "--spec", str(container_spec),
    ]
    return subprocess.run(command, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--spec", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "worker":
            return run_worker(args.spec)
        raise PreliminaryEnsembleError(f"unsupported command {args.command}")
    except PreliminaryEnsembleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
