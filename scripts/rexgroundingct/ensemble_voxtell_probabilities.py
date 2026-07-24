#!/usr/bin/env python3
"""Average VoxTell probability maps and threshold them for ReXGroundingCT eval."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
from pathlib import Path

import nibabel as nib
import numpy as np

from common import sha256_file, write_json


DEFAULT_THRESHOLDS = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]


def parse_label_path(values: list[str], flag: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} expects LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"{flag} got an empty label in {value!r}")
        if label in parsed:
            raise ValueError(f"{flag} label {label!r} was provided more than once")
        parsed[label] = Path(path)
    return parsed


def load_dataset_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    entries = data.get("test")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected evaluator-compatible JSON with list key 'test'")
    for index, entry in enumerate(entries):
        if "name" not in entry or "findings" not in entry:
            raise ValueError(f"{path}: entry {index} is missing name or findings")
    return entries


def threshold_label(threshold: float) -> str:
    return f"thr{int(round(threshold * 100)):03d}"


def threshold_dir_name(input_count: int, threshold: float) -> str:
    return f"probavg{input_count}_{threshold_label(threshold)}"


def load_probability(path: Path, expected_findings: int, reference: dict | None, name: str, label: str) -> tuple[np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing probability file for {name}: {path}")
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32, copy=False)
    if data.ndim != 4:
        raise ValueError(f"{label} {name}: expected 4D probability map, got shape {data.shape}")
    if data.shape[0] != expected_findings:
        raise ValueError(f"{label} {name}: expected {expected_findings} findings, got {data.shape[0]}")
    if not np.isfinite(data).all():
        raise ValueError(f"{label} {name}: probability map contains non-finite values")
    min_value = float(np.min(data))
    max_value = float(np.max(data))
    if min_value < -1e-4 or max_value > 1.0 + 1e-4:
        raise ValueError(f"{label} {name}: probability range [{min_value}, {max_value}] is outside [0, 1]")
    if reference is not None:
        if tuple(data.shape) != tuple(reference["shape"]):
            raise ValueError(f"{label} {name}: shape {data.shape} != reference {reference['shape']}")
        if not np.allclose(img.affine, reference["affine"], atol=1e-5):
            raise ValueError(f"{label} {name}: affine does not match reference")
    metadata = {
        "shape": list(data.shape),
        "affine": img.affine,
        "header": img.header.copy(),
        "min": min_value,
        "max": max_value,
    }
    return data, metadata


def save_mask(mask: np.ndarray, reference: dict, output_path: Path) -> None:
    header = reference["header"].copy()
    header.set_data_dtype(np.uint8)
    img = nib.Nifti1Image(mask.astype(np.uint8, copy=False), reference["affine"], header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))


def complete_prediction_dir(prediction_dir: Path, entries: list[dict]) -> bool:
    if not prediction_dir.is_dir():
        return False
    return all((prediction_dir / entry["name"]).is_file() for entry in entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--input", dest="inputs", action="append", required=True, help="LABEL=PROBABILITY_DIR")
    parser.add_argument("--model-dir", dest="model_dirs", action="append", default=[], help="LABEL=MODEL_DIR")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest-json", type=Path, default=None)
    args = parser.parse_args()

    input_dirs = parse_label_path(args.inputs, "--input")
    model_dirs = parse_label_path(args.model_dirs, "--model-dir") if args.model_dirs else {}
    labels = list(input_dirs)
    if args.weights is None:
        weights = np.full(len(labels), 1.0 / len(labels), dtype=np.float32)
    else:
        if len(args.weights) != len(labels):
            raise ValueError("--weights must have the same length as --input")
        weights = np.asarray(args.weights, dtype=np.float32)
        if np.any(weights < 0):
            raise ValueError("--weights must be non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("--weights must sum to a positive value")
        weights = weights / weights.sum()

    thresholds = sorted(float(value) for value in args.thresholds)
    if any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("--thresholds must be within [0, 1]")
    if len(set(threshold_label(value) for value in thresholds)) != len(thresholds):
        raise ValueError("Threshold labels collide after rounding to percent")

    entries = load_dataset_entries(args.dataset_json)
    args.output_root.mkdir(parents=True, exist_ok=True)
    threshold_prediction_dirs = {
        threshold: args.output_root / threshold_dir_name(len(labels), threshold) / "predictions"
        for threshold in thresholds
    }

    input_hashes: dict[str, dict[str, str]] = {label: {} for label in labels}
    case_summaries = []
    for entry in entries:
        name = entry["name"]
        expected_findings = len(entry.get("findings", {}))
        weighted_sum: np.ndarray | None = None
        reference: dict | None = None
        source_ranges = {}
        for index, label in enumerate(labels):
            probability_path = input_dirs[label] / name
            input_hashes[label][name] = sha256_file(probability_path)
            probabilities, metadata = load_probability(probability_path, expected_findings, reference, name, label)
            source_ranges[label] = [metadata["min"], metadata["max"]]
            if reference is None:
                reference = metadata
                weighted_sum = np.zeros_like(probabilities, dtype=np.float32)
            weighted_sum += probabilities * weights[index]

        assert weighted_sum is not None and reference is not None
        average_min = float(np.min(weighted_sum))
        average_max = float(np.max(weighted_sum))
        if average_min < -1e-4 or average_max > 1.0 + 1e-4:
            raise ValueError(f"{name}: averaged probability range [{average_min}, {average_max}] is outside [0, 1]")

        for threshold, prediction_dir in threshold_prediction_dirs.items():
            output_path = prediction_dir / name
            if output_path.exists() and not args.overwrite:
                continue
            mask = weighted_sum >= threshold
            save_mask(mask, reference, output_path)

        case_summaries.append(
            {
                "name": name,
                "findings": expected_findings,
                "shape": reference["shape"],
                "input_probability_ranges": source_ranges,
                "average_probability_range": [average_min, average_max],
            }
        )

    output_dirs = {}
    for threshold, prediction_dir in threshold_prediction_dirs.items():
        if not complete_prediction_dir(prediction_dir, entries):
            raise RuntimeError(f"Incomplete threshold prediction directory: {prediction_dir}")
        threshold_root = prediction_dir.parent
        record = {
            "threshold": threshold,
            "threshold_label": threshold_label(threshold),
            "prediction_dir": str(prediction_dir),
            "dataset_json": str(args.dataset_json),
            "dataset_json_sha256": sha256_file(args.dataset_json),
            "inputs": {label: str(path) for label, path in input_dirs.items()},
            "weights": {label: float(weight) for label, weight in zip(labels, weights)},
        }
        write_json(threshold_root / "ensemble_manifest.json", record)
        output_dirs[threshold_label(threshold)] = str(threshold_root)

    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "dataset_json": str(args.dataset_json),
        "dataset_json_sha256": sha256_file(args.dataset_json),
        "case_count": len(entries),
        "finding_count": sum(len(entry.get("findings", {})) for entry in entries),
        "thresholds": thresholds,
        "input_labels": labels,
        "input_probability_dirs": {label: str(path) for label, path in input_dirs.items()},
        "model_dirs": {label: str(model_dirs[label]) for label in labels if label in model_dirs},
        "weights": {label: float(weight) for label, weight in zip(labels, weights)},
        "input_probability_sha256": input_hashes,
        "case_summaries": case_summaries,
        "threshold_output_dirs": output_dirs,
    }
    manifest_json = args.manifest_json or args.output_root / "manifest.json"
    write_json(manifest_json, manifest)
    print(f"Ensemble manifest: {manifest_json}", flush=True)
    print(f"Threshold outputs: {len(threshold_prediction_dirs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
