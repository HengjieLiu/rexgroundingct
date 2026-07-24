#!/usr/bin/env python3
"""Run VoxTell on a ReXGroundingCT split and save one 4D prediction per case."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from tqdm import tqdm
from voxtell.inference.predictor import VoxTellPredictor

from common import CT_ROOT, REX_METADATA, REX_SEG_DIR, ct_rate_abs_path, load_split_entries, sorted_prompts, write_json


DEFAULT_EVAL_LOCK_STALE_SECONDS = 12 * 60 * 60


def _serializable_ornt(ornt: np.ndarray) -> list[list[int | float]]:
    return [[int(axis), float(direction)] for axis, direction in ornt.tolist()]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _eval_root_from_output_dir(output_dir: Path) -> Path | None:
    if output_dir.name == "predictions":
        return output_dir.parent
    return None


def _expected_finding_count(entries: list[dict]) -> int:
    return sum(len(entry.get("findings", {})) for entry in entries)


def _prediction_names_complete(output_dir: Path, entries: list[dict]) -> bool:
    expected = [entry["name"] for entry in entries]
    if not output_dir.is_dir():
        return False
    actual_count = sum(1 for _ in output_dir.glob("*.nii.gz"))
    return actual_count == len(expected) and all((output_dir / name).is_file() for name in expected)


def _completed_summary(eval_root: Path, output_dir: Path, split: str, entries: list[dict]) -> bool:
    summary_json = eval_root / "reports" / f"{split}_quick_global_eval_summary.json"
    if not summary_json.is_file() or not _prediction_names_complete(output_dir, entries):
        return False
    try:
        summary = json.loads(summary_json.read_text())
    except json.JSONDecodeError:
        return False
    return (
        int(summary.get("total_cases", -1)) == len(entries)
        and int(summary.get("total_findings", -1)) == _expected_finding_count(entries)
    )


def _lock_metadata(phase: str) -> dict:
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "phase": phase,
        "command": sys.argv,
    }


def _write_lock(path: Path, phase: str) -> None:
    path.write_text(json.dumps(_lock_metadata(phase), indent=2, sort_keys=True) + "\n")


def _lock_age_seconds(path: Path) -> float:
    return max(0.0, time.time() - path.stat().st_mtime)


def _release_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _acquire_eval_lock(
    eval_root: Path,
    output_dir: Path,
    split: str,
    entries: list[dict],
    poll_seconds: float,
    stale_seconds: float,
) -> Path | None:
    if _env_flag("EVAL_LOCK_HELD") or _env_flag("EVAL_LOCK_DISABLE"):
        return None
    lock_path = eval_root / ".eval.lock"
    eval_root.mkdir(parents=True, exist_ok=True)
    while True:
        if _completed_summary(eval_root, output_dir, split, entries):
            print(f"Completed eval already exists under {eval_root}; skipping inference.", flush=True)
            return None
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_age_seconds(lock_path) > stale_seconds:
                print(f"Removing stale eval lock: {lock_path}", flush=True)
                _release_lock(lock_path)
                continue
            print(f"Waiting for eval lock: {lock_path}", flush=True)
            time.sleep(poll_seconds)
            continue
        with os.fdopen(fd, "w") as handle:
            json.dump(_lock_metadata("inference"), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return lock_path


def export_prediction_to_gt_layout(
    prediction: np.ndarray,
    gt_img: nib.Nifti1Image,
    ct_properties: dict,
    name: str,
    output_dtype: np.dtype | type | None = np.uint8,
) -> tuple[np.ndarray, dict]:
    """Convert VoxTell/nnU-Net output to ReXGroundingCT evaluator layout.

    VoxTell receives CTs through nnU-Net's ``NibabelIOWithReorient`` reader. The
    network output is therefore channel-first in nnU-Net order ``(F, Z, Y, X)``
    and in the reader's reoriented voxel space. ReXGroundingCT masks are stored
    as channel-first raw CT voxel arrays ``(F, X, Y, Z)``. This export path first
    undoes the nnU-Net axis transpose and then applies the nibabel orientation
    transform from the reoriented CT affine back to the original CT affine.
    """
    gt_shape = tuple(int(dim) for dim in gt_img.shape)
    metadata: dict = {
        "raw_prediction_shape": list(prediction.shape),
        "gt_shape": list(gt_shape),
        "gt_axcodes": list(nib.aff2axcodes(gt_img.affine)),
        "applied_transform": None,
    }

    if prediction.ndim != 4 or len(gt_shape) != 4:
        raise ValueError(f"{name}: expected 4D prediction and GT, got {prediction.shape} and {gt_shape}")
    if prediction.shape[0] != gt_shape[0]:
        raise ValueError(f"{name}: {prediction.shape[0]} predictions but {gt_shape[0]} GT findings")

    nibabel_stuff = ct_properties.get("nibabel_stuff", {})
    original_affine = nibabel_stuff.get("original_affine")
    reoriented_affine = nibabel_stuff.get("reoriented_affine", original_affine)
    if original_affine is None or reoriented_affine is None:
        raise ValueError(f"{name}: missing CT affine metadata from NibabelIOWithReorient")

    expected_nnunet_shape = (gt_shape[0], gt_shape[3], gt_shape[2], gt_shape[1])
    if tuple(prediction.shape) != expected_nnunet_shape:
        raise ValueError(
            f"{name}: prediction shape {prediction.shape} is not expected nnU-Net layout "
            f"{expected_nnunet_shape} for GT shape {gt_shape}"
        )

    original_ornt = io_orientation(original_affine)
    reoriented_ornt = io_orientation(reoriented_affine)
    transform = ornt_transform(reoriented_ornt, original_ornt)
    prediction_xyz = np.transpose(prediction, (0, 3, 2, 1))
    exported = np.stack(
        [apply_orientation(prediction_xyz[index], transform) for index in range(prediction_xyz.shape[0])],
        axis=0,
    )
    if output_dtype is not None:
        exported = exported.astype(output_dtype, copy=False)
    exported = np.ascontiguousarray(exported)

    metadata.update(
        {
            "applied_transform": "nnunet_FZYX_to_reoriented_FXYZ_then_original_ct_orientation",
            "ct_original_axcodes": list(nib.aff2axcodes(original_affine)),
            "ct_reoriented_axcodes": list(nib.aff2axcodes(reoriented_affine)),
            "ct_original_orientation": _serializable_ornt(original_ornt),
            "ct_reoriented_orientation": _serializable_ornt(reoriented_ornt),
            "orientation_transform": _serializable_ornt(transform),
            "pre_orientation_shape": list(prediction_xyz.shape),
            "final_prediction_shape": list(exported.shape),
        }
    )
    if tuple(exported.shape) != gt_shape:
        raise ValueError(f"{name}: exported prediction shape {exported.shape} != GT shape {gt_shape}")
    return exported, metadata


def save_4d_prediction(
    segmentation: np.ndarray,
    gt_path: Path,
    output_path: Path,
    output_dtype: np.dtype | type = np.uint8,
) -> None:
    """Save evaluator-compatible 4D NIfTI using GT affine/header."""
    gt_img = nib.load(str(gt_path))
    header = gt_img.header.copy()
    header.set_data_dtype(output_dtype)
    out_img = nib.Nifti1Image(segmentation.astype(output_dtype, copy=False), gt_img.affine, header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))


def load_dataset_json_entries(path: Path, split_label: str) -> list[dict]:
    data = json.loads(path.read_text())
    if "test" not in data or not isinstance(data["test"], list):
        raise ValueError(f"{path}: expected evaluator-compatible JSON with a list under key 'test'")
    entries = []
    for index, item in enumerate(data["test"]):
        entry = dict(item)
        if "name" not in entry or "findings" not in entry:
            raise ValueError(f"{path}: entry {index} is missing required 'name' or 'findings'")
        entry["_split"] = split_label
        entry["_index"] = index
        entries.append(entry)
    return entries


def predict_single_image_probabilities(
    predictor: VoxTellPredictor,
    data: np.ndarray,
    prompts: list[str],
) -> np.ndarray:
    """Run the same VoxTell path as mask inference, but return sigmoid probabilities."""
    text_embeddings = predictor.embed_text_prompts(prompts)
    data_tensor, bbox, orig_shape = predictor.preprocess(data)
    logits = predictor.predict_sliding_window_return_logits(data_tensor, text_embeddings).to("cpu")
    with torch.no_grad():
        probabilities = torch.sigmoid(logits.float()).numpy().astype(np.float32, copy=False)

    probabilities_reverted_cropping = np.zeros(
        [probabilities.shape[0], *orig_shape],
        dtype=np.float32,
    )
    probabilities_reverted_cropping = insert_crop_into_image(
        probabilities_reverted_cropping,
        probabilities,
        bbox,
    )
    probabilities_reverted_cropping = np.asarray(probabilities_reverted_cropping, dtype=np.float32)
    if not np.isfinite(probabilities_reverted_cropping).all():
        raise RuntimeError("Encountered non-finite VoxTell probabilities")
    return probabilities_reverted_cropping


def run_case(
    predictor: VoxTellPredictor,
    reader: NibabelIOWithReorient,
    entry: dict,
    ct_root: Path,
    seg_dir: Path,
    output_dir: Path,
    overwrite: bool,
    runtime_metadata: dict,
    output_type: str,
) -> dict:
    name = entry["name"]
    ct_path = ct_rate_abs_path(name, ct_root)
    gt_path = seg_dir / name
    pred_path = output_dir / name
    prompts = sorted_prompts(entry)

    result = {
        "name": name,
        "ct_path": str(ct_path),
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "num_prompts": len(prompts),
        "split": entry.get("_split"),
        "split_index": entry.get("_index"),
        "runtime": runtime_metadata,
        "output_type": output_type,
        "status": "pending",
    }
    if pred_path.exists() and not overwrite:
        result["status"] = "skipped_existing"
        return result
    if not ct_path.is_file():
        result["status"] = "missing_ct"
        return result
    if not gt_path.is_file():
        result["status"] = "missing_gt"
        return result

    image, ct_properties = reader.read_images([str(ct_path)])
    if output_type == "mask":
        prediction = predictor.predict_single_image(image, prompts)
        output_dtype = np.uint8
    elif output_type == "probability":
        prediction = predict_single_image_probabilities(predictor, image, prompts)
        output_dtype = np.float32
    else:
        raise ValueError(f"Unsupported output_type: {output_type}")
    gt_img = nib.load(str(gt_path))
    prediction, export_metadata = export_prediction_to_gt_layout(
        prediction,
        gt_img,
        ct_properties,
        name,
        output_dtype=output_dtype,
    )
    if output_type == "probability":
        min_probability = float(np.min(prediction))
        max_probability = float(np.max(prediction))
        if min_probability < -1e-6 or max_probability > 1.0 + 1e-6:
            raise RuntimeError(
                f"{name}: probability range [{min_probability}, {max_probability}] is outside [0, 1]"
            )
        result["probability_range"] = [min_probability, max_probability]
    save_4d_prediction(prediction, gt_path, pred_path, output_dtype=output_dtype)
    result["status"] = "written"
    result["shape"] = list(prediction.shape)
    result["orientation"] = export_metadata
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--case-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--no-precomputed", action="store_true")
    parser.add_argument("--output-type", choices=["mask", "probability"], default="mask")
    parser.add_argument("--status-json", type=Path, default=None)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard < num_shards")

    selected_by_case = args.case_index is not None or args.case_name is not None
    if args.dataset_json is not None:
        if selected_by_case or args.limit is not None:
            raise ValueError("--dataset-json cannot be combined with --limit, --case-index, or --case-name")
        entries = load_dataset_json_entries(args.dataset_json, args.split)
        entries = [
            entry for index, entry in enumerate(entries)
            if index % args.num_shards == args.shard_index
        ]
    else:
        entries = load_split_entries(args.metadata, [args.split])
        if args.case_index is not None:
            if args.case_index < 0 or args.case_index >= len(entries):
                raise ValueError(f"--case-index {args.case_index} is out of range for split {args.split}")
            entries = [entries[args.case_index]]
        if args.case_name is not None:
            matching = [entry for entry in entries if entry["name"] == args.case_name]
            if not matching:
                raise ValueError(f"--case-name {args.case_name!r} not found in selected {args.split} entries")
            entries = matching
        if not selected_by_case:
            entries = [
                entry for index, entry in enumerate(entries)
                if index % args.num_shards == args.shard_index
            ]
            if args.limit is not None:
                entries = entries[: args.limit]
        elif args.limit is not None:
            raise ValueError("--limit cannot be combined with --case-index or --case-name")

    if args.split == "test":
        print("WARNING: test split has no released masks; shape validation/evaluation may fail.")

    eval_root = _eval_root_from_output_dir(args.output_dir)
    lock_path = None
    if eval_root is not None:
        if _completed_summary(eval_root, args.output_dir, args.split, entries):
            print(f"Completed eval already exists under {eval_root}; skipping inference.", flush=True)
            return 0
        lock_path = _acquire_eval_lock(
            eval_root=eval_root,
            output_dir=args.output_dir,
            split=args.split,
            entries=entries,
            poll_seconds=float(os.environ.get("EVAL_LOCK_POLL_SECONDS", "60")),
            stale_seconds=float(os.environ.get("EVAL_LOCK_STALE_SECONDS", str(DEFAULT_EVAL_LOCK_STALE_SECONDS))),
        )
        if lock_path is None and _completed_summary(eval_root, args.output_dir, args.split, entries):
            return 0

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    runtime_metadata = {
        "gpu": args.gpu,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else None,
    }
    print(f"Runtime: {runtime_metadata}", flush=True)
    predictor = VoxTellPredictor(
        model_dir=str(args.model_dir) if args.model_dir else None,
        device=device,
        embedding_bank=str(args.embeddings) if args.embeddings else None,
        use_precomputed_embeddings=not args.no_precomputed,
    )
    reader = NibabelIOWithReorient()

    statuses = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in tqdm(entries, desc=f"{args.split} shard {args.shard_index}/{args.num_shards}"):
        try:
            statuses.append(
                run_case(
                    predictor=predictor,
                    reader=reader,
                    entry=entry,
                    ct_root=args.ct_root,
                    seg_dir=args.seg_dir,
                    output_dir=args.output_dir,
                    overwrite=args.overwrite,
                    runtime_metadata=runtime_metadata,
                    output_type=args.output_type,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch progress resumable
            statuses.append(
                {
                    "name": entry.get("name"),
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"ERROR {entry.get('name')}: {exc}", flush=True)
            if args.limit == 1:
                raise
        if args.status_json:
            write_json(args.status_json, {"cases": statuses})
    if args.status_json:
        write_json(args.status_json, {"cases": statuses})
    failures = [status for status in statuses if status["status"] in {"missing_ct", "missing_gt", "error"}]
    print(f"Processed {len(statuses)} cases; failures={len(failures)}")
    if lock_path is not None:
        if failures:
            _release_lock(lock_path)
        else:
            _write_lock(lock_path, "inference_complete_pending_eval")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
