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
from voxtell_dual_branch import (
    DualBranchVoxTellPredictor,
    load_model_spec as load_dual_model_spec,
)
from voxtell_s3_attention import (
    S3AttentionVoxTellPredictor,
    load_model_spec as load_s3_model_spec,
)


DEFAULT_EVAL_LOCK_STALE_SECONDS = 12 * 60 * 60


def load_model_spec_kind(model_dir: Path | None) -> tuple[str, dict] | None:
    if model_dir is None:
        return None
    path = Path(model_dir) / "model_spec.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    model_type = data.get("model_type")
    if model_type == "voxtell_dual_branch_v1":
        return "dual", load_dual_model_spec(Path(model_dir))
    if model_type == "voxtell_s3_attention_v1":
        return "s3", load_s3_model_spec(Path(model_dir))
    raise ValueError(f"Unsupported model type in {path}: {model_type!r}")


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not thresholds:
        raise ValueError("At least one proposal threshold is required")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("Proposal thresholds must be between 0 and 1")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Proposal thresholds must be unique")
    return thresholds


def threshold_label(threshold: float) -> str:
    return f"thr{int(round(float(threshold) * 100)):03d}"


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


def _all_required_outputs_complete(
    eval_root: Path,
    output_dir: Path,
    split: str,
    entries: list[dict],
    additional_output_dirs: list[Path] | None = None,
) -> bool:
    return _completed_summary(eval_root, output_dir, split, entries) and all(
        _prediction_names_complete(required_dir, entries)
        for required_dir in (additional_output_dirs or [])
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
    additional_output_dirs: list[Path] | None,
    poll_seconds: float,
    stale_seconds: float,
) -> Path | None:
    if _env_flag("EVAL_LOCK_HELD") or _env_flag("EVAL_LOCK_DISABLE"):
        return None
    lock_path = eval_root / ".eval.lock"
    eval_root.mkdir(parents=True, exist_ok=True)
    while True:
        if _all_required_outputs_complete(
            eval_root,
            output_dir,
            split,
            entries,
            additional_output_dirs,
        ):
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


def predict_preprocessed_crop_probabilities(
    predictor: VoxTellPredictor,
    image_czyx: np.ndarray,
    prompts: list[str],
) -> np.ndarray:
    """Run VoxTell on an already cropped and normalized cached image."""
    text_embeddings = predictor.embed_text_prompts(prompts)
    data_tensor = torch.from_numpy(np.ascontiguousarray(image_czyx, dtype=np.float32))
    logits = predictor.predict_sliding_window_return_logits(data_tensor, text_embeddings).to("cpu")
    with torch.no_grad():
        probabilities = torch.sigmoid(logits.float()).numpy().astype(np.float32, copy=False)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Encountered non-finite cached VoxTell probabilities")
    return np.ascontiguousarray(probabilities)


def predict_preprocessed_crop_branch_probabilities(
    predictor: DualBranchVoxTellPredictor,
    image_czyx: np.ndarray,
    prompts: list[str],
) -> dict[str, np.ndarray]:
    """Run one dual model pass and return proposal and final probabilities."""
    text_embeddings = predictor.embed_text_prompts(prompts)
    data_tensor = torch.from_numpy(np.ascontiguousarray(image_czyx, dtype=np.float32))
    logits = predictor.predict_sliding_window_return_branch_logits(
        data_tensor,
        text_embeddings,
    )
    probabilities: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for branch, branch_logits in logits.items():
            value = (
                torch.sigmoid(branch_logits.float())
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if not np.isfinite(value).all():
                raise RuntimeError(
                    f"Encountered non-finite cached VoxTell {branch} probabilities"
                )
            probabilities[branch] = np.ascontiguousarray(value)
    return probabilities


def restore_cached_native_crop(
    prediction_crop: np.ndarray,
    metadata: dict,
) -> np.ndarray:
    """Insert native cached crop predictions back into reoriented full image space."""
    native_shape = tuple(int(value) for value in metadata["native_cropped_shape_zyx"])
    cached_shape = tuple(int(value) for value in metadata["resampled_shape_zyx"])
    if cached_shape != native_shape:
        raise ValueError(
            "run_voxtell_val_inference.py only supports native cached inference. "
            f"Cached shape {cached_shape} differs from native cropped shape {native_shape}; "
            "use the 2 mm inference wrapper for resampled caches."
        )
    orig_shape = tuple(int(value) for value in metadata["original_reoriented_shape_zyx"])
    restored = np.zeros([prediction_crop.shape[0], *orig_shape], dtype=prediction_crop.dtype)
    return np.asarray(
        insert_crop_into_image(restored, prediction_crop, metadata["crop_bbox_zyx"]),
        dtype=prediction_crop.dtype,
    )


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
    probability_output_dir: Path | None = None,
    threshold: float = 0.5,
    preprocessed_cache_dir: Path | None = None,
    proposal_output_root: Path | None = None,
    proposal_thresholds: tuple[float, ...] = (0.1, 0.3, 0.5),
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
    probability_path = probability_output_dir / name if probability_output_dir is not None else None
    proposal_paths = {
        threshold_label(proposal_threshold): (
            proposal_output_root
            / threshold_label(proposal_threshold)
            / "predictions"
            / name
        )
        for proposal_threshold in proposal_thresholds
    } if proposal_output_root is not None else {}
    if (
        pred_path.exists()
        and (probability_path is None or probability_path.exists())
        and all(path.exists() for path in proposal_paths.values())
        and not overwrite
    ):
        result["status"] = "skipped_existing"
        return result
    if not ct_path.is_file():
        result["status"] = "missing_ct"
        return result
    if not gt_path.is_file():
        result["status"] = "missing_gt"
        return result

    probability_prediction = None
    proposal_probability_prediction = None
    cache_metadata = None
    if preprocessed_cache_dir is not None:
        from voxtell_preprocessed_cache import load_cached_case

        image, _targets, cache_metadata = load_cached_case(
            preprocessed_cache_dir,
            name,
            require_targets=False,
        )
        if cache_metadata.get("preprocess_id") != "crop_zscore_native_v1":
            raise ValueError(
                f"{name}: --preprocessed-cache-dir for this wrapper requires "
                f"crop_zscore_native_v1, got {cache_metadata.get('preprocess_id')!r}"
            )
        if proposal_output_root is not None:
            if not isinstance(predictor, DualBranchVoxTellPredictor):
                raise ValueError(
                    "--proposal-output-root requires a dual-branch model directory"
                )
            crop_probabilities = predict_preprocessed_crop_branch_probabilities(
                predictor,
                image,
                prompts,
            )
            probability_prediction = restore_cached_native_crop(
                crop_probabilities["final"],
                cache_metadata,
            )
            proposal_probability_prediction = restore_cached_native_crop(
                crop_probabilities["proposal"],
                cache_metadata,
            )
        else:
            crop_probability = predict_preprocessed_crop_probabilities(
                predictor,
                image,
                prompts,
            )
            probability_prediction = restore_cached_native_crop(
                crop_probability,
                cache_metadata,
            )
        ct_properties = cache_metadata["ct_properties"]
        if output_type == "mask":
            prediction = (probability_prediction >= threshold).astype(np.uint8, copy=False)
            output_dtype = np.uint8
        elif output_type == "probability":
            prediction = probability_prediction
            output_dtype = np.float32
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
    else:
        if proposal_output_root is not None:
            raise ValueError(
                "Proposal-output export currently requires --preprocessed-cache-dir"
            )
        image, ct_properties = reader.read_images([str(ct_path)])
        if output_type == "mask" and probability_output_dir is None:
            prediction = predictor.predict_single_image(image, prompts)
            output_dtype = np.uint8
        elif output_type == "mask":
            probability_prediction = predict_single_image_probabilities(predictor, image, prompts)
            prediction = (probability_prediction >= threshold).astype(np.uint8, copy=False)
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
    if probability_prediction is not None and probability_output_dir is not None:
        exported_probability, _ = export_prediction_to_gt_layout(
            probability_prediction,
            gt_img,
            ct_properties,
            name,
            output_dtype=np.float32,
        )
        probability_path = probability_output_dir / name
        save_4d_prediction(
            exported_probability,
            gt_path,
            probability_path,
            output_dtype=np.float32,
        )
        result["probability_path"] = str(probability_path)
        result["probability_range"] = [
            float(np.min(exported_probability)),
            float(np.max(exported_probability)),
        ]
    if proposal_probability_prediction is not None:
        exported_proposal_probability, _ = export_prediction_to_gt_layout(
            proposal_probability_prediction,
            gt_img,
            ct_properties,
            name,
            output_dtype=np.float32,
        )
        ground_truth = np.asanyarray(gt_img.dataobj)
        if ground_truth.ndim == 3:
            ground_truth = ground_truth[None]
        ground_truth = ground_truth > 0
        if ground_truth.shape != exported_proposal_probability.shape:
            raise ValueError(
                f"{name}: GT shape {ground_truth.shape} does not match exported "
                f"proposal shape {exported_proposal_probability.shape}"
            )
        proposal_metrics: dict[str, dict] = {}
        for proposal_threshold in proposal_thresholds:
            label = threshold_label(proposal_threshold)
            proposal_mask = (
                exported_proposal_probability >= proposal_threshold
            ).astype(np.uint8, copy=False)
            proposal_path = proposal_paths[label]
            save_4d_prediction(
                proposal_mask,
                gt_path,
                proposal_path,
                output_dtype=np.uint8,
            )
            finding_metrics = []
            for finding_index in range(proposal_mask.shape[0]):
                gt_mask = ground_truth[finding_index]
                pred_mask = proposal_mask[finding_index] > 0
                gt_voxels = int(gt_mask.sum())
                pred_voxels = int(pred_mask.sum())
                intersection = int(np.logical_and(gt_mask, pred_mask).sum())
                finding_metrics.append(
                    {
                        "finding_index": finding_index,
                        "gt_voxels": gt_voxels,
                        "proposal_voxels": pred_voxels,
                        "intersection_voxels": intersection,
                        "gt_voxel_coverage": (
                            intersection / gt_voxels if gt_voxels else None
                        ),
                        "proposal_to_gt_volume_ratio": (
                            pred_voxels / gt_voxels if gt_voxels else None
                        ),
                        "has_overlap": bool(intersection > 0),
                    }
                )
            proposal_metrics[label] = {
                "threshold": proposal_threshold,
                "prediction_path": str(proposal_path),
                "findings": finding_metrics,
            }
        result["proposal_probability_range"] = [
            float(np.min(exported_proposal_probability)),
            float(np.max(exported_proposal_probability)),
        ]
        result["proposal_metrics"] = proposal_metrics
    result["status"] = "written"
    result["shape"] = list(prediction.shape)
    result["orientation"] = export_metadata
    if cache_metadata is not None:
        result["preprocessed_cache"] = {
            "cache_root": str(preprocessed_cache_dir),
            "preprocess_id": cache_metadata.get("preprocess_id"),
            "image_sha256": cache_metadata.get("image_sha256"),
            "targets_sha256": cache_metadata.get("targets_sha256"),
        }
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
    parser.add_argument("--probability-output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--preprocessed-cache-dir", type=Path, default=None)
    parser.add_argument("--sliding-window-batch-size", type=int, default=1)
    parser.add_argument("--proposal-output-root", type=Path, default=None)
    parser.add_argument("--proposal-thresholds", default="0.1,0.3,0.5")
    parser.add_argument("--status-json", type=Path, default=None)
    args = parser.parse_args()
    proposal_thresholds = parse_thresholds(args.proposal_thresholds)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard < num_shards")
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be between 0 and 1")
    if args.output_type == "probability" and args.probability_output_dir is not None:
        raise ValueError("--probability-output-dir is only valid with --output-type mask")
    if args.sliding_window_batch_size < 1:
        raise ValueError("--sliding-window-batch-size must be >= 1")
    if args.proposal_output_root is not None and args.output_type != "mask":
        raise ValueError("--proposal-output-root requires --output-type mask")

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
        additional_output_dirs = []
        if args.probability_output_dir is not None:
            additional_output_dirs.append(args.probability_output_dir)
        if args.proposal_output_root is not None:
            for proposal_threshold in proposal_thresholds:
                additional_output_dirs.append(
                    args.proposal_output_root
                    / threshold_label(proposal_threshold)
                    / "predictions"
                )
        complete_outputs = _all_required_outputs_complete(
            eval_root,
            args.output_dir,
            args.split,
            entries,
            additional_output_dirs,
        )
        if complete_outputs:
            print(f"Completed eval already exists under {eval_root}; skipping inference.", flush=True)
            return 0
        lock_path = _acquire_eval_lock(
            eval_root=eval_root,
            output_dir=args.output_dir,
            split=args.split,
            entries=entries,
            additional_output_dirs=additional_output_dirs,
            poll_seconds=float(os.environ.get("EVAL_LOCK_POLL_SECONDS", "60")),
            stale_seconds=float(os.environ.get("EVAL_LOCK_STALE_SECONDS", str(DEFAULT_EVAL_LOCK_STALE_SECONDS))),
        )
        if lock_path is None and _all_required_outputs_complete(
            eval_root,
            args.output_dir,
            args.split,
            entries,
            additional_output_dirs,
        ):
            return 0

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    runtime_metadata = {
        "gpu": args.gpu,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else None,
        "sliding_window_batch_size": args.sliding_window_batch_size,
    }
    print(f"Runtime: {runtime_metadata}", flush=True)
    model_spec_record = load_model_spec_kind(args.model_dir)
    if model_spec_record is not None and model_spec_record[0] == "dual":
        predictor = DualBranchVoxTellPredictor(
            model_dir=args.model_dir,
            device=device,
            embedding_bank=str(args.embeddings) if args.embeddings else None,
            use_precomputed_embeddings=not args.no_precomputed,
            sliding_window_batch_size=args.sliding_window_batch_size,
        )
    elif model_spec_record is not None and model_spec_record[0] == "s3":
        if args.proposal_output_root is not None:
            raise ValueError("--proposal-output-root requires a dual-branch model directory")
        if args.sliding_window_batch_size != 1:
            raise ValueError("Sliding-window batching is currently implemented for dual-branch models only")
        predictor = S3AttentionVoxTellPredictor(
            model_dir=args.model_dir,
            device=device,
            embedding_bank=str(args.embeddings) if args.embeddings else None,
            use_precomputed_embeddings=not args.no_precomputed,
        )
    else:
        if args.proposal_output_root is not None:
            raise ValueError(
                "--proposal-output-root requires a model directory with model_spec.json"
            )
        if args.sliding_window_batch_size != 1:
            raise ValueError(
                "Sliding-window batching is currently implemented for dual-branch models only"
            )
        predictor = VoxTellPredictor(
            model_dir=str(args.model_dir) if args.model_dir else None,
            device=device,
            embedding_bank=str(args.embeddings) if args.embeddings else None,
            use_precomputed_embeddings=not args.no_precomputed,
        )
    reader = NibabelIOWithReorient()

    statuses = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.probability_output_dir is not None:
        args.probability_output_dir.mkdir(parents=True, exist_ok=True)
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
                    probability_output_dir=args.probability_output_dir,
                    threshold=args.threshold,
                    preprocessed_cache_dir=args.preprocessed_cache_dir,
                    proposal_output_root=args.proposal_output_root,
                    proposal_thresholds=proposal_thresholds,
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
