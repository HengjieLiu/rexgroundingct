#!/usr/bin/env python3
"""Run VoxTell with resampled source-window multiscale inference."""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from tqdm import tqdm
from voxtell.inference.predictor import VoxTellPredictor

from common import CT_ROOT, REX_SEG_DIR, ct_rate_abs_path, sorted_prompts, write_json
from run_voxtell_val_inference import export_prediction_to_gt_layout, load_dataset_json_entries, save_4d_prediction


MODEL_PATCH_SIZE = (192, 192, 192)


def _runtime_metadata(gpu: int, device: torch.device, scale: int) -> dict:
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "command": sys.argv,
        "gpu": gpu,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(gpu) if torch.cuda.is_available() else None,
        "source_window_size": int(scale),
        "model_patch_size": list(MODEL_PATCH_SIZE),
        "interpolation": "trilinear",
        "align_corners": False,
        "antialias": False,
        "normalization_scope": "full crop-to-nonzero image before multiscale window sampling",
    }


def _window_slicers(spatial_shape: tuple[int, int, int], source_window_size: int, step_size: float) -> list[tuple]:
    patch_size = (int(source_window_size),) * 3
    steps = compute_steps_for_sliding_window(spatial_shape, patch_size, step_size)
    return [
        tuple([slice(None), slice(z, z + patch_size[0]), slice(y, y + patch_size[1]), slice(x, x + patch_size[2])])
        for z in steps[0]
        for y in steps[1]
        for x in steps[2]
    ]


def _torch_size(source_window_size: int) -> tuple[int, int, int]:
    value = int(source_window_size)
    return (value, value, value)


@torch.inference_mode()
def predict_multiscale_probabilities(
    predictor: VoxTellPredictor,
    image: np.ndarray,
    prompts: list[str],
    source_window_size: int,
    tile_step_size: float,
    device: torch.device,
    aggregate_dtype: torch.dtype,
) -> tuple[np.ndarray, dict]:
    """Resize source windows to 192^3, run VoxTell, and blend in source space."""
    text_embeddings = predictor.embed_text_prompts(prompts)
    data, bbox, orig_shape = predictor.preprocess(image)
    source_patch = _torch_size(source_window_size)
    padded, slicer_revert_padding = pad_nd_image(data, source_patch, "constant", {"value": 0}, True, None)
    slicers = _window_slicers(tuple(int(v) for v in padded.shape[1:]), source_window_size, tile_step_size)

    logits_sum = torch.zeros((text_embeddings.shape[1], *padded.shape[1:]), dtype=aggregate_dtype, device="cpu")
    weight_sum = torch.zeros(tuple(padded.shape[1:]), dtype=aggregate_dtype, device="cpu")
    gaussian = compute_gaussian(
        source_patch,
        sigma_scale=1.0 / 8,
        value_scaling_factor=10,
        device=torch.device("cpu"),
    ).to(dtype=aggregate_dtype)

    predictor.network = predictor.network.to(device)
    empty_cache(device)
    start = time.perf_counter()
    autocast_context = torch.autocast(device.type, enabled=True) if device.type == "cuda" else dummy_context()
    with autocast_context:
        for tile_slice in tqdm(slicers, desc=f"source{source_window_size}", leave=False):
            source_patch_tensor = torch.clone(padded[tile_slice][None], memory_format=torch.contiguous_format).to(device)
            model_patch = F.interpolate(
                source_patch_tensor.float(),
                size=MODEL_PATCH_SIZE,
                mode="trilinear",
                align_corners=False,
            )
            model_logits = predictor.network(model_patch, text_embeddings)[0].float()
            source_logits = F.interpolate(
                model_logits[None],
                size=source_patch,
                mode="trilinear",
                align_corners=False,
            )[0].to("cpu", dtype=aggregate_dtype)
            logits_sum[tile_slice] += source_logits * gaussian
            weight_sum[tile_slice[1:]] += gaussian
            del source_patch_tensor, model_patch, model_logits, source_logits

    logits_sum = logits_sum[(slice(None), *slicer_revert_padding[1:])]
    weight_sum = weight_sum[slicer_revert_padding[1:]]
    logits = logits_sum / torch.clamp(weight_sum[None], min=torch.finfo(aggregate_dtype).eps)
    probabilities = torch.sigmoid(logits.float()).numpy().astype(np.float32, copy=False)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Encountered non-finite multiscale probabilities")

    reverted = np.zeros([probabilities.shape[0], *orig_shape], dtype=np.float32)
    reverted = insert_crop_into_image(reverted, probabilities, bbox)
    reverted = np.asarray(reverted, dtype=np.float32)
    if not np.isfinite(reverted).all():
        raise RuntimeError("Encountered non-finite uncropped probabilities")

    elapsed = time.perf_counter() - start
    metadata = {
        "num_prompts": len(prompts),
        "text_embeddings_shape": list(text_embeddings.shape),
        "preprocessed_shape": list(data.shape),
        "padded_shape": list(padded.shape),
        "source_window_size": int(source_window_size),
        "source_patch_size": list(source_patch),
        "model_patch_size": list(MODEL_PATCH_SIZE),
        "tile_step_size": float(tile_step_size),
        "tile_count": len(slicers),
        "bbox": [[int(v) for v in axis] for axis in bbox],
        "orig_shape": [int(v) for v in orig_shape],
        "inference_seconds": elapsed,
        "probability_range": [float(np.min(reverted)), float(np.max(reverted))],
    }
    return reverted, metadata


def run_case(
    predictor: VoxTellPredictor,
    reader: NibabelIOWithReorient,
    entry: dict,
    ct_root: Path,
    seg_dir: Path,
    output_dir: Path,
    overwrite: bool,
    runtime_metadata: dict,
    source_window_size: int,
    tile_step_size: float,
    device: torch.device,
    aggregate_dtype: torch.dtype,
) -> dict:
    name = entry["name"]
    ct_path = ct_rate_abs_path(name, ct_root)
    gt_path = seg_dir / name
    pred_path = output_dir / name
    result = {
        "name": name,
        "ct_path": str(ct_path),
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "num_prompts": len(sorted_prompts(entry)),
        "split": entry.get("_split"),
        "split_index": entry.get("_index"),
        "runtime": runtime_metadata,
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
    probabilities, predict_metadata = predict_multiscale_probabilities(
        predictor=predictor,
        image=image,
        prompts=sorted_prompts(entry),
        source_window_size=source_window_size,
        tile_step_size=tile_step_size,
        device=device,
        aggregate_dtype=aggregate_dtype,
    )
    gt_img = nib.load(str(gt_path))
    exported, export_metadata = export_prediction_to_gt_layout(
        probabilities,
        gt_img,
        ct_properties,
        name,
        output_dtype=np.float32,
    )
    save_4d_prediction(exported, gt_path, pred_path, output_dtype=np.float32)
    result.update(
        {
            "status": "written",
            "shape": list(exported.shape),
            "multiscale": predict_metadata,
            "orientation": export_metadata,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-window-size", type=int, required=True)
    parser.add_argument("--tile-step-size", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--status-json", type=Path, default=None)
    args = parser.parse_args()

    if args.source_window_size <= 0:
        raise ValueError("--source-window-size must be positive")
    if args.source_window_size == MODEL_PATCH_SIZE[0]:
        raise ValueError("Use run_voxtell_val_inference.py for native 192 probability inference")
    if not (0 < args.tile_step_size <= 1):
        raise ValueError("--tile-step-size must satisfy 0 < value <= 1")

    entries = load_dataset_json_entries(args.dataset_json, "val")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    aggregate_dtype = torch.float32
    runtime_metadata = _runtime_metadata(args.gpu, device, args.source_window_size)
    print(f"Runtime: {runtime_metadata}", flush=True)
    predictor = VoxTellPredictor(
        model_dir=str(args.model_dir),
        device=device,
        embedding_bank=str(args.embeddings) if args.embeddings else None,
        use_precomputed_embeddings=not bool(args.embeddings),
    )
    reader = NibabelIOWithReorient()

    statuses = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in tqdm(entries, desc=f"multiscale source {args.source_window_size}"):
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
                    source_window_size=args.source_window_size,
                    tile_step_size=args.tile_step_size,
                    device=device,
                    aggregate_dtype=aggregate_dtype,
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
        if args.status_json:
            write_json(args.status_json, {"cases": statuses})

    if args.status_json:
        write_json(args.status_json, {"cases": statuses})
    failures = [status for status in statuses if status["status"] in {"missing_ct", "missing_gt", "error"}]
    print(f"Processed {len(statuses)} cases; failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
