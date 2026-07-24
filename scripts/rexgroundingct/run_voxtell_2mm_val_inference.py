#!/usr/bin/env python3
"""Run VoxTell on cached 2 mm volumes and restore predictions to native geometry."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
from voxtell.inference.predictor import VoxTellPredictor

from common import REX_SEG_DIR, sorted_prompts, write_json
from run_voxtell_val_inference import (
    export_prediction_to_gt_layout,
    load_dataset_json_entries,
    save_4d_prediction,
)
from voxtell_2mm import load_cached_case, restore_2mm_probabilities_to_native_crop


def insert_native_crop(
    cropped_fzyx: np.ndarray,
    original_shape_zyx: tuple[int, int, int],
    bbox_zyx: list[list[int]],
) -> np.ndarray:
    output = np.zeros((cropped_fzyx.shape[0], *original_shape_zyx), dtype=cropped_fzyx.dtype)
    slices = tuple(slice(int(start), int(stop)) for start, stop in bbox_zyx)
    output[(slice(None), *slices)] = cropped_fzyx
    return output


@torch.inference_mode()
def predict_cached_probabilities(
    predictor: VoxTellPredictor,
    image_czyx: np.ndarray,
    prompts: list[str],
) -> np.ndarray:
    embeddings = predictor.embed_text_prompts(prompts)
    logits = predictor.predict_sliding_window_return_logits(
        torch.from_numpy(np.ascontiguousarray(image_czyx)),
        embeddings,
    ).to("cpu")
    probabilities = torch.sigmoid(logits.float()).numpy().astype(np.float32, copy=False)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Encountered non-finite 2 mm probabilities")
    return probabilities


def run_case(
    predictor: VoxTellPredictor,
    entry: dict,
    cache_root: Path,
    seg_dir: Path,
    output_dir: Path,
    probability_output_dir: Path | None,
    threshold: float,
    overwrite: bool,
) -> dict:
    name = entry["name"]
    gt_path = seg_dir / name
    pred_path = output_dir / name
    probability_path = probability_output_dir / name if probability_output_dir is not None else None
    if pred_path.is_file() and (probability_path is None or probability_path.is_file()) and not overwrite:
        return {"name": name, "status": "skipped_existing"}

    image, _targets, metadata = load_cached_case(cache_root, name)
    probabilities_2mm = predict_cached_probabilities(predictor, image, sorted_prompts(entry))
    native_crop = restore_2mm_probabilities_to_native_crop(
        probabilities_2mm,
        tuple(int(value) for value in metadata["native_cropped_shape_zyx"]),
    )
    native_reoriented = insert_native_crop(
        native_crop,
        tuple(int(value) for value in metadata["original_reoriented_shape_zyx"]),
        metadata["crop_bbox_zyx"],
    )
    gt_img = nib.load(str(gt_path))
    ct_properties = {
        "nibabel_stuff": {
            key: np.asarray(value, dtype=np.float64)
            for key, value in metadata["ct_properties"]["nibabel_stuff"].items()
        }
    }
    exported_probability, orientation = export_prediction_to_gt_layout(
        native_reoriented,
        gt_img,
        ct_properties,
        name,
        output_dtype=np.float32,
    )
    exported_mask = (exported_probability >= threshold).astype(np.uint8, copy=False)
    save_4d_prediction(exported_mask, gt_path, pred_path, output_dtype=np.uint8)
    if probability_path is not None:
        save_4d_prediction(exported_probability, gt_path, probability_path, output_dtype=np.float32)
    return {
        "name": name,
        "status": "written",
        "prediction": str(pred_path),
        "probability": str(probability_path) if probability_path is not None else None,
        "threshold": threshold,
        "probability_range": [
            float(np.min(exported_probability)),
            float(np.max(exported_probability)),
        ],
        "resampled_shape_zyx": metadata["resampled_shape_zyx"],
        "native_cropped_shape_zyx": metadata["native_cropped_shape_zyx"],
        "orientation": orientation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probability-output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--status-json", type=Path, default=None)
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard < num-shards")
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be between 0 and 1")
    entries = load_dataset_json_entries(args.dataset_json, "val")
    entries = [
        entry for index, entry in enumerate(entries)
        if index % args.num_shards == args.shard_index
    ]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    predictor = VoxTellPredictor(
        model_dir=str(args.model_dir),
        device=device,
        embedding_bank=str(args.embeddings) if args.embeddings else None,
        use_precomputed_embeddings=not bool(args.embeddings),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.probability_output_dir is not None:
        args.probability_output_dir.mkdir(parents=True, exist_ok=True)

    statuses = []
    for entry in tqdm(entries, desc=f"2mm shard {args.shard_index}/{args.num_shards}"):
        try:
            statuses.append(
                run_case(
                    predictor=predictor,
                    entry=entry,
                    cache_root=args.cache_root,
                    seg_dir=args.seg_dir,
                    output_dir=args.output_dir,
                    probability_output_dir=args.probability_output_dir,
                    threshold=args.threshold,
                    overwrite=args.overwrite,
                )
            )
        except Exception as exc:  # noqa: BLE001
            statuses.append(
                {
                    "name": entry.get("name"),
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"ERROR {entry.get('name')}: {exc}", flush=True)
        if args.status_json is not None:
            write_json(args.status_json, {"cases": statuses})
    if args.status_json is not None:
        write_json(args.status_json, {"cases": statuses})
    failures = [item for item in statuses if item["status"] == "error"]
    print(json.dumps({"processed": len(statuses), "failures": len(failures)}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
