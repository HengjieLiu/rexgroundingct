#!/usr/bin/env python3
"""Run VoxTell on a ReXGroundingCT split and save one 4D prediction per case."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from tqdm import tqdm
from voxtell.inference.predictor import VoxTellPredictor

from common import CT_ROOT, REX_METADATA, REX_SEG_DIR, ct_rate_abs_path, load_split_entries, sorted_prompts, write_json


def _serializable_ornt(ornt: np.ndarray) -> list[list[int | float]]:
    return [[int(axis), float(direction)] for axis, direction in ornt.tolist()]


def export_prediction_to_gt_layout(
    prediction: np.ndarray,
    gt_img: nib.Nifti1Image,
    ct_properties: dict,
    name: str,
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
    exported = np.ascontiguousarray(exported.astype(np.uint8, copy=False))

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


def save_4d_prediction(segmentation: np.ndarray, gt_path: Path, output_path: Path) -> None:
    """Save prediction as evaluator-compatible 4D uint8 NIfTI using GT affine/header."""
    gt_img = nib.load(str(gt_path))
    header = gt_img.header.copy()
    header.set_data_dtype(np.uint8)
    out_img = nib.Nifti1Image(segmentation.astype(np.uint8, copy=False), gt_img.affine, header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))


def run_case(
    predictor: VoxTellPredictor,
    reader: NibabelIOWithReorient,
    entry: dict,
    ct_root: Path,
    seg_dir: Path,
    output_dir: Path,
    overwrite: bool,
    runtime_metadata: dict,
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
    prediction = predictor.predict_single_image(image, prompts)
    gt_img = nib.load(str(gt_path))
    prediction, export_metadata = export_prediction_to_gt_layout(prediction, gt_img, ct_properties, name)
    save_4d_prediction(prediction, gt_path, pred_path)
    result["status"] = "written"
    result["shape"] = list(prediction.shape)
    result["orientation"] = export_metadata
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
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
    parser.add_argument("--status-json", type=Path, default=None)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard < num_shards")

    entries = load_split_entries(args.metadata, [args.split])
    selected_by_case = args.case_index is not None or args.case_name is not None
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
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
