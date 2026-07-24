#!/usr/bin/env python3
"""Run exp005 global proposals followed by local native-resolution VoxTell."""

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

from common import CT_ROOT, REX_METADATA, REX_SEG_DIR, load_split_entries, sorted_prompts, write_json
from global_proposal import GLOBAL_SHAPE, GlobalProposalUNet, extract_candidate_starts, load_global_case
from run_voxtell_val_inference import export_prediction_to_gt_layout, load_dataset_json_entries, save_4d_prediction
from train_global_proposal import load_embeddings
from train_text_conditioned_voxtell import RexVoxTellPatchSampler, build_negative_prompt_pool


def extract_patch(array: np.ndarray, starts: list[int], patch_size: int = 192) -> np.ndarray:
    output = np.zeros((array.shape[0], patch_size, patch_size, patch_size), dtype=array.dtype)
    source_slices = []
    target_slices = []
    for axis, start in enumerate(starts):
        stop = min(start + patch_size, array.shape[axis + 1])
        source_slices.append(slice(start, stop))
        target_slices.append(slice(0, stop - start))
    output[(slice(None), *target_slices)] = array[(slice(None), *source_slices)]
    return output


def insert_patch_max(
    output: np.ndarray,
    patch: np.ndarray,
    starts: list[int],
) -> None:
    source_slices = []
    target_slices = []
    for axis, start in enumerate(starts):
        stop = min(start + patch.shape[axis], output.shape[axis])
        target_slices.append(slice(start, stop))
        source_slices.append(slice(0, stop - start))
    destination = output[tuple(target_slices)]
    np.maximum(destination, patch[tuple(source_slices)], out=destination)


def insert_native_crop(
    cropped_fzyx: np.ndarray,
    original_shape_zyx: list[int],
    bbox_zyx: list[list[int]],
) -> np.ndarray:
    output = np.zeros((cropped_fzyx.shape[0], *original_shape_zyx), dtype=cropped_fzyx.dtype)
    slices = tuple(slice(int(low), int(high)) for low, high in bbox_zyx)
    output[(slice(None), *slices)] = cropped_fzyx
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-model-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probability-output-dir", type=Path, default=None)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--nms-radius", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    entries = load_dataset_json_entries(args.dataset_json, "val")
    all_train = load_split_entries(args.metadata, ["train"])
    entries_by_name = {entry["name"]: entry for entry in load_split_entries(args.metadata, ["val"])}
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.proposal_checkpoint, map_location="cpu", weights_only=False)
    base_channels = int(checkpoint.get("model", {}).get("base_channels", 8))
    proposal_model = GlobalProposalUNet(base_channels=base_channels).to(device)
    proposal_model.load_state_dict(checkpoint["network_weights"])
    proposal_model.eval()
    embedding_bank = load_embeddings(args.embeddings)

    stage2 = VoxTellPredictor(
        model_dir=str(args.stage2_model_dir),
        device=device,
        embedding_bank=str(args.embeddings),
        use_precomputed_embeddings=False,
    )
    stage2.network = stage2.network.to(device).eval()
    native_sampler = RexVoxTellPatchSampler(
        entries=list(entries_by_name.values()),
        negative_prompt_pool=build_negative_prompt_pool(all_train),
        ct_root=args.ct_root,
        seg_dir=args.seg_dir,
        preprocessed_cache_dir=None,
        patch_size=(192, 192, 192),
        foreground_oversample_prob=0.85,
        seed=20260723,
        cache_size=1,
        require_positive_crop=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.probability_output_dir is not None:
        args.probability_output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for entry in tqdm(entries, desc="global-to-local cascade"):
        name = entry["name"]
        output_path = args.output_dir / name
        probability_path = (
            args.probability_output_dir / name if args.probability_output_dir is not None else None
        )
        if output_path.is_file() and (probability_path is None or probability_path.is_file()) and not args.overwrite:
            records.append({"name": name, "status": "skipped_existing"})
            continue
        try:
            source_entry = entries_by_name[name]
            image_global, _targets, metadata = load_global_case(args.global_cache, name)
            native_case = native_sampler.load_case(source_entry)
            prompts = sorted_prompts(entry)
            embeddings_np = np.stack([embedding_bank[prompt.lower()] for prompt in prompts])
            embeddings = torch.from_numpy(embeddings_np)[None].to(device)
            image_tensor = torch.from_numpy(np.asarray(image_global, dtype=np.float32))[None].to(device)
            with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
                proposal_logits = proposal_model(image_tensor, embeddings)
            proposal_prob = torch.sigmoid(proposal_logits.float()).cpu().numpy()[0]

            case_probabilities = np.zeros(
                (len(prompts), *native_case.image.shape[1:]),
                dtype=np.float32,
            )
            finding_records = []
            for index, prompt in enumerate(prompts):
                candidates = extract_candidate_starts(
                    proposal_prob[index],
                    metadata,
                    top_k=args.top_k,
                    nms_radius=args.nms_radius,
                )
                for candidate in candidates:
                    starts = candidate["native_patch_starts_zyx"]
                    patch = extract_patch(native_case.image, starts)
                    patch_tensor = torch.from_numpy(patch)[None].to(device)
                    prompt_embedding = embeddings[:, index : index + 1]
                    with torch.inference_mode(), torch.autocast(
                        device.type, enabled=device.type == "cuda"
                    ):
                        logits = stage2.network(patch_tensor, prompt_embedding)
                        if isinstance(logits, list):
                            logits = logits[0]
                        probability_patch = torch.sigmoid(logits[0, 0].float()).cpu().numpy()
                    insert_patch_max(case_probabilities[index], probability_patch, starts)
                finding_records.append(
                    {
                        "finding_index": index,
                        "prompt": prompt,
                        "candidates": candidates,
                    }
                )

            full_reoriented = insert_native_crop(
                case_probabilities,
                metadata["native_original_reoriented_shape_zyx"],
                metadata["native_crop_bbox_zyx"],
            )
            gt_path = args.seg_dir / name
            gt_img = nib.load(str(gt_path))
            ct_properties = {
                "nibabel_stuff": {
                    key: np.asarray(value, dtype=np.float64)
                    for key, value in metadata["ct_properties"]["nibabel_stuff"].items()
                }
            }
            exported_probability, orientation = export_prediction_to_gt_layout(
                full_reoriented,
                gt_img,
                ct_properties,
                name,
                output_dtype=np.float32,
            )
            mask = (exported_probability >= args.threshold).astype(np.uint8, copy=False)
            save_4d_prediction(mask, gt_path, output_path, output_dtype=np.uint8)
            if probability_path is not None:
                save_4d_prediction(
                    exported_probability,
                    gt_path,
                    probability_path,
                    output_dtype=np.float32,
                )
            records.append(
                {
                    "name": name,
                    "status": "written",
                    "findings": finding_records,
                    "orientation": orientation,
                }
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "name": name,
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"ERROR {name}: {exc}", flush=True)
        write_json(args.candidate_json, {"cases": records})
    failures = [record for record in records if record["status"] == "error"]
    print(json.dumps({"cases": len(records), "failures": len(failures)}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
