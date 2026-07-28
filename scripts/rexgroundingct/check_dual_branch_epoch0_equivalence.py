#!/usr/bin/env python3
"""Compare source, proposal, and final epoch-0 logits on one fixed patch."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from voxtell.inference.predictor import VoxTellPredictor

from common import sorted_prompts, write_json
from voxtell_dual_branch import DualBranchVoxTellPredictor
from voxtell_preprocessed_cache import load_cached_case


def centered_patch(image: np.ndarray, patch_size: tuple[int, int, int]) -> np.ndarray:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected image shape (1, Z, Y, X), got {image.shape}")
    pad_width = [(0, 0)]
    for size, required in zip(image.shape[1:], patch_size):
        total = max(0, required - int(size))
        pad_width.append((total // 2, total - total // 2))
    padded = np.pad(image, pad_width, mode="constant", constant_values=0)
    starts = [
        max(0, (int(size) - required) // 2)
        for size, required in zip(padded.shape[1:], patch_size)
    ]
    slices = tuple(
        slice(start, start + required)
        for start, required in zip(starts, patch_size)
    )
    patch = np.ascontiguousarray(padded[(slice(None), *slices)], dtype=np.float32)
    if tuple(patch.shape[1:]) != patch_size:
        raise RuntimeError(f"Unexpected patch shape: {patch.shape}")
    return patch


def compare_outputs(
    reference: list[torch.Tensor],
    candidate: list[torch.Tensor],
) -> list[dict]:
    if len(reference) != len(candidate):
        raise ValueError(
            f"Deep-supervision output count differs: {len(reference)} != "
            f"{len(candidate)}"
        )
    comparisons = []
    for scale, (expected, actual) in enumerate(zip(reference, candidate)):
        if expected.shape != actual.shape:
            raise ValueError(
                f"Scale {scale} shape differs: {expected.shape} != {actual.shape}"
            )
        expected_float = expected.float()
        actual_float = actual.float()
        difference = (actual_float - expected_float).abs()
        comparisons.append(
            {
                "scale": scale,
                "shape": list(expected.shape),
                "max_abs_logit_difference": float(difference.max()),
                "mean_abs_logit_difference": float(difference.mean()),
                "logits_bitwise_equal": bool(torch.equal(expected, actual)),
                "threshold_mask_agreement": float(
                    torch.mean(
                        (
                            (expected_float >= 0.0)
                            == (actual_float >= 0.0)
                        ).float()
                    )
                ),
            }
        )
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--dual-model-dir", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(192, 192, 192))
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    entries = json.loads(args.dataset_json.read_text()).get("test", [])
    entry = next(
        (candidate for candidate in entries if candidate["name"] == args.case_name),
        None,
    )
    if entry is None:
        raise ValueError(f"{args.case_name!r} is absent from {args.dataset_json}")
    image, _targets, _metadata = load_cached_case(
        args.cache_root,
        args.case_name,
        require_targets=False,
    )
    patch = centered_patch(image, tuple(args.patch_size))
    prompts = sorted_prompts(entry)
    if not prompts:
        raise ValueError(f"{args.case_name}: no prompts")
    device = torch.device(f"cuda:{args.gpu}")
    image_tensor = torch.from_numpy(patch[None]).to(device)

    source_predictor = VoxTellPredictor(
        model_dir=str(args.source_model_dir),
        device=device,
        embedding_bank=str(args.embeddings),
        use_precomputed_embeddings=False,
    )
    text_embedding = source_predictor.embed_text_prompts(prompts[:1])
    source_predictor.network.to(device)
    source_predictor.network.deep_supervision = True
    source_predictor.network.decoder.deep_supervision = True
    source_predictor.network.eval()
    with torch.inference_mode(), torch.autocast("cuda", enabled=True):
        source_first = [
            output.detach().cpu()
            for output in source_predictor.network(image_tensor, text_embedding)
        ]
        source_second = [
            output.detach().cpu()
            for output in source_predictor.network(image_tensor, text_embedding)
        ]

    source_predictor.network.to("cpu")
    del source_predictor
    torch.cuda.empty_cache()
    gc.collect()

    dual_predictor = DualBranchVoxTellPredictor(
        model_dir=args.dual_model_dir,
        device=device,
        embedding_bank=str(args.embeddings),
        use_precomputed_embeddings=False,
        sliding_window_batch_size=1,
    )
    dual_predictor.network.to(device)
    dual_predictor.network.deep_supervision = True
    dual_predictor.network.eval()
    with torch.inference_mode(), torch.autocast("cuda", enabled=True):
        branches = dual_predictor.network(
            image_tensor,
            text_embedding,
            return_branches=True,
        )
    proposal = [output.detach().cpu() for output in branches["proposal"]]
    final = [output.detach().cpu() for output in branches["final"]]

    report = {
        "source_model_dir": str(args.source_model_dir),
        "dual_model_dir": str(args.dual_model_dir),
        "dataset_json": str(args.dataset_json),
        "cache_root": str(args.cache_root),
        "case_name": args.case_name,
        "prompt": prompts[0],
        "patch_size": list(args.patch_size),
        "source_repeatability": compare_outputs(source_first, source_second),
        "source_vs_proposal": compare_outputs(source_first, proposal),
        "source_vs_final": compare_outputs(source_first, final),
        "proposal_vs_final": compare_outputs(proposal, final),
    }
    write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
