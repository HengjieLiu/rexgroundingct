#!/usr/bin/env python3
"""Benchmark intra-case sliding-window batches for a dual-branch VoxTell model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import sorted_prompts, write_json
from voxtell_dual_branch import DualBranchVoxTellPredictor
from voxtell_preprocessed_cache import load_cached_case


def parse_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("Batch sizes must be positive integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--case-name", action="append", required=True)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-reserved-gib", type=float, default=42.0)
    parser.add_argument("--minimum-speedup", type=float, default=0.1)
    parser.add_argument("--minimum-mask-agreement", type=float, default=0.99999)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    batch_sizes = parse_ints(args.batch_sizes)
    if 1 not in batch_sizes:
        raise ValueError("Batch size 1 is required as the benchmark reference")
    dataset = json.loads(args.dataset_json.read_text()).get("test", [])
    by_name = {entry["name"]: entry for entry in dataset}
    missing = [name for name in args.case_name if name not in by_name]
    if missing:
        raise ValueError(f"Benchmark cases are absent from dataset JSON: {missing}")

    device = torch.device(f"cuda:{args.gpu}")
    predictor = DualBranchVoxTellPredictor(
        model_dir=args.model_dir,
        device=device,
        embedding_bank=str(args.embeddings),
        use_precomputed_embeddings=False,
        sliding_window_batch_size=1,
    )
    results = {
        batch_size: {
            "batch_size": batch_size,
            "status": "ok",
            "total_seconds": 0.0,
            "max_reserved_gib": 0.0,
            "case_results": [],
        }
        for batch_size in batch_sizes
    }

    for name in args.case_name:
        entry = by_name[name]
        image, _targets, _metadata = load_cached_case(
            args.cache_root,
            name,
            require_targets=False,
        )
        text_embeddings = predictor.embed_text_prompts(sorted_prompts(entry))
        data = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        baseline_masks = None
        for batch_size in batch_sizes:
            predictor.sliding_window_batch_size = batch_size
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            try:
                outputs = predictor.predict_sliding_window_return_branch_logits(
                    data,
                    text_embeddings,
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                masks = {
                    branch: (torch.sigmoid(logits.float()) >= 0.5).cpu().numpy()
                    for branch, logits in outputs.items()
                }
                if baseline_masks is None:
                    if batch_size != 1:
                        raise RuntimeError("Batch size 1 must be benchmarked first")
                    baseline_masks = masks
                    agreement = 1.0
                else:
                    agreements = []
                    for branch in ("proposal", "final"):
                        agreements.append(
                            float(np.mean(masks[branch] == baseline_masks[branch]))
                        )
                    agreement = min(agreements)
                reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
                case_result = {
                    "name": name,
                    "seconds": elapsed,
                    "max_reserved_gib": reserved_gib,
                    "minimum_branch_mask_agreement": agreement,
                    "finite": all(
                        bool(torch.isfinite(logits).all().item())
                        for logits in outputs.values()
                    ),
                }
                result = results[batch_size]
                result["total_seconds"] += elapsed
                result["max_reserved_gib"] = max(
                    result["max_reserved_gib"],
                    reserved_gib,
                )
                result["case_results"].append(case_result)
                del outputs, masks
            except torch.cuda.OutOfMemoryError as exc:
                results[batch_size]["status"] = "oom"
                results[batch_size]["error"] = str(exc)
                torch.cuda.empty_cache()
                break

    baseline_seconds = results[1]["total_seconds"]
    passing = []
    for batch_size in batch_sizes:
        result = results[batch_size]
        result["speedup_fraction_vs_batch1"] = (
            1.0 - result["total_seconds"] / baseline_seconds
            if result["status"] == "ok" and baseline_seconds > 0
            else None
        )
        result["minimum_mask_agreement"] = min(
            (
                case["minimum_branch_mask_agreement"]
                for case in result["case_results"]
            ),
            default=0.0,
        )
        result["passes"] = bool(
            result["status"] == "ok"
            and result["max_reserved_gib"] <= args.max_reserved_gib
            and result["minimum_mask_agreement"] >= args.minimum_mask_agreement
            and (
                batch_size == 1
                or result["speedup_fraction_vs_batch1"] >= args.minimum_speedup
            )
            and all(case["finite"] for case in result["case_results"])
            and len(result["case_results"]) == len(args.case_name)
        )
        if result["passes"]:
            passing.append(batch_size)

    selected = max(passing) if passing else 1
    report = {
        "model_dir": str(args.model_dir),
        "dataset_json": str(args.dataset_json),
        "cache_root": str(args.cache_root),
        "case_names": args.case_name,
        "criteria": {
            "max_reserved_gib": args.max_reserved_gib,
            "minimum_speedup": args.minimum_speedup,
            "minimum_mask_agreement": args.minimum_mask_agreement,
        },
        "results": [results[batch_size] for batch_size in batch_sizes],
        "selected_sliding_window_batch_size": selected,
    }
    write_json(args.output_json, report)
    lines = [
        "# Dual-Branch Inference Batch Benchmark",
        "",
        "| Window batch | Status | Seconds | Speedup | Reserved GiB | Agreement | Pass |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report["results"]:
        speedup = result["speedup_fraction_vs_batch1"]
        lines.append(
            f"| {result['batch_size']} | {result['status']} | "
            f"{result['total_seconds']:.2f} | "
            f"{speedup:.3f} | " if speedup is not None else
            f"| {result['batch_size']} | {result['status']} | "
            f"{result['total_seconds']:.2f} | n/a | "
        )
        prefix = lines.pop()
        lines.append(
            prefix
            + f"{result['max_reserved_gib']:.2f} | "
            + f"{result['minimum_mask_agreement']:.6f} | "
            + f"{result['passes']} |"
        )
    lines.extend(
        [
            "",
            f"Selected batch size: `{selected}`",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
