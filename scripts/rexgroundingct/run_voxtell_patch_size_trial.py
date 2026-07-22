#!/usr/bin/env python3
"""Trial VoxTell inference with alternate patch sizes on one ReXGroundingCT case."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from einops import rearrange
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from positional_encodings.torch_encodings import PositionalEncoding3D
from voxtell.inference.predictor import VoxTellPredictor, download_voxtell_model

from common import CT_ROOT, EXP_ROOT, REX_DATA_ROOT, REX_METADATA, REX_SEG_DIR
from common import ct_rate_abs_path, load_split_entries, sorted_prompts, utc_now_iso, write_json
from run_voxtell_val_inference import export_prediction_to_gt_layout, save_4d_prediction


DEFAULT_EXP_DIR = EXP_ROOT / "trial_20260722_voxtell_patch_size_val_idx000"
CANONICAL_EXP001_EVAL = EXP_ROOT / "001_voxtell_v1_1_miccai200_val_eval" / "eval" / "val_quick_global_eval.json"


def patch_label(patch_size: int) -> str:
    return f"patch{patch_size:03d}"


def resolve_model_dir(model_dir: Path | None) -> Path:
    if model_dir is not None:
        return model_dir
    env_model = os.environ.get("VOXTELL_MODEL")
    if env_model:
        return Path(env_model)
    return Path(download_voxtell_model())


def feature_grid_for_patch(patch_size: list[int], decoder_layer: int) -> list[int]:
    reference_patch = [192, 192, 192]
    reference_grid = {
        0: [192, 192, 192],
        1: [96, 96, 96],
        2: [48, 48, 48],
        3: [24, 24, 24],
        4: [12, 12, 12],
        5: [6, 6, 6],
    }[decoder_layer]
    factors = [patch // grid for patch, grid in zip(reference_patch, reference_grid)]
    grid: list[int] = []
    for size, factor in zip(patch_size, factors):
        if size % factor:
            raise ValueError(f"Patch size {patch_size} is not divisible by decoder-layer factors {factors}")
        grid.append(size // factor)
    return grid


def set_trial_patch_size(predictor: VoxTellPredictor, patch_size: int) -> dict[str, Any]:
    patch = [int(patch_size)] * 3
    network = predictor.network
    decoder_layer = int(network.selected_decoder_layer)
    feature_grid = feature_grid_for_patch(patch, decoder_layer)
    h, w, d = feature_grid
    device = network.pos_embed.device
    dtype = network.pos_embed.dtype
    pos_embed = PositionalEncoding3D(network.query_dim)(
        torch.zeros(1, h, w, d, network.query_dim, dtype=dtype)
    )
    pos_embed = rearrange(pos_embed, "b h w d c -> (h w d) b c")
    network.pos_embed = pos_embed.to(device=device, dtype=dtype)
    predictor.patch_size = patch
    return {
        "patch_size": patch,
        "decoder_layer": decoder_layer,
        "feature_grid_shape": feature_grid,
        "pos_embed_shape": list(network.pos_embed.shape),
    }


def prediction_from_preprocessed(
    predictor: VoxTellPredictor,
    image: np.ndarray,
    prompts: list[str],
    aggregate_on_gpu: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    text_embeddings = predictor.embed_text_prompts(prompts)
    data, bbox, orig_shape = predictor.preprocess(image)
    padded, _ = pad_nd_image(data, predictor.patch_size, "constant", {"value": 0}, True, None)
    tile_count = len(predictor._internal_get_sliding_window_slicers(padded.shape[1:]))

    predictor.perform_everything_on_device = aggregate_on_gpu
    prediction = predictor.predict_sliding_window_return_logits(data, text_embeddings).to("cpu")
    with torch.no_grad():
        prediction = (torch.sigmoid(prediction.float()) > 0.5).numpy().astype(np.uint8, copy=False)

    segmentation = np.zeros([prediction.shape[0], *orig_shape], dtype=np.uint8)
    segmentation = insert_crop_into_image(segmentation, prediction, bbox)
    return segmentation, {
        "num_prompts": len(prompts),
        "text_embeddings_shape": list(text_embeddings.shape),
        "preprocessed_shape": list(data.shape),
        "padded_shape": list(padded.shape),
        "tile_count": int(tile_count),
        "bbox": [[int(v) for v in axis] for axis in bbox],
        "orig_shape": [int(v) for v in orig_shape],
        "aggregate_on_gpu": bool(aggregate_on_gpu),
    }


def reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_memory_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.set_device(device)
    return int(torch.cuda.max_memory_allocated())


def run_inference_variant(
    predictor: VoxTellPredictor,
    reader: NibabelIOWithReorient,
    entry: dict[str, Any],
    patch_size: int,
    exp_dir: Path,
    device: torch.device,
    overwrite: bool,
) -> dict[str, Any]:
    label = patch_label(patch_size)
    pred_dir = exp_dir / "predictions" / label
    log_dir = exp_dir / "logs"
    status_path = log_dir / f"{label}_status.json"
    pred_path = pred_dir / entry["name"]
    ct_path = ct_rate_abs_path(entry["name"], CT_ROOT)
    gt_path = REX_SEG_DIR / entry["name"]
    prompts = sorted_prompts(entry)
    result: dict[str, Any] = {
        "label": label,
        "case_name": entry["name"],
        "case_index": entry.get("_index"),
        "patch_size": [patch_size, patch_size, patch_size],
        "ct_path": str(ct_path),
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "status_json": str(status_path),
        "started_at": utc_now_iso(),
        "status": "pending",
        "fallback": None,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if pred_path.exists() and not overwrite:
        result["status"] = "skipped_existing"
        write_json(status_path, result)
        return result

    try:
        patch_metadata = set_trial_patch_size(predictor, patch_size)
        result.update(patch_metadata)
        image, ct_properties = reader.read_images([str(ct_path)])
        gt_img = nib.load(str(gt_path))
        result["gt_shape"] = list(gt_img.shape)
        result["ct_shape_after_reader"] = list(image.shape)

        attempts = [("gpu_aggregation", True)]
        if patch_size == 256:
            attempts.append(("cpu_aggregation_after_oom", False))

        last_error: str | None = None
        for attempt_label, aggregate_on_gpu in attempts:
            try:
                gc.collect()
                reset_cuda_peak_memory(device)
                start = time.perf_counter()
                raw_prediction, runtime = prediction_from_preprocessed(
                    predictor=predictor,
                    image=image,
                    prompts=prompts,
                    aggregate_on_gpu=aggregate_on_gpu,
                )
                elapsed = time.perf_counter() - start
                peak_bytes = peak_cuda_memory_bytes(device)
                exported, orientation = export_prediction_to_gt_layout(raw_prediction, gt_img, ct_properties, entry["name"])
                save_4d_prediction(exported, gt_path, pred_path)
                result.update(
                    {
                        "status": "written",
                        "attempt": attempt_label,
                        "inference_seconds": elapsed,
                        "peak_cuda_memory_bytes": int(peak_bytes),
                        "peak_cuda_memory_gib": float(peak_bytes / (1024**3)),
                        "raw_prediction_shape": list(raw_prediction.shape),
                        "final_prediction_shape": list(exported.shape),
                        "runtime": runtime,
                        "orientation": orientation,
                        "completed_at": utc_now_iso(),
                    }
                )
                write_json(status_path, result)
                return result
            except torch.cuda.OutOfMemoryError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                result["fallback"] = "retry_cpu_aggregation" if patch_size == 256 and aggregate_on_gpu else None
                result["last_error"] = last_error
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                    torch.cuda.empty_cache()
                gc.collect()
                if not (patch_size == 256 and aggregate_on_gpu):
                    raise

        raise RuntimeError(last_error or "inference failed without an exception")
    except Exception as exc:  # noqa: BLE001 - trial should record failures
        result.update(
            {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "completed_at": utc_now_iso(),
            }
        )
        write_json(status_path, result)
        return result


def one_case_dataset_json(entry: dict[str, Any], path: Path) -> None:
    write_json(
        path,
        {
            "test": [
                {
                    "name": entry["name"],
                    "seg_path": entry["name"],
                    "findings": entry.get("findings", {}),
                    "categories": entry.get("categories", {}),
                }
            ]
        },
    )


def run_quick_eval(exp_dir: Path, label: str, dataset_json: Path) -> dict[str, Any]:
    eval_json = exp_dir / "eval" / f"{label}_quick_global_eval.json"
    if eval_json.exists():
        eval_json.unlink()
    cmd = [
        sys.executable,
        str(REX_DATA_ROOT / "rexrank_eval.py"),
        "--gt_dir",
        str(REX_SEG_DIR),
        "--pred_dir",
        str(exp_dir / "predictions" / label),
        "--dataset_json",
        str(dataset_json),
        "--output_json",
        str(eval_json),
        "--num_workers",
        "1",
        "--global_only",
    ]
    subprocess.run(cmd, check=True)
    data = json.loads(eval_json.read_text())
    data["eval_json"] = str(eval_json)
    return data


def canonical_case_metrics(case_name: str, eval_json: Path) -> dict[str, Any] | None:
    if not eval_json.exists():
        return None
    data = json.loads(eval_json.read_text())
    for case in data.get("cases", []):
        if case.get("file") == case_name:
            return {
                "eval_json": str(eval_json),
                "file": case_name,
                "summary": case.get("case_stats", {}),
                "findings": case.get("findings", {}),
            }
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(
    exp_dir: Path,
    entry: dict[str, Any],
    prompts: list[str],
    variants: list[dict[str, Any]],
    evals: dict[str, dict[str, Any]],
    canonical: dict[str, Any] | None,
    model_dir: Path,
) -> None:
    summary: dict[str, Any] = {
        "trial": exp_dir.name,
        "created_at": utc_now_iso(),
        "case_name": entry["name"],
        "case_index": entry.get("_index"),
        "model_dir": str(model_dir),
        "prompts": prompts,
        "variants": variants,
        "evals": {
            label: {
                "summary": data.get("summary", {}),
                "cases": data.get("cases", []),
                "eval_json": data.get("eval_json"),
            }
            for label, data in evals.items()
        },
        "canonical_reference": canonical,
    }
    summary_json = exp_dir / "reports" / "patch_size_trial_summary.json"
    write_json(summary_json, summary)

    lines = [
        "# VoxTell Patch-Size Trial",
        "",
        "This is a trial-only single-case comparison. It changes only the runtime",
        "sliding-window patch size and regenerates VoxTell's deterministic",
        "sinusoidal positional encoding for the corresponding feature grid. It does",
        "not retrain or modify checkpoint weights.",
        "",
        "## Case",
        "",
        f"- Validation index: `{entry.get('_index')}`",
        f"- Case: `{entry['name']}`",
        f"- Findings/prompts: `{len(prompts)}`",
        f"- Model directory: `{model_dir}`",
        "",
        "## Summary Metrics",
        "",
        "| Variant | Status | Patch | Feature grid | Tiles | Time (s) | Peak CUDA GiB | Mean Dice/finding | Mean Dice/case | Hit rate | Hits | Fallback |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant in variants:
        label = variant["label"]
        eval_summary = evals.get(label, {}).get("summary", {})
        runtime = variant.get("runtime", {})
        lines.append(
            "| {label} | {status} | `{patch}` | `{grid}` | {tiles} | {seconds} | {memory} | {dice_f} | {dice_c} | {hit} | {hits} | {fallback} |".format(
                label=label,
                status=variant.get("status"),
                patch=variant.get("patch_size"),
                grid=variant.get("feature_grid_shape"),
                tiles=fmt(runtime.get("tile_count")),
                seconds=fmt(variant.get("inference_seconds")),
                memory=fmt(variant.get("peak_cuda_memory_gib")),
                dice_f=fmt(eval_summary.get("mean_global_dice_per_finding")),
                dice_c=fmt(eval_summary.get("mean_global_dice_per_case")),
                hit=fmt(eval_summary.get("hit_rate")),
                hits=fmt(eval_summary.get("total_hits")),
                fallback=variant.get("fallback") or "",
            )
        )
    if canonical:
        case_stats = canonical.get("summary", {})
        lines.append(
            "| canonical_exp001_case | reference | `192` | `n/a` | n/a | n/a | n/a | {dice} | {case_dice} | {hit} | {hits} | existing full-val quick eval |".format(
                dice=fmt(case_stats.get("mean_global_dice")),
                case_dice=fmt(case_stats.get("mean_global_dice")),
                hit=fmt(case_stats.get("hit_rate")),
                hits=fmt(case_stats.get("hits")),
            )
        )
    lines.extend(["", "## Per-Finding Metrics", ""])
    lines.extend(["| Finding | Prompt | " + " | ".join(f"{v['label']} Dice/Hit" for v in variants) + " |", "| ---: | --- | " + " | ".join("---" for _ in variants) + " |"])
    for finding_idx, prompt in enumerate(prompts):
        cells = []
        for variant in variants:
            label = variant["label"]
            case = (evals.get(label, {}).get("cases") or [{}])[0]
            finding = (case.get("findings") or {}).get(f"finding_{finding_idx}", {})
            cells.append(f"{fmt(finding.get('global_dice'))} / {finding.get('global_hit')}")
        lines.append(f"| {finding_idx} | {prompt} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Machine-readable summary: `{summary_json}`",
            f"- One-case dataset JSON: `{exp_dir / 'eval' / 'val_idx000_as_test_dataset.json'}`",
            f"- Predictions root: `{exp_dir / 'predictions'}`",
            f"- Logs/status root: `{exp_dir / 'logs'}`",
            "",
        ]
    )
    report_md = exp_dir / "reports" / "patch_size_trial_report.md"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote report: {report_md}")
    print(f"Wrote summary: {summary_json}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--metadata", type=Path, default=REX_METADATA)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--case-name", default="train_13082_a_1.nii.gz")
    parser.add_argument("--patch-sizes", nargs="+", type=int, default=[96, 192, 256])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    for child in ["config", "eval", "logs", "predictions", "reports"]:
        (args.exp_dir / child).mkdir(parents=True, exist_ok=True)

    entries = load_split_entries(args.metadata, [args.split])
    if args.case_index < 0 or args.case_index >= len(entries):
        raise ValueError(f"case index {args.case_index} out of range for {args.split}")
    entry = entries[args.case_index]
    if entry["name"] != args.case_name:
        raise ValueError(f"Expected {args.case_name} at val index {args.case_index}, found {entry['name']}")
    prompts = sorted_prompts(entry)
    model_dir = resolve_model_dir(args.model_dir)
    config = {
        "trial": args.exp_dir.name,
        "created_at": utc_now_iso(),
        "case_index": args.case_index,
        "case_name": entry["name"],
        "patch_sizes": args.patch_sizes,
        "gpu": args.gpu,
        "model_dir": str(model_dir),
        "ct_root": str(CT_ROOT),
        "seg_dir": str(REX_SEG_DIR),
        "metadata": str(args.metadata),
        "canonical_reference_eval": str(CANONICAL_EXP001_EVAL),
    }
    write_json(args.exp_dir / "config" / "patch_size_trial_config.json", config)
    dataset_json = args.exp_dir / "eval" / "val_idx000_as_test_dataset.json"
    one_case_dataset_json(entry, dataset_json)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    predictor = VoxTellPredictor(model_dir=str(model_dir), device=device)
    reader = NibabelIOWithReorient()

    variants: list[dict[str, Any]] = []
    evals: dict[str, dict[str, Any]] = {}
    for patch_size in args.patch_sizes:
        variant = run_inference_variant(
            predictor=predictor,
            reader=reader,
            entry=entry,
            patch_size=patch_size,
            exp_dir=args.exp_dir,
            device=device,
            overwrite=args.overwrite,
        )
        variants.append(variant)
        if variant.get("status") in {"written", "skipped_existing"}:
            evals[variant["label"]] = run_quick_eval(args.exp_dir, variant["label"], dataset_json)

    canonical = canonical_case_metrics(entry["name"], CANONICAL_EXP001_EVAL)
    write_report(args.exp_dir, entry, prompts, variants, evals, canonical, model_dir)
    failures = [variant for variant in variants if variant.get("status") not in {"written", "skipped_existing"}]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
