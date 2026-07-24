#!/usr/bin/env python3
"""Summarize experiment 004 against the original exp003 v123 baseline."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from common import REX_SEG_DIR, write_json


def category(prompt: str) -> str:
    text = prompt.lower()
    rules = [
        ("nodule", r"\bnodul"),
        ("ground_glass", r"ground[- ]glass"),
        ("consolidation", r"consolidat|infiltrat"),
        ("effusion", r"effusion"),
        ("atelectasis", r"atelect"),
        ("cyst_cavity", r"\bcyst|cavit"),
        ("lymph_node", r"lymph|adenopath"),
    ]
    for label, pattern in rules:
        if re.search(pattern, text):
            return label
    return "other"


def size_stratum(voxels: int) -> str:
    if voxels <= 8:
        return "tiny_1_8_voxels"
    if voxels <= 64:
        return "small_9_64_voxels"
    if voxels <= 512:
        return "medium_65_512_voxels"
    return "large_over_512_voxels"


def aggregate(rows: list[dict], key: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        label: {
            "findings": len(items),
            "mean_dice": float(np.mean([item["dice"] for item in items])),
            "hit_rate": float(np.mean([item["hit"] for item in items])),
            "mean_gt_voxels": float(np.mean([item["gt_voxels"] for item in items])),
            "mean_gt_volume_mm3": float(np.mean([item["gt_volume_mm3"] for item in items])),
        }
        for label, items in sorted(grouped.items())
    }


def load_rows(eval_json: Path, dataset_entries: list[dict], seg_dir: Path) -> tuple[dict, list[dict]]:
    data = json.loads(eval_json.read_text())
    cases = {case["file"]: case for case in data["cases"]}
    rows = []
    for entry in dataset_entries:
        name = entry["name"]
        gt_img = nib.load(str(seg_dir / name))
        gt = np.asanyarray(gt_img.dataobj)
        voxel_mm3 = float(abs(np.linalg.det(gt_img.affine[:3, :3])))
        prompts = [entry["findings"][key] for key in sorted(entry["findings"], key=lambda x: int(x))]
        for index, prompt in enumerate(prompts):
            metrics = cases[name]["findings"][f"finding_{index}"]
            voxels = int(np.count_nonzero(gt[index]))
            rows.append(
                {
                    "name": name,
                    "finding_index": index,
                    "prompt": prompt,
                    "category": category(prompt),
                    "size_stratum": size_stratum(voxels),
                    "sparse_mask": voxels <= 64,
                    "gt_voxels": voxels,
                    "gt_volume_mm3": voxels * voxel_mm3,
                    "dice": float(metrics["global_dice"]),
                    "hit": bool(metrics["global_hit"]),
                }
            )
    return data["summary"], rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, default=REX_SEG_DIR)
    parser.add_argument("--input", action="append", required=True, help="label=/path/to/eval.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    dataset_entries = json.loads(args.dataset_json.read_text())["test"]
    results = {}
    for item in args.input:
        label, path = item.split("=", 1)
        summary, rows = load_rows(Path(path), dataset_entries, args.seg_dir)
        results[label] = {
            "eval_json": path,
            "summary": summary,
            "by_category": aggregate(rows, "category"),
            "by_size": aggregate(rows, "size_stratum"),
            "by_sparse_mask": aggregate(rows, "sparse_mask"),
        }
    payload = {
        "dataset_json": str(args.dataset_json),
        "strata_note": (
            "Size and sparse strata use native GT voxel counts: tiny 1-8, small 9-64, "
            "medium 65-512, large >512; sparse means <=64 voxels."
        ),
        "results": results,
    }
    write_json(args.output_json, payload)

    lines = [
        "# Experiment 004 Paired Comparison",
        "",
        "Primary results use threshold 0.5 on the fixed val200 set.",
        "",
        "| Model | Dice | Hit rate | Hits / findings |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, result in results.items():
        summary = result["summary"]
        lines.append(
            f"| {label} | {summary['mean_global_dice_per_finding']:.4f} | "
            f"{summary['hit_rate']:.4f} | {summary['total_hits']} / {summary['total_findings']} |"
        )
    for section, key in [
        ("Category", "by_category"),
        ("Native GT Size", "by_size"),
        ("Sparse Mask", "by_sparse_mask"),
    ]:
        labels = sorted({name for result in results.values() for name in result[key]})
        lines.extend(["", f"## {section}", "", "| Stratum | Model | N | Dice | Hit rate |", "| --- | --- | ---: | ---: | ---: |"])
        for stratum in labels:
            for model, result in results.items():
                if stratum not in result[key]:
                    continue
                item = result[key][stratum]
                lines.append(
                    f"| {stratum} | {model} | {item['findings']} | "
                    f"{item['mean_dice']:.4f} | {item['hit_rate']:.4f} |"
                )
    lines.extend(["", payload["strata_note"], ""])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
