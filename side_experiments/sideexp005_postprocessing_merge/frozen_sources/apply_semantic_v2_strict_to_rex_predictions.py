#!/usr/bin/env python3
"""Emit deployable frozen semantic-v2 masks from frozen semantic-v1 masks.

The v2 policy is intentionally narrow: it retains semantic-v1 unless a finding
has one of the five scope signatures that was frozen before the step-9000
checkpoint transfer.  It never reads ground-truth masks.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from audit_fine_spatial_modifiers import (
    build_proxy,
    relevant_sides,
    split_lobes,
    world_axis_info,
)
from audit_segment_v2_scopeaware import SIMPLE_PATTERNS, combine_scope_proxy, scope_plan
from evaluate_anatomy_prior_ablation import channel_first_mask
from evaluate_anatomy_prior_diagnostic import LOBE_FILES, geometry_reference, load_prior
from evaluate_semantic_aware_lobe_constraint_part1 import semantic_parse


FROZEN_SIGNATURES = {
    "apical",
    "basal",
    "superior_within_base",
    "laterobasal",
    "subpleural_peripheral+posterior",
}


def signature(plan: dict) -> str:
    return " OR ".join(
        "+".join(item["name"] for item in group) for group in plan["atom_groups"]
    )


def case_worker(task: tuple[str, list[dict], dict]) -> tuple[list[dict], str | None]:
    case, findings, settings = task
    try:
        pred_path = Path(settings["pred_dir"]) / case
        pred, pred_ref = channel_first_mask(pred_path, 0.5, False)
        ref = geometry_reference(case, pred_ref, Path(settings["ct_dir"]))
        sampling = tuple(float(item) for item in nib.affines.voxel_sizes(ref.affine)[:3])
        axes = world_axis_info(ref.affine, tuple(int(item) for item in pred.shape[1:]))
        priors = {
            "whole": load_prior(Path(settings["prior_dir"]), case, "whole_lung", ref),
            "whole10": load_prior(Path(settings["prior_dir"]), case, "lung_dilated_10mm", ref),
        }
        for lobe, filename in LOBE_FILES.items():
            priors[lobe] = load_prior(Path(settings["prior_dir"]), case, filename, ref)
        priors["left"] = priors["LUL"] | priors["LLL"]
        priors["right"] = priors["RUL"] | priors["RML"] | priors["RLL"]

        out = pred.copy()
        cache: dict[str, np.ndarray] = {}
        rows: list[dict] = []
        for finding in findings:
            finding_id = int(finding["finding_idx"])
            if finding_id >= pred.shape[0]:
                raise IndexError(f"{case}: finding {finding_id} outside {pred.shape[0]} channels")
            parsed = semantic_parse(str(finding["prompt"]))
            side = parsed["side"] if parsed["side"] in {"left", "right"} else None
            lobes = split_lobes(";".join(parsed["lobes"]))
            if parsed["group"] == "high_conf_exact_lobe" and lobes:
                strict_base = np.logical_or.reduce([priors[lobe] for lobe in lobes])
                semantic_roi = ndimage.distance_transform_edt(~strict_base, sampling=sampling) <= 15.0
            elif parsed["group"] in {"high_conf_exact_lobe", "ambiguous_lobe", "laterality_only"} and side:
                strict_base = priors[side]
                semantic_roi = ndimage.distance_transform_edt(~strict_base, sampling=sampling) <= 15.0
            else:
                strict_base = priors["whole"]
                semantic_roi = priors["whole10"]

            # Passing all known lexical clusters is safe: scope_plan retains only
            # patterns actually present in the prompt.
            plan = scope_plan(str(finding["prompt"]), set(SIMPLE_PATTERNS))
            scope_signature = signature(plan)
            selected = bool(plan["hard_eligible"] and scope_signature in FROZEN_SIGNATURES)
            if selected:
                scope_roi, _ = combine_scope_proxy(
                    plan,
                    semantic_roi,
                    strict_base,
                    priors,
                    axes,
                    sampling,
                    relevant_sides(side, lobes),
                    cache,
                )
                out[finding_id] &= scope_roi
            rows.append(
                {
                    "case_id": case,
                    "finding_id": finding_id,
                    "scope_signature": scope_signature,
                    "hard_eligible": bool(plan["hard_eligible"]),
                    "semantic_v2_strict_selected": selected,
                    "voxels_before": int(pred[finding_id].sum()),
                    "voxels_after": int(out[finding_id].sum()),
                }
            )

        original = nib.load(str(pred_path))
        source = np.asanyarray(original.dataobj)
        if source.ndim == 4 and source.shape[0] <= 64:
            saved = out.astype(np.uint8)
        elif source.ndim == 4 and source.shape[-1] <= 64:
            saved = np.moveaxis(out, 0, -1).astype(np.uint8)
        elif source.ndim == 3 and out.shape[0] == 1:
            saved = out[0].astype(np.uint8)
        else:
            raise ValueError(f"Unsupported prediction layout for {case}: {source.shape}")
        output_path = Path(settings["output_dir"]) / case
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = original.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(saved, original.affine, header), str(output_path))
        return rows, None
    except Exception as exc:  # pragma: no cover - surfaced by main
        return [], f"{case}: {type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--findings-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignments-out", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    findings = pd.read_csv(args.findings_csv)
    required = {"case_id", "finding_idx", "prompt"}
    if missing := required - set(findings.columns):
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    tasks = [
        (str(case), frame.to_dict(orient="records"), {
            "pred_dir": str(args.pred_dir), "prior_dir": str(args.prior_dir),
            "ct_dir": str(args.ct_dir), "output_dir": str(args.output_dir),
        })
        for case, frame in findings.groupby("case_id", sort=True)
    ]
    rows: list[dict] = []
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
        futures = [pool.submit(case_worker, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="semantic-v2 strict"):
            case_rows, failure = future.result()
            rows.extend(case_rows)
            if failure:
                failures.append(failure)
    if failures:
        raise RuntimeError(f"{len(failures)} v2 failures; first: {failures[0]}")
    frame = pd.DataFrame(rows).sort_values(["case_id", "finding_id"])
    if len(frame) != len(findings):
        raise RuntimeError(f"Expected {len(findings)} findings, emitted {len(frame)}")
    args.assignments_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.assignments_out, index=False)
    print(json.dumps({"cases": len(tasks), "findings": len(frame), "v2_selected": int(frame.semantic_v2_strict_selected.sum())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
