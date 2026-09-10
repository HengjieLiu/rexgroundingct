"""Frozen-method input adapter. No labels or scores are accepted here."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "frozen_sources"))
sys.path.insert(0, str(REPO / "scripts/rexgroundingct"))
sys.path.insert(0, str(REPO / "side_experiments/sideexp003_ensemble_method_hub"))

import nibabel as nib
import numpy as np
from scipy import ndimage

from preliminary_ensemble import sigmoid_float32, dice_from_counts, summarize_rows
from run_024_test_inference_anatomy import physical_dilation, route_prompt, LABELS
from evaluate_semantic_aware_lobe_constraint_part1 import semantic_parse
from audit_fine_spatial_modifiers import world_axis_info, relevant_sides, split_lobes
from audit_segment_v2_scopeaware import scope_plan, combine_scope_proxy, SIMPLE_PATTERNS
from apply_semantic_v2_strict_to_rex_predictions import signature, FROZEN_SIGNATURES
from run_totalseg_lung_priors import dilate_mm as source_dilate_zyx

LOBE_LABELS = {"LUL": 10, "LLL": 11, "RUL": 12, "RML": 13, "RLL": 14}
VARIANTS = ("d1", "d2", "d3", "d11", "d12")
CATEGORIES = ("1a", "1b", "1c", "1d", "1e", "1f", "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h")
CATEGORY_NAMES = dict(zip(CATEGORIES, (
    "Bronchial wall thickening", "Bronchiectasis", "Emphysema", "Septal thickening",
    "Micronodules", "Other (non-focal)", "Linear / scarring / fibrosis",
    "Atelectasis / consolidation", "Groundglass opacity", "Pulmonary nodules/masses",
    "Pleural effusion or thickening", "Honeycombing", "Pneumothorax", "Other (focal)")))


def ordered_findings(case):
    keys = sorted(case["findings"], key=int)
    if [int(k) for k in keys] != list(range(len(keys))) or set(keys) != set(case["categories"]):
        raise ValueError("finding indices must be unique contiguous numeric indices")
    return [{"finding_idx": int(k), "prompt": case["findings"][k],
             "category": case["categories"][k]} for k in keys]


def parse_finding(prompt):
    parsed = semantic_parse(prompt)
    if parsed["group"] == "high_conf_exact_lobe" and parsed["lobes"]:
        key, kind, margin = tuple(sorted(parsed["lobes"])), "lobe", 15
    elif parsed["group"] in {"high_conf_exact_lobe", "ambiguous_lobe", "laterality_only"} and parsed["side"] in {"left", "right"}:
        key, kind, margin = (parsed["side"],), "side", 15
    else:
        key, kind, margin = ("whole",), "whole", 10
    plan = scope_plan(prompt, set(SIMPLE_PATTERNS))
    sig = signature(plan)
    return {"parsed": parsed, "base_key": list(key), "roi_kind": kind,
            "dilation_mm": margin, "scope_plan": plan, "signature": sig,
            "v2_selected": bool(plan["hard_eligible"] and sig in FROZEN_SIGNATURES)}


def anatomy_priors(anatomy):
    p = {name: anatomy == label for name, label in LOBE_LABELS.items()}
    p["left"] = p["LUL"] | p["LLL"]
    p["right"] = p["RUL"] | p["RML"] | p["RLL"]
    p["whole"] = p["left"] | p["right"]
    return p


def whole10(priors, sampling):
    # Preserve the source generator's XYZ-spacing / ZYX-array contract.
    return source_dilate_zyx(priors["whole"].transpose(2, 1, 0), sampling, 10.0).transpose(2, 1, 0)


def postprocess(raw, anatomy, affine, findings, routes):
    """Return four native FXYZ uint8 arrays and outcome-blind routing details.

    Parsers, fine proxies and our physical dilation are imported unchanged.
    The nested v1 selection is lifted into parse_finding; support caching does
    not alter source inequalities. Reference-worker parity tests cover both.
    """
    sampling = tuple(float(x) for x in nib.affines.voxel_sizes(affine)[:3])
    axes = world_axis_info(affine, anatomy.shape)
    priors = anatomy_priors(anatomy)
    output = {v: np.array(raw, dtype=np.uint8, copy=True) for v in VARIANTS[1:]}
    support_cache, base_cache, proxy_cache = {}, {}, {}
    details = []
    for finding in findings:
        idx = finding["finding_idx"]
        route = routes[idx]
        for variant, eligible, labels in (
            ("d2", route["eligible"], LABELS["W"]),
            ("d3", route["fine_eligible"], route["selected_labels"]),
        ):
            # Match transform_case_dir, including the empty-label bypass.
            if eligible and labels:
                cache_key = ("ours", tuple(labels))
                if cache_key not in support_cache:
                    support_cache[cache_key] = physical_dilation(np.isin(anatomy, labels), affine, 20.0)
                output[variant][idx] &= support_cache[cache_key]

        info = parse_finding(finding["prompt"])
        key = tuple(info["base_key"])
        if key not in base_cache:
            base_cache[key] = np.logical_or.reduce([priors[k] for k in key]) if len(key) > 1 else priors[key[0]]
        base = base_cache[key]
        cache_key = ("collaborator", key, info["dilation_mm"])
        if cache_key not in support_cache:
            dist = ndimage.distance_transform_edt(~base, sampling=sampling)
            support_cache[cache_key] = base | (dist <= float(info["dilation_mm"]))
            del dist
        roi = support_cache[cache_key]
        output["d11"][idx] &= roi
        output["d12"][idx] = output["d11"][idx]
        if info["v2_selected"]:
            # V2 uses the original precomputed whole10 for whole-lung routes.
            if info["roi_kind"] == "whole":
                if "whole10" not in priors:
                    priors["whole10"] = whole10(priors, sampling)
                roi = priors["whole10"]
            parsed = info["parsed"]
            side = parsed["side"] if parsed["side"] in {"left", "right"} else None
            lobes = split_lobes(";".join(parsed["lobes"]))
            scope, proxy_details = combine_scope_proxy(
                info["scope_plan"], roi, base, priors, axes, sampling,
                relevant_sides(side, lobes), proxy_cache)
            output["d12"][idx] &= scope
            info["proxy_details"] = proxy_details
        details.append({"finding_idx": idx, "ours": route, "collaborator": info})
    return output, details
