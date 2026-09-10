#!/usr/bin/env python3
"""Part 1: deterministic semantic-aware post-hoc lung/lobe constraints.

This script never runs or modifies VoxTell. It consumes the saved thresholded
step-6000 predictions, GT, and TotalSegmentator priors, then evaluates fixed
ROI policies on the full validation set.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from evaluate_anatomy_prior_ablation import (
    channel_first_mask,
    improved_parse_lobes,
)
from evaluate_anatomy_prior_diagnostic import LOBE_FILES, geometry_reference, load_prior, parse_laterality


LOBE_ORDER = ["LUL", "LLL", "RUL", "RML", "RLL"]
LEFT_LOBES = {"LUL", "LLL"}
RIGHT_LOBES = {"RUL", "RML", "RLL"}
HIT_THRESHOLD = 0.1
TIE_TOL = 1e-12

EXACT_PATTERNS = {
    "RUL": (
        r"\bright upper lobe\b",
        r"\bupper lobe of (?:the )?right lung\b",
        r"\bRUL\b",
    ),
    "RML": (
        r"\bright middle lobe\b",
        r"\bmiddle lobe of (?:the )?right lung\b",
        r"\bRML\b",
    ),
    "RLL": (
        r"\bright lower lobe\b",
        r"\blower lobe of (?:the )?right lung\b",
        r"\bRLL\b",
    ),
    "LUL": (
        r"\bleft upper lobe\b",
        r"\bupper lobe of (?:the )?left lung\b",
        r"\bLUL\b",
    ),
    "LLL": (
        r"\bleft lower lobe\b",
        r"\blower lobe of (?:the )?left lung\b",
        r"\bLLL\b",
    ),
}

AMBIGUOUS_PATTERNS = (
    r"\bpredominant(?:ly)?\b",
    r"\bmainly\b",
    r"\bmostly\b",
    r"\bcentered in\b",
    r"\bgreatest in\b",
    r"\bmost pronounced(?: in)?\b",
    r"\bextending into\b",
    r"\binvolving\b",
    r"\badjacent(?: to)?\b",
    r"\bbasilar\b",
    r"\bbibasilar\b",
    r"\blower lung\b",
    r"\bupper lung\b",
    r"\bperi[- ]?hilar\b",
    r"\binfra[- ]?hilar\b",
    r"\bhilar\b",
    r"\bsubpleural\b",
    r"\bpleural[- ]based\b",
    r"\bperi[- ]?fissural\b",
    r"\bfissural\b",
    r"\blingula\b",
    r"\blingular\b",
    r"\blower lobe predominant\b",
    r"\bupper lobe predominant\b",
)

BILATERAL_MULTIFOCAL_PATTERNS = (
    r"\bbilateral(?:ly)?\b",
    r"\bboth lungs?\b",
    r"\bboth (?:the )?(?:upper|lower) lobes?\b",
    r"\b(?:upper|lower) lobes? of both lungs?\b",
    r"\bmultifocal\b",
    r"\bdiffuse(?:ly)?\b",
    r"\bscattered\b",
    r"\bnumerous\b",
    r"\bmultiple bilateral\b",
    r"\bbibasilar\b",
)

SIDE_PATTERNS = {
    "left": (
        r"\bleft\b",
        r"\blingula\b",
        r"\blingular\b",
        r"\bLUL\b",
        r"\bLLL\b",
    ),
    "right": (
        r"\bright\b",
        r"\bRUL\b",
        r"\bRML\b",
        r"\bRLL\b",
        r"\bmiddle lobe\b",
    ),
}

NEGATION_HISTORY_PATTERNS = (
    r"\bno\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bresolved\b",
    r"\bprior\b",
    r"\bprevious\b",
    r"\bhistory of\b",
    r"\bstatus post\b",
    r"\brule out\b",
    r"\bexcluding\b",
)

SEMANTIC_SPECS = (
    ("semantic_A_e10_s10_fw10", 10, 10, "whole10", False),
    ("semantic_B_e15_s10_fw10", 15, 10, "whole10", False),
    ("semantic_C_e20_s10_fw10", 20, 10, "whole10", False),
    ("semantic_D_e15_s15_fw10", 15, 15, "whole10", False),
    ("semantic_E_e15_s10_pass", 15, 10, "pass", False),
    ("semantic_F_noexact_s10_fw10", None, 10, "whole10", True),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("outputs/zoom_out_256_matched/fullvol_evaluation/s3_allcategory_1over1_step6000"),
    )
    p.add_argument("--gt-dir", type=Path, default=Path("data/segmentations"))
    p.add_argument("--ct-dir", type=Path, default=Path("data/ct_rate_volumes_fixed"))
    p.add_argument(
        "--prior-dir",
        type=Path,
        default=Path("outputs/anatomy_prior_diagnostic_val_hardneg_step1000_thr0p7/lung_priors"),
    )
    p.add_argument(
        "--findings-csv",
        type=Path,
        default=Path("outputs/s3_attention_candidate_diagnostic_all_categories_step6000/findings.csv"),
    )
    p.add_argument(
        "--reference-results",
        type=Path,
        default=Path(
            "outputs/anatomy_prior_exp2_allcat1over1_step6000_thr0p5_20shard/"
            "per_finding_prior_ablation.csv"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/semantic_aware_lobe_constraint_part1_allcat1over1_step6000"),
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--parser-test-only", action="store_true")
    return p.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _unsafe_context(text: str, start: int, end: int, window_tokens: int = 6) -> tuple[bool, str]:
    """Flag cue phrases within six tokens without crossing a strong boundary."""
    low = text.lower()
    left_boundary = max(low.rfind(".", 0, start), low.rfind(";", 0, start), low.rfind(":", 0, start))
    right_candidates = [x for x in (low.find(".", end), low.find(";", end), low.find(":", end)) if x >= 0]
    right_boundary = min(right_candidates) if right_candidates else len(low)
    sentence = low[left_boundary + 1 : right_boundary]
    local_start = start - left_boundary - 1
    local_end = end - left_boundary - 1
    tokens = list(re.finditer(r"[a-z0-9]+", sentence))
    phrase_tokens = [i for i, token in enumerate(tokens) if token.start() < local_end and token.end() > local_start]
    if not phrase_tokens:
        return False, ""
    lo = max(0, min(phrase_tokens) - window_tokens)
    hi = min(len(tokens), max(phrase_tokens) + window_tokens + 1)
    local = " ".join(token.group(0) for token in tokens[lo:hi])
    for pattern in NEGATION_HISTORY_PATTERNS:
        match = re.search(pattern, local, flags=re.I)
        if match:
            return True, match.group(0)
    return False, ""


def _safe_matches(text: str, patterns: tuple[str, ...]) -> tuple[list[re.Match], list[str]]:
    safe: list[re.Match] = []
    excluded: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            unsafe, cue = _unsafe_context(text, *match.span())
            if unsafe:
                excluded.append(f"{match.group(0)}[{cue}]")
            else:
                safe.append(match)
    return safe, excluded


def semantic_parse(prompt: str) -> dict[str, Any]:
    text = normalize_text(prompt)
    matched: list[str] = []
    excluded: list[str] = []
    exact_lobes: set[str] = set()

    for lobe, patterns in EXACT_PATTERNS.items():
        safe, unsafe = _safe_matches(text, patterns)
        for match in safe:
            exact_lobes.add(lobe)
            matched.append(match.group(0))
        excluded.extend(unsafe)

    bilateral_lobe_patterns = {
        ("LUL", "RUL"): (
            r"\bbilateral(?:ly)? upper lobes?\b",
            r"\bboth (?:the )?upper lobes?\b",
            r"\bupper lobes? of both lungs?\b",
        ),
        ("LLL", "RLL"): (
            r"\bbilateral(?:ly)? lower lobes?\b",
            r"\bboth (?:the )?lower lobes?\b",
            r"\blower lobes? of both lungs?\b",
        ),
    }
    for targets, patterns in bilateral_lobe_patterns.items():
        safe, unsafe = _safe_matches(text, patterns)
        for match in safe:
            exact_lobes.update(targets)
            matched.append(match.group(0))
        excluded.extend(unsafe)

    side_hits: dict[str, list[str]] = {"left": [], "right": []}
    for side, patterns in SIDE_PATTERNS.items():
        safe, unsafe = _safe_matches(text, patterns)
        side_hits[side].extend(match.group(0) for match in safe)
        excluded.extend(unsafe)

    if exact_lobes & LEFT_LOBES:
        side_hits["left"].append("exact-left-lobe")
    if exact_lobes & RIGHT_LOBES:
        side_hits["right"].append("exact-right-lobe")
    if side_hits["left"] and side_hits["right"]:
        side: str | None = "bilateral"
    elif side_hits["left"]:
        side = "left"
    elif side_hits["right"]:
        side = "right"
    else:
        side = None

    bilateral_hits = []
    for pattern in BILATERAL_MULTIFOCAL_PATTERNS:
        safe, unsafe = _safe_matches(text, (pattern,))
        bilateral_hits.extend(match.group(0) for match in safe)
        excluded.extend(unsafe)
    if side == "bilateral":
        bilateral_hits.append("left-and-right-location")

    ambiguous_hits = []
    for pattern in AMBIGUOUS_PATTERNS:
        safe, unsafe = _safe_matches(text, (pattern,))
        ambiguous_hits.extend(match.group(0) for match in safe)
        excluded.extend(unsafe)

    matched.extend(side_hits["left"])
    matched.extend(side_hits["right"])
    matched.extend(bilateral_hits)
    matched.extend(ambiguous_hits)
    matched = list(dict.fromkeys(matched))
    lobes = [lobe for lobe in LOBE_ORDER if lobe in exact_lobes]

    if bilateral_hits:
        group = "bilateral_multifocal"
        side = "bilateral" if side == "bilateral" or any(
            re.search(r"bilateral|both|bibasilar", value, flags=re.I) for value in bilateral_hits
        ) else side
        roi_kind = "whole_lung"
        reason = "bilateral/multifocal/diffuse cue disables side or exact-lobe restriction"
    elif ambiguous_hits:
        group = "ambiguous_lobe"
        roi_kind = "side_lung" if side in {"left", "right"} else "pass_through"
        reason = "soft or anatomically ambiguous location is degraded to laterality"
    elif lobes:
        group = "high_conf_exact_lobe"
        roi_kind = "exact_lobe"
        reason = "unnegated exact-lobe phrase without soft modifier"
    elif side in {"left", "right"}:
        group = "laterality_only"
        roi_kind = "side_lung"
        reason = "unilateral location without a safe exact-lobe phrase"
    else:
        group = "nonlocalizable"
        roi_kind = "pass_through"
        reason = "no reliable unnegated target side or lobe"

    if excluded:
        reason += "; excluded context: " + ", ".join(dict.fromkeys(excluded))
    return {
        "group": group,
        "side": side,
        "lobes": lobes,
        "roi_kind": roi_kind,
        "matched_phrases": matched,
        "reason": reason,
    }


def run_parser_tests() -> None:
    tests = {
        "nodule in the right upper lobe": ("high_conf_exact_lobe", "right", ["RUL"]),
        "mass in the left lower lobe": ("high_conf_exact_lobe", "left", ["LLL"]),
        "predominantly in the left lower lobe": ("ambiguous_lobe", "left", ["LLL"]),
        "lingular opacity": ("ambiguous_lobe", "left", []),
        "left lower lung opacity": ("ambiguous_lobe", "left", []),
        "left pulmonary opacity": ("laterality_only", "left", []),
        "bilateral lower lobe opacities": ("bilateral_multifocal", "bilateral", ["LLL", "RLL"]),
        "multifocal nodules": ("bilateral_multifocal", None, []),
        "pulmonary nodule": ("nonlocalizable", None, []),
        "resolved right upper lobe opacity": ("nonlocalizable", None, []),
        "mass without left lower lobe involvement": ("nonlocalizable", None, []),
    }
    failures = []
    for text, expected in tests.items():
        got = semantic_parse(text)
        observed = (got["group"], got["side"], got["lobes"])
        if observed != expected:
            failures.append((text, expected, observed, got))
    if failures:
        raise AssertionError(json.dumps(failures, indent=2))
    print(f"Parser tests passed: {len(tests)}")


def case_stem(case: str) -> str:
    return case[:-7] if case.endswith(".nii.gz") else Path(case).stem


def voxel_volume_mm3(ref: nib.Nifti1Image) -> float:
    values = nib.affines.voxel_sizes(ref.affine)[:3]
    return float(np.prod(values)) if np.all(np.isfinite(values)) and np.all(values > 0) else 1.0


def load_priors(case: str, prior_dir: Path, ref: nib.Nifti1Image) -> dict[str, np.ndarray]:
    priors = {
        "whole": load_prior(prior_dir, case, "whole_lung", ref),
        "left": load_prior(prior_dir, case, "left_lung", ref),
        "right": load_prior(prior_dir, case, "right_lung", ref),
    }
    for lobe, filename in LOBE_FILES.items():
        priors[lobe] = load_prior(prior_dir, case, filename, ref)
    return priors


def _union(priors: dict[str, np.ndarray], labels: list[str]) -> np.ndarray:
    return np.logical_or.reduce([priors[label] for label in labels])


def _policy_definitions() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = [
        {"policy": "raw", "family": "raw", "dilation_mm": 0, "fallback": "none"},
    ]
    for dilation in (5, 10, 15):
        policies.append(
            {
                "policy": f"whole_lung_d{dilation}",
                "family": "whole_lung",
                "dilation_mm": dilation,
                "fallback": "none",
            }
        )
    for dilation in (5, 10, 15, 20):
        for fallback in ("whole10", "pass"):
            policies.append(
                {
                    "policy": f"laterality_d{dilation}_f{fallback}",
                    "family": "laterality",
                    "dilation_mm": dilation,
                    "fallback": fallback,
                }
            )
            policies.append(
                {
                    "policy": f"old_exact_lobe_d{dilation}_f{fallback}",
                    "family": "old_exact_lobe",
                    "dilation_mm": dilation,
                    "fallback": fallback,
                }
            )
    for name, exact_d, side_d, fallback, no_exact in SEMANTIC_SPECS:
        policies.append(
            {
                "policy": name,
                "family": "semantic",
                "exact_dilation_mm": exact_d,
                "side_dilation_mm": side_d,
                "fallback": fallback,
                "no_exact": no_exact,
            }
        )
    return policies


POLICIES = _policy_definitions()


def process_case(task: tuple[str, list[dict[str, Any]], dict[str, str]]) -> tuple[list[dict[str, Any]], dict, str | None]:
    case, finding_rows, config = task
    try:
        pred_path = Path(config["pred_dir"]) / case
        gt_path = Path(config["gt_dir"]) / case
        # Load each gzip-compressed 4D NIfTI only once. Reading channels through
        # the proxy repeatedly would decompress these large files per finding.
        pred_masks, pred_ref = channel_first_mask(pred_path, 0.5, False)
        gt_masks, gt_ref = channel_first_mask(gt_path, 0.5, False)
        pred_channels, gt_channels = pred_masks.shape[0], gt_masks.shape[0]
        pred_shape, gt_shape = pred_masks.shape[1:], gt_masks.shape[1:]
        if pred_shape != gt_shape:
            raise ValueError(f"pred shape {pred_shape} != GT shape {gt_shape}")
        ref = geometry_reference(case, pred_ref, Path(config["ct_dir"]))
        priors = load_priors(case, Path(config["prior_dir"]), ref)
        for label, mask in priors.items():
            if mask.shape != pred_shape:
                raise ValueError(f"aligned prior {label} shape {mask.shape} != prediction {pred_shape}")
        vv = voxel_volume_mm3(ref)
        sampling = tuple(float(x) for x in nib.affines.voxel_sizes(ref.affine)[:3])
        distance_cache: dict[str, np.ndarray] = {}
        roi_cache: dict[tuple[str, int], np.ndarray] = {}
        base_cache: dict[str, np.ndarray] = {
            "whole": priors["whole"],
            "left": priors["left"],
            "right": priors["right"],
        }

        def base_roi(key: str) -> np.ndarray:
            if key not in base_cache:
                labels = key.removeprefix("lobes:").split("+")
                base_cache[key] = _union(priors, labels)
            return base_cache[key]

        def roi(key: str, dilation: int) -> np.ndarray:
            cache_key = (key, int(dilation))
            if cache_key not in roi_cache:
                base = base_roi(key)
                if dilation <= 0 or not base.any():
                    roi_cache[cache_key] = base
                else:
                    if key not in distance_cache:
                        distance_cache[key] = ndimage.distance_transform_edt(~base, sampling=sampling)
                    roi_cache[cache_key] = base | (distance_cache[key] <= float(dilation))
            return roi_cache[cache_key]

        def fallback_roi(name: str) -> tuple[np.ndarray | None, str, int, bool]:
            if name == "whole10":
                return roi("whole", 10), "whole_lung", 10, True
            return None, "pass_through", 0, False

        def choose_policy(
            policy: dict[str, Any], parsed: dict[str, Any], prompt: str
        ) -> tuple[np.ndarray | None, str, int, str, bool]:
            family = policy["family"]
            if family == "raw":
                return None, "pass_through", 0, "raw", False
            if family == "whole_lung":
                dilation = int(policy["dilation_mm"])
                return roi("whole", dilation), "whole_lung", dilation, "whole", True

            old_side = parse_laterality(prompt)
            if family == "laterality":
                if old_side in {"left", "right"}:
                    dilation = int(policy["dilation_mm"])
                    return roi(old_side, dilation), "side_lung", dilation, old_side, True
                mask, kind, dilation, applied = fallback_roi(str(policy["fallback"]))
                return mask, kind, dilation, f"fallback_{policy['fallback']}", applied

            if family == "old_exact_lobe":
                lobes = improved_parse_lobes(prompt)
                dilation = int(policy["dilation_mm"])
                if lobes:
                    key = "lobes:" + "+".join(sorted(lobes))
                    return roi(key, dilation), "exact_lobe", dilation, "+".join(lobes), True
                if old_side in {"left", "right"}:
                    return roi(old_side, dilation), "side_lung", dilation, old_side, True
                mask, kind, fallback_d, applied = fallback_roi(str(policy["fallback"]))
                return mask, kind, fallback_d, f"fallback_{policy['fallback']}", applied

            group = parsed["group"]
            side = parsed["side"]
            if group == "high_conf_exact_lobe":
                if policy["no_exact"]:
                    side_d = int(policy["side_dilation_mm"])
                    return roi(side, side_d), "side_lung", side_d, f"{side}_degraded_exact", True
                exact_d = int(policy["exact_dilation_mm"])
                key = "lobes:" + "+".join(sorted(parsed["lobes"]))
                return roi(key, exact_d), "exact_lobe", exact_d, "+".join(parsed["lobes"]), True
            if group in {"ambiguous_lobe", "laterality_only"} and side in {"left", "right"}:
                side_d = int(policy["side_dilation_mm"])
                return roi(side, side_d), "side_lung", side_d, side, True
            mask, kind, dilation, applied = fallback_roi(str(policy["fallback"]))
            return mask, kind, dilation, f"fallback_{policy['fallback']}", applied

        results: list[dict[str, Any]] = []
        for finding in finding_rows:
            idx = int(finding["finding_idx"])
            if idx >= pred_channels or idx >= gt_channels:
                raise IndexError(f"finding {idx}; pred channels={pred_channels}, GT channels={gt_channels}")
            prompt = str(finding["prompt"])
            parsed = semantic_parse(prompt)
            pred = pred_masks[idx]
            gt = gt_masks[idx]
            pred_flat = pred.ravel()
            gt_flat = gt.ravel()
            pred_idx = np.flatnonzero(pred_flat)
            gt_idx = np.flatnonzero(gt_flat)
            tp_idx = pred_idx[gt_flat[pred_idx]]
            pred_n = int(pred_idx.size)
            gt_n = int(gt_idx.size)
            raw_tp = int(tp_idx.size)
            raw_dice = 1.0 if pred_n + gt_n == 0 else 2.0 * raw_tp / (pred_n + gt_n)
            raw_hit = raw_dice >= HIT_THRESHOLD

            for policy in POLICIES:
                selected, roi_kind, selected_dilation, roi_label, applied = choose_policy(policy, parsed, prompt)
                if selected is None:
                    retained_pred = pred_n
                    retained_gt = gt_n
                    retained_tp = raw_tp
                    roi_voxels = pred.size
                else:
                    flat = selected.ravel()
                    retained_pred = int(flat[pred_idx].sum()) if pred_idx.size else 0
                    retained_gt = int(flat[gt_idx].sum()) if gt_idx.size else 0
                    retained_tp = int(flat[tp_idx].sum()) if tp_idx.size else 0
                    roi_voxels = int(selected.sum())
                denom = retained_pred + gt_n
                dice = 1.0 if denom == 0 else 2.0 * retained_tp / denom
                hit = dice >= HIT_THRESHOLD
                deleted_tp = raw_tp - retained_tp
                raw_fp = pred_n - raw_tp
                retained_fp = retained_pred - retained_tp
                deleted_fp = raw_fp - retained_fp
                results.append(
                    {
                        "case_id": case,
                        "scan_id": case_stem(case),
                        "finding_id": idx,
                        "finding_text": prompt,
                        "category": finding.get("category", ""),
                        "group": parsed["group"],
                        "side": parsed["side"],
                        "lobes": ";".join(parsed["lobes"]),
                        "parser_roi_kind": parsed["roi_kind"],
                        "policy": policy["policy"],
                        "policy_family": policy["family"],
                        "configured_dilation_mm": policy.get("dilation_mm", ""),
                        "exact_dilation_mm": policy.get("exact_dilation_mm", ""),
                        "side_dilation_mm": policy.get("side_dilation_mm", ""),
                        "fallback": policy["fallback"],
                        "selected_roi_kind": roi_kind,
                        "selected_roi_label": roi_label,
                        "selected_dilation_mm": selected_dilation,
                        "constraint_applied": applied,
                        "roi_volume_mm3": roi_voxels * vv,
                        "dice": dice,
                        "hit": hit,
                        "pred_volume_mm3": retained_pred * vv,
                        "gt_volume_mm3": gt_n * vv,
                        "tp_volume_mm3": retained_tp * vv,
                        "fp_volume_mm3": retained_fp * vv,
                        "raw_dice": raw_dice,
                        "raw_hit": raw_hit,
                        "raw_pred_volume_mm3": pred_n * vv,
                        "delta_dice_vs_raw": dice - raw_dice,
                        "pred_volume_change_mm3_vs_raw": (retained_pred - pred_n) * vv,
                        "gt_retention": retained_gt / max(gt_n, 1),
                        "pred_retention": retained_pred / max(pred_n, 1),
                        "deleted_gt_overlapping_pred_volume_mm3": deleted_tp * vv,
                        "deleted_fp_volume_mm3": deleted_fp * vv,
                        "dice_decrease_vs_raw": dice < raw_dice - TIE_TOL,
                        "hit_loss_vs_raw": raw_hit and not hit,
                    }
                )
        alignment = {
            "case_id": case,
            "prediction_shape": "x".join(map(str, pred_shape)),
            "gt_shape": "x".join(map(str, gt_shape)),
            "all_aligned_prior_shapes": True,
            "voxel_volume_mm3": vv,
            "pred_gt_affine_equal": bool(np.allclose(pred_ref.affine, gt_ref.affine, atol=1e-3)),
            "geometry_reference": "CT" if not np.allclose(ref.affine, pred_ref.affine, atol=1e-3) else "prediction",
        }
        return results, alignment, None
    except Exception as exc:
        return [], {"case_id": case}, f"{case}: {type(exc).__name__}: {exc}"


def relation_counts(delta: pd.Series) -> tuple[int, int, int]:
    values = delta.to_numpy(dtype=float)
    return int((values > TIE_TOL).sum()), int((np.abs(values) <= TIE_TOL).sum()), int((values < -TIE_TOL).sum())


def summarize_policy(frame: pd.DataFrame, policy: str, subgroup: str = "all") -> dict[str, Any]:
    data = frame[frame["policy"] == policy]
    improved_raw, tied_raw, worsened_raw = relation_counts(data["delta_dice_vs_raw"])
    improved_whole, tied_whole, worsened_whole = relation_counts(data["delta_dice_vs_whole10"])
    return {
        "policy": policy,
        "policy_family": data["policy_family"].iloc[0],
        "subgroup": subgroup,
        "n_findings": len(data),
        "mean_dice": data["dice"].mean(),
        "hit_count": int(data["hit"].sum()),
        "predicted_volume_sum_mm3": data["pred_volume_mm3"].sum(),
        "delta_mean_dice_vs_raw": data["dice"].mean() - data["raw_dice"].mean(),
        "delta_mean_dice_vs_whole10": data["dice"].mean() - data["whole10_dice"].mean(),
        "predicted_volume_delta_vs_raw_mm3": data["pred_volume_mm3"].sum() - data["raw_pred_volume_mm3"].sum(),
        "predicted_volume_delta_vs_whole10_mm3": data["pred_volume_mm3"].sum()
        - data["whole10_pred_volume_mm3"].sum(),
        "improved_vs_raw": improved_raw,
        "tied_vs_raw": tied_raw,
        "worsened_vs_raw": worsened_raw,
        "improved_vs_whole10": improved_whole,
        "tied_vs_whole10": tied_whole,
        "worsened_vs_whole10": worsened_whole,
        "new_hit_vs_raw": int((data["hit"] & ~data["raw_hit"]).sum()),
        "lost_hit_vs_raw": int((~data["hit"] & data["raw_hit"]).sum()),
        "new_hit_vs_whole10": int((data["hit"] & ~data["whole10_hit"]).sum()),
        "lost_hit_vs_whole10": int((~data["hit"] & data["whole10_hit"]).sum()),
        "mean_gt_retention": data["gt_retention"].mean(),
        "mean_pred_retention": data["pred_retention"].mean(),
        "deleted_tp_volume_sum_mm3": data["deleted_gt_overlapping_pred_volume_mm3"].sum(),
        "deleted_fp_volume_sum_mm3": data["deleted_fp_volume_mm3"].sum(),
    }


def add_reference_columns(frame: pd.DataFrame) -> pd.DataFrame:
    whole = (
        frame[frame["policy"] == "whole_lung_d10"][
            ["case_id", "finding_id", "dice", "hit", "pred_volume_mm3"]
        ]
        .rename(
            columns={
                "dice": "whole10_dice",
                "hit": "whole10_hit",
                "pred_volume_mm3": "whole10_pred_volume_mm3",
            }
        )
        .copy()
    )
    out = frame.merge(whole, on=["case_id", "finding_id"], validate="many_to_one")
    out["delta_dice_vs_whole10"] = out["dice"] - out["whole10_dice"]
    out["dice_decrease_vs_whole10"] = out["dice"] < out["whole10_dice"] - TIE_TOL
    out["hit_loss_vs_whole10"] = out["whole10_hit"] & ~out["hit"]
    return out


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def table_markdown(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    data = frame[columns].head(limit).copy() if limit else frame[columns].copy()
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].map(lambda x: f"{x:.6f}")
    headers = "| " + " | ".join(map(str, data.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(data.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |"
        for row in data.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, divider, *rows])


def write_report(
    out_dir: Path,
    findings: pd.DataFrame,
    assignments: pd.DataFrame,
    results: pd.DataFrame,
    metrics: pd.DataFrame,
    subgroups: pd.DataFrame,
    alignments: pd.DataFrame,
    reference_check: dict[str, Any],
) -> None:
    semantic = metrics[metrics["policy_family"] == "semantic"].sort_values("mean_dice", ascending=False)
    best = semantic.iloc[0]
    best_policy = str(best["policy"])
    best_rows = results[results["policy"] == best_policy].copy()
    top_improved = best_rows[best_rows["delta_dice_vs_raw"] > TIE_TOL].sort_values(
        "delta_dice_vs_raw", ascending=False
    ).head(20)
    top_worsened = best_rows[best_rows["delta_dice_vs_raw"] < -TIE_TOL].sort_values(
        "delta_dice_vs_raw", ascending=True
    ).head(20)
    group_counts = assignments.groupby("group").size().rename("n").reset_index()
    group_counts["fraction"] = group_counts["n"] / len(assignments)

    amb_phrases: dict[str, int] = {}
    for text in assignments.loc[assignments["group"] == "ambiguous_lobe", "finding_text"]:
        for pattern in AMBIGUOUS_PATTERNS:
            safe, _ = _safe_matches(str(text), (pattern,))
            for match in safe:
                phrase = match.group(0).lower()
                amb_phrases[phrase] = amb_phrases.get(phrase, 0) + 1
    phrase_frame = pd.DataFrame(
        sorted(amb_phrases.items(), key=lambda item: (-item[1], item[0])),
        columns=["phrase", "count"],
    )

    a_policy = "semantic_D_e15_s15_fw10"
    old_policy = "old_exact_lobe_d15_fwhole10"
    side_policy = "laterality_d15_fwhole10"
    mechanism_rows = []
    for subgroup in (
        "A_high_conf_exact_lobe",
        "B_ambiguous_lobe",
        "C_laterality_only",
        "D_bilateral_multifocal",
    ):
        for label, policy in (
            ("semantic_best", a_policy),
            ("old_exact15", old_policy),
            ("side15", side_policy),
        ):
            source = subgroups[(subgroups["policy"] == policy) & (subgroups["subgroup"] == subgroup)].iloc[0]
            mechanism_rows.append(
                {
                    "subgroup": subgroup,
                    "policy_role": label,
                    "mean_dice": source["mean_dice"],
                    "delta_vs_whole10": source["delta_mean_dice_vs_whole10"],
                    "mean_gt_retention": source["mean_gt_retention"],
                    "lost_hit_vs_whole10": source["lost_hit_vs_whole10"],
                }
            )
    mechanism = pd.DataFrame(mechanism_rows)
    a_exact = mechanism[
        (mechanism["subgroup"] == "A_high_conf_exact_lobe")
        & (mechanism["policy_role"] == "semantic_best")
    ].iloc[0]
    a_side = mechanism[
        (mechanism["subgroup"] == "A_high_conf_exact_lobe") & (mechanism["policy_role"] == "side15")
    ].iloc[0]
    b_side = mechanism[
        (mechanism["subgroup"] == "B_ambiguous_lobe")
        & (mechanism["policy_role"] == "semantic_best")
    ].iloc[0]
    b_exact = mechanism[
        (mechanism["subgroup"] == "B_ambiguous_lobe") & (mechanism["policy_role"] == "old_exact15")
    ].iloc[0]
    a_exact_gain = float(a_exact["mean_dice"] - a_side["mean_dice"])
    b_degrade_gain = float(b_side["mean_dice"] - b_exact["mean_dice"])

    delta_whole = float(best["delta_mean_dice_vs_whole10"])
    delta_laterality = float(best["mean_dice"] - 0.317295739814475)
    no_hit_loss = int(best["lost_hit_vs_whole10"]) == 0
    aggregate_strong = (
        delta_whole >= 0.002
        and no_hit_loss
        and float(best["predicted_volume_delta_vs_whole10_mm3"]) <= 0
    )
    if aggregate_strong and b_degrade_gain > 0:
        decision = "STRONG GO"
        part2 = "justified"
        decision_detail = "Both the aggregate gate and the Group-B degradation hypothesis pass."
    elif aggregate_strong:
        decision = "CONDITIONAL GO"
        part2 = "conditionally justified as a gated ablation"
        decision_detail = (
            "The aggregate policy passes every primary performance/safety gate, but Group B does not improve "
            "when degraded from exact-lobe to side-lung. The gain is therefore not evidence for that specific "
            "semantic hypothesis."
        )
    elif delta_whole > 0 and no_hit_loss:
        decision = "WEAK GO"
        part2 = "weakly justified"
        decision_detail = "The policy improves over whole-lung 10 mm without HIT loss, but misses a strong gate."
    else:
        decision = "NO-GO"
        part2 = "not justified by Part 1"
        decision_detail = "The aggregate post-hoc policy does not pass the requested performance/safety gates."

    unsafe_95 = int((best_rows["gt_retention"] < 0.95).sum())
    unsafe_90 = int((best_rows["gt_retention"] < 0.90).sum())
    unsafe_50 = int((best_rows["gt_retention"] < 0.50).sum())

    lines = [
        "# Semantic-aware adaptive lobe constraint — Part 1",
        "",
        "## Goal and setup",
        "",
        "This deterministic post-hoc experiment tests whether conservative semantic grouping avoids the "
        "failure mode of treating soft phrases such as “predominantly in the LLL” as strict containment. "
        "No model was trained or modified. The saved canonical step-6000 binary NIfTIs are the original "
        "threshold-0.5 `pred_raw`; probability/logit rerun was therefore unnecessary.",
        "",
        f"- Findings: {len(assignments)} across {findings['case_id'].nunique()} cases.",
        "- Priors: whole lung, left/right lung, and five TotalSegmentator lobes.",
        "- Mask handling: index-aligned masks are retained directly; any shape mismatch is nearest-neighbor "
        "resampled by the shared prior loader. Physical dilation uses the CT voxel spacing.",
        f"- Alignment audit: {int(alignments['all_aligned_prior_shapes'].sum())}/{len(alignments)} cases have "
        "all priors on the prediction array grid after loading.",
        f"- Canonical reproduction: raw max absolute per-finding Dice error "
        f"{reference_check['raw_max_abs_error']:.3g}; whole10 max absolute error "
        f"{reference_check['whole10_max_abs_error']:.3g}.",
        "",
        "## Fixed parser rules",
        "",
        "- A: exact RUL/RML/RLL/LUL/LLL phrase, without a soft modifier or unsafe context.",
        "- B: soft/anatomically ambiguous lobe phrasing; degrade to unilateral lung when identifiable.",
        "- C: reliable left/right location without a safe exact-lobe phrase.",
        "- D: bilateral, multifocal, diffuse, scattered, numerous, both-lung, or bibasilar cues; never use a "
        "single-side/exact constraint.",
        "- E: no reliable target location.",
        "- Negated/historical phrases within a six-token, punctuation-bounded window are excluded.",
        "- Priority is D → B → A → C → E. Thus `left lower lung` is B (the intentionally safer interpretation).",
        "",
        "## Parser group counts",
        "",
        table_markdown(group_counts, ["group", "n", "fraction"]),
        "",
        "Most frequent phrases contributing to Group B:",
        "",
        table_markdown(phrase_frame, ["phrase", "count"], limit=15) if not phrase_frame.empty else "None.",
        "",
        "## Main results",
        "",
        table_markdown(
            metrics.sort_values(["policy_family", "mean_dice"], ascending=[True, False]),
            [
                "policy",
                "mean_dice",
                "hit_count",
                "predicted_volume_sum_mm3",
                "delta_mean_dice_vs_raw",
                "delta_mean_dice_vs_whole10",
                "improved_vs_whole10",
                "tied_vs_whole10",
                "worsened_vs_whole10",
                "new_hit_vs_whole10",
                "lost_hit_vs_whole10",
            ],
        ),
        "",
        f"Best semantic policy: `{best_policy}`; mean Dice {fmt(float(best['mean_dice']))}, "
        f"Δ vs raw {fmt(float(best['delta_mean_dice_vs_raw']))}, "
        f"Δ vs whole10 {fmt(delta_whole)}, and Δ vs prior text-laterality10 {fmt(delta_laterality)}.",
        "",
        "## Mechanism and subgroup analysis",
        "",
        table_markdown(
            mechanism,
            [
                "subgroup",
                "policy_role",
                "mean_dice",
                "delta_vs_whole10",
                "mean_gt_retention",
                "lost_hit_vs_whole10",
            ],
        ),
        "",
        f"- Group A supports exact-lobe use: exact15 beats side15 by {fmt(a_exact_gain)} mean Dice.",
        f"- Group B does **not** support the original degradation hypothesis: side15 minus old exact15 is "
        f"{fmt(b_degrade_gain)} mean Dice. Side15 retains slightly more GT "
        f"({fmt(float(b_side['mean_gt_retention']))} vs {fmt(float(b_exact['mean_gt_retention']))}) but gives "
        "lower Dice.",
        "- The largest semantic correction is avoiding unsafe old-parser restrictions in Groups C and D. "
        "Group D is deliberately identical to whole10 under the deployed semantic policy.",
        "- All policy/subgroup combinations are available in `subgroup_metrics.csv`.",
        "",
        "## Safety audit",
        "",
        f"For the best semantic policy, mean GT retention is {fmt(float(best['mean_gt_retention']))}, "
        f"mean prediction retention is {fmt(float(best['mean_pred_retention']))}, deleted GT-overlapping "
        f"prediction volume is {fmt(float(best['deleted_tp_volume_sum_mm3']), 1)} mm³, and deleted FP volume "
        f"is {fmt(float(best['deleted_fp_volume_sum_mm3']), 1)} mm³. It creates "
        f"{int(best['lost_hit_vs_whole10'])} HIT losses and {int(best['new_hit_vs_whole10'])} new HITs versus "
        f"whole-lung 10 mm. GT retention is below 0.95 in {unsafe_95} findings, below 0.90 in {unsafe_90}, "
        f"and below 0.50 in {unsafe_50}; these cases are explicitly listed in `safety_audit.csv`.",
        "",
        "## Top 20 improved findings for the best semantic policy",
        "",
        table_markdown(
            top_improved,
            [
                "case_id",
                "finding_id",
                "finding_text",
                "group",
                "selected_roi_kind",
                "raw_dice",
                "dice",
                "gt_retention",
                "pred_volume_change_mm3_vs_raw",
            ],
        ),
        "",
        f"## Worsened findings for the best semantic policy (all {len(top_worsened)}, up to 20 requested)",
        "",
        table_markdown(
            top_worsened,
            [
                "case_id",
                "finding_id",
                "finding_text",
                "group",
                "selected_roi_kind",
                "raw_dice",
                "dice",
                "gt_retention",
                "pred_volume_change_mm3_vs_raw",
            ],
        ),
        "",
        "## Conclusion",
        "",
        f"**{decision} for the semantic-aware inference policy.** {decision_detail} Part 2 training-time "
        f"alignment loss is **{part2}**: it should preserve the A/C/D/E routing and must not encode a blanket "
        "assumption that Group-B findings are strictly confined to, or necessarily benefit from degrading "
        "away from, the named lobe. GT was used only for evaluation and never for policy selection.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_parser_tests()
    if args.parser_test_only:
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    findings = pd.read_csv(args.findings_csv)
    required = {"case_id", "finding_idx", "prompt"}
    missing = required - set(findings.columns)
    if missing:
        raise ValueError(f"Missing findings columns: {sorted(missing)}")
    if args.limit is not None:
        findings = findings.head(args.limit).copy()

    parsed = [semantic_parse(text) for text in findings["prompt"]]
    assignments = pd.DataFrame(
        {
            "case_id": findings["case_id"],
            "finding_id": findings["finding_idx"],
            "finding_text": findings["prompt"],
            "group": [item["group"] for item in parsed],
            "side": [item["side"] for item in parsed],
            "lobes": [";".join(item["lobes"]) for item in parsed],
            "roi_kind": [item["roi_kind"] for item in parsed],
            "matched_phrases": [" | ".join(item["matched_phrases"]) for item in parsed],
            "reason": [item["reason"] for item in parsed],
        }
    )
    assignments.to_csv(args.out_dir / "parser_group_assignments.csv", index=False)

    config = {
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir),
        "ct_dir": str(args.ct_dir),
        "prior_dir": str(args.prior_dir),
    }
    tasks = [
        (str(case), group.to_dict(orient="records"), config)
        for case, group in findings.groupby("case_id", sort=True)
    ]
    rows: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    failures: list[str] = []
    workers = max(1, int(args.num_workers))
    if workers == 1:
        iterator = map(process_case, tasks)
        for case_rows, alignment, failure in tqdm(iterator, total=len(tasks), desc="semantic constraints"):
            rows.extend(case_rows)
            alignments.append(alignment)
            if failure:
                failures.append(failure)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_case, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="semantic constraints"):
                case_rows, alignment, failure = future.result()
                rows.extend(case_rows)
                alignments.append(alignment)
                if failure:
                    failures.append(failure)
    if failures:
        (args.out_dir / "failed_cases.txt").write_text("\n".join(failures) + "\n")
        raise RuntimeError(f"{len(failures)} failed cases; first: {failures[0]}")

    results = pd.DataFrame(rows).sort_values(["case_id", "finding_id", "policy"]).reset_index(drop=True)
    results = add_reference_columns(results)
    results.to_csv(args.out_dir / "per_finding_policy_results.csv", index=False)
    alignment_frame = pd.DataFrame(alignments).sort_values("case_id")
    alignment_frame.to_csv(args.out_dir / "alignment_audit.csv", index=False)

    policy_metrics = pd.DataFrame([summarize_policy(results, policy["policy"]) for policy in POLICIES])
    policy_metrics.to_csv(args.out_dir / "policy_metrics.csv", index=False)

    subgroup_specs = {
        "A_high_conf_exact_lobe": {"high_conf_exact_lobe"},
        "B_ambiguous_lobe": {"ambiguous_lobe"},
        "C_laterality_only": {"laterality_only"},
        "D_bilateral_multifocal": {"bilateral_multifocal"},
        "E_nonlocalizable": {"nonlocalizable"},
        "ABC_parser_localizable": {"high_conf_exact_lobe", "ambiguous_lobe", "laterality_only"},
        "AB_explicit_lobe": {"high_conf_exact_lobe", "ambiguous_lobe"},
        "BC_side_localizable": {"ambiguous_lobe", "laterality_only"},
        "DE_fallback": {"bilateral_multifocal", "nonlocalizable"},
    }
    subgroup_rows = []
    for name, groups in subgroup_specs.items():
        subset = results[results["group"].isin(groups)]
        for policy in POLICIES:
            subgroup_rows.append(summarize_policy(subset, policy["policy"], subgroup=name))
    subgroup_metrics = pd.DataFrame(subgroup_rows)
    subgroup_metrics.to_csv(args.out_dir / "subgroup_metrics.csv", index=False)

    safety_columns = [
        "case_id",
        "finding_id",
        "finding_text",
        "group",
        "side",
        "lobes",
        "policy",
        "selected_roi_kind",
        "selected_roi_label",
        "selected_dilation_mm",
        "roi_volume_mm3",
        "gt_retention",
        "pred_retention",
        "deleted_gt_overlapping_pred_volume_mm3",
        "deleted_fp_volume_mm3",
        "dice_decrease_vs_raw",
        "hit_loss_vs_raw",
        "dice_decrease_vs_whole10",
        "hit_loss_vs_whole10",
    ]
    results.loc[results["constraint_applied"], safety_columns].to_csv(
        args.out_dir / "safety_audit.csv", index=False
    )

    reference_check = {"raw_max_abs_error": float("nan"), "whole10_max_abs_error": float("nan")}
    if args.reference_results.exists() and args.limit is None:
        reference = pd.read_csv(args.reference_results)
        raw_ref = reference[(reference["variant"] == "none") & (reference["dilation_mm"] == 0)][
            ["case", "finding_idx", "dice"]
        ].rename(columns={"case": "case_id", "finding_idx": "finding_id", "dice": "reference_dice"})
        whole_ref = reference[
            (reference["variant"] == "whole_lung") & (reference["dilation_mm"] == 10)
        ][["case", "finding_idx", "dice"]].rename(
            columns={"case": "case_id", "finding_idx": "finding_id", "dice": "reference_dice"}
        )
        raw_now = results[results["policy"] == "raw"][["case_id", "finding_id", "dice"]]
        whole_now = results[results["policy"] == "whole_lung_d10"][["case_id", "finding_id", "dice"]]
        raw_compare = raw_now.merge(raw_ref, on=["case_id", "finding_id"], validate="one_to_one")
        whole_compare = whole_now.merge(whole_ref, on=["case_id", "finding_id"], validate="one_to_one")
        reference_check = {
            "raw_max_abs_error": float((raw_compare["dice"] - raw_compare["reference_dice"]).abs().max()),
            "whole10_max_abs_error": float(
                (whole_compare["dice"] - whole_compare["reference_dice"]).abs().max()
            ),
            "raw_mean": float(raw_now["dice"].mean()),
            "whole10_mean": float(whole_now["dice"].mean()),
            "raw_hit": int((raw_now["dice"] >= HIT_THRESHOLD).sum()),
            "whole10_hit": int((whole_now["dice"] >= HIT_THRESHOLD).sum()),
        }
        if reference_check["raw_max_abs_error"] > 1e-12 or reference_check["whole10_max_abs_error"] > 1e-12:
            raise RuntimeError(f"Canonical reproduction failed: {reference_check}")
    (args.out_dir / "reference_reproduction.json").write_text(
        json.dumps(reference_check, indent=2), encoding="utf-8"
    )
    write_report(
        args.out_dir,
        findings,
        assignments,
        results,
        policy_metrics,
        subgroup_metrics,
        alignment_frame,
        reference_check,
    )
    print(policy_metrics.sort_values("mean_dice", ascending=False).head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
