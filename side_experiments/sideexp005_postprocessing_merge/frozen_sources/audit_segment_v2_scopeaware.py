#!/usr/bin/env python3
"""Scope-aware segment-v2 audit from existing masks and predictions.

This intentionally differs from the first fine-modifier audit: it builds one
complete prompt ROI per finding.  Compound segment names are intersections,
explicitly distinct locations are unions, and non-exclusive language is never
used for hard deletion.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from audit_fine_spatial_modifiers import (
    LOBE_FILES,
    LOBE_ORDER,
    build_proxy,
    dice_score,
    relevant_sides,
    safe_ratio,
    split_lobes,
    world_axis_info,
)
from evaluate_anatomy_prior_ablation import channel_first_mask
from evaluate_anatomy_prior_diagnostic import geometry_reference, load_prior


HIT_THRESHOLD = 0.1
POLICIES = ("raw", "whole10", "semantic_v1")
UNRESOLVED = {"paraspinal_or_costovertebral", "retrocardiac"}

# A lexical compound is one anatomical atom, so its component proxies are
# intersected.  Distinct atoms elsewhere in the finding are later unioned.
# name, lexical pattern, proxy components to intersect, discovery clusters
# consumed by the compound.  The final field prevents e.g. "inferior lingular"
# from being added again as a separate inferior-half atom.
COMPOUNDS: list[tuple[str, re.Pattern[str], tuple[str, ...], tuple[str, ...]]] = [
    ("anteromedial_basal", re.compile(r"\bantero\s*-?\s*medial\s+basal\b", re.I), ("basal", "anterior", "medial"), ("basal", "anterior", "medial")),
    ("posterobasal", re.compile(r"\b(?:postero\s*-?\s*basal|posterior\s+basal)\b", re.I), ("basal", "posterior"), ("basal", "posterior")),
    ("anterobasal", re.compile(r"\b(?:antero\s*-?\s*basal|anterior\s+basal)\b", re.I), ("basal", "anterior"), ("basal", "anterior")),
    ("laterobasal", re.compile(r"\b(?:latero\s*-?\s*basal|lateral\s+basal)\b", re.I), ("basal", "lateral"), ("basal", "lateral")),
    ("mediobasal", re.compile(r"\b(?:medio\s*-?\s*basal|medial\s+basal)\b", re.I), ("basal", "medial"), ("basal", "medial")),
    ("apicoposterior", re.compile(r"\bapico\s*-?\s*posterior\b", re.I), ("apical", "posterior"), ("apical", "posterior")),
    ("inferior_lingular", re.compile(r"\b(?:inferior\s+lingular|lingular\s+inferior)\b", re.I), ("lingular",), ("lingular", "inferior_within_base")),
    ("dependent_basal", re.compile(r"\b(?:dependent\s+basal|basal\s+dependent)\b", re.I), ("dependent", "basal"), ("dependent", "basal")),
    ("anteromedial", re.compile(r"\bantero\s*-?\s*medial\b", re.I), ("anterior", "medial"), ("anterior", "medial")),
]

SIMPLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "apical": re.compile(r"\b(?:biapical|apices?|apical)\b", re.I),
    "basal": re.compile(r"\b(?:basal|basilar|lung base|basal levels?)\b", re.I),
    "diaphragmatic": re.compile(r"\b(?:juxtadiaphragmatic|diaphragmatic|adjacent to the diaphragm)\b", re.I),
    "subpleural_peripheral": re.compile(r"\b(?:subpleural|peripheral(?:ly)?|pleural-based|pleural surface)\b", re.I),
    "central": re.compile(r"\b(?:central(?:ly)?|central zones?|central levels?|peri-?hilar|hilar)\b", re.I),
    "anterior": re.compile(r"\banterior\b", re.I),
    "posterior": re.compile(r"\bposterior\b", re.I),
    "medial": re.compile(r"\bmedial\b", re.I),
    "lateral": re.compile(r"\blateral\b", re.I),
    "fissural": re.compile(r"\b(?:peri-?fissural|fissure-based|fissural|fissures?)\b", re.I),
    "lingular": re.compile(r"\b(?:lingula|lingular)\b", re.I),
    "dependent": re.compile(r"\b(?:dependent|dependently)\b", re.I),
    "paramediastinal": re.compile(r"\b(?:paramediastinal|paramediastinum)\b", re.I),
    "superior_within_base": re.compile(r"\b(?:superior|uppermost)\b", re.I),
    "inferior_within_base": re.compile(r"\b(?:inferior|lowermost)\b", re.I),
    "paraspinal_or_costovertebral": re.compile(r"\b(?:paraspinal|paravertebral|costovertebral|adjacent to osteophytes)\b", re.I),
    "retrocardiac": re.compile(r"\b(?:retrocardiac|paracardiac)\b", re.I),
}

# These expressions locate an example or dominant component, not the complete
# target extent.  They are retained for analysis but prohibited from hard ROI
# clipping.  "including" is conservatively treated as non-exhaustive.
NONEXCLUSIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    "predominantly": re.compile(r"\b(?:predominantly|primarily|mainly)\b", re.I),
    "most_prominent": re.compile(r"\b(?:most prominent|more prominent|greatest)\b", re.I),
    "largest_anchor": re.compile(r"\blargest\b.*\b(?:in|at|within)\b", re.I),
    "including_nonexhaustive": re.compile(r"\b(?:including|includes?|among)\b", re.I),
    "example_language": re.compile(r"\b(?:such as|for example|one of)\b", re.I),
}

DECORATOR_CLUSTERS = {
    "subpleural_peripheral", "central", "fissural", "dependent",
    "diaphragmatic", "paramediastinal",
}

STRONG_LIST_CONNECTOR = re.compile(r"(?:;|\band\b|\bor\b)", re.I)
COARSE_LOCATION = re.compile(
    r"\b(?:both\s+lungs?|bilateral(?:ly)?|(?:left|right)\s+(?:upper|middle|lower)\s+lobe|"
    r"(?:upper|middle|lower)\s+lobe\s+of\s+the\s+(?:left|right)\s+lung)\b",
    re.I,
)
CLAUSE_SPLIT = re.compile(r"\s*(?:,|;|\band\b)\s*", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--ct-dir", type=Path, required=True)
    p.add_argument("--prior-dir", type=Path, required=True)
    p.add_argument("--modifier-assignments", type=Path, required=True)
    p.add_argument(
        "--full-parser-assignments",
        type=Path,
        default=None,
        help=(
            "Optional one-row-per-finding (or one-row-per-finding-per-policy) "
            "semantic-v1 assignment table. Findings absent from the fine-modifier "
            "table are included with an empty modifier set, allowing full-cohort "
            "raw/whole10/semantic-v1 reproduction."
        ),
    )
    p.add_argument("--dataset-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    return p.parse_args()


def dataset_entity_counts(path: Path) -> dict[tuple[str, int], int]:
    payload = json.loads(path.read_text())
    found: dict[tuple[str, int], int] = {}

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if "name" in value and "findings" in value:
                for key in value.get("findings", {}):
                    found[(str(value["name"]), int(key))] = int(value.get("entity_counts", {}).get(key, 1))
            for item in value.values():
                if isinstance(item, (dict, list)):
                    walk(item)

    walk(payload)
    return found


def scope_plan(text: str, available_clusters: set[str]) -> dict[str, Any]:
    residual = str(text)
    atoms: list[dict[str, Any]] = []
    covered_clusters: set[str] = set()
    accepted_spans: list[tuple[int, int]] = []
    compound_matches: list[tuple[int, int, str, tuple[str, ...], tuple[str, ...], str]] = []
    for name, pattern, components, consumes in COMPOUNDS:
        for match in pattern.finditer(text):
            compound_matches.append((*match.span(), name, components, consumes, match.group(0)))
    # Longest-match-first for overlapping lexical compounds such as
    # "anteromedial basal" versus "anteromedial".
    compound_matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    for start, end, name, components, consumes, source in compound_matches:
        if any(start < prior_end and prior_start < end for prior_start, prior_end in accepted_spans):
            continue
        atoms.append({
            "name": name, "components": list(components),
            "consumed_clusters": list(consumes), "source": source, "span": [start, end],
        })
        covered_clusters.update(consumes)
        accepted_spans.append((start, end))
    if accepted_spans:
        chars = list(residual)
        for start, end in accepted_spans:
            chars[start:end] = " " * (end - start)
        residual = "".join(chars)

    decorator_items: list[dict[str, Any]] = []
    for cluster in sorted(available_clusters):
        if cluster in UNRESOLVED:
            continue
        pattern = SIMPLE_PATTERNS.get(cluster)
        # If the cluster occurred only as part of a lexical compound, do not
        # add it again as a separate union atom.
        if cluster in covered_clusters and (pattern is None or not pattern.search(residual)):
            continue
        match = pattern.search(residual) if pattern is not None else None
        if pattern is not None and match is None:
            continue
        if cluster in DECORATOR_CLUSTERS:
            span = list(match.span()) if match is not None else None
            source = match.group(0) if match is not None else cluster
            decorator_items.append({
                "name": cluster, "components": [cluster], "consumed_clusters": [cluster],
                "source": source, "span": span, "kind": "decorator",
            })
        else:
            span = list(match.span()) if match is not None else None
            source = match.group(0) if match is not None else cluster
            atoms.append({
                "name": cluster, "components": [cluster], "consumed_clusters": [cluster],
                "source": source, "span": span, "kind": "atom",
            })

    # A discovery label that cannot be recovered from the text by the v2
    # lexical rules is audited as a parser disagreement, not silently turned
    # into a hard ROI.  This catches errors such as v1 mapping "peripherally
    # located" to lateral in addition to peripheral.
    represented = (
        {component for atom in atoms for component in atom["components"]}
        | {item["name"] for item in decorator_items}
    )
    unmatched_discovery = []
    for cluster in sorted(available_clusters - represented - UNRESOLVED):
        if cluster not in covered_clusters:
            unmatched_discovery.append(cluster)

    # Deduplicate identical atoms while preserving distinct compound atoms.
    unique_atoms, seen = [], set()
    for atom in atoms:
        atom.setdefault("kind", "atom")
        key = (atom["name"], tuple(atom["components"]))
        if key not in seen:
            unique_atoms.append(atom)
            seen.add(key)
    unique_decorators, seen_decorators = [], set()
    for item in decorator_items:
        key = (item["name"], tuple(item["components"]))
        if key not in seen_decorators:
            unique_decorators.append(item)
            seen_decorators.add(key)

    # Nearby descriptors form one conjunctive location; explicitly enumerated
    # locations form separate groups whose masks are unioned.  This handles
    # both "fissural ... in the superior segment" (intersection) and
    # "centrally and peripherally" (union).
    positioned = sorted(
        unique_atoms + unique_decorators,
        key=lambda item: item["span"][0] if item["span"] is not None else 10**9,
    )
    atom_groups: list[list[dict[str, Any]]] = []
    for atom in positioned:
        if not atom_groups:
            atom_groups.append([atom])
            continue
        previous = atom_groups[-1][-1]
        if previous["span"] is None or atom["span"] is None:
            atom_groups.append([atom])
            continue
        between = text[previous["span"][1]:atom["span"][0]]
        # A comma is a list separator only after a primary location has
        # already appeared.  This avoids treating measurement punctuation in
        # "peripheral nodule, 5 mm, in the lateral segment" as a scope union.
        is_list = bool(STRONG_LIST_CONNECTOR.search(between)) or (
            "," in between and any(item["kind"] == "atom" for item in atom_groups[-1])
        )
        if is_list:
            atom_groups.append([atom])
        else:
            atom_groups[-1].append(atom)

    # A leading distribution/decorator attached to the first listed location
    # scopes the complete following list: "subpleural opacities in A, B and C".
    # A trailing decorator remains local to its own group.
    if len(atom_groups) > 1:
        first_primary_position = next(
            (index for index, item in enumerate(atom_groups[0]) if item["kind"] == "atom"),
            None,
        )
        if first_primary_position is not None:
            leading = [item for item in atom_groups[0][:first_primary_position] if item["kind"] == "decorator"]
            for group in atom_groups[1:]:
                if leading and any(item["kind"] == "atom" for item in group):
                    present = {item["name"] for item in group}
                    group[:0] = [item for item in leading if item["name"] not in present]

    flags = [name for name, pattern in NONEXCLUSIVE_PATTERNS.items() if pattern.search(text)]
    fine_patterns = [pattern for pattern in SIMPLE_PATTERNS.values()] + [item[1] for item in COMPOUNDS]
    unmodified_coarse_clauses = [
        clause.strip()
        for clause in CLAUSE_SPLIT.split(text)
        if COARSE_LOCATION.search(clause)
        and not any(pattern.search(clause) for pattern in fine_patterns)
    ]
    unresolved = sorted(available_clusters & UNRESOLVED)
    known_scope = bool(atom_groups)
    if unresolved and not known_scope:
        mode = "unresolved_location"
        hard_eligible = False
    elif flags:
        mode = "soft_only_nonexclusive"
        hard_eligible = False
    elif unmodified_coarse_clauses:
        mode = "soft_only_incomplete_scope"
        hard_eligible = False
    elif len(atom_groups) > 1:
        mode = "hard_union"
        hard_eligible = True
    elif any(len(group) > 1 for group in atom_groups):
        mode = "hard_intersection"
        hard_eligible = True
    elif known_scope:
        mode = "hard_single"
        hard_eligible = True
    else:
        mode = "unresolved_location"
        hard_eligible = False
    return {
        "scope_mode": mode,
        "hard_eligible": hard_eligible,
        "atoms": unique_atoms,
        "atom_groups": atom_groups,
        "decorators": sorted({item["name"] for item in unique_decorators}),
        "nonexclusive_flags": flags,
        "unresolved_clusters": unresolved,
        "unmatched_discovery_clusters": unmatched_discovery,
        "unmodified_coarse_clauses": unmodified_coarse_clauses,
    }


def combine_scope_proxy(
    plan: dict[str, Any], semantic_roi: np.ndarray, strict_base: np.ndarray,
    priors: dict[str, np.ndarray], axes: dict[int, tuple[int, np.ndarray, float]],
    sampling: tuple[float, float, float], sides: list[str], case_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not plan["hard_eligible"]:
        return semantic_roi.copy(), []
    details: list[dict[str, Any]] = []
    group_masks: list[np.ndarray] = []
    for group_index, group in enumerate(plan["atom_groups"]):
        masks_within_group: list[np.ndarray] = []
        for atom in group:
            component_masks: list[np.ndarray] = []
            component_details = []
            for cluster in atom["components"]:
                mask, name, params = build_proxy(
                    cluster, semantic_roi, strict_base, priors, axes, sampling, sides, case_cache
                )
                if mask is None:
                    continue
                component_masks.append(mask)
                component_details.append({"cluster": cluster, "proxy": name, "parameters": params})
            if component_masks:
                masks_within_group.append(np.logical_and.reduce(component_masks))
                details.append({**atom, "group": group_index, "operation": "component_intersection", "component_details": component_details})
        if masks_within_group:
            group_masks.append(np.logical_and.reduce(masks_within_group))
    scope = np.logical_or.reduce(group_masks) if group_masks else semantic_roi.copy()
    return scope, details


def process_case(case: str, finding_groups: list[dict[str, Any]], args: argparse.Namespace,
                 entities: dict[tuple[str, int], int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pred, pred_ref = channel_first_mask(args.pred_dir / case, 0.5, False)
    gt, _ = channel_first_mask(args.gt_dir / case, 0.5, False)
    if pred.shape != gt.shape:
        raise ValueError(f"{case}: pred {pred.shape} != GT {gt.shape}")
    ref = geometry_reference(case, pred_ref, args.ct_dir)
    spatial_shape = tuple(int(x) for x in pred.shape[1:])
    sampling = tuple(float(x) for x in nib.affines.voxel_sizes(ref.affine)[:3])
    voxel_volume = float(abs(np.linalg.det(ref.affine[:3, :3])))
    axes = world_axis_info(ref.affine, spatial_shape)
    priors = {
        "whole": load_prior(args.prior_dir, case, "whole_lung", ref),
        "whole10": load_prior(args.prior_dir, case, "lung_dilated_10mm", ref),
    }
    for lobe, filename in LOBE_FILES.items():
        priors[lobe] = load_prior(args.prior_dir, case, filename, ref)
    priors["left"] = priors["LUL"] | priors["LLL"]
    priors["right"] = priors["RUL"] | priors["RML"] | priors["RLL"]
    case_cache: dict[str, np.ndarray] = {}
    gt_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []

    for finding in finding_groups:
        finding_id = int(finding["finding_id"])
        if finding_id >= pred.shape[0]:
            raise IndexError(f"{case}: finding {finding_id} outside channels")
        group = str(finding["parser_group"])
        side = None if pd.isna(finding.get("parsed_side")) else str(finding.get("parsed_side"))
        lobes = split_lobes(finding.get("parsed_lobes"))
        exact_base = np.logical_or.reduce([priors[lobe] for lobe in lobes]) if lobes else None
        if group == "high_conf_exact_lobe" and exact_base is not None:
            strict_base = exact_base
            semantic_roi = ndimage.distance_transform_edt(~strict_base, sampling=sampling) <= 15.0
            base_kind, base_label, base_dilation = "exact_lobe", ";".join(lobes), 15
        elif group in {"high_conf_exact_lobe", "ambiguous_lobe", "laterality_only"} and side in {"left", "right"}:
            strict_base = priors[side]
            semantic_roi = ndimage.distance_transform_edt(~strict_base, sampling=sampling) <= 15.0
            base_kind, base_label, base_dilation = "side_lung", side, 15
        else:
            strict_base = priors["whole"]
            semantic_roi = priors["whole10"]
            base_kind, base_label, base_dilation = "whole_lung", "whole", 10
        plan = scope_plan(str(finding["finding_text"]), set(finding["modifier_clusters"]))
        active_clusters = sorted({
            cluster
            for group_items in plan["atom_groups"]
            for item in group_items
            for cluster in item.get("consumed_clusters", item["components"])
        })
        sides = relevant_sides(side, lobes)
        scope_roi, operation_details = combine_scope_proxy(
            plan, semantic_roi, strict_base, priors, axes, sampling, sides, case_cache
        )
        gt_mask = gt[finding_id]
        raw = pred[finding_id]
        gt_n = int(gt_mask.sum())
        inside_n = int((gt_mask & scope_roi).sum())
        if gt_n:
            center = ndimage.center_of_mass(gt_mask)
            center_index = tuple(min(max(int(round(x)), 0), spatial_shape[i] - 1) for i, x in enumerate(center))
            centroid_inside: bool | float = bool(scope_roi[center_index])
        else:
            centroid_inside = float("nan")
        entity_count = int(entities.get((case, finding_id), 1))
        singular_nodule = bool(re.search(r"\bnodule\b", str(finding["finding_text"]), re.I)) and not bool(re.search(r"\bnodules\b", str(finding["finding_text"]), re.I))
        common = {
            "case_id": case, "finding_id": finding_id, "finding_text": finding["finding_text"],
            "parser_group": group, "parsed_side": side, "parsed_lobes": ";".join(lobes),
            "modifier_clusters": ";".join(sorted(finding["modifier_clusters"])),
            "active_modifier_clusters": ";".join(active_clusters),
            "matched_raw_phrases": ";".join(sorted(finding["matched_raw_phrases"])),
            "scope_mode": plan["scope_mode"], "hard_eligible": bool(plan["hard_eligible"]),
            "scope_atoms": json.dumps(plan["atoms"], sort_keys=True),
            "scope_atom_groups": json.dumps(plan["atom_groups"], sort_keys=True),
            "scope_decorators": ";".join(plan["decorators"]),
            "scope_operations": json.dumps(operation_details, sort_keys=True),
            "nonexclusive_flags": ";".join(plan["nonexclusive_flags"]),
            "unresolved_clusters": ";".join(plan["unresolved_clusters"]),
            "unmatched_discovery_clusters": ";".join(plan["unmatched_discovery_clusters"]),
            "unmodified_coarse_clauses": ";".join(plan["unmodified_coarse_clauses"]),
            "base_roi_type": base_kind, "base_roi_label": base_label,
            "base_roi_dilation_mm": base_dilation, "entity_count": entity_count,
            "singular_nodule_multi_entity_flag": bool(singular_nodule and entity_count > 1),
            "roi_selection_uses_gt": False, "voxel_volume_mm3": voxel_volume,
        }
        gt_rows.append({
            **common,
            "gt_volume_total": gt_n * voxel_volume,
            "gt_volume_inside_scope_roi": inside_n * voxel_volume,
            "gt_retention_inside_scope_roi": safe_ratio(inside_n, gt_n),
            "gt_centroid_inside_scope_roi": centroid_inside,
            "gt_any_overlap_scope_roi": inside_n > 0,
            "scope_roi_volume": int(scope_roi.sum()) * voxel_volume,
            "base_roi_volume": int(semantic_roi.sum()) * voxel_volume,
            "scope_roi_fraction_of_base": safe_ratio(int(scope_roi.sum()), int(semantic_roi.sum())),
            "no_gt": gt_n == 0,
        })
        policies = {"raw": raw, "whole10": raw & priors["whole10"], "semantic_v1": raw & semantic_roi}
        for policy, policy_pred in policies.items():
            fp = policy_pred & ~gt_mask
            tp = policy_pred & gt_mask
            clipped = policy_pred & scope_roi
            pred_n, fp_n, tp_n = int(policy_pred.sum()), int(fp.sum()), int(tp.sum())
            fp_out = int((fp & ~scope_roi).sum())
            pred_out = int((policy_pred & ~scope_roi).sum())
            tp_in = int((tp & scope_roi).sum())
            clipped_n = int(clipped.sum())
            before, after = dice_score(policy_pred, gt_mask), dice_score(clipped, gt_mask)
            fp_rows.append({
                **common, "policy": policy,
                "prediction_volume": pred_n * voxel_volume,
                "fp_volume_total": fp_n * voxel_volume,
                "tp_volume_total": tp_n * voxel_volume,
                "fp_volume_outside_scope_roi": fp_out * voxel_volume,
                "prediction_volume_outside_scope_roi": pred_out * voxel_volume,
                "fp_outside_scope_roi_ratio": safe_ratio(fp_out, fp_n),
                "prediction_outside_scope_roi_ratio": safe_ratio(pred_out, pred_n),
                "tp_retention_inside_scope_roi": safe_ratio(tp_in, tp_n),
                "gt_retention_inside_scope_roi": safe_ratio(inside_n, gt_n),
                "dice_before_clipping": before, "dice_after_clipping": after,
                "dice_delta": after - before,
                "hit_before": before >= HIT_THRESHOLD, "hit_after": after >= HIT_THRESHOLD,
                "new_hit": before < HIT_THRESHOLD <= after,
                "lost_hit": before >= HIT_THRESHOLD > after,
                "clipped_prediction_volume": clipped_n * voxel_volume,
                "volume_ratio_after_before": safe_ratio(clipped_n, pred_n),
                "no_prediction": pred_n == 0, "no_fp": fp_n == 0, "no_gt": gt_n == 0,
            })
    quality = {
        "case_id": case, "spatial_shape": list(spatial_shape),
        "affine_axis_codes": "".join(str(x) for x in nib.aff2axcodes(ref.affine)),
        "max_off_axis_ratio": max(value[2] for value in axes.values()),
        "mask_alignment_pass": True, "gt_used_for_roi_selection": False,
        "findings": len(gt_rows), "policy_rows": len(fp_rows),
    }
    return gt_rows, fp_rows, quality


def main() -> int:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    assignments = pd.read_csv(args.modifier_assignments)
    grouped: list[dict[str, Any]] = []
    for (case, finding_id), data in assignments.groupby(["case_id", "finding_id"], sort=True):
        first = data.iloc[0]
        grouped.append({
            "case_id": case, "finding_id": int(finding_id), "finding_text": first["finding_text"],
            "parser_group": first["parser_group"], "parsed_side": first["parsed_side"],
            "parsed_lobes": first["parsed_lobes"],
            "modifier_clusters": sorted(set(data["modifier_cluster"].astype(str))),
            "matched_raw_phrases": sorted({p for value in data["matched_raw_phrases"].fillna("") for p in str(value).split(";") if p}),
        })
    if args.full_parser_assignments is not None:
        full = pd.read_csv(args.full_parser_assignments)
        if "policy" in full.columns:
            raw_rows = full[full["policy"].astype(str) == "raw"]
            full = raw_rows if len(raw_rows) else full
        full = full.drop_duplicates(["case_id", "finding_id"])
        existing = {(str(row["case_id"]), int(row["finding_id"])) for row in grouped}
        for _, row in full.iterrows():
            key = (str(row["case_id"]), int(row["finding_id"]))
            if key in existing:
                continue
            grouped.append({
                "case_id": key[0],
                "finding_id": key[1],
                "finding_text": row["finding_text"],
                "parser_group": row["parser_group"],
                "parsed_side": row.get("parsed_side"),
                "parsed_lobes": row.get("parsed_lobes"),
                "modifier_clusters": [],
                "matched_raw_phrases": [],
            })
    cases = sorted({row["case_id"] for row in grouped})
    selected = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_index]
    entities = dataset_entity_counts(args.dataset_json)
    gt_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    quality_rows = []
    for case in tqdm(selected, desc=f"segment-v2 shard {args.shard_index}"):
        findings = [row for row in grouped if row["case_id"] == case]
        case_gt, case_fp, quality = process_case(case, findings, args, entities)
        gt_rows.extend(case_gt); fp_rows.extend(case_fp); quality_rows.append(quality)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gt_rows).sort_values(["case_id", "finding_id"]).to_csv(args.output_dir / "segment_v2_scope_gt.csv", index=False)
    pd.DataFrame(fp_rows).sort_values(["case_id", "finding_id", "policy"]).to_csv(args.output_dir / "segment_v2_scope_fp.csv", index=False)
    payload = {
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "cases": len(selected), "findings": len(gt_rows), "policy_rows": len(fp_rows),
        "mask_alignment_failures": sum(not row["mask_alignment_pass"] for row in quality_rows),
        "gt_used_for_roi_selection": False, "cases_detail": quality_rows,
    }
    (args.output_dir / "quality_checks.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "cases_detail"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
