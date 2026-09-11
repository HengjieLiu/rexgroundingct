#!/usr/bin/env python3
"""CPU closeout: exact-run checks and explicitly retrospective matched-retention curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exp027_common import atomic_csv, atomic_json, digest, read_json, sha256
from fit_027_deletion import metrics, retention_threshold, summarize


def load_vectors(paths):
    result = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            result.append({"labels": data["labels"].astype(bool), "scores": data["scores"],
                           "base_logits": data["base_logits"], "total_gt": int(data["total_gt"])})
    return result


def compare(vectors, target):
    tp_scores = np.concatenate([v["scores"][v["labels"]] for v in vectors])
    tp_base = np.concatenate([v["base_logits"][v["labels"]] for v in vectors])
    delete_threshold = retention_threshold(tp_scores, target)
    base_threshold = retention_threshold(tp_base, target, False)
    editor = summarize([metrics(v["labels"], v["scores"] > delete_threshold, v["total_gt"]) for v in vectors])
    simple = summarize([metrics(v["labels"], v["base_logits"] < base_threshold, v["total_gt"]) for v in vectors])
    return {"target_tp_retention": target, "editor_threshold": delete_threshold,
            "base_logit_threshold": base_threshold, "editor": editor, "simple_threshold": simple,
            "interpretation": "Both thresholds use GT in this same diagnostic scope; retrospective tradeoff, not held-out performance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    manifest = read_json(root / "manifest.json")
    config = manifest["config"]
    assert read_json(root / "status.json")["phase"] == "pending_user_review"
    history = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
    assert [r["update"] for r in history] == list(range(1, config["updates"] + 1))
    assert [r["patch"] for r in history] == [manifest["patches"][i]["id"] for i in manifest["schedule"]]
    assert all(np.isfinite(r[k]) for r in history for k in ("loss", "grad_norm_before_clip", "step_seconds"))
    assert all(r["half"] == "A" for r in manifest["findings"])
    assert len({r["patient"] for r in manifest["findings"]}) == config["findings"]
    assert [int(p.parent.name.split("_")[-1]) for p in sorted((root / "patch_evaluations").glob("*/summary.json"))] == config["checkpoints"]
    for name, expected in manifest["provenance"]["code"].items():
        assert sha256(Path(__file__).parent / name) == expected, name
    for receipt in manifest["cache_verification"]:
        for path, expected in receipt["files"].items():
            stat = Path(path).stat()
            assert {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected, path
    full = read_json(root / "full_volume/summary.json")
    assert full["findings_done"] == full["findings_total"] == config["findings"]
    assert {r["key"] for r in full["findings"]} == {r["key"] for r in manifest["findings"]}
    for row in full["findings"]:
        for policy in ("fixed", "patch_calibrated", "simple_threshold"):
            m = row[policy]
            assert m["fp"] <= m["base_fp"] and m["tp"] <= m["base_tp"]
            assert m["new_foreground"] == 0
    scopes = {"fitting_patches": sorted((root / "patch_evaluations/update_0200").glob("*.npz")),
              "full_training_volumes": [root / "full_volume" / digest(r["key"]) / "base_positive_scores.npz"
                                         for r in manifest["findings"]]}
    comparisons = {}
    for name, paths in scopes.items():
        vectors = load_vectors(paths)
        comparisons[name] = [compare(vectors, target) for target in (.99, .95, .90)]
    patch_history = []
    for path in sorted((root / "patch_evaluations").glob("*/summary.json")):
        saved = read_json(path)
        patch_history.append({"update": saved["update"], "all_patches": saved["metrics"],
                              "gt_nonempty_patches": summarize([r for r in saved["patches"] if r["total_gt"] > 0]),
                              "gt_empty_patches": summarize([r for r in saved["patches"] if r["total_gt"] == 0]),
                              "gt_empty_completely_cleared": sum(r["total_gt"] == 0 and r["fp"] == 0 for r in saved["patches"])})
    for update in config["checkpoints"]:
        state = torch.load(root / "checkpoints" / f"update_{update:04d}.pth", map_location="cpu", weights_only=False)
        assert state["update"] == state["sampling_cursor"] == update
        assert state["manifest_sha256"] == digest(manifest)
        assert state["precision"] == "fp32" and state["tf32"] is False
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in state["model"].values())
        assert state["optimizer"]["param_groups"][0]["lr"] == 1e-4
    context = []
    for record in manifest["findings"]:
        with np.load(root / "full_volume" / digest(record["key"]) / "base_positive_scores.npz") as volume:
            indices, scores, shape = volume["native_indices"], volume["scores"], volume["native_shape"]
            for patch in (r for r in manifest["patches"] if r["key"] == record["key"]):
                x = np.load(root / "patches" / patch["id"] / "x.npy", mmap_mode="r")
                coordinates = np.array(np.nonzero(x[1] >= 0)) + np.array(patch["starts"])[:, None]
                native_indices = np.ravel_multi_index(tuple(coordinates), shape)
                positions = np.searchsorted(indices, native_indices)
                np.testing.assert_array_equal(indices[positions], native_indices)
                blended = scores[positions]
                with np.load(root / "patch_evaluations/update_0200" / f"{patch['id']}.npz") as data:
                    isolated = data["scores"]
                    difference = np.abs(blended - isolated)
                    context.append({"key": record["key"], "patch": patch["id"], "base_positive_voxels": len(isolated),
                                    "mean_abs_score_change": float(difference.mean()), "max_abs_score_change": float(difference.max()),
                                    "decision_disagreement_fraction_at_0_5": float(np.mean((isolated > .5) != (blended > .5)))})
    steps = np.array([r["step_seconds"] for r in history])
    checks = {"status": "passed", "updates": len(history), "findings": len(manifest["findings"]),
              "patches": len(manifest["patches"]), "scope": "A_only_fitting_data",
              "source_code_and_cache_files_unchanged": True,
              "checkpoints": {p.name: sha256(p) for p in sorted((root / "checkpoints").glob("*.pth"))},
              "step_seconds": {"mean": float(steps.mean()), "p50": float(np.median(steps)),
                               "p95": float(np.quantile(steps, .95)), "max": float(steps.max())},
              "mean_loading_seconds": float(np.mean([r["loading_seconds"] for r in history])),
              "finite_gradients_and_losses": True, "precision": "fp32", "tf32": False}
    checks["analysis_source_sha256"] = sha256(__file__)
    atomic_json(root / "verification.json", checks)
    atomic_json(root / "matched_retention_comparison.json", comparisons)
    atomic_json(root / "tile_context_comparison.json", context)
    atomic_json(root / "patch_subgroup_history.json", patch_history)
    atomic_csv(root / "tile_context_comparison.csv", context)
    print(json.dumps({"checks": checks, "matched_retention": comparisons}, indent=2))


if __name__ == "__main__":
    main()
