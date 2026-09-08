#!/usr/bin/env python3
"""CPU-only tests for the SideExp003 foundation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("sideexp003_hub", HERE / "hub.py")
assert SPEC is not None and SPEC.loader is not None
hub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def synthetic_dataset() -> dict:
    cases = []
    finding_number = 0
    categories = [category for category in hub.OFFICIAL_CATEGORY_ORDER if category != "2f"]
    for case_index in range(200):
        count = 2 if case_index < 181 else 1
        findings = {}
        category_map = {}
        for local_index in range(count):
            findings[str(local_index)] = f"finding {finding_number}"
            category_map[str(local_index)] = categories[finding_number % len(categories)]
            finding_number += 1
        name = f"case_{case_index:03d}.nii.gz"
        cases.append(
            {"name": name, "seg_path": name, "findings": findings, "categories": category_map}
        )
    assert finding_number == 381
    return {"test": cases}


def synthetic_evaluation(dataset: dict, *, delta: float = 0.0, reverse: bool = True) -> dict:
    cases = []
    dice_values = []
    source_cases = list(dataset["test"])
    if reverse:
        source_cases.reverse()
    for case in source_cases:
        findings = {}
        items = list(case["findings"].items())
        if reverse:
            items.reverse()
        for raw_index, _text in items:
            ordinal = int(case["name"].split("_")[1].split(".")[0]) + int(raw_index)
            dice = min(0.99, 0.05 + (ordinal % 10) / 20 + delta)
            findings[f"finding_{raw_index}"] = {
                "global_dice": dice,
                "global_hit": dice >= hub.HIT_THRESHOLD,
            }
            dice_values.append(dice)
        cases.append({"file": case["name"], "findings": findings})
    hits = sum(value >= hub.HIT_THRESHOLD for value in dice_values)
    return {
        "summary": {
            "params": {"global_only": True, "global_hit_thr": hub.HIT_THRESHOLD},
            "total_cases": 200,
            "total_findings": 381,
            "total_hits": hits,
            "total_misses": 381 - hits,
            "hit_rate": hits / 381,
            "mean_global_dice_per_finding": sum(dice_values) / 381,
        },
        "cases": cases,
    }


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_path = self.root / "val200.json"
        self.dataset = synthetic_dataset()
        write_json(self.dataset_path, self.dataset)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_source(
        self, *, run: str, checkpoint_bytes: bytes = b"same", delta: float = 0.0
    ) -> Path:
        arm = self.root / "runtime/001_test/runs" / run / "arm"
        checkpoint = arm / "model_epoch001/fold_0/checkpoint_final.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(checkpoint_bytes)
        evaluation = arm / "eval_epoch001_val200/eval/val_quick_global_eval.json"
        write_json(evaluation, synthetic_evaluation(self.dataset, delta=delta))
        return evaluation

    def test_unordered_alignment_and_official_categories(self) -> None:
        evaluation = self.make_source(run="run_a")
        dataset = hub.load_dataset_contract(self.dataset_path)
        metrics = hub.recompose_evaluator(evaluation, dataset)
        self.assertEqual(metrics["overall"]["findings"], 381)
        self.assertEqual(list(metrics["categories"]), list(hub.OFFICIAL_CATEGORY_ORDER))
        self.assertFalse(metrics["categories"]["2f"]["available"])

    def test_checkpoint_sha_deduplicates_consistent_evaluators(self) -> None:
        first = self.make_source(run="run_a")
        second = self.make_source(run="run_b")
        rows, dataset = hub.scan_catalog_candidates(
            [second, first],
            dataset_path=self.dataset_path,
            runtime_root=self.root / "runtime",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["evaluator_aliases"]), 2)
        self.assertEqual(rows[0]["eligibility"], "needs_review")
        catalog = hub.make_catalog(rows, dataset)
        self.assertEqual(
            hub.render_checkpoint_leaderboard(catalog),
            hub.render_checkpoint_leaderboard(catalog),
        )
        self.assertEqual(
            hub.render_subcategory_leaderboard(catalog),
            hub.render_subcategory_leaderboard(catalog),
        )

    def test_conflicting_duplicate_evaluator_stops(self) -> None:
        first = self.make_source(run="run_a")
        second = self.make_source(run="run_b", delta=0.01)
        with self.assertRaises(hub.CatalogConflict) as context:
            hub.scan_catalog_candidates(
                [first, second],
                dataset_path=self.dataset_path,
                runtime_root=self.root / "runtime",
            )
        override = next(iter(context.exception.stub["evaluator_overrides"].values()))
        self.assertIsNone(override["canonical_evaluator_path"])

    def test_reviewed_override_resolves_only_named_conflict_source(self) -> None:
        first = self.make_source(run="run_a")
        second = self.make_source(run="run_b", delta=0.01)
        checkpoint = second.parents[2] / "model_epoch001/fold_0/checkpoint_final.pth"
        checksum = hub.sha256_file(checkpoint)
        rows, _dataset = hub.scan_catalog_candidates(
            [first, second],
            dataset_path=self.dataset_path,
            runtime_root=self.root / "runtime",
            evaluator_overrides={
                checksum: {
                    "canonical_evaluator_path": str(second),
                    "reason": "reviewed synthetic conflict",
                    "reviewed_by": "unit-test",
                    "reviewed_at_utc": "2026-09-07T00:00:00Z",
                }
            },
        )
        self.assertEqual(rows[0]["canonical_evaluator"]["path"], str(second))
        self.assertIn(
            "conflicting_duplicate_evaluators_resolved_by_reviewed_override",
            rows[0]["warnings"],
        )

    def test_missing_threshold_is_needs_review(self) -> None:
        config = self.root / "config.json"
        write_json(config, {"validation": {}})
        provenance = hub.threshold_provenance(config)
        self.assertEqual(provenance["status"], "missing")
        self.assertIsNone(provenance["value"])

    def test_verified_threshold_provenance(self) -> None:
        config = self.root / "config.json"
        write_json(config, {"validation": {"primary_threshold": 0.5}})
        provenance = hub.threshold_provenance(config)
        self.assertEqual(provenance["status"], "verified")
        self.assertEqual(provenance["json_path"], "validation.primary_threshold")

    def test_non_200_evaluator_is_rejected(self) -> None:
        evaluation = self.make_source(run="run_a")
        value = json.loads(evaluation.read_text())
        value["cases"].pop()
        value["summary"]["total_cases"] = 199
        write_json(evaluation, value)
        dataset = hub.load_dataset_contract(self.dataset_path)
        with self.assertRaises(hub.HubError):
            hub.recompose_evaluator(evaluation, dataset)

    def test_cache_key_binds_every_contract_component(self) -> None:
        contract = {
            "dataset_sha256": "dataset",
            "checkpoint_sha256": "checkpoint",
            "inference_contract_sha256": "inference",
            "preprocessing_manifest_sha256": "preprocessing",
            "code_version": "commit",
            "storage_contract": {
                "layout": "FXYZ",
                "dtype": "float16",
                "clip": [-30, 30],
            },
        }
        baseline = hub.make_cache_key(**contract)
        self.assertEqual(baseline, hub.make_cache_key(**contract))
        for key in (
            "dataset_sha256", "checkpoint_sha256", "inference_contract_sha256",
            "preprocessing_manifest_sha256", "code_version",
        ):
            changed = dict(contract)
            changed[key] = contract[key] + "-changed"
            self.assertNotEqual(baseline, hub.make_cache_key(**changed))
        changed_storage = dict(contract)
        changed_storage["storage_contract"] = {**contract["storage_contract"], "dtype": "float32"}
        self.assertNotEqual(baseline, hub.make_cache_key(**changed_storage))


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "sideexp002/logits"
        self.destination = self.root / "sideexp003/cache/logits/legacy_sideexp002"
        for index in range(2):
            candidate = self.source / f"candidate_{index}"
            array_path = candidate / "cases" / f"case_{index}.npy"
            array_path.parent.mkdir(parents=True, exist_ok=True)
            array_path.write_bytes(b"npy" + bytes([index]))
            write_json(
                candidate / "export_manifest.json",
                {
                    "candidate_id": candidate.name,
                    "case_count": 1,
                    "finding_count": 1,
                    "dtype": "float16",
                    "status": "complete",
                    "cases": [{"array_path": str(array_path)}],
                },
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dry_run_does_not_move(self) -> None:
        result = hub.migrate_legacy_cache(self.source, self.destination, apply=False)
        self.assertEqual(result["status"], "dry_run")
        self.assertTrue(self.source.is_dir())
        self.assertFalse(self.source.is_symlink())
        self.assertFalse(self.destination.exists())

    def test_atomic_move_symlink_and_idempotent_audit(self) -> None:
        result = hub.migrate_legacy_cache(self.source, self.destination, apply=True)
        self.assertEqual(result["status"], "migrated")
        self.assertTrue(self.source.is_symlink())
        self.assertEqual(self.source.resolve(), self.destination.resolve())
        self.assertEqual(result["inventory"]["array_count"], 2)
        second = hub.migrate_legacy_cache(self.source, self.destination, apply=True)
        self.assertEqual(second["status"], "already_migrated")

    def test_failure_rolls_back_move(self) -> None:
        with self.assertRaises(hub.HubError):
            hub.migrate_legacy_cache(
                self.source, self.destination, apply=True, fail_after_move=True
            )
        self.assertTrue(self.source.is_dir())
        self.assertFalse(self.source.is_symlink())
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
