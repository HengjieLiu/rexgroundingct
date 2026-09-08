#!/usr/bin/env python3
"""CPU-only contract tests for SideExp003 fresh-cache scheduling and state."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fresh_cache


def candidate(rank: int) -> dict:
    return {
        "id": f"candidate_{rank:02d}",
        "rank": rank,
        "checkpoint": {"sha256": f"checkpoint_{rank:02d}"},
        "config": {"sha256": f"config_{rank:02d}"},
        "inference_contract_sha256": f"inference_{rank:02d}",
        "cache": {
            "id": "crop_zscore_native_v1",
            "manifest_sha256": "preprocessing",
        },
    }


def roster(size: int = 20) -> dict:
    value = {
        "schema_version": fresh_cache.ROSTER_SCHEMA_VERSION,
        "roster_id": "synthetic_top20",
        "created_at_utc": "2026-09-07T00:00:00Z",
        "catalog": {"sha256": "catalog", "schema_version_at_freeze": 1},
        "dataset": {
            "split": "val200",
            "path": "val200.json",
            "sha256": fresh_cache.VAL200_SHA256,
            "cases": 200,
            "findings": 381,
        },
        "test300": {
            "status": "deferred",
            "metric_status": "not_available_withheld_labels",
        },
        "container": {
            "image": fresh_cache.EXPECTED_CONTAINER_IMAGE,
            "image_id": fresh_cache.EXPECTED_CONTAINER_IMAGE_ID,
        },
        "source_bundle": {"sha256": "source"},
        "reuse_policy": "fresh_only",
        "N": size,
        "candidates": [candidate(index) for index in range(1, size + 1)],
    }
    value["roster_sha256"] = fresh_cache.json_sha256(value)
    return value


class FreshScheduleTests(unittest.TestCase):
    def test_top20_is_five_contiguous_four_gpu_waves(self) -> None:
        job = fresh_cache.build_job(roster(), job_id="j001", wave_size=4)
        self.assertEqual(len(job["waves"]), 5)
        self.assertEqual(
            [(value["rank"], value["wave"], value["gpu"]) for value in job["candidates"]],
            [
                (rank, (rank - 1) // 4 + 1, (rank - 1) % 4)
                for rank in range(1, 21)
            ],
        )
        self.assertEqual(job["reuse_policy"], "fresh_only")
        self.assertEqual(fresh_cache.validate_job(job), [])

    def test_cache_key_changes_for_each_checkpoint(self) -> None:
        value = roster(size=2)
        first = fresh_cache.make_cache_key(value["candidates"][0], value)
        second = fresh_cache.make_cache_key(value["candidates"][1], value)
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("v2_"))

    def test_audited_continuation_preserves_original_ranks_and_new_keys(self) -> None:
        parent = fresh_cache.build_job(roster(), job_id="j001", wave_size=4)
        current_bundle = {"sha256": "source_v2", "files": []}
        audit = {
            "audit_sha256": "audit",
            "parent_source_bundle_sha256": "source",
            "new_source_bundle_sha256": "source_v2",
        }
        continuation = fresh_cache.build_continuation_job(
            parent,
            job_id="j002",
            start_rank=9,
            current_source_bundle=current_bundle,
            source_drift_audit=audit,
        )

        self.assertEqual(
            [value["rank"] for value in continuation["candidates"]],
            list(range(9, 21)),
        )
        self.assertEqual(
            [value["wave"] for value in continuation["candidates"]],
            [3] * 4 + [4] * 4 + [5] * 4,
        )
        self.assertEqual(
            [value["gpu"] for value in continuation["candidates"]],
            [0, 1, 2, 3] * 3,
        )
        self.assertNotEqual(
            continuation["candidates"][0]["cache_key"],
            parent["candidates"][8]["cache_key"],
        )
        self.assertEqual(
            continuation["candidates"][0]["parent_cache_key"],
            parent["candidates"][8]["cache_key"],
        )
        self.assertEqual(fresh_cache.validate_job(continuation), [])


class CatalogStateTests(unittest.TestCase):
    def catalog_candidate(self) -> dict:
        return {
            "candidate_id": "candidate_01",
            "config": {"path": None},
            "logit_cache": {
                "status": "strict_gate_passed",
                "candidate_id": "legacy_candidate",
                "physical_path": "/runtime/legacy_candidate",
                "logical_path": "/legacy/logits/legacy_candidate",
            },
        }

    def test_schema_migration_retains_legacy_version(self) -> None:
        catalog = fresh_cache.migrate_catalog(
            {"schema_version": 1, "candidates": [self.catalog_candidate()]}
        )
        migrated = catalog["candidates"][0]
        self.assertNotIn("logit_cache", migrated)
        versions = migrated["inference_artifacts"]["val200"]["versions"]
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["origin"], "legacy_sideexp002")
        self.assertEqual(versions[0]["status"], "strict_passed")
        self.assertEqual(
            migrated["inference_artifacts"]["test300"]["metric_status"],
            "not_available_withheld_labels",
        )

    def test_fresh_job_never_reuses_or_overwrites_legacy(self) -> None:
        value = roster(size=1)
        job = fresh_cache.build_job(value, job_id="j001", wave_size=1, gpus=[0])
        catalog = {
            "schema_version": 1,
            "candidates": [self.catalog_candidate()],
        }
        planned = fresh_cache.apply_job_to_catalog(catalog, job)
        split = planned["candidates"][0]["inference_artifacts"]["val200"]
        self.assertEqual(len(split["versions"]), 2)
        self.assertEqual(split["versions"][0]["origin"], "legacy_sideexp002")
        self.assertEqual(split["versions"][1]["origin"], "sideexp003")
        self.assertEqual(split["versions"][1]["status"], "queued")
        self.assertNotEqual(
            split["versions"][0]["cache_version_id"],
            split["versions"][1]["cache_version_id"],
        )

        progress = {
            "candidate_01": {
                "status": "strict_passed",
                "dtype": "float16",
                "cases": 200,
                "findings": 381,
            }
        }
        completed = fresh_cache.update_catalog_progress(planned, job, progress)
        final_split = completed["candidates"][0]["inference_artifacts"]["val200"]
        self.assertEqual(final_split["active_cache_version"], job["candidates"][0]["cache_version_id"])
        self.assertEqual(final_split["versions"][0]["origin"], "legacy_sideexp002")

    def test_failed_fresh_version_does_not_fall_back_to_legacy(self) -> None:
        value = roster(size=1)
        job = fresh_cache.build_job(value, job_id="j001", wave_size=1, gpus=[0])
        planned = fresh_cache.apply_job_to_catalog(
            {"schema_version": 1, "candidates": [self.catalog_candidate()]}, job
        )
        active_before = planned["candidates"][0]["inference_artifacts"]["val200"]["active_cache_version"]
        failed = fresh_cache.update_catalog_progress(
            planned,
            job,
            {"candidate_01": {"status": "failed", "error": "synthetic"}},
        )
        split = failed["candidates"][0]["inference_artifacts"]["val200"]
        self.assertEqual(split["active_cache_version"], active_before)
        self.assertEqual(split["status"], "failed")


if __name__ == "__main__":
    unittest.main()
