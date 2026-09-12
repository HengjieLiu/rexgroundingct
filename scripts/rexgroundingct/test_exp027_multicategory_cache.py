"""CPU tests of the separate multicategory cache and guarded launcher."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from exp027_common import REPO, atomic_json, lock, read_json, sha256
from exp027_data import native_to_original, original_to_native, save_npy, stored_logits
from exp027_multicategory_data import (DEFAULT_CONFIG, FindingStore, build_index, cache_contract,
    census, file_proof, finalize, finding_index, finding_records, immutable_json, shard_plan,
    summarize, validate_membership, verify_case, verify_validation_geometry)
from exp027_multicategory_report import report
from exp027_multicategory_worker import MemoryGuard, process_case, require_gpu
from run_027_multicategory_cache import dry_run


def fixture(root):
    root = Path(root)
    name = "train_10_a_1.nii.gz"
    shape = (5, 6, 7)
    config = read_json(DEFAULT_CONFIG)
    config["runtime"] = str(root)
    config["model"]["patch_size"] = [4] * 3
    config["preprocessing"]["cache_root"] = str(root / "native")
    native = root / "native/cases/train_10_a_1"
    image = np.arange(np.prod(shape), dtype=np.float32).reshape((1, *shape))
    from voxtell_preprocessed_cache import sha256_array
    save_npy(native / "image.npy", image)
    ct = {"preprocess_id": "crop_zscore_native_v1", "native_cropped_shape_zyx": list(shape),
          "resampled_shape_zyx": list(shape), "original_reoriented_shape_zyx": list(shape),
          "crop_bbox_zyx": [[0, n] for n in shape], "prompts": ["unused", "b", "d"],
          "image_sha256": sha256_array(image),
          "ct_properties": {"nibabel_stuff": {"original_affine": np.eye(4).tolist(),
                                               "reoriented_affine": np.eye(4).tolist()}}}
    atomic_json(native / "metadata.json", ct)
    y = np.zeros((2, *shape), np.uint8)
    y[0, 1, 2, 3] = 1
    y[1, 2, 3, 4] = 1
    z = np.full(y.shape, -1, np.float16)
    z[0, 1:3, 2:4, 3:5] = 1
    records = [{"name": name, "key": f"{name}::{fid}", "finding_id": fid, "channel": ch,
                "prompt": prompt, "category": category, "patient": "train_10", "split": "train",
                "voxels": 1, "entities": 1} for fid, ch, prompt, category in [("3", 1, "b", "2b"), ("9", 2, "d", "2d")]]
    folder = root / "cases/train_10_a_1"
    for filename, array in (("logits.npy", z), ("targets.npy", y)):
        save_npy(folder / filename, array)
    meta = {"contract": "synthetic", "name": name, "records": records,
            "hashes": {n: sha256(folder / n) for n in ("logits.npy", "targets.npy")},
            "points": {r["key"]: {"gt": [[1, 2, 3]], "prediction": []} for r in records},
            "image_path": str(native / "image.npy"), "ct_metadata": ct, "shape": list(shape),
            "origin": {"kind": "frozen_exp007_inference", "dtype": "float16",
                       "checkpoint_sha256": config["base"]["checkpoint_sha256"], "same_pass_mask_mismatch_voxels": 0}}
    atomic_json(folder / "metadata.json", meta)
    prepared = {"records": records, "halves": {}, "source_baseline": {},
                "cases": {name: {"keys": [r["key"] for r in records],
                                 "native_metadata_sha256": sha256(native / "metadata.json")}},
                "split_sha256": "fixture"}
    context = {"cache_contract": "synthetic", "sha256": "fixture"}
    atomic_json(root / "config.json", config)
    return config, context, prepared, folder, z, y


class MetadataTests(unittest.TestCase):
    def test_original_identifier_order_and_shared_ct(self):
        row = {"name": "train_11_a_1.nii.gz", "findings": {"9": "d", "3": "b", "1": "skip"},
               "categories": {"1": "2a", "3": "2b", "9": "2d"},
               "pixels": {"1": 1, "3": 2, "9": 3}, "entity_counts": {"1": 1, "3": 1, "9": 2}}
        records = finding_records({"train": [row], "val": []}, ["2b", "2d"])
        self.assertEqual([(r["finding_id"], r["channel"]) for r in records], [("3", 1), ("9", 2)])
        self.assertEqual(len({r["name"] for r in records}), 1)

    def test_real_census_and_patient_separation(self):
        config = read_json(DEFAULT_CONFIG)
        data = read_json(config["metadata"])
        halves = read_json(REPO / config["split_manifest"])["split"]["halves"]
        records = finding_records(data, config["categories"])
        validate_membership(records, halves, config)
        self.assertEqual(len(records), 5106)
        self.assertEqual(census(records, halves)["2e"]["B"], 3)
        bad = copy.deepcopy(records)
        next(r for r in bad if r["split"] == "val")["patient"] = records[0]["patient"]
        with self.assertRaisesRegex(ValueError, "patient overlap"):
            validate_membership(bad, halves, config)

    def test_disjoint_shards_and_smoke_workloads(self):
        cases = {str(i): {"split": "train", "shape": [i + 1, 2, 3], "prompt_count": 20 - i} for i in range(16)}
        plans = shard_plan(cases, 4)
        all_names = [n for p in plans for n in p["names"]]
        self.assertEqual(sorted(all_names), sorted(cases))
        for p in plans:
            self.assertEqual(p["names"][:len(p["smoke"])], p["smoke"])
            self.assertIn(max(p["names"], key=lambda n: (np.prod(cases[n]["shape"]), n)), p["smoke"])

    def test_contract_independent_of_training_and_split_views(self):
        config = read_json(DEFAULT_CONFIG)
        changed = copy.deepcopy(config)
        changed.update({"loss": "future", "runtime": "/tmp/elsewhere", "split_manifest": "new_view", "schedule": [1]})
        self.assertEqual(cache_contract(config), cache_contract(changed))
        changed["base"]["checkpoint_sha256"] = "other"
        self.assertNotEqual(cache_contract(config), cache_contract(changed))

    def test_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "snapshot.json"
            immutable_json(p, {"a": 1})
            immutable_json(p, {"a": 1})
            with self.assertRaises(ValueError):
                immutable_json(p, {"a": 2})


class ArrayTests(unittest.TestCase):
    def test_index_matches_historical_tiles(self):
        from deletion027_data import eligible_tiles
        rng = np.random.default_rng(22)
        for shape in ((3, 4, 5), (9, 10, 11)):
            z, y = rng.normal(size=shape), rng.integers(0, 2, size=shape)
            new = finding_index(z, y, (6, 6, 6))
            old = eligible_tiles(z, y, (6, 6, 6))
            self.assertEqual({k: new[k] for k in old}, old)
            self.assertEqual(new["base_fn"], int(((z < 0) & (y > 0)).sum()))

    def test_empty_base_and_false_negative_counts(self):
        z, y = -np.ones((3, 4, 5)), np.ones((3, 4, 5))
        index = finding_index(z, y, (6, 6, 6))
        self.assertFalse(index["deletion_eligible"])
        self.assertEqual(index["base_fn"], 60)
        self.assertEqual(index["tiles"], [])

    def test_exact_source_smoothed_dice(self):
        z = np.array([1., 1., -1.]).reshape(1, 1, 3)
        y = np.array([1, 0, 1]).reshape(1, 1, 3)
        index = finding_index(z, y, (4, 4, 4))
        self.assertEqual(index["dice"], (2 + 1e-6) / (4 + 1e-6))
        self.assertNotEqual(index["dice"], .5)

    def test_dense_small_volume_padding(self):
        index = finding_index(np.zeros((3, 4, 5)), np.ones((3, 4, 5)), (6, 6, 6))
        self.assertEqual(index["tiles"], [{"starts": [-2, -1, -1], "tp": 60, "fp": 0}])
        self.assertEqual(index["dice"], 1.)

    def test_storage_clipping_and_signed_zero(self):
        value = np.array([-100., -1e-9, 0., .2, 100.], np.float32)
        result = stored_logits(value, [-30, 30])
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result >= 0, value >= 0)
        self.assertEqual(result.min(), -30.)
        self.assertEqual(stored_logits(np.array([-1., 0., 1.]), [-30, 30]).dtype, np.float16)

    def test_orientation_round_trip(self):
        value = np.arange(60).reshape(3, 4, 5)
        props = {"nibabel_stuff": {"original_affine": np.diag([-1., -1., 1., 1.]).tolist(),
                                  "reoriented_affine": np.eye(4).tolist()}}
        np.testing.assert_array_equal(native_to_original(original_to_native(value, props), props), value)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config, self.context, self.prepared, self.folder, self.z, self.y = fixture(self.root)
        self.name = self.prepared["records"][0]["name"]

    def tearDown(self):
        self.temp.cleanup()

    def complete(self):
        build_index(self.folder, self.config, self.prepared["records"])
        return verify_case(self.root, self.name, self.config, self.context, self.prepared, {}, completing=True)

    def test_completion_and_exact_coverage(self):
        proof = self.complete()
        self.assertEqual(len(proof["keys"]), 2)
        self.assertEqual(verify_case(self.root, self.name, self.config, self.context, self.prepared, {}), proof)
        self.assertTrue((self.folder / "complete.json").exists())

    def test_incomplete_case_is_not_ready(self):
        with self.assertRaisesRegex(ValueError, "Incomplete cache"):
            verify_case(self.root, self.name, self.config, self.context, self.prepared, {})

    def test_resume_partial_without_new_inference(self):
        with patch("exp027_multicategory_worker.write_case_cache", side_effect=AssertionError("unexpected inference")):
            _, proof, reused = process_case(self.root, self.name, self.config, self.context, self.prepared, {}, {}, None)
        self.assertFalse(reused)
        self.assertEqual(len(proof["keys"]), 2)

    def test_resume_completed_without_new_inference(self):
        self.complete()
        with patch("exp027_multicategory_worker.write_case_cache", side_effect=AssertionError("unexpected inference")):
            _, proof, reused = process_case(self.root, self.name, self.config, self.context, self.prepared, {}, {}, None)
        self.assertTrue(reused)

    def test_corrupt_completed_array_rejected(self):
        self.complete()
        with (self.folder / "logits.npy").open("ab") as f:
            f.write(b"bad")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            process_case(self.root, self.name, self.config, self.context, self.prepared, {}, {}, None)

    def test_corrupt_tile_index_rejected(self):
        self.complete()
        atomic_json(self.folder / "tile_index.json", {})
        with self.assertRaisesRegex(ValueError, "SHA256"):
            verify_case(self.root, self.name, self.config, self.context, self.prepared, {})

    def test_cache_only_loader_preserves_original_channel(self):
        proof = self.complete()
        atomic_json(self.root / "input_manifest.json", {"status": "verified", "context_sha256": "fixture", "case_proofs": [proof]})
        with patch("exp027_multicategory_data.check_context", return_value=(self.context, self.prepared)):
            store = FindingStore(self.root)
            image, z, y, item, meta = store.get(self.prepared["records"][1]["key"])
            np.testing.assert_array_equal(z, self.z[1])
            self.assertFalse(item["deletion_eligible"])
            self.assertEqual(meta["records"][1]["channel"], 2)

    def test_loader_rejects_changed_inputs(self):
        proof = self.complete()
        atomic_json(self.root / "input_manifest.json", {"status": "verified", "context_sha256": "fixture", "case_proofs": [proof]})
        with patch("exp027_multicategory_data.check_context", return_value=(self.context, self.prepared)):
            store = FindingStore(self.root)
        (self.folder / "targets.npy").touch()
        with self.assertRaisesRegex(ValueError, "changed"):
            store.get(self.prepared["records"][0]["key"])

    def test_validation_original_mask_and_baseline(self):
        self.complete()
        records = self.prepared["records"]
        # Source includes the omitted original finding channel at index zero.
        source = np.stack([np.zeros((7, 6, 5)), self.z[0].transpose(2, 1, 0), self.z[1].transpose(2, 1, 0)])
        expected = {r["key"]: finding_index(self.z[i], self.y[i], (4, 4, 4))["dice"] for i, r in enumerate(records)}
        with patch("exp027_multicategory_data.load_validation_array", return_value=source):
            verify_validation_geometry(self.folder, records, {}, expected)
            source[1, 0, 0, 0] = 1
            with self.assertRaisesRegex(ValueError, "mask changed"):
                verify_validation_geometry(self.folder, records, {}, expected)

    def test_full_gate_rejects_missing_receipts(self):
        atomic_json(self.root / "status.json", {"mode": "export"})
        with patch("exp027_multicategory_data.check_context", return_value=(self.context, self.prepared)):
            with self.assertRaisesRegex(ValueError, "coverage"):
                finalize(self.config, self.root)

    def test_full_gate_and_manifest_publication(self):
        proof = self.complete()
        atomic_json(self.root / "receipts/train0.json", {"context_sha256": "fixture", "cases": [proof]})
        atomic_json(self.root / "status.json", {"mode": "export"})
        self.config["categories"] = ["2b", "2d"]
        # Synthetic train-only fixture has no validation baseline expectations.
        with patch("exp027_multicategory_data.check_context", return_value=(self.context, self.prepared)), \
             patch("exp027_multicategory_data.summarize", return_value=[]):
            value = finalize(self.config, self.root)
        self.assertEqual((value["cases"], value["findings"]), (1, 2))
        self.assertTrue((self.root / "manifests/2b/train.json").exists())


class ReportingAndLaunchTests(unittest.TestCase):
    def test_summary_recomposes_a_plus_b(self):
        rows = [{"record": {"category": "2b", "split": "val", "name": n}, "dice": d,
                 "base_tp": 1, "base_fp": 2, "base_fn": 3, "deletion_eligible": True} for n, d in [("a", .2), ("b", .4)]]
        s = summarize(rows, {"halves": {"a": "A", "b": "B"}}, ["2b"])
        self.assertIsNone(s[0]["mean_dice"])
        self.assertAlmostEqual(s[-1]["mean_dice"], .3)

    def test_pending_failure_report_and_single_writer(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            atomic_json(root / "status.json", {"status": "failed", "error": "SYNTHETIC OOM", "synthetic_fixture": True})
            report(root)
            text = (root / "reports/live_dashboard.md").read_text()
            self.assertIn("SYNTHETIC FIXTURE", text)
            self.assertIn("pending", text)
            self.assertIn("SYNTHETIC OOM", text)
            self.assertLess(text.index("| 2b |"), text.index("| 2e |"))
            with lock(root / "locks/report.lock"):
                with self.assertRaisesRegex(RuntimeError, "Another process owns"):
                    report(root)

    def test_explicit_gpu_and_sharing_gate(self):
        with patch.dict(os.environ, {"START_GPU_WORK": "0"}):
            with self.assertRaises(PermissionError):
                require_gpu(True, True)
        with patch.dict(os.environ, {"START_GPU_WORK": "1"}):
            with self.assertRaises(PermissionError):
                require_gpu(True, False)
            require_gpu(True, True)

    def test_memory_start_gate_without_allocation(self):
        with tempfile.TemporaryDirectory() as t:
            config = read_json(DEFAULT_CONFIG)
            guard = MemoryGuard(t, Path(t) / "attempt", "train0", config)
            with patch("torch.cuda.mem_get_info", return_value=(31 * 1024**3, 48 * 1024**3)):
                with self.assertRaisesRegex(RuntimeError, "32 GiB"):
                    guard.start()

    def test_memory_runtime_reserve(self):
        with tempfile.TemporaryDirectory() as t:
            guard = MemoryGuard(t, Path(t) / "attempt", "train0", read_json(DEFAULT_CONFIG))
            with patch("torch.cuda.mem_get_info", return_value=(3 * 1024**3, 48 * 1024**3)), \
                 patch("torch.cuda.max_memory_allocated", return_value=1), patch("torch.cuda.max_memory_reserved", return_value=1):
                with self.assertRaisesRegex(RuntimeError, "reserve breached"):
                    guard.sample()

    def test_dry_run_no_gpu_or_writes(self):
        with tempfile.TemporaryDirectory() as t:
            config = read_json(DEFAULT_CONFIG)
            config["runtime"] = str(Path(t) / "absent")
            self.assertFalse(dry_run(config, "orchestrate")["gpu_work"])
            self.assertFalse(Path(config["runtime"]).exists())

    def test_host_rejects_gpu_launch_without_optin_before_docker(self):
        env = {**os.environ, "START_GPU_WORK": "0"}
        result = subprocess.run(["bash", str(REPO / "scripts/rexgroundingct/run_027_multicategory_cache_host.sh"), "orchestrate"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("START_GPU_WORK", result.stderr)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
