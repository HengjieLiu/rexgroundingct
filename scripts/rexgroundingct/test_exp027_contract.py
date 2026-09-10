"""CPU contract tests; no torch, CUDA, real-data exports or GPU workers."""
import contextlib
import copy
import io
import json
import math
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from exp027_common import (ARMS, DEFAULT_CONFIG, REPO, aggregate, atomic_json, digest, finding_metrics,
                          finding_records, full_schedule, gpu_gate, lock, make_split, patient_id,
                          read_json, schedule, segmentation_counts)
from exp027_data import FindingStore, cache_contract, extract_patch, stored_logits, prepare, load_validation_array
from exp027_report import collect, render, report_command
from run_027_residual_refinement import benchmark_plan, main, parallel_workers


def fake_rows():
    result = []
    for i in range(12):
        # Two reconstructions of each patient must travel together.
        result.append({"name": f"train_{i // 2 + 10}_a_{i % 2 + 1}.nii.gz",
                       "findings": {"2": "left linear opacity", "9": "right linear opacity", "12": "nodule"},
                       "categories": {"2": "2a", "9": "2a", "12": "2d"},
                       "pixels": {"2": 10 + i * 4, "9": 20 + i * 3, "12": 50},
                       "entity_counts": {"2": i % 5 + 1, "9": 1, "12": 1}})
    return result


def fake_prepared():
    split = make_split(fake_rows(), candidates=50)
    train = finding_records([{**r, "name": r["name"].replace("train_", "valid_")} for r in fake_rows()], "train")
    return {"train": train, "val": split["findings"], "split": split}


class ContractTests(unittest.TestCase):
    def test_validation_cache_payload_hash_excludes_npy_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logits.npy"
            array = np.linspace(-30, 30, 24, dtype=np.float16).reshape(2, 3, 4)
            np.save(path, array)
            checksum = hashlib.sha256(array.tobytes()).hexdigest()
            record = {"array_path": str(path), "array_sha256": checksum, "shape": list(array.shape),
                      "dtype": str(array.dtype), "npy_bytes": path.stat().st_size}
            self.assertNotEqual(checksum, hashlib.sha256(path.read_bytes()).hexdigest())
            np.testing.assert_array_equal(load_validation_array(record), array)
            for changed in ({"array_sha256": "0" * 64}, {"shape": [24]}, {"dtype": "float32"}, {"npy_bytes": 1}):
                with self.assertRaises(ValueError):
                    load_validation_array({**record, **changed})
            corrupt = array.copy()
            corrupt.flat[0] = 0
            np.save(path, corrupt)
            with self.assertRaisesRegex(ValueError, "payload SHA256"):
                load_validation_array(record)

    def test_prepare_is_cpu_only_and_preserves_source_config_bytes(self):
        config = read_json(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            metadata = {"val": fake_rows(), "train": [{**r, "name": r["name"].replace("train_", "valid_")}
                                                       for r in fake_rows()]}
            atomic_json(folder / "metadata.json", metadata)
            atomic_json(folder / "export_manifest.json", {})
            config["metadata"] = str(folder / "metadata.json")
            config["base"]["validation_cache"] = str(folder)
            config["split"] = {"candidates": 20}
            config["expected"] = {"train_findings": 24, "val_findings": 24, "val_cases": 12, "val_total_cases": 12}
            source = folder / "source.json"
            source.write_text(json.dumps(config, indent=1) + "\n")
            manifest = {"cases": [{"name": r["name"], "status": "complete", "same_pass_mask_mismatch_voxels": 0,
                                    "same_pass_reference": {"finding_dice": [.2, .3, .4]}} for r in metadata["val"]]}
            root = folder / "runtime"
            with patch("exp027_data.validate_sources", return_value=manifest):
                prepared = prepare(config, root, source)
                legacy = copy.deepcopy(prepared)
                legacy["baseline"]["full"]["dice"] = np.nextafter(legacy["baseline"]["full"]["dice"], 1.)
                atomic_json(root / "prepared.json", legacy)
                self.assertEqual(prepare(config, root, source), prepared)
            self.assertEqual(source.read_bytes(), (root / "config" / DEFAULT_CONFIG.name).read_bytes())
            self.assertFalse((root / "cache").exists())
            self.assertEqual(read_json(root / "status.json")["status"], config["status"])
            self.assertNotIn("torch", sys.modules)

    def test_frozen_validation_manifest_complete(self):
        config = read_json(DEFAULT_CONFIG)
        frozen = read_json(REPO / config["split"]["manifest"])
        split = frozen["split"]
        self.assertEqual(frozen["metadata_sha256"], config["metadata_sha256"])
        self.assertEqual(frozen["split_sha256"], digest(split))
        self.assertEqual(len(split["halves"]), 200)
        self.assertEqual(len({r["key"] for r in split["findings"]}), 69)
        self.assertEqual(len({r["name"] for r in split["findings"]}), 63)
        patient_halves = {}
        for name, half in split["halves"].items():
            patient_halves.setdefault(patient_id(name), set()).add(half)
        self.assertTrue(all(len(halves) == 1 for halves in patient_halves.values()))
        self.assertEqual([split["counts"][h]["findings_2a"] for h in ("A", "B")], [35, 34])

    def test_split_reproducible_and_patient_separated(self):
        a = make_split(fake_rows(), candidates=100)
        self.assertEqual(a, make_split(list(reversed(fake_rows())), candidates=100))
        halves = a["halves"]
        for n, half in halves.items():
            self.assertEqual({h for other, h in halves.items() if patient_id(other) == patient_id(n)}, {half})
        self.assertEqual(sum(c["findings_2a"] for c in a["counts"].values()), 24)
        self.assertEqual(sum(c["all_cases"] for c in a["counts"].values()), 12)

    def test_original_ids_and_channels_are_preserved(self):
        records = finding_records(fake_rows(), "val")
        self.assertEqual({r["finding_id"] for r in records}, {"2", "9"})
        self.assertEqual({r["channel"] for r in records}, {0, 1})
        self.assertEqual(len({r["key"] for r in records}), len(records))
        with self.assertRaises(ValueError):
            patient_id("unrecognized.nii.gz")

    def test_all_memberships_exact_mixtures_and_prefixes(self):
        p = fake_prepared()
        train = {r["key"] for r in p["train"]}
        a = {r["key"] for r in p["val"] if r["half"] == "A"}
        b = {r["key"] for r in p["val"] if r["half"] == "B"}
        allowed = (train, train | a, a, a | b)
        for arm, keys in zip(ARMS, allowed):
            events = schedule(p, arm, 200, 20260909)
            self.assertEqual(events[:100], schedule(p, arm, 100, 20260909))
            self.assertTrue({e["key"] for e in events} <= keys)
            for block in (events[:100], events[100:]):
                self.assertEqual([sum(e["mode"] == k for e in block) for k in ("gt", "prediction", "random")], [50, 25, 25])
                if arm == ARMS[1]:
                    self.assertEqual(sum(e["source"] == "train" for e in block), 50)
                    self.assertEqual(sum(e["source"] == "A" for e in block), 50)

    def test_padding_and_gt_patch_contains_anchor(self):
        array = np.ones((3, 4, 5), dtype=np.float32)
        crop, valid = extract_patch(array, (-2, -1, -1), (8, 8, 8), -30)
        self.assertEqual(valid.sum(), array.size)
        np.testing.assert_array_equal(crop[valid == 1], 1)
        self.assertTrue(np.all(crop[valid == 0] == -30))

    def test_sampler_empty_prediction_fallback_flips_and_replay(self):
        p, c = fake_prepared(), read_json(DEFAULT_CONFIG)
        r = p["train"][0]
        image = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        target = np.zeros_like(image, dtype=np.uint8)
        target[1, 1, 1] = 1
        store = FindingStore("/unused", p, c)
        points = {"gt": [[1, 1, 1]], "prediction": []}
        event = {"key": r["key"], "patch_seed": 12, "mode": "prediction"}
        with patch.object(store, "get", return_value=(image, np.full_like(image, -2), target, points, {})):
            x, y, valid, info = store.patch(event, (5, 5, 5))
            again = store.patch(event, (5, 5, 5))
            self.assertTrue(info["fallback"])
            self.assertEqual(info["actual_mode"], "random")
            self.assertEqual(valid.sum(), 27)
            self.assertEqual(y.sum(), 1)
            for left, right in zip((x, y, valid), again[:3]):
                np.testing.assert_array_equal(left, right)
            _, gy, _, gi = store.patch({**event, "mode": "gt"}, (2, 2, 2))
            self.assertEqual(gy.sum(), 1)
            self.assertFalse(gi["fallback"])

    def test_missing_cache_never_generates_logits(self):
        p, c = fake_prepared(), read_json(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                FindingStore(root, p, c).get(p["train"][0]["key"])

    def test_storage_sign_fallback_and_nonfinite(self):
        ordinary = np.array([-100, -1, 0, 100], dtype=np.float32)
        stored = stored_logits(ordinary, [-30, 30])
        self.assertEqual(stored.dtype, np.float16)
        np.testing.assert_array_equal(stored >= 0, ordinary >= 0)
        near_zero = np.array([-1e-12, 1e-12], dtype=np.float32)
        self.assertEqual(stored_logits(near_zero, [-30, 30]).dtype, np.float32)
        with self.assertRaises(ValueError):
            stored_logits(np.array([np.nan]), [-30, 30])

    def test_metrics_help_harm_and_outside_crop(self):
        base = np.array([1, -1, 1, -1, 1, -1], dtype=float)
        final = np.array([-1, 1, -1, 1, 1, -1], dtype=float)
        gt = np.array([0, 1, 1, 0, 1, 0])
        m = finding_metrics(base, final, gt, final - base, total_gt=4)
        for key in ("fp_removed", "fn_recovered", "tp_removed", "tn_added"):
            self.assertEqual(m[key], 1)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["fn"], 2)
        self.assertAlmostEqual(m["dice"], (4 + 1e-6) / (7 + 1e-6))
        self.assertEqual(segmentation_counts(np.zeros(2), np.zeros(2))["dice"], 1)

    def test_full_recomposition_is_finding_weighted(self):
        rows = []
        for i, value in enumerate((.1, .3, .9)):
            rows.append({"name": str(i), "half": "A" if i < 2 else "B", "dice": value,
                         "base_dice": .2, "hit": int(value >= .1), "base_hit": 1})
        result = aggregate(rows)
        self.assertAlmostEqual(result["full"]["dice"], (2 * result["A"]["dice"] + result["B"]["dice"]) / 3)
        self.assertIsNone(aggregate([])["A"]["dice"])
        repeated = [dict(rows[0], dice=.1) for _ in range(69)]
        self.assertEqual(aggregate(repeated)["full"]["dice"], math.fsum([.1] * 69) / 69)

    def test_budget_and_gpu_gates(self):
        config = read_json(DEFAULT_CONFIG)
        self.assertEqual(full_schedule(config), (10000, list(range(1000, 10001, 1000))))
        config["training"].update(total_updates=None, evaluation_updates=None)
        with self.assertRaises(ValueError):
            full_schedule(config)
        with self.assertRaises(PermissionError):
            gpu_gate(False)
        config["training"].update(total_updates=300, evaluation_updates=[100, 200])
        self.assertEqual(full_schedule(config), (300, [100, 200, 300]))

    def test_dry_run_does_not_write_or_import_torch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "not_created"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["benchmark", "--dry-run", "--root", str(root)]), 0)
            self.assertFalse(root.exists())
            for command in (["benchmark"], ["train", "--phase", "benchmark", "--arm", ARMS[0], "--until", "100"],
                            ["evaluate", "--phase", "benchmark", "--arm", ARMS[0], "--update", "100"]):
                with self.assertRaises(PermissionError):
                    main([*command, "--root", str(root)])
            self.assertFalse(root.exists())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["orchestrate", "--dry-run", "--root", str(root)]), 0)
            unset = read_json(DEFAULT_CONFIG)
            unset["training"].update(total_updates=None, evaluation_updates=None)
            path = Path(folder) / "unset.json"
            atomic_json(path, unset)
            with self.assertRaises(ValueError):
                main(["orchestrate", "--dry-run", "--config", str(path), "--root", str(root)])
        self.assertNotIn("torch", sys.modules)

    def test_parallel_failure_stops_peer_and_reports(self):
        with tempfile.TemporaryDirectory() as root:
            commands = [("fail", [sys.executable, "-c", "raise SystemExit(7)"], os.environ.copy()),
                        ("peer", [sys.executable, "-c", "import time; time.sleep(20)"], os.environ.copy())]
            with self.assertRaisesRegex(RuntimeError, "Worker failure"):
                parallel_workers(commands, root=root, refresh=lambda: None)
            self.assertEqual(len(list((Path(root) / "logs").glob("*.log"))), 2)

    def test_single_writer_lock(self):
        with tempfile.TemporaryDirectory() as root:
            with lock(Path(root) / ".report_writer.lock"):
                with self.assertRaises(RuntimeError):
                    report_command(root)

    def test_dashboard_missing_partial_failed_fixed_order(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            atomic_json(root / "status.json", {"status": "failed", "active_phase": "benchmark", "synthetic_fixture": True})
            atomic_json(root / "benchmark" / ARMS[1] / "training.json",
                        {"updates": [{"update": 1, "loss": 1.0, "patch_dice": .2, "residual_mean_abs": .01, "source": "A"}]})
            atomic_json(root / "benchmark" / ARMS[1] / "status.json", {"status": "failed", "update": 1, "error": "fixture failure"})
            with lock(root / ".report_writer.lock"):
                result = render(root)
            self.assertEqual([a["arm"] for a in result["arms"]], list(ARMS))
            md = (root / "reports/live_dashboard.md").read_text()
            self.assertIn("SYNTHETIC FIXTURE", md)
            self.assertIn("fixture failure", md)
            self.assertIn("pending / pending", md)
            self.assertTrue((root / "reports/live_dashboard.png").stat().st_size > 1000)
            positions = [md.index(f"| {arm} |") for arm in ARMS]
            self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
