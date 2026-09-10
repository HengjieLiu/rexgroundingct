#!/usr/bin/env python3
"""Test label-free contracts, publication recovery, and coordinator ownership."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fresh_cache as fc
import test300_cache as tc
import test300_runner as tr


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = {
            "name": "test_1_a_1.nii.gz",
            "shape": [2, 2, 2],
            "findings": {"0": "a prompt"},
        }
        self.job = {
            "job_spec_sha256": "job-sha",
            "dataset": {"cases": 1, "findings": 1},
        }
        self.c = {
            "cache_key": "key-one",
            "cache_root": str(self.root / "published"),
            "staging_root": str(self.root / "stage"),
        }
        Path(self.c["cache_root"]).mkdir()

    def stage(self, values=None):
        a = np.array(
            values if values is not None else [-2, -1, -0.1, 0, 0.1, 1, 2, 30],
            dtype=np.float32,
        ).reshape(1, 2, 2, 2)
        ap, rp = tc.stage_paths(self.c, self.case["name"])
        tc.atomic_save_npy(ap, a)
        r = {
            "name": self.case["name"],
            "shape": list(a.shape),
            "dtype": "float32",
            "job_spec_sha256": "job-sha",
            "cache_key": "key-one",
            "array_sha256": tc.array_sha256(a),
            **tc.prompt_contract(self.case),
        }
        fc.atomic_write_json(rp, r)
        return a, r

    def test_split_isolation_and_numeric_order(self):
        c = copy.deepcopy(self.case)
        c["findings"] = {str(i): f"prompt{i}" for i in reversed(range(12))}
        p = self.root / "metadata.json"
        fc.atomic_write_json(p, {"test": [c], "val": [self.case]})
        got = tc.load_cases(str(p), 1, 12)
        self.assertEqual(tc.prompt_contract(got[0])["finding_indices"], list(range(12)))
        self.assertEqual(len(got), 1)
        with self.assertRaises(fc.FreshCacheError):
            tc.load_cases(p, 2, 13)
        c["findings"]["01"] = "duplicate numeric index"
        fc.atomic_write_json(p, {"test": [c]})
        with self.assertRaises(fc.FreshCacheError):
            tc.load_cases(p, 1, 13)

    def test_duplicates_and_non_test_dataset_rejected(self):
        p = self.root / "metadata.json"
        for data in ({"test": [self.case, self.case]}, {"val": [self.case]}):
            fc.atomic_write_json(p, data)
            with self.assertRaises(fc.FreshCacheError):
                tc.load_cases(p, 2, 2)

    def test_float16_exact_masks_and_atomic_cross_filesystem_publication(self):
        original, _ = self.stage()
        with patch.object(
            tc.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**4)
        ), patch("os.link", side_effect=AssertionError("hard links forbidden")):
            dtype, records = tc.publish(self.job, self.c, [self.case])
            self.assertEqual(dtype, "float16")
            output = tc.check_array(
                Path(records[0]["array_path"]),
                records[0],
                self.case,
                "job-sha",
                "key-one",
            )
            np.testing.assert_array_equal(output >= 0, original >= 0)
            # Publication can restart without replacing a verified destination array.
            with patch.object(
                tc, "atomic_save_npy", side_effect=AssertionError("should reuse")
            ):
                self.assertEqual(
                    tc.publish(self.job, self.c, [self.case])[0], "float16"
                )
        self.assertFalse(list(Path(self.c["cache_root"]).rglob("*.tmp")))

    def test_float32_fallback_reuses_inference(self):
        original, _ = self.stage([-1e-12, -1, -0.1, 0, 0.1, 1, 2, 30])
        with patch.object(
            tc.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**4)
        ):
            dtype, records = tc.publish(self.job, self.c, [self.case])
        self.assertEqual(dtype, "float32")
        np.testing.assert_array_equal(np.load(records[0]["array_path"]), original)

    def test_actual_cross_filesystem_publication(self):
        try:
            destination = tempfile.TemporaryDirectory(dir="/dev/shm")
        except (PermissionError, FileNotFoundError):
            self.skipTest("separate writable tmpfs required; run in Docker")
        self.addCleanup(destination.cleanup)
        if Path(destination.name).stat().st_dev == self.root.stat().st_dev:
            self.skipTest("source and destination are on the same filesystem")
        self.c["cache_root"] = destination.name
        original, _ = self.stage()
        with patch.object(
            tc.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**4)
        ):
            _, records = tc.publish(self.job, self.c, [self.case])
        np.testing.assert_array_equal(
            np.load(records[0]["array_path"]) >= 0, original >= 0
        )

    def test_restart_detects_corruption_wrong_checkpoint_and_orphan(self):
        a, rec = self.stage()
        self.assertIsNotNone(tc.load_stage(self.c, self.case, "job-sha"))
        ap, rp = tc.stage_paths(self.c, self.case["name"])
        rec["cache_key"] = "other"
        fc.atomic_write_json(rp, rec)
        with self.assertRaises(fc.FreshCacheError):
            tc.load_stage(self.c, self.case, "job-sha")
        rec["cache_key"] = "key-one"
        fc.atomic_write_json(rp, rec)
        a[0, 0, 0, 0] = 3
        tc.atomic_save_npy(ap, a)
        with self.assertRaises(fc.FreshCacheError):
            tc.load_stage(self.c, self.case, "job-sha")
        rp.unlink()
        self.assertIsNone(tc.load_stage(self.c, self.case, "job-sha"))

    def test_insufficient_space_preserves_stage_and_publishes_nothing(self):
        self.stage()
        with patch.object(
            tc.shutil, "disk_usage", return_value=SimpleNamespace(free=tc.MIN_SHARED)
        ):
            with self.assertRaises(fc.FreshCacheError):
                tc.publish(self.job, self.c, [self.case])
        self.assertTrue(tc.stage_paths(self.c, self.case["name"])[0].exists())
        self.assertFalse(list(Path(self.c["cache_root"]).rglob("*.npy")))
        self.assertFalse(tc.wave_space_ok(tc.MIN_SHARED + 500, 10**15, 4, 1000))
        self.assertTrue(
            tc.wave_space_ok(100 * 1024**4, 2 * 1024**4, 4, 58536820736)
        )

    def test_another_live_owner_is_rejected(self):
        with tc.exclusive(self.root / "launch.lock"):
            with self.assertRaises(fc.FreshCacheError):
                with tc.exclusive(self.root / "launch.lock"):
                    pass

    def test_cleanup_preserves_bind_mount_and_unrelated_sibling(self):
        self.stage()
        sibling = self.root / "unrelated.npy"
        sibling.write_bytes(b"keep")
        stage = Path(self.c["staging_root"])
        tc.cleanup_stage(stage)
        self.assertTrue(stage.is_dir())
        self.assertFalse(list(stage.iterdir()))
        self.assertEqual(sibling.read_bytes(), b"keep")

    def test_smokes_cover_different_input_and_output_maxima(self):
        small = copy.deepcopy(self.case)
        big = copy.deepcopy(self.case)
        big["name"] = "test_2_a_1.nii.gz"
        big["shape"] = [4, 4, 4]
        for c, shape in [(small, [8, 8, 8]), (big, [4, 4, 4])]:
            fc.atomic_write_json(
                self.root
                / "cases"
                / c["name"].removesuffix(".nii.gz")
                / "metadata.json",
                {"resampled_shape_zyx": shape},
            )
        self.assertEqual(
            tc.smoke_cases([small, big], self.root), [small["name"], big["name"]]
        )

    def test_catalog_sync_leaves_val200_and_roster_memberships_identical(self):
        c = {
            **self.c,
            "id": "candidate",
            "rank": 1,
            "wave": 1,
            "gpu": 0,
            "progress_path": str(self.root / "progress.json"),
            "cache_version_id": "test-version",
        }
        job = {**self.job, "job_id": "j003", "candidates": [c]}
        catalog = {
            "candidates": [
                {
                    "candidate_id": "candidate",
                    "roster_memberships": [{"job_id": "val-job"}],
                    "inference_artifacts": {
                        "val200": {
                            "active_cache_version": "val-key",
                            "versions": [{"key": "val-key"}],
                        },
                        "test300": {"status": "deferred", "versions": []},
                    },
                }
            ]
        }
        old = copy.deepcopy(catalog)
        result = tc.sync_catalog(catalog, job)
        self.assertEqual(catalog, old)
        self.assertEqual(
            result["candidates"][0]["inference_artifacts"]["val200"],
            old["candidates"][0]["inference_artifacts"]["val200"],
        )
        self.assertEqual(
            result["candidates"][0]["roster_memberships"],
            old["candidates"][0]["roster_memberships"],
        )
        self.assertIsNone(
            result["candidates"][0]["inference_artifacts"]["test300"]["versions"][0][
                "metrics_path"
            ]
        )

    def runner(self):
        r = object.__new__(tr.Runner)
        r.root = self.root
        r.paused = None
        r.paused_id = None
        r.degraded = 0
        r.recovered = 0
        r.baseline = {"candidate": 1}
        return r

    def test_runner_reads_frozen_val_timing_manifest_from_json_path(self):
        val = self.root / "val_export.json"
        fc.atomic_write_json(
            val, {"cases": [{"elapsed_seconds": 8, "shape": [1, 2, 2, 2]}]}
        )
        job = self.root / "job.json"
        fc.atomic_write_json(
            job,
            {
                "runtime_root": str(self.root),
                "candidates": [
                    {
                        "id": "candidate",
                        "paired_val200": {
                            "export_manifest": str(val),
                            "export_sha256": fc.sha256_file(val),
                        },
                    }
                ],
            },
        )
        r = tr.Runner(str(job))
        self.assertEqual(r.baseline["candidate"], 1)

    def test_worker_launch_uses_immutable_image_id_and_hides_segmentations(self):
        r = self.runner()
        r.path = self.root / "job_spec.json"
        r.job = {
            "job_id": "test-job",
            "job_spec_sha256": "job-sha",
            "container": {"image": "mutable-tag", "image_id": "sha256:frozen-image"},
        }
        r.active = []
        c = {
            **self.c,
            "id": "candidate",
            "wave": 1,
            "gpu": 0,
            "progress_path": str(self.root / "progress.json"),
        }
        with patch.object(tc, "REPO", self.root), patch.object(
            tr, "docker_state", return_value=None
        ), patch.object(tr, "command") as command:
            r.launch(c)
        argv = command.call_args.args[0]
        self.assertIn("sha256:frozen-image", argv)
        self.assertNotIn("mutable-tag", argv)
        self.assertIn(
            "HF_HUB_CACHE=/data/hengjie/datasets/rexgroundingct/.hf_home/hub", argv
        )
        self.assertIn(
            "/data/hengjie/datasets/rexgroundingct/segmentations:ro,size=1m", argv
        )

    def test_contention_pause_and_recovery_own_only_cpu_production(self):
        r = self.runner()
        cp = self.root / "cpu.json"
        fc.atomic_write_json(
            cp, {"host": "shenggpu8", "container": "rex-resume-session-production"}
        )
        d = {
            "Id": "immutable-container-id",
            "State": {"Running": True, "Paused": False},
        }

        def records(cost):
            return {
                "candidate": {
                    "status": "exporting",
                    "recent_case_costs": [{"seconds": cost, "elements": 1}] * 10,
                }
            }

        with patch.object(tr, "CPU_STATE", cp), patch.object(
            tr, "docker_state", return_value=d
        ), patch.object(tr, "command") as command:
            r.contention(records(3))
            command.assert_not_called()
            r.contention(records(3))
            self.assertEqual(
                command.call_args.args[0],
                ["docker", "pause", "rex-resume-session-production"],
            )
            d["State"]["Paused"] = True
            r.contention(records(1))
            r.contention(records(1))
            self.assertEqual(
                command.call_args.args[0],
                ["docker", "unpause", "rex-resume-session-production"],
            )
            self.assertIsNone(r.paused)
            command.reset_mock()
            fc.atomic_write_json(
                cp, {"host": "shenggpu8", "container": "unrelated-job"}
            )
            r.contention(records(3))
            r.contention(records(3))
            command.assert_not_called()

    def test_never_unpause_replacement_container_or_existing_user_pause(self):
        r = self.runner()
        r.paused = "rex-resume-session-production"
        r.paused_id = "old"
        with patch.object(
            tr,
            "docker_state",
            return_value={"Id": "new", "State": {"Running": True, "Paused": True}},
        ), patch.object(tr, "command") as command:
            r.release_pause()
            command.assert_not_called()

    def test_cpu_benchmark_selection_is_required(self):
        r = self.runner()
        cp = self.root / "cpu.json"
        session = self.root / "session"
        session.mkdir()
        fc.atomic_write_json(cp, {"session": str(session), "status": "benchmarking"})
        with patch.object(tr, "CPU_STATE", cp):
            self.assertFalse(r.cpu_ready())
            fc.atomic_write_json(session / "selected_execution.json", {"workers": 5})
            self.assertFalse(r.cpu_ready())
            fc.atomic_write_json(
                cp, {"session": str(session), "status": "computing_new_cases"}
            )
            self.assertTrue(r.cpu_ready())


@unittest.skipUnless(
    importlib.util.find_spec("nibabel"),
    "run in frozen VoxTell image for geometry tests",
)
class LabelFreeGeometry(unittest.TestCase):
    setUp = Contracts.setUp

    def test_orientation_and_prediction_without_segmentation_access(self):
        import nibabel as nib

        ex = tc.inference_imports()
        import common
        import run_voxtell_val_inference as infer
        import voxtell_preprocessed_cache as prep

        affine = np.diag([-1.0, 1.0, 1.0, 1.0])
        affine[0, 3] = 1
        ct_path = self.root / "ct.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine), ct_path
        )
        root = (
            Path(self.c["cache_root"])
            / "cases"
            / self.case["name"].removesuffix(".nii.gz")
        )
        a = np.ones((1, 2, 2, 2), dtype=np.float32)
        meta = {
            "image_sha256": prep.sha256_array(a),
            "normalization": {"name": "zscore"},
            "ct_properties": {
                "nibabel_stuff": {
                    "original_affine": affine.tolist(),
                    "reoriented_affine": np.eye(4).tolist(),
                }
            },
        }
        tc.atomic_save_npy(root / "image.npy", a)
        fc.atomic_write_json(root / "metadata.json", meta)
        (root / ".complete").write_text(meta["image_sha256"])
        # Even a deliberately malformed target file must never be opened.
        (root / "targets.npz").write_text("forbidden target file")
        c = {
            **self.c,
            "cache": {"root": self.c["cache_root"]},
            "test_preprocessing_cases": {
                self.case["name"]: {
                    "metadata_sha256": fc.sha256_file(root / "metadata.json"),
                    "image_sha256": meta["image_sha256"],
                }
            },
        }
        logits = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2) - 3
        with patch.object(
            common, "ct_rate_abs_path", return_value=ct_path
        ), patch.object(
            prep,
            "load_cached_case",
            side_effect=AssertionError("target-aware loader forbidden"),
        ), patch.object(
            prep, "image_padding_value", return_value=0
        ), patch.object(
            infer, "predict_preprocessed_crop_logits", return_value=logits
        ), patch.object(
            infer, "restore_cached_native_crop", side_effect=lambda x, *a, **kw: x
        ):
            output, details = tc.predict_test(object(), c, self.case)
        np.testing.assert_array_equal(
            output, np.transpose(logits, (0, 3, 2, 1))[:, ::-1, :, :]
        )
        self.assertEqual(details["metric_status"], tc.METRIC_STATUS)
        self.assertNotIn("dice", details)
        # Exercise complete publication validation with a real CT header.
        ap, rp = tc.stage_paths(c, self.case["name"])
        tc.atomic_save_npy(ap, output)
        fc.atomic_write_json(
            rp,
            {
                **details,
                "name": self.case["name"],
                "shape": list(output.shape),
                "dtype": "float32",
                "job_spec_sha256": "job-sha",
                "cache_key": "key-one",
                "array_sha256": tc.array_sha256(output),
            },
        )
        with patch.object(
            tc.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**4)
        ), patch.object(common, "ct_rate_abs_path", return_value=ct_path):
            tc.publish(self.job, c, [self.case])
            v, _ = tc.validate_published(self.job, c, [self.case])
        self.assertEqual(v["status"], "passed")


if __name__ == "__main__":
    unittest.main()
