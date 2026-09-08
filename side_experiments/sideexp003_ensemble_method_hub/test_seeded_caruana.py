from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import preliminary_ensemble as pe
import seeded_caruana as sc


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class SeededCaruanaTests(unittest.TestCase):
    def synthetic_sources(self, root: Path) -> tuple[Path, list[Path]]:
        roster = {
            "schema_version": 1,
            "roster_id": "synthetic_top20",
            "reuse_policy": "fresh_only",
            "N": 20,
            "dataset": {
                "path": str(root / "dataset.json"),
                "sha256": "dataset-sha",
                "cases": 200,
                "findings": 381,
                "split": "val200",
            },
            "candidates": [
                {
                    "rank": rank,
                    "id": f"c{rank:02d}",
                    "checkpoint": {"path": f"/checkpoint/{rank}", "sha256": f"sha-{rank}"},
                }
                for rank in range(1, 21)
            ],
        }
        roster["roster_sha256"] = sc._identity_sha(roster, "roster_sha256")
        roster_path = root / "roster.json"
        write_json(roster_path, roster)
        job_paths = []
        for job_number, ranks in ((1, range(1, 9)), (2, range(9, 21))):
            candidates = []
            for rank in ranks:
                cache = root / f"cache-{rank}"
                progress = root / f"progress-{rank}.json"
                export = {
                    "status": "strict_passed",
                    "cases": [
                        {"name": f"case-{index:03d}", "array_path": f"/array/{rank}/{index}"}
                        for index in range(200)
                    ],
                }
                validation = {
                    "status": "passed",
                    "storage_reproduction_status": "passed",
                    "array_hashes_verified": True,
                    "same_pass_mask_mismatch_voxels": 0,
                    "cases": 200,
                    "findings": 381,
                    "dtype": "float16",
                    "array_bytes": 1000 + rank,
                    "same_pass_mean_global_dice_per_finding": 0.3,
                    "same_pass_total_hits": 280,
                }
                write_json(cache / "export_manifest.json", export)
                write_json(cache / "reproduction_validation.json", validation)
                write_json(progress, {"status": "strict_passed", "cases": 200})
                candidates.append({
                    "rank": rank,
                    "id": f"c{rank:02d}",
                    "cache_key": f"key-{rank}",
                    "cache_root": str(cache),
                    "progress_path": str(progress),
                    "wave": (rank - 1) // 4 + 1,
                })
            job = {
                "job_id": f"j{job_number}",
                "roster_id": roster["roster_id"],
                "roster_sha256": roster["roster_sha256"],
                "dataset": roster["dataset"],
                "container": {"image": "synthetic"},
                "candidates": candidates,
            }
            path = root / f"job-{job_number}.json"
            write_json(path, job)
            job_paths.append(path)
        return roster_path, job_paths

    def test_launch_freezes_exact_per_scope_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roster_path, jobs = self.synthetic_sources(Path(directory))
            spec = sc.create_launch_spec(
                roster_path, jobs, top=16, workers=5, job_id="a001"
            )
            self.assertEqual(spec["seed_candidate_ids"]["all"], ["c01", "c02", "c03", "c04"])
            self.assertEqual(spec["seed_candidate_ids"]["2a"], ["c04", "c03", "c15", "c02"])
            self.assertEqual(spec["seed_candidate_ids"]["2b"], ["c07", "c04", "c11", "c01"])
            self.assertEqual(spec["seed_candidate_ids"]["2c"], ["c01", "c16", "c13", "c08"])
            self.assertEqual(spec["seed_candidate_ids"]["2d"], ["c01", "c03", "c02", "c05"])
            self.assertEqual(sc.validate_launch_spec(spec), [])

    def test_two_job_readiness_and_merged_roster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roster_path, jobs = self.synthetic_sources(Path(directory))
            spec = sc.create_launch_spec(
                roster_path, jobs, top=16, workers=5, job_id="a001"
            )
            readiness = sc.readiness_report(spec)
            self.assertTrue(readiness["ready"])
            self.assertEqual(readiness["ready_count"], 16)
            merged = sc.build_merged_roster(spec, readiness)
            self.assertEqual([row["source_job"] for row in merged["candidates"][:8]], ["j1"] * 8)
            self.assertEqual([row["source_job"] for row in merged["candidates"][8:]], ["j2"] * 8)
            self.assertEqual(sc.validate_merged_roster(merged), [])

    def test_scope_summary_filters_findings(self) -> None:
        rows = [
            {"case": "a", "finding_index": index, "category": "2a", "dice": 1.0}
            for index in range(69)
        ]
        summary = sc._summary_for_scope(rows, "2a")
        self.assertEqual(summary["findings"], 69)
        self.assertEqual(summary["hits"], 69)
        self.assertEqual(summary["dice"], 1.0)

    def test_initial_case_uses_probability_average_and_scope_seed(self) -> None:
        try:
            import nibabel as nib
        except ImportError:
            self.skipTest("nibabel not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = [f"c{index}" for index in range(4)]
            paths = {}
            logits = [10.0, 10.0, -10.0, -10.0]
            for candidate, value in zip(ids, logits):
                path = root / f"{candidate}.npy"
                np.save(path, np.full((1, 2, 1, 1), value, dtype=np.float16))
                paths[candidate] = str(path)
            old_seg = pe.SEG_DIR
            pe.SEG_DIR = root
            try:
                truth = np.array([[[[1]], [[0]]]], dtype=np.uint8)
                nib.save(nib.Nifti1Image(truth, np.eye(4)), root / "case.nii.gz")
                task = {
                    "case": {
                        "name": "case.nii.gz",
                        "findings": {"0": "finding"},
                        "categories": {"0": "2a"},
                    },
                    "paths": paths,
                    "candidate_ids": ids,
                    "pass_root": str(root / "pass"),
                    "fingerprint": "test",
                    "mode": "initial",
                    "seeds": {scope: ids for scope in sc.SCOPES},
                    "counts": None,
                    "basket_size": None,
                    "chunk_elements": 1,
                    "segmentation_dir": str(root),
                }
                result = sc._evaluate_case(task)
            finally:
                pe.SEG_DIR = old_seg
            self.assertIn("uniform", result["rows"])
            self.assertIn("seed:all", result["rows"])
            self.assertIn("seed:2a", result["rows"])
            self.assertNotIn("seed:2b", result["rows"])

    def test_one_and_five_worker_initial_passes_match(self) -> None:
        try:
            import nibabel as nib
        except ImportError:
            self.skipTest("nibabel not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = [f"c{index}" for index in range(4)]
            cases = []
            paths = {candidate: {} for candidate in ids}
            for case_index in range(5):
                name = f"case-{case_index}.nii.gz"
                cases.append({
                    "name": name,
                    "findings": {"0": "finding"},
                    "categories": {"0": "2a"},
                })
                truth = np.array([[[[1]], [[0]]]], dtype=np.uint8)
                nib.save(nib.Nifti1Image(truth, np.eye(4)), root / name)
                for model_index, candidate in enumerate(ids):
                    path = root / f"{case_index}-{candidate}.npy"
                    value = 10.0 if (case_index + model_index) % 3 else -10.0
                    np.save(path, np.full((1, 2, 1, 1), value, dtype=np.float16))
                    paths[candidate][name] = path
            roster = {"candidates": [{"id": candidate} for candidate in ids]}
            seeds = {scope: ids for scope in sc.SCOPES}
            common = {
                "execution_spec_sha256": "synthetic",
                "chunk_elements": 1,
                "segmentation_dir": str(root),
            }
            one = sc._run_pass(
                execution={**common, "runtime_root": str(root / "one"), "workers": 1},
                roster=roster, cases=cases, paths=paths, pass_name="initial",
                mode="initial", seeds=seeds,
            )
            five = sc._run_pass(
                execution={**common, "runtime_root": str(root / "five"), "workers": 5},
                roster=roster, cases=cases, paths=paths, pass_name="initial",
                mode="initial", seeds=seeds,
            )
            self.assertEqual(one["rows"], five["rows"])


if __name__ == "__main__":
    unittest.main()
