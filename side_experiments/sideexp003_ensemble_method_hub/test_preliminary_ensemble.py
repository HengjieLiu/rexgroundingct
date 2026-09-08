from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

import preliminary_ensemble as pe

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - exercised in the pinned container
    nib = None


class PreliminaryEnsembleTests(unittest.TestCase):
    def test_sigmoid_and_uniform_probability_average(self) -> None:
        logits = {
            "a": np.array([-2.0, 2.0, 0.0, 1.0], dtype=np.float16),
            "b": np.array([-1.0, 1.0, 0.0, -1.0], dtype=np.float16),
            "c": np.array([0.0, 0.0, 2.0, -2.0], dtype=np.float16),
            "d": np.array([1.0, -1.0, -2.0, 2.0], dtype=np.float16),
        }
        probabilities = {key: pe.sigmoid_float32(value) for key, value in logits.items()}
        generated = pe.generate_probability_ensembles(
            probabilities, {"uniform": {key: 1 for key in logits}}
        )
        summed, basket_size = generated["uniform"]
        expected = sum(probabilities.values()) / 4.0
        np.testing.assert_allclose(summed / basket_size, expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(
            summed >= 0.5 * basket_size, expected >= 0.5
        )

    def test_chunked_threshold_counts_equal_direct(self) -> None:
        rng = np.random.default_rng(7)
        probabilities = {
            key: pe.sigmoid_float32(rng.normal(size=23).astype(np.float32))
            for key in ("a", "b", "c", "d")
        }
        truth = rng.random(23) > 0.55
        recipe = {"r": {"a": 2, "b": 1, "d": 1}}
        direct, size = pe.generate_probability_ensembles(probabilities, recipe)["r"]
        direct_mask = direct >= 0.5 * size
        pred = intersection = 0
        for start in range(0, truth.size, 5):
            stop = min(start + 5, truth.size)
            chunks = {key: value[start:stop] for key, value in probabilities.items()}
            ensemble, chunk_size = pe.generate_probability_ensembles(chunks, recipe)["r"]
            mask = ensemble >= 0.5 * chunk_size
            pred += int(np.count_nonzero(mask))
            intersection += int(np.count_nonzero(mask & truth[start:stop]))
        self.assertEqual(pred, int(np.count_nonzero(direct_mask)))
        self.assertEqual(intersection, int(np.count_nonzero(direct_mask & truth)))
        self.assertAlmostEqual(
            pe.dice_from_counts(int(np.count_nonzero(truth)), pred, intersection),
            pe.dice_from_counts(
                int(np.count_nonzero(truth)),
                int(np.count_nonzero(direct_mask)),
                int(np.count_nonzero(direct_mask & truth)),
            ),
        )

    def test_replacement_frequency_becomes_weight(self) -> None:
        sequence = ["a", "b", "a", "a"]
        self.assertEqual(pe.basket_weights(sequence), {"a": 0.75, "b": 0.25})

    def test_dice_then_hits_then_id_tie_break(self) -> None:
        trials = {
            "c": {"metrics": {"dice": 0.5, "hits": 8}},
            "b": {"metrics": {"dice": 0.5, "hits": 9}},
            "a": {"metrics": {"dice": 0.5, "hits": 9}},
            "d": {"metrics": {"dice": 0.49, "hits": 99}},
        }
        self.assertEqual(pe.select_trial(trials), "a")

    def test_three_replacement_rounds_can_repeat_best(self) -> None:
        candidate_ids = ["a", "b", "c", "d"]
        sequence = ["a"]
        rounds = [
            {"a": (0.60, 8), "b": (0.55, 9), "c": (0.50, 10), "d": (0.45, 11)},
            {"a": (0.61, 8), "b": (0.60, 9), "c": (0.59, 10), "d": (0.58, 11)},
            {"a": (0.62, 8), "b": (0.61, 9), "c": (0.60, 10), "d": (0.59, 11)},
        ]
        evaluated = 0
        for round_values in rounds:
            trials = {
                candidate_id: {
                    "metrics": {
                        "dice": round_values[candidate_id][0],
                        "hits": round_values[candidate_id][1],
                    }
                }
                for candidate_id in candidate_ids
            }
            evaluated += len(trials)
            sequence.append(pe.select_trial(trials))
        self.assertEqual(evaluated, 12)
        self.assertEqual(sequence, ["a", "a", "a", "a"])
        self.assertEqual(pe.basket_weights(sequence), {"a": 1.0})

    def test_empty_empty_dice_is_one(self) -> None:
        self.assertEqual(pe.dice_from_counts(0, 0, 0), 1.0)

    def test_code_provenance_freezes_entrypoint(self) -> None:
        provenance = pe.code_provenance()
        self.assertEqual(provenance["entrypoint"], "side_experiments/sideexp003_ensemble_method_hub/preliminary_ensemble.py")
        self.assertEqual(provenance["entrypoint_sha256"], pe.sha256_file(Path(pe.__file__)))
        self.assertEqual(len(provenance["entrypoint_sha256"]), 64)

    @unittest.skipUnless(nib is not None, "nibabel is available in the pinned container")
    def test_streaming_end_to_end_metrics_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seg_dir = root / "segmentations"
            seg_dir.mkdir()
            candidate_ids = ["a", "b", "c", "d"]
            paths = {candidate_id: {} for candidate_id in candidate_ids}
            cases = []
            for case_index in range(200):
                findings = 2 if case_index < 181 else 1
                name = f"case_{case_index:03d}.nii.gz"
                shape = (findings, 1, 1, 1)
                truth = np.ones(shape, dtype=np.uint8)
                nib.save(nib.Nifti1Image(truth, np.eye(4)), seg_dir / name)
                cases.append(
                    {
                        "name": name,
                        "findings": {str(index): "x" for index in range(findings)},
                        "categories": {str(index): "2a" for index in range(findings)},
                    }
                )
                for candidate_id in candidate_ids:
                    path = root / f"{candidate_id}_{case_index:03d}.npy"
                    np.save(path, np.ones(shape, dtype=np.float16))
                    paths[candidate_id][name] = path
            roster = {
                "roster_sha256": "synthetic",
                "candidates": [{"id": candidate_id} for candidate_id in candidate_ids],
            }
            job_path = root / "job.json"
            job_path.write_text(json.dumps({"candidates": []}))
            old_seg, old_runtime, old_minimum = pe.SEG_DIR, pe.RUNTIME_ROOT, pe.MINIMUM_FREE_BYTES
            try:
                pe.SEG_DIR = seg_dir
                pe.RUNTIME_ROOT = root
                pe.MINIMUM_FREE_BYTES = 0
                result = pe._evaluate_pass(
                    roster=roster,
                    cases=cases,
                    paths=paths,
                    recipes={"uniform": {candidate_id: 1 for candidate_id in candidate_ids}},
                    partial_path=root / "partial.json",
                    job_path=job_path,
                    chunk_elements=1,
                )
            finally:
                pe.SEG_DIR, pe.RUNTIME_ROOT, pe.MINIMUM_FREE_BYTES = old_seg, old_runtime, old_minimum
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["summaries"]["uniform"]["findings"], 381)
            self.assertEqual(result["summaries"]["uniform"]["dice"], 1.0)
            self.assertGreater(result["timing"]["active_wall_seconds"], 0)
            resumed = pe._evaluate_pass(
                roster=roster,
                cases=cases,
                paths=paths,
                recipes={"uniform": {candidate_id: 1 for candidate_id in candidate_ids}},
                partial_path=root / "partial.json",
                job_path=job_path,
                chunk_elements=1,
            )
            self.assertEqual(len(resumed["rows"]["uniform"]), 381)
            self.assertEqual(resumed["summaries"]["uniform"]["dice"], 1.0)
            self.assertFalse(any(root.glob("**/*ensemble*.npy")))


if __name__ == "__main__":
    unittest.main()
