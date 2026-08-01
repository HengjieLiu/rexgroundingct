from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_stage.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "asd000_run_stage_test_module", SCRIPT_PATH
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)

from error_atlas import (  # noqa: E402
    AtlasInvariantError,
    DEFAULT_BOOTSTRAP_DRAWS,
    ErrorAtlasError,
    SupervisionLabel,
    UnverifiedLogitsError,
    Val120AccessError,
    build_error_atlas,
    build_tristate_supervision,
    component_records,
    deterministic_case_cluster_bootstrap,
    finding_metrics,
    label_components_26,
    partition_prompt_off_location_fp,
    reconstruct_raw_mask,
    split_tristate_supervision,
    summarize_prompt_off_location_fp,
    threshold_curves,
    verify_raw_mask_reconstruction,
    verify_threshold_mask_equivalence,
    voxel_indices_to_world,
    voxel_volume_mm3,
)


class PhysicalGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.affine = np.asarray(
            [
                [-2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, -6.0],
                [0.0, 0.0, 4.0, 5.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def test_affine_controls_world_coordinates_and_volume(self) -> None:
        points = voxel_indices_to_world(
            np.asarray([[0, 0, 0], [1, 2, 3]]), self.affine
        )
        np.testing.assert_allclose(points[0], [10.0, -6.0, 5.0])
        np.testing.assert_allclose(points[1], [8.0, 0.0, 17.0])
        self.assertAlmostEqual(voxel_volume_mm3(self.affine), 24.0)

    def test_component_records_use_26_connectivity_and_physical_centroids(
        self,
    ) -> None:
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[0, 0, 0] = True
        mask[1, 1, 1] = True
        mask[3, 3, 3] = True
        labels, count = label_components_26(mask)
        self.assertEqual(count, 2)
        self.assertEqual(labels[0, 0, 0], labels[1, 1, 1])
        self.assertNotEqual(labels[1, 1, 1], labels[3, 3, 3])

        records = component_records(mask, self.affine)
        self.assertEqual([record["component_id"] for record in records], [1, 2])
        self.assertEqual(records[0]["voxel_count"], 2)
        self.assertAlmostEqual(records[0]["volume_mm3"], 48.0)
        np.testing.assert_allclose(
            records[0]["centroid_world_xyz_mm"], [9.0, -4.5, 7.0]
        )

    def test_raw_mask_component_reconstruction_is_exact_and_strict(self) -> None:
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        mask[0, 0, 0] = 1
        mask[1, 1, 1] = 1
        labels, _ = label_components_26(mask)
        np.testing.assert_array_equal(reconstruct_raw_mask(labels), mask.astype(bool))
        report = verify_raw_mask_reconstruction(mask, labels)
        self.assertTrue(report["exact"])
        self.assertEqual(report["component_count"], 1)

        tampered = labels.copy()
        tampered[1, 1, 1] = 0
        report = verify_raw_mask_reconstruction(
            mask, tampered, require_exact=False
        )
        self.assertFalse(report["exact"])
        self.assertEqual(report["mismatch_voxel_count"], 1)
        with self.assertRaises(AtlasInvariantError):
            verify_raw_mask_reconstruction(mask, tampered)

        split_component = labels.copy()
        split_component[1, 1, 1] = 2
        split_report = verify_raw_mask_reconstruction(
            mask, split_component, require_exact=False
        )
        self.assertEqual(split_report["mismatch_voxel_count"], 0)
        self.assertFalse(split_report["component_partition_exact"])
        with self.assertRaises(AtlasInvariantError):
            verify_raw_mask_reconstruction(mask, split_component)
        with self.assertRaises(ErrorAtlasError):
            verify_raw_mask_reconstruction(np.full((2, 2, 2), 0.5))


class ConservativeSupervisionTests(unittest.TestCase):
    def test_unlabeled_remainder_stays_unknown(self) -> None:
        positive = np.zeros((2, 2, 2), dtype=bool)
        certified_negative = np.zeros_like(positive)
        positive[0, 0, 0] = True
        certified_negative[1, 1, 1] = True

        labels = build_tristate_supervision(positive, certified_negative)
        split = split_tristate_supervision(labels)
        self.assertEqual(
            int(labels[0, 0, 0]), int(SupervisionLabel.KNOWN_POSITIVE)
        )
        self.assertEqual(
            int(labels[1, 1, 1]), int(SupervisionLabel.CERTIFIED_NEGATIVE)
        )
        self.assertEqual(int(labels[0, 1, 1]), int(SupervisionLabel.UNKNOWN))
        self.assertEqual(np.count_nonzero(split["unknown"]), 6)
        self.assertFalse(np.array_equal(split["certified_negative"], ~positive))

    def test_overlap_or_incomplete_explicit_tristate_is_rejected(self) -> None:
        positive = np.zeros((2, 2, 2), dtype=bool)
        negative = np.zeros_like(positive)
        positive[0, 0, 0] = True
        negative[0, 0, 0] = True
        with self.assertRaises(AtlasInvariantError):
            build_tristate_supervision(positive, negative)

        negative[0, 0, 0] = False
        explicit_unknown = np.zeros_like(positive)
        with self.assertRaises(AtlasInvariantError):
            build_tristate_supervision(
                positive, negative, unknown_mask=explicit_unknown
            )

    def test_prompt_partitions_keep_compatible_predictions_unknown(self) -> None:
        shape = (3, 3, 3)
        prediction = np.zeros(shape, dtype=bool)
        positive = np.zeros(shape, dtype=bool)
        target = np.zeros(shape, dtype=bool)
        ipsilateral = np.zeros(shape, dtype=bool)
        contralateral = np.zeros(shape, dtype=bool)
        outside_lung = np.zeros(shape, dtype=bool)
        other_certified = np.zeros(shape, dtype=bool)

        positive[0, 0, 0] = True
        target[0, 0, 0] = True
        target[0, 0, 1] = True
        ipsilateral[1, 0, 0] = True
        contralateral[2, 0, 0] = True
        outside_lung[2, 2, 2] = True
        other_certified[0, 2, 2] = True
        for index in (
            (0, 0, 0),
            (0, 0, 1),
            (1, 0, 0),
            (2, 0, 0),
            (2, 2, 2),
            (0, 2, 2),
            (1, 1, 2),
        ):
            prediction[index] = True
        certified = ipsilateral | contralateral | outside_lung | other_certified

        partitions = partition_prompt_off_location_fp(
            prediction,
            positive,
            prompt_target_mask=target,
            ipsilateral_nontarget_mask=ipsilateral,
            contralateral_mask=contralateral,
            outside_lung_mask=outside_lung,
            certified_negative_mask=certified,
        )
        self.assertTrue(partitions["unknown_prediction"][0, 0, 1])
        self.assertTrue(partitions["unknown_prediction"][1, 1, 2])
        self.assertFalse(partitions["ipsilateral_nontarget_fp"][0, 0, 1])
        self.assertEqual(
            sum(np.count_nonzero(mask) for mask in partitions.values()), 7
        )

        summary = summarize_prompt_off_location_fp(
            partitions, np.diag([2.0, 2.0, 2.0, 1.0])
        )
        self.assertEqual(summary["off_location_fp_voxel_count"], 3)
        self.assertAlmostEqual(summary["off_location_fp_volume_mm3"], 24.0)
        self.assertEqual(summary["certified_negative_fp_voxel_count"], 4)
        self.assertAlmostEqual(summary["prompt_off_location_fp_fraction"], 3 / 7)
        self.assertAlmostEqual(summary["certified_precision"], 1 / 5)

        invalid_certified = certified.copy()
        invalid_certified[0, 0, 0] = True
        with self.assertRaises(AtlasInvariantError):
            partition_prompt_off_location_fp(
                prediction,
                positive,
                prompt_target_mask=target,
                ipsilateral_nontarget_mask=ipsilateral,
                contralateral_mask=contralateral,
                outside_lung_mask=outside_lung,
                certified_negative_mask=invalid_certified,
            )

    def test_finding_metrics_expose_unknown_and_certified_counts_separately(
        self,
    ) -> None:
        prediction = np.zeros((2, 2, 2), dtype=bool)
        positive = np.zeros_like(prediction)
        certified = np.zeros_like(prediction)
        prediction[0, 0, 0] = True
        prediction[0, 0, 1] = True
        prediction[1, 1, 1] = True
        positive[0, 0, 0] = True
        certified[1, 1, 1] = True
        metrics = finding_metrics(
            prediction,
            positive,
            np.eye(4),
            hit_threshold=0.1,
            prompt_target_mask=positive,
            outside_lung_mask=certified,
            certified_negative_mask=certified,
        )
        self.assertEqual(metrics["unknown_prediction_voxels"], 1)
        self.assertEqual(metrics["certified_negative_fp_voxels"], 1)
        self.assertEqual(metrics["supervision_unknown_voxels"], 6)
        self.assertAlmostEqual(metrics["certified_precision"], 0.5)
        self.assertTrue(metrics["raw_mask_reconstruction_exact"])


class VerifiedThresholdCurveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = np.full((2, 2, 2), 0.1, dtype=np.float64)
        self.scores[0, 0, 0] = 0.9
        self.scores[0, 0, 1] = 0.8
        self.scores[0, 1, 0] = 0.7
        self.positive = np.zeros_like(self.scores, dtype=bool)
        self.positive[0, 0, 0] = True
        self.certified = np.zeros_like(self.positive)
        self.certified[0, 0, 1] = True

    def test_unverified_scores_cannot_create_curves(self) -> None:
        self.assertIsNone(
            threshold_curves(
                None,
                [0.5],
                self.positive,
                np.eye(4),
                logits_verified=False,
                score_kind="probability",
                hit_threshold=0.1,
            )
        )
        with self.assertRaises(UnverifiedLogitsError):
            threshold_curves(
                self.scores,
                [0.5],
                self.positive,
                np.eye(4),
                logits_verified=False,
                score_kind="probability",
                hit_threshold=0.1,
            )

    def test_verified_curves_preserve_unknown_predictions(self) -> None:
        raw_mask = self.scores >= 0.5
        equivalence = verify_threshold_mask_equivalence(
            self.scores,
            raw_mask,
            threshold=0.5,
            score_kind="probability",
            logits_verified=True,
        )
        self.assertTrue(equivalence["exact"])
        curves = threshold_curves(
            self.scores,
            [0.85, 0.5],
            self.positive,
            np.eye(4),
            logits_verified=True,
            score_kind="probability",
            hit_threshold=0.1,
            certified_negative_mask=self.certified,
        )
        self.assertIsNotNone(curves)
        assert curves is not None
        self.assertEqual([row["threshold"] for row in curves], [0.5, 0.85])
        self.assertEqual(curves[0]["prediction_voxels"], 3)
        self.assertEqual(curves[0]["certified_negative_fp_voxels"], 1)
        self.assertEqual(curves[0]["unknown_prediction_voxels"], 1)
        self.assertAlmostEqual(curves[0]["certified_precision"], 0.5)

        wrong_mask = raw_mask.copy()
        wrong_mask[1, 1, 1] = True
        with self.assertRaises(AtlasInvariantError):
            verify_threshold_mask_equivalence(
                self.scores,
                wrong_mask,
                threshold=0.5,
                score_kind="probability",
                logits_verified=True,
            )


class ClusterBootstrapTests(unittest.TestCase):
    def test_case_cluster_bootstrap_is_seeded_and_order_invariant(self) -> None:
        rows = [
            {"case_id": "case_b", "metric": 1.0},
            {"case_id": "case_a", "metric": 0.0},
            {"case_id": "case_b", "metric": 0.8},
            {"case_id": "case_c", "metric": 0.2},
        ]
        first = deterministic_case_cluster_bootstrap(
            rows, ["metric"], seed=17, draws=512
        )
        reordered = deterministic_case_cluster_bootstrap(
            list(reversed(rows)), ["metric"], seed=17, draws=512
        )
        self.assertEqual(first, reordered)
        self.assertEqual(first["case_ids_sorted"], ["case_a", "case_b", "case_c"])
        self.assertEqual(first["draws"], 512)
        self.assertEqual(first["metrics"]["metric"]["valid_draws"], 512)

        other_seed = deterministic_case_cluster_bootstrap(
            rows, ["metric"], seed=18, draws=512
        )
        self.assertNotEqual(
            first["metrics"]["metric"]["bootstrap_mean"],
            other_seed["metrics"]["metric"]["bootstrap_mean"],
        )
        self.assertEqual(DEFAULT_BOOTSTRAP_DRAWS, 10_000)

    def test_nonfinite_bootstrap_values_are_rejected(self) -> None:
        with self.assertRaises(ErrorAtlasError):
            deterministic_case_cluster_bootstrap(
                [{"case_id": "case", "metric": float("nan")}],
                ["metric"],
                seed=3,
                draws=8,
            )


class AtlasBuilderTests(unittest.TestCase):
    @staticmethod
    def _finding(finding_id: str, prediction_indices: tuple[tuple[int, ...], ...]):
        prediction = np.zeros((2, 2, 2), dtype=bool)
        positive = np.zeros_like(prediction)
        positive[0, 0, 0] = True
        for index in prediction_indices:
            prediction[index] = True
        return {
            "finding_id": finding_id,
            "prediction_mask": prediction,
            "known_positive_mask": positive,
        }

    def test_builder_is_json_safe_sorted_and_omits_unavailable_curves(self) -> None:
        cases = [
            {
                "case_id": "case_b",
                "cohort": "val80",
                "affine": np.eye(4),
                "findings": [
                    self._finding("finding_2", ((0, 0, 0), (1, 1, 1))),
                ],
            },
            {
                "case_id": "case_a",
                "affine": np.eye(4),
                "findings": [
                    self._finding("finding_3", ()),
                    self._finding("finding_1", ((0, 0, 0),)),
                ],
            },
        ]
        atlas = build_error_atlas(
            cases,
            bootstrap_seed=11,
            bootstrap_draws=128,
            primary_threshold=0.5,
            hit_threshold=0.1,
            connectivity=26,
            thresholds=[0.5],
            expected_case_count=2,
            expected_finding_count=3,
        )
        self.assertEqual(atlas["case_count"], 2)
        self.assertEqual(atlas["finding_count"], 3)
        self.assertEqual(
            [
                (row["case_id"], row["finding_id"])
                for row in atlas["rows"]
            ],
            [
                ("case_a", "finding_1"),
                ("case_a", "finding_3"),
                ("case_b", "finding_2"),
            ],
        )
        self.assertFalse(atlas["logit_status"]["threshold_curves_included"])
        self.assertNotIn("threshold_curves", atlas)
        json.dumps(atlas, allow_nan=False)

        with self.assertRaises(AtlasInvariantError):
            build_error_atlas(
                cases,
                bootstrap_seed=11,
                bootstrap_draws=8,
                primary_threshold=0.5,
                hit_threshold=0.1,
                expected_case_count=80,
            )


class RunnerContractTests(unittest.TestCase):
    @staticmethod
    def _write_nifti(path: Path, array: np.ndarray) -> None:
        header = bytearray(352)
        struct.pack_into("<i", header, 0, 348)
        dimensions = [array.ndim, *array.shape, *([1] * (7 - array.ndim))]
        struct.pack_into("<8h", header, 40, *dimensions)
        struct.pack_into("<h", header, 70, 2)
        struct.pack_into("<h", header, 72, 8)
        struct.pack_into("<8f", header, 76, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        struct.pack_into("<f", header, 108, 352.0)
        struct.pack_into("<h", header, 254, 1)
        struct.pack_into("<4f", header, 280, 1.0, 0.0, 0.0, 0.0)
        struct.pack_into("<4f", header, 296, 0.0, 1.0, 0.0, 0.0)
        struct.pack_into("<4f", header, 312, 0.0, 0.0, 1.0, 0.0)
        header[344:348] = b"n+1\x00"
        with gzip.open(path, "wb") as stream:
            stream.write(header)
            stream.write(array.tobytes(order="F"))

    def test_bounded_nifti_reader_preserves_f_order_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.nii.gz"
            expected = np.arange(24, dtype=np.uint8).reshape((2, 3, 4), order="F")
            self._write_nifti(path, expected)
            header = RUNNER.read_nifti_header(path)
            self.assertEqual(header.shape, (2, 3, 4))
            self.assertEqual(header.dtype, np.dtype("u1"))
            np.testing.assert_array_equal(RUNNER.read_nifti_data(header), expected)
            np.testing.assert_array_equal(header.affine, np.eye(4))

    def test_metadata_projection_never_materializes_label_metrics(self) -> None:
        metadata = {
            "case": {
                "case_key": "case_a",
                "ct_path": "/declared/ct/case_a.nii.gz",
                "finding_count": 1,
                "findings": {"0": "secret label"},
                "categories": {"0": "secret category"},
                "name": "case_a.nii.gz",
                "source_axcodes": ["R", "A", "S"],
                "source_shape_xyz": [2, 3, 4],
                "source_spacing_xyz_mm": [1.0, 1.0, 1.0],
                "val_order": 0,
            },
            "class_map": {"10": "lung_upper_lobe_left"},
            "geometry": {
                "class_volume_mm3": {"10": 123.0},
                "class_voxels": {"10": 123},
                "observed_label_ids": [10],
                "source_affine": np.eye(4).tolist(),
                "output_affine": np.eye(4).tolist(),
                "source_axcodes": ["R", "A", "S"],
                "output_axcodes": ["R", "A", "S"],
                "source_shape_xyz": [2, 3, 4],
                "output_shape_xyz": [2, 3, 4],
                "source_spacing_xyz_mm": [1.0, 1.0, 1.0],
                "output_spacing_xyz_mm": [1.0, 1.0, 1.0],
            },
            "output_sha256": "a" * 64,
            "statistics": {"lung": {"volume": 123}},
            "task": "total",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            projected = RUNNER._load_anatomy_metadata_projection(path)
        serialized = json.dumps(projected, sort_keys=True)
        for forbidden in (
            "findings",
            "categories",
            "class_map",
            "class_volume_mm3",
            "class_voxels",
            "observed_label_ids",
            "statistics",
            "secret",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(projected["case"]["name"], "case_a.nii.gz")

    def test_resolved_path_self_evidence_is_narrowly_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "result.json"
            other = root / "other.json"
            other.write_text("{}\n", encoding="utf-8")
            stage = {
                "id": "synthetic_stage",
                "outputs": [
                    {"id": "self", "path": str(root / "." / "result.json")},
                    {"id": "other", "path": str(other)},
                ],
                "completion": {"evidence_file": str(evidence)},
                "acceptance_checks": ["synthetic_check"],
            }
            RUNNER._write_passed_evidence(
                stage, summary="synthetic", metrics={"ok": True}
            )
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual([item["id"] for item in payload["artifacts"]], ["other"])
            self.assertTrue(RUNNER._is_within(other, root))
            self.assertFalse(RUNNER._is_within(root.parent / "outside", root))

    def test_exact_geometry_and_closeout_contract_fail_closed(self) -> None:
        affine = np.eye(4).tolist()
        metadata = {
            "case": {
                "name": "case_a.nii.gz",
                "source_shape_xyz": [2, 3, 4],
                "source_spacing_xyz_mm": [1.0, 1.0, 1.0],
                "source_axcodes": ["R", "A", "S"],
            },
            "geometry": {
                "source_shape_xyz": [2, 3, 4],
                "output_shape_xyz": [2, 3, 4],
                "source_affine": affine,
                "output_affine": affine,
                "source_spacing_xyz_mm": [1.0, 1.0, 1.0],
                "output_spacing_xyz_mm": [1.0, 1.0, 1.0],
                "source_axcodes": ["R", "A", "S"],
                "output_axcodes": ["R", "A", "S"],
            },
        }
        result = RUNNER._validate_geometry_mapping(
            metadata,
            case_name="case_a.nii.gz",
            source_header=None,
            output_header=None,
        )
        self.assertEqual(result["left_right_axis"]["positive_voxel_direction"], "right")
        tampered = json.loads(json.dumps(metadata))
        tampered["geometry"]["output_shape_xyz"] = [2, 3, 5]
        with self.assertRaises(RUNNER.StageContractError):
            RUNNER._validate_geometry_mapping(
                tampered,
                case_name="case_a.nii.gz",
                source_header=None,
                output_header=None,
            )

        stage = {"id": "prior", "acceptance_checks": ["gate_a"]}
        evidence = {
            "experiment_id": RUNNER.EXPERIMENT_ID,
            "stage_id": "prior",
            "status": "passed",
            "checks": [{"id": "gate_a", "status": "passed"}],
            "artifacts": [],
        }
        RUNNER._validate_prior_evidence_contract(stage, evidence)
        evidence["checks"][0]["status"] = "failed"
        with self.assertRaises(RUNNER.StageContractError):
            RUNNER._validate_prior_evidence_contract(stage, evidence)

    def test_builder_rejects_val120_before_processing(self) -> None:
        with self.assertRaises(Val120AccessError):
            build_error_atlas(
                [],
                bootstrap_seed=11,
                primary_threshold=0.5,
                hit_threshold=0.1,
                cohort="val120",
            )

    def test_builder_includes_curves_only_for_complete_verified_logits(self) -> None:
        logits = np.full((2, 2, 2), -2.0, dtype=np.float64)
        logits[0, 0, 0] = 2.0
        logits[0, 0, 1] = 1.0
        prediction = logits >= 0.0
        positive = np.zeros_like(prediction)
        positive[0, 0, 0] = True
        records = [
            {
                "case_id": "case_a",
                "finding_id": "finding_1",
                "prediction_mask": prediction,
                "known_positive_mask": positive,
                "affine": np.eye(4),
                "logits": logits,
                "logits_verified": True,
            }
        ]
        atlas = build_error_atlas(
            records,
            bootstrap_seed=7,
            bootstrap_draws=32,
            primary_threshold=0.5,
            hit_threshold=0.1,
            thresholds=[0.5, 0.8],
        )
        self.assertTrue(atlas["logit_status"]["verified_complete"])
        self.assertEqual(
            [row["threshold"] for row in atlas["threshold_curves"]],
            [0.5, 0.8],
        )
        json.dumps(atlas, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
