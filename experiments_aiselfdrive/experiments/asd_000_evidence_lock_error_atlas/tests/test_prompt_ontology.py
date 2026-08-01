from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

import prompt_ontology as ontology  # noqa: E402


class ConservativeParserTests(unittest.TestCase):
    def assert_unknown(self, text: str, expected_group: str = "unknown") -> None:
        parsed = ontology.parse_prompt(text)
        self.assertEqual(parsed.group, expected_group, text)
        self.assertFalse(parsed.is_restrictive, text)
        self.assertIsNone(parsed.restrictive_target, text)
        self.assertEqual(parsed.confidence, "unknown", text)

    def test_defaults_to_unknown(self) -> None:
        for text in (
            "",
            "Pulmonary nodule",
            "Ground-glass opacity",
            "A finding near an unspecified structure",
        ):
            with self.subTest(text=text):
                self.assert_unknown(text)

    def test_exact_lobe_literal_is_restrictive(self) -> None:
        parsed = ontology.parse_prompt(
            "Peripheral opacity in the posterior right lower lobe"
        )
        self.assertEqual(parsed.group, "exact_lobe")
        self.assertTrue(parsed.is_restrictive)
        self.assertEqual(parsed.target["laterality"], "right")
        self.assertEqual(parsed.target["lobe"], "right_lower")
        self.assertEqual(parsed.target["radial"], "peripheral")

    def test_exact_lobe_alternate_word_order_and_lingula(self) -> None:
        alternate = ontology.parse_prompt("Nodule in the upper lobe of the left lung")
        self.assertEqual(alternate.group, "exact_lobe")
        self.assertEqual(alternate.target["lobe"], "left_upper")

        lingula = ontology.parse_prompt("Linear atelectasis in the lingular segment")
        self.assertEqual(lingula.group, "exact_lobe")
        self.assertEqual(lingula.target["lobe"], "left_upper")
        self.assertEqual(lingula.target["laterality"], "left")

    def test_safe_literal_groups(self) -> None:
        fixtures = {
            "Small right pleural effusion": (
                "laterality_only",
                {"laterality": "right"},
            ),
            "Apical fibrotic band": (
                "apical_or_basal",
                {"vertical": "apical"},
            ),
            "Peripheral pulmonary opacity": (
                "central_or_peripheral",
                {"radial": "peripheral"},
            ),
            "Subpleural nodule": (
                "subpleural",
                {"relation": "subpleural"},
            ),
            "Perihilar opacity": ("hilar", {"relation": "hilar"}),
            "Peribronchial thickening": (
                "peribronchial",
                {"relation": "peribronchial"},
            ),
        }
        for text, (group, expected_target) in fixtures.items():
            with self.subTest(text=text):
                parsed = ontology.parse_prompt(text)
                self.assertEqual(parsed.group, group)
                self.assertTrue(parsed.is_restrictive)
                for key, value in expected_target.items():
                    self.assertEqual(parsed.target[key], value)

    def test_negation_and_ambiguity_are_unknown(self) -> None:
        fixtures = (
            "No nodule in the right upper lobe",
            "Opacity is not in the left lower lobe",
            "Possible right lower lobe nodule",
            "Nodule in either the right or left lung",
            "Indeterminate opacity in the left upper lobe",
            "Subcentimeter nodule in the left lower lobe, difficult to distinguish from a vessel",
        )
        for text in fixtures:
            with self.subTest(text=text):
                self.assert_unknown(text)

    def test_bilateral_multifocal_and_diffuse_are_nonrestrictive(self) -> None:
        bilateral_fixtures = (
            "Bilateral upper-lobe opacities",
            "Fibrosis at the apices of both upper lobes",
            "Nodules in both lungs",
            "Multiple nodules in the right upper lobe",
            "Multilobar right lung consolidation",
            "Multifocal right lower lobe consolidation",
            "Right and left pleural effusions",
        )
        for text in bilateral_fixtures:
            with self.subTest(text=text):
                self.assert_unknown(text, "bilateral_or_multifocal")

        for text in (
            "Diffuse opacity in the right lung",
            "Ground-glass throughout the left lung",
            "Widespread subpleural reticulation",
        ):
            with self.subTest(text=text):
                self.assert_unknown(text, "diffuse")

    def test_conflicting_literal_spans_are_unknown(self) -> None:
        fixtures = (
            "Right upper lobe and left lower lobe nodules",
            "Central and peripheral right lung opacities",
            "Apical and basal left lung scarring",
            "Subpleural and perihilar right lung nodules",
            "Right upper and middle lobes",
            "Upper and lower lobes of the right lung",
        )
        for text in fixtures:
            with self.subTest(text=text):
                self.assert_unknown(text)

    def test_mapping_and_dictionary_interfaces_are_consistent(self) -> None:
        parsed = ontology.parse_prompt("Left lower lobe nodule")
        as_dict = ontology.classify_prompt("Left lower lobe nodule")
        self.assertEqual(dict(parsed), as_dict)
        self.assertEqual(parsed["group"], "exact_lobe")
        self.assertEqual(as_dict["restrictive_target"]["laterality"], "left")

    def test_manifest_is_deterministic_and_atomic_writer_compatible(self) -> None:
        first = ontology.build_ontology_manifest()
        second = ontology.build_ontology_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["default_group"], "unknown")
        self.assertEqual(len(first["payload_sha256"]), 64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ontology.json"
            written = ontology.write_ontology_manifest(path)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                written,
            )


class RasGeometryTests(unittest.TestCase):
    def test_identity_ras_affine_uses_physical_x_for_laterality(self) -> None:
        affine = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(ontology.voxel_to_ras((3, 4, 5), affine), (3.0, 4.0, 5.0))
        self.assertEqual(
            ontology.voxel_laterality((3, 4, 5), affine, midline_ras_x=1.0),
            "right",
        )
        self.assertEqual(
            ontology.voxel_laterality((-3, 4, 5), affine, midline_ras_x=1.0),
            "left",
        )
        self.assertEqual(
            ontology.voxel_laterality((1, 4, 5), affine, midline_ras_x=1.0),
            "midline",
        )

    def test_flipped_voxel_axis_does_not_flip_physical_laterality(self) -> None:
        # Increasing voxel i moves toward physical left because RAS x decreases.
        affine = (
            (-2.0, 0.0, 0.0, 10.0),
            (0.0, 3.0, 0.0, -6.0),
            (0.0, 0.0, 4.0, 2.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(ontology.voxel_to_ras((1, 2, 3), affine), (8.0, 0.0, 14.0))
        self.assertEqual(
            ontology.voxel_laterality((1, 2, 3), affine, midline_ras_x=5.0),
            "right",
        )
        self.assertEqual(
            ontology.voxel_laterality((4, 2, 3), affine, midline_ras_x=5.0),
            "left",
        )
        axis = ontology.left_right_axis_from_affine(affine)
        self.assertEqual(axis.voxel_axis, 0)
        self.assertEqual(axis.positive_voxel_direction, "left")
        self.assertEqual(axis.ras_x_mm_per_voxel, -2.0)

    def test_oblique_transform_uses_full_affine(self) -> None:
        affine = (
            (2.0, 0.5, 0.0, -5.0),
            (0.0, 2.0, 0.0, 1.0),
            (0.0, 0.0, 2.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        ras = ontology.physical_point_from_voxel((2, 4, 1), affine)
        self.assertEqual(ras, (1.0, 9.0, 5.0))
        self.assertEqual(
            ontology.physical_laterality(ras, midline_ras_x=0.0),
            "right",
        )

    def test_invalid_affines_and_laterality_inputs_are_rejected(self) -> None:
        invalid_affines = (
            ((1, 0), (0, 1)),
            (
                (0, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ),
            (
                (1, 0, 0, 0),
                (0, math.inf, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ),
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 1, 1),
            ),
        )
        for affine in invalid_affines:
            with self.subTest(affine=affine):
                with self.assertRaises(ontology.PromptOntologyError):
                    ontology.validate_ras_affine(affine)

        with self.assertRaises(ontology.PromptOntologyError):
            ontology.ras_laterality(1.0, tolerance_mm=-1.0)


if __name__ == "__main__":
    unittest.main()
