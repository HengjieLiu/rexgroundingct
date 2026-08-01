from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

import cohorts  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(name: str, finding_count: int = 1) -> dict[str, object]:
    return {
        "name": name,
        "seg_path": name,
        "findings": {
            str(index): f"finding {index}" for index in range(finding_count)
        },
        "categories": {str(index): "2d" for index in range(finding_count)},
    }


class CaseIdentityReaderTests(unittest.TestCase):
    def test_reads_only_direct_case_names_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.json"
            _write_json(
                path,
                {
                    "metadata": {"name": "not-a-case"},
                    "test": [
                        {
                            "name": "case-b.nii.gz",
                            "findings": {
                                "0": {
                                    "name": "nested-name-must-not-be-selected",
                                    "text": "right lower lobe",
                                }
                            },
                        },
                        {"name": "case-a.nii.gz", "findings": {"0": "left lung"}},
                    ],
                },
            )
            self.assertEqual(
                cohorts.load_case_names_only(path),
                ["case-b.nii.gz", "case-a.nii.gz"],
            )

    def test_rejects_duplicate_case_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.json"
            _write_json(path, {"test": [{"name": "x"}, {"name": "x"}]})
            with self.assertRaisesRegex(cohorts.CohortError, "duplicate"):
                cohorts.load_case_names_only(path)

    def test_rejects_malformed_and_trailing_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.json"
            path.write_text('{"test":[{"name":"x"}]} null', encoding="utf-8")
            with self.assertRaisesRegex(cohorts.CohortError, "trailing"):
                cohorts.load_case_names_only(path)


class PartitionTests(unittest.TestCase):
    def test_complement_is_disjoint_exhaustive_and_parent_ordered(self) -> None:
        parent = ["c3", "c1", "c4", "c2"]
        development = ["c2", "c3"]
        partition = cohorts.validate_partition(
            parent,
            development,
            expected_val200_cases=4,
            expected_val80_cases=2,
            expected_val120_cases=2,
        )
        self.assertEqual(partition.confirmatory_case_names, ("c1", "c4"))
        self.assertEqual(
            set(partition.development_case_names)
            | set(partition.confirmatory_case_names),
            set(parent),
        )
        self.assertFalse(
            set(partition.development_case_names)
            & set(partition.confirmatory_case_names)
        )

    def test_rejects_development_case_outside_parent(self) -> None:
        with self.assertRaisesRegex(cohorts.CohortError, "not a subset"):
            cohorts.derive_case_complement(["a", "b"], ["a", "outside"])

    def test_rejects_count_drift(self) -> None:
        with self.assertRaisesRegex(cohorts.CohortError, "expected 3"):
            cohorts.validate_partition(
                ["a", "b"], ["a"], expected_val200_cases=3
            )


class CohortSealingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.val200 = self.root / "val200.json"
        self.val80 = self.root / "val80.json"
        self.val80_output = self.root / "val80.locked.json"
        self.val120_output = self.root / "val120.sealed.json"

        parent_records = [
            _case("case-c.nii.gz", 2),
            _case("case-a.nii.gz"),
            _case("case-e.nii.gz"),
            _case("case-b.nii.gz"),
            _case("case-d.nii.gz"),
        ]
        # A nested name proves the parent identity reader does not select
        # labels or nested values as case identities.
        parent_records[1]["extra"] = {"name": "nested-not-a-case"}
        development_records = [parent_records[3], parent_records[0]]
        _write_json(self.val200, {"test": parent_records})
        _write_json(self.val80, {"test": development_records})

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _seal(self) -> dict[str, object]:
        return cohorts.seal_cohorts(
            self.val200,
            self.val80,
            self.val80_output,
            self.val120_output,
            expected_val200_cases=5,
            expected_val80_cases=2,
            expected_val80_findings=3,
            expected_val120_cases=3,
            expected_val200_sha256=_sha256(self.val200),
            expected_val80_sha256=_sha256(self.val80),
            declared_val200_findings=6,
        )

    def test_seals_identity_only_complement_and_full_development_records(self) -> None:
        result = self._seal()
        development = json.loads(self.val80_output.read_text(encoding="utf-8"))
        confirmatory = json.loads(self.val120_output.read_text(encoding="utf-8"))

        self.assertEqual(
            development["case_names"],
            ["case-b.nii.gz", "case-c.nii.gz"],
        )
        self.assertEqual([case["name"] for case in development["cases"]], development["case_names"])
        self.assertEqual(development["finding_count"], 3)
        self.assertEqual(
            confirmatory["case_names"],
            ["case-a.nii.gz", "case-e.nii.gz", "case-d.nii.gz"],
        )
        self.assertEqual(confirmatory["case_count"], 3)
        self.assertTrue(result["val120"]["identity_only"])
        cohorts.assert_identity_only_manifest(confirmatory)

        serialized = json.dumps(confirmatory, sort_keys=True).casefold()
        for forbidden in (
            '"findings"',
            '"categories"',
            '"seg_path"',
            '"labels"',
            '"predictions"',
            '"metrics"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_outputs_are_byte_deterministic(self) -> None:
        first = self._seal()
        first_val80 = self.val80_output.read_bytes()
        first_val120 = self.val120_output.read_bytes()
        second = self._seal()
        self.assertEqual(self.val80_output.read_bytes(), first_val80)
        self.assertEqual(self.val120_output.read_bytes(), first_val120)
        self.assertEqual(first, second)

    def test_hash_mismatch_fails_before_outputs(self) -> None:
        with self.assertRaisesRegex(cohorts.CohortError, "SHA-256 mismatch"):
            cohorts.seal_cohorts(
                self.val200,
                self.val80,
                self.val80_output,
                self.val120_output,
                expected_val200_cases=5,
                expected_val80_cases=2,
                expected_val80_findings=3,
                expected_val120_cases=3,
                expected_val200_sha256="0" * 64,
            )
        self.assertFalse(self.val80_output.exists())
        self.assertFalse(self.val120_output.exists())

    def test_api_cannot_accept_a_val120_input(self) -> None:
        parameters = inspect.signature(cohorts.seal_cohorts).parameters
        self.assertNotIn("val120_path", parameters)
        self.assertNotIn("val120_labels", parameters)
        self.assertNotIn("val120_predictions", parameters)
        self.assertNotIn("val120_metrics", parameters)

    def test_identity_guard_rejects_forbidden_nested_fields(self) -> None:
        with self.assertRaisesRegex(cohorts.CohortError, "forbidden key"):
            cohorts.assert_identity_only_manifest(
                {"case_names": ["a"], "private": {"metrics": {"dice": 1.0}}}
            )


if __name__ == "__main__":
    unittest.main()
