from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

from hardware_contract import (  # noqa: E402
    atomic_runtime_durability_probe,
    validate_cuda_inventory,
)


class CudaInventoryTests(unittest.TestCase):
    expected_name = "NVIDIA RTX 6000 Ada Generation"

    def test_exact_four_expected_devices_pass(self) -> None:
        result = validate_cuda_inventory(
            [self.expected_name] * 4,
            expected_count=4,
            expected_name=self.expected_name,
            require_exact=True,
        )
        self.assertTrue(result["cuda_inventory_exact"])
        self.assertEqual(result["visible_cuda_device_count"], 4)

    def test_count_and_name_drift_fail_exact_preflight(self) -> None:
        for names in (
            [self.expected_name] * 3,
            [self.expected_name] * 3 + ["NVIDIA H100 80GB HBM3"],
        ):
            with self.subTest(names=names), self.assertRaises(RuntimeError):
                validate_cuda_inventory(
                    names,
                    expected_count=4,
                    expected_name=self.expected_name,
                    require_exact=True,
                )

    def test_single_selected_device_is_allowed_only_after_exact_preflight(self) -> None:
        result = validate_cuda_inventory(
            [self.expected_name],
            expected_count=4,
            expected_name=self.expected_name,
            require_exact=False,
        )
        self.assertFalse(result["cuda_inventory_exact"])


class RuntimeDurabilityTests(unittest.TestCase):
    def test_probe_performs_durable_round_trip_and_leaves_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            proof = atomic_runtime_durability_probe(root)
            self.assertEqual(proof["status"], "passed")
            self.assertEqual(proof["runtime_root"], str(root))
            for key in (
                "file_fsync",
                "atomic_replace",
                "directory_fsync_after_replace",
                "readback_exact",
                "delete_verified",
                "directory_fsync_after_delete",
            ):
                self.assertIs(proof[key], True)
            self.assertEqual(list(root.iterdir()), [])

    def test_probe_rejects_relative_runtime_root(self) -> None:
        with self.assertRaises(ValueError):
            atomic_runtime_durability_probe(Path("relative-runtime"))


if __name__ == "__main__":
    unittest.main()
