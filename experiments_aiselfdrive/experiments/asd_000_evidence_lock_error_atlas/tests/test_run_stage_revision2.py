from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_stage.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "asd000_run_stage_revision2_test_module", SCRIPT_PATH
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


class AcceptedLogitSourceTests(unittest.TestCase):
    def _accepted_candidate(self) -> dict[str, object]:
        return {
            "candidate_id": "exp009_baseline_e100",
            "accepted": True,
            "status": "accepted",
            "cases": 200,
            "findings": 381,
            "array_hashes_verified": True,
            "source_matches_manifest": True,
            "storage_reproduction_status": "passed",
            "same_pass_mask_reproduction_status": "passed",
        }

    def test_accepted_candidate_requires_every_export_and_reproduction_proof(self) -> None:
        candidate = self._accepted_candidate()
        selected = RUNNER._select_accepted_logit_candidate(
            {"candidates": [candidate]}, "exp009_baseline_e100"
        )
        self.assertEqual(selected, candidate)

        incomplete = dict(candidate)
        incomplete["same_pass_mask_reproduction_status"] = None
        with self.assertRaises(RUNNER.StageContractError):
            RUNNER._select_accepted_logit_candidate(
                {"candidates": [incomplete]}, "exp009_baseline_e100"
            )

    def test_standardized_accepted_val80_manifest_is_fully_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            val80_path = root / "val80.json"
            prediction_root = root / "predictions"
            prediction_root.mkdir()
            cases: list[dict[str, object]] = []
            records: list[dict[str, object]] = []
            arrays: dict[str, np.ndarray] = {}
            for order in range(80):
                finding_count = 3 if order < 35 else 2
                name = f"case_{order:03d}.nii.gz"
                findings = {
                    str(index): f"finding {index}"
                    for index in range(finding_count)
                }
                cases.append(
                    {
                        "name": name,
                        "seg_path": name,
                        "findings": findings,
                    }
                )
                values = np.ones((finding_count, 1, 1, 1), dtype=np.float16)
                if order % 2:
                    values[0, 0, 0, 0] = -1.0
                array_path = root / f"case_{order:03d}.npy"
                np.save(array_path, values, allow_pickle=False)
                arrays[name] = values
                records.append(
                    {
                        "status": "complete",
                        "case_order": order,
                        "case_name": name,
                        "finding_count": finding_count,
                        "array_path": str(array_path),
                        "file_sha256": RUNNER.sha256_file(array_path),
                        "finite": True,
                        "embedding_bank_sha256": "b" * 64,
                        "checkpoint_sha256": "c" * 64,
                        "geometry": {"verified": True},
                    }
                )
            val80_path.write_text(
                json.dumps({"test": cases}, sort_keys=True), encoding="utf-8"
            )
            protocol = {
                "cohorts": {
                    "val80": {
                        "path": str(val80_path),
                        "sha256": RUNNER.sha256_file(val80_path),
                    }
                },
                "embedding_bank": {"sha256": "b" * 64},
                "baseline": {
                    "predictions": str(prediction_root),
                    "checkpoint_sha256": "c" * 64,
                },
                "runtime_root": str(root / "runtime"),
            }
            manifest = {
                "status": "verified_complete",
                "source": "accepted_sideexp002_export",
                "val80_sha256": protocol["cohorts"]["val80"]["sha256"],
                "case_count": 80,
                "finding_count": 195,
                "embedding_bank": {"verified": True},
                "candidate_lineage": {"verified": True},
                "logical_uncompressed_bytes": 390,
                "logical_array_count": 80,
                "threshold_mask_equivalence": "exact",
                "cases": records,
            }

            def prediction_loader(path: Path) -> np.ndarray:
                return (arrays[path.name] >= 0.0).astype(np.uint8)

            layout = {
                "logical_uncompressed_bytes": 390,
                "array_count": 80,
            }
            with patch.object(
                RUNNER, "read_nifti_data", side_effect=prediction_loader
            ), patch.object(
                RUNNER,
                "_validate_logit_embedding_proof",
                return_value={"verified": True},
            ), patch.object(
                RUNNER,
                "_validate_logit_candidate_proof",
                return_value={"verified": True},
            ), patch.object(
                RUNNER,
                "_derive_locked_val80_float16_layout",
                return_value=layout,
            ):
                verified = RUNNER._validate_val80_logit_manifest(
                    protocol, manifest
                )
            self.assertEqual(
                verified["host_revalidation"]["hash_verified_array_count"], 80
            )
            self.assertEqual(
                verified["host_revalidation"]["exact_threshold_mask_case_count"],
                80,
            )

            corrupted = {**manifest, "cases": [dict(row) for row in records]}
            corrupted["cases"][0]["file_sha256"] = "0" * 64
            with patch.object(
                RUNNER, "read_nifti_data", side_effect=prediction_loader
            ), patch.object(
                RUNNER,
                "_validate_logit_embedding_proof",
                return_value={"verified": True},
            ), patch.object(
                RUNNER,
                "_validate_logit_candidate_proof",
                return_value={"verified": True},
            ), patch.object(
                RUNNER,
                "_derive_locked_val80_float16_layout",
                return_value=layout,
            ), self.assertRaises(RUNNER.StageContractError):
                RUNNER._validate_val80_logit_manifest(protocol, corrupted)

    def test_missing_embedding_proof_forces_local_fallback(self) -> None:
        failure = RUNNER.AcceptedLogitSourceUnavailable(
            "accepted export lacks exact embedding-bank provenance"
        )
        with patch.object(
            RUNNER, "_project_accepted_sideexp_val80", side_effect=failure
        ):
            projected, reason = RUNNER._try_accepted_sideexp_projection({}, {})
        self.assertIsNone(projected)
        self.assertIn("embedding-bank provenance", str(reason))


class DockerFallbackContractTests(unittest.TestCase):
    def _controller_environment(self) -> dict[str, str]:
        lease_id = "lease-test-000"
        return {
            "REXGROUNDINGCT_AISELFDRIVE_EXPERIMENT_ID": RUNNER.EXPERIMENT_ID,
            "REXGROUNDINGCT_AISELFDRIVE_LEASE_ID": lease_id,
            "REXGROUNDINGCT_AISELFDRIVE_DOCKER_LABEL": (
                f"rexgroundingct.aiselfdrive.lease={lease_id}"
            ),
        }

    def test_fallback_is_an_argument_array_scoped_to_asd000_runtime(self) -> None:
        config = (
            Path(__file__).resolve().parents[1] / "configs" / "protocol.yaml"
        )
        protocol = RUNNER.load_protocol(config)
        with patch.dict(os.environ, self._controller_environment(), clear=False):
            argv = RUNNER._docker_val80_export_argv(
                protocol, preflight_only=True
            )
        self.assertEqual(argv[:2], ["docker", "run"])
        self.assertEqual(argv[argv.index("--gpus") + 1], "all")
        self.assertIn("rexgroundingct-voxtell:cu126", argv)
        self.assertIn("--preflight-only", argv)
        self.assertIn(
            "/workspace/experiments_aiselfdrive/experiments/"
            "asd_000_evidence_lock_error_atlas/scripts/export_val80_logits.py",
            argv,
        )
        self.assertIn(protocol["runtime_root"], argv)
        self.assertEqual(
            argv[argv.index("--max-durable-storage-gib") + 1], "64"
        )
        local_export = protocol["logits"]["local_export"]
        self.assertEqual(
            local_export["resource_class"], "bounded_inference_export"
        )
        self.assertEqual(local_export["max_durable_storage_gib"], 64)
        self.assertEqual(local_export["disk_preflight_free_gib_min"], 64)
        self.assertEqual(
            local_export["full_layout_float16_uncompressed_bytes"],
            37_666_291_712,
        )
        self.assertEqual(
            local_export["full_layout_float16_uncompressed_gib"],
            37_666_291_712 / 1024**3,
        )
        self.assertEqual(local_export["minimum_case_float16_bytes"], 103_809_024)
        self.assertEqual(local_export["maximum_case_float16_bytes"], 1_274_019_840)
        self.assertEqual(argv[argv.index("--network") + 1], "none")
        self.assertIn("/mnt/shengdata1:/mnt/shengdata1:ro", argv)
        runtime_rw = (
            f"{protocol['runtime_root']}:{protocol['runtime_root']}:rw"
        )
        volumes = [
            argv[index + 1]
            for index, value in enumerate(argv[:-1])
            if value == "--volume"
        ]
        self.assertEqual(
            [volume for volume in volumes if volume.endswith(":rw")],
            [runtime_rw],
        )
        environment = [
            argv[index + 1]
            for index, value in enumerate(argv[:-1])
            if value == "--env"
        ]
        self.assertEqual(
            sorted(value.split("=", 1)[0] for value in environment),
            sorted(
                [
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "REXGROUNDINGCT_AISELFDRIVE_DOCKER_LABEL",
                    "REXGROUNDINGCT_AISELFDRIVE_EXPERIMENT_ID",
                    "REXGROUNDINGCT_AISELFDRIVE_LEASE_ID",
                ]
            ),
        )
        self.assertIn("HF_HUB_OFFLINE=1", environment)
        self.assertIn("TRANSFORMERS_OFFLINE=1", environment)
        self.assertEqual(
            argv[argv.index("--embeddings") + 1],
            protocol["embedding_bank"]["path"],
        )
        self.assertEqual(
            argv[argv.index("--embedding-lookup-normalization") + 1],
            "python_str_lower",
        )
        self.assertEqual(
            argv[argv.index("--expected-checkpoint-path") + 1],
            protocol["baseline"]["checkpoint"],
        )
        self.assertEqual(
            argv[argv.index("--val200-sha256") + 1],
            protocol["cohorts"]["val200"]["sha256"],
        )
        cidfile = Path(argv[argv.index("--cidfile") + 1])
        self.assertTrue(
            RUNNER._is_within(
                cidfile, Path(protocol["runtime_root"]) / ".controller"
            )
        )
        self.assertNotIn("bash", argv)
        self.assertNotIn("-c", argv)
        self.assertFalse(any("slurm" in value.casefold() for value in argv))
        labels = [
            argv[index + 1]
            for index, value in enumerate(argv[:-1])
            if value == "--label"
        ]
        self.assertEqual(
            labels,
            [
                "rexgroundingct.aiselfdrive.lease=lease-test-000",
                (
                    "rexgroundingct.aiselfdrive.experiment="
                    f"{RUNNER.EXPERIMENT_ID}"
                ),
            ],
        )
        with patch.dict(os.environ, self._controller_environment(), clear=False):
            proof = RUNNER._docker_launch_contract_proof(
                protocol, preflight_only=True
            )
        self.assertTrue(proof["verified"])
        self.assertEqual(proof["only_writable_mount"], runtime_rw)

    def test_fallback_rejects_missing_or_mismatched_controller_lease(self) -> None:
        config = (
            Path(__file__).resolve().parents[1] / "configs" / "protocol.yaml"
        )
        protocol = RUNNER.load_protocol(config)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            RUNNER.StageContractError, "EXPERIMENT_ID"
        ):
            RUNNER._docker_val80_export_argv(protocol, preflight_only=True)

        environment = self._controller_environment()
        environment["REXGROUNDINGCT_AISELFDRIVE_DOCKER_LABEL"] = (
            "rexgroundingct.aiselfdrive.lease=wrong"
        )
        with patch.dict(os.environ, environment, clear=True), self.assertRaisesRegex(
            RUNNER.StageContractError, "active controller lease label"
        ):
            RUNNER._docker_val80_export_argv(protocol, preflight_only=False)

    def test_docker_image_id_must_match_hardware_profile_exactly(self) -> None:
        expected = (
            "sha256:8ff421d05fbf6044553ba987d065a4d6fdaaaab8e2913272e620d08c4c286e0f"
        )
        completed = subprocess.CompletedProcess(
            ["docker", "image", "inspect"], 0, stdout=expected + "\n", stderr=""
        )
        with patch.object(RUNNER, "_run_checked", return_value=completed):
            observed = RUNNER._verify_docker_image_id(
                "rexgroundingct-voxtell:cu126", expected
            )
        self.assertEqual(observed["observed_id"], expected)

        wrong = subprocess.CompletedProcess(
            ["docker", "image", "inspect"],
            0,
            stdout="sha256:" + "0" * 64 + "\n",
            stderr="",
        )
        with patch.object(RUNNER, "_run_checked", return_value=wrong):
            with self.assertRaisesRegex(
                RUNNER.StageContractError, "image ID mismatch"
            ):
                RUNNER._verify_docker_image_id(
                    "rexgroundingct-voxtell:cu126", expected
                )

    def test_subprocess_failures_are_translated_to_stage_contract_errors(self) -> None:
        error = subprocess.CalledProcessError(7, ["docker", "run"])
        with patch.object(RUNNER.subprocess, "run", side_effect=error):
            with self.assertRaisesRegex(
                RUNNER.StageContractError, "exit code 7"
            ):
                RUNNER._run_checked(
                    ["docker", "run"], timeout=1, capture_output=True
                )

    def test_child_geometry_stderr_preserves_nonretryable_failure_class(self) -> None:
        error = subprocess.CalledProcessError(
            9,
            ["docker", "run"],
            stderr="fatal: geometry mismatch in locked case",
        )
        with patch.object(RUNNER.subprocess, "run", side_effect=error):
            with self.assertRaises(RUNNER.StageContractError) as raised:
                RUNNER._run_checked(
                    ["docker", "run"], timeout=1, capture_output=False
                )
        self.assertIn("geometry mismatch", str(raised.exception))
        self.assertEqual(
            RUNNER._classify_stage_failure(raised.exception),
            "geometry_mismatch",
        )

    def test_legacy_cached_export_without_host_launch_attestation_is_rejected(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs" / "protocol.yaml"
        protocol = RUNNER.load_protocol(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "export_manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            missing = root / "host_launch_attestation.json"
            with patch.object(
                RUNNER, "_local_launch_attestation_path", return_value=missing
            ), self.assertRaisesRegex(
                RUNNER.StageContractError, "launch attestation"
            ):
                RUNNER._validate_local_launch_attestation(
                    protocol, manifest_path
                )

    def test_timeout_stops_only_the_exact_owned_cidfile_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cidfile = Path(directory) / "owned.cid"
            container_id = "a" * 64
            calls: list[list[str]] = []

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                calls.append(list(command))
                if command[:2] == ["docker", "run"]:
                    cidfile.write_text(container_id + "\n", encoding="utf-8")
                    raise subprocess.TimeoutExpired(command, 1)
                if command[:2] == ["docker", "inspect"]:
                    labels = {
                        "rexgroundingct.aiselfdrive.lease": "lease-test-000",
                        "rexgroundingct.aiselfdrive.experiment": RUNNER.EXPERIMENT_ID,
                    }
                    return subprocess.CompletedProcess(
                        command, 0, stdout=json.dumps(labels), stderr=""
                    )
                if command[:2] == ["docker", "stop"]:
                    return subprocess.CompletedProcess(
                        command, 0, stdout=container_id, stderr=""
                    )
                raise AssertionError(command)

            with patch.dict(
                os.environ, self._controller_environment(), clear=False
            ), patch.object(
                RUNNER.subprocess, "run", side_effect=fake_run
            ), patch.object(RUNNER, "_RESOURCE_GPU_SECONDS", 0.0):
                with self.assertRaisesRegex(
                    RUNNER.StageContractError, "owned_container_stopped"
                ):
                    RUNNER._run_checked(
                        ["docker", "run"],
                        timeout=1,
                        capture_output=True,
                        docker_cidfile=cidfile,
                        account_gpu_time=True,
                    )
                self.assertGreaterEqual(RUNNER._RESOURCE_GPU_SECONDS, 0.0)
            self.assertFalse(cidfile.exists())
            self.assertTrue(any(call[:2] == ["docker", "inspect"] for call in calls))
            self.assertTrue(any(call[:2] == ["docker", "stop"] for call in calls))


class ScientificLockContractTests(unittest.TestCase):
    @staticmethod
    def _protocol() -> dict[str, object]:
        config = Path(__file__).resolve().parents[1] / "configs" / "protocol.yaml"
        return RUNNER.load_protocol(config)

    def test_locked_val80_float16_layout_is_derived_from_prediction_headers(self) -> None:
        layout = RUNNER._derive_locked_val80_float16_layout(self._protocol())
        self.assertEqual(layout["case_count"], 80)
        self.assertEqual(layout["finding_count"], 195)
        self.assertEqual(layout["array_count"], 80)
        self.assertEqual(layout["logical_uncompressed_bytes"], 37_666_291_712)
        self.assertEqual(layout["logical_uncompressed_gib"], 35.0794677734375)
        self.assertEqual(layout["minimum_case_bytes"], 103_809_024)
        self.assertEqual(layout["maximum_case_bytes"], 1_274_019_840)

    def test_locked_embedding_bank_has_exact_lowercase_coverage(self) -> None:
        proof = RUNNER._audit_locked_embedding_bank(self._protocol())
        self.assertEqual(proof["shape"], [6467, 2560])
        self.assertEqual(proof["dtype"], "float16")
        self.assertEqual(proof["lookup_normalization"], "python_str_lower")
        self.assertEqual(
            proof["coverage"]["val80"]["exact_case_covered_occurrences"], 7
        )
        self.assertEqual(
            proof["coverage"]["val200"]["exact_case_covered_occurrences"], 14
        )
        self.assertEqual(
            proof["coverage"]["val80"]["lowercase_covered_occurrences"], 195
        )
        self.assertEqual(
            proof["coverage"]["val200"]["lowercase_covered_occurrences"], 381
        )
        self.assertFalse(proof["qwen_fallback_permitted"])

    def test_local_export_policy_pins_cap_mounts_offline_mode_and_probe(self) -> None:
        protocol = self._protocol()
        policy = RUNNER._validate_local_export_policy(protocol)
        self.assertEqual(policy["max_durable_storage_gib"], 64)
        self.assertEqual(policy["network_mode"], "none")
        self.assertEqual(
            policy["offline_environment"],
            {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        drifted = copy.deepcopy(protocol)
        drifted["logits"]["local_export"]["network_mode"] = "bridge"
        with self.assertRaisesRegex(RUNNER.StageContractError, "policy drifted"):
            RUNNER._validate_local_export_policy(drifted)

    def test_runtime_probe_proof_requires_explicit_pre_cuda_ordering(self) -> None:
        protocol = self._protocol()
        proof = {
            "status": "passed",
            "runtime_root": protocol["runtime_root"],
            "probe_contract": (
                "atomic_replace_file_fsync_directory_fsync_read_delete_v1"
            ),
            "file_fsync": True,
            "atomic_replace": True,
            "directory_fsync_after_replace": True,
            "readback_exact": True,
            "delete_verified": True,
            "directory_fsync_after_delete": True,
            "completed_before_cuda_and_predictor_initialization": True,
        }
        self.assertTrue(
            RUNNER._validate_runtime_durability_proof(
                protocol,
                {"runtime_durability_probe": proof},
                label="synthetic",
            )["completed_before_cuda_and_predictor_initialization"]
        )
        proof.pop("completed_before_cuda_and_predictor_initialization")
        with self.assertRaisesRegex(RUNNER.StageContractError, "durability proof"):
            RUNNER._validate_runtime_durability_proof(
                protocol,
                {"runtime_durability_probe": proof},
                label="synthetic",
            )

    def test_transitive_sources_and_voxtell_submodule_are_locked(self) -> None:
        protocol = self._protocol()
        dependencies = protocol["logits"]["dependency_hashes"]
        for declared, expected in dependencies.items():
            with self.subTest(path=declared):
                self.assertEqual(
                    RUNNER.sha256_file(RUNNER._resolve_declared_path(declared)),
                    expected,
                )
        proof = RUNNER._validate_voxtell_submodule(protocol)
        self.assertEqual(proof["head"], "ab284be17c002c106075227ca655f4da3269427f")
        self.assertTrue(proof["clean"])


class FailedEvidenceContractTests(unittest.TestCase):
    def test_failure_evidence_is_attempt_and_plan_bound_with_all_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "failed.json"
            stage = {
                "id": "verify_or_export_val80_logits",
                "outputs": [],
                "allowed_write_paths": [str(evidence_path)],
                "completion": {"evidence_file": str(evidence_path)},
                "acceptance_checks": ["lineage_gate", "resource_gate"],
            }
            plan_path = root / "experiment.yaml"
            plan_path.write_text("schema_version: '1.0'\n", encoding="utf-8")
            plan_sha256 = RUNNER.sha256_file(plan_path)
            state = {
                "experiment_id": RUNNER.EXPERIMENT_ID,
                "plan_sha256": plan_sha256,
                "plan_revision": 2,
                "revision": 17,
                "status": "running",
                "current_stage_id": stage["id"],
                "stages": [
                    {
                        "id": stage["id"],
                        "attempts": 2,
                        "status": "running",
                        "started_at": "2026-07-31T00:00:00Z",
                    }
                ],
            }
            (root / "state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            with patch.object(RUNNER, "EXPERIMENT_ROOT", root), patch.object(
                RUNNER, "SELFDRIVE_ROOT", root
            ), patch.object(
                RUNNER, "_ACTIVE_RUNTIME_ROOT", None
            ), patch.object(RUNNER, "_ACTIVE_DURABLE_FILES", ()), patch.object(
                RUNNER, "_DURABLE_STORAGE_HIGH_WATER_BYTES", 0
            ), patch.object(RUNNER, "_RESOURCE_GPU_SECONDS", 0.125 * 3600):
                RUNNER._write_failed_evidence(
                    stage,
                    RUNNER.StageContractError(
                        "locked baseline checkpoint hash drifted"
                    ),
                )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed")
            self.assertEqual(evidence["experiment_id"], RUNNER.EXPERIMENT_ID)
            self.assertEqual(evidence["stage_id"], stage["id"])
            self.assertEqual(evidence["attempt"], 2)
            self.assertEqual(evidence["plan_sha256"], plan_sha256)
            self.assertEqual(
                evidence["error"]["class"], "checkpoint_hash_drift"
            )
            self.assertIn("checkpoint hash drifted", evidence["error"]["message"])
            resources = evidence["metrics"]["resources"]
            self.assertEqual(
                set(resources), {"gpu_hours", "cpu_hours", "storage_gib"}
            )
            self.assertGreaterEqual(resources["gpu_hours"], 0.125)
            self.assertTrue(
                all(
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value >= 0
                    for value in resources.values()
                )
            )
            self.assertTrue(
                all(check["status"] == "failed" for check in evidence["checks"])
            )

    def test_failure_classes_separate_nonretryable_resource_timeout_and_io(self) -> None:
        cases = [
            (
                RUNNER.StageContractError("geometry mismatch for case"),
                "geometry_mismatch",
            ),
            (
                RUNNER.StageContractError("Docker executable is unavailable"),
                "resource_unavailable",
            ),
            (
                RUNNER.StageContractError("subprocess timed out after 10s"),
                "timeout",
            ),
            (TimeoutError("operation timed out"), "timeout"),
            (
                OSError(28, "no space left on device"),
                "resource_unavailable",
            ),
            (PermissionError(13, "permission denied"), "write_scope_violation"),
            (OSError("temporary read failure"), "transient_io"),
            (
                RUNNER.StageContractError("candidate lineage drifted"),
                "lineage_mismatch",
            ),
        ]
        for error, expected in cases:
            with self.subTest(error=error):
                self.assertEqual(RUNNER._classify_stage_failure(error), expected)

    def test_failed_writer_rejects_completion_path_outside_allowed_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "outside.json"
            stage = {
                "id": "synthetic_stage",
                "allowed_write_paths": [],
                "completion": {"evidence_file": str(outside)},
                "acceptance_checks": ["scope"],
            }
            with self.assertRaisesRegex(
                RUNNER.StageContractError,
                "outside experiments_aiselfdrive|allowed_write_path",
            ):
                RUNNER._write_failed_evidence(
                    stage, RUNNER.StageContractError("synthetic failure")
                )
            self.assertFalse(outside.exists())

    def test_unknown_implementation_exception_has_normalized_class(self) -> None:
        self.assertEqual(
            RUNNER._classify_stage_failure(KeyError("missing internal key")),
            "command_runtime",
        )


class CloseoutTransactionTests(unittest.TestCase):
    def test_snapshot_restores_existing_files_and_removes_partial_new_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "claims.yaml"
            created = root / "closeout.json"
            existing.write_bytes(b"pending\n")
            snapshot = RUNNER._snapshot_files([existing, created])
            existing.write_bytes(b"supported\n")
            created.write_bytes(b"partial\n")
            RUNNER._restore_file_snapshot(snapshot)
            self.assertEqual(existing.read_bytes(), b"pending\n")
            self.assertFalse(created.exists())


class EvaluationContractTests(unittest.TestCase):
    def test_lineage_contract_has_stable_standard_json_pointer_targets(self) -> None:
        config = (
            Path(__file__).resolve().parents[1] / "configs" / "protocol.yaml"
        )
        protocol = RUNNER.load_protocol(config)
        contract = RUNNER._build_evaluation_contract(protocol)
        for pointer_target in (
            "prompt_form",
            "inference_window",
            "threshold_policy",
            "postprocessing",
        ):
            self.assertIn(pointer_target, contract)
        hash_scope = {
            key: value
            for key, value in contract.items()
            if key not in {"contract_sha256", "hash_algorithm", "verified"}
        }
        self.assertEqual(
            contract["contract_sha256"], RUNNER._canonical_sha256(hash_scope)
        )

        drifted = copy.deepcopy(protocol)
        drifted["evaluation_contract"]["threshold_policy"][
            "probability_threshold"
        ] = 0.6
        with self.assertRaisesRegex(
            RUNNER.StageContractError, "threshold-policy"
        ):
            RUNNER._build_evaluation_contract(drifted)


class ResourceEvidenceTests(unittest.TestCase):
    def test_every_passed_evidence_has_measured_nonnegative_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence.json"
            stage = {
                "id": "synthetic_resource_stage",
                "outputs": [],
                "completion": {"evidence_file": str(evidence)},
                "acceptance_checks": ["resource_contract"],
            }
            RUNNER._write_passed_evidence(
                stage,
                summary="resource contract",
                metrics={"ok": True},
            )
            resources = json.loads(evidence.read_text(encoding="utf-8"))[
                "metrics"
            ]["resources"]
            self.assertEqual(
                set(resources), {"gpu_hours", "cpu_hours", "storage_gib"}
            )
            self.assertEqual(resources["gpu_hours"], 0.0)
            for value in resources.values():
                self.assertIsInstance(value, float)
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)

    def test_measured_resources_preserve_optional_gpu_attempt_increment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence.json"
            stage = {
                "id": "synthetic_gpu_resource_stage",
                "outputs": [],
                "completion": {"evidence_file": str(evidence)},
                "acceptance_checks": ["resource_contract"],
            }
            RUNNER._write_passed_evidence(
                stage,
                summary="resource contract",
                metrics={
                    "resources": {
                        "gpu_hours": 0.25,
                        "cpu_hours": 999,
                        "storage_gib": 999,
                    }
                },
            )
            resources = json.loads(evidence.read_text(encoding="utf-8"))[
                "metrics"
            ]["resources"]
            self.assertEqual(resources["gpu_hours"], 0.25)
            self.assertLess(resources["cpu_hours"], 999)
            self.assertLess(resources["storage_gib"], 999)


if __name__ == "__main__":
    unittest.main()
