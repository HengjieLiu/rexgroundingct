#!/usr/bin/env python3
"""Acquire and audit official CT-RATE ``ts_total`` masks for ReX test300.

This isolated Exp023 entrypoint deliberately reuses the tested Exp020 audit
engine without modifying it.  The Exp020 source file is sealed as a hash-pinned
dependency in Exp023's source lock, while this entrypoint and its test-specific
population contract are sealed alongside it.  It never reads ReX test finding
labels, resamples inputs, or changes downstream model behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import audit_ct_rate_ts_total_rex_val200 as base


EXPERIMENT_ID = "023_ct_rate_ts_total_rex_test300_audit"
SCRIPT_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
DEFAULT_RUNTIME_ROOT = Path(
    "/mnt/shengdata1/hengjie/experiments/rexgroundingct/"
    "023_ct_rate_ts_total_rex_test300_audit"
)
PARENT_EXPERIMENT_ID = "020_ct_rate_ts_total_rex_val200_audit"
PARENT_AUDIT_SCRIPT = Path(base.__file__).resolve()
PARENT_AUDIT_REPO_PATH = "scripts/rexgroundingct/audit_ct_rate_ts_total_rex_val200.py"
SOURCE_LOCK_ENTRYPOINT_KEY = "test300_entrypoint"

# Re-exported stable helpers make the focused contract tests explicit and let
# this entrypoint retain the audited remote-path construction implementation.
ct_rate_ct_remote_path = base.ct_rate_ct_remote_path
ct_rate_ts_total_remote_path = base.ct_rate_ts_total_remote_path
parse_volume_name = base.parse_volume_name
source_lock_sha256 = base.source_lock_sha256


def _sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def _canonical_json_sha256(value: Any) -> str:
    return base.canonical_json_sha256(value)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the test-only canonical configuration."""
    config = base.read_json(config_path)
    if config.get("experiment") != EXPERIMENT_ID:
        raise ValueError(
            f"Config {config_path} is for {config.get('experiment')!r}, "
            f"not {EXPERIMENT_ID!r}"
        )
    population = config.get("population")
    if not isinstance(population, Mapping) or population.get("rex_split") != "test":
        raise ValueError("Exp023 requires population.rex_split to be exactly 'test'")
    if "evaluation_manifest" in population or "evaluation_manifest_sha256" in population:
        raise ValueError("Exp023 must not reuse a validation evaluation manifest")
    return dict(config)


def _expected_split_counts(config: Mapping[str, Any]) -> dict[str, int]:
    raw = config["population"].get("expected_ct_rate_split_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("population.expected_ct_rate_split_counts must be an object")
    return {str(key): int(value) for key, value in raw.items()}


def load_population(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Lock the exact ReX test metadata population without evaluator coupling."""
    population = config["population"]
    metadata_path = Path(str(population["metadata_path"]))
    expected_metadata_hash = str(population["metadata_sha256"])
    observed_metadata_hash = _sha256_file(metadata_path)
    if observed_metadata_hash != expected_metadata_hash:
        raise ValueError(
            "Local ReX metadata hash mismatch: "
            f"expected={expected_metadata_hash} observed={observed_metadata_hash}"
        )
    metadata = base.read_json(metadata_path)
    split = str(population["rex_split"])
    if split != "test":
        raise ValueError("Exp023 only accepts the ReX test split")
    source_rows = metadata.get(split)
    if not isinstance(source_rows, list):
        raise ValueError(f"Expected list metadata[{split!r}] in {metadata_path}")

    names: list[str] = []
    for row in source_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise ValueError(f"Malformed row in metadata[{split!r}]: {row!r}")
        names.append(str(row["name"]))
    expected_cases = int(population["expected_unique_cases"])
    if len(names) != expected_cases or len(set(names)) != expected_cases:
        raise ValueError(
            f"Expected exactly {expected_cases} unique {split} cases, "
            f"found rows={len(names)} unique={len(set(names))}"
        )

    output: list[dict[str, Any]] = []
    for metadata_index, name in enumerate(names):
        parsed = parse_volume_name(name)
        output.append(
            {
                "metadata_index": metadata_index,
                "volume_name": name,
                "rex_split": split,
                "ct_rate_original_split": parsed["ct_rate_original_split"],
                "ct_rate_fixed_split": parsed["ct_rate_fixed_split"],
                "remote_ct_path": ct_rate_ct_remote_path(name),
                "remote_ts_total_path": ct_rate_ts_total_remote_path(name),
                "known_upstream_missing": name in base.KNOWN_UPSTREAM_MISSING,
            }
        )

    expected_splits = _expected_split_counts(config)
    observed_splits = Counter(row["ct_rate_fixed_split"] for row in output)
    normalized_observed = {
        split_name: int(observed_splits.get(split_name, 0)) for split_name in expected_splits
    }
    unexpected_splits = sorted(set(observed_splits) - set(expected_splits))
    if normalized_observed != expected_splits or unexpected_splits:
        raise ValueError(
            f"Unexpected CT-RATE fixed split counts: expected={expected_splits} "
            f"observed={dict(observed_splits)}"
        )
    return output


def _entrypoint_record() -> dict[str, Any]:
    """Record every executable dependency that Exp023 relies on."""
    return {
        "script_version": SCRIPT_VERSION,
        "entrypoint": Path(__file__).name,
        "entrypoint_sha256": _sha256_file(Path(__file__)),
        "parent_experiment": PARENT_EXPERIMENT_ID,
        "parent_audit_script": PARENT_AUDIT_REPO_PATH,
        "parent_audit_script_sha256": _sha256_file(PARENT_AUDIT_SCRIPT),
    }


def _configure_base() -> None:
    """Bind the audited engine to Exp023's isolated population and report."""
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.load_population = load_population
    base.write_aggregate_report = write_aggregate_report


def _assert_preflight_summary(
    config: Mapping[str, Any], manifest: Mapping[str, Any], source_lock: Mapping[str, Any]
) -> None:
    expected = config["preflight"]
    summary = source_lock.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError("Preflight source lock has no summary")
    expected_cases = int(expected["requested_masks"])
    expected_bytes = int(expected["expected_remote_bytes"])
    expected_missing = int(expected["expected_known_upstream_missing_in_population"])
    if len(manifest.get("cases", [])) != expected_cases:
        raise RuntimeError("Preflight manifest count differs from the locked test population")
    if int(summary.get("requested_masks", -1)) != expected_cases:
        raise RuntimeError("Preflight requested-mask count differs from the canonical config")
    if int(summary.get("remote_ts_total_bytes", -1)) != expected_bytes:
        raise RuntimeError("Preflight remote-byte total differs from the canonical config")
    expected_splits = _expected_split_counts(config)
    observed_splits = {
        str(key): int(value) for key, value in dict(summary.get("split_counts", {})).items()
    }
    normalized_observed_splits = {
        split_name: int(observed_splits.get(split_name, 0)) for split_name in expected_splits
    }
    if normalized_observed_splits != expected_splits or set(observed_splits) - set(expected_splits):
        raise RuntimeError("Preflight CT-RATE split counts differ from the canonical config")
    if int(summary.get("known_upstream_missing_in_population", -1)) != expected_missing:
        raise RuntimeError("Preflight known-missing count differs from the canonical config")
    if int(summary.get("local_ct_present", -1)) != expected_cases:
        raise RuntimeError("Not every locked test CT is locally present")
    if int(summary.get("local_ct_sidecar_matches", -1)) != expected_cases:
        raise RuntimeError("Not every locked test CT has matching pinned provenance")
    if any(row.get("rex_split") != "test" for row in manifest["cases"]):
        raise RuntimeError("Preflight manifest contains a non-test ReX record")


def preflight(config: Mapping[str, Any], runtime_root: Path) -> dict[str, Any]:
    """Build a sealed, authenticated source lock without downloading masks."""
    _configure_base()
    result = base.preflight(config, runtime_root)
    manifest = dict(result["manifest"])
    source_lock = dict(result["source_lock"])
    _assert_preflight_summary(config, manifest, source_lock)
    if manifest.get("experiment") != EXPERIMENT_ID or source_lock.get("experiment") != EXPERIMENT_ID:
        raise RuntimeError("Preflight did not create an Exp023 source lock")

    source_lock[SOURCE_LOCK_ENTRYPOINT_KEY] = _entrypoint_record()
    source_lock["source_lock_sha256"] = source_lock_sha256(source_lock)
    lock_path = runtime_root / "config" / "source_lock.json"
    base.write_json_atomic(lock_path, source_lock)
    return {"manifest": manifest, "source_lock": source_lock, "paths": result["paths"]}


def verify_test_source_lock(
    runtime_root: Path, config: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Reject a validation/mixed runtime and verify both Exp023 code layers."""
    _configure_base()
    source_lock, manifest, paths = base._require_preflight(runtime_root)
    if source_lock.get("experiment") != EXPERIMENT_ID or manifest.get("experiment") != EXPERIMENT_ID:
        raise RuntimeError("Runtime source lock/manifest does not belong to Exp023")
    record = source_lock.get(SOURCE_LOCK_ENTRYPOINT_KEY)
    if not isinstance(record, Mapping):
        raise RuntimeError("Exp023 source lock lacks its entrypoint dependency record")
    if record.get("entrypoint") != Path(__file__).name:
        raise RuntimeError("Exp023 source lock was produced by a different entrypoint")
    if record.get("entrypoint_sha256") != _sha256_file(Path(__file__)):
        raise RuntimeError("Exp023 entrypoint differs from the sealed source lock")
    if record.get("parent_experiment") != PARENT_EXPERIMENT_ID:
        raise RuntimeError("Exp023 source lock has an unexpected parent experiment")
    if record.get("parent_audit_script") != PARENT_AUDIT_REPO_PATH:
        raise RuntimeError("Exp023 source lock has an unexpected parent audit path")
    if record.get("parent_audit_script_sha256") != _sha256_file(PARENT_AUDIT_SCRIPT):
        raise RuntimeError("Exp023 parent audit dependency differs from the sealed source lock")
    if source_lock.get("script_sha256") != _sha256_file(PARENT_AUDIT_SCRIPT):
        raise RuntimeError("Exp023 base-script hash differs from the sealed source lock")
    active_config = dict(config) if config is not None else load_config(DEFAULT_CONFIG)
    if _canonical_json_sha256(active_config) != source_lock.get("config_canonical_sha256"):
        raise RuntimeError("Live Exp023 config differs from the sealed preflight config")
    return source_lock, manifest, paths


def _test_aggregate_report(
    source_lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    visual_review: Mapping[str, Any] | None,
) -> tuple[str, str]:
    report, conclusion = base._aggregate_report(source_lock, manifest, results, visual_review)
    report = report.replace(
        "# CT-RATE `ts_total` / ReX validation audit",
        "# CT-RATE `ts_total` / ReX test300 audit",
    )
    report = report.replace(
        "No model, preprocessing, inference, or anatomy-prior integration was modified by this audit.",
        "ReX test finding masks remain withheld and untouched. No model, preprocessing, "
        "inference, submission behavior, or anatomy-prior integration was modified by this audit.",
    )
    return report, conclusion


def write_aggregate_report(runtime_root: Path) -> str:
    """Write the external, aggregate-only Exp023 audit report."""
    source_lock, manifest, paths = verify_test_source_lock(runtime_root)
    results_path = paths["private_reports"] / "audit_results.json"
    if not results_path.exists():
        raise RuntimeError("No audit results found; run --audit first")
    results = base.verified_audit_results(base.read_json(results_path), manifest, source_lock)
    visual_review = base._verified_manual_visual_review(
        paths, manifest, source_lock, results_path, results
    )
    report, conclusion = _test_aggregate_report(source_lock, manifest, results, visual_review)
    report_path = paths["reports"] / "ct_rate_ts_total_rex_test300_audit_report.md"
    temporary = report_path.with_name(f".{report_path.name}.{base.os.getpid()}.tmp")
    temporary.write_text(report)
    base.os.replace(temporary, report_path)
    return conclusion


def _print_preflight_summary(result: Mapping[str, Any]) -> None:
    source_lock = result["source_lock"]
    summary = source_lock["summary"]
    manifest = result["manifest"]
    print("Exp023 preflight complete (no masks downloaded).")
    print(f"Requested: {summary['requested_masks']}")
    print(f"Remote ts_total bytes: {summary['remote_ts_total_bytes']}")
    print(f"CT-RATE split counts: {summary['split_counts']}")
    print(f"Local CTs ready and provenance-matched: {summary['local_ct_sidecar_matches']}")
    print(f"LUT provenance: {source_lock['lut_provenance']['status']}")
    print(f"Manifest approval SHA-256: {manifest['manifest_sha256']}")
    print("First 10 resolved private-manifest rows:")
    for row in list(manifest["cases"])[:10]:
        print(
            " | ".join(
                (
                    str(row["volume_name"]),
                    str(row["ct_rate_fixed_split"]),
                    str(row["local_ct_path"]),
                    str(row["remote_ts_total_path"]),
                    "exists=yes",
                )
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    stages = parser.add_argument_group("exactly one stage is required")
    stages.add_argument("--preflight", action="store_true", help="Resolve and lock sources; do not download masks.")
    stages.add_argument("--download", action="store_true", help="Download only the preflight-approved masks.")
    stages.add_argument("--audit", action="store_true", help="Audit downloaded masks against native local CTs.")
    stages.add_argument("--visual-qc", action="store_true", help="Generate deterministic native-grid lung overlays.")
    stages.add_argument("--record-visual-review", action="store_true", help="Record a human visual-review outcome.")
    parser.add_argument(
        "--approve-manifest-sha256",
        help="Required exact preflight manifest SHA-256 for --download.",
    )
    parser.add_argument(
        "--derive-lungs",
        action="store_true",
        help="With --audit, derive native left/right lungs only after LUT evidence passes.",
    )
    parser.add_argument(
        "--skip-local-ct-hash",
        action="store_true",
        help="Do not perform the default full local CT content hashes (not suitable for final PASS).",
    )
    parser.add_argument("--representative-count", type=int, default=20)
    parser.add_argument("--reviewer")
    parser.add_argument("--review-notes")
    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument("--manual-review-passed", action="store_true")
    review_group.add_argument("--manual-review-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected_stages = sum(
        bool(value)
        for value in (
            args.preflight,
            args.download,
            args.audit,
            args.visual_qc,
            args.record_visual_review,
        )
    )
    if selected_stages != 1:
        parser.error("Choose exactly one stage")
    if args.derive_lungs and not args.audit:
        parser.error("--derive-lungs is valid only with --audit")
    if args.skip_local_ct_hash and not args.audit:
        parser.error("--skip-local-ct-hash is valid only with --audit")
    if (args.manual_review_passed or args.manual_review_failed) and not args.record_visual_review:
        parser.error("Manual review outcome flags require --record-visual-review")

    try:
        config = load_config(args.config)
        _configure_base()
        if args.preflight:
            _print_preflight_summary(preflight(config, args.runtime_root))
        else:
            verify_test_source_lock(args.runtime_root, config)
            if args.download:
                outcome = base.download(config, args.runtime_root, args.approve_manifest_sha256)
                print(f"Download status summary: {outcome['summary']}")
            elif args.audit:
                outcome = base.audit(
                    config,
                    args.runtime_root,
                    derive_lungs=args.derive_lungs,
                    hash_local_ct=not args.skip_local_ct_hash,
                )
                print(f"Audit conclusion: {outcome['conclusion']}")
            elif args.visual_qc:
                outcome = base.visual_qc(args.runtime_root, args.representative_count)
                print(f"Visual QC selected: {outcome['selected']}; conclusion: {outcome['conclusion']}")
            else:
                passed: bool | None
                if args.manual_review_passed:
                    passed = True
                elif args.manual_review_failed:
                    passed = False
                else:
                    passed = None
                conclusion = base.record_visual_review(
                    args.runtime_root, args.reviewer, args.review_notes, passed
                )
                print(f"Recorded manual visual review; conclusion: {conclusion}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
