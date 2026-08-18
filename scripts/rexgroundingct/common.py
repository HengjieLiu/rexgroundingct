#!/usr/bin/env python3
"""Shared ReXGroundingCT experiment utilities."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REX_DATA_ROOT = Path("/data/hengjie/datasets/rexgroundingct")
REX_METADATA = REX_DATA_ROOT / "MICCAI_challenge_dataset.json"
REX_SEG_DIR = REX_DATA_ROOT / "segmentations"
CT_ROOT = Path("/mnt/shengdata1/hengjie/datasets/rexgroundingct/ct")
EXP_ROOT = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct")
REPO_EXPERIMENT_ROOT = REPO_ROOT / "experiments"
CANONICAL_CONFIG_ROOT = REPO_ROOT / "configs" / "experiments"
VOXTELL_SUBMODULE = REPO_ROOT / "external" / "VoxTell"

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "001_voxtell_v1_1_miccai200_val_eval": {
        "title": "VoxTell v1.1 MICCAI 200-case validation evaluation",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT / "001_voxtell_v1_1_miccai200_val_eval.json"
        ),
        "primary_report": Path("reports/val_evaluation_report.md"),
        "primary_eval_json": Path("eval/val_quick_global_eval.json"),
        "readiness_json": Path("config/val_ct_readiness.json"),
    },
    "002_voxtell_text_ft_miccai_train_val": {
        "title": "VoxTell text-conditioned fine-tuning on MICCAI train and val",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT / "002_voxtell_text_ft_miccai_train_val.json"
        ),
        "primary_report": Path("reports/train_val_finetuning_report.md"),
        "primary_eval_json": Path("eval/val_official_eval.json"),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
    "003_voxtell_rex_ft_rescue_ablation": {
        "title": "VoxTell ReX fine-tuning rescue ablation",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT / "003_voxtell_rex_ft_rescue_ablation.json"
        ),
        "primary_report": Path("reports/train_val_finetuning_report.md"),
        "primary_eval_json": Path("eval/val_quick_global_eval.json"),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
    "004_voxtell_v123_native_vs_2mm_global_context_ft": {
        "title": "VoxTell v123 native versus 2 mm global-context continuation",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT
            / "004_voxtell_v123_native_vs_2mm_global_context_ft.json"
        ),
        "primary_report": Path("reports/paired_comparison_report.md"),
        "primary_eval_json": Path(
            "runs/latest/native192_cont100/eval_epoch100_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
    "005_voxtell_global_proposal_local_cascade": {
        "title": "VoxTell global proposal to local segmentation cascade",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT / "005_voxtell_global_proposal_local_cascade.json"
        ),
        "primary_report": Path("reports/final_comparison.md"),
        "primary_eval_json": Path(
            "runs/latest/strict_inclusion/cascade_step05000_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
    "006_voxtell_cached_native_v123_lr_ablation": {
        "title": "VoxTell cached-native v123 learning-rate ablation",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT / "006_voxtell_cached_native_v123_lr_ablation.json"
        ),
        "primary_report": Path("reports/lr_ablation_report.md"),
        "primary_eval_json": Path(
            "runs/latest/v123_cached_e5_d4/eval_epoch100_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
    "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched": {
        "title": "VoxTell cached-native v123 e5/d4 DDP batch4 update-matched run",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT
            / "007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched.json"
        ),
        "primary_report": Path("reports/ddp_bs4_update_matched_report.md"),
        "primary_eval_json": Path(
            "runs/latest/ddp_bs4/eval_epoch100_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
    "008_voxtell_dual_branch_proposal_refinement_ablation": {
        "title": "VoxTell dual-branch proposal/refinement ablation",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT
            / "008_voxtell_dual_branch_proposal_refinement_ablation.json"
        ),
        "primary_report": Path("reports/dual_branch_ablation_report.md"),
        "primary_eval_json": Path(
            "runs/latest/v3_dualfusion_softguide_joint/"
            "eval_epoch100_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
    "009_voxtell_s3_attention_coupling_ablation": {
        "title": "VoxTell S3 attention coupling ablation",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT
            / "009_voxtell_s3_attention_coupling_ablation.json"
        ),
        "primary_report": Path("reports/s3_attention_coupling_report.md"),
        "primary_eval_json": Path(
            "runs/latest/s3v3_logit_residual_half_quarter/"
            "eval_epoch100_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
    "010_voxtell_public_anatomy_prior_fusion": {
        "title": "VoxTell public anatomy-prior fusion",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT
            / "010_voxtell_public_anatomy_prior_fusion.json"
        ),
        "primary_report": Path(
            "reports/totalsegmentator_val200_anatomy_audit.md"
        ),
        "primary_eval_json": Path(
            "reports/totalsegmentator_val200_anatomy_audit.json"
        ),
        "readiness_json": Path("reports/val200_progress.json"),
    },
    "011_voxtell_v123_e4d4_ct_normalization_ablation": {
        "title": "VoxTell v123 CT normalization and encoder-LR ablation",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT
            / "011_voxtell_v123_e4d4_ct_normalization_ablation.json"
        ),
        "primary_report": Path("reports/ct_normalization_ablation_report.md"),
        "primary_eval_json": Path(
            "runs/latest_e4d4/v123_e4d4_zscore/"
            "eval_epoch100_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
    "012_voxtell_category_specialists_replay50_cont100": {
        "title": "VoxTell category-specialist continuation with 50% replay",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT
            / "012_voxtell_category_specialists_replay50_cont100.json"
        ),
        "primary_report": Path("reports/latest_progress.md"),
        "primary_eval_json": Path(
            "runs/latest/category_1alldiffuse_replay50/"
            "eval_epoch100_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("runs/latest/run_group_manifest.json"),
    },
    "013_voxtell_public_category_only_specialists": {
        "title": "VoxTell public-start category-only specialists",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT
            / "013_voxtell_public_category_only_specialists.json"
        ),
        "primary_report": Path("reports/latest_progress.md"),
        "primary_eval_json": Path(
            "runs/latest/category_1all_diffuse_target100/"
            "eval_epoch100_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("runs/latest/run_group_manifest.json"),
    },
    "014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched": {
        "title": "VoxTell cached-native v123 e5/d4 DDP batch16 update-matched run",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT
            / "014_voxtell_cached_native_v123_e5_d4_ddp_bs16_update_matched.json"
        ),
        "primary_report": Path("reports/ddp_bs16_update_matched_report.md"),
        "primary_eval_json": Path(
            "runs/latest/ddp_bs16/eval_epoch100_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
    "015_voxtell_isotropic_resolution_audit": {
        "title": "VoxTell native-resolution audit for isotropic planning",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT / "015_voxtell_isotropic_resolution_audit.json"
        ),
        "primary_report": Path("reports/native_resolution_audit_report.md"),
        "primary_eval_json": Path("reports/native_resolution_summary.json"),
        "readiness_json": Path("reports/native_resolution_summary.json"),
    },
    "016_voxtell_iso07_hu_preprocessing": {
        "title": "VoxTell 0.7 mm isotropic HU preprocessing",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT / "016_voxtell_iso07_hu_preprocessing.json"
        ),
        "primary_report": Path("reports/preprocessing_report.md"),
        "primary_eval_json": Path("cache/manifest.json"),
        "readiness_json": Path("cache/manifest.json"),
    },
    "017_voxtell_iso07_hu_ddp_bs4_update_matched": {
        "title": "VoxTell 0.7 mm HU v123 e5/d4 DDP batch4 run",
        "canonical_config": (
            CANONICAL_CONFIG_ROOT
            / "017_voxtell_iso07_hu_ddp_bs4_update_matched.json"
        ),
        "primary_report": Path("reports/iso07_hu_ddp_bs4_report.md"),
        "primary_eval_json": Path(
            "runs/latest/ddp_bs4/eval_epoch100_val200/eval/val_quick_global_eval.json"
        ),
        "readiness_json": Path("config/train_val_ct_readiness.json"),
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, sort_keys=True) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def experiment_definition(experiment: str) -> dict[str, Any]:
    if experiment not in EXPERIMENTS:
        valid = ", ".join(sorted(EXPERIMENTS))
        raise KeyError(f"Unknown experiment {experiment!r}; valid experiments: {valid}")
    return EXPERIMENTS[experiment]


def canonical_config_path(experiment: str) -> Path:
    return Path(experiment_definition(experiment)["canonical_config"])


def runtime_experiment_dir(experiment: str, exp_root: Path = EXP_ROOT) -> Path:
    return exp_root / experiment


def snapshot_experiment_config(
    experiment: str,
    exp_dir: Path | None = None,
    source_config: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy the repo-canonical config into runtime and record exact hashes.

    The runtime copy is a provenance snapshot. Edit the repo config, then start a
    new run or explicitly overwrite the snapshot when intentional.
    """
    source_config = source_config or canonical_config_path(experiment)
    exp_dir = exp_dir or runtime_experiment_dir(experiment)
    if not source_config.exists():
        raise FileNotFoundError(f"Canonical config not found: {source_config}")

    target = exp_dir / "config" / source_config.name
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source_config)
    status = "created"

    if target.exists():
        target_hash = sha256_file(target)
        if target_hash == source_hash:
            status = "exists_identical"
        elif overwrite:
            target.chmod(target.stat().st_mode | 0o200)
            target.write_bytes(source_config.read_bytes())
            status = "overwritten"
        else:
            raise RuntimeError(
                "Runtime config snapshot differs from canonical config. "
                f"canonical={source_config} runtime={target}. "
                "Edit only the canonical config, then use --overwrite only for an "
                "intentional new snapshot."
            )
    else:
        target.write_bytes(source_config.read_bytes())

    try:
        target.chmod(0o444)
    except OSError:
        pass

    record = {
        "experiment": experiment,
        "source_config_path": str(source_config),
        "source_config_repo_path": repo_relative(source_config),
        "source_config_sha256": source_hash,
        "runtime_config_snapshot_path": str(target),
        "runtime_config_snapshot_sha256": sha256_file(target),
        "status": status,
        "snapshotted_at_utc": utc_now_iso(),
    }
    write_json(exp_dir / "config" / f"{experiment}.config_snapshot.json", record)
    return record


def update_run_manifest(exp_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    manifest = exp_dir / "run_manifest.json"
    if manifest.exists():
        data = read_json(manifest)
    else:
        data = {
            "created_at_utc": utc_now_iso(),
            "experiment": exp_dir.name,
            "repo_root": str(REPO_ROOT),
            "repo_commit": git_commit(REPO_ROOT),
            "voxtell_submodule": str(VOXTELL_SUBMODULE),
            "voxtell_commit": git_commit(VOXTELL_SUBMODULE),
        }
    data.update(updates)
    data["updated_at_utc"] = utc_now_iso()
    write_json(manifest, data)
    return data


def ct_rate_rel_path(filename: str) -> Path:
    """Map a ReXGroundingCT filename to the CT-RATE fixed-volume path."""
    if not filename.endswith(".nii.gz"):
        raise ValueError(f"Expected a .nii.gz filename, got {filename!r}")
    stem = filename[: -len(".nii.gz")]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected CT-RATE filename format: {filename!r}")
    split = parts[0]
    patient = "_".join(parts[:2])
    scan = "_".join(parts[:3])
    return Path("dataset") / f"{split}_fixed" / patient / scan / filename


def ct_rate_abs_path(filename: str, ct_root: Path = CT_ROOT) -> Path:
    return ct_root / ct_rate_rel_path(filename)


def load_split_entries(metadata: Path = REX_METADATA, splits: list[str] | None = None) -> list[dict[str, Any]]:
    data = read_json(metadata)
    selected = splits or ["train", "val", "test"]
    entries: list[dict[str, Any]] = []
    for split in selected:
        if split not in data:
            raise KeyError(f"Split {split!r} not found in {metadata}")
        for index, item in enumerate(data[split]):
            copied = dict(item)
            copied["_split"] = split
            copied["_index"] = index
            entries.append(copied)
    return entries


def sorted_prompts(entry: dict[str, Any]) -> list[str]:
    findings = entry["findings"]
    return [findings[key] for key in sorted(findings, key=lambda value: int(value))]


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def command_string(argv: list[str] | None = None) -> str:
    import shlex
    import sys

    return " ".join(shlex.quote(part) for part in (argv or sys.argv))


def env_snapshot() -> dict[str, str]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "VOXTELL_MODEL",
        "PYTHONPATH",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}
