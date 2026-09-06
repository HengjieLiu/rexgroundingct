from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from summarize_007_phase3_results import (
    ALL_CATEGORIES,
    EXPECTED_FINDINGS,
    EXPECTED_CASES,
    build_summary,
    evaluation_records,
)


def make_dataset(path: Path) -> dict:
    entries = []
    for case_index in range(EXPECTED_CASES):
        count = 1 if case_index < 199 else 182
        findings = {str(index): {} for index in range(count)}
        categories = {
            str(index): ALL_CATEGORIES[index % len(ALL_CATEGORIES)]
            for index in range(count)
        }
        entries.append({"name": f"case_{case_index:03d}.nii.gz", "findings": findings, "categories": categories})
    payload = {"test": entries}
    path.write_text(json.dumps(payload))
    assert sum(len(item["findings"]) for item in entries) == EXPECTED_FINDINGS
    return payload


def make_evaluation(path: Path, dataset: dict, dice: float = 0.2) -> None:
    cases = []
    for entry in dataset["test"]:
        findings = {
            f"finding_{index}": {"global_dice": dice + (int(index) % 3) * 0.1}
            for index in entry["findings"]
        }
        cases.append({"file": entry["name"], "findings": findings})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": {"total_cases": 200, "total_findings": 381}, "cases": cases}))


def test_evaluation_records_validate_case_and_finding_census(tmp_path: Path) -> None:
    dataset_path = tmp_path / "val200.json"
    dataset = make_dataset(dataset_path)
    evaluation_path = tmp_path / "eval.json"
    make_evaluation(evaluation_path, dataset)

    records = evaluation_records(evaluation_path, dataset_path)

    assert len(records) == EXPECTED_FINDINGS
    assert len({record["case_name"] for record in records}) == EXPECTED_CASES
    assert {record["category"] for record in records} == set(ALL_CATEGORIES)


def test_phase3_summary_has_ten_milestones_and_nodule_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "val200.json"
    dataset = make_dataset(dataset_path)
    source_eval = tmp_path / "source_eval.json"
    make_evaluation(source_eval, dataset, dice=0.3)
    group_dir = tmp_path / "group"
    e10 = group_dir / "ddp_bs4" / "eval_epoch010_val200" / "eval" / "val_quick_global_eval.json"
    make_evaluation(e10, dataset, dice=0.4)
    state_path = group_dir / "phase3_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"status": "running", "action": "val200_complete", "relative_epoch": 10}))
    metrics_path = group_dir / "ddp_bs4" / "reports" / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(json.dumps({
        "status": "running", "completed_updates": 1000, "total_updates": 10000,
        "world_size": 4, "batch_size": 1, "grad_accum": 1,
        "mean_update_seconds": 2.0, "last_loss": 0.5,
    }))
    source_checkpoint = tmp_path / "source.pth"
    source_checkpoint.write_bytes(b"checkpoint")
    schedule_manifest = tmp_path / "schedule.manifest.json"
    schedule_manifest.write_text(json.dumps({"events": 40000}))

    args = type("Args", (), {
        "exp_dir": tmp_path,
        "group_dir": group_dir,
        "source_run_dir": tmp_path / "source",
        "source_checkpoint": source_checkpoint,
        "source_evaluation_json": source_eval,
        "val200_json": dataset_path,
        "absolute_epoch_offset": 200,
        "steps_per_epoch": 100,
        "schedule_manifest": schedule_manifest,
        "state_json": state_path,
    })()
    summary = build_summary(args)

    assert summary["status"] == "running"
    assert summary["latest_completed_milestone"] == 10
    assert list(summary["milestones"]) == [str(value) for value in range(10, 101, 10)]
    assert summary["milestones"]["10"]["absolute_epoch"] == 210
    assert summary["milestones"]["10"]["evaluation"]["overall"]["findings"] == 381
    assert summary["milestones"]["10"]["evaluation"]["categories"]["2d"] is not None
    assert summary["training"]["effective_global_batch_size"] == 4


def test_cli_writes_atomic_readable_json_and_markdown(tmp_path: Path) -> None:
    dataset_path = tmp_path / "val200.json"
    dataset = make_dataset(dataset_path)
    source_eval = tmp_path / "source_eval.json"
    make_evaluation(source_eval, dataset)
    source_checkpoint = tmp_path / "source.pth"
    source_checkpoint.write_bytes(b"checkpoint")
    group_dir = tmp_path / "group"
    output_json = tmp_path / "out" / "status.json"
    output_md = tmp_path / "out" / "status.md"
    command = [
        "python", "scripts/rexgroundingct/summarize_007_phase3_results.py",
        "--exp-dir", str(tmp_path), "--group-dir", str(group_dir),
        "--source-run-dir", str(tmp_path / "source"),
        "--source-checkpoint", str(source_checkpoint),
        "--source-evaluation-json", str(source_eval), "--val200-json", str(dataset_path),
        "--schedule-manifest", str(tmp_path / "schedule.json"),
        "--output-json", str(output_json), "--output-md", str(output_md),
    ]
    subprocess.run(command, check=True)
    assert json.loads(output_json.read_text())["evaluation_contract"]["findings"] == 381
    assert "Exp007 Phase 3" in output_md.read_text()
