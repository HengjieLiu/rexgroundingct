from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("poll_exp021_then_launch_007_phase3.sh")
VAL200 = Path(__file__).resolve().parents[2] / "configs/evaluation/rexgroundingct_val200_seed20260723.json"


def run_poller(exp021: Path, exp007: Path, state: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({
        "EXP021_DIR": str(exp021), "EXP007_DIR": str(exp007),
        "REPO_ROOT": str(Path(__file__).resolve().parents[2]),
        "AUTOSTART_DIR": str(state), "POLL_ONCE": "1",
        "POLL_INTERVAL_SECONDS": "1", **extra,
    })
    return subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=True)


def make_exp021_fixture(root: Path, complete: bool) -> Path:
    exp021 = root / "exp021"
    group = exp021 / "runs" / "exp021_fixture"
    run_dir = group / "ddp_bs4"
    (exp021 / "config").mkdir(parents=True)
    (exp021 / "config/latest_run_group.txt").write_text("exp021_fixture\n")
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints/checkpoint_update_010000.pth").write_bytes(b"checkpoint")
    if complete:
        (run_dir / ".train_complete").touch()
        (group / ".experiment_complete").touch()
        eval_dir = run_dir / "eval_epoch100_val200"
        (eval_dir / "eval").mkdir(parents=True)
        (eval_dir / "reports").mkdir(parents=True)
        (eval_dir / "predictions").mkdir(parents=True)
        (eval_dir / "eval/val_quick_global_eval.json").write_text("{}")
        (eval_dir / "reports/val_quick_global_eval_summary.json").write_text(json.dumps({"total_cases": 200, "total_findings": 381}))
        for index in range(200):
            (eval_dir / "predictions" / f"case_{index:03d}.nii.gz").touch()
        report = exp021 / "reports/nodule_specialist_summary.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({"status": "complete", "milestones": {"100": {"evaluation": {}}}}))
    return exp021


def test_poller_waits_for_current_incomplete_exp021(tmp_path: Path) -> None:
    exp021 = make_exp021_fixture(tmp_path, complete=False)
    exp007 = tmp_path / "exp007"
    result = run_poller(exp021, exp007, tmp_path / "state", NO_LAUNCH="1")

    assert "status=waiting_exp021" in result.stdout
    status = json.loads((tmp_path / "state/poller_status.json").read_text())
    assert status["status"] == "waiting_exp021"
    assert status["exp021_gate"]["complete"] is False


def test_poller_launches_exactly_once_after_complete_gate(tmp_path: Path) -> None:
    exp021 = make_exp021_fixture(tmp_path, complete=True)
    exp007 = tmp_path / "exp007"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"$*\" >> {docker_log}\n"
        "exit 0\n"
    )
    fake_docker.chmod(0o755)
    env_path = f"{fake_bin}:{os.environ['PATH']}"

    run_poller(exp021, exp007, tmp_path / "state", PATH=env_path, IMAGE="fixture-image")
    launch = json.loads((tmp_path / "state/phase3_launch_state.json").read_text())
    assert launch["status"] == "launched"
    first_log = docker_log.read_text().splitlines()
    assert sum(line.startswith("run ") for line in first_log) == 1

    launch["status"] = "launching"
    (tmp_path / "state/phase3_launch_state.json").write_text(json.dumps(launch))
    run_poller(exp021, exp007, tmp_path / "state", PATH=env_path, IMAGE="fixture-image")
    assert sum(line.startswith("run ") for line in docker_log.read_text().splitlines()) == 1

    run_poller(exp021, exp007, tmp_path / "state", PATH=env_path, IMAGE="fixture-image")
    second_log = docker_log.read_text().splitlines()
    assert sum(line.startswith("run ") for line in second_log) == 1

    phase3_group = Path(launch["phase3_group_dir"])
    phase3_group.mkdir(parents=True)
    (phase3_group / ".experiment_complete").touch()
    run_poller(exp021, exp007, tmp_path / "state", PATH=env_path, IMAGE="fixture-image")
    status = json.loads((tmp_path / "state/poller_status.json").read_text())
    assert status["status"] == "complete"
