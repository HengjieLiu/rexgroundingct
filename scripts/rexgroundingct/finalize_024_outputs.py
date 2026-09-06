#!/usr/bin/env python3
"""Finalize the external Exp024 output manifest after native verification."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit")
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/experiments/024_test_inference_anatomy_audit.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    config = json.loads(CONFIG.read_text())
    expected = {"cases": 300, "findings": 582}
    sets = {}
    for name in ("a1", "a2", "a3", "b1", "b2", "b3"):
        directory = ROOT / "outputs" / name
        files = sorted(directory.glob("*.nii.gz"))
        if len(files) != expected["cases"]:
            raise SystemExit(f"{name}: expected 300 NIfTI files, found {len(files)}")
        verification = directory.with_suffix(".verification.json")
        if not verification.exists():
            raise SystemExit(f"{name}: missing native verification manifest {verification}")
        check = json.loads(verification.read_text())
        if check.get("cases") != expected["cases"] or check.get("findings") != expected["findings"]:
            raise SystemExit(f"{name}: verification census mismatch: {check.get('cases')}/{check.get('findings')}")
        sets[name] = {
            "directory": str(directory),
            "file_count": len(files),
            "finding_count": int(check["findings"]),
            "verification_manifest": str(verification),
            "verification_sha256": sha256(verification),
        }
    marker = {
        "experiment": "Exp024",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_status": config["source_status"],
        "metadata_sha256": config["metadata_sha256"],
        "prompt_routing_test": str(ROOT / "config/prompt_routing_test.json"),
        "prompt_routing_test_sha256": sha256(ROOT / "config/prompt_routing_test.json"),
        "candidate_manifest": str(ROOT / "config/candidate_manifest.json"),
        "candidate_manifest_sha256": sha256(ROOT / "config/candidate_manifest.json"),
        "b_validation_report": str(ROOT / "reports/b_validation_report.md"),
        "b_validation_metrics": str(ROOT / "reports/b_validation_metrics.json"),
        "outputs": sets,
        "oracle_qualification": "retrospective validation-selected and optimistic; not an unbiased test estimate",
    }
    out = ROOT / "test_output_manifest.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    done = ROOT / "completion.json"
    done.write_text(json.dumps({"experiment": "Exp024", "status": "COMPLETE", "output_manifest": str(out), "six_sets": list(sets)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "sets": list(sets), "manifest": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
