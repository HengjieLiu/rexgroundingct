"""Single CPU writer for local cache progress; no GPU imports or model work."""
from __future__ import annotations

import time
from pathlib import Path

from exp027_common import atomic_json, atomic_text, lock, now, read_json
from exp027_multicategory_data import CATEGORIES


def render(root):
    root = Path(root)
    config = read_json(root / "config.json") if (root / "config.json").exists() else {}
    state = read_json(root / "status.json") if (root / "status.json").exists() else {"status": "pending"}
    progress = [read_json(p) for p in sorted((root / "progress").glob("*.json")) if not p.name.endswith("_memory.json")]
    order = {n: i for i, n in enumerate(["train0", "train1", "train2", "train3", "val", "verify0", "verify1", "verify2", "verify3"])}
    progress.sort(key=lambda r: order.get(r["worker"], 99))
    memory = {p.name.removesuffix("_memory.json"): read_json(p) for p in (root / "progress").glob("*_memory.json")}
    current_kind = "verify" if state.get("mode") == "verify" else "export"
    progress = [p for p in progress if p["worker"].startswith("verify") == (current_kind == "verify")]
    receipts = [read_json(p) for p in (root / "receipts").glob("*.json")
                if p.name.startswith("verify") == (current_kind == "verify")]
    cases = {c["name"]: c for r in receipts for c in r["cases"]}
    rows = [r for c in cases.values() for r in c["baseline_findings"]]
    refreshed = now()
    lines = ["# Frozen Exp007 cache preparation: 2b–2e", "",
             f"Status: **{state['status']}**. Last refreshed: {refreshed}.", "",
             "Local file viewers may require reload. Missing values are pending.", "",
             "This export shares GPUs and storage bandwidth with the 2a experiment. No category training is launched.", ""]
    if state.get("synthetic_fixture"):
        lines += ["**SYNTHETIC FIXTURE — not experimental results.**", ""]
    if state.get("error"):
        lines += [f"Failure: {state['error']}", ""]
    lines += [f"Verified: {len(cases):,} / 2,730 CTs; {len(rows):,} / 5,106 findings.",
              f"Recorded cache storage: {sum(c['cache_bytes'] for c in cases.values()) / 1024**3:.2f} GiB.", "",
              "| Category | Train findings | Val findings | A / B expected |",
              "| --- | ---: | ---: | ---: |"]
    for category in CATEGORIES:
        counts = config.get("category_counts", {}).get(category, {})
        train = sum(r["record"]["category"] == category and r["record"]["split"] == "train" for r in rows)
        val = sum(r["record"]["category"] == category and r["record"]["split"] == "val" for r in rows)
        lines += [f"| {category} | {train} / {counts.get('train_findings', 'pending')} | {val} / {counts.get('val_findings', 'pending')} | {counts.get('A', 'pending')} / {counts.get('B', 'pending')} |"]
    lines += ["", "The existing patient partition is retained. Category 2e has only three B findings.", "",
              "Original train/validation overlap: **2d patient train_2936** (one training and two validation findings, different CTs). Account for this before patient-held-out 2d training/evaluation.", "",
              "| Worker | Phase | CTs | Reused / generated | Elapsed | ETA | GPU free / minimum |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for p in progress:
        eta = ((p["total"] - p["done"]) * p["elapsed_seconds"] / p["done"] if p["done"] >= 2 else None)
        m = memory.get(p["worker"])
        mem = f"{m['free_mib']/1024:.1f} / {m['minimum_free_mib']/1024:.1f} GiB" if m else "pending"
        eta_text = f"{eta/3600:.2f} h" if eta is not None else "pending"
        lines += [f"| {p['worker']} | {p['status']} | {p['done']} / {p['total']} | {p['reused']} / {p['generated']} | {p['elapsed_seconds']/60:.1f} min | {eta_text} | {mem} |"]
        if p.get("error"):
            lines += [f"\n{p['worker']} failure: {p['error']}\n"]
    lines += ["", "Current CTs:", ""]
    lines += [f"- {p['worker']}: {p.get('case') or 'pending'}" for p in progress]
    lines += ["", "Final artifacts: [baselines](baselines.csv), [input manifest](../input_manifest.json), [category manifests](../manifests/).", ""]
    atomic_text(root / "reports/live_dashboard.md", "\n".join(lines))
    atomic_json(root / "reports/progress.json", {"status": state, "workers": progress,
                "verified_cases": len(cases), "verified_findings": len(rows), "refreshed_at": refreshed})


def report(root, interval=5, watch=False):
    root = Path(root)
    with lock(root / "locks/report.lock"):
        while True:
            render(root)
            if not watch:
                return
            for _ in range(max(1, int(interval))):
                time.sleep(1)
                state = read_json(root / "status.json") if (root / "status.json").exists() else {}
                if state.get("status") in ("failed", "cache_ready_for_training", "interrupted"):
                    render(root)
                    return
