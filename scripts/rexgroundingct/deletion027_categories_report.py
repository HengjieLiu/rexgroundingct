"""Local category dashboard: only complete evaluations enter Dice plots."""
from __future__ import annotations

import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deletion027_categories_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp027_common import atomic_json, atomic_text, lock, now, read_json, read_updates
from deletion027_report import fmt, optional, smooth
from deletion027_categories_data import ARMS
from deletion027_categories_worker import aggregate

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def collect(root):
    prepared = optional(root / "prepared.json", {})
    items = []
    for arm in ARMS:
        folder = root / "runs" / arm
        evaluations = [read_json(p) for p in sorted((folder / "evaluations").glob("*/summary.json"))]
        expected = len(prepared.get(arm, {}).get("val", []))
        evaluations = [e for e in evaluations if e.get("status") == "complete"
                       and e.get("findings_done") == e.get("findings_total") == expected]
        state = optional(folder / "status.json", {"status": "pending", "update": 0})
        final_updates = {e["update"] for e in evaluations}
        pending = state.get("update") if state.get("status") == "evaluating" and state.get("update") not in final_updates else None
        items.append({"arm": arm, "history": read_updates(folder / "training.jsonl"), "state": state,
                      "evaluations": evaluations, "pending_update": pending,
                      "baseline": aggregate(prepared.get(arm, {}).get("baseline_findings", [])),
                      "train_findings": len(prepared.get(arm, {}).get("train_keys", [])), "expected_findings": expected,
                      "exposure": prepared.get(arm, {}).get("exposure", {})})
    return {"state": optional(root / "status.json", {"status": "preparing"}), "arms": items, "updated_at": now()}


def plot_points(item, threshold):
    return [(e["update"], e["threshold_metrics"][str(threshold)]["full"]["dice"]) for e in item["evaluations"]]


def render(root):
    data = collect(root)
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    training = [(0, 0, "loss", "Dice + BCE TP2 total loss"), (0, 1, "remove_loss", "Unweighted FP BCE"),
                (0, 2, "preserve_loss", "Unweighted TP BCE"), (1, 0, "dice_loss", "Finding-aware Dice loss"),
                (2, 0, "grad_norm_before_clip", "Gradient norm before clipping")]
    for row, col, field, title in training:
        ax = axes[row, col]
        ax.set_title(title)
        for item, color in zip(data["arms"], COLORS):
            xs, ys = [r["update"] for r in item["history"]], [r.get(field, np.nan) for r in item["history"]]
            if xs:
                ax.plot(xs, ys, color=color, alpha=.16, linewidth=.6)
                ax.plot(xs, smooth(xs, ys), color=color, label=item["arm"], linewidth=1.1)
    for item, color, position in zip(data["arms"], COLORS, ((1, 1), (1, 2), (2, 1), (2, 2))):
        ax = axes[position]
        ax.set_title(f"{item['arm']}: full-validation Dice")
        baseline = item["baseline"]["full"]["dice"]
        if baseline is not None:
            ax.axhline(baseline, color="black", linestyle=":", label=f"Cached base {baseline:.6f}")
        for threshold, style in ((.5, "o-"), (.9, "s--")):
            points = plot_points(item, threshold)
            if points:
                ax.plot(*zip(*points), style, color=color, label=f"Removal {threshold:.2f}")
        if item["pending_update"] is not None:
            state = item["state"]
            ax.text(.03, .97, f"Update {item['pending_update']}: pending\n{state.get('findings_done', 0)}/{item['expected_findings']} findings",
                    transform=ax.transAxes, va="top", fontsize=9)
        elif not item["evaluations"]:
            ax.text(.5, .65, "Evaluation pending", transform=ax.transAxes, ha="center")
    maximum = max([100] + [r["update"] for a in data["arms"] for r in a["history"]]
                  + [e["update"] for a in data["arms"] for e in a["evaluations"]])
    for ax in axes.flat:
        ax.set_xlabel("Optimizer update")
        ax.set_xlim(0, maximum*1.02)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    title = "SYNTHETIC FIXTURE — NOT EXPERIMENT RESULTS" if data["state"].get("synthetic_fixture") else "Exp027 — category-specific Dice + BCE TP2"
    fig.suptitle(f"{title}\n{data['state']['status']} | complete full-validation points only", fontsize=14)
    tmp = output / ".live_dashboard.tmp.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, output / "live_dashboard.png")
    lines = [f"# {title}", "", f"Refreshed: {data['updated_at']}", "", f"Status: **{data['state']['status']}**.", "",
             "![Live category curves](live_dashboard.png)", "",
             "Each category fits its eligible original-training findings for 5000 updates (50 epochs). "
             "All use D + 0.25(F + 2K)/3, FP32/no TF32, LR 1e-4 and clipping 1.0. "
             "Faint training curves are raw; solid curves show a trailing 100-update mean. Different category difficulty affects loss magnitude.", "",
             "Validation points appear only after the entire category evaluation completes. An unfinished evaluation is pending; "
             "the preceding completed points remain visible. A/B/full metrics are calculated from the same finding records and saved, "
             "without separate A/B plots. Thresholds do not enter training. Reload the viewer if needed.", "",
             "The original 2d training finding from patient train_2936 is excluded; all validation findings remain. "
             "A/B are reused development subsets. Category 2e B has only three findings.", "",
             "| Category | Phase | Update / epoch | Loss / F / K / D | Step / load seconds | Phase elapsed seconds | Peak GiB | Training coverage | Evaluation progress |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in data["arms"]:
        state = item["state"]
        row = item["history"][-1] if item["history"] else {}
        update = max(state.get("update", 0), row.get("update", 0))
        memory = state.get("peak_memory_bytes", row.get("peak_memory_bytes"))
        losses = " / ".join(fmt(row.get(k)) for k in ("loss", "remove_loss", "preserve_loss", "dice_loss"))
        progress = f"{state.get('findings_done', 0)}/{item['expected_findings']} findings" if "findings_total" in state else "pending"
        if "tiles_total" in state:
            progress += f"; tiles {state.get('tiles_done', 0)}/{state['tiles_total']}"
        coverage = len({r["key"] for r in item["history"]})
        lines.append(f"| {item['arm']} | {state['status']} | {update} / {update/100:.2f} | {losses} | "
                     f"{fmt(row.get('step_seconds'))} / {fmt(row.get('loading_seconds'))} | {fmt(state.get('elapsed_seconds'))} | {fmt(memory/2**30 if memory else None)} | "
                     f"{coverage}/{item['train_findings']} | {progress} |")
    lines += ["", "## Latest completed full-validation results", "",
              "| Category | Completed update | Baseline Dice | Dice at 0.50 | Change | Dice at 0.90 | Change | Next evaluation |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for item in data["arms"]:
        final = item["evaluations"][-1] if item["evaluations"] else None
        metrics = [final["threshold_metrics"][str(t)]["full"] if final else {} for t in (.5, .9)]
        pending = f"Update {item['pending_update']}: pending" if item["pending_update"] is not None else "pending"
        lines.append(f"| {item['arm']} | {final['update'] if final else 'pending'} | {fmt(item['baseline']['full']['dice'])} | "
                     f"{fmt(metrics[0].get('dice'))} | {fmt(metrics[0].get('delta_dice'))} | "
                     f"{fmt(metrics[1].get('dice'))} | {fmt(metrics[1].get('delta_dice'))} | {pending} |")
    lines += ["", "[All thresholds, baseline tables and A/B/full records](threshold_sweep/index.md)", "",
              "No automatic threshold selection, ranking, checkpoint promotion or continuation."]
    for item in data["arms"]:
        lines += ["", f"[{item['arm']} training CSV](../runs/{item['arm']}/training.csv)"]
        for e in item["evaluations"]:
            lines += [f"[{item['arm']} update {e['update']} A/B/full records](../runs/{item['arm']}/evaluations/update_{e['update']:07d}/summary.json)"]
        if item["state"].get("error"):
            lines += [f"\n**{item['arm']} failure:** {item['state']['error']}"]
    if data["state"].get("error"):
        lines += [f"\n**Group failure:** {data['state']['error']}"]
    atomic_text(output / "live_dashboard.md", "\n".join(lines) + "\n")
    atomic_json(output / "dashboard.json", data)


def watch(root, stop_file):
    with lock(root / ".report_writer.lock"):
        last, prior = 0., None
        while True:
            paths = [root / "status.json", *root.glob("runs/*/evaluations/*/summary.json")]
            event = [(str(p), p.stat().st_mtime_ns) for p in paths if p.exists()]
            done = stop_file.exists()
            if done or event != prior or time.monotonic() - last >= 5:
                last = time.monotonic()
                render(root)
                prior = event
            if done:
                return
            time.sleep(1)
