"""Independent CPU live report for four deletion runs; no GPU work or web server."""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deletion027_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp027_common import ARMS, atomic_json, atomic_text, lock, now, read_json, read_updates
from deletion027_worker import aggregate

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
EXPOSURE = ("Train; A development / B gradient-held-out", "Train+A; A fitted / B gradient-held-out",
            "A fitted / B gradient-held-out", "A+B fitted; capacity diagnostic")


def optional(path, default):
    return read_json(path) if path.exists() else default


def fmt(value, percent=False):
    return "pending" if value is None else f"{value:.2%}" if percent else f"{value:.5f}" if isinstance(value, float) else str(value)


def collect(root):
    state = optional(root / "status.json", {"status": "preparing"})
    manifest = optional(root / "input_manifest.json", {})
    arms = []
    for arm in ARMS:
        folder = root / "runs" / arm
        history = read_updates(folder / "training.jsonl")
        finals = [read_json(p) for p in sorted((folder / "evaluations").glob("*/summary.json"))]
        partials = [read_json(p) for p in sorted((folder / "evaluations").glob("*/partial.json")) if not (p.parent / "summary.json").exists()]
        arms.append({"arm": arm, "history": history, "evaluations": finals, "partial": partials[-1] if partials else None,
                     "state": optional(folder / "status.json", {"status": "pending", "update": 0})})
    return {"state": state, "arms": arms, "baseline": aggregate(manifest.get("baseline_findings", [])),
            "preparation": [read_json(p) for p in sorted(root.glob("prepare_progress_*.json"))],
            "eligible_counts": manifest.get("eligible_counts", {}), "updated_at": now()}


def smooth(xs, ys, window=100):
    xs, ys = np.asarray(xs), np.asarray(ys, float)
    starts = np.searchsorted(xs, xs-window, side="right")
    ends = np.arange(1, len(xs)+1)
    sums = np.r_[0., np.cumsum(np.nan_to_num(ys))]
    counts = np.r_[0, np.cumsum(np.isfinite(ys))]
    n = counts[ends]-counts[starts]
    return np.divide(sums[ends]-sums[starts], n, out=np.full(len(xs), np.nan), where=n > 0)


def render(root):
    data = collect(root)
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(17, 15), constrained_layout=True)
    for i, title in enumerate(("Training loss", "GT-containing patch Dice", "GT-empty patch FP removal")):
        axes[0, i].set_title(title)
    for row, title in ((1, "Full-volume Dice"), (2, "Pooled TP retention"), (3, "FP removal")):
        for col, half in enumerate(("A", "B", "full")):
            axes[row, col].set_title(f"{title} — {half}")
    for item, color in zip(data["arms"], COLORS):
        history = item["history"]
        series = (history, [r for r in history if not r["gt_empty"]], [r for r in history if r["gt_empty"]])
        for col, (rows, metric) in enumerate(zip(series, ("loss", "patch_dice", "fp_removal"))):
            xs = [r["update"] for r in rows]
            ys = [np.nan if r[metric] is None else r[metric] for r in rows]
            if xs:
                axes[0, col].plot(xs, ys, color=color, alpha=.18, linewidth=.7)
                axes[0, col].plot(xs, smooth(xs, ys), color=color, label=item["arm"])
        for col, half in enumerate(("A", "B", "full")):
            for row, metric in ((1, "dice"), (2, "tp_retention"), (3, "fp_removal")):
                points = [(e["update"], e["metrics"][half].get(metric)) for e in item["evaluations"]]
                points = [(x, y) for x, y in points if y is not None]
                if points:
                    axes[row, col].plot(*zip(*points), "o-", color=color, label=item["arm"])
                p = item["partial"]
                if p and p["metrics"][half].get(metric) is not None:
                    m = p["metrics"][half]
                    axes[row, col].plot(p["update"], m[metric], marker="D", markerfacecolor="none", color=color,
                                        linestyle="none", label=item["arm"] + " provisional")
                    baseline = m["base_dice"] if metric == "dice" else 1. if metric == "tp_retention" else 0.
                    axes[row, col].plot(p["update"], baseline, marker="x", color=color, linestyle="none")
    for row, metric in ((1, "dice"), (2, "tp_retention"), (3, "fp_removal")):
        for col, half in enumerate(("A", "B", "full")):
            baseline = data["baseline"][half].get(metric)
            if baseline is not None:
                axes[row, col].axhline(baseline, color="black", linestyle="--", linewidth=.8, label="Cached base")
            axes[row, col].set_ylim(-.02, 1.02)
    for ax in axes.flat:
        ax.set_xlabel("Optimizer update")
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "pending", ha="center", transform=ax.transAxes)
    label = "SYNTHETIC FIXTURE — NOT EXPERIMENT RESULTS" if data["state"].get("synthetic_fixture") else "Exp027 deletion — four data-source arms"
    fig.suptitle(f"{label}\n{data['state']['status']} | main removal threshold 0.90; A is development data", fontsize=14)
    tmp = output / ".live_dashboard.tmp.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, output / "live_dashboard.png")
    lines = [f"# {label}", "", f"Refreshed: {data['updated_at']}", "",
             f"Status: **{data['state']['status']}**. Stop after the epoch20 evaluation barrier for user review.", "",
             "![Live curves](live_dashboard.png)", "",
             "Training refreshes about every5 seconds; finding results appear independently of other arms. Reload your viewer if needed.",
             "Faint curves are raw; solid curves use a trailing100-update window. Patch metrics are separate from full-volume metrics.",
             "Loss components are shown unweighted: total loss = FP term + 2 × TP term.",
             "Hollow diamonds are provisional; crosses show their matching-subset base. Missing/undefined values remain pending.", "",
             "| Run | Phase | Update / epoch | Loss (FP / TP) | Unique findings / tiles | Step s | Phase elapsed s | Peak GiB | Progress |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in data["arms"]:
        state = item["state"]
        latest = item["history"][-1] if item["history"] else {}
        update = max(state.get("update", 0), latest.get("update", 0))
        progress = f"{state.get('findings_done', 0)}/{state.get('findings_total', '?')} findings" if "findings_total" in state else "pending"
        if "tiles_total" in state:
            progress += f"; {state['tiles_done']}/{state['tiles_total']} tiles"
        memory = state.get("peak_memory_bytes", latest.get("peak_memory_bytes"))
        unique_keys = len({r["key"] for r in item["history"]}) if item["history"] else None
        unique_tiles = len({(r["key"], r["tile_index"]) for r in item["history"]}) if item["history"] else None
        lines.append(f"| {item['arm']} | {state['status']} | {update} / {update/100:.2f} | "
                     f"{fmt(latest.get('loss'))} ({fmt(latest.get('remove_loss'))} / {fmt(latest.get('preserve_loss'))}) | "
                     f"{fmt(unique_keys)} / {fmt(unique_tiles)} | {fmt(latest.get('step_seconds'))} | {fmt(state.get('elapsed_seconds'))} | "
                     f"{fmt(memory/2**30 if memory is not None else None)} | {progress} |")
    if data["preparation"]:
        lines += ["", "CPU preparation: " + "; ".join(f"worker{i}: {p['done']}/{p['total']} cases" for i, p in enumerate(data["preparation"]))]
    lines += ["", "| Run | Exposure |", "| --- | --- |"]
    lines += [f"| {arm} | {exposure} |" for arm, exposure in zip(ARMS, EXPOSURE)]
    lines += ["", "## Latest full-volume results at threshold0.90", "",
              "| Run | Update | Subset | Completed | Dice / matching base | Hit rate | TP retained | FP removed | Min finding TP retention | <95% / <80% |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    details = []
    for item in data["arms"]:
        result = item["partial"] or (item["evaluations"][-1] if item["evaluations"] else None)
        for half in ("A", "B", "full"):
            m = result["metrics"][half] if result else {}
            count = f"{m.get('findings', 0)}/{result['expected_counts'][half]}" if result else "pending"
            label_status = " provisional" if item["partial"] else ""
            lines.append(f"| {item['arm']} | {result['update'] if result else 'pending'}{label_status} | {half} | {count} | "
                         f"{fmt(m.get('dice'))} / {fmt(m.get('base_dice'))} | {fmt(m.get('hit_rate'), True)} | "
                         f"{fmt(m.get('tp_retention'), True)} | {fmt(m.get('fp_removal'), True)} | "
                         f"{fmt(m.get('tp_retention_min'), True)} | {fmt(m.get('below_95'))} / {fmt(m.get('below_80'))} |")
        if result:
            flags = []
            for record in result["findings"]:
                m = record["thresholds"]["0.9"]
                if m["tp_retention"] is not None and m["tp_retention"] < .95:
                    flags.append((record, m))
            if flags:
                details += ["", f"### {item['arm']} individual TP losses ({len(flags)} findings)", "",
                          "| Finding | Half | TP retained | TP removed / base TP | Dice change | Severity |",
                          "| --- | --- | --- | --- | --- | --- |"]
                for record, m in sorted(flags, key=lambda x: x[1]["tp_retention"]):
                    details.append(f"| {record['key']} | {record['half']} | {m['tp_retention']:.2%} | {m['tp_removed']} / {m['base_tp']} | "
                                 f"{m['delta_dice']:+.5f} | {'SEVERE >20% TP loss' if m['tp_retention'] < .8 else '>5% TP loss'} |")
            path = f"../runs/{item['arm']}/evaluations/update_{result['update']:07d}"
            details += ["", f"[{item['arm']} complete threshold records]({path}/{'partial' if item['partial'] else 'summary'}.json)", ""]
        if item["state"].get("error"):
            details += ["", f"**{item['arm']} failure:** {item['state']['error']}", ""]
    lines += details
    if data["state"].get("error"):
        lines += ["", f"**Group failure:** {data['state']['error']}", ""]
    lines += ["", "No automatic checkpoint selection or ranking. A+B metrics mix exposure in runs2–3; run4 is in-sample.", ""]
    atomic_text(output / "live_dashboard.md", "\n".join(lines))
    atomic_json(output / "dashboard.json", data)


def watch(root, stop_file):
    with lock(root / ".report_writer.lock"):
        last = 0.
        prior_event = None
        while True:
            event_files = [root / "status.json", *root.glob("runs/*/evaluations/*/partial.json"),
                           *root.glob("runs/*/evaluations/*/summary.json")]
            event = [(str(p), p.stat().st_mtime_ns) for p in event_files if p.exists()]
            done = stop_file.exists()
            if done or event != prior_event or time.monotonic()-last >= 5:
                last = time.monotonic()
                render(root)
                prior_event = event
            if done:
                return
            time.sleep(min(1., max(.05, 5-(time.monotonic()-last))))
