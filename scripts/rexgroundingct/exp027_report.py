"""Local Markdown and PNG dashboard; CPU only, no server or telemetry."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rexgroundingct-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp027_common import ARMS, EXPOSURE, atomic_json, atomic_text, lock, now, read_json, read_updates

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def optional_json(path, default):
    return read_json(path) if Path(path).is_file() else default


def collect(root):
    root = Path(root)
    state = optional_json(root / "status.json", {"status": "awaiting_timing_approval", "active_phase": "benchmark"})
    phase = state.get("active_phase", "benchmark")
    prepared = optional_json(root / "prepared.json", {})
    arms = []
    for arm in ARMS:
        folder = root / phase / arm
        evaluations = []
        for p in sorted((folder / "evaluations").glob("update_*/summary.json")):
            evaluations.append(read_json(p))
        partials = [read_json(p) for p in sorted((folder / "evaluations").glob("update_*/partial.json"))
                    if not (p.parent / "summary.json").exists()]
        journal = folder / "training.jsonl"
        arms.append({"arm": arm, "exposure": EXPOSURE[arm],
                     "status": optional_json(folder / "status.json", {"status": "pending", "update": 0}),
                     "training": read_updates(journal) if journal.exists() else
                                 optional_json(folder / "training.json", {"updates": []})["updates"],
                     "evaluations": evaluations, "partial": partials[-1] if partials else None})
    return {"updated_at": now(), "state": state, "phase": phase, "arms": arms,
            "cache_progress": [read_json(p) for p in sorted(root.glob("cache_progress_gpu*.json"))],
            "baseline": prepared.get("baseline", {}), "split_counts": prepared.get("split", {}).get("counts", {}),
            "synthetic_fixture": bool(state.get("synthetic_fixture", False)), "ranking": "pending_user_review"}


def fmt(value):
    return "pending" if value is None else f"{value:.5f}" if isinstance(value, float) else str(value)


def render(root):
    """Caller owns .report_writer.lock; safe for a single coordinator or watch process."""
    root = Path(root)
    report = collect(root)
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    titles = (("Training patch loss", "Training patch Dice", "Training mean |residual|"),
              ("Evaluation Dice — A", "Evaluation Dice — B", "Evaluation Dice — full 2a"),
              ("Evaluation hit rate — A", "Evaluation hit rate — B", "Evaluation hit rate — full 2a"))
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            ax.set_title(titles[row][col])
            ax.set_xlabel("Optimizer updates")
            ax.grid(alpha=.2)
            if row:
                ax.set_ylim(0, 1)
    for item, color in zip(report["arms"], COLORS):
        arm, history = item["arm"], item["training"]
        for col, field in enumerate(("loss", "patch_dice", "residual_mean_abs")):
            if history:
                xs = [r["update"] for r in history]
                ys = np.asarray([r.get(field, np.nan) for r in history], dtype=float)
                axes[0, col].plot(xs, ys, color=color, alpha=.22, linewidth=.7)
                sums = np.r_[0., np.cumsum(np.nan_to_num(ys))]
                counts = np.r_[0, np.cumsum(np.isfinite(ys))]
                ends = np.arange(1, len(ys) + 1)
                starts = np.maximum(0, ends - 100)
                smoothed = (sums[ends] - sums[starts]) / np.maximum(1, counts[ends] - counts[starts])
                axes[0, col].plot(xs, smoothed, color=color, label=arm, linewidth=1.4)
        for col, half in enumerate(("A", "B", "full")):
            for row, metric in ((1, "dice"), (2, "hit_rate")):
                points = [(r["update"], r["metrics"][half].get(metric)) for r in item["evaluations"]]
                points = [(x, y) for x, y in points if y is not None]
                if points:
                    axes[row, col].plot(*zip(*points), "o-", color=color, label=arm, markersize=4)
                partial = item["partial"]
                if partial and partial["metrics"][half].get(metric) is not None:
                    m = partial["metrics"][half]
                    axes[row, col].plot(partial["update"], m[metric], marker="D", markerfacecolor="none",
                                        color=color, linestyle="none", label=f"{arm} provisional")
                    axes[row, col].plot(partial["update"], m["base_" + metric], marker="x", color=color,
                                        linestyle="none", label=f"{arm} subset base")
    for col, half in enumerate(("A", "B", "full")):
        for row, metric in ((1, "dice"), (2, "hit_rate")):
            value = report["baseline"].get(half, {}).get(metric)
            if value is not None:
                axes[row, col].axhline(value, color="black", linestyle="--", linewidth=1, label="Cached base")
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(.5, .5, "pending", transform=ax.transAxes, ha="center", color="gray")
    label = "SYNTHETIC FIXTURE — NOT EXPERIMENT RESULTS" if report["synthetic_fixture"] else "Exp027 — pending user review"
    fig.suptitle(f"{label}\n{report['state']['status']} | {report['phase']}", fontsize=14)
    fd, tmp = tempfile.mkstemp(dir=output, suffix=".png")
    os.close(fd)
    try:
        fig.savefig(tmp, dpi=120)
        os.replace(tmp, output / "live_dashboard.png")
    finally:
        plt.close(fig)
        if os.path.exists(tmp):
            os.unlink(tmp)
    lines = [f"# {label}", "", f"Updated: {report['updated_at']}", "",
             f"Status: **{report['state']['status']}**. Active phase: `{report['phase']}`.", "",
             "![Four-run curves](live_dashboard.png)", "",
             "Files refresh during execution (training ≤5 seconds; validation after each finding); your viewer may need a reload. Faint training lines are raw;",
             "solid training lines use a trailing window of up to 100 updates. Patch metrics are not full-volume metrics.", "",
             "| Run | Phase | Update | 100-update epochs | Latest patch loss | Step seconds | Elapsed seconds | Peak GiB | Evaluation progress |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for item in report["arms"]:
        s = item["status"]
        latest = item["training"][-1] if item["training"] else {}
        update = s.get("update", latest.get("update", 0))
        memory = s.get("peak_memory_bytes", s.get("timing", {}).get("peak_memory_bytes"))
        elapsed = s.get("elapsed_seconds", s.get("measured_wall_seconds", s.get("timing", {}).get("total_seconds")))
        progress = f"{s.get('findings_done', 0)}/{s.get('findings_total', '?')} findings" if "findings_total" in s else "pending"
        if "tiles_total" in s:
            progress += f"; {s['tiles_done']}/{s['tiles_total']} tiles"
        lines.append(f"| {item['arm']} | {s['status']} | {update} | {update / 100:.2f} | {fmt(latest.get('loss'))} | "
                     f"{fmt(latest.get('step_seconds'))} | {fmt(elapsed)} | "
                     f"{fmt(memory / 2**30 if memory is not None else None)} | {progress} |")
    if report["cache_progress"]:
        lines += ["", "Cache preparation: " + "; ".join(f"{r['done']}/{r['total']} verified cases ({r['name']}); "
                                                       f"reused {r.get('reused', 'pending')}, generated {r.get('generated', 'pending')}, "
                                                       f"{fmt(r['cache_bytes'] / 2**30 if 'cache_bytes' in r else None)} GiB verified; "
                                                       f"elapsed {fmt(r.get('elapsed_seconds'))} s; {r.get('error') or r.get('status', '')}"
                                                       for r in report["cache_progress"]) + "."]
    lines += ["", "## Latest evaluation (fixed arm order)", "",
              "| Run | Update | A Dice / hit rate | B Dice / hit rate | Full Dice / hit rate | A / B / full exposure |",
              "| --- | ---: | --- | --- | --- | --- |"]
    for item in report["arms"]:
        last = item["evaluations"][-1] if item["evaluations"] else None
        cells = []
        for half in ("A", "B", "full"):
            m = last["metrics"][half] if last else {}
            cells.append(f"{fmt(m.get('dice'))} / {fmt(m.get('hit_rate'))}")
        lines.append(f"| {item['arm']} | {fmt(last['update'] if last else None)} | " + " | ".join(cells) +
                     " | " + " / ".join(item["exposure"][h] for h in ("A", "B", "full")) + " |")
    lines += ["", "## Validation in progress — provisional", "",
              "Open diamonds are provisional scores; crosses are cached-base scores for exactly the same completed findings.",
              "Partial scores are not directly comparable to full-cohort scores. Each arm publishes independently.", "",
              "| Run | Update | A done/total: Dice / hit rate (subset base) | B done/total: Dice / hit rate (subset base) | Full done/total: Dice / hit rate (subset base) |",
              "| --- | ---: | --- | --- | --- |"]
    for item in report["arms"]:
        partial = item["partial"]
        cells = []
        for half in ("A", "B", "full"):
            m = partial["metrics"][half] if partial else {}
            expected = partial["expected_counts"][half] if partial else "pending"
            cells.append(f"{m.get('findings', 'pending')}/{expected}: {fmt(m.get('dice'))} / {fmt(m.get('hit_rate'))} "
                         f"(base {fmt(m.get('base_dice'))} / {fmt(m.get('base_hit_rate'))})")
        lines.append(f"| {item['arm']} | {fmt(partial['update'] if partial else None)} | " + " | ".join(cells) + " |")
    lines += ["", "## Training details and artifacts", ""]
    for item in report["arms"]:
        h = item["training"]
        last = h[-1] if h else {}
        counts = {source: sum(r.get("source") == source for r in h) for source in ("train", "A", "val")}
        lines += [f"- **{item['arm']}**: source updates `{counts}`; BCE {fmt(last.get('bce'))}, "
                  f"Dice loss {fmt(last.get('dice_loss'))}, residual penalty {fmt(last.get('residual_penalty'))}, "
                  f"LR {fmt(last.get('lr'))}. [Training CSV](../{report['phase']}/{item['arm']}/training.csv)."]
        if item["evaluations"]:
            last_eval = item["evaluations"][-1]
            lines.append(f"  [Per-finding metrics](../{report['phase']}/{item['arm']}/evaluations/"
                         f"update_{last_eval['update']:07d}/per_finding.csv).")
        if item["status"].get("error"):
            lines.append(f"  Failure: `{item['status']['error']}`")
    if report["state"].get("error"):
        lines += ["", f"Group failure: `{report['state']['error']}`"]
    timing = optional_json(root / "timing_report.json", None)
    if timing:
        lines += ["", "## Timing benchmark", "", "[Timing report](timing_report.md)"]
    lines += ["", "All scores and checkpoints are pending user review. No ranking, best-model selection, or automatic promotion.",
              "Run 4 scores measure training-set fitting capacity. Runs 2–3 full-validation scores mix training and held-out findings.", ""]
    atomic_json(output / "live_dashboard.json", report)
    atomic_text(output / "live_dashboard.md", "\n".join(lines))
    return report


def report_signature(root):
    """Cheap polling: per-update history is coalesced, finding/phase events are urgent."""
    root = Path(root)
    state = optional_json(root / "status.json", {})
    phase = state.get("active_phase", "benchmark")
    urgent = [("group", state)]
    history = []
    paths = list(root.glob("cache_progress_gpu*.json"))
    for arm in ARMS:
        folder = root / phase / arm
        status = optional_json(folder / "status.json", {})
        urgent.append((arm, status.get("status"), status.get("error")))
        paths += list((folder / "evaluations").glob("update_*/partial.json"))
        paths += list((folder / "evaluations").glob("update_*/summary.json"))
        for name in ("training.jsonl", "training.json", "status.json"):
            path = folder / name
            if path.exists():
                history.append((str(path), path.stat().st_mtime_ns, path.stat().st_size))
    for path in paths:
        urgent.append((str(path), path.stat().st_mtime_ns, path.stat().st_size))
    return urgent, history


def report_command(root, watch=False, interval=5, stop_file=None):
    with lock(Path(root) / ".report_writer.lock"):
        prior = None
        last = 0.0
        render_seconds = 0.0
        while True:
            signature = report_signature(root)
            stop = stop_file is not None and Path(stop_file).exists()
            if (prior is None or not watch or stop or signature[0] != prior[0]
                    or time.monotonic() - last >= max(0., interval - render_seconds - 1.)):
                started = time.monotonic()
                render(root)
                last, prior = time.monotonic(), signature
                render_seconds = last - started
            if not watch or stop:
                break
            time.sleep(1)


def timing_report(root, payload):
    atomic_json(Path(root) / "timing_report.json", payload)
    lines = ["# Exp027 timing benchmark", "", "Diagnostic only; training schedule awaits user decision.", "",
             f"Preparation: {payload['preparation_seconds']:.3f} seconds.",
             f"Four-arm training wall time: {payload['training_group_seconds']:.3f} seconds.",
             f"Four-arm evaluation wall time: {payload['evaluation_group_seconds']:.3f} seconds.", "",
             f"Combined four-arm training + evaluation wall time (including worker startup/warm-up): "
             f"{payload['training_group_seconds'] + payload['evaluation_group_seconds']:.3f} seconds.",
             "Per-arm measured updates below exclude setup and warm-up. Full step distributions are in training.csv.", "",
             "| Run | 100-update measured seconds | Setup | Warm-up | Step median / p95 | Data loading | Eval total | Eval inference / metrics / output | Peak training GiB |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |"]
    for arm in ARMS:
        t, e = payload["arms"][arm]["training"], payload["arms"][arm]["evaluation"]
        lines.append(f"| {arm} | {t['measured_wall_seconds']:.3f} | {t['setup_seconds']:.3f} | {t['warmup_seconds']:.3f} | "
                     f"{t['step_median_seconds']:.3f} / {t['step_p95_seconds']:.3f} | {t['data_seconds']:.3f} | "
                     f"{e['total_seconds']:.3f} | {e['inference_seconds']:.3f} / {e['metric_seconds']:.3f} / "
                     f"{e['output_seconds']:.3f} | {t['peak_memory_bytes'] / 2**30:.3f} |")
    lines += ["", "Process launch/import time is separate from per-arm training and evaluation:", ""]
    for arm in ARMS:
        t, e = payload["arms"][arm]["training"], payload["arms"][arm]["evaluation"]
        lines.append(f"- {arm}: train startup {fmt(t.get('process_startup_seconds'))} s; "
                     f"eval startup {fmt(e.get('process_startup_seconds'))} s; "
                     f"eval loading {fmt(e.get('loading_seconds'))} s; eval peak {e['peak_memory_bytes'] / 2**30:.3f} GiB.")
    atomic_text(Path(root) / "reports/timing_report.md", "\n".join(lines) + "\n")
