"""Single CPU train-only loss dashboard writer, with provisional A/B/full scores."""
from __future__ import annotations

import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deletion027_ablation_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp027_common import atomic_json, atomic_text, lock, now, read_json, read_updates
from deletion027_report import COLORS, fmt, optional, smooth
from deletion027_train_ablation_data import ARMS, LOSSES
from deletion027_train_ablation_worker import aggregate


def collect(root):
    manifest = optional(root/"input_manifest.json", {})
    arms = []
    for arm in ARMS:
        folder = root/"runs"/arm
        finals = [read_json(p) for p in sorted((folder/"evaluations").glob("*/summary.json"))]
        partials = [read_json(p) for p in sorted((folder/"evaluations").glob("*/partial.json"))
                    if not (p.parent/"summary.json").exists()]
        arms.append({"arm": arm, "history": read_updates(folder/"training.jsonl"),
                     "evaluations": finals, "partial": partials[-1] if partials else None,
                     "state": optional(folder/"status.json", {"status": "pending", "update": 0})})
    return {"state": optional(root/"status.json", {"status": "preparing"}), "arms": arms,
            "config": optional(root/"config.json", {"total_updates": 10000}),
            "baseline": aggregate(manifest.get("baseline_findings", [])), "updated_at": now(),
            "preparation": [read_json(p) for p in sorted(root.glob("prepare_progress_*.json"))]}


def curve(ax, history, field, color, label):
    xs = [r["update"] for r in history]
    ys = [r.get(field, np.nan) for r in history]
    if xs:
        ax.plot(xs, ys, color=color, alpha=.16, linewidth=.6)
        ax.plot(xs, smooth(xs, ys), color=color, label=label, linewidth=1.1)


def render(root):
    data = collect(root)
    output = root/"reports"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    training = ((0, 0, "loss", "Total objective (different definitions)"),
                (0, 1, "remove_loss", "Unweighted FP BCE"),
                (0, 2, "preserve_loss", "Unweighted TP BCE"),
                (1, 0, "dice_loss", "Finding-aware Dice loss (all arms)"),
                (2, 0, "grad_norm_before_clip", "Gradient norm before clipping"))
    for row, col, field, title in training:
        axes[row, col].set_title(title)
        for item, color in zip(data["arms"], COLORS):
            curve(axes[row, col], item["history"], field, color, item["arm"])
    for row, threshold in ((1, .5), (2, .9)):
        for col, half in ((1, "A"), (2, "B")):
            ax = axes[row, col]
            ax.set_title(f"{half} full-volume Dice — removal threshold {threshold:.2f}")
            baseline = data["baseline"][half]["dice"]
            if baseline is not None:
                ax.axhline(baseline, color="black", linestyle="--", linewidth=.8, label="Cached base")
            has_results = False
            for item, color in zip(data["arms"], COLORS):
                points = [(e["update"], e["threshold_metrics"][str(threshold)][half]["dice"])
                          for e in item["evaluations"]]
                points = [(x, y) for x, y in points if y is not None]
                if points:
                    has_results = True
                    ax.plot(*zip(*points), "o-", color=color, label=item["arm"])
                partial = item["partial"]
                if partial:
                    m = partial["threshold_metrics"][str(threshold)][half]
                    if m["dice"] is not None:
                        has_results = True
                        ax.plot(partial["update"], m["dice"], marker="D", markerfacecolor="none",
                                color=color, linestyle="none", label=item["arm"]+" provisional")
                        ax.plot(partial["update"], m["base_dice"], marker="x", color=color, linestyle="none")
            if not has_results:
                ax.text(.5, .65, "pending", transform=ax.transAxes, ha="center")
    last_update = max([100] + [r["update"] for a in data["arms"] for r in a["history"]]
                      + [e["update"] for a in data["arms"] for e in a["evaluations"]]
                      + [a["partial"]["update"] for a in data["arms"] if a["partial"]])
    for ax in axes.flat:
        ax.set_xlabel("Optimizer update")
        ax.set_xlim(0, last_update*1.02)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "pending", transform=ax.transAxes, ha="center")
    title = ("SYNTHETIC FIXTURE — NOT EXPERIMENT RESULTS" if data["state"].get("synthetic_fixture")
             else "Exp027 — four losses fitted on original training data")
    fig.suptitle(f"{title}\n{data['state']['status']} | A and B: held-out development", fontsize=14)
    tmp = output/".live_dashboard.tmp.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    os.replace(tmp, output/"live_dashboard.png")
    lines = [f"# {title}", "", f"Refreshed: {data['updated_at']}", "",
             f"Status: **{data['state']['status']}**. Stop at update {data['config']['total_updates']:,} after all final evaluations and analysis.", "",
             "![Live 3×3 curves](live_dashboard.png)", "",
             "All four arms fit the same 1,119 eligible original training findings. "
             "A (35 findings) and B (34 findings) are both excluded from gradients. "
             "Full = A+B; all are repeatedly inspected development data, not an untouched final test. "
             "One original training finding has no base prediction and is excluded. "
             "An epoch is 100 updates, not a full dataset pass.", "",
             "Updates are appended immediately. This independent CPU board polls each second and refreshes "
             "within about five seconds, with prompt provisional finding results. Reload the viewer if needed.",
             "Faint curves are raw; solid training curves use a trailing 100-update window. "
             "Hollow diamonds are provisional and crosses show matching-subset baselines.",
             "Total objectives have different definitions and cannot be ranked by magnitude. "
             "Dice arms also rescale BCE. Finding-aware Dice edits one patch with the outside prediction held fixed; "
             "it is not exact blended full-volume Dice. Thresholds do not enter training.", "",
             "| Arm | Objective | Phase | Update / epoch | Total / F / K / D | Step / load s | Elapsed s | Peak GiB | Progress / train coverage |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in data["arms"]:
        state = item["state"]
        row = item["history"][-1] if item["history"] else {}
        update = max(state.get("update", 0), row.get("update", 0))
        memory = state.get("peak_memory_bytes", row.get("peak_memory_bytes"))
        progress = (f"{state['findings_done']}/{state['findings_total']} findings"
                    if "findings_total" in state else "pending")
        if "tiles_total" in state:
            progress += f"; {state['tiles_done']}/{state['tiles_total']} tiles"
        keys = {r['key'] for r in item['history'] if 'key' in r}
        if keys:
            progress += f"; train {len(keys)}/1119 findings"
        losses = " / ".join(fmt(row.get(k)) for k in ("loss", "remove_loss", "preserve_loss", "dice_loss"))
        lines.append(f"| {item['arm']} | {LOSSES[item['arm']]['formula']} | {state['status']} | "
                     f"{update} / {update/100:.2f} | {losses} | {fmt(row.get('step_seconds'))} / {fmt(row.get('loading_seconds'))} | "
                     f"{fmt(state.get('elapsed_seconds'))} | {fmt(memory/2**30 if memory is not None else None)} | {progress} |")
    if data["preparation"]:
        lines += ["", "CPU complete train+validation cache verification: "+"; ".join(
            f"worker {i}: {p['done']}/{p['total']} CTs" for i, p in enumerate(data["preparation"]))]
        size = sum(p.get("verified_cache_bytes", 0) for p in data["preparation"])
        elapsed = max(p.get("elapsed_seconds", 0) for p in data["preparation"])
        lines += [f"Verified logit/GT cache storage: {size/2**30:.2f} GiB; elapsed verification: {elapsed:.1f} s. "
                  "Existing arrays reused; generated logits: 0. All 864 CTs must pass before training."]
    for threshold in (.5, .9):
        gaps = []
        lines += ["", f"## Full-volume results at removal threshold {threshold:.2f}", "",
                  "| Arm | Update / status | Subset | Findings | Dice / matching base | Change | Hits gained / lost | TP lost | FP removed | Improved / worse |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for item in data["arms"]:
            result = item["partial"] or (item["evaluations"][-1] if item["evaluations"] else None)
            for half in ("A", "B", "full"):
                m = result["threshold_metrics"][str(threshold)][half] if result else {}
                progress = f"{m.get('findings', 0)}/{result['expected_counts'][half]}" if result else "pending"
                status = f"{result['update']} / {'provisional' if item['partial'] else 'complete'}" if result else "pending"
                lines.append(f"| {item['arm']} | {status} | {half} | {progress} | {fmt(m.get('dice'))} / "
                             f"{fmt(m.get('base_dice'))} | {fmt(m.get('delta_dice'))} | "
                             f"{fmt(m.get('hits_gained'))} / {fmt(m.get('hits_lost'))} | "
                             f"{fmt(m.get('tp_loss'), True)} | {fmt(m.get('fp_removal'), True)} | "
                             f"{fmt(m.get('improved'))} / {fmt(m.get('worsened'))} |")
            if result:
                a, b = (result["threshold_metrics"][str(threshold)][h] for h in ("A", "B"))
                gap = a["delta_dice"]-b["delta_dice"] if a.get("delta_dice") is not None and b.get("delta_dice") is not None else None
                gaps.append(f"{item['arm']} A-minus-B improvement gap: {fmt(gap)} "
                            f"({'provisional' if item['partial'] else 'complete'}; both subsets are held out from these refiners).")
        for gap in gaps:
            lines += ["", gap]
    lines += ["", "## Records and diagnostics", "",
              "[All thresholds and tradeoff figures](threshold_sweep/index.md) · "
              "[Matched-update comparison with A-only training](comparison_with_a.md) · "
              "[Historical train-only reference checks](reference_checks.json) · "
              "[Completion audit](completion.json)", "",
              "TP-retention flags are diagnostics, not rejection gates. Full per-finding records include damage, "
              "precision/recall, residual FN and baseline comparisons. No automatic threshold selection, model ranking or continuation."]
    for item in data["arms"]:
        lines += ["", f"[{item['arm']} training CSV](../runs/{item['arm']}/training.csv)"]
        for result in item["evaluations"]:
            lines += [f"[{item['arm']} update {result['update']} per-finding results]"
                      f"(../runs/{item['arm']}/evaluations/update_{result['update']:07d}/summary.json)"]
        if item["state"].get("error"):
            lines += ["", f"**{item['arm']} failure:** {item['state']['error']}"]
    if data["state"].get("error"):
        lines += ["", f"**Group failure:** {data['state']['error']}"]
    atomic_text(output/"live_dashboard.md", "\n".join(lines)+"\n")
    atomic_json(output/"dashboard.json", data)


def watch(root, stop_file):
    with lock(root/".report_writer.lock"):
        last, prior = 0., None
        while True:
            paths = [root/"status.json", *root.glob("runs/*/evaluations/*/partial.json"),
                     *root.glob("runs/*/evaluations/*/summary.json")]
            event = [(str(p), p.stat().st_mtime_ns) for p in paths if p.exists()]
            done = stop_file.exists()
            if done or event != prior or time.monotonic()-last >= 5:
                last = time.monotonic()
                render(root)
                prior = event
            if done:
                return
            time.sleep(min(1., max(.05, 5-(time.monotonic()-last))))
