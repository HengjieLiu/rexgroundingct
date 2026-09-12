"""Independent CPU threshold sweeps and completion audit; no inference or selection."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from exp027_common import atomic_csv, atomic_json, atomic_text, digest, lock, now, read_json, read_updates, sha256
from deletion027_categories_data import ARMS, LOSSES, check_context
from deletion027_categories_worker import aggregate, edit_metrics


def threshold_counts(labels, scores, total_gt, thresholds, *, base=False):
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, np.float32)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid saved score vectors")
    ts, fs = np.sort(scores[labels]), np.sort(scores[~labels])
    # Match strict float32 comparisons, including values tied at the cutoff.
    cuts = np.asarray(thresholds, np.float32)
    rt = np.searchsorted(ts, cuts, side="left" if base else "right")
    rf = np.searchsorted(fs, cuts, side="left" if base else "right")
    if not base:
        rt, rf = len(ts)-rt, len(fs)-rf
    return [edit_metrics(len(ts), len(fs), total_gt, int(a), int(b)) for a, b in zip(rt, rf)]


def analyze_evaluation(config, root, arm, summary_path):
    summary = read_json(summary_path)
    out = root/"reports/threshold_sweep"/arm/f"update_{summary['update']:07d}"
    marker = out/"complete.json"
    source_hash = sha256(summary_path)
    if marker.exists():
        done = read_json(marker)
        if done["source_sha256"] != source_hash:
            raise ValueError("Completed analysis source changed")
        for name, checksum in done["outputs"].items():
            if sha256(out/name) != checksum:
                raise ValueError("Completed analysis artifact changed")
        return done
    if summary["status"] != "complete" or summary["arm"] != arm or len(summary["findings"]) != config["expected_val"][arm]:
        raise ValueError("Analysis requires the complete category evaluation")
    thresholds = [i/200 for i in range(201)]
    rows, base_rows = [], []
    for finding in summary["findings"]:
        score_path = summary_path.parent/"scores"/f"{digest(finding['key'])}.npz"
        if sha256(score_path) != finding["scores_sha256"]:
            raise ValueError("Saved score hash differs")
        with np.load(score_path, allow_pickle=False) as values:
            if values["scores"].dtype != np.float32 or np.any((values["scores"] < 0) | (values["scores"] > 1)):
                raise ValueError("Removal scores must be finite FP32 probabilities")
            metrics = threshold_counts(values["labels"], values["scores"], int(values["total_gt"]), thresholds)
            controls = threshold_counts(values["labels"], values["base_logits"], int(values["total_gt"]),
                                        config["base_thresholds"], base=True)
        identity = {k: finding[k] for k in ("key", "name", "half", "patient", "prompt")}
        for t, m in zip(thresholds, metrics):
            rows.append({"arm": arm, "update": summary["update"], "threshold": t, **identity, **m})
            if str(t) in finding["thresholds"] and m != finding["thresholds"][str(t)]:
                raise ValueError("Dense sweep differs from recorded live threshold")
        for t, m in zip(config["base_thresholds"], controls):
            if m != finding["base_thresholds"][str(t)]:
                raise ValueError("Base control differs from recorded metrics")
            base_rows.append({"arm": arm, "update": summary["update"], "base_logit_threshold": t, **identity, **m})
        if metrics[-1]["tp_removed"] or metrics[-1]["fp_removed"]:
            raise ValueError("Threshold-one identity failed")
    aggregates = []
    for t in thresholds:
        for half, m in aggregate([r for r in rows if r["threshold"] == t]).items():
            aggregates.append({"arm": arm, "update": summary["update"], "threshold": t, "subset": half, **m})
    base_aggregates = []
    for t in config["base_thresholds"]:
        for half, m in aggregate([r for r in base_rows if r["base_logit_threshold"] == t]).items():
            base_aggregates.append({"base_logit_threshold": t, "subset": half, **m})
    atomic_csv(out/"per_finding.csv", rows)
    atomic_csv(out/"metrics.csv", aggregates)
    atomic_json(out/"metrics.json", aggregates)
    atomic_csv(out/"base_per_finding.csv", base_rows)
    atomic_json(out/"base_metrics.json", base_aggregates)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for half, color in (("full", ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")[ARMS.index(arm)]),):
        subset = [r for r in aggregates if r["subset"] == half]
        axes[0, 0].plot(thresholds, [r["dice"] for r in subset], color=color, label=half)
        axes[0, 0].axhline(subset[0]["base_dice"], color=color, linestyle="--")
        axes[0, 1].plot([r["tp_loss"] for r in subset], [r["fp_removal"] for r in subset], color=color, label=half)
        axes[1, 0].plot(thresholds, [r["tp_loss"] for r in subset], color=color, label=half)
        axes[1, 1].plot(thresholds, [r["fp_removal"] for r in subset], color=color, label=half)
    titles = ("Mean finding Dice; dashed = cached base", "FP removal versus TP loss", "TP loss", "FP removal")
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title); ax.grid(alpha=.2); ax.legend()
        ax.set_xlabel("Removal threshold")
    axes[0, 1].set_xlabel("Pooled TP loss"); axes[0, 1].set_ylabel("Pooled FP removal")
    fig.suptitle(f"{arm}, update {summary['update']} — trained on original train; full validation (A+B)\nRetrospective curves; no operating point selected")
    tmp = out/".tradeoffs.tmp.png"
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    os.replace(tmp, out/"tradeoffs.png")
    outputs = {p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.name != "complete.json"}
    result = {"arm": arm, "update": summary["update"], "source_sha256": source_hash,
              "checkpoint_sha256": summary["checkpoint_sha256"], "findings": len(summary["findings"]),
              "thresholds": 201, "per_finding_rows": len(rows), "outputs": outputs, "completed_at": now()}
    atomic_json(marker, result)
    return result


def analyze_available(config, root):
    found = []
    for arm in ARMS:
        for path in sorted((root / "runs" / arm / "evaluations").glob("*/summary.json")):
            found.append(analyze_evaluation(config, root, arm, path))
    lines = ["# Category threshold sweeps", "", "Full-validation figures; A/B/full metrics retained in CSV/JSON. "
             "All thresholds are retrospective diagnostics, with no automatic selection.", "",
             "| Category | Update | All thresholds with baseline | Per-finding records | Full-cohort tradeoffs |",
             "| --- | --- | --- | --- | --- |"]
    for result in found:
        relative = f"{result['arm']}/update_{result['update']:07d}"
        folder = root / "reports/threshold_sweep" / relative
        metrics = read_json(folder / "metrics.json")
        table = [f"# {result['arm']} update {result['update']}: full-validation threshold sweep", "",
                 "A/B results are retained in metrics.csv; no extra inference was used.", "",
                 "| Removal threshold | Dice | Baseline Dice | Change | TP loss | FP removal |",
                 "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in metrics:
            if row["subset"] == "full":
                tp = "pending" if row["tp_loss"] is None else f"{row['tp_loss']:.6f}"
                fp = "pending" if row["fp_removal"] is None else f"{row['fp_removal']:.6f}"
                table.append(f"| {row['threshold']:.3f} | {row['dice']:.6f} | {row['base_dice']:.6f} | {row['delta_dice']:+.6f} | {tp} | {fp} |")
        atomic_text(folder / "full_table.md", "\n".join(table) + "\n")
        lines.append(f"| {result['arm']} | {result['update']} | [Table]({relative}/full_table.md) / [CSV]({relative}/metrics.csv) | "
                     f"[CSV]({relative}/per_finding.csv) | [Figure]({relative}/tradeoffs.png) |")
    if not found:
        lines += ["", "Evaluations pending."]
    atomic_text(root / "reports/threshold_sweep/index.md", "\n".join(lines) + "\n")
    return found


def watch(config, root):
    with lock(root / ".analysis.lock"):
        previous = None
        while True:
            paths = sorted(root.glob("runs/*/evaluations/*/summary.json"))
            signature = [(str(p), p.stat().st_mtime_ns) for p in paths]
            stopped = (root / "reports/.analysis_stop").exists()
            if signature != previous or stopped:
                analyze_available(config, root)
                previous = signature
            if stopped:
                return
            time.sleep(1)


def audit(config, root):
    import torch
    from fit_027_deletion import make_editor
    from deletion027_worker import weight_hash
    from run_027_deletion_categories import verify_barrier
    context = check_context(config, root)
    prepared = read_json(root / "prepared.json")
    for update in config["evaluation_updates"]:
        verify_barrier(root, update, prepared, context["sha256"])
    results = []
    for arm in ARMS:
        folder = root / "runs" / arm
        history = read_updates(folder / "training.jsonl")
        fixed = read_json(folder / "schedule.json")
        if len(history) != config["total_updates"] or digest(fixed["events"]) != context["schedule_hashes"][arm]:
            raise ValueError("Completion update/schedule count differs")
        for event, row in zip(fixed["events"], history):
            if (any(row[k] != v for k, v in event.items() if k not in ("tp", "fp"))
                    or row["scheduled_tp"] != event["tp"] or row["scheduled_fp"] != event["fp"]
                    or row["category"] != arm or row["loss_arm"] != config["loss_arm"]):
                raise ValueError("Training journal differs from immutable schedule")
            if not np.isfinite([row[k] for k in ("loss", "remove_loss", "preserve_loss", "dice_loss", "grad_norm_before_clip")]).all():
                raise ValueError("Non-finite training history")
        for update in range(0, config["total_updates"] + 1, 100):
            state = torch.load(folder / "checkpoints" / f"update_{update:07d}.pth", map_location="cpu", weights_only=False)
            if (state["update"] != update or state["sampling_cursor"] != update
                    or state["initial_sha256"] != config["initial_weights_sha256"]
                    or state["schedule_sha256"] != context["schedule_hashes"][arm]
                    or state["arm"] != arm or state["loss_spec"] != LOSSES[arm]
                    or state["context_sha256"] != context["sha256"] or len(state["history"]) != update
                    or state["precision"] != "fp32" or state["tf32"]):
                raise ValueError("Completion checkpoint provenance differs")
            model = make_editor(config, context["base_config"])
            model.load_state_dict(state["model"])
            if weight_hash(model) != state["model_sha256"]:
                raise ValueError("Checkpoint model digest differs")
            if update == config["total_updates"] and state["history"] != history:
                raise ValueError("Final checkpoint/history discrepancy")
        for update in config["evaluation_updates"]:
            summary = read_json(folder / "evaluations" / f"update_{update:07d}" / "summary.json")
            for threshold in config["display_thresholds"]:
                for half in ("A", "B", "full"):
                    results.append({"category": arm, "update": update, "threshold": threshold,
                                    "subset": half, **summary["threshold_metrics"][str(threshold)][half]})
    analyses = analyze_available(config, root)
    expected = len(ARMS) * len(config["evaluation_updates"])
    if len(analyses) != expected:
        raise ValueError("Missing dense threshold analyses")
    result = {"status": "pending_user_review", "updates": len(ARMS)*config["total_updates"],
              "evaluations": expected, "dense_analyses": expected, "completed_at": now()}
    atomic_csv(root / "reports/fixed_order_results.csv", results)
    atomic_json(root / "reports/fixed_order_results.json", results)
    atomic_json(root / "reports/completion.json", result)
    return result
