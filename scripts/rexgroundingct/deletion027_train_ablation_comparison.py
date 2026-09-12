"""CPU comparison tables and strict historical train-only reference checks."""
from pathlib import Path

from exp027_common import atomic_csv, atomic_json, atomic_text, now, read_json, sha256
from deletion027_train_ablation_data import ARMS, REFERENCE_UPDATES, check_context


def checked_reference(context, path):
    path = Path(path)
    if sha256(path) != context["reference_files"][str(path)]:
        raise ValueError(f"Historical reference changed: {path}")
    return path


def verify_reference(config, root, update):
    """Called by the coordinator before releasing an early evaluation barrier."""
    if update not in REFERENCE_UPDATES:
        return None
    import torch
    from deletion027_worker import weight_hash
    from fit_027_deletion import make_editor
    context = check_context(config, root)
    old = Path(config["historical_runtime"])/"runs/run1_train"
    new = root/"runs"/ARMS[0]
    tail = f"checkpoints/update_{update:07d}.pth"
    prior = torch.load(checked_reference(context, old/tail), map_location="cpu", weights_only=False)
    current = torch.load(new/tail, map_location="cpu", weights_only=False)
    model = make_editor(config, context["base_config"])
    model.load_state_dict(current["model"])
    current_hash = weight_hash(model)
    model.load_state_dict(prior["model"])
    prior_hash = weight_hash(model)
    tail = f"evaluations/update_{update:07d}/summary.json"
    previous = read_json(checked_reference(context, old/tail))
    actual = read_json(new/tail)
    keys = {r["key"] for r in read_json(root/"prepared.json")["val"]}
    left = {r["key"]: r["thresholds"] for r in previous["findings"]}
    right = {r["key"]: r["thresholds"] for r in actual["findings"]}
    result = {"update": update, "weights_identical": current_hash == prior_hash,
              "metrics_identical": left == right and set(left) == keys,
              "current_model_sha256": current_hash, "historical_model_sha256": prior_hash,
              "current_checkpoint_sha256": sha256(new/f"checkpoints/update_{update:07d}.pth"),
              "historical_checkpoint_sha256": sha256(old/f"checkpoints/update_{update:07d}.pth"),
              "checked_at": now()}
    if (current["model_sha256"] != current_hash or prior["model_sha256"] != prior_hash
            or current["context_sha256"] != context["sha256"]
            or actual["checkpoint_sha256"] != result["current_checkpoint_sha256"]
            or previous["checkpoint_sha256"] != result["historical_checkpoint_sha256"]):
        result["integrity_failure"] = True
    path = root/"reports/reference_checks.json"
    checks = read_json(path).get("checks", []) if path.exists() else []
    checks = sorted([r for r in checks if r["update"] != update] + [result], key=lambda r: r["update"])
    passed = all(r["weights_identical"] and r["metrics_identical"] and not r.get("integrity_failure") for r in checks)
    atomic_json(path, {"status": "passed_so_far" if passed else "reference_discrepancy_requires_investigation",
                       "expected_updates": REFERENCE_UPDATES, "checks": checks, "updated_at": now()})
    if not passed:
        raise ValueError("Historical train-only BCE reference differs; inspect reports/reference_checks.json")
    return result


def number(value):
    return "pending" if value is None else f"{value:.6f}"


def threshold_comparisons(config, root, found):
    """Publish all 201 rows as individual arms arrive; never choose a threshold."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from deletion027_report import COLORS
    output = root/"reports/threshold_sweep"
    updates = sorted({r["update"] for r in found})
    links = []
    for update in updates:
        available = {r["arm"]: r for r in found if r["update"] == update}
        signature = {arm: row["source_sha256"] for arm, row in available.items()}
        folder = output/f"comparison_update_{update:07d}"
        marker = folder/"published.json"
        links.append((update, len(available), folder.name))
        if marker.exists() and read_json(marker).get("sources") == signature:
            for name, checksum in read_json(marker)["outputs"].items():
                if sha256(folder/name) != checksum:
                    raise ValueError("Consolidated threshold table changed")
            continue
        records, lookup = [], {}
        for arm in ARMS:
            if arm not in available:
                continue
            rows = read_json(output/arm/f"update_{update:07d}/metrics.json")
            if len(rows) != 603:
                raise ValueError("Dense threshold grid incomplete")
            for row in rows:
                lookup[arm, row["subset"], row["threshold"]] = row
            records.extend(rows)
        text = [f"# Update {update}: full threshold comparison", "",
                f"Completed analyses: {len(available)}/4 arms. Missing arms are pending. "
                "All models train on original train only; A and B are held-out development.", "",
                "All 201 thresholds from 0 to 1 at step 0.005. Delete only score > threshold. "
                "Frozen base uses its unchanged probability threshold 0.5. No operating point is selected.", "",
                "[All metrics CSV](metrics.csv) · [JSON](metrics.json)", "",
                "![Threshold comparison](comparison.png)", ""]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for col, subset in enumerate(("A", "B", "full")):
            base = next(r["base_dice"] for r in records if r["subset"] == subset)
            text += [f"## {subset} mean per-finding Dice", "", f"[CSV]({subset}.csv)", "",
                     "| Threshold | Frozen base | " + " | ".join(ARMS) + " |",
                     "| --- | --- | --- | --- | --- | --- |"]
            wide = []
            for i in range(201):
                t = i/200
                values = {arm: lookup[arm, subset, t]["dice"] if arm in available else None for arm in ARMS}
                for arm in available:
                    if abs(lookup[arm, subset, t]["base_dice"]-base) > 1e-12:
                        raise ValueError("Threshold table baseline mismatch")
                wide.append({"threshold": t, "baseline": base, **values})
                text.append(f"| {t:.3f} | {base:.6f} | " + " | ".join(number(values[a]) for a in ARMS) + " |")
            atomic_csv(folder/f"{subset}.csv", wide)
            for row in range(2):
                ax = axes[row, col]
                for arm, color in zip(ARMS, COLORS):
                    if arm in available:
                        ax.plot([r["threshold"] for r in wide], [r[arm] for r in wide], label=arm, color=color)
                ax.axhline(base, color="black", ls="--", label="Frozen base")
                if row:
                    values = [r["dice"] for r in records if r["subset"] == subset and r["threshold"] >= .25]
                    ax.set_ylim(max(0, base-.035), max(base+.02, max(values)+.005))
                else:
                    ax.set_ylim(0, max(.4, max(r["dice"] for r in records if r["subset"] == subset)+.02))
                ax.set_title(f"{subset}: {'enlarged range' if row else 'full range'}")
                ax.set_xlabel("Removal threshold"); ax.set_ylabel("Mean finding Dice")
                ax.set_xlim(0, 1); ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.suptitle(f"Original train-only losses, update {update} — {len(available)}/4 analyses available")
        tmp = folder/".comparison.tmp.png"
        fig.savefig(tmp, dpi=120); plt.close(fig); os.replace(tmp, folder/"comparison.png")
        atomic_csv(folder/"metrics.csv", records)
        atomic_json(folder/"metrics.json", records)
        atomic_text(folder/"all_thresholds.md", "\n".join(text)+"\n")
        names = ["A.csv", "B.csv", "full.csv", "metrics.csv", "metrics.json", "comparison.png", "all_thresholds.md"]
        atomic_json(marker, {"sources": signature, "updated_at": now(),
                             "outputs": {name: sha256(folder/name) for name in names}})
    return links


def compare_with_a(config, root):
    context = check_context(config, root)
    rows = []
    for update in REFERENCE_UPDATES:
        for arm in ARMS:
            path = root/"runs"/arm/f"evaluations/update_{update:07d}/summary.json"
            if not path.exists():
                continue
            old_path = Path(config["a_only_runtime"])/"runs"/arm/f"evaluations/update_{update:07d}/summary.json"
            old = read_json(checked_reference(context, old_path))
            current = read_json(path)
            for threshold in config["display_thresholds"]:
                for subset in ("A", "B", "full"):
                    a, b = (s["threshold_metrics"][str(threshold)][subset] for s in (old, current))
                    if abs(a["base_dice"]-b["base_dice"]) > 1e-12 or a["findings"] != b["findings"]:
                        raise ValueError("Matched-update A-only baseline/coverage differs")
                    rows.append({"update": update, "arm": arm, "threshold": threshold, "subset": subset,
                                 "baseline": b["base_dice"], "trained_on_A_dice": a["dice"],
                                 "trained_on_train_dice": b["dice"], "train_minus_A": b["dice"]-a["dice"],
                                 "train_fp_removal": b["fp_removal"], "train_tp_loss": b["tp_loss"],
                                 "A_fp_removal": a["fp_removal"], "A_tp_loss": a["tp_loss"]})
    text = ["# Matched-update comparison with A-only training", "",
            "Previous models fit A; current models fit original train only. A/B/full are held out "
            "from the current refiners. B has repeatedly informed development. Data source and its "
            "patch schedule differ; this comparison holds the loss recipe and update count fixed.", "",
            "Updates above 2,000 are additional training exposure and have no matched A-only checkpoint. "
            "Rows retain arm order. No ranking or selection.", "",
            "[CSV](comparison_with_a.csv) · [JSON](comparison_with_a.json)", "",
            "| Update | Arm | Threshold | Subset | Base | Trained on A | Trained on train | Difference |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        text.append(f"| {r['update']} | {r['arm']} | {r['threshold']:.2f} | {r['subset']} | "
                    f"{r['baseline']:.6f} | {r['trained_on_A_dice']:.6f} | {r['trained_on_train_dice']:.6f} | {r['train_minus_A']:+.6f} |")
    if not rows:
        text += ["", "Evaluations pending."]
    atomic_text(root/"reports/comparison_with_a.md", "\n".join(text)+"\n")
    atomic_json(root/"reports/comparison_with_a.json", rows)
    atomic_csv(root/"reports/comparison_with_a.csv", rows)
