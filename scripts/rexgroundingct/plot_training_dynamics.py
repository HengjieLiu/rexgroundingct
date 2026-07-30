#!/usr/bin/env python3
"""Generate canonical within-experiment training-dynamics comparisons."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rexgroundingct-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import EXP_ROOT, sha256_file, utc_now_iso


SCRIPT_VERSION = "training_dynamics_v1"
TOTAL_UPDATES = 10_000
ROLLING_WINDOW = 100
ROBUST_PERCENTILE = 99.5
EXPERIMENT_IDS = {
    "008": "008_voxtell_dual_branch_proposal_refinement_ablation",
    "009": "009_voxtell_s3_attention_coupling_ablation",
    "011": "011_voxtell_v123_e4d4_ct_normalization_ablation",
}
MILESTONES = {
    "008": (2_000, 4_000, 6_000, 8_000, 10_000),
    "009": (500, 2_000, 4_000, 6_000, 8_000, 10_000),
    "011": (500, 2_000, 4_000, 6_000, 8_000, 10_000),
}
MILESTONE_DIR_RE = re.compile(r"epoch(\d{3})$")
COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)


@dataclass(frozen=True)
class ArmHistory:
    experiment: str
    profile: str
    arm: str
    label: str
    arm_dir: Path
    run_group: str
    updates: tuple[dict[str, Any], ...]
    source_files: tuple[dict[str, Any], ...]

    @property
    def global_updates(self) -> np.ndarray:
        return np.asarray(
            [int(item["global_update"]) for item in self.updates],
            dtype=np.int64,
        )

    @property
    def last_update(self) -> int:
        return int(self.updates[-1]["global_update"])

    def series(
        self,
        field: str,
        *,
        fallbacks: Sequence[str] = (),
        default: float = math.nan,
    ) -> np.ndarray:
        values: list[float] = []
        fields = (field, *fallbacks)
        for update in self.updates:
            value: Any = None
            found = False
            for candidate in fields:
                if candidate in update and update[candidate] is not None:
                    value = update[candidate]
                    found = True
                    break
            values.append(float(value) if found else float(default))
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class Curve:
    history: ArmHistory
    values: np.ndarray
    label: str
    color: str
    linestyle: str = "-"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=path.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(
            temporary,
            format=path.suffix.lstrip("."),
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def values_equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            values_equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            values_equivalent(a, b) for a, b in zip(left, right)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        left_float = float(left)
        right_float = float(right)
        if math.isnan(left_float) and math.isnan(right_float):
            return True
        return math.isclose(left_float, right_float, rel_tol=1e-7, abs_tol=1e-7)
    return left == right


def metric_paths(arm_dir: Path) -> list[Path]:
    reports_dir = arm_dir / "reports"
    paths = sorted(reports_dir.glob("training_metrics_segment_epoch*.json"))
    paths.extend(sorted(reports_dir.glob("training_metrics_recovery_epoch*.json")))
    final = reports_dir / "training_metrics.json"
    if final.is_file():
        paths.append(final)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def load_arm_history(
    *,
    experiment: str,
    profile: str,
    arm: str,
    label: str,
    arm_dir: Path,
) -> ArmHistory:
    paths = metric_paths(arm_dir)
    if not paths:
        raise FileNotFoundError(f"No training metric files found under {arm_dir}")

    updates_by_index: dict[int, dict[str, Any]] = {}
    update_sources: dict[int, Path] = {}
    source_files: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        updates = payload.get("updates")
        if not isinstance(updates, list):
            raise ValueError(f"{path} does not contain an updates list")
        source_files.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "updates": len(updates),
                "start_global_update": payload.get("start_global_update"),
                "completed_updates": payload.get("completed_updates"),
            }
        )
        for item in updates:
            if not isinstance(item, dict) or item.get("global_update") is None:
                raise ValueError(f"{path} contains an update without global_update")
            index = int(item["global_update"])
            if index <= 0:
                raise ValueError(f"{path} contains invalid global_update {index}")
            if index in updates_by_index:
                if not values_equivalent(updates_by_index[index], item):
                    raise ValueError(
                        "Conflicting duplicate global_update "
                        f"{index} in {update_sources[index]} and {path}"
                    )
                continue
            updates_by_index[index] = item
            update_sources[index] = path

    indices = sorted(updates_by_index)
    if not indices:
        raise ValueError(f"No update records found under {arm_dir}")
    expected = list(range(1, indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected).difference(indices))
        raise ValueError(
            f"{arm_dir} is not a contiguous prefix from update 1; "
            f"first missing updates: {missing[:10]}"
        )
    if indices[-1] > TOTAL_UPDATES:
        raise ValueError(
            f"{arm_dir} contains update {indices[-1]} beyond {TOTAL_UPDATES}"
        )

    return ArmHistory(
        experiment=experiment,
        profile=profile,
        arm=arm,
        label=label,
        arm_dir=arm_dir.resolve(),
        run_group=arm_dir.resolve().parent.name,
        updates=tuple(updates_by_index[index] for index in indices),
        source_files=tuple(source_files),
    )


def resolve_group(exp_root: Path, experiment: str, alias: str) -> Path:
    path = exp_root / experiment / "runs" / alias
    if not path.exists():
        raise FileNotFoundError(f"Missing run selector: {path}")
    return path.resolve()


def latest_reported_update(group_dir: Path) -> int:
    completed_epochs: list[int] = []
    for marker in group_dir.glob("milestones/epoch*/report.complete"):
        match = MILESTONE_DIR_RE.match(marker.parent.name)
        if match is not None:
            completed_epochs.append(int(match.group(1)))
    if not completed_epochs:
        raise ValueError(f"No completed report milestone found under {group_dir}")
    return max(completed_epochs) * 100


def trim_history(history: ArmHistory, maximum_update: int) -> ArmHistory:
    updates = tuple(
        item
        for item in history.updates
        if int(item["global_update"]) <= maximum_update
    )
    if not updates or int(updates[-1]["global_update"]) != maximum_update:
        raise ValueError(
            f"{history.arm_dir} does not contain completed report update "
            f"{maximum_update}"
        )
    return ArmHistory(
        experiment=history.experiment,
        profile=history.profile,
        arm=history.arm,
        label=history.label,
        arm_dir=history.arm_dir,
        run_group=history.run_group,
        updates=updates,
        source_files=history.source_files,
    )


def load_histories(
    exp_root: Path,
    experiments: Sequence[str],
) -> dict[str, list[ArmHistory]]:
    result: dict[str, list[ArmHistory]] = {}
    if "008" in experiments:
        experiment = EXPERIMENT_IDS["008"]
        group = resolve_group(exp_root, experiment, "latest")
        arms = (
            ("v1_sharedfusion_softguide", "Shared fusion, soft guide"),
            ("v1_dualfusion_softguide", "Dual fusion, soft guide"),
            ("v2_dualfusion_precision", "Dual fusion, precision"),
            ("v3_dualfusion_softguide_joint", "Dual fusion, joint guide"),
        )
        result["008"] = [
            load_arm_history(
                experiment=experiment,
                profile="dual_branch",
                arm=arm,
                label=label,
                arm_dir=group / arm,
            )
            for arm, label in arms
        ]
    if "009" in experiments:
        experiment = EXPERIMENT_IDS["009"]
        group = resolve_group(exp_root, experiment, "latest")
        arms = (
            ("baseline_cont100", "Continuation baseline"),
            (
                "s3v1_fixedrho_suppress_half_quarter",
                "S3 v1 fixed-rho suppression",
            ),
            (
                "s3v2_balanced_feature_half_quarter",
                "S3 v2 balanced feature",
            ),
            (
                "s3v3_logit_residual_half_quarter",
                "S3 v3 logit residual",
            ),
        )
        result["009"] = [
            load_arm_history(
                experiment=experiment,
                profile="s3_attention",
                arm=arm,
                label=label,
                arm_dir=group / arm,
            )
            for arm, label in arms
        ]
    if "011" in experiments:
        experiment = EXPERIMENT_IDS["011"]
        e4_group = resolve_group(exp_root, experiment, "latest_e4d4")
        e5_group = resolve_group(exp_root, experiment, "latest_e5d4")
        normalization_arms = (
            ("zscore", "Native z-score"),
            ("clip1024_zscore", "Clipped z-score"),
            ("clip1024_linear", "Clipped linear HU"),
        )
        histories: list[ArmHistory] = []
        for profile, group in (("e4d4", e4_group), ("e5d4", e5_group)):
            for suffix, label in normalization_arms:
                arm = f"v123_{profile}_{suffix}"
                histories.append(
                    load_arm_history(
                        experiment=experiment,
                        profile=profile,
                        arm=arm,
                        label=f"{label} ({profile})",
                        arm_dir=group / arm,
                    )
                )
        latest_e5_update = latest_reported_update(e5_group)
        histories = [
            (
                trim_history(item, latest_e5_update)
                if item.profile == "e5d4"
                else item
            )
            for item in histories
        ]
        e5_lengths = {
            history.last_update
            for history in histories
            if history.profile == "e5d4"
        }
        if len(e5_lengths) != 1:
            raise ValueError(
                "Exp011 e5d4 arms are not at an equal completed barrier: "
                f"{sorted(e5_lengths)}"
            )
        result["011"] = histories

        exp006 = "006_voxtell_cached_native_v123_lr_ablation"
        exp006_group = resolve_group(exp_root, exp006, "latest")
        result["006_reference"] = [
            load_arm_history(
                experiment=exp006,
                profile="e5d4_reference",
                arm="v123_cached_e5_d4",
                label="Exp006 native z-score e5d4",
                arm_dir=exp006_group / "v123_cached_e5_d4",
            )
        ]
    return result


def rolling_mean(values: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    finite = np.isfinite(values)
    sums = np.cumsum(np.where(finite, values, 0.0), dtype=np.float64)
    counts = np.cumsum(finite.astype(np.int64))
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        total = sums[index] - (sums[start - 1] if start else 0.0)
        count = counts[index] - (counts[start - 1] if start else 0)
        if count:
            result[index] = total / float(count)
    return result


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    finite_values = values[np.isfinite(values)]
    first = finite_values[:ROLLING_WINDOW]
    last = finite_values[-ROLLING_WINDOW:]
    first_mean = float(np.mean(first)) if len(first) else None
    last_mean = float(np.mean(last)) if len(last) else None
    return {
        "observations": int(len(values)),
        "finite_observations": int(len(finite_values)),
        "nonfinite_observations": int(len(values) - len(finite_values)),
        "first100_mean": first_mean,
        "last100_mean": last_mean,
        "overall_mean": (
            float(np.mean(finite_values)) if len(finite_values) else None
        ),
        "maximum": float(np.max(finite_values)) if len(finite_values) else None,
        "relative_last100_vs_first100": (
            (last_mean - first_mean) / first_mean
            if first_mean not in (None, 0.0) and last_mean is not None
            else None
        ),
    }


def total_values(history: ArmHistory) -> np.ndarray:
    if history.experiment.startswith("008_"):
        return history.series("loss_total", fallbacks=("loss",))
    return history.series("loss")


def curve(
    history: ArmHistory,
    values: np.ndarray,
    color_index: int,
    *,
    label: str | None = None,
    linestyle: str = "-",
) -> Curve:
    return Curve(
        history=history,
        values=values,
        label=label or history.label,
        color=COLORS[color_index % len(COLORS)],
        linestyle=linestyle,
    )


def plot_panel(
    axis: plt.Axes,
    curves: Sequence[Curve],
    *,
    title: str,
    milestones: Sequence[int],
    ylabel: str,
    show_raw: bool = True,
) -> dict[str, Any]:
    finite_arrays = [
        item.values[np.isfinite(item.values)]
        for item in curves
        if np.isfinite(item.values).any()
    ]
    if not finite_arrays:
        raise ValueError(f"No finite values available for panel {title!r}")
    combined = np.concatenate(finite_arrays)
    upper = float(np.percentile(combined, ROBUST_PERCENTILE))
    rolling_max = max(
        float(np.nanmax(rolling_mean(item.values)))
        for item in curves
        if np.isfinite(item.values).any()
    )
    upper = max(upper * 1.05, rolling_max * 1.05, 1e-3)
    lower_data = float(np.min(combined))
    lower = min(0.0, lower_data * 1.05)
    clipped = int(np.sum(combined > upper))

    for milestone in milestones:
        axis.axvline(
            milestone,
            color="#999999",
            linewidth=0.7,
            alpha=0.35,
            zorder=0,
        )
    for item in curves:
        updates = item.history.global_updates
        finite = np.isfinite(item.values)
        if not finite.any():
            continue
        if show_raw:
            axis.plot(
                updates[finite],
                item.values[finite],
                color=item.color,
                linestyle=item.linestyle,
                linewidth=0.45,
                alpha=0.10,
            )
        smoothed = rolling_mean(item.values)
        smooth_finite = np.isfinite(smoothed)
        axis.plot(
            updates[smooth_finite],
            smoothed[smooth_finite],
            color=item.color,
            linestyle=item.linestyle,
            linewidth=2.0,
            label=item.label,
        )
    axis.set_xlim(0, TOTAL_UPDATES)
    axis.set_ylim(lower, upper)
    axis.set_title(title)
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.8)
    axis.legend(fontsize=8, frameon=True, loc="upper right")
    axis.text(
        0.01,
        0.02,
        (
            f"Raw (thin), {ROLLING_WINDOW}-update mean (bold); "
            f"p{ROBUST_PERCENTILE:g} cap; {clipped} above."
            if show_raw
            else f"Trailing {ROLLING_WINDOW}-update means; panel-specific y-scale."
        ),
        transform=axis.transAxes,
        fontsize=7,
        color="#444444",
        va="bottom",
    )
    return {
        "title": title,
        "y_min": lower,
        "y_max": upper,
        "robust_percentile": ROBUST_PERCENTILE,
        "raw_points_above_y_max": clipped,
    }


def summary_row(
    history: ArmHistory,
    metric: str,
    values: np.ndarray,
) -> dict[str, Any]:
    stats = finite_stats(values)
    return {
        "experiment": history.experiment,
        "run_group": history.run_group,
        "profile": history.profile,
        "arm": history.arm,
        "label": history.label,
        "metric": metric,
        "last_update": history.last_update,
        **stats,
    }


def write_experiment_summary(
    *,
    output_dir: Path,
    experiment: str,
    histories: Sequence[ArmHistory],
    rows: Sequence[dict[str, Any]],
    panels: Sequence[dict[str, Any]],
    figures: Sequence[str],
    notes: Sequence[str],
) -> None:
    payload = {
        "created_at_utc": utc_now_iso(),
        "script_version": SCRIPT_VERSION,
        "experiment": experiment,
        "run_groups": sorted({item.run_group for item in histories}),
        "figures": list(figures),
        "panels": list(panels),
        "notes": list(notes),
        "arms": [
            {
                "arm": item.arm,
                "label": item.label,
                "profile": item.profile,
                "run_group": item.run_group,
                "arm_dir": str(item.arm_dir),
                "last_update": item.last_update,
                "source_files": list(item.source_files),
            }
            for item in histories
        ],
        "metrics": list(rows),
    }
    atomic_write_json(output_dir / "training_dynamics_summary.json", payload)
    fieldnames = (
        "experiment",
        "run_group",
        "profile",
        "arm",
        "label",
        "metric",
        "last_update",
        "observations",
        "finite_observations",
        "nonfinite_observations",
        "first100_mean",
        "last100_mean",
        "overall_mean",
        "maximum",
        "relative_last100_vs_first100",
    )
    atomic_write_csv(
        output_dir / "training_dynamics_summary.csv",
        rows,
        fieldnames,
    )
    lines = [
        f"# {experiment} Training Dynamics",
        "",
        "Canonical CPU-generated training-loss figures for this experiment.",
        "",
    ]
    for figure in figures:
        lines.extend([f"![{figure}]({figure})", ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            *[f"- {note}" for note in notes],
            "- Thin lines are raw per-update values; bold lines are trailing "
            f"{ROLLING_WINDOW}-update means.",
            "- The x-axis is optimizer update, not samples seen or wall time.",
            "- Exact source hashes and numeric summaries are in "
            "`training_dynamics_summary.json`.",
            "",
        ]
    )
    atomic_write_text(output_dir / "README.md", "\n".join(lines))


def render_exp008(
    exp_root: Path,
    histories: Sequence[ArmHistory],
) -> tuple[list[dict[str, Any]], list[Path]]:
    experiment = EXPERIMENT_IDS["008"]
    output_dir = exp_root / experiment / "reports" / "training_dynamics"
    total_curves = [
        curve(item, total_values(item), index)
        for index, item in enumerate(histories)
    ]
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    panels = [
        plot_panel(
            axis,
            total_curves,
            title="Exp008 dual-branch total training objective",
            milestones=MILESTONES["008"],
            ylabel="Total training loss",
        )
    ]
    total_path = output_dir / "total_loss_vs_optimizer_update.png"
    atomic_savefig(fig, total_path)
    plt.close(fig)

    component_builders: tuple[
        tuple[str, str, Callable[[ArmHistory], np.ndarray]], ...
    ] = (
        ("Total objective", "Loss", total_values),
        (
            "Final-branch v123",
            "Loss",
            lambda item: item.series("loss_final_v123"),
        ),
        (
            "Proposal-branch v123",
            "Loss",
            lambda item: item.series("loss_proposal_v123"),
        ),
        (
            "Weighted auxiliary contribution",
            "Weighted loss",
            lambda item: np.nan_to_num(
                item.series("loss_weighted_proposal"), nan=0.0
            )
            + np.nan_to_num(
                item.series("loss_weighted_final_precision"), nan=0.0
            ),
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    component_rows: list[dict[str, Any]] = []
    for panel_index, (title, ylabel, builder) in enumerate(component_builders):
        panel_curves = [
            curve(item, builder(item), index)
            for index, item in enumerate(histories)
        ]
        panels.append(
            plot_panel(
                axes.flat[panel_index],
                panel_curves,
                title=title,
                milestones=MILESTONES["008"],
                ylabel=ylabel,
            )
        )
        metric_name = title.lower().replace(" ", "_").replace("-", "_")
        component_rows.extend(
            summary_row(item, metric_name, builder(item))
            for item in histories
        )
    component_path = output_dir / "objective_components_vs_optimizer_update.png"
    atomic_savefig(fig, component_path)
    plt.close(fig)

    rows = [
        summary_row(item, "total_loss", total_values(item))
        for item in histories
    ]
    rows.extend(component_rows)
    write_experiment_summary(
        output_dir=output_dir,
        experiment=experiment,
        histories=histories,
        rows=rows,
        panels=panels,
        figures=(total_path.name, component_path.name),
        notes=(
            "Compare curves within Exp008 only; this objective includes final, "
            "proposal, and variant-dependent precision terms.",
            "The weighted auxiliary panel is weighted proposal plus weighted "
            "final-precision contribution.",
        ),
    )
    return rows, [total_path, component_path]


def render_exp009(
    exp_root: Path,
    histories: Sequence[ArmHistory],
) -> tuple[list[dict[str, Any]], list[Path]]:
    experiment = EXPERIMENT_IDS["009"]
    output_dir = exp_root / experiment / "reports" / "training_dynamics"
    total_curves = [
        curve(item, total_values(item), index)
        for index, item in enumerate(histories)
    ]
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    panels = [
        plot_panel(
            axis,
            total_curves,
            title="Exp009 S3 attention total training objective",
            milestones=MILESTONES["009"],
            ylabel="Total training loss",
        )
    ]
    total_path = output_dir / "total_loss_vs_optimizer_update.png"
    atomic_savefig(fig, total_path)
    plt.close(fig)

    def base_v123(item: ArmHistory) -> np.ndarray:
        if item.arm == "baseline_cont100":
            return item.series("loss")
        return item.series("loss_s3_base_v123")

    def weighted_auxiliary(item: ArmHistory) -> np.ndarray:
        if item.arm == "baseline_cont100":
            return np.zeros(len(item.updates), dtype=np.float64)
        return item.series("loss_s3_weighted_auxiliary_loss")

    component_builders: tuple[
        tuple[str, str, Callable[[ArmHistory], np.ndarray]], ...
    ] = (
        ("Total objective", "Loss", total_values),
        ("Base v123 segmentation loss", "Loss", base_v123),
        ("Weighted S3 auxiliary loss", "Weighted loss", weighted_auxiliary),
    )
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
    component_rows: list[dict[str, Any]] = []
    for panel_index, (title, ylabel, builder) in enumerate(component_builders):
        panel_curves = [
            curve(item, builder(item), index)
            for index, item in enumerate(histories)
        ]
        panels.append(
            plot_panel(
                axes.flat[panel_index],
                panel_curves,
                title=title,
                milestones=MILESTONES["009"],
                ylabel=ylabel,
            )
        )
        metric_name = title.lower().replace(" ", "_")
        component_rows.extend(
            summary_row(item, metric_name, builder(item))
            for item in histories
        )
    component_path = output_dir / "objective_components_vs_optimizer_update.png"
    atomic_savefig(fig, component_path)
    plt.close(fig)

    rows = [
        summary_row(item, "total_loss", total_values(item))
        for item in histories
    ]
    rows.extend(component_rows)
    write_experiment_summary(
        output_dir=output_dir,
        experiment=experiment,
        histories=histories,
        rows=rows,
        panels=panels,
        figures=(total_path.name, component_path.name),
        notes=(
            "The continuation baseline has no S3 auxiliary term, represented "
            "as zero in the weighted-auxiliary panel.",
            "Base-v123 loss is directly comparable within Exp009; total "
            "objective values include variant-specific S3 auxiliary terms.",
        ),
    )
    return rows, [total_path, component_path]


def render_exp011(
    exp_root: Path,
    histories: Sequence[ArmHistory],
    reference: ArmHistory,
) -> tuple[list[dict[str, Any]], list[Path]]:
    experiment = EXPERIMENT_IDS["011"]
    output_dir = exp_root / experiment / "reports" / "training_dynamics"
    by_profile = {
        profile: [item for item in histories if item.profile == profile]
        for profile in ("e4d4", "e5d4")
    }
    panels: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    for panel_index, profile in enumerate(("e4d4", "e5d4")):
        panel_curves = [
            curve(item, total_values(item), index)
            for index, item in enumerate(by_profile[profile])
        ]
        if profile == "e5d4":
            panel_curves.append(
                curve(
                    reference,
                    total_values(reference),
                    7,
                    label="Exp006 native e5d4 reference",
                    linestyle="--",
                )
            )
        panels.append(
            plot_panel(
                axes[panel_index],
                panel_curves,
                title=f"Exp011 {profile} normalization comparison",
                milestones=MILESTONES["011"],
                ylabel="v123 training loss",
            )
        )
    total_path = output_dir / "total_loss_vs_optimizer_update.png"
    atomic_savefig(fig, total_path)
    plt.close(fig)

    suffixes = (
        ("zscore", "Native z-score"),
        ("clip1024_zscore", "Clipped z-score"),
        ("clip1024_linear", "Clipped linear HU"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
    for panel_index, (suffix, title) in enumerate(suffixes):
        e4 = next(item for item in histories if item.arm == f"v123_e4d4_{suffix}")
        e5 = next(item for item in histories if item.arm == f"v123_e5d4_{suffix}")
        panel_curves = [
            curve(e4, total_values(e4), 0, label="Exp011 e4d4"),
            curve(e5, total_values(e5), 3, label="Exp011 e5d4"),
        ]
        if suffix == "zscore":
            panel_curves.append(
                curve(
                    reference,
                    total_values(reference),
                    7,
                    label="Exp006 e5d4 replication reference",
                    linestyle="--",
                )
            )
        panels.append(
            plot_panel(
                axes[panel_index],
                panel_curves,
                title=title,
                milestones=MILESTONES["011"],
                ylabel="v123 training loss",
            )
        )
    paired_path = output_dir / "paired_loss_comparisons.png"
    atomic_savefig(fig, paired_path)
    plt.close(fig)

    all_histories = [*histories, reference]
    rows = [
        summary_row(item, "total_loss", total_values(item))
        for item in all_histories
    ]
    write_experiment_summary(
        output_dir=output_dir,
        experiment=experiment,
        histories=all_histories,
        rows=rows,
        panels=panels,
        figures=(total_path.name, paired_path.name),
        notes=(
            "All Exp011 curves use the same aggregate v123 objective; internal "
            "Dice/BCE/deep-supervision components were not logged separately.",
            "The Exp006 dashed curve is a native-z-score e5d4 replication "
            "reference and is not part of the Exp011 three-arm barrier.",
            "The active e5d4 curves stop at the latest completed synchronous "
            "report milestone and refresh independently from training.",
        ),
    )
    return rows, [total_path, paired_path]


def overview_curves(
    key: str,
    histories: dict[str, list[ArmHistory]],
) -> list[Curve]:
    if key in {"008", "009"}:
        return [
            curve(item, total_values(item), index)
            for index, item in enumerate(histories[key])
        ]
    result: list[Curve] = []
    normalization_colors = {
        "zscore": 0,
        "clip1024_zscore": 1,
        "clip1024_linear": 2,
    }
    for item in histories["011"]:
        suffix = item.arm.removeprefix(f"v123_{item.profile}_")
        result.append(
            curve(
                item,
                total_values(item),
                normalization_colors[suffix],
                label=item.label,
                linestyle=":" if item.profile == "e4d4" else "-",
            )
        )
    reference = histories["006_reference"][0]
    result.append(
        curve(
            reference,
            total_values(reference),
            7,
            label="Exp006 native e5d4 reference",
            linestyle="--",
        )
    )
    return result


def render_central_gallery(
    *,
    exp_root: Path,
    output_root: Path,
    selected: Sequence[str],
    histories: dict[str, list[ArmHistory]],
    all_rows: Sequence[dict[str, Any]],
    per_experiment_figures: dict[str, list[Path]],
    input_hash: str,
    input_files: Sequence[dict[str, Any]],
) -> list[Path]:
    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(17, 5.5 * len(selected)),
        constrained_layout=True,
    )
    if len(selected) == 1:
        axes = np.asarray([axes])
    titles = {
        "008": "Exp008 dual-branch objective",
        "009": "Exp009 S3-attention objective",
        "011": "Exp011 normalization and encoder-LR objective",
    }
    panels = []
    for axis, key in zip(axes, selected):
        panels.append(
            plot_panel(
                axis,
                overview_curves(key, histories),
                title=titles[key],
                milestones=MILESTONES[key],
                ylabel="Experiment-specific training loss",
                show_raw=False,
            )
        )
    overview_path = output_root / "training_loss_overview.png"
    atomic_savefig(fig, overview_path)
    plt.close(fig)

    total_rows = [row for row in all_rows if row["metric"] == "total_loss"]
    fieldnames = (
        "experiment",
        "run_group",
        "profile",
        "arm",
        "label",
        "metric",
        "last_update",
        "observations",
        "finite_observations",
        "nonfinite_observations",
        "first100_mean",
        "last100_mean",
        "overall_mean",
        "maximum",
        "relative_last100_vs_first100",
    )
    atomic_write_csv(
        output_root / "training_dynamics_summary.csv",
        total_rows,
        fieldnames,
    )
    manifest = {
        "created_at_utc": utc_now_iso(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__)),
        "input_hash": input_hash,
        "selected_experiments": list(selected),
        "panels": panels,
        "inputs": list(input_files),
        "outputs": {
            "overview": str(overview_path),
            "summary_csv": str(output_root / "training_dynamics_summary.csv"),
            "per_experiment_figures": {
                key: [str(path) for path in paths]
                for key, paths in per_experiment_figures.items()
            },
        },
        "comparability_warning": (
            "Compare loss values within each experiment panel only. Exp008, "
            "Exp009, and Exp011 optimize different objectives."
        ),
    }
    atomic_write_json(output_root / "manifest.json", manifest)

    lines = [
        "# VoxTell Training-Dynamics Gallery",
        "",
        "![Cross-experiment training-loss overview](training_loss_overview.png)",
        "",
        "> Compare curves within each panel only. Exp008, Exp009, and Exp011 "
        "optimize different objectives, so their absolute loss values are not "
        "cross-experiment scores.",
        "",
        "## Detailed Figures",
        "",
    ]
    for key in selected:
        experiment = EXPERIMENT_IDS[key]
        lines.append(f"### {experiment}")
        lines.append("")
        for path in per_experiment_figures[key]:
            relative = os.path.relpath(path, output_root)
            lines.extend([f"![{path.name}]({relative})", ""])
    lines.extend(
        [
            "## Provenance",
            "",
            f"- Generated by `{Path(__file__).name}` version `{SCRIPT_VERSION}`.",
            f"- Input manifest hash: `{input_hash}`.",
            "- Machine-readable provenance: `manifest.json`.",
            "- Numeric arm summaries: `training_dynamics_summary.csv`.",
            "",
        ]
    )
    readme_path = output_root / "README.md"
    atomic_write_text(readme_path, "\n".join(lines))
    return [
        overview_path,
        output_root / "training_dynamics_summary.csv",
        output_root / "manifest.json",
        readme_path,
    ]


def collect_input_files(
    histories: dict[str, list[ArmHistory]],
) -> list[dict[str, Any]]:
    inputs: dict[str, dict[str, Any]] = {}
    for items in histories.values():
        for history in items:
            for source in history.source_files:
                inputs[source["path"]] = dict(source)
    return [inputs[path] for path in sorted(inputs)]


def input_digest(
    selected: Sequence[str],
    histories: dict[str, list[ArmHistory]],
) -> str:
    plotted_histories = []
    for key in sorted(histories):
        for history in histories[key]:
            update_hash = hashlib.sha256(
                json.dumps(
                    history.updates,
                    sort_keys=True,
                    allow_nan=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            plotted_histories.append(
                {
                    "experiment": history.experiment,
                    "run_group": history.run_group,
                    "profile": history.profile,
                    "arm": history.arm,
                    "last_update": history.last_update,
                    "updates_sha256": update_hash,
                }
            )
    payload = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__)),
        "selected": list(selected),
        "plotted_histories": plotted_histories,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def expected_outputs(
    exp_root: Path,
    output_root: Path,
    selected: Sequence[str],
) -> list[Path]:
    result = [
        output_root / "README.md",
        output_root / "training_loss_overview.png",
        output_root / "training_dynamics_summary.csv",
        output_root / "manifest.json",
    ]
    for key in selected:
        output_dir = (
            exp_root
            / EXPERIMENT_IDS[key]
            / "reports"
            / "training_dynamics"
        )
        result.extend(
            [
                output_dir / "README.md",
                output_dir / "total_loss_vs_optimizer_update.png",
                output_dir / "training_dynamics_summary.json",
                output_dir / "training_dynamics_summary.csv",
            ]
        )
        result.append(
            output_dir
            / (
                "paired_loss_comparisons.png"
                if key == "011"
                else "objective_components_vs_optimizer_update.png"
            )
        )
    return result


def render(
    *,
    exp_root: Path,
    output_root: Path,
    selected: Sequence[str],
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    histories = load_histories(exp_root, selected)
    input_files = collect_input_files(histories)
    digest = input_digest(selected, histories)
    validation = {
        key: [
            {
                "arm": item.arm,
                "profile": item.profile,
                "run_group": item.run_group,
                "last_update": item.last_update,
                "metric_files": len(item.source_files),
            }
            for item in histories[key]
        ]
        for key in selected
    }
    if "006_reference" in histories:
        validation["006_reference"] = [
            {
                "arm": item.arm,
                "profile": item.profile,
                "run_group": item.run_group,
                "last_update": item.last_update,
                "metric_files": len(item.source_files),
            }
            for item in histories["006_reference"]
        ]
    status = {
        "status": "dry_run" if dry_run else "rendered",
        "selected_experiments": list(selected),
        "input_hash": digest,
        "validation": validation,
    }
    if dry_run:
        return status

    manifest_path = output_root / "manifest.json"
    expected = expected_outputs(exp_root, output_root, selected)
    if not force and manifest_path.is_file() and all(path.is_file() for path in expected):
        existing = json.loads(manifest_path.read_text())
        if existing.get("input_hash") == digest:
            status["status"] = "already_current"
            status["outputs"] = [str(path) for path in expected]
            return status

    all_rows: list[dict[str, Any]] = []
    figures: dict[str, list[Path]] = {}
    if "008" in selected:
        rows, paths = render_exp008(exp_root, histories["008"])
        all_rows.extend(rows)
        figures["008"] = paths
    if "009" in selected:
        rows, paths = render_exp009(exp_root, histories["009"])
        all_rows.extend(rows)
        figures["009"] = paths
    if "011" in selected:
        rows, paths = render_exp011(
            exp_root,
            histories["011"],
            histories["006_reference"][0],
        )
        all_rows.extend(rows)
        figures["011"] = paths
    central = render_central_gallery(
        exp_root=exp_root,
        output_root=output_root,
        selected=selected,
        histories=histories,
        all_rows=all_rows,
        per_experiment_figures=figures,
        input_hash=digest,
        input_files=input_files,
    )
    status["outputs"] = [
        *[str(path) for paths in figures.values() for path in paths],
        *[str(path) for path in central],
    ]
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENT_IDS),
        default=sorted(EXPERIMENT_IDS),
    )
    parser.add_argument("--exp-root", type=Path, default=EXP_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXP_ROOT / "comparisons" / "training_dynamics",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = tuple(dict.fromkeys(args.experiments))
    status = render(
        exp_root=args.exp_root,
        output_root=args.output_root,
        selected=selected,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
