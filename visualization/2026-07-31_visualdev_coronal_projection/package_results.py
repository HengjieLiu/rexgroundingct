#!/usr/bin/env python3
"""Package the val200 category gallery as a browsable, Git-trackable result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SOURCE = Path(
    "/data/hengjie/datasets/rexgroundingct/visualizations/"
    "2026-07-31_visualdev_coronal_projection/full_val200_mean_by_category"
)
DEFAULT_DESTINATION = SCRIPT_DIR / "results" / "full_val200_mean_by_category"
DEFAULT_SPLIT_AUDIT = (
    REPO_ROOT
    / "side_experiments"
    / "sideexp001_validation_probe_design"
    / "outputs"
    / "split_category_audit.csv"
)
DEFAULT_ANALYSIS_SUMMARY = DEFAULT_SPLIT_AUDIT.with_name("analysis_summary.json")

HIT_THRESHOLD = 0.1
SOURCE_ROOT_FILES = (
    "category_case_index.csv",
    "category_case_index.md",
    "run_manifest.json",
    "val200_finding_metrics.csv",
    "val200_finding_metrics.md",
)
DERIVED_ROOT_FILES = (
    "four_model_overall_metrics.csv",
    "four_model_category_metrics.csv",
)
CATEGORY_LABELS = {
    "1a": "Bronchial wall thickening",
    "1b": "Bronchiectasis",
    "1c": "Emphysema (including Centrilobular, Paraseptal, Bullous)",
    "1d": "Septal thickening (including Interlobular, Reticulation)",
    "1e": "Micronodules (including Centrilobular, Tree-in-bud, Perilymphatic)",
    "1f": "Other",
    "2a": "Linear (including subsegmental atelectasis, scarring, fibrosis)",
    "2b": "Atelectasis, consolidation",
    "2c": "Groundglass opacity",
    "2d": "Pulmonary nodules/masses",
    "2e": "Pleural effusion or thickening",
    "2f": "Honeycombing",
    "2g": "Pneumothorax",
    "2h": "Other",
}
CATEGORY_CODES = tuple(CATEGORY_LABELS)
CATEGORY_DIRECTORIES = tuple("2f_empty" if code == "2f" else code for code in CATEGORY_CODES)
CATEGORY_FIGURE_COUNTS = {
    "1a": 3,
    "1b": 9,
    "1c": 15,
    "1d": 6,
    "1e": 11,
    "1f": 4,
    "2a": 63,
    "2b": 40,
    "2c": 53,
    "2d": 119,
    "2e": 11,
    "2f": 0,
    "2g": 1,
    "2h": 7,
}
CATEGORY_FINDING_COUNTS = {
    "1a": 3,
    "1b": 11,
    "1c": 17,
    "1d": 6,
    "1e": 11,
    "1f": 4,
    "2a": 69,
    "2b": 49,
    "2c": 60,
    "2d": 132,
    "2e": 11,
    "2f": 0,
    "2g": 1,
    "2h": 7,
}
SPLIT_TOTALS = {
    "train": {"cases": 2992, "findings": 7687},
    "validation": {"cases": 200, "findings": 381},
    "test": {"cases": 300, "findings": 582},
}
MODEL_DEFINITIONS = (
    {
        "key": "public_v11",
        "display_name": "Public VoxTell v1.1",
        "description": "Original pretrained model; no project epoch",
    },
    {
        "key": "exp009_s3v1_e100",
        "display_name": "100+100 attention",
        "description": (
            "Exp009 s3v1_fixedrho_suppress_half_quarter; Exp006 e5d4 epoch 100 "
            "+ continuation epoch 100 (10,000 updates)"
        ),
    },
    {
        "key": "noddp_best",
        "display_name": "100+100 plain best no-DDP",
        "description": (
            "Exp009 baseline_cont100; Exp006 e5d4 epoch 100 + plain continuation "
            "epoch 100 (10,000 updates)"
        ),
    },
    {
        "key": "ddp_best",
        "display_name": "Best DDP",
        "description": (
            "Exp007 weights-only DDP continuation relative epoch 50 / absolute "
            "epoch 150 (5,000 continuation updates)"
        ),
    },
)
MODEL_KEYS = tuple(model["key"] for model in MODEL_DEFINITIONS)
COLOR_DEFINITION_SECTION = """## Color Definition

> [!IMPORTANT]
> Prediction overlay colors are assigned per AP projection ray after voxelwise
> 3D TP/FP/FN classification. Green wins whenever the ray contains any real
> voxelwise TP, so slight AP over/under-segmentation still shows overlap.
> Purple marks rays where FN and FP both occur at different AP depths but no
> voxel overlaps.

| Color | Meaning |
| --- | --- |
| Green | Ray contains any real voxelwise TP, `gt & pred` |
| Red | No TP; ray contains FP only |
| Blue | No TP; ray contains FN only |
| Purple | No TP; ray contains depth-disjoint FN+FP |
"""
VAL_FILENAME_RE = re.compile(r"^val(?P<index>\d{3})_.+\.png$")
MARKDOWN_LINK_RE = re.compile(r"(?:!\[[^\]]*\]|\[[^\]]+\])\(([^)]+)\)")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+\.png)\)")


@dataclass(frozen=True)
class PackageContext:
    source: Path
    manifest: dict[str, Any]
    finding_rows: list[dict[str, str]]
    category_index_rows: list[dict[str, str]]
    split_rows: dict[str, dict[str, str]]
    split_summaries: dict[str, dict[str, Any]]
    source_pngs: dict[str, tuple[Path, ...]]
    overall_rows: list[dict[str, Any]]
    category_metric_rows: list[dict[str, Any]]


def category_directory(code: str) -> str:
    if code not in CATEGORY_LABELS:
        raise ValueError(f"unknown category code: {code!r}")
    return "2f_empty" if code == "2f" else code


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected True or False, found {value!r}")


def val_index(path: Path) -> int:
    match = VAL_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"figure does not follow valNNN naming: {path}")
    return int(match.group("index"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_png(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG signature: {path}")


def batch_items(items: list[Path], batch_size: int = 10) -> list[list[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def validate_source_tree(source: Path) -> dict[str, tuple[Path, ...]]:
    if not source.is_dir():
        raise FileNotFoundError(f"source gallery does not exist: {source}")

    expected_root_names = set(SOURCE_ROOT_FILES) | set(CATEGORY_DIRECTORIES)
    actual_root_names = {path.name for path in source.iterdir()}
    if actual_root_names != expected_root_names:
        missing = sorted(expected_root_names - actual_root_names)
        unexpected = sorted(actual_root_names - expected_root_names)
        raise ValueError(f"source root drift: missing={missing}, unexpected={unexpected}")

    p75_paths = [path for path in source.rglob("*") if "p75" in path.name.lower()]
    if p75_paths:
        raise ValueError(f"P75 output is not allowed: {p75_paths[0]}")

    source_pngs: dict[str, tuple[Path, ...]] = {}
    for code in CATEGORY_CODES:
        directory = source / category_directory(code)
        unexpected = sorted(path.name for path in directory.iterdir() if path.suffix.lower() != ".png")
        if unexpected:
            raise ValueError(f"unexpected non-PNG files in {directory}: {unexpected}")
        pngs = sorted(directory.glob("*.png"), key=lambda path: (val_index(path), path.name))
        expected_count = CATEGORY_FIGURE_COUNTS[code]
        if len(pngs) != expected_count:
            raise ValueError(f"{code}: expected {expected_count} PNGs, found {len(pngs)}")
        for path in pngs:
            verify_png(path)
        source_pngs[code] = tuple(pngs)

    if sum(len(paths) for paths in source_pngs.values()) != 342:
        raise ValueError("source gallery must contain exactly 342 PNGs")
    return source_pngs


def summarize_metrics(
    finding_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_category = {code: [row for row in finding_rows if row["category"] == code] for code in CATEGORY_CODES}
    overall_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []

    for model_order, model in enumerate(MODEL_DEFINITIONS, start=1):
        key = model["key"]
        dice_values = [float(row[f"{key}_dice"]) for row in finding_rows]
        hit_values = [parse_bool(row[f"{key}_hit"]) for row in finding_rows]
        overall_rows.append(
            {
                "model_order": model_order,
                "model_key": key,
                "method": model["display_name"],
                "training_description": model["description"],
                "finding_count": len(finding_rows),
                "mean_dice": sum(dice_values) / len(dice_values),
                "hit_count": sum(hit_values),
                "hit_rate": sum(hit_values) / len(hit_values),
            }
        )

        for category_order, code in enumerate(CATEGORY_CODES, start=1):
            rows = by_category[code]
            dice_values = [float(row[f"{key}_dice"]) for row in rows]
            hit_values = [parse_bool(row[f"{key}_hit"]) for row in rows]
            category_rows.append(
                {
                    "category_order": category_order,
                    "category": code,
                    "category_label": CATEGORY_LABELS[code],
                    "finding_count": len(rows),
                    "model_order": model_order,
                    "model_key": key,
                    "method": model["display_name"],
                    "mean_dice": sum(dice_values) / len(dice_values) if rows else None,
                    "hit_count": sum(hit_values),
                    "hit_rate": sum(hit_values) / len(hit_values) if rows else None,
                }
            )
    category_rows.sort(key=lambda row: (row["category_order"], row["model_order"]))
    return overall_rows, category_rows


def validate_metrics(
    finding_rows: list[dict[str, str]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(finding_rows) != 381:
        raise ValueError(f"expected 381 finding rows, found {len(finding_rows)}")
    if {row["category"] for row in finding_rows} != set(CATEGORY_CODES) - {"2f"}:
        raise ValueError("finding metrics contain an unexpected category set")

    for row in finding_rows:
        for key in MODEL_KEYS:
            dice = float(row[f"{key}_dice"])
            hit = parse_bool(row[f"{key}_hit"])
            if hit != (dice >= HIT_THRESHOLD):
                raise ValueError(
                    f"hit mismatch for {row['case_name']} finding {row['finding_id']} model {key}"
                )

    actual_finding_counts = {
        code: sum(row["category"] == code for row in finding_rows) for code in CATEGORY_CODES
    }
    if actual_finding_counts != CATEGORY_FINDING_COUNTS:
        raise ValueError(
            f"category finding-count drift: expected={CATEGORY_FINDING_COUNTS}, "
            f"found={actual_finding_counts}"
        )

    overall_rows, category_rows = summarize_metrics(finding_rows)
    manifest_models = manifest.get("models", [])
    if [model.get("key") for model in manifest_models] != list(MODEL_KEYS):
        raise ValueError("manifest model order does not match the four-model comparison")
    for summary, manifest_model in zip(overall_rows, manifest_models, strict=True):
        if abs(summary["mean_dice"] - float(manifest_model["overall_val200_dice"])) > 1e-6:
            raise ValueError(f"overall Dice mismatch for {summary['model_key']}")
        if abs(summary["hit_rate"] - float(manifest_model["overall_val200_hit_rate"])) > 1e-12:
            raise ValueError(f"overall hit-rate mismatch for {summary['model_key']}")
    return overall_rows, category_rows


def validate_category_index(
    rows: list[dict[str, str]],
    source_pngs: dict[str, tuple[Path, ...]],
) -> None:
    if len(rows) != 342:
        raise ValueError(f"expected 342 category-index rows, found {len(rows)}")
    indexed_figures: set[tuple[str, str]] = set()
    for row in rows:
        code = row["category"]
        if code not in CATEGORY_CODES:
            raise ValueError(f"category index contains unknown code: {code!r}")
        if int(row["finding_count"]) > 3:
            raise ValueError(f"category figure exceeds three findings: {row}")
        figure_name = Path(row["figure_path"]).name
        if int(row["reshuffled_val_index"]) != val_index(Path(figure_name)):
            raise ValueError(f"category-index val index does not match filename: {row}")
        indexed_figures.add((code, figure_name))

    actual_figures = {
        (code, path.name) for code, paths in source_pngs.items() for path in paths
    }
    if indexed_figures != actual_figures:
        raise ValueError("category index figure paths do not match the source PNGs")
    indexed_findings = {
        code: sum(int(row["finding_count"]) for row in rows if row["category"] == code)
        for code in CATEGORY_CODES
    }
    if indexed_findings != CATEGORY_FINDING_COUNTS:
        raise ValueError("category-index finding counts do not match val200")


def validate_manifest(
    manifest: dict[str, Any],
    source_pngs: dict[str, tuple[Path, ...]],
) -> None:
    required_values = {
        "mode": "val200-by-category",
        "projection_methods": ["mean"],
        "case_count": 200,
        "finding_count": 381,
        "figure_count": 342,
        "category_order": list(CATEGORY_DIRECTORIES),
        "empty_category_directory": "2f_empty",
        "filename_pattern": "valNNN_<case-stem>.png",
    }
    for key, expected in required_values.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} drift: expected={expected!r}, found={manifest.get(key)!r}")

    expected_case_counts = {
        category_directory(code): count for code, count in CATEGORY_FIGURE_COUNTS.items()
    }
    expected_finding_counts = {
        category_directory(code): count for code, count in CATEGORY_FINDING_COUNTS.items()
    }
    if manifest.get("category_case_counts") != expected_case_counts:
        raise ValueError("manifest category case counts do not match the expected gallery")
    if manifest.get("category_finding_counts") != expected_finding_counts:
        raise ValueError("manifest category finding counts do not match val200")

    manifest_figures = {
        (Path(path).parent.name, Path(path).name) for path in manifest.get("figure_paths", [])
    }
    actual_figures = {
        (category_directory(code), path.name)
        for code, paths in source_pngs.items()
        for path in paths
    }
    if manifest_figures != actual_figures:
        raise ValueError("manifest figure paths do not match the source gallery")


def validate_split_sources(
    split_audit: Path,
    analysis_summary: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    rows = read_csv_rows(split_audit)
    split_rows = {row["category"]: row for row in rows}
    if tuple(split_rows) != CATEGORY_CODES:
        raise ValueError("split audit category order does not match the official order")

    analysis = read_json(analysis_summary)
    split_summaries = analysis["metadata"]["split_summaries"]
    for split in ("train", "validation", "test"):
        summary = split_summaries[split]
        if int(summary["cases"]) != SPLIT_TOTALS[split]["cases"]:
            raise ValueError(f"{split} case total drift")
        if int(summary["findings"]) != SPLIT_TOTALS[split]["findings"]:
            raise ValueError(f"{split} finding total drift")

    column_prefixes = {"train": "train", "validation": "validation", "test": "test"}
    for code, row in split_rows.items():
        for split, prefix in column_prefixes.items():
            count = int(row[f"{prefix}_count"])
            ratio = float(row[f"{prefix}_ratio"])
            if count != int(split_summaries[split]["counts"][code]):
                raise ValueError(f"{split} count mismatch for {code}")
            expected_ratio = count / SPLIT_TOTALS[split]["findings"]
            if abs(ratio - expected_ratio) > 1e-12:
                raise ValueError(f"{split} ratio mismatch for {code}")
    return split_rows, split_summaries


def load_context(source: Path, split_audit: Path, analysis_summary: Path) -> PackageContext:
    source_pngs = validate_source_tree(source)
    manifest = read_json(source / "run_manifest.json")
    validate_manifest(manifest, source_pngs)
    finding_rows = read_csv_rows(source / "val200_finding_metrics.csv")
    overall_rows, category_metric_rows = validate_metrics(finding_rows, manifest)
    category_index_rows = read_csv_rows(source / "category_case_index.csv")
    validate_category_index(category_index_rows, source_pngs)
    split_rows, split_summaries = validate_split_sources(split_audit, analysis_summary)
    return PackageContext(
        source=source,
        manifest=manifest,
        finding_rows=finding_rows,
        category_index_rows=category_index_rows,
        split_rows=split_rows,
        split_summaries=split_summaries,
        source_pngs=source_pngs,
        overall_rows=overall_rows,
        category_metric_rows=category_metric_rows,
    )


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def metric_cell(row: dict[str, Any]) -> str:
    if row["mean_dice"] is None:
        return "—"
    return (
        f"{row['mean_dice']:.6f} · {row['hit_count']}/{row['finding_count']} "
        f"({percentage(row['hit_rate'])})"
    )


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    header_values = list(headers)
    lines = [
        "| " + " | ".join(header_values) + " |",
        "| " + " | ".join("---" for _ in header_values) + " |",
    ]
    for row in rows:
        values = [str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def category_metric_lookup(context: PackageContext, code: str) -> dict[str, dict[str, Any]]:
    return {
        row["model_key"]: row
        for row in context.category_metric_rows
        if row["category"] == code
    }


def build_overview(context: PackageContext) -> str:
    split_table = markdown_table(
        ("Split", "Cases", "Findings"),
        (
            ("Train", SPLIT_TOTALS["train"]["cases"], SPLIT_TOTALS["train"]["findings"]),
            (
                "Validation",
                SPLIT_TOTALS["validation"]["cases"],
                SPLIT_TOTALS["validation"]["findings"],
            ),
            ("Test", SPLIT_TOTALS["test"]["cases"], SPLIT_TOTALS["test"]["findings"]),
        ),
    )
    distribution_rows = []
    for code in CATEGORY_CODES:
        row = context.split_rows[code]
        distribution_rows.append(
            (
                code,
                CATEGORY_LABELS[code],
                f"{row['train_count']} ({percentage(float(row['train_ratio']))})",
                f"{row['validation_count']} ({percentage(float(row['validation_ratio']))})",
                f"{row['test_count']} ({percentage(float(row['test_ratio']))})",
            )
        )
    distribution_table = markdown_table(
        ("Category", "Finding type", "Train n (%)", "Validation n (%)", "Test n (%)"),
        distribution_rows,
    )

    overall_table = markdown_table(
        ("Order", "Method", "Training/checkpoint description", "Mean Dice", "Hits", "Hit rate"),
        (
            (
                row["model_order"],
                row["method"],
                row["training_description"],
                f"{row['mean_dice']:.6f}",
                f"{row['hit_count']}/{row['finding_count']}",
                percentage(row["hit_rate"]),
            )
            for row in context.overall_rows
        ),
    )

    comparison_rows = []
    gallery_rows = []
    for code in CATEGORY_CODES:
        lookup = category_metric_lookup(context, code)
        comparison_rows.append(
            (
                code,
                CATEGORY_FINDING_COUNTS[code],
                *(metric_cell(lookup[key]) for key in MODEL_KEYS),
            )
        )
        directory = category_directory(code)
        gallery_rows.append(
            (
                f"[{code} — {CATEGORY_LABELS[code]}]"
                f"(full_val200_mean_by_category/{directory}/README.md)",
                CATEGORY_FINDING_COUNTS[code],
                CATEGORY_FIGURE_COUNTS[code],
            )
        )
    comparison_table = markdown_table(
        (
            "Category",
            "Findings",
            "Public VoxTell v1.1",
            "100+100 attention",
            "100+100 plain best no-DDP",
            "Best DDP",
        ),
        comparison_rows,
    )
    gallery_table = markdown_table(("Category gallery", "Findings", "Figures"), gallery_rows)

    return f"""# Full Val200 Mean Coronal Projection Results

This is the committed, browsable package for the category-organized full-val200
coronal projection comparison. It contains 342 figures covering all 381
validation findings. CT panels use the mean lung-windowed anterior-posterior
projection; GT and prediction panels use binary maximum-intensity projections.
Displayed Dice values remain three-dimensional.

Radiology orientation is fixed throughout: superior is up and patient right is
on screen left. Each case/category figure contains only that category's
findings and has at most three finding rows.

{COLOR_DEFINITION_SECTION}

## Dataset category distribution

Counts and percentages are finding-level within each split. Case totals are
included separately because a scan can have multiple findings.

{split_table}

{distribution_table}

Sources: [split category audit](../../../side_experiments/sideexp001_validation_probe_design/outputs/split_category_audit.csv)
and [analysis summary](../../../side_experiments/sideexp001_validation_probe_design/outputs/analysis_summary.json).
The released test metadata includes finding categories, but test masks are
hidden; therefore the tables below report measured model performance on the
200-case validation set only.

## Four-method val200 comparison

A hit is a finding with 3D Dice greater than or equal to `0.1`.

{overall_table}

Machine-readable table: [four_model_overall_metrics.csv](full_val200_mean_by_category/four_model_overall_metrics.csv).

## Per-category val200 comparison

Each model cell is `mean Dice · hits/support (hit rate)`. Category 2f has no
validation findings, so measured performance is not available.

{comparison_table}

Machine-readable table: [four_model_category_metrics.csv](full_val200_mean_by_category/four_model_category_metrics.csv).

## Category galleries

{gallery_table}

## Canonical data and provenance

- [Finding-level metrics](full_val200_mean_by_category/val200_finding_metrics.csv)
- [Case/category figure index](full_val200_mean_by_category/category_case_index.csv)
- [Renderer run manifest](full_val200_mean_by_category/run_manifest.json)
- [Visualization development README](../README.md)

The source gallery remains at:

- Host: `/data/hengjie/datasets/rexgroundingct/visualizations/2026-07-31_visualdev_coronal_projection/full_val200_mean_by_category`
- Docker: `/database/datasets/rexgroundingct/visualizations/2026-07-31_visualdev_coronal_projection/full_val200_mean_by_category`

Rebuild and verify this committed package from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python \\
  visualization/2026-07-31_visualdev_coronal_projection/package_results.py
```
"""


def build_category_readme(context: PackageContext, code: str) -> str:
    directory = category_directory(code)
    split_row = context.split_rows[code]
    pngs = list(context.source_pngs[code])
    split_table = markdown_table(
        ("Split", "Finding count", "Within-split ratio"),
        (
            ("Train", split_row["train_count"], percentage(float(split_row["train_ratio"]))),
            (
                "Validation",
                split_row["validation_count"],
                percentage(float(split_row["validation_ratio"])),
            ),
            ("Test", split_row["test_count"], percentage(float(split_row["test_ratio"]))),
        ),
    )
    lookup = category_metric_lookup(context, code)
    metric_table = markdown_table(
        ("Order", "Method", "Mean Dice", "Hits", "Hit rate"),
        (
            (
                lookup[key]["model_order"],
                lookup[key]["method"],
                "—" if lookup[key]["mean_dice"] is None else f"{lookup[key]['mean_dice']:.6f}",
                (
                    "—"
                    if lookup[key]["mean_dice"] is None
                    else f"{lookup[key]['hit_count']}/{lookup[key]['finding_count']}"
                ),
                "—" if lookup[key]["hit_rate"] is None else percentage(lookup[key]["hit_rate"]),
            )
            for key in MODEL_KEYS
        ),
    )
    lines = [
        f"# {code} — {CATEGORY_LABELS[code]}",
        "",
        "[← Results overview](../../README.md) · "
        "[Case/category index](../category_case_index.csv) · "
        "[Finding metrics](../val200_finding_metrics.csv)",
        "",
        "## Split distribution",
        "",
        split_table,
        "",
        "## Val200 model summary",
        "",
        "A hit is a finding with 3D Dice greater than or equal to `0.1`.",
        "",
        metric_table,
        "",
        COLOR_DEFINITION_SECTION,
        "",
        f"## Figures ({len(pngs)})",
        "",
    ]

    if not pngs:
        lines.extend(
            [
                "There are no category-2f findings in validation or test, so this",
                "gallery intentionally contains zero PNGs. The training split has 16",
                "category-2f findings.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "Figures are sorted by fixed validation index. Each figure shows only",
            f"category {code} findings and uses radiology coronal orientation.",
            "",
        ]
    )
    item_number = 0
    for batch in batch_items(pngs, batch_size=10):
        first = item_number + 1
        last = item_number + len(batch)
        lines.extend(["<details>", f"<summary>Figures {first}–{last} of {len(pngs)}</summary>", ""])
        for png in batch:
            item_number += 1
            lines.extend(
                [
                    f"### {item_number}. {png.stem}",
                    "",
                    f"![{code} {png.stem}]({png.name})",
                    "",
                    f"[Open full-resolution PNG]({png.name})",
                    "",
                ]
            )
        lines.extend(["</details>", ""])
    return "\n".join(lines)


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.12f}"
    if value is None:
        return ""
    return value


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: csv_value(row[key]) for key in fieldnames} for row in rows)


def expected_destination_files(context: PackageContext, destination: Path) -> set[Path]:
    expected = {destination / name for name in SOURCE_ROOT_FILES + DERIVED_ROOT_FILES}
    for code, pngs in context.source_pngs.items():
        directory = destination / category_directory(code)
        expected.add(directory / "README.md")
        expected.update(directory / png.name for png in pngs)
    return expected


def reject_destination_drift(context: PackageContext, destination: Path) -> None:
    if not destination.exists():
        return
    expected = expected_destination_files(context, destination)
    actual = {path for path in destination.rglob("*") if path.is_file()}
    unexpected = sorted(str(path.relative_to(destination)) for path in actual - expected)
    if unexpected:
        raise ValueError(f"destination contains unexpected files: {unexpected}")


def copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise ValueError(f"copy hash mismatch: {source} -> {destination}")


def package_results(context: PackageContext, destination: Path) -> None:
    reject_destination_drift(context, destination)
    destination.mkdir(parents=True, exist_ok=True)
    for code in CATEGORY_CODES:
        (destination / category_directory(code)).mkdir(parents=True, exist_ok=True)

    for name in SOURCE_ROOT_FILES:
        copy_verified(context.source / name, destination / name)
    for code, pngs in context.source_pngs.items():
        for png in pngs:
            copy_verified(png, destination / category_directory(code) / png.name)

    write_csv(
        destination / "four_model_overall_metrics.csv",
        (
            "model_order",
            "model_key",
            "method",
            "training_description",
            "finding_count",
            "mean_dice",
            "hit_count",
            "hit_rate",
        ),
        context.overall_rows,
    )
    write_csv(
        destination / "four_model_category_metrics.csv",
        (
            "category_order",
            "category",
            "category_label",
            "finding_count",
            "model_order",
            "model_key",
            "method",
            "mean_dice",
            "hit_count",
            "hit_rate",
        ),
        context.category_metric_rows,
    )
    for code in CATEGORY_CODES:
        readme = destination / category_directory(code) / "README.md"
        readme.write_text(build_category_readme(context, code), encoding="utf-8")
    results_root = destination.parent
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "README.md").write_text(build_overview(context), encoding="utf-8")


def verify_markdown_links(readme: Path) -> None:
    for target in MARKDOWN_LINK_RE.findall(readme.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        resolved = (readme.parent / target).resolve()
        if not resolved.exists():
            raise ValueError(f"broken Markdown link in {readme}: {target}")


def verify_package(context: PackageContext, destination: Path) -> None:
    expected = expected_destination_files(context, destination)
    actual = {path for path in destination.rglob("*") if path.is_file()}
    if actual != expected:
        missing = sorted(str(path.relative_to(destination)) for path in expected - actual)
        unexpected = sorted(str(path.relative_to(destination)) for path in actual - expected)
        raise ValueError(f"destination drift: missing={missing}, unexpected={unexpected}")

    for name in SOURCE_ROOT_FILES:
        if sha256_file(context.source / name) != sha256_file(destination / name):
            raise ValueError(f"canonical file changed during packaging: {name}")
    for code, pngs in context.source_pngs.items():
        for source_png in pngs:
            destination_png = destination / category_directory(code) / source_png.name
            if sha256_file(source_png) != sha256_file(destination_png):
                raise ValueError(f"PNG changed during packaging: {source_png.name}")
            verify_png(destination_png)

    results_readme = destination.parent / "README.md"
    if results_readme.read_text(encoding="utf-8") != build_overview(context):
        raise ValueError("results overview is not deterministic")
    verify_markdown_links(results_readme)

    embedded_pngs: set[Path] = set()
    for code in CATEGORY_CODES:
        readme = destination / category_directory(code) / "README.md"
        if readme.read_text(encoding="utf-8") != build_category_readme(context, code):
            raise ValueError(f"category README is not deterministic: {code}")
        verify_markdown_links(readme)
        text = readme.read_text(encoding="utf-8")
        images = MARKDOWN_IMAGE_RE.findall(text)
        if len(images) != CATEGORY_FIGURE_COUNTS[code]:
            raise ValueError(f"{code}: not every PNG is embedded exactly once")
        for target in images:
            resolved = (readme.parent / target).resolve()
            if resolved in embedded_pngs:
                raise ValueError(f"PNG embedded more than once: {resolved}")
            embedded_pngs.add(resolved)
        for details in re.findall(r"<details>(.*?)</details>", text, flags=re.DOTALL):
            if len(MARKDOWN_IMAGE_RE.findall(details)) > 10:
                raise ValueError(f"{code}: collapsible batch exceeds ten figures")

    expected_embeds = {
        (destination / category_directory(code) / png.name).resolve()
        for code, pngs in context.source_pngs.items()
        for png in pngs
    }
    if embedded_pngs != expected_embeds:
        raise ValueError("category pages do not embed the exact destination PNG set")
    if any("p75" in path.name.lower() for path in destination.rglob("*")):
        raise ValueError("destination contains a P75 artifact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--split-audit", type=Path, default=DEFAULT_SPLIT_AUDIT)
    parser.add_argument("--analysis-summary", type=Path, default=DEFAULT_ANALYSIS_SUMMARY)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate the source and an existing destination without writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = load_context(args.source, args.split_audit, args.analysis_summary)
    if not args.verify_only:
        package_results(context, args.destination)
    verify_package(context, args.destination)
    print(
        f"verified {len(context.finding_rows)} findings and "
        f"{sum(len(paths) for paths in context.source_pngs.values())} PNGs in {args.destination}"
    )


if __name__ == "__main__":
    main()
