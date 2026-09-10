#!/usr/bin/env python3
"""Generate lung anatomical priors for ReXGroundingCT CT volumes.

This script runs TotalSegmentator, combines lung-lobe masks, and writes
whole-lung/left/right/dilated lung priors with CT geometry preserved.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover - dependency message
    raise SystemExit("SimpleITK is required. Install with: pip install SimpleITK") from exc

try:
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover - dependency message
    raise SystemExit("scipy is required. Install with: pip install scipy") from exc


LUNG_LOBES = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]
LEFT_LOBES = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
RIGHT_LOBES = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
DILATION_MM = [5, 10, 15]
SUPPORTED_ADDITIONAL_ROIS = ["trachea", "esophagus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct_dir", type=Path, required=True, help="Directory containing CT .nii/.nii.gz files.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for lung priors.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for summary/failure CSVs. Defaults to --out_dir.",
    )
    parser.add_argument("--scan_list_csv", type=Path, default=None, help="Optional CSV listing scans to process.")
    parser.add_argument("--splits", nargs="+", default=None, help="Optional split names to keep if scan list has a split column.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default=None, help="TotalSegmentator device, usually gpu or cpu.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Pass --fast to TotalSegmentator.")
    parser.add_argument(
        "--robust-crop",
        action="store_true",
        help="Pass --robust_crop to TotalSegmentator for more reliable ROI localization.",
    )
    parser.add_argument(
        "--scan-id-column",
        default=None,
        help="Optional column in --scan_list_csv containing scan filenames/ids. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--additional-rois",
        nargs="*",
        choices=SUPPORTED_ADDITIONAL_ROIS,
        default=[],
        help="Additional same-part TotalSegmentator ROIs to save with the five lung lobes.",
    )
    parser.add_argument("--no-roi-subset", action="store_true", help="Do not pass lung lobe roi_subset to TotalSegmentator.")
    parser.add_argument(
        "--disable-full-task-fallback",
        action="store_true",
        help="Fail instead of retrying all TotalSegmentator classes if roi_subset inference fails.",
    )
    parser.add_argument(
        "--reuse-totalseg",
        action="store_true",
        help="Reuse masks already present in CASE/totalseg instead of rerunning TotalSegmentator.",
    )
    return parser.parse_args()


def case_id_from_path(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def candidate_ct_paths(
    ct_dir: Path,
    scan_list_csv: Path | None,
    scan_id_column: str | None,
    splits: list[str] | None,
) -> list[Path]:
    if scan_list_csv is None:
        return sorted([*ct_dir.glob("*.nii.gz"), *ct_dir.glob("*.nii")])

    df = pd.read_csv(scan_list_csv)
    if splits is not None and "split" in df.columns:
        df = df[df["split"].astype(str).isin(set(splits))]
    if scan_id_column is None:
        for col in ["case", "scan_id", "name", "file", "filename", "volume"]:
            if col in df.columns:
                scan_id_column = col
                break
    if scan_id_column is None or scan_id_column not in df.columns:
        raise ValueError(
            f"Could not infer scan id column from {scan_list_csv}. "
            f"Columns: {list(df.columns)}. Use --scan-id-column."
        )

    paths: list[Path] = []
    for raw in df[scan_id_column].dropna().astype(str).unique():
        p = Path(raw)
        names = [p.name]
        if not p.name.endswith((".nii.gz", ".nii")):
            names = [f"{p.name}.nii.gz", f"{p.name}.nii"]
        found = None
        for name in names:
            candidate = ct_dir / name
            if candidate.exists():
                found = candidate
                break
        if found is None:
            raise FileNotFoundError(f"Could not find CT for {raw!r} in {ct_dir}")
        paths.append(found)
    return sorted(set(paths))


def infer_device(user_device: str | None) -> str:
    if user_device:
        return user_device
    try:
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "gpu"


def run_totalseg(
    ct_path: Path,
    totalseg_dir: Path,
    device: str,
    fast: bool,
    robust_crop: bool,
    roi_subset: list[str] | None,
    allow_full_task_fallback: bool,
) -> None:
    cmd = ["TotalSegmentator", "-i", str(ct_path), "-o", str(totalseg_dir), "--device", device]
    if fast:
        cmd.append("--fast")
    if robust_crop:
        cmd.append("--robust_crop")
    if roi_subset:
        cmd += ["--roi_subset", *roi_subset]

    try:
        subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as exc:
        if roi_subset and allow_full_task_fallback:
            fallback = ["TotalSegmentator", "-i", str(ct_path), "-o", str(totalseg_dir), "--device", device]
            if fast:
                fallback.append("--fast")
            if robust_crop:
                fallback.append("--robust_crop")
            subprocess.run(fallback, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        else:
            raise RuntimeError(
                f"TotalSegmentator failed for {ct_path}\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}"
            ) from exc


def read_mask_like(mask_path: Path, reference: sitk.Image) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    img = sitk.ReadImage(str(mask_path))
    if img.GetSize() != reference.GetSize() or img.GetSpacing() != reference.GetSpacing():
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        img = resampler.Execute(img)
    return sitk.GetArrayFromImage(img) > 0


def write_mask(mask: np.ndarray, reference: sitk.Image, path: Path) -> tuple[int, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(path))
    voxel_count = int(mask.sum())
    volume_mm3 = voxel_count * float(np.prod(reference.GetSpacing()))
    return voxel_count, volume_mm3


def dilate_mm(mask: np.ndarray, spacing_xyz: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    if radius_mm <= 0:
        return mask.copy()
    if not mask.any():
        return mask.copy()
    # SimpleITK arrays are z, y, x; spacing is x, y, z.
    sampling_zyx = np.asarray((spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]), dtype=float)
    foreground = np.argwhere(mask)
    margin = np.ceil(float(radius_mm) / sampling_zyx).astype(int)
    lower = np.maximum(foreground.min(axis=0) - margin, 0)
    upper = np.minimum(foreground.max(axis=0) + margin + 1, np.asarray(mask.shape))
    crop_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    cropped = mask[crop_slices]
    distance_to_foreground = ndimage.distance_transform_edt(~cropped, sampling=sampling_zyx)
    result = mask.copy()
    result[crop_slices] = cropped | (distance_to_foreground <= float(radius_mm))
    return result


def process_case(
    ct_path: Path,
    out_dir: Path,
    device: str,
    fast: bool,
    robust_crop: bool,
    skip_existing: bool,
    use_roi_subset: bool,
    additional_rois: list[str],
    allow_full_task_fallback: bool,
    reuse_totalseg: bool,
) -> dict:
    scan_id = case_id_from_path(ct_path)
    case_dir = out_dir / scan_id
    totalseg_dir = case_dir / "totalseg"
    final_mask = case_dir / "lung_dilated_10mm.nii.gz"
    required_outputs = [
        final_mask,
        case_dir / "whole_lung.nii.gz",
        *[case_dir / f"{name}.nii.gz" for name in LUNG_LOBES],
        *[case_dir / f"{name}.nii.gz" for name in additional_rois],
    ]
    if skip_existing and all(path.is_file() for path in required_outputs):
        return {"scan_id": scan_id, "status": "skipped_existing"}

    case_dir.mkdir(parents=True, exist_ok=True)
    roi_subset = [*LUNG_LOBES, *additional_rois] if use_roi_subset else None
    if reuse_totalseg:
        if not totalseg_dir.is_dir():
            raise FileNotFoundError(f"Cannot reuse missing TotalSegmentator output: {totalseg_dir}")
    else:
        run_totalseg(
            ct_path,
            totalseg_dir,
            device=device,
            fast=fast,
            robust_crop=robust_crop,
            roi_subset=roi_subset,
            allow_full_task_fallback=allow_full_task_fallback,
        )

    reference = sitk.ReadImage(str(ct_path))
    lobe_masks = {name: read_mask_like(totalseg_dir / f"{name}.nii.gz", reference) for name in LUNG_LOBES}
    additional_masks = {
        name: read_mask_like(totalseg_dir / f"{name}.nii.gz", reference)
        for name in additional_rois
    }
    left = np.logical_or.reduce([lobe_masks[name] for name in LEFT_LOBES])
    right = np.logical_or.reduce([lobe_masks[name] for name in RIGHT_LOBES])
    whole = left | right
    if not whole.any():
        raise ValueError("Combined whole lung mask is empty")

    rows = []
    masks_to_write = (
        [(name, lobe_masks[name]) for name in LUNG_LOBES]
        + [(name, additional_masks[name]) for name in additional_rois]
        + [
            ("left_lung", left),
            ("right_lung", right),
            ("whole_lung", whole),
        ]
    )
    for label, mask in masks_to_write:
        vox, vol = write_mask(mask, reference, case_dir / f"{label}.nii.gz")
        rows.append({"scan_id": scan_id, "mask": label, "voxels": vox, "volume_mm3": vol})
        if vox == 0:
            # A lobe or auxiliary structure can legitimately be outside a
            # limited-FOV scan. The combined lung mask remains the hard check.
            print(f"{scan_id}: warning: {label} is empty", flush=True)

    for radius in DILATION_MM:
        dilated = dilate_mm(whole, reference.GetSpacing(), radius)
        vox, vol = write_mask(dilated, reference, case_dir / f"lung_dilated_{radius}mm.nii.gz")
        rows.append({"scan_id": scan_id, "mask": f"lung_dilated_{radius}mm", "voxels": vox, "volume_mm3": vol})
        print(f"{scan_id}: lung_dilated_{radius}mm voxels={vox} volume_mm3={vol:.1f}", flush=True)

    return {"scan_id": scan_id, "status": "ok", "rows": rows}


def write_failures(path: Path, failures: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scan_id", "ct_path", "error"])
        writer.writeheader()
        writer.writerows(failures)


def main() -> int:
    args = parse_args()
    if shutil.which("TotalSegmentator") is None:
        raise SystemExit("TotalSegmentator executable not found. Install with: pip install TotalSegmentator")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.report_dir or args.out_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    ct_paths = candidate_ct_paths(args.ct_dir, args.scan_list_csv, args.scan_id_column, args.splits)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < num-shards")
    total_cts = len(ct_paths)
    ct_paths = ct_paths[args.shard_index :: args.num_shards]
    device = infer_device(args.device)
    print(
        f"Processing {len(ct_paths)}/{total_cts} CT(s); "
        f"shard={args.shard_index}/{args.num_shards}; "
        f"device={device}; out_dir={args.out_dir}; "
        f"additional_rois={args.additional_rois}"
    )

    summary_rows: list[dict] = []
    failures: list[dict] = []
    workers = max(1, int(args.num_workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                process_case,
                p,
                args.out_dir,
                device,
                args.fast,
                args.robust_crop,
                args.skip_existing,
                not args.no_roi_subset,
                args.additional_rois,
                not args.disable_full_task_fallback,
                args.reuse_totalseg,
            ): p
            for p in ct_paths
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="TotalSegmentator", unit="case"):
            p = futures[fut]
            try:
                result = fut.result()
                if result.get("rows"):
                    summary_rows.extend(result["rows"])
                print(f"{result['scan_id']}: {result['status']}", flush=True)
            except Exception as exc:
                failures.append({"scan_id": case_id_from_path(p), "ct_path": str(p), "error": f"{type(exc).__name__}: {exc}"})

    pd.DataFrame(summary_rows).to_csv(report_dir / "lung_prior_summary.csv", index=False)
    write_failures(report_dir / "failed_cases.csv", failures)
    print(f"Done. ok rows={len(summary_rows)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
