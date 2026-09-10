"""CPU-only contracts, deterministic sampling and metrics for Exp027."""
from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EXPERIMENT = "027_voxtell_2a_residual_refinement"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/experiments" / f"{EXPERIMENT}.json"
ARMS = ("run1_train", "run2_train_val_a", "run3_val_a", "run4_val_all")
EXPOSURE = {
    ARMS[0]: {"A": "held_out_from_refiner", "B": "held_out_from_refiner", "full": "held_out_from_refiner"},
    ARMS[1]: {"A": "training", "B": "held_out_from_refiner", "full": "mixed"},
    ARMS[2]: {"A": "training", "B": "held_out_from_refiner", "full": "mixed"},
    ARMS[3]: {"A": "training", "B": "training", "full": "training"},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path, rows):
    if not rows:
        atomic_text(path, "")
        return
    stream = io.StringIO()
    fields = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def append_update(path, row):
    """One flushed record per actual optimizer update; one trainer owns the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_updates(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break  # A killed writer may leave an incomplete final record.
            row = json.loads(line)
            if row["update"] != len(rows) + 1:
                raise ValueError(f"Non-contiguous training history: {path}")
            rows.append(row)
    return rows


def reconcile_updates(path, history):
    """Checkpoint state owns recovery; archive observations beyond its cursor."""
    path = Path(path)
    value = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in history)
    if path.exists():
        previous = path.read_text()
        if previous == value:
            return
        if not previous.startswith(value):
            raise ValueError("Journal disagrees with checkpoint history")
        atomic_text(path.with_name(f"recovered_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.jsonl"), previous)
    atomic_text(path, value)


def materialize_schedule(path, prepared, arm, updates, seed):
    value = {"events": schedule(prepared, arm, updates, seed)}
    value["sha256"] = digest(value["events"])
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable sampling schedule changed: {path}")
    else:
        atomic_json(path, value)
    return value["events"]


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"Another process owns {path}") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def patient_id(name):
    match = re.fullmatch(r"((?:train|valid|test)_\d+)_[^_]+_\d+\.nii\.gz", name)
    if not match:
        raise ValueError(f"Unrecognized CT-RATE patient name: {name}")
    return match.group(1)


def finding_records(rows, split):
    result = []
    for row in rows:
        keys = sorted(row["findings"], key=int)
        for channel, key in enumerate(keys):
            if row["categories"].get(key) != "2a":
                continue
            result.append({"key": f"{row['name']}::{key}", "name": row["name"],
                           "finding_id": key, "channel": channel, "prompt": row["findings"][key],
                           "patient": patient_id(row["name"]), "split": split,
                           "voxels": int(row["pixels"][key]), "entities": int(row["entity_counts"][key])})
    return sorted(result, key=lambda r: (r["name"], int(r["finding_id"])))


def make_split(rows, seed=20260909, candidates=5000):
    names = [r["name"] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate validation case")
    patients = sorted({patient_id(n) for n in names})
    if len(patients) < 2:
        raise ValueError("Need at least two validation patients")
    records = finding_records(rows, "val")
    if not records:
        raise ValueError("Validation has no 2a findings")
    cuts = np.quantile([r["voxels"] for r in records], [.25, .5, .75])
    features = np.zeros((len(patients), 9), dtype=np.int64)
    lookup = {p: i for i, p in enumerate(patients)}
    for name in names:
        features[lookup[patient_id(name)], 1] += 1
    for r in records:
        i = lookup[r["patient"]]
        features[i, 0] += 1
        features[i, 2 + int(np.searchsorted(cuts, r["voxels"], side="right"))] += 1
        features[i, 6 + (0 if r["entities"] == 1 else 1 if r["entities"] <= 3 else 2)] += 1
    total = features.sum(0)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(candidates):
        indices = sorted(rng.permutation(len(patients))[:len(patients) // 2].tolist())
        delta = np.abs(2 * features[indices].sum(0) - total)
        score = (int(delta[0]), int(delta[1]), int(delta[2:].sum()), tuple(indices))
        if best is None or score < best[0]:
            best = (score, indices)
    if best is None:
        raise ValueError("Candidate count must be positive")
    a = {patients[i] for i in best[1]}
    halves = {n: "A" if patient_id(n) in a else "B" for n in sorted(names)}
    for r in records:
        r["half"] = halves[r["name"]]
    counts = {}
    for half in ("A", "B"):
        selected = [r for r in records if r["half"] == half]
        selected_names = [n for n, h in halves.items() if h == half]
        counts[half] = {"all_cases": len(selected_names), "all_patients": len({patient_id(n) for n in selected_names}),
                        "findings_2a": len(selected), "cases_2a": len({r["name"] for r in selected}),
                        "patients_2a": len({r["patient"] for r in selected})}
    if any(counts[h]["findings_2a"] == 0 for h in counts):
        raise ValueError("A validation half has no 2a findings")
    return {"seed": seed, "candidates": candidates, "score": list(best[0][:3]),
            "volume_quartiles_voxels": cuts.tolist(), "halves": halves, "counts": counts, "findings": records}


def schedule(prepared, arm, updates, seed):
    if arm not in ARMS or updates < 1:
        raise ValueError("Invalid arm or update count")
    pools = {"train": prepared["train"], "A": [r for r in prepared["val"] if r["half"] == "A"],
             "val": prepared["val"]}
    events = []
    for block in range((updates + 99) // 100):
        rng = np.random.default_rng(np.random.SeedSequence([seed, ARMS.index(arm), block]))
        sources = (["train"] * 100 if arm == ARMS[0] else ["train"] * 50 + ["A"] * 50
                   if arm == ARMS[1] else ["A"] * 100 if arm == ARMS[2] else ["val"] * 100)
        modes = ["gt"] * 50 + ["prediction"] * 25 + ["random"] * 25
        rng.shuffle(sources)
        rng.shuffle(modes)
        for source, mode in zip(sources, modes):
            pool = pools[source]
            if not pool:
                raise ValueError(f"Empty source pool: {source}")
            record = pool[int(rng.integers(len(pool)))]
            events.append({"update": len(events) + 1, "key": record["key"], "source": source,
                           "mode": mode, "patch_seed": int(rng.integers(2**32))})
    return events[:updates]


def full_schedule(config):
    train = config["training"]
    total, milestones = train["total_updates"], train["evaluation_updates"]
    if not isinstance(total, int) or isinstance(total, bool) or total < 1 or not milestones:
        raise ValueError("Full-run duration and evaluation_updates are unset; await the user's timing review")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 or x > total for x in milestones):
        raise ValueError("Invalid evaluation_updates")
    if milestones != sorted(set(milestones)):
        raise ValueError("evaluation_updates must be increasing and unique")
    return total, sorted(set(milestones + [total]))


def gpu_gate(allowed):
    if not allowed:
        raise PermissionError("GPU work is disabled. Explicit user approval and --allow-gpu are required")


def code_fingerprint():
    paths = sorted(Path(__file__).parent.glob("exp027_*.py"))
    paths += [Path(__file__).parent / "run_027_residual_refinement.py"]
    paths += [Path(__file__).parent / name for name in (
        "run_027_host.sh", "run_voxtell_val_inference.py", "voxtell_preprocessed_cache.py",
        "train_text_conditioned_voxtell.py")]
    return {p.name: sha256(p) for p in paths if p.is_file()}


def context(config, prepared):
    return {"config": config, "prepared_sha256": digest(prepared), "code": code_fingerprint()}


def segmentation_counts(pred, target, total_gt=None):
    p, y = np.asarray(pred, dtype=bool), np.asarray(target, dtype=bool)
    if p.shape != y.shape:
        raise ValueError("Metric shape mismatch")
    tp = int(np.count_nonzero(p & y))
    npred = int(np.count_nonzero(p))
    ngt = int(np.count_nonzero(y)) if total_gt is None else int(total_gt)
    if ngt < int(np.count_nonzero(y)):
        raise ValueError("Total GT cannot be smaller than cropped GT")
    return {"tp": tp, "fp": npred - tp, "fn": ngt - tp,
            "dice": (2 * tp + 1e-6) / (npred + ngt + 1e-6),
            "precision": tp / npred if npred else (1.0 if ngt == 0 else 0.0),
            "recall": tp / ngt if ngt else 1.0}


def finding_metrics(base, final, target, residual, total_gt=None, mask_transform=None):
    b, f, y = base >= 0, final >= 0, target > 0
    if mask_transform is not None:
        b, f, y = (mask_transform(mask) for mask in (b, f, y))
    before, after = segmentation_counts(b, y, total_gt), segmentation_counts(f, y, total_gt)
    changed = b != f
    union = b | f | y
    outside_gt = 0 if total_gt is None else total_gt - int(np.count_nonzero(y))
    return {**after, "base_dice": before["dice"], "delta_dice": after["dice"] - before["dice"],
            "hit": int(after["dice"] >= .1), "base_hit": int(before["dice"] >= .1),
            "fp_removed": int(np.count_nonzero(b & ~f & ~y)), "fn_recovered": int(np.count_nonzero(~b & f & y)),
            "tp_removed": int(np.count_nonzero(b & ~f & y)), "tn_added": int(np.count_nonzero(~b & f & ~y)),
            "edited_fraction": float(changed.mean()),
            "edited_fraction_union": float(np.count_nonzero(changed) / max(1, np.count_nonzero(union) + outside_gt)),
            "metric_geometry": "original_CT" if mask_transform else "input_array",
            "residual_statistics_domain": "native_preprocessed_extent",
            "residual_mean_abs": float(np.abs(residual).mean(dtype=np.float64)),
            "residual_max_abs": float(np.abs(residual).max()),
            "residual_p95_abs": float(np.percentile(np.abs(residual), 95))}


def aggregate(rows):
    result = {}
    for half in ("A", "B", "full"):
        items = [r for r in rows if half == "full" or r["half"] == half]
        n = len(items)
        if not n:
            result[half] = {"findings": 0, "dice": None, "hit_rate": None, "base_dice": None, "base_hit_rate": None}
            continue
        out = {"findings": n, "cases": len({r["name"] for r in items})}
        for key in ("dice", "base_dice", "precision", "recall", "delta_dice", "residual_mean_abs", "edited_fraction", "edited_fraction_union"):
            if all(key in r for r in items):
                out[key] = float(math.fsum(r[key] for r in items) / n)
        for key in ("hit", "base_hit", "tp", "fp", "fn", "fp_removed", "fn_recovered", "tp_removed", "tn_added"):
            if all(key in r for r in items):
                out[key + "s" if key in ("hit", "base_hit") else key] = sum(r[key] for r in items)
        out["hit_rate"] = sum(r["hit"] for r in items) / n
        out["base_hit_rate"] = sum(r["base_hit"] for r in items) / n
        if all("delta_dice" in r for r in items):
            out.update(improved=sum(r["delta_dice"] > 1e-12 for r in items),
                       worsened=sum(r["delta_dice"] < -1e-12 for r in items),
                       unchanged=sum(abs(r["delta_dice"]) <= 1e-12 for r in items))
        result[half] = out
    return result
