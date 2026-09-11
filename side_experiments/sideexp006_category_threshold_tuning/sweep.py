#!/usr/bin/env python3
"""Source-bound, resumable raw Dice/HIT threshold sweeps; never runs inference."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"

import argparse
import csv
import fcntl
import hashlib
import json
import math
import multiprocessing
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HUB = REPO / "side_experiments/sideexp003_ensemble_method_hub"
CACHE_ROOT = Path("/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp003_ensemble_method_hub/cache/logits/by_cache_key")
RUN_ROOT = Path("/mnt/shengdata1/hengjie/side_experiments/rexgroundingct/sideexp006_category_threshold_tuning/runs/r001_a1_val200")
DATASET = REPO / "configs/evaluation/rexgroundingct_val200_seed20260723.json"
DATASET_SHA = "7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897"
SEG_ROOT = Path("/data/hengjie/datasets/rexgroundingct/segmentations")
A1_KEY = "v2_d6dcddf705766db234bf631fa0925bfd69a5e18388767448b02a31b6484fcdbe"
A1_SHA = "a499ad1c0fada9a7e4d78e352fa5510d31295e82749e00ef0a4eee6da685e4aa"
BASE_DICE = 0.34586800803778955
PERCENTAGES = tuple(range(5, 100, 5))
THRESHOLDS = np.array(PERCENTAGES, dtype=np.float64) / 100
CATEGORIES = {
    "1a": "Bronchial wall thickening", "1b": "Bronchiectasis",
    "1c": "Emphysema", "1d": "Septal thickening / reticulation",
    "1e": "Micronodules / tree-in-bud", "1f": "Other diffuse lung/airway/pleural abnormality",
    "2a": "Linear opacity, scarring, fibrosis", "2b": "Atelectasis / consolidation",
    "2c": "Ground-glass opacity", "2d": "Pulmonary nodules / masses",
    "2e": "Pleural effusion / thickening", "2f": "Honeycombing",
    "2g": "Pneumothorax", "2h": "Other focal lung/airway/pleural finding",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(array):
    """SideExp003 logical hash: C-order values, excluding the NPY header."""
    view = memoryview(np.ascontiguousarray(array)).cast('B')
    h = hashlib.sha256()
    for start in range(0, len(view), 16*1024**2):
        h.update(view[start:start+16*1024**2])
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def dice(gt, pred, intersection):
    return (2 * intersection + 1e-6) / (gt + pred + 1e-6) if gt + pred else 1.0


def validate_cases(cases, expected_cases=200, expected_findings=381):
    if len(cases) != expected_cases:
        raise ValueError("Unexpected case count")
    names = set()
    n = 0
    for c in cases:
        name = c["name"]
        if name in names or Path(name).name != name or not name.endswith(".nii.gz"):
            raise ValueError("Duplicate or invalid case name")
        names.add(name)
        keys = sorted(c["findings"], key=int)
        if keys != [str(i) for i in range(len(keys))] or set(keys) != set(c["categories"]):
            raise ValueError("Finding/category keys or numeric channel order invalid")
        if any(c["categories"][k] not in CATEGORIES for k in keys):
            raise ValueError("Unknown official category")
        n += len(keys)
    if n != expected_findings:
        raise ValueError("Unexpected finding count")


def choose_sources(recipe, registry, catalog):
    """Resolve immutable recipe members by SHA, never by mutable leaderboard rank."""
    family = registry["families"][recipe[0]]
    result = []
    for model in family["models"]:
        sha = model["checkpoint"]["sha256"]
        if recipe == "a1":
            if sha != A1_SHA:
                raise ValueError("A1 checkpoint changed")
            key = A1_KEY
        elif recipe in ("d1", "e1"):
            key = model["paired_val200"]["cache_key"]
        else:
            matches = [c for c in catalog["candidates"] if c["checkpoint"]["sha256"] == sha]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous checkpoint: {sha}")
            artifact = matches[0]["inference_artifacts"]["val200"]
            versions = [v for v in artifact["versions"] if v["cache_version_id"] == artifact["active_cache_version"] and v["status"] == "strict_passed" and v.get("cache_key")]
            if len(versions) != 1:
                raise ValueError(f"Missing strict val200 cache for {model['candidate_id']}; separate export required")
            key = versions[0]["cache_key"]
        result.append({"candidate_id": model["candidate_id"], "checkpoint_sha256": sha, "cache_key": key})
    count = {"a1": 1, "b1": 9, "d1": 4, "e1": 8}[recipe]
    if len(result) != count or len({m['checkpoint_sha256'] for m in result}) != count:
        raise ValueError("Unexpected recipe member count or duplicate checkpoint")
    if recipe in ("d1", "e1"):
        weights = family["recipe"]["weights"]
        if len(weights) != count or any(w != 1/count for w in weights):
            raise ValueError("Recipe is not the expected equal probability average")
    return result, family["recipe"]


def channel_index(record, original_index):
    ids = record.get('finding_ids', list(range(record['shape'][0])))
    if len(ids) != record['shape'][0] or len(set(ids)) != len(ids):
        raise ValueError('Invalid subset channel map')
    if 'channel_to_finding' in record and record['channel_to_finding'] != {str(i): k for i, k in enumerate(ids)}:
        raise ValueError('Subset channel map mismatch')
    if original_index not in ids:
        raise ValueError('Missing routed finding')
    return ids.index(original_index)


def needed_sources(sources, spec, recipe, case):
    if recipe != 'b1':
        return sources
    ids = {spec['categories'][category] for category in case['categories'].values()}
    return [s for s in sources if s['candidate_id'] in ids]


def validate_routing(sources, spec, cases):
    ids = [s['candidate_id'] for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate routed source')
    for case in cases:
        for k, category in case['categories'].items():
            matches = [s for s in sources if s['candidate_id'] == spec['categories'][category]]
            if len(matches) != 1 or case['name'] not in matches[0]['cases']:
                raise ValueError('Missing or duplicate routed prediction')
            channel_index(matches[0]['cases'][case['name']], int(k))


def baseline_references(contract):
    recipe = contract['recipe']
    if recipe == 'a1':
        return dict(dice=BASE_DICE, hits=295, source='Pinned SideExp003 strict cache'), dict(
            dice=0.3460234857111323, hits=296,
            source='/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched/runs/exp007_cont100_from_ddp100_20260730T062051Z/ddp_bs4/eval_epoch050_val200/eval/val_quick_global_eval.json')
    if recipe == 'd1':
        return dict(dice=0.3575041942161927, hits=296,
                    source='SideExp005 r001_d1_val200_frozen_postprocessing raw baseline'), None
    if recipe == 'b1':
        refs = []
        for c in contract['cases']:
            for k, category in c['categories'].items():
                source = next(s for s in contract['sources'] if s['candidate_id'] == contract['recipe_spec']['categories'][category])
                r = source['cases'][c['name']]
                refs.append(r['same_pass_reference']['finding_dice'][channel_index(r, int(k))])
        return dict(dice=math.fsum(refs)/len(refs), hits=sum(d >= .1 for d in refs),
                    source='Independent composition of full-cache and targeted same-pass finding references'), dict(
            dice=0.3602780325749142, hits=298,
            source='/mnt/shengdata1/hengjie/experiments/rexgroundingct/024_test_inference_anatomy_audit/reports/b_validation_metrics.json')
    return None, None


def prepare(recipe, source_manifest=None):
    import nibabel as nib
    if file_hash(DATASET) != DATASET_SHA:
        raise ValueError("Fixed dataset SHA mismatch")
    cases = read(DATASET)["test"]
    validate_cases(cases)
    if source_manifest:
        if recipe != 'b1':
            raise ValueError('Mixed source manifests are scoped to B1')
        frozen = read(source_manifest)
        if frozen['dataset_sha256'] != DATASET_SHA or frozen['status'] != 'complete':
            raise ValueError('Invalid mixed source manifest')
        sources, recipe_spec = frozen['sources'], frozen['recipe_spec']
    else:
        sources, recipe_spec = choose_sources(recipe, read(REPO / "submissions/registry.json"), read(HUB / "checkpoint_catalog.json"))
    expected_names = {c["name"] for c in cases}
    for source in sources:
        if source.get('scope') == 'routed_subset':
            export_path = Path(source['export_manifest_path'])
            if file_hash(export_path) != source['export_manifest_sha256']:
                raise ValueError('Subset source manifest changed')
            export = read(export_path)
            if export['status'] != 'complete' or export['scope'] != 'routed_subset' or export['checkpoint_sha256'] != source['checkpoint_sha256'] or export['dataset_sha256'] != DATASET_SHA:
                raise ValueError('Subset identity mismatch')
            source['cases'] = {r['name']: r for r in export['cases']}
            if len(source['cases']) != len(export['cases']):
                raise ValueError('Duplicate subset case')
            continue
        root = CACHE_ROOT / source["cache_key"]
        export = read(root / "export_manifest.json")
        validation = read(root / "reproduction_validation.json")
        if export["status"] != "strict_passed" or export["dataset"]["sha256"] != DATASET_SHA:
            raise ValueError("Cache status or dataset mismatch")
        if export["candidate_source"]["checkpoint"]["sha256"] != source["checkpoint_sha256"]:
            raise ValueError("Cache checkpoint mismatch")
        if export["case_count"] != 200 or export["finding_count"] != 381:
            raise ValueError("Incomplete cache")
        if validation["status"] != "passed" or not validation["array_hashes_verified"] or validation["storage_reproduction_status"] != "passed":
            raise ValueError("Cache has no strict storage validation")
        if export["candidate_source"]["inference_contract"]["output_layout"] != "F,X,Y,Z":
            raise ValueError("Unsupported cache geometry")
        entries = export["cases"]
        if len(entries) != 200 or {c['name'] for c in entries} != expected_names:
            raise ValueError("Cache case set mismatch")
        source.update(export_manifest_sha256=file_hash(root / "export_manifest.json"),
                      validation_sha256=file_hash(root / "reproduction_validation.json"),
                      reference={k: validation[k] for k in ("mean_global_dice_per_finding", "total_hits")},
                      cases={c["name"]: c for c in entries})
    if recipe == 'b1':
        validate_routing(sources, recipe_spec, cases)
    for c in cases:
        path = SEG_ROOT / c.get("seg_path", c["name"])
        if path.parent != SEG_ROOT or not path.is_file():
            raise ValueError("Missing or invalid GT path")
        shape = list(nib.load(str(path)).shape)
        if len(shape) != 4 or shape[0] != len(c["findings"]):
            raise ValueError("GT finding-first shape mismatch")
        c.update(gt_path=str(path), gt_sha256=file_hash(path), shape=shape)
        for source in needed_sources(sources, recipe_spec, recipe, c):
            record = source['cases'][c['name']]
            p = Path(record['array_path'])
            if p.stat().st_size != record['npy_bytes'] or record['status'] != 'complete':
                raise ValueError(f"Incomplete array: {p}")
            a = np.load(p, mmap_mode='r', allow_pickle=False)
            expected_shape = [len(record.get('finding_ids', c['findings'])), *shape[1:]]
            if list(a.shape) != expected_shape or list(a.shape) != record['shape'] or str(a.dtype) != record['dtype'] or a.dtype not in (np.float16, np.float32):
                raise ValueError(f"Array header mismatch: {p}")
            del a
    evaluator = SEG_ROOT.parent / "rexrank_eval.py"
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    contract = {"schema_version": 1, "recipe": recipe, "recipe_spec": recipe_spec,
                "sources": sources, "cases": cases, "dataset_sha256": DATASET_SHA,
                "threshold_percentages": list(PERCENTAGES), "dice_epsilon": 1e-6,
                "hit_threshold": .1, "comparison": ">=", "component_filtering": False,
                "sweep_sha256": file_hash(__file__), "source_manifest_sha256": file_hash(source_manifest) if source_manifest else None, "evaluator_sha256": file_hash(evaluator),
                "code_commit": commit, "numpy_version": np.__version__, "nibabel_version": nib.__version__}
    return {"identity": digest(contract), "contract": contract,
            "environment": {"python": platform.python_version(), "hostname": platform.node(),
                            "container_image": os.environ.get("SWEEP_IMAGE_ID", "unrecorded")}}


def cutoffs(domain, models=1):
    if domain == 'logit':
        return np.log(THRESHOLDS / (1 - THRESHOLDS)).astype(np.float32)
    return (THRESHOLDS * models).astype(np.float32)


def count_chunk(values, gt, boundaries):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt, dtype=bool).reshape(-1)
    if values.size != gt.size or not np.isfinite(values).all():
        raise ValueError("Mismatched chunk or nonfinite logits/probabilities")
    # Bin j contains values passing exactly j inclusive cutoffs.
    bins = np.searchsorted(boundaries, values, side='right')
    total = np.bincount(bins, minlength=len(boundaries)+1)
    positive = np.bincount(bins[gt], minlength=len(boundaries)+1)
    return (np.cumsum(total[::-1], dtype=np.int64)[::-1][1:],
            np.cumsum(positive[::-1], dtype=np.int64)[::-1][1:])


def sigmoid(values):
    p = np.array(values, dtype=np.float32, copy=True)
    np.clip(p, -30, 30, out=p)
    np.negative(p, out=p)
    np.exp(p, out=p)
    p += np.float32(1)
    np.reciprocal(p, out=p)
    return p


def validate_record(record, identity, case):
    if record.get('identity') != identity or record.get('case') != case['name']:
        raise ValueError("Partial result identity mismatch")
    rows = record['rows']
    if record.get('rows_sha256') != digest(rows):
        raise ValueError("Partial result content digest mismatch")
    expected = {(int(k), p) for k in case['findings'] for p in PERCENTAGES}
    if len(rows) != len(expected) or {(r['finding_index'], r['threshold_pct']) for r in rows} != expected:
        raise ValueError("Partial result finding/threshold coverage mismatch")
    for r in rows:
        if r['case'] != case['name'] or r['category'] != case['categories'][str(r['finding_index'])]:
            raise ValueError("Partial result category mismatch")
        g, p, i = (r[k] for k in ('gt_voxels', 'pred_voxels', 'intersection'))
        if min(g, p, i) < 0 or i > min(g, p):
            raise ValueError("Invalid voxel counts")
        d = dice(g, p, i)
        if abs(d-r['dice']) > 1e-12 or r['hit'] != int(d >= .1):
            raise ValueError("Partial result metric mismatch")
    return rows


def evaluate_case(manifest, case, output_root, chunk_elements):
    import nibabel as nib
    t = time.monotonic()
    identity, contract = manifest['identity'], manifest['contract']
    output = Path(output_root) / 'cases' / f"{case['name']}.json"
    if output.exists():
        result = read(output)
        validate_record(result, identity, case)
    if file_hash(case['gt_path']) != case['gt_sha256']:
        raise ValueError('GT changed after preflight')
    gt_all = np.asanyarray(nib.load(case['gt_path']).dataobj)
    if list(gt_all.shape) != case['shape'] or not np.isfinite(gt_all).all():
        raise ValueError('GT shape or finite-value check failed')
    sources = needed_sources(contract['sources'], contract['recipe_spec'], contract['recipe'], case)
    arrays = {}
    for s in sources:
        r = s['cases'][case['name']]
        array = np.load(r['array_path'], mmap_mode='r', allow_pickle=False)
        if array_hash(array) != r['array_sha256']:
            raise ValueError(f"Array hash mismatch: {r['array_path']}")
        arrays[s['candidate_id']] = array
    if output.exists():
        return case['name'], True, time.monotonic()-t
    rows = []
    recipe = contract['recipe']
    for k in sorted(case['findings'], key=int):
        index = int(k)
        category = case['categories'][k]
        if recipe == 'b1':
            selected_sources = [s for s in sources if s['candidate_id'] == contract['recipe_spec']['categories'][category]]
        else:
            selected_sources = sources
        selected = [arrays[s['candidate_id']] for s in selected_sources]
        flats = [a[channel_index(s['cases'][case['name']], index)].reshape(-1) for s, a in zip(selected_sources, selected)]
        gt = (gt_all[index] > 0).reshape(-1)
        g = int(gt.sum())
        bounds = cutoffs('logit' if len(selected) == 1 else 'probability_sum', len(selected))
        pred_count = np.zeros(19, dtype=np.int64)
        intersections = np.zeros(19, dtype=np.int64)
        for start in range(0, gt.size, chunk_elements):
            stop = min(gt.size, start+chunk_elements)
            if len(flats) == 1:
                values = np.asarray(flats[0][start:stop], dtype=np.float32)
            else:
                values = np.zeros(stop-start, dtype=np.float32)
                for flat in flats:
                    part = flat[start:stop]
                    if not np.isfinite(part).all():
                        raise ValueError('Nonfinite ensemble source')
                    values += sigmoid(part)
            p, i = count_chunk(values, gt[start:stop], bounds)
            pred_count += p
            intersections += i
        for j, pct in enumerate(PERCENTAGES):
            p, i = int(pred_count[j]), int(intersections[j])
            d = dice(g, p, i)
            rows.append(dict(case=case['name'], finding_index=index, category=category,
                             threshold_pct=pct, threshold=pct/100, gt_voxels=g,
                             pred_voxels=p, intersection=i, dice=d, hit=int(d >= .1)))
    result = dict(identity=identity, case=case['name'], rows=rows, rows_sha256=digest(rows),
                  elapsed_seconds=time.monotonic()-t)
    validate_record(result, identity, case)
    if recipe == 'b1':
        for row in (r for r in rows if r['threshold_pct'] == 50):
            source = next(s for s in sources if s['candidate_id'] == contract['recipe_spec']['categories'][row['category']])
            entry = source['cases'][case['name']]
            expected = entry['same_pass_reference']['finding_dice'][channel_index(entry, row['finding_index'])]
            if abs(row['dice']-expected) > 1e-10 or row['hit'] != int(expected >= .1):
                raise ValueError('B1 routed source reference mismatch')
    if recipe == 'a1':
        reference = sources[0]['cases'][case['name']]['same_pass_reference']
        baseline = [r for r in rows if r['threshold_pct'] == 50]
        if any(abs(r['dice']-d) > 1e-10 for r, d in zip(baseline, reference['finding_dice'])) or sum(r['hit'] for r in baseline) != reference['hits']:
            raise ValueError(f"Case 0.50 reference mismatch: {case['name']}")
    atomic_json(output, result)
    return case['name'], False, time.monotonic()-t


def best_row(rows, metric):
    return min(rows, key=lambda r: (-r[metric], abs(r['threshold_pct']-50), r['threshold_pct']))


def aggregate(rows):
    result = []
    for scope in ['overall', *CATEGORIES]:
        for pct in PERCENTAGES:
            selected = [r for r in rows if r['threshold_pct'] == pct and (scope == 'overall' or r['category'] == scope)]
            n = len(selected)
            hits = sum(r['hit'] for r in selected)
            result.append(dict(category=scope, label=CATEGORIES.get(scope, 'All findings'),
                               threshold_pct=pct, threshold=pct/100, findings=n, hits=hits,
                               dice=math.fsum(r['dice'] for r in selected)/n if n else None,
                               hit_rate=hits/n if n else None))
    comparisons = []
    for scope in ['overall', *CATEGORIES]:
        rs = [r for r in result if r['category'] == scope]
        base = next(r for r in rs if r['threshold_pct'] == 50)
        item = dict(category=scope, label=base['label'], findings=base['findings'],
                    baseline_dice=base['dice'], baseline_hit_rate=base['hit_rate'])
        for metric in ['dice', 'hit_rate']:
            best = best_row(rs, metric) if base['findings'] else None
            item.update({f'best_{metric}_threshold': best['threshold'] if best else None,
                         f'best_{metric}': best[metric] if best else None,
                         f'{metric}_delta': best[metric]-base[metric] if best else None})
        comparisons.append(item)
    return result, comparisons


def write_csv(path, rows):
    path = Path(path)
    tmp = path.with_name('.'+path.name+'.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def collect(manifest, root):
    rows = []
    for c in manifest['contract']['cases']:
        rows.extend(validate_record(read(root / 'cases' / f"{c['name']}.json"), manifest['identity'], c))
    if len(rows) != 7239:
        raise ValueError('Expected 7239 finding-threshold records')
    metrics, comparisons = aggregate(rows)
    baseline = next(r for r in metrics if r['category'] == 'overall' and r['threshold_pct'] == 50)
    expected, historical = baseline_references(manifest['contract'])
    if expected and (abs(baseline['dice']-expected['dice']) > 1e-10 or baseline['hits'] != expected['hits']):
        raise ValueError(f"0.50 baseline gate failed: {baseline}; expected {expected}")
    for pct in PERCENTAGES:
        cats = [r for r in metrics if r['category'] != 'overall' and r['threshold_pct'] == pct]
        all_row = next(r for r in metrics if r['category'] == 'overall' and r['threshold_pct'] == pct)
        if sum(r['findings'] for r in cats) != 381 or sum(r['hits'] for r in cats) != all_row['hits']:
            raise ValueError('Category recomposition failed')
        if abs(math.fsum((r['dice'] or 0)*r['findings'] for r in cats)/381-all_row['dice']) > 1e-12:
            raise ValueError('Category Dice recomposition failed')
    write_csv(root/'finding_metrics.csv', rows)
    write_csv(root/'category_metrics.csv', [r for r in metrics if r['category'] != 'overall'])
    write_csv(root/'overall_metrics.csv', [r for r in metrics if r['category'] == 'overall'])
    write_csv(root/'comparison.csv', comparisons)
    summary = dict(identity=manifest['identity'], cases=200, findings=381, thresholds=19,
                   finding_threshold_records=len(rows), baseline=baseline, comparisons=comparisons,
                   qualification='Same-validation exploratory tuning; no thresholds adopted.',
                   expected_baseline=expected, historical_reference=historical, baseline_gate='passed')
    atomic_json(root/'summary.json', summary)
    outputs = {p.name: file_hash(p) for p in [root/'summary.json', *sorted(root.glob('*.csv'))]}
    atomic_json(root/'completion.json', dict(identity=manifest['identity'], status='complete',
                                            completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), outputs=outputs))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=['a1', 'b1', 'd1', 'e1'], default='a1')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--chunk-elements', type=int, default=4*1024**2)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Score and retain the first case only')
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_elements < 1:
        parser.error('workers and chunk size must be positive')
    if args.recipe != 'a1' and args.output_root is None:
        parser.error('Future recipes require an explicit --output-root')
    print('Validating frozen recipe, cache headers, and label hashes...', flush=True)
    manifest = prepare(args.recipe, args.source_manifest)
    if args.preflight:
        print(json.dumps(dict(identity=manifest['identity'], cases=200, findings=381,
                              recipe=args.recipe, sources=len(manifest['contract']['sources']))))
        return
    root = args.output_root or RUN_ROOT
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mpath = root/'run_manifest.json'
        if mpath.exists() and read(mpath)['identity'] != manifest['identity']:
            raise ValueError('Existing run belongs to a different source/scoring identity; use a new output root')
        if not mpath.exists():
            manifest['execution'] = dict(workers=args.workers, chunk_elements=args.chunk_elements,
                                         command=os.sys.argv, started_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            atomic_json(mpath, manifest)
        else:
            manifest = read(mpath)
        start = time.monotonic()
        try:
            if args.smoke:
                result = evaluate_case(manifest, manifest['contract']['cases'][0], str(root), args.chunk_elements)
                atomic_json(root/'smoke.json', dict(identity=manifest['identity'], status='passed', case=result[0], seconds=result[2]))
                print('Retained smoke case passed: ' + str(result), flush=True)
                return
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                cases = iter(manifest['contract']['cases'])
                pending = {pool.submit(evaluate_case, manifest, c, str(root), args.chunk_elements)
                           for c in [next(cases) for _ in range(min(args.workers, 200))]}
                done = 0
                while pending:
                    completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                    # Observe every result before scheduling more work; failures stop dispatch.
                    results = [future.result() for future in completed]
                    for name, reused, seconds in results:
                        done += 1
                        progress = dict(status='running', cases_complete=done, total_cases=200,
                                        last_case=name, reused=reused, last_case_seconds=seconds,
                                        elapsed_seconds=time.monotonic()-start)
                        atomic_json(root/'progress.json', progress)
                        print(json.dumps(progress), flush=True)
                        case = next(cases, None)
                        if case is not None:
                            pending.add(pool.submit(evaluate_case, manifest, case, str(root), args.chunk_elements))
            summary = collect(manifest, root)
            atomic_json(root/'progress.json', dict(status='complete', cases_complete=200,
                        elapsed_seconds=time.monotonic()-start, baseline=summary['baseline']))
            print(json.dumps(summary['baseline']), flush=True)
        except BaseException as exc:
            atomic_json(root/'failure.json', dict(identity=manifest['identity'], error=str(exc)))
            raise


if __name__ == '__main__':
    main()
