#!/usr/bin/env python3
"""Frozen prompt review and native-grid CPU-only counterfactual mask audit.

No input is ever written. Runtime case records are content-addressed by a seal.
Distances are evaluated exactly at GT/prediction foreground voxel centers using
a KD tree of the six-connected anatomical boundary, avoiding full-volume EDTs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/experiments/022_exp007_official_anatomy_val200_audit.json'
SCOPES = {'N': [], 'W': [10, 11, 12, 13, 14], 'B': [10, 11, 12, 13, 14],
          'L': [10, 11], 'R': [12, 13, 14], 'U': [10, 12], 'D': [11, 14],
          'LU': [10], 'LL': [11], 'RU': [12], 'RM': [13], 'RL': [14]}
CATEGORIES = ['1a', '1b', '1c', '1d', '1e', '1f', '2a', '2b', '2c', '2d', '2e', '2f', '2g', '2h']
RISKS = {
    'peripheral': r'peripheral|subpleural|pleural.based|juxtapleural|paraseptal',
    'pleural': r'pleur', 'fissural': r'fissur',
    'central_hilar': r'central|hilar|mediastin|paracardia|pericardia',
    'airway': r'bronch|tree.in.bud|centrilobular|centriacinar|centriacinar',
    'vascular': r'vascular|vessel|aorta|sequestration',
    'diaphragm_chestwall': r'diaphragm|costovertebr|paraspin|osteophyte',
    'distorted_dense_anatomy': r'atelecta|fibro|scarr|consolidat|distort',
    'cyst_bleb': r'cyst|bleb|bulla',
    'halo_surrounding': r'halo|surrounding',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def scope_labels(scope):
    return sorted({label for part in scope.split('+') for label in SCOPES[part]})


def side_labels(labels):
    left, right = bool(set(labels) & {10, 11}), bool(set(labels) & {12, 13, 14})
    return SCOPES['B' if left and right else 'L' if left else 'R' if right else 'N']


def census(val):
    rows = []
    seen = set()
    for e in val['test']:
        if set(e['findings']) != set(e['categories']):
            raise ValueError('Finding/category key mismatch')
        for k, prompt in e['findings'].items():
            key = (e['name'], int(k))
            if key in seen:
                raise ValueError('Duplicate finding key')
            seen.add(key)
            rows.append(dict(case=e['name'], finding_id=int(k), prompt=prompt, category=e['categories'][k]))
    rows.sort(key=lambda r: (r['category'], r['case'], r['finding_id']))
    for i, r in enumerate(rows):
        r['id'] = i
    return rows


def annotate(rows, review_path):
    lines = [s for s in Path(review_path).read_text().splitlines() if s.strip()]
    if len(lines) != len(rows):
        raise ValueError('Manual review must account for every finding')
    for row, line in zip(rows, lines):
        fields, _, note = line.partition('|')
        idx, scope, extent = fields.split()
        if int(idx) != row['id'] or extent not in {'focal', 'multifocal', 'diffuse', 'unclear'}:
            raise ValueError('Manual review ID/extent mismatch')
        labels = scope_labels(scope)
        prompt = row['prompt'].lower()
        risks = [k for k, pattern in RISKS.items() if re.search(pattern, prompt)]
        uncertainty = re.findall(r'\b(?:indeterminate|nonspecific|non-specific|faint\w*|poorly\s+\w+|difficult\s+to\s+\w+|too small to characterize|consistent with|atelectasis-like)\b', prompt)
        temporal = bool(re.search(r'prior|previous|stable|unchanged|newly|new |decreased|progressive', prompt))
        unavailable = []
        if re.search(r'segment|lingul|apic|basal|base', prompt):
            unavailable.append('No exact segment/zone/lingula-only label; lobe is only a broader parent when explicit.')
        if 'airway' in risks:
            unavailable.append('No complete bronchial tree/wall label in total; trachea is not equivalent.')
        if 'pleural' in risks or 'fissural' in risks or row['category'] == '2g':
            unavailable.append('No exact pleura/fissure/pleural-fluid/pleural-air target label in total.')
        if 'vascular' in risks:
            unavailable.append('Aorta/pulmonary-vein labels do not identify an aberrant vessel or vessel-on-end nodule.')
        if not labels:
            rationale = 'No prompt-routed hard restriction: target compartment is not represented by a suitable total label.'
        else:
            rationale = f'Compare whole lung, supported side, and {scope} as broad anatomical supports, not lesion boundaries.'
            if scope in {'B', 'W'}:
                rationale += ' No exclusive narrower compartment is established.'
            if risks:
                rationale += ' Boundary/quality risks: ' + ', '.join(risks) + '; inspect loss, not just average Dice.'
        row.update(scope=scope, extent=extent, labels=labels, side_labels=side_labels(labels),
                   eligible=bool(labels), risks=risks, uncertainty=uncertainty,
                   temporal_comparison=temporal,
                   current_target_negation=False,
                   negation_note='No prompt negates the current target; prior absence/stability is temporal, not exclusion.' if temporal else 'No current-target negation.',
                   target_relation='unsupported target; anatomical mentions cannot substitute for it' if not labels else 'target within broad pulmonary support; named adjacent structures are landmarks',
                   unavailable_anatomy=unavailable, semantic_note=note.strip(),
                   rationale=rationale + (' ' + note.strip() if note.strip() else ''),
                   reviewer='Codex semantic review of every complete prompt; not human clinical review')
    return rows


def config():
    return read(CONFIG)


def freeze(review_path):
    c = config()
    root = Path(c['runtime_root'])
    dest = root / 'frozen_review.json'
    if dest.exists():
        raise FileExistsError('Frozen review already exists; never overwrite outcome-blind evidence')
    if sha(REPO / c['val_json']) != c['val_sha256'] or sha(c['eval_json']) != c['eval_sha256']:
        raise ValueError('Pinned input changed')
    rows = annotate(census(read(REPO / c['val_json'])), review_path)
    if len(rows) != 381 or len({r['case'] for r in rows}) != 200:
        raise ValueError('Unexpected cohort')
    audit_path = Path(c['source_audit_root']) / 'reports/private/audit_results.json'
    lock_path = Path(c['source_audit_root']) / 'config/source_lock.json'
    sources = {r['volume_name']: r for r in read(audit_path)['cases']}
    if set(sources) != {r['case'] for r in rows} or read(audit_path)['lut_status'] != 'PASS':
        raise ValueError('Official anatomy cohort/LUT mismatch')
    lock = read(lock_path)
    write(root / 'config_snapshot.json', c)
    write(root / 'source_lut.json', lock['lut_provenance']['total_task_label_map'])
    write(dest, dict(created_at=now(), purpose='Frozen before new GT/prediction outcome measurements',
                     review_sha256=sha(review_path), rows=rows))
    # Private hand-reviewed source table is retained verbatim for reproducibility.
    (root / 'semantic_review.tsv').write_text(Path(review_path).read_text())
    pins = {str(p): sha(p) for p in [CONFIG, Path(__file__), REPO / c['val_json'],
                                    Path(c['eval_json']), audit_path, lock_path, dest]}
    write(root / 'seal.json', dict(created_at=now(), pins=pins, sources=sources,
                                 source_review_status=c['source_review_status'], argv=sys.argv,
                                 environment=dict(python=sys.version, numpy=np.__version__, nibabel=nib.__version__)))
    print('Frozen', len(rows), 'semantic reviews;', sha(dest), flush=True)


def verify_seal(c):
    root = Path(c['runtime_root'])
    seal_path = root / ('measurement_seal.json' if (root / 'measurement_seal.json').exists() else 'seal.json')
    seal = read(seal_path)
    for path, digest in seal['pins'].items():
        if sha(path) != digest:
            raise ValueError(f'Sealed source changed: {path}')
    return seal, sha(seal_path)


def revise_implementation(reason):
    """Preserve the original outcome-blind seal; revise only code before cases."""
    root = Path(config()['runtime_root'])
    if list((root / 'cases').glob('*.json')) or (root / 'measurement_seal.json').exists():
        raise ValueError('Revision is only allowed before any completed measurement and only once')
    original = read(root / 'seal.json')
    for p, digest in original['pins'].items():
        if p != str(Path(__file__)) and sha(p) != digest:
            raise ValueError('Revision cannot change review, config, or data')
    original['implementation_revision'] = dict(reason=reason, original_seal_sha256=sha(root / 'seal.json'),
                                               old_code_sha256=original['pins'][str(Path(__file__))], at=now())
    original['pins'][str(Path(__file__))] = sha(__file__)
    write(root / 'measurement_seal.json', original)


def geometry(ct, anatomy, gt, pred, ids):
    if ct.shape != anatomy.shape or len(ct.shape) != 3:
        raise ValueError('CT/anatomy grid mismatch')
    if not np.allclose(ct.affine, anatomy.affine, atol=1e-5, rtol=0):
        raise ValueError('CT/anatomy affine mismatch; no automatic resampling')
    if len(gt.shape) != 4 or gt.shape != pred.shape or gt.shape[1:] != ct.shape:
        raise ValueError('Finding-first GT/prediction/native spatial shape mismatch')
    if sorted(ids) != list(range(gt.shape[0])):
        raise ValueError('Finding-axis census mismatch')
    if not np.array_equal(gt.affine, np.eye(4)) or not np.array_equal(pred.affine, gt.affine):
        raise ValueError('Unexpected legacy finding-first header; investigate export, do not resample')
    scales = np.linalg.norm(ct.affine[:3, :3], axis=0)
    norm = ct.affine[:3, :3] / scales
    if not np.allclose(norm.T @ norm, np.eye(3), atol=1e-6):
        raise ValueError('Non-orthogonal CT grid incompatible with axis-spacing distance')
    return scales


def support_distances(mask, coords, spacing, maximum=20):
    """Exact minimum Euclidean distance to foreground voxel centers, capped >20.

    Any closest foreground point for an exterior grid point lies on a 6-neighbor
    boundary. Inside points have distance zero. KDTree eps=0 is exact; queries
    beyond the largest margin can safely remain infinity.
    """
    if not np.any(mask):
        return np.full(len(coords), np.inf), np.full(len(coords), np.inf)
    lo = np.array([np.flatnonzero(np.any(mask, axis=tuple(j for j in range(3) if j != i)))[0] for i in range(3)])
    hi = np.array([np.flatnonzero(np.any(mask, axis=tuple(j for j in range(3) if j != i)))[-1] for i in range(3)])
    box = np.maximum(np.maximum((lo - coords) * spacing, (coords - hi) * spacing), 0).max(axis=1)
    distances = np.full(len(coords), np.inf)
    inside = mask[tuple(coords.T)]
    distances[inside] = 0
    query = (~inside) & (box <= maximum + 1e-7)
    if query.any():
        boundary = mask & ~binary_erosion(mask, structure=generate_binary_structure(3, 1), border_value=0)
        points = np.argwhere(boundary) * spacing
        tree = cKDTree(points)
        distances[query] = tree.query(coords[query] * spacing, eps=0,
                                      distance_upper_bound=maximum + 1e-6, workers=1)[0]
    return distances, box


def metrics(g, p, allowed):
    tp0, p0, g0 = int((g & p).sum()), int(p.sum()), int(g.sum())
    tp, pn, gn = int((g & p & allowed).sum()), int((p & allowed).sum()), int((g & allowed).sum())
    dice = float((2 * tp + 1e-6) / (pn + g0 + 1e-6)) if pn + g0 else 1.0
    return dict(dice=dice, hit=dice >= .1, precision=tp / pn if pn else 0.,
                recall=tp / g0 if g0 else 1., gt_coverage=gn / g0 if g0 else 1.,
                gt_voxels=g0, pred_voxels=pn, tp=tp, fp=pn-tp,
                tp_removed=tp0-tp, fp_removed=(p0-tp0)-(pn-tp),
                gt_completely_excluded=bool(g0 and not gn), prediction_emptied=bool(p0 and not pn))


def worker(args):
    c, case, reviews, source, seal_hash = args
    dest = Path(c['runtime_root']) / 'cases' / (case + '.json')
    if dest.exists():
        old = read(dest)
        if old['seal_sha256'] != seal_hash:
            raise ValueError('Stale case evidence; refusing reuse')
        for key, path in old['input_paths'].items():
            if sha(path) != old['hashes'][key]:
                raise ValueError('Previously measured input changed')
        return case, 'reused'
    paths = dict(ct=source['source_provenance']['path'], anatomy=source['integrity']['path'],
                 gt=str(Path(c['gt_root']) / case),
                 pred=str(Path(c['eval_json']).parent.parent / 'predictions' / case))
    hashes = {k: sha(v) for k, v in paths.items()}
    if hashes['anatomy'] != source['integrity']['observed_sha256'] or hashes['ct'] != source['source_provenance']['expected_sha256']:
        raise ValueError('Exp020 source content changed')
    if source['geometry']['index_grid_status'] != 'PASS' or source['geometry']['world_header_status'] != 'PASS':
        raise ValueError('Exp020 geometry did not pass')
    images = {k: nib.load(v) for k, v in paths.items()}
    spacing = geometry(images['ct'], images['anatomy'], images['gt'], images['pred'], [r['finding_id'] for r in reviews])
    anatomy = np.asanyarray(images['anatomy'].dataobj)
    if not np.isin(anatomy, np.arange(118)).all():
        raise ValueError('Unexpected total label')
    gt_data, pred_data = np.asanyarray(images['gt'].dataobj), np.asanyarray(images['pred'].dataobj)
    saved = next(x['findings'] for x in read(c['eval_json'])['cases'] if x['file'] == case)
    findings = []
    query = []
    for r in reviews:
        i = r['finding_id']
        if not np.isin(pred_data[i], [0, 1]).all():
            raise ValueError('Expected binary prediction masks')
        if not np.isfinite(gt_data[i]).all() or (gt_data[i] < 0).any() or not np.equal(gt_data[i], np.floor(gt_data[i])).all():
            raise ValueError('Expected nonnegative integer GT instance labels; evaluator foreground is >0')
        gb, pb = gt_data[i] > 0, pred_data[i] > 0
        coords = np.argwhere(gb | pb).astype(np.int32)
        g, p = gb[tuple(coords.T)], pb[tuple(coords.T)]
        base = metrics(g, p, np.ones(len(coords), bool))
        expected = saved[f'finding_{i}']
        if abs(base['dice'] - expected['global_dice']) > c['baseline_tolerance'] or base['hit'] != expected['global_hit']:
            raise ValueError(f'Baseline mismatch for {case}/{i}: {base["dice"]} != {expected["global_dice"]}')
        counts = np.bincount(anatomy[gb].astype(np.int64), minlength=118)
        findings.append(dict(id=r['id'], finding_id=i, baseline=base, candidates={},
                             all_label_gt_voxels=counts.tolist(), saved_dice=expected['global_dice']))
        query.append((coords, g, p))
    del gt_data, pred_data, gb, pb
    offsets = np.cumsum([0] + [len(q[0]) for q in query])
    coords = np.concatenate([q[0] for q in query])
    scopes = {tuple(SCOPES['W'])}
    for r in reviews:
        if r['eligible']:
            scopes.add(tuple(r['labels']))
            scopes.add(tuple(r['side_labels']))
    computed = {}
    for labels in sorted(scopes):
        mask = np.isin(anatomy, labels)
        available = bool(mask.any())
        ds, db = support_distances(mask, coords, spacing, max(c['margins_mm']))
        for j, (r, (_, g, p)) in enumerate(zip(reviews, query)):
            for shape, dist in [('mask', ds), ('bbox', db)]:
                for margin in c['margins_mm']:
                    allowed = dist[offsets[j]:offsets[j+1]] <= margin + 1e-7
                    computed[(j, labels, shape, margin)] = (metrics(g, p, allowed), available)
        del mask, ds, db
    for j, (r, out) in enumerate(zip(reviews, findings)):
        for family in c['families']:
            labels = SCOPES['W'] if family in {'all_lung', 'prompt_lung'} else r['side_labels'] if family == 'prompt_side' else r['labels']
            eligible = family == 'all_lung' or r['eligible']
            for shape in c['shapes']:
                for margin in c['margins_mm']:
                    key = f'{family}:{shape}:{margin}'
                    m, available = computed[(j, tuple(labels), shape, margin)] if eligible else (out['baseline'], False)
                    selected = eligible and available
                    result = dict(m if selected else out['baseline'])
                    result.update(selected=selected, available=available, labels=labels,
                                  delta=(m['dice'] if selected else out['baseline']['dice']) - out['baseline']['dice'],
                                  reason='applied diagnostic constraint' if selected else 'unsupported target' if not eligible else 'empty anatomical region; bypass')
                    out['candidates'][key] = result
    write(dest, dict(case=case, seal_sha256=seal_hash, input_paths=paths, hashes=hashes,
                     spacing_mm=spacing.tolist(), geometry='native CT index grid; legacy finding-first identity GT/pred headers; no resampling',
                     lung_flags={s: source['lungs'][s]['flags'] for s in ['left', 'right']}, findings=findings))
    return case, 'measured'


def measure(workers, limit):
    c = config()
    seal, seal_hash = verify_seal(c)
    rows = read(Path(c['runtime_root']) / 'frozen_review.json')['rows']
    names = sorted({r['case'] for r in rows})
    if limit:
        names = names[:limit]
    tasks = [(c, n, [r for r in rows if r['case'] == n], seal['sources'][n], seal_hash) for n in names]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(worker, t) for t in tasks]
        for i, f in enumerate(as_completed(jobs), 1):
            print(i, '/', len(tasks), *f.result(), flush=True)
    print('MEASUREMENT_COMPLETE', len(tasks), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'measure', 'revise-implementation'])
    p.add_argument('--review', type=Path)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--reason')
    a = p.parse_args()
    if a.stage == 'freeze':
        freeze(a.review)
    elif a.stage == 'revise-implementation':
        if not a.reason:
            p.error('--reason is required')
        revise_implementation(a.reason)
    else:
        measure(a.workers, a.limit)


if __name__ == '__main__':
    main()
