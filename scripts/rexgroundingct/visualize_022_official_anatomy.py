#!/usr/bin/env python3
"""Select diagnostic extrema and render native-grid CT/GT/prediction panels.

This creates images and a pending-review manifest, never claims inspection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from audit_022_official_anatomy import CATEGORIES, config, now, read, sha, support_distances, write
from report_022_official_anatomy import load_evidence


def select(rows):
    selected = defaultdict(list)
    def add(r, k, reason):
        selected[(r['id'], k)].append(reason)
    for cat in CATEGORIES:
        pairs = [(r, k, m) for r in rows if r['category'] == cat for k, m in r['candidates'].items()]
        if pairs:
            for title, fn in [('category_largest_gain', max), ('category_largest_harm', min)]:
                r, k, _ = fn(pairs, key=lambda t: t[2]['delta'])
                add(r, k, title)
    for r in rows:
        lost = [(k, m) for k, m in r['candidates'].items() if r['baseline']['hit'] and not m['hit']]
        excluded = [(k, m) for k, m in r['candidates'].items() if m['selected'] and m['gt_completely_excluded']]
        if lost:
            k, _ = min(lost, key=lambda t: t[1]['dice'])
            add(r, k, 'lost_hit_finding; visual shows most harmful losing policy')
        if excluded:
            k, _ = min(excluded, key=lambda t: t[1]['dice'])
            add(r, k, 'completely_excluded_GT_finding')
        if any(r['lung_flags'].values()):
            add(r, 'all_lung:mask:0', 'Exp020_fragmentation_flag')
    byid = {r['id']: r for r in rows}
    return [dict(id=i, policy=k, reasons=v, finding=byid[i]) for (i, k), v in sorted(selected.items())]


def contours(ax, arr, color, extent):
    if np.any(arr) and not np.all(arr):
        ax.contour(arr.T.astype(float), [.5], colors=[color], linewidths=.65,
                   origin='lower', extent=extent)


def render_one(args):
    record, case, root_s = args
    root = Path(root_s); r = record['finding']; k = record['policy']; m = r['candidates'][k]
    images = {key: nib.load(path) for key, path in case['input_paths'].items()}
    ct = np.asanyarray(images['ct'].dataobj)
    anatomy = np.asanyarray(images['anatomy'].dataobj)
    # Array proxies preserve the finding-first native CT indices; GT >0 joins instances.
    g = np.asanyarray(images['gt'].dataobj[r['finding_id']]) > 0
    p = np.asanyarray(images['pred'].dataobj[r['finding_id']]) > 0
    spacing = np.array(r['spacing_mm'])
    _, shape, margin_s = k.split(':'); margin = int(margin_s)
    mask = np.isin(anatomy, m['labels']) if m['selected'] else np.ones(ct.shape, bool)
    coords = np.argwhere(g | p)
    ds, db = support_distances(mask, coords, spacing)
    allowed = (ds if shape == 'mask' else db) <= margin + 1e-7
    gq, pq = g[tuple(coords.T)], p[tuple(coords.T)]
    removed_gt = gq & ~allowed
    removed_fp = pq & ~gq & ~allowed
    reason = 'largest removed GT cross-section'
    points = coords[removed_gt]
    if not len(points):
        points = coords[removed_fp]; reason = 'removed FP context'
    if not len(points):
        points = np.argwhere(g); reason = 'GT context (no removed foreground)'
    if not len(points):
        points = np.array([np.array(ct.shape) // 2]); reason = 'volume center (empty GT)'
    # A densest native axial plane, then median in-plane location: deterministic.
    z = int(np.argmax(np.bincount(points[:, 2], minlength=ct.shape[2])))
    center = np.array([*np.median(points[points[:, 2] == z, :2], axis=0).astype(int), z])
    half = np.ceil(90 / spacing).astype(int)
    lo = np.maximum(0, center - half); hi = np.minimum(ct.shape, center + half + 1)
    plane_coords = []
    for axis in [2, 1, 0]:
        indices = [np.arange(lo[j], hi[j]) if j != axis else np.array([center[j]]) for j in range(3)]
        grid = np.stack(np.meshgrid(*indices, indexing='ij'), axis=-1).reshape(-1, 3)
        plane_coords.append(grid)
    all_plane = np.concatenate(plane_coords)
    ds, db = support_distances(mask, all_plane, spacing)
    ap = (ds if shape == 'mask' else db) <= margin + 1e-7
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    offset = 0
    codes = nib.aff2axcodes(images['ct'].affine)
    for col, (axis, grid) in enumerate(zip([2, 1, 0], plane_coords)):
        remain = [j for j in range(3) if j != axis]
        sh = tuple(hi[j] - lo[j] for j in remain)
        c2, g2, p2 = [v[tuple(grid.T)].reshape(sh) for v in [ct, g, p]]
        a2 = ap[offset:offset+len(grid)].reshape(sh); offset += len(grid)
        extent = (0, sh[0] * spacing[remain[0]], 0, sh[1] * spacing[remain[1]])
        for row in range(2):
            ax = axes[row, col]
            ax.imshow(c2.T, cmap='gray', vmin=-1000, vmax=300, origin='lower', extent=extent)
            contours(ax, g2, 'lime', extent)
            contours(ax, p2 if row == 0 else p2 & a2, 'magenta' if row == 0 else 'orange', extent)
            if row == 1:
                contours(ax, a2, 'cyan', extent)
            ax.set_title(f"{'Baseline' if row == 0 else 'Constrained'}; native axis {axis}={center[axis]}", fontsize=9)
            ax.set_xlabel(f'+{codes[remain[0]]} (mm)', fontsize=8)
            ax.set_ylabel(f'+{codes[remain[1]]} (mm)', fontsize=8)
            ax.tick_params(labelsize=7)
    title = f"{r['id']:03} / {r['category']} | {k} | Dice {r['baseline']['dice']:.4f} -> {m['dice']:.4f} | GT retained {m['gt_coverage']:.1%}"
    fig.suptitle(title + '\nGT green; baseline magenta; constrained orange; allowed region cyan', fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.text(.02, .01, reason + '; native index directions, not standard radiological display', fontsize=9)
    path = root / 'visuals' / (f"{r['id']:03}_" + k.replace(':', '_') + '.png')
    fig.savefig(path, dpi=110); plt.close(fig)
    return dict(id=r['id'], policy=k, reasons=record['reasons'], path=str(path), sha256=sha(path),
                center_index=center.tolist(), focus=reason, baseline_dice=r['baseline']['dice'],
                after_dice=m['dice'], gt_coverage=m['gt_coverage'],
                inspection_status='PENDING', limitations='Three orthogonal local cuts; not exhaustive 3D review or clinical adjudication.')


def contact_sheets(records, root):
    sheets = []
    for start in range(0, len(records), 4):
        canvas = Image.new('RGB', (1760, 1240), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, record in enumerate(records[start:start+4]):
            im = Image.open(record['path']).convert('RGB'); im.thumbnail((880, 600))
            x, y = (j % 2) * 880, (j // 2) * 620
            canvas.paste(im, (x, y))
            draw.text((x+5, y+602), f"{record['id']:03} {record['policy']} | " + '; '.join(record['reasons']), fill='black')
        path = root / 'visuals' / f'contact_{start//4:03}.jpg'
        canvas.save(path, quality=93)
        sheets.append(dict(path=str(path), sha256=sha(path), record_indices=list(range(start, min(start+4, len(records))))))
    return sheets


def record_review(notes_path):
    """Record explicit notes after actual image inspection; never infer review."""
    root = Path(config()['runtime_root'])
    path = root / 'reports/visual_manifest.json'
    manifest = read(path)
    notes = {}
    for line in Path(notes_path).read_text().splitlines():
        if not line.strip():
            continue
        fields, _, observation = line.partition('|')
        idx, policy, mechanism = fields.split()
        key = int(idx), policy
        if key in notes or not observation.strip():
            raise ValueError('Duplicate or empty visual note')
        notes[key] = dict(mechanism=mechanism, observation=observation.strip())
    required = {(r['id'], r['policy']) for r in manifest['records']}
    if set(notes) != required:
        raise ValueError(f'Visual notes mismatch: missing {required-set(notes)}, extra {set(notes)-required}')
    reviewed = []
    for i, r in enumerate(manifest['records']):
        if sha(r['path']) != r['sha256']:
            raise ValueError('Image changed after generation')
        sheet = next(s for s in manifest['contact_sheets'] if i in s['record_indices'])
        reviewed.append(dict(id=r['id'], policy=r['policy'], image_sha256=r['sha256'],
                             contact_sheet=sheet['path'], contact_sheet_sha256=sheet['sha256'],
                             **notes[(r['id'], r['policy'])]))
    source_root = Path(config()['source_audit_root'])
    source_audit = read(source_root / 'reports/private/audit_results.json')
    flagged = {r['volume_name'] for r in source_audit['cases'] if any(r['lungs'][s]['flags'] for s in ['left', 'right'])}
    panels = [r for r in read(source_root / 'reports/private/visual_qc_manifest.json')['selected'] if r['volume_name'] in flagged]
    if len(panels) != 1:
        raise ValueError('Source-flag review contract requires revisiting if its population changes')
    source_panel = Path(panels[0]['path'])
    result = dict(status='AI_INSPECTED_HUMAN_REVIEW_PENDING', reviewer='Codex; not a clinician or human QC sign-off',
                  recorded_at=now(), notes_sha256=sha(notes_path), visual_manifest_sha256=sha(path), records=reviewed,
                  source_flagged_panel=dict(path=str(source_panel), sha256=sha(source_panel),
                      observation='Inspected full-resolution Exp020 panel: marked asymmetric aeration and dense hemithoracic opacity; masks follow remaining aerated compartments. Three views do not adjudicate every disconnected component. No source-QC status is changed.'),
                  limitations='Contact sheets and selected full-resolution panels show three local orthogonal cuts, not exhaustive 3D review. Mask error versus annotation error often remains uncertain. No human review is claimed.')
    write(root / 'reports/ai_visual_review.json', result)
    lines = ['# Diagnostic visual review', '', '**AI inspection complete for the selected panels; official source human review remains pending.**', '', result['limitations'], '',
             'Every category extremum, every finding losing a hit or fully losing GT, and the fragmentation-flagged scan are covered. With several losing policies for one finding, the most harmful losing policy is shown. Selections are diagnostic, not representative population estimates.', '']
    for r, note in zip(manifest['records'], reviewed):
        lines += [f'<a id="finding-{r["id"]:03}"></a>', f'## {r["id"]:03} — {r["policy"]}', '',
                  '; '.join(r['reasons']), '',
                  f"Dice {r['baseline_dice']:.4f} → {r['after_dice']:.4f}; GT retained {r['gt_coverage']:.1%}.", '',
                  f"![Native CT overlay](../visuals/{Path(r['path']).name})", '',
                  '**Observed:** ' + note['observation'], '', '**Mechanism (provisional):** ' + note['mechanism'].replace('_', ' ') + '.', '']
    lines += ['## Original fragmentation-flag panel', '', result['source_flagged_panel']['observation'], '',
              f"![Source Exp020 QC]({source_panel})", '']
    (root / 'reports/visual_review.md').write_text('\n'.join(lines))
    print('RECORDED_AI_INSPECTION', len(reviewed), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--review-notes', type=Path)
    a = p.parse_args()
    if a.review_notes:
        record_review(a.review_notes)
        return
    _, root, rows, cases = load_evidence()
    (root / 'visuals').mkdir(exist_ok=True)
    selected = select(rows); case_map = {c['case']: c for c in cases}
    tasks = [(r, case_map[r['finding']['case']], str(root)) for r in selected]
    records = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(render_one, t) for t in tasks]
        for f in as_completed(futures):
            record = f.result(); records.append(record)
            print('visual', len(records), '/', len(tasks), record['id'], record['policy'], flush=True)
    records.sort(key=lambda r: (r['id'], r['policy']))
    sheets = contact_sheets(records, root)
    write(root / 'reports/visual_manifest.json', dict(created_at=now(), script_sha256=sha(__file__), records=records, contact_sheets=sheets))
    lines = ['# Diagnostic visual review', '', 'Status: generated; AI inspection pending. Exp020 human review is a separate, still-pending requirement.', '',
             'Every category gain/harm extreme, every finding with a lost hit or completely excluded GT, and every finding on the fragmentation-flagged CT is represented. For a finding losing hits under several policies, show the most harmful losing policy. These diagnostic selections are not a population-quality sample.', '',
             'Three cuts focus on the largest removed-GT cross-section, removed FP, or GT if unchanged. This can suggest an error mechanism but does not establish exhaustive 3D anatomy quality.', '']
    for r in records:
        lines += [f'<a id="finding-{r["id"]:03}"></a>', f'## {r["id"]:03} — {r["policy"]}', '',
                  '; '.join(r['reasons']), '', f"Dice {r['baseline_dice']:.4f} → {r['after_dice']:.4f}; GT retained {r['gt_coverage']:.1%}.", '',
                  f"![Native CT overlay](../visuals/{Path(r['path']).name})", '', 'Inspection: pending.', '']
    (root / 'reports/visual_review.md').write_text('\n'.join(lines))
    print('VISUALS_COMPLETE', len(records), 'panels;', len(sheets), 'contact sheets', flush=True)


if __name__ == '__main__':
    main()
