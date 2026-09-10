#!/usr/bin/env python3
"""Measure GT outside exact downloaded whole lungs, with auditable plots."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path('/mnt/shengdata1/hengjie/experiments/rexgroundingct')
DATA = Path('/data/hengjie/datasets/rexgroundingct')
COHORT = REPO / 'configs/evaluation/rexgroundingct_val200_seed20260723.json'
COHORT_SHA = '7c6db98a2548165cf12448c4b6bc3701f5db8641b5a8627678f69691d8322897'
AUDIT = ROOT / '020_ct_rate_ts_total_rex_val200_audit/reports/private/audit_results.json'
REFERENCE = ROOT / '022_exp007_official_anatomy_val200_audit/cases'
CATEGORIES = ['1a', '1b', '1c', '1d', '1e', '1f', '2a', '2b', '2c', '2d', '2e', '2f', '2g', '2h']
EXPECTED = dict(zip(CATEGORIES, [3, 11, 17, 6, 11, 4, 69, 49, 60, 132, 11, 0, 1, 7]))
LUNG_LABELS = [10, 11, 12, 13, 14]
SOURCE_STATUS = 'PASS_PENDING_MANUAL_VISUAL_REVIEW'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def voxel_volume(affine, units):
    if units != 'mm':
        raise ValueError(f'Expected CT spatial units mm, got {units}')
    value = float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Invalid voxel volume')
    return value


def geometry(ct, anatomy, gt, findings):
    if len(ct.shape) != 3 or anatomy.shape != ct.shape:
        raise ValueError('CT/anatomy spatial shape mismatch')
    if not np.allclose(ct.affine, anatomy.affine, atol=1e-5, rtol=0):
        raise ValueError('CT/anatomy affine mismatch; resampling prohibited')
    if gt.shape != (findings, *ct.shape):
        raise ValueError('GT finding-first spatial shape mismatch')
    if not (np.allclose(gt.affine, np.eye(4), atol=1e-5, rtol=0)
            or np.allclose(gt.affine, ct.affine, atol=1e-5, rtol=0)):
        raise ValueError('Unexpected GT affine')
    if anatomy.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Anatomy spatial units must be mm')
    return voxel_volume(ct.affine, ct.header.get_xyzt_units()[0])


def foreground(data):
    if not np.isfinite(data).all() or (data < 0).any() or not np.equal(data, np.floor(data)).all():
        raise ValueError('Expected nonnegative finite integer GT instance IDs')
    return data > 0


def measure(gt, lung, vv):
    if gt.shape != lung.shape or gt.dtype != bool or lung.dtype != bool:
        raise ValueError('Expected matching boolean masks')
    total = int(np.count_nonzero(gt))
    outside = int(np.count_nonzero(gt & ~lung))
    percentage = 100.0 * outside / total if total else None
    row = dict(voxel_volume_mm3=vv, total_gt_voxels=total,
               total_gt_volume_mm3=total * vv, total_gt_volume_ml=total * vv / 1000,
               outside_voxels=outside, outside_volume_mm3=outside * vv,
               outside_volume_ml=outside * vv / 1000, outside_percentage=percentage,
               has_outside=outside > 0, status='OK' if total else 'EMPTY_GT')
    if total and not np.isclose(percentage, 100 * row['outside_volume_mm3'] / row['total_gt_volume_mm3']):
        raise ValueError('Count and volume percentages disagree')
    return row


def worker(job):
    index, entry, source = job
    name = entry['name']
    ref_path = REFERENCE / (name + '.json')
    ref = read(ref_path)
    paths = {'ct': Path(source['source_provenance']['path']),
             'anatomy': Path(source['integrity']['path']), 'gt': DATA / 'segmentations' / name}
    if source['labels']['status'] != 'PASS' or source['integrity']['status'] != 'VALID':
        raise ValueError(f'Source integrity/labels failed: {name}')
    for key in ('index_grid_status', 'world_header_status'):
        if source['geometry'][key] != 'PASS':
            raise ValueError(f'Source geometry failed: {name}')
    hashes = {k: sha(paths[k]) for k in ('anatomy', 'gt')}
    if hashes['anatomy'] != source['integrity']['observed_sha256']:
        raise ValueError('Exp020 anatomy hash mismatch')
    if any(hashes[k] != ref['hashes'][k] for k in hashes):
        raise ValueError('Exp022 source hash mismatch')
    images = {k: nib.load(p) for k, p in paths.items()}
    ids = sorted(map(int, entry['findings']))
    if ids != list(range(len(ids))):
        raise ValueError('Finding IDs must match GT channels')
    vv = geometry(images['ct'], images['anatomy'], images['gt'], len(ids))
    if not np.allclose(images['ct'].affine, source['geometry']['ct_affine'], atol=1e-5, rtol=0):
        raise ValueError('CT header changed since Exp020')
    if list(images['ct'].shape) != source['geometry']['ct_shape']:
        raise ValueError('CT shape changed since Exp020')
    anatomy = np.asanyarray(images['anatomy'].dataobj)
    if not np.isin(anatomy, np.arange(118)).all():
        raise ValueError('Invalid anatomy label')
    lung = np.isin(anatomy, LUNG_LABELS)
    if not lung.any():
        raise ValueError('Empty lung mask')
    del anatomy
    gt_data = np.asanyarray(images['gt'].dataobj)
    union = np.zeros(lung.shape, dtype=bool)
    refs = {r['finding_id']: r for r in ref['findings']}
    if set(refs) != set(ids):
        raise ValueError('Exp022 finding census mismatch')
    rows = []
    for fid in ids:
        gt = foreground(gt_data[fid])
        union |= gt
        row = dict(case=name, validation_index=index, finding_index=fid,
                   prompt=entry['findings'][str(fid)], category=entry['categories'][str(fid)],
                   **measure(gt, lung, vv))
        counts = refs[fid]['all_label_gt_voxels']
        expected_total = sum(counts)
        expected_outside = expected_total - sum(counts[i] for i in LUNG_LABELS)
        if (row['total_gt_voxels'], row['outside_voxels']) != (expected_total, expected_outside):
            raise ValueError(f'Exp022 voxel-count disagreement: {name}/{fid}')
        rows.append(row)
    case = dict(case=name, validation_index=index, finding_count=len(rows),
                findings_with_outside=sum(r['has_outside'] for r in rows), **measure(union, lung, vv))
    evidence = dict(case=name, paths={k: str(v) for k, v in paths.items()}, hashes=hashes,
                    exp022_path=str(ref_path), exp022_sha256=sha(ref_path),
                    ct_affine=images['ct'].affine.tolist(), ct_shape=list(images['ct'].shape),
                    ct_units='mm', ct_payload_hashing='not repeated; header-only CT use',
                    source_visual_qc_status=source['visual_qc_status'],
                    geometry='native index grid; finding-first GT; no resampling',
                    exp022_count_check='PASS')
    return rows, case, evidence


def statistics(values):
    if not values:
        return dict(sum=0, mean=None, median=None, p95=None, max=None)
    a = np.asarray(values)
    return dict(sum=float(a.sum()), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def summarize(rows):
    summaries = []
    for category in CATEGORIES:
        group = [r for r in rows if r['category'] == category]
        n = len(group)
        m = sum(r['has_outside'] for r in group)
        s = dict(category=category, M=m, N=n, M_over_N=f'{m}/{n}',
                 affected_percentage=100*m/n if n else None,
                 empty_gt_count=sum(r['status'] == 'EMPTY_GT' for r in group),
                 status='OK' if n else 'NO_FINDINGS')
        for field in ('outside_voxels', 'outside_volume_mm3', 'outside_volume_ml', 'outside_percentage'):
            for stat, value in statistics([r[field] for r in group if r[field] is not None]).items():
                if field == 'outside_percentage' and stat == 'sum':
                    continue
                s[f'{field}_{stat}'] = value
        summaries.append(s)
    return summaries


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fmt(x):
    return '—' if x is None else f'{x:,.3f}'


def report(summaries, rows, cases):
    m = sum(r['has_outside'] for r in rows)
    lines = ['# Val200 GT outside the downloaded whole-lung mask', '',
             f'**{m}/{len(rows)} findings** have at least one GT voxel outside lung, across {len(cases)} scans.', '',
             'Exact CT-RATE `ts_total` labels 10–14; native GT > 0; no dilation or resampling. '
             'M/N counts findings, not separate lesion IDs. Volumes use each CT affine determinant and are displayed in mL. '
             'The source remains `PASS_PENDING_MANUAL_VISUAL_REVIEW`.', '',
             '| Category | M/N | Affected % | Outside % mean | Median | P95 | Max |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        label = s['M_over_N'] + (' — no findings' if not s['N'] else '')
        vals = [s['affected_percentage']] + [s['outside_percentage_'+k] for k in ('mean','median','p95','max')]
        lines.append(f"| {s['category']} | {label} | " + ' | '.join(map(fmt, vals)) + ' |')
    lines += ['', '## Outside voxel counts and volumes', '',
              'Each statistic is paired as **voxels / mL**. Sums are across findings and may count overlapping voxels repeatedly. '
              'Per-case CSV measurements use the GT union instead.', '',
              '| Category | Sum voxels | Sum mL | Mean voxels | Mean mL | Median voxels | Median mL | P95 voxels | P95 mL | Max voxels | Max mL |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        vals = [s[f'{field}_{stat}'] for stat in ('sum','mean','median','p95','max')
                for field in ('outside_voxels','outside_volume_ml')]
        lines.append(f"| {s['category']} | " + ' | '.join(map(fmt, vals)) + ' |')
    lines += ['', '## Distribution figures', '',
              '| Category | Outside voxels and volume, side by side | Outside percentage |',
              '| --- | --- | --- |']
    for s in summaries:
        cat = s['category']
        links = [' / '.join(f'[{ext.upper()}](figures/{cat}_{kind}.{ext})' for ext in ('png', 'pdf'))
                 for kind in ('outside_voxels_volume', 'outside_percentage')]
        lines.append(f'| {cat} | ' + ' | '.join(links) + ' |')
    lines += ['', 'Combined overviews: '
              '[voxels/volume 1](figures/overview_voxels_volume_1.png), '
              '[voxels/volume 2](figures/overview_voxels_volume_2.png), '
              '[percentage 1](figures/overview_percentage_1.png), '
              '[percentage 2](figures/overview_percentage_2.png).']
    lines += ['', '## Verification and artifacts', '',
              'All 200 native grids and 381 per-finding voxel counts were checked against Exp020/Exp022. '
              'CT spatial units are mm. Source-mask and GT hashes match the prior audit; CT intensity payloads were not rehashed. '
              f"Empty GT findings: {sum(r['status'] == 'EMPTY_GT' for r in rows)}.", '',
              'External outputs: `per_finding.csv`, `per_case.csv`, `category_summary.csv`, '
              '`measurements.json`, `run_manifest.json`, and `figures/` (PNG/PDF). '
              'Count/volume plots pair voxels left and mL right; percentage plots are separate. '
              'All valid findings, including zeros, enter the distributions.', '']
    return '\n'.join(lines)


def positive_bins(values):
    positive = np.asarray([v for v in values if v > 0], dtype=float)
    if not len(positive):
        return np.geomspace(1, 10, 13)
    lo, hi = float(positive.min()), float(positive.max())
    if lo == hi:
        lo, hi = lo / 2, hi * 2
    return np.geomspace(lo * (1-1e-8), hi * (1+1e-8), 17)


def histogram(ax, values, bins, unit):
    values = np.asarray(values)
    if not len(values):
        ax.text(.5, .5, 'No findings', ha='center', va='center', transform=ax.transAxes)
        return
    zero = int(np.count_nonzero(values == 0))
    positive = values[values > 0]
    # The zero bar is an independent inset, not a fake positive log bin.
    zero_ax = ax.inset_axes([0, 0, .12, 1])
    positive_ax = ax.inset_axes([.23, 0, .77, 1])
    ax.set_axis_off()
    counts, _ = np.histogram(positive, bins=bins)
    assert counts.sum() + zero == len(values)
    zero_ax.bar([0], [zero], width=.7, color='#82919f')
    zero_ax.set_xticks([0], ['0'])
    zero_ax.set_ylabel('Findings')
    positive_ax.hist(positive, bins=bins, color='#277da8', edgecolor='white', linewidth=.3)
    positive_ax.set_xscale('log')
    positive_ax.set_xlim(bins[0], bins[-1])
    positive_ax.set_xlabel(f'Outside {unit} (positive; log scale)')
    top = max(zero, int(counts.max()), 1) * 1.18
    zero_ax.set_ylim(0, top)
    positive_ax.set_ylim(0, top)
    positive_ax.tick_params(axis='y', labelleft=False)
    zero_ax.set_title(f'Zero: {zero}', fontsize=9)
    positive_ax.set_title(f'Positive: {len(positive)}', fontsize=9)
    return dict(N=len(values), zero=zero, positive=int(counts.sum()))


def ecdf(ax, values, unit, maximum, percentage=False):
    x = np.sort(values)
    if len(x):
        ax.step(np.r_[0, x, maximum], np.r_[0, np.arange(1, len(x)+1)/len(x), 1], where='post', color='#277da8')
    else:
        ax.text(.5, .5, 'No findings', ha='center', transform=ax.transAxes)
    if not percentage:
        ax.set_xscale('function', functions=(np.log1p, np.expm1))
        # Sub-unit decades collapse against zero under log1p; omit those ticks.
        ticks = [0] + [10.**k for k in range(0, 10) if 10.**k <= maximum]
        if len(ticks) > 7:
            ticks = [0] + ticks[1:][::2]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel(f'Outside {unit}' + ('' if percentage else ' (log1p scale)'))
    ax.set_ylabel('Cumulative fraction')
    ax.grid(alpha=.2)


def plots(out, rows, summaries, support_label='Exact lung mask'):
    dest = out / 'figures'
    dest.mkdir(exist_ok=True)
    fields = [('outside_voxels', 'voxels'), ('outside_volume_ml', 'volume (mL)')]
    bins = {f: positive_bins([r[f] for r in rows]) for f, _ in fields}
    maxs = {f: max(max(r[f] for r in rows)*1.05, 1) for f, _ in fields}
    accounting = []

    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(dest / f'{name}.{ext}', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Split overview across two pages to keep printed plots readable.
    for start in (0, 7):
        overview, axes = plt.subplots(7, 4, figsize=(22, 28), layout='constrained')
        pct_overview, pct_axes = plt.subplots(7, 2, figsize=(13, 25), layout='constrained')
        for local, s in enumerate(summaries[start:start+7]):
            cat = s['category']
            group = [r for r in rows if r['category'] == cat]
            title = f"Category {cat} | M/N = {s['M_over_N']}"
            fig, a = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
            fig.suptitle(title + ' | ' + support_label, fontsize=14)
            for col, (field, unit) in enumerate(fields):
                values = [r[field] for r in group]
                count = histogram(a[0, col], values, bins[field], unit)
                ecdf(a[1, col], values, unit, maxs[field])
                histogram(axes[local, col], values, bins[field], unit)
                ecdf(axes[local, col+2], values, unit, maxs[field])
                accounting.append(dict(category=cat, field=field, **(count or dict(N=0, zero=0, positive=0))))
            axes[local, 0].text(0, 1.23, title, transform=axes[local, 0].transAxes, fontsize=12)
            save(fig, f'{cat}_outside_voxels_volume')
            fig, a = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
            values = [r['outside_percentage'] for r in group if r['outside_percentage'] is not None]
            fig.suptitle(title + f' | {support_label} | Exact zeros: {sum(v == 0 for v in values)}')
            for ha, ea in ((a[0], a[1]), (pct_axes[local, 0], pct_axes[local, 1])):
                if values:
                    counts, _, _ = ha.hist(values, bins=np.arange(0, 105, 5), color='#277da8', edgecolor='white')
                    assert int(counts.sum()) == len(values)
                else:
                    ha.text(.5, .5, 'No findings', transform=ha.transAxes, ha='center')
                ha.set(xlim=(0,100), xlabel='GT outside lung (%)', ylabel='Findings')
                ecdf(ea, values, 'GT (%)', 100, percentage=True)
            pct_axes[local, 0].set_title(title + f' | zeros: {sum(v == 0 for v in values)}')
            accounting.append(dict(category=cat, field='outside_percentage', N=len(values),
                                   zero=sum(v == 0 for v in values), positive=sum(v > 0 for v in values)))
            save(fig, f'{cat}_outside_percentage')
        overview.suptitle(support_label + ': outside-GT voxel and volume histograms (left pair), ECDFs (right pair)', fontsize=16)
        save(overview, f'overview_voxels_volume_{start//7+1}')
        pct_overview.suptitle(support_label + ': outside-GT percentage distributions', fontsize=16)
        save(pct_overview, f'overview_percentage_{start//7+1}')
    return accounting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'visualizations' / HERE.name)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(REPO):
        raise ValueError('Detailed results must remain outside repository')
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if sha(COHORT) != COHORT_SHA:
        raise ValueError('Fixed val200 manifest hash mismatch')
    entries = read(COHORT)['test']
    metadata_path = DATA / 'MICCAI_challenge_dataset.json'
    metadata = {e['name']: e for e in read(metadata_path)['val']}
    census = Counter(c for e in entries for c in e['categories'].values())
    if len(entries) != 200 or len({e['name'] for e in entries}) != 200 or dict(census) != {k:v for k,v in EXPECTED.items() if v}:
        raise ValueError('Fixed cohort census mismatch')
    for e in entries:
        if set(e['findings']) != set(e['categories']):
            raise ValueError('Finding/category keys mismatch')
        if any(e[k] != metadata[e['name']][k] for k in ('findings', 'categories')):
            raise ValueError('Metadata mismatch')
    audit = read(AUDIT)
    if audit['lut_status'] != 'PASS':
        raise ValueError('LUT provenance must pass')
    sources = {e['volume_name']: e for e in audit['cases']}
    if set(sources) != set(metadata):
        raise ValueError('Exp020 cohort mismatch')
    inputs = {str(p): sha(p) for p in (COHORT, metadata_path, AUDIT)}
    code_sha = sha(Path(__file__))
    if args.plots_only:
        previous = read(out / 'run_manifest.json')
        if sha(out / 'measurements.json') != previous['outputs']['measurements.json']:
            raise ValueError('Measured data hash mismatch')
        measured = read(out / 'measurements.json')
        if measured['inputs'] != inputs:
            raise ValueError('Measured provenance differs from current inputs')
        # Plot revisions can reuse immutable measured counts; preserve both hashes.
        if sha(out / 'measurement_source.py') != measured['code_sha256']:
            raise ValueError('Archived measurement source hash mismatch')
        for e in measured['evidence']:
            for k, h in e['hashes'].items():
                if sha(e['paths'][k]) != h:
                    raise ValueError('Measured source changed')
            if sha(e['exp022_path']) != e['exp022_sha256']:
                raise ValueError('Exp022 reference changed')
    else:
        if (out / 'measurements.json').exists():
            raise ValueError('Existing measurements: use --plots-only or a new output directory')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'measurement_source.py').write_bytes(Path(__file__).read_bytes())
        rows, cases, evidence = [], [], []
        jobs = [(i, e, sources[e['name']]) for i, e in enumerate(entries)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, (finding_rows, case, ev) in enumerate(pool.map(worker, jobs), 1):
                rows.extend(finding_rows)
                cases.append(case)
                evidence.append(ev)
                if i % 10 == 0:
                    print(f'Measured {i}/200 scans, {len(rows)}/381 findings', flush=True)
        measured = dict(rows=rows, cases=cases, evidence=evidence, inputs=inputs, code_sha256=code_sha)
        write_json(out / 'measurements.json', measured)
    rows, cases = measured['rows'], measured['cases']
    if len(rows) != 381 or len({(r['case'], r['finding_index']) for r in rows}) != 381 or len(cases) != 200:
        raise ValueError('Output census mismatch')
    summaries = summarize(rows)
    (out / 'render_source.py').write_bytes(Path(__file__).read_bytes())
    for s in summaries:
        if s['N'] != EXPECTED[s['category']]:
            raise ValueError('Output category census mismatch')
    write_csv(out / 'per_finding.csv', rows)
    write_csv(out / 'per_case.csv', cases)
    # Interleave count/volume statistics for side-by-side CSV inspection.
    summary_fields = ['category','M','N','M_over_N','affected_percentage','empty_gt_count','status']
    summary_fields += [f'{f}_{s}' for s in ('sum','mean','median','p95','max')
                       for f in ('outside_voxels','outside_volume_mm3','outside_volume_ml')]
    summary_fields += ['outside_percentage_'+s for s in ('mean','median','p95','max')]
    write_csv(out / 'category_summary.csv', [{k:s[k] for k in summary_fields} for s in summaries])
    (out / 'report.md').write_text(report(summaries, rows, cases))
    print('Rendering paired distributions...', flush=True)
    accounting = plots(out, rows, summaries)
    for a in accounting:
        assert a['N'] == a['zero'] + a['positive']
    outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run_manifest.json'}
    manifest = dict(generated_at=datetime.now(timezone.utc).isoformat(), source_status=SOURCE_STATUS,
                    code_sha256=code_sha, measurement_code_sha256=measured['code_sha256'],
                    command=os.sys.argv, input_sha256=inputs, lung_labels=LUNG_LABELS,
                    scans=200, findings=381, affected_findings=sum(r['has_outside'] for r in rows),
                    checks=dict(census='PASS', native_geometry='PASS', source_hashes='PASS',
                                exp022_counts='PASS', volume_percentage_equivalence='PASS', plot_accounting='PASS'),
                    plot_accounting=accounting, outputs=outputs,
                    environment=dict(python=os.sys.version, numpy=np.__version__, nibabel=nib.__version__,
                                     matplotlib=matplotlib.__version__, gpu_used=False),
                    repo_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip())
    write_json(out / 'run_manifest.json', manifest)
    print(json.dumps({'output':str(out), 'M':manifest['affected_findings'], 'N':381, 'files':len(outputs)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
