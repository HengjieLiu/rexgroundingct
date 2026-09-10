#!/usr/bin/env python3
"""Verified +20 mm rerun, with zero-margin comparison and frozen PP2 eligibility."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


base = module('outside_lung_base', HERE / 'analyze.py')
PP2_PATH = base.REPO / 'scripts/rexgroundingct/run_024_test_inference_anatomy.py'
pp2 = module('outside_lung_pp2', PP2_PATH)
BASE_ROOT = base.ROOT / 'visualizations' / HERE.name
ROUTES = base.ROOT / '024_test_inference_anatomy_audit/config/prompt_routing_val.json'
ROUTES25 = base.ROOT / '025_iso07_best_anatomy_audit/config/prompt_routing_val.json'


def outside_count(candidate, total):
    coverage = candidate['gt_coverage']
    if (candidate['gt_voxels'] != total or not np.isfinite(coverage)
            or not 0 <= coverage <= 1 or not candidate['selected']
            or candidate['labels'] != base.LUNG_LABELS):
        raise ValueError('Invalid Exp022 coverage provenance')
    inside = total * coverage
    rounded = round(inside)
    if abs(inside - rounded) > 1e-6:
        raise ValueError('Coverage does not recover an unambiguous integer count')
    return total - rounded


def expanded_row(old, outside, eligible):
    total, vv = old['total_gt_voxels'], old['voxel_volume_mm3']
    if not 0 <= outside <= old['outside_voxels'] <= total:
        raise ValueError('Expansion must not increase outside GT')
    result = dict(old)
    for field in ('outside_voxels', 'outside_volume_mm3', 'outside_volume_ml', 'outside_percentage', 'has_outside'):
        result['baseline_0mm_' + field] = old[field]
    result.update(outside_voxels=outside, outside_volume_mm3=outside*vv,
                  outside_volume_ml=outside*vv/1000,
                  outside_percentage=100*outside/total if total else None,
                  has_outside=outside > 0, margin_mm=20.0, method2_eligible=eligible,
                  method2_gt_at_risk_voxels=outside if eligible else 0,
                  method2_gt_at_risk_volume_mm3=outside*vv if eligible else 0,
                  method2_gt_at_risk_volume_ml=outside*vv/1000 if eligible else 0,
                  method2_gt_at_risk_percentage=(100*outside/total if eligible else 0) if total else None,
                  measurement_source='Exp022 all_lung:mask:20; source hashes verified')
    return result


def verify_sources(original):
    for path, expected in original['inputs'].items():
        if base.sha(Path(path)) != expected:
            raise ValueError('Original cohort/metadata/audit input changed')
    source = {r['volume_name']:r for r in base.read(base.AUDIT)['cases']}
    for i, e in enumerate(original['evidence'], 1):
        if e['paths']['anatomy'] != source[e['case']]['integrity']['path']:
            raise ValueError('Different source lung mask')
        for key in ('anatomy', 'gt'):
            if base.sha(Path(e['paths'][key])) != e['hashes'][key]:
                raise ValueError('Source/GT hash changed')
        if e['hashes']['anatomy'] != source[e['case']]['integrity']['observed_sha256']:
            raise ValueError('Exp020 source hash mismatch')
        if base.sha(Path(e['exp022_path'])) != e['exp022_sha256']:
            raise ValueError('Exp022 cached measurement changed')
        ct = nib.load(e['paths']['ct'])
        if (list(ct.shape) != e['ct_shape'] or not np.allclose(ct.affine, e['ct_affine'], atol=1e-5, rtol=0)
                or ct.header.get_xyzt_units()[0] != 'mm'):
            raise ValueError('CT header changed')
        if i % 50 == 0:
            print(f'Verified {i}/200 source-mask/GT pairs and CT headers', flush=True)


def direct_check(job):
    evidence, rows, old_case = job
    images = {k:nib.load(v) for k,v in evidence['paths'].items()}
    vv = base.geometry(images['ct'], images['anatomy'], images['gt'], len(rows))
    spacing = np.linalg.norm(images['ct'].affine[:3,:3], axis=0)
    normalized = images['ct'].affine[:3,:3] / spacing
    if not np.allclose(normalized.T @ normalized, np.eye(3), atol=1e-6):
        raise ValueError('Nonorthogonal grid incompatible with PP2 EDT')
    anatomy = np.asanyarray(images['anatomy'].dataobj)
    lung = np.isin(anatomy, base.LUNG_LABELS)
    del anatomy
    support = pp2.physical_dilation(lung, images['ct'].affine, 20.0)
    if not lung.any() or np.any(lung & ~support):
        raise ValueError('Invalid expanded lung support')
    del lung
    gt_data = np.asanyarray(images['gt'].dataobj)
    union = np.zeros(support.shape, dtype=bool)
    for row in rows:
        gt = base.foreground(gt_data[row['finding_index']])
        union |= gt
        result = base.measure(gt, support, vv)
        if (result['outside_voxels'] != row['outside_voxels']
                or result['total_gt_voxels'] != row['total_gt_voxels']):
            raise ValueError('Actual PP2 EDT differs from cached Exp022 coverage')
    case_result = base.measure(union, support, vv)
    if case_result['total_gt_voxels'] != old_case['total_gt_voxels']:
        raise ValueError('Case GT union differs from original analysis')
    return dict(case=evidence['case'], findings_checked=len(rows),
                status='PASS', method='actual PP2 physical_dilation EDT at 20 mm',
                union_measurements=case_result)


def select_direct_cases(rows):
    affected = {r['case'] for r in rows if r['has_outside']}
    selected = set(affected)
    for cat in base.CATEGORIES:
        candidates = [r for r in rows if r['category'] == cat and r['case'] not in affected]
        if candidates:
            selected.add(min(candidates, key=lambda r:r['validation_index'])['case'])
    return affected, selected


def comparison_rows(old_rows, rows):
    old_summaries = {s['category']:s for s in base.summarize(old_rows)}
    summaries = base.summarize(rows)
    result = []
    for s in summaries:
        cat = s['category']
        eligible = [r for r in rows if r['category'] == cat and r['method2_eligible']]
        row = dict(category=cat, N=s['N'], M_0mm=old_summaries[cat]['M'], M_20mm=s['M'],
                   method2_eligible_N=len(eligible), method2_eligible_M=sum(r['has_outside'] for r in eligible),
                   method2_bypassed_N=s['N']-len(eligible))
        for stat in ('sum','mean','median','p95','max'):
            for margin, table in ((0, old_summaries[cat]), (20, s)):
                for field in ('outside_voxels','outside_volume_ml','outside_volume_mm3'):
                    row[f'{field}_{stat}_{margin}mm'] = table[f'{field}_{stat}']
        for margin, table in ((0, old_summaries[cat]), (20, s)):
            for stat in ('mean','median','p95','max'):
                row[f'outside_percentage_{stat}_{margin}mm'] = table[f'outside_percentage_{stat}']
        result.append(row)
    return summaries, result


def make_report(rows, cases, comparison, direct):
    eligible = [r for r in rows if r['method2_eligible']]
    m = sum(r['has_outside'] for r in rows)
    risk = sum(r['has_outside'] for r in eligible)
    lines = ['# Val200 GT outside the lung mask expanded by 20 mm', '',
             f'Outside GT decreases from **271/381 findings (71.1%) to {m}/381 ({100*m/381:.1f}%)**. '
             f'At scan-union level, **{sum(r["has_outside"] for r in cases)}/200 scans** retain outside GT.', '',
             f'Actual method 2 has **{risk}/{len(eligible)} eligible findings** with GT outside its support; '
             f'{381-len(eligible)} findings bypass clipping. These are GT-coverage measurements, '
             'not observed prediction losses or Dice regressions.', '',
             'Same downloaded Exp020 CT-RATE `ts_total` masks, labels 10–14. Expansion uses '
             'the actual PP2 physical Euclidean dilation helper with native spacing and a 20 mm radius '
             '(voxel-center distance ≤20 mm + 1e-7). No resampling or inference. '
             'Source status remains `PASS_PENDING_MANUAL_VISUAL_REVIEW`.', '',
             '| Category | M/N, 0 mm | M/N, +20 mm | Mean outside %, +20 mm | Outside voxels, +20 mm | Outside mL, +20 mm | Eligible M/N, +20 mm |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in comparison:
        empty = ' — no findings' if not c['N'] else ''
        lines.append(f"| {c['category']} | {c['M_0mm']}/{c['N']} | {c['M_20mm']}/{c['N']}{empty} | "
                     f"{base.fmt(c['outside_percentage_mean_20mm'])} | {base.fmt(c['outside_voxels_sum_20mm'])} | "
                     f"{base.fmt(c['outside_volume_ml_sum_20mm'])} | {c['method2_eligible_M']}/{c['method2_eligible_N']} |")
    lines += ['', '## Paired outside voxel and volume summaries at +20 mm', '',
              'Each statistic has adjacent voxel and mL columns. Category sums count findings '
              'and may count overlapping voxels repeatedly. Per-scan rows use GT unions.', '',
              '| Category | Sum voxels | Sum mL | Mean voxels | Mean mL | Median voxels | Median mL | P95 voxels | P95 mL | Max voxels | Max mL |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in comparison:
        values = [c[f'{field}_{stat}_20mm'] for stat in ('sum','mean','median','p95','max')
                  for field in ('outside_voxels','outside_volume_ml')]
        lines.append(f"| {c['category']} | " + ' | '.join(map(base.fmt,values)) + ' |')
    lines += ['', '## Distributions', '',
              'All 381 findings enter these geometric distributions, including PP2-bypassed findings and zeros. '
              'The paired plots show voxel counts left and volume in mL right, with histograms above ECDFs.', '',
              '| Category | Paired voxels and volume | Percentage |', '| --- | --- | --- |']
    for cat in base.CATEGORIES:
        links = [' / '.join(f'[{ext.upper()}](figures/{cat}_{kind}.{ext})' for ext in ('png','pdf'))
                 for kind in ('outside_voxels_volume','outside_percentage')]
        lines.append(f'| {cat} | ' + ' | '.join(links) + ' |')
    lines += ['', 'Combined overviews: [paired page 1](figures/overview_voxels_volume_1.png), '
              '[paired page 2](figures/overview_voxels_volume_2.png), '
              '[percentage page 1](figures/overview_percentage_1.png), '
              '[percentage page 2](figures/overview_percentage_2.png).', '',
              '## Verification and detailed outputs', '',
              'All 200 source-mask/GT pairs were rehashed; CT headers, Exp022 records, and fixed '
              'cohort metadata match the zero-margin analysis. Per-finding +20 mm counts reuse '
              'Exp022’s exact whole-lung coverage with integer-recovery validation. '
              f'The actual method-2 EDT helper independently recomputed **{len(direct)} scans / '
              f'{sum(r["findings_checked"] for r in direct)} findings**, including every scan with remaining '
              'outside GT and a deterministic fully-contained sample by category. All counts match. '
              'Case unions were recomputed for these scans; elsewhere all individual findings are fully '
              'contained, so their union has zero outside voxels.', '',
              '`per_finding.csv` retains 0 mm and +20 mm measurements, total GT sizes, native voxel volume, '
              'PP2 eligibility, and GT-at-risk measurements. `per_case.csv` records unions. '
              '`category_summary.csv` contains paired 0/+20 mm counts, volumes, and percentages. '
              '`remaining_outside_findings.csv` lists every finding with outside GT. Detailed records, '
              'figures, and provenance remain external.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE_ROOT / 'lung20mm')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(base.REPO) or out == BASE_ROOT or args.workers < 1:
        raise ValueError('Use a new external output directory and positive workers')
    if (out / 'measurements.json').exists():
        raise ValueError('Existing completed measurements; choose a new output directory')
    original = base.read(BASE_ROOT / 'measurements.json')
    old_manifest = base.read(BASE_ROOT / 'run_manifest.json')
    if base.sha(BASE_ROOT / 'measurements.json') != old_manifest['outputs']['measurements.json']:
        raise ValueError('Zero-margin measurement hash mismatch')
    if len(original['rows']) != 381 or len(original['cases']) != 200:
        raise ValueError('Unexpected zero-margin cohort size')
    verify_sources(original)
    routes = {(r['case'],int(r['finding_id'])):r for r in base.read(ROUTES)['rows']}
    routes25 = {(r['case'],int(r['finding_id'])):r for r in base.read(ROUTES25)['rows']}
    if len(routes) != 381 or set(routes) != set(routes25):
        raise ValueError('Frozen routing census mismatch')
    rows = []
    for e in original['evidence']:
        refs = {r['finding_id']:r for r in base.read(e['exp022_path'])['findings']}
        for old in (r for r in original['rows'] if r['case'] == e['case']):
            key = (old['case'],old['finding_index'])
            route = routes[key]
            if (route['category'] != old['category'] or route['prompt'] != old['prompt']
                    or route['eligible'] != routes25[key]['eligible']):
                raise ValueError('Frozen prompt/category/eligibility mismatch')
            count = outside_count(refs[old['finding_index']]['candidates']['all_lung:mask:20'], old['total_gt_voxels'])
            rows.append(expanded_row(old, count, bool(route['eligible'])))
    rows.sort(key=lambda r:(r['validation_index'],r['finding_index']))
    if sum(r['method2_eligible'] for r in rows) != 366:
        raise ValueError('Expected 366 eligible / 15 bypassed findings')
    if Counter(r['category'] for r in rows) != {c:n for c,n in base.EXPECTED.items() if n}:
        raise ValueError('Category census mismatch')
    affected, selected = select_direct_cases(rows)
    old_cases = {r['case']:r for r in original['cases']}
    jobs = [(e,[r for r in rows if r['case']==e['case']],old_cases[e['case']])
            for e in original['evidence'] if e['case'] in selected]
    out.mkdir(parents=True, exist_ok=True)
    snapshots = {'measurement_source.py':Path(__file__), 'shared_source.py':HERE/'analyze.py', 'pp2_source.py':PP2_PATH}
    for dest, src in snapshots.items():
        (out/dest).write_bytes(src.read_bytes())
    print(f'Direct PP2 EDT checks: {len(affected)} affected scans + {len(selected-affected)} fully-contained representatives', flush=True)
    direct = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(direct_check,jobs),1):
            direct.append(result)
            print(f'Direct EDT verified {i}/{len(jobs)} scans', flush=True)
    bycase = {r['case']:r for r in direct}
    cases = []
    for old in original['cases']:
        outside = bycase[old['case']]['union_measurements']['outside_voxels'] if old['case'] in bycase else 0
        row = expanded_row(old, outside, False)
        for field in [k for k in row if k.startswith('method2_')]:
            row.pop(field)
        group = [r for r in rows if r['case']==old['case']]
        row['findings_with_outside'] = sum(r['has_outside'] for r in group)
        row['measurement_source'] = 'direct PP2 EDT and GT union' if old['case'] in bycase else 'all finding outside counts are zero; union outside is zero'
        if not max(r['outside_voxels'] for r in group) <= outside <= sum(r['outside_voxels'] for r in group):
            raise ValueError('Case union count bounds failed')
        cases.append(row)
    for row in rows:
        if row['case'] in bycase:
            row['measurement_source'] += '; independently matched actual PP2 EDT'
    summaries, comparison = comparison_rows(original['rows'],rows)
    inputs = {str(p):base.sha(p) for p in (BASE_ROOT/'measurements.json', BASE_ROOT/'run_manifest.json',base.AUDIT,ROUTES,ROUTES25)}
    base.write_json(out/'measurements.json',dict(rows=rows,cases=cases,direct_checks=direct,inputs=inputs))
    base.write_csv(out/'per_finding.csv',rows)
    base.write_csv(out/'per_case.csv',cases)
    base.write_csv(out/'category_summary.csv',comparison)
    remaining = [r for r in rows if r['has_outside']]
    if remaining:
        base.write_csv(out/'remaining_outside_findings.csv',remaining)
    (out/'report.md').write_text(make_report(rows,cases,comparison,direct))
    print('Rendering +20 mm paired voxel/volume and percentage distributions',flush=True)
    accounting = base.plots(out,rows,summaries,support_label='Lung mask +20 mm')
    for record in accounting:
        if record['N'] != base.EXPECTED[record['category']] or record['N'] != record['zero']+record['positive']:
            raise ValueError('Plot census mismatch')
    manifest = dict(generated_at=datetime.now(timezone.utc).isoformat(),command=sys.argv,margin_mm=20,
                    lung_labels=base.LUNG_LABELS,source_status=base.SOURCE_STATUS,
                    findings=len(rows),scans=len(cases),outside_findings=len(remaining),outside_scans=len(affected),
                    method2_eligible_findings=366,method2_bypassed_findings=15,
                    eligible_outside_findings=sum(r['has_outside'] and r['method2_eligible'] for r in rows),
                    direct_edt_scans=len(direct),direct_edt_findings=sum(r['findings_checked'] for r in direct),
                    input_sha256=inputs,plot_accounting=accounting,
                    checks=dict(source_hashes='PASS',ct_headers='PASS',coverage_integer_recovery='PASS',
                                direct_pp2_edt='PASS',monotonicity='PASS',case_union_bounds='PASS',plot_census='PASS'),
                    outputs={str(p.relative_to(out)):base.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='run_manifest.json'})
    base.write_json(out/'run_manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('outside_findings','outside_scans','eligible_outside_findings','direct_edt_scans','direct_edt_findings')},indent=2),flush=True)


if __name__ == '__main__':
    main()
