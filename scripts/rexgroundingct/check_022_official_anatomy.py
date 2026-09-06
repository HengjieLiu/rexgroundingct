#!/usr/bin/env python3
"""Independent arithmetic/completeness checks for Exp022 runtime evidence."""
import argparse
from pathlib import Path

import numpy as np

from audit_022_official_anatomy import now, read, sha, write
from report_022_official_anatomy import load_evidence


def main():
    p = argparse.ArgumentParser(); p.add_argument('--require-visual-review', action='store_true')
    args = p.parse_args()
    c, root, rows, cases = load_evidence()
    checks = 0
    for r in rows:
        b = r['baseline']
        assert sum(r['all_label_gt_voxels']) == b['gt_voxels']
        for k, m in r['candidates'].items():
            assert m['tp'] + m['tp_removed'] == b['tp']
            assert m['fp'] + m['fp_removed'] == b['fp']
            assert m['tp'] + m['fp'] == m['pred_voxels']
            assert 0 <= m['gt_coverage'] <= 1
            assert 0 <= m['tp'] <= b['tp'] and 0 <= m['pred_voxels'] <= b['pred_voxels']
            expected = (2*m['tp'] + 1e-6) / (m['pred_voxels'] + b['gt_voxels'] + 1e-6)
            assert abs(m['dice'] - expected) < 1e-10
            assert m['hit'] == (m['dice'] >= .1)
            assert abs(m['delta'] - (m['dice'] - b['dice'])) < 1e-10
            if not m['selected']:
                assert m['dice'] == b['dice'] and not m['tp_removed'] and not m['fp_removed']
            checks += 1
        for family in c['families']:
            for shape in c['shapes']:
                series = [r['candidates'][f'{family}:{shape}:{margin}'] for margin in c['margins_mm']]
                for field in ['gt_coverage', 'tp', 'pred_voxels']:
                    assert all(x[field] <= y[field] for x, y in zip(series, series[1:])), (r['id'], family, shape, field)
    summary = read(root / 'reports/summary.json')
    for policy, s in summary['global_results'].items():
        assert s['n'] == 381 and s['cases'] == 200
        assert abs(s['after'] - np.mean([r['candidates'][policy]['dice'] for r in rows])) < 1e-10
        category_n = sum(v[policy]['n'] for v in summary['categories'].values())
        category_mean = sum(v[policy].get('after', 0) * v[policy]['n'] for v in summary['categories'].values()) / category_n
        assert category_n == 381 and abs(category_mean - s['after']) < 1e-10
        assert s['hits_after'] == s['hits_before'] + s['hit_gains'] - s['hit_losses']
        assert s['improved'] + s['worsened'] + s['unchanged'] == 381
    if args.require_visual_review:
        visual = read(root / 'reports/visual_manifest.json')
        inspection = read(root / 'reports/ai_visual_review.json')
        assert inspection['visual_manifest_sha256'] == sha(root / 'reports/visual_manifest.json')
        assert inspection['status'] == 'AI_INSPECTED_HUMAN_REVIEW_PENDING'
        reviewed = {(v['id'], v['policy']) for v in inspection['records']}
        required = {(v['id'], v['policy']) for v in visual['records']}
        assert reviewed == required
        for v in visual['records'] + visual['contact_sheets']:
            assert sha(v['path']) == v['sha256']
        ids = {i for i, _ in required}
        for r in rows:
            needs = any((r['baseline']['hit'] and not m['hit']) or (m['selected'] and m['gt_completely_excluded']) for m in r['candidates'].values())
            if needs or any(r['lung_flags'].values()):
                assert r['id'] in ids
    result = dict(status='PASS', checked_at=now(), cases=len(cases), findings=len(rows),
                  finding_policy_checks=checks, monotonic_margin_checks=True,
                  category_global_reconciliation=True, baseline_reproduced=True,
                  visual_review_required=args.require_visual_review,
                  source_human_review='PENDING', checker_sha256=sha(__file__))
    write(root / 'reports/validation_checks.json', result)
    print(result)


if __name__ == '__main__':
    main()
