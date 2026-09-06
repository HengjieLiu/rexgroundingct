#!/usr/bin/env python3
"""Add diagnostic views of every residual harm in the highest-observed policy.

This is explicitly outcome-selected error analysis, not policy tuning. Existing
images, contact sheets and the original visual manifest are preserved.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from audit_022_official_anatomy import now, read, sha, write
from report_022_official_anatomy import load_evidence
from visualize_022_official_anatomy import contact_sheets, render_one


def main():
    _, root, rows, cases = load_evidence()
    summary = read(root / 'reports/summary.json')
    policy, _ = max(((k, v) for k, v in summary['global_results'].items() if not k.startswith('all_lung')), key=lambda x: x[1]['after'])
    path = root / 'reports/visual_manifest.json'
    manifest = read(path)
    seen = {(r['id'], r['policy']) for r in manifest['records']}
    selected = [dict(id=r['id'], policy=policy, reasons=['Residual_harm_in_highest_observed_routed_policy; post-outcome diagnostic selection only'], finding=r)
                for r in rows if r['candidates'][policy]['delta'] < -1e-7 and (r['id'], policy) not in seen]
    if not selected:
        print('All residual harms already have panels'); return
    case_map = {c['case']: c for c in cases}
    tasks = [(r, case_map[r['finding']['case']], str(root)) for r in selected]
    with ProcessPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(render_one, tasks))
    extra_root = root / 'additional_review'
    (extra_root / 'visuals').mkdir(parents=True, exist_ok=True)
    sheets = contact_sheets(records, extra_root)
    offset = len(manifest['records'])
    for sheet in sheets:
        sheet['record_indices'] = [i + offset for i in sheet['record_indices']]
    original = root / 'reports/visual_manifest_initial.json'
    if original.exists():
        raise ValueError('Unexpected second extension; preserve prior provenance')
    write(original, manifest)
    manifest['extension'] = dict(at=now(), policy=policy, added_panels=len(records), script_sha256=sha(__file__),
                                 initial_manifest_sha256=sha(original), reason='Inspect all remaining regressions in the reported highest-observed fixed policy; no routing or measurement changes')
    manifest['records'].extend(records); manifest['contact_sheets'].extend(sheets)
    write(path, manifest)
    print('ADDED_RESIDUAL_HARM_PANELS', len(records), flush=True)


if __name__ == '__main__':
    main()
