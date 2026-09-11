#!/usr/bin/env python3
"""Maintain submission metadata; never modify predictions or run experiments."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / 'registry.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(value)
    os.replace(tmp, path)


def write(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(registry):
    require(registry['schema_version'] == 1, 'unsupported registry schema')
    require(registry['dataset']['cases'] == 300 and registry['dataset']['findings'] == 582,
            'test cohort mismatch')
    ids, paths, events = set(), set(), set()
    for family, definition in registry['families'].items():
        require(definition['models'], 'family lacks model identities')
        hashes = []
        for model in definition['models']:
            checkpoint = model['checkpoint']
            require(Path(checkpoint['path']).is_absolute(), 'checkpoint path must be absolute')
            require(re.fullmatch(r'[0-9a-f]{64}', checkpoint['sha256']), 'invalid checkpoint hash')
            require(model['preprocessing']['id'], 'missing preprocessing')
            hashes.append(checkpoint['sha256'])
        require(len(hashes) == len(set(hashes)), 'duplicate checkpoint in family')
        recipe = definition['recipe']
        if recipe['kind'] == 'probability_average':
            weights = recipe['weights']
            require(len(weights) == len(hashes) and all(w > 0 for w in weights)
                    and abs(sum(weights) - 1) < 1e-9, 'invalid ensemble weights')
        if recipe['kind'] == 'category_routing':
            candidates = {m['candidate_id'] for m in definition['models']}
            require(all(v is None or v in candidates for v in recipe['categories'].values()),
                    'category references unknown checkpoint')
    for row in registry['submissions']:
        sid = row['id']
        match = re.fullmatch(r'([a-z]+)([1-9][0-9]*)', sid)
        require(match is not None, 'invalid submission ID')
        require(sid not in ids, 'duplicate submission ID: ' + sid)
        ids.add(sid)
        family, suffix = match.groups()
        require(row['family'] == family and family in registry['families'], 'family mismatch')
        policies = {'1': 'raw', '2': 'whole_lung20', '3': 'fine20',
                    '11': 'semantic_v1', '12': 'semantic_v2_strict'}
        require(suffix in policies and (suffix in ('1', '2', '3') or family in ('d', 'e')), 'unknown submission method ID')
        expected_policy = policies[suffix]
        require(row['postprocessing'] == expected_policy, 'suffix/postprocessing mismatch')
        require(expected_policy in registry['postprocessing'], 'missing postprocessing definition')
        if family == 'e':
            require(row.get('parent_id') == (None if suffix == '1' else 'e11' if suffix == '12' else 'e1'), 'e postprocessing parent mismatch')
            source = row['evidence']
            require(source.get('kind') == 'frozen_e_ensemble'
                    and source['run_id'] == 'r004_top8_test300_frozen_postprocessing'
                    and re.fullmatch(r'[0-9a-f]{64}', source['job_sha256']), 'invalid e producer')
            require(source['completion']['path'] and source['verification']['path'] and source['job']['sha256'], 'missing e producer evidence')
            require(source['authorization'] == {'validation_score_requirement': 'none', 'zip_policy': 'skipped_by_request'}, 'e authorization drift')
            require(row['zip'] == {'path': None, 'sha256': None, 'policy': 'skipped_by_request'}, 'e ZIPs must remain skipped')
            require(len(registry['families']['e']['models']) == 8
                    and registry['families']['e']['recipe']['weights'] == [.125]*8, 'e ensemble drift')
        if sid in ('d11', 'd12'):
            require(row.get('parent_id') == {'d11': 'd1', 'd12': 'd11'}[sid], 'postprocessing parent mismatch')
            source = row['evidence']
            require(source.get('kind') in ('planned_postprocessing', 'frozen_test_postprocessing'), 'unknown postprocessing producer')
            if source['kind'] == 'planned_postprocessing':
                require(source['completion']['path'] is None and source['verification']['path'] is None
                        and not source.get('job_sha256'), 'planned variant cannot inherit another producer')
            else:
                require(source['run_id'] == 'r002_d1_test300_d11_d12_frozen_postprocessing'
                        and re.fullmatch(r'[0-9a-f]{64}', source['job_sha256']), 'invalid test postprocessing producer')
                require(source['completion']['path'] and source['verification']['path'] and source['job']['sha256'], 'missing producer evidence')
                require(source['authorization'] == {'validation_score_requirement': 'none', 'zip_policy': 'skipped_by_request'}, 'test authorization drift')
                require(row['zip'] == {'path': None, 'sha256': None, 'policy': 'skipped_by_request'}, 'ZIPs must remain skipped')
        pred = row['prediction']
        path = Path(pred['path'])
        require(path.is_absolute() and path.name == pred['directory_name'], 'prediction path/name mismatch')
        require(str(path) not in paths, 'conflicting prediction aliases: ' + str(path))
        paths.add(str(path))
        if sid in ('c1', 'c2', 'c3'):
            require(path.name == {'c1': 'baseline', 'c2': 'whole_lung20', 'c3': 'fine20'}[sid],
                    'c-series alias mismatch')
        require(row['expected'] == {'cases': 300, 'findings': 582}, 'expected count mismatch')
        for event in row['submission_history']:
            require(event['event_id'] not in events, 'duplicate submission event')
            events.add(event['event_id'])
            require(event['submitted_name'] and event['submitted_at_utc'], 'upload needs name and time')
            require(isinstance(event['result_references'], list), 'invalid result references')
    return ids


def pinned(rec):
    actual = sha(rec['path'])
    require(actual == rec['sha256'], 'source hash mismatch: ' + rec['path'])
    return actual


def observe(row, registry, expected_names):
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    obs = {'checked_at_utc': now, 'prediction_status': 'unavailable', 'cases': None,
           'findings': None, 'verification_sha256': None, 'completion_sha256': None,
           'zip_status': 'unrecorded' if row['zip']['path'] is None else 'pending',
           'zip_sha256': None, 'zip_bytes': None, 'runner_status': None, 'error': None}
    if row['zip'].get('policy') == 'skipped_by_request':
        obs['zip_status'] = 'skipped_by_request'
    try:
        family = registry['families'][row['family']]
        for rec in family['sources']:
            pinned(rec)
        source = row['evidence']
        if source.get('kind') == 'planned_postprocessing':
            for key in ('implementation', 'frozen_policy', 'source_archive'):
                pinned(row['postprocessing_sources'][key])
            directory = Path(row['prediction']['path'])
            unexpected = directory.exists() or (row['zip']['path'] and Path(row['zip']['path']).exists())
            obs.update(prediction_status='conflict' if unexpected else 'planned_pending_review',
                       zip_status='conflict' if unexpected else 'planned_pending_review',
                       error='Unexpected test artifacts without a reviewed producer.' if unexpected
                       else 'Test generation is pending review of the separate SideExp005 val200 audit.')
            return obs
        for key in ('implementation', 'routing'):
            pinned(row['postprocessing_sources'][key])
        if source.get('kind') in ('frozen_test_postprocessing', 'frozen_e_ensemble'):
            if source['kind'] == 'frozen_test_postprocessing':
                for key in ('frozen_policy', 'source_archive'):
                    pinned(row['postprocessing_sources'][key])
            pinned(source['job'])
            job = read(source['job']['path'])
            require(job['job_sha256'] == source['job_sha256'] and job['run_id'] == source['run_id'], 'foreign postprocessing job')
            require([c['checkpoint']['sha256'] for c in job['candidates']]
                    == [m['checkpoint']['sha256'] for m in family['models']], 'postprocessing checkpoint mapping drift')
        runtime_state = source.get('runner_state')
        if runtime_state and Path(runtime_state).is_file():
            state = read(runtime_state)
            require(state.get('job_sha256') == source['job_sha256'], 'foreign producer state')
            obs['runner_status'] = state.get('status')
        directory = Path(row['prediction']['path'])
        exists = directory.is_dir()
        files = {p.name for p in directory.glob('*.nii.gz') if p.is_file()} if exists else set()
        obs['cases'] = len(files) if exists else None
        obs['prediction_status'] = 'pending' if row['family'] in ('d', 'e') else 'unavailable'
        if source.get('kind') in ('frozen_test_postprocessing', 'frozen_e_ensemble') and obs['runner_status']:
            obs['prediction_status'] = obs['runner_status']
        completion = Path(source['completion']['path'])
        verification = Path(source['verification']['path'])
        if exists:
            obs['prediction_status'] = 'partial' if files else 'pending'
        if not completion.is_file() or not verification.is_file():
            obs['error'] = 'Awaiting completion/verification evidence or source mount.'
            return obs
        for key in ('completion', 'verification'):
            rec = source[key]
            if rec.get('sha256'):
                pinned(rec)
        done = read(completion)
        require(done.get('status', '').lower() == 'complete', 'producer is not complete')
        if row['family'] in ('a', 'b'):
            manifest_ref = source['output_manifest']
            pinned(manifest_ref)
            declared = read(manifest_ref['path'])['outputs'][row['id']]
            require(declared['directory'] == str(directory)
                    and declared['verification_manifest'] == str(verification)
                    and declared['verification_sha256'] == sha(verification), 'producer output mismatch')
        elif row['family'] == 'c':
            declared = done['test_sets'][row['prediction']['directory_name']]
            require(declared['directory'] == str(directory)
                    and declared['verification'] == str(verification)
                    and done['checkpoint_sha256'] == family['models'][0]['checkpoint']['sha256'],
                    'iso07 output/checkpoint mismatch')
        else:
            require(done.get('job_sha256') == source['job_sha256'], 'foreign producer completion')
            if source.get('kind') in ('frozen_test_postprocessing', 'frozen_e_ensemble'):
                declared = done['outputs'][row['id']]
                require(declared['directory'] == str(directory) and declared['verification'] == str(verification)
                        and declared['verification_sha256'] == sha(verification), 'postprocessing output binding drift')
                require(done['files'] == (1500 if source['kind'] == 'frozen_e_ensemble' else 600) and done['cases'] == 300 and done['findings'] == 582
                        and done['zip_policy'] == 'skipped_by_request' and done['archives'] == {}, 'postprocessing completion contract')
        verified = read(verification)
        records = verified['files']
        names = [r.get('name') or Path(r['path']).name for r in records]
        require(verified['cases'] == len(records) == len(set(names)) == 300
                and verified['findings'] == 582, 'verification counts mismatch')
        require(set(names) == expected_names, 'verification test cohort mismatch')
        require(all(r['dtype'] == 'uint8' and len(r['shape']) == 4 for r in records)
                and sum(r.get('findings', r['shape'][0]) for r in records) == 582,
                'verification finding/dtype mismatch')
        if not exists or files != expected_names:
            obs['error'] = 'Prediction directory missing or file set differs from verification.'
            return obs
        obs.update(prediction_status='ready', cases=300, findings=582,
                   verification_sha256=sha(verification), completion_sha256=sha(completion))
        previous = row.get('observed', {})
        for key in ('verification_sha256', 'completion_sha256'):
            if previous.get(key):
                require(previous[key] == obs[key], 'previously observed evidence changed: ' + key)
        try:
            archive_path = row['zip']['path']
            if archive_path is not None and Path(archive_path).is_file():
                archive = Path(archive_path)
                actual = sha(archive)
                declared_sha = row['zip'].get('sha256') or previous.get('zip_sha256')
                if row['family'] == 'd':
                    declared = done['archives'][row['id']]
                    require(declared['path'] == str(archive), 'archive identity mismatch')
                    declared_sha = declared['sha256']
                if declared_sha:
                    require(actual == declared_sha, 'ZIP hash mismatch')
                with zipfile.ZipFile(archive) as z:
                    require(len(z.namelist()) == 300 and set(z.namelist()) == expected_names,
                            'ZIP root file set mismatch')
                obs.update(zip_status='ready', zip_sha256=actual, zip_bytes=archive.stat().st_size)
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
            obs.update(zip_status='conflict', error=str(exc))
    except FileNotFoundError as exc:
        obs.update(prediction_status='unavailable', error=str(exc))
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
        obs.update(prediction_status='conflict', error=str(exc))
    return obs


def refresh(registry):
    result = copy.deepcopy(registry)
    validate(result)
    pinned(result['dataset'])
    cases = read(result['dataset']['path'])['test']
    expected_names = {c['name'] for c in cases}
    require(len(cases) == len(expected_names) == 300
            and sum(len(c['findings']) for c in cases) == 582, 'dataset cohort mismatch')
    for row in result['submissions']:
        row['observed'] = observe(row, result, expected_names)
    result['last_refreshed_at_utc'] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    return result


def clean(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def link(path, label=None):
    if path is None:
        return 'Unrecorded (not created)'
    return f'[{clean(label or path)}](<{path}>)'


def id_order(row):
    match = re.fullmatch(r'([a-z]+)([0-9]+)', row['id'])
    return match.group(1), int(match.group(2))


def render(registry):
    lines = ['# Central submission register', '',
             'Generated from [registry.json](registry.json); edit that file and run `python submissions/manage.py render`.', '',
             'Last artifact refresh: **' + (registry.get('last_refreshed_at_utc') or 'Not checked') + '**.', '',
             'Each set targets **300 test CTs / 582 prompts**. Readiness reflects recorded validation and current file presence; it does not mean uploaded.', '',
             '| ID | Display name | Actual prediction folder | Prediction status | ZIP | Upload history |',
             '| --- | --- | --- | --- | --- | --- |']
    for row in sorted(registry['submissions'], key=id_order):
        obs = row.get('observed', {})
        pred = row['prediction']
        status = obs.get('prediction_status', 'not_checked')
        if obs.get('cases') is not None:
            status += f" ({obs['cases']}/300 CTs"
            if obs.get('findings') is not None:
                status += f", {obs['findings']}/582 prompts"
            status += ')'
        archive = obs.get('zip_status', 'unrecorded')
        if row['zip']['path']:
            archive = link(row['zip']['path'], archive)
        history = row['submission_history']
        uploaded = '; '.join(clean(e['submitted_name']) for e in history) if history else 'Unrecorded'
        lines.append('| ' + ' | '.join([row['id'], clean(row['display_name']),
                     link(pred['path'], pred['directory_name']), status, archive, uploaded]) + ' |')
    lines += ['', '## Naming and postprocessing', '']
    for key in sorted(registry['postprocessing'], key=lambda k: registry['postprocessing'][k]['suffix']):
        p = registry['postprocessing'][key]
        lines.append(f"- **{p['suffix']} — {p['name']}:** {p['description']}")
    lines += ['', 'c1/c2/c3 are aliases for baseline/whole_lung20/fine20; existing folders retain their names.',
              '', '## Model families and provenance', '']
    for family, definition in sorted(registry['families'].items()):
        lines += [f"### {family} — {definition['name']}", '', definition['summary'], '']
        recipe = definition['recipe']
        if recipe['kind'] == 'category_routing':
            lines += ['Category routing: ' + '; '.join(f'{k} → {v or "no assigned checkpoint"}'
                      for k,v in sorted(recipe['categories'].items())) + '.', '']
        if recipe['kind'] == 'probability_average':
            lines += ['Checkpoint weights in the listed order: ' + ', '.join(map(str, recipe['weights']))
                      + '; sigmoid probabilities averaged before threshold ≥ 0.5.', '']
        for model in definition['models']:
            c = model['checkpoint']
            lines.append('- ' + link(c['path'], model['candidate_id']) +
                         f" — SHA256 `{c['sha256']}`; preprocessing `{model['preprocessing']['id']}`.")
        lines += ['', 'Source records: ' + ', '.join(link(r['path'], Path(r['path']).name)
                  for r in definition['sources']) + '.', '']
    lines += ['## Artifact evidence and recorded uploads', '']
    for row in sorted(registry['submissions'], key=id_order):
        obs = row.get('observed', {})
        lines += [f"### {row['id']}", '',
                  '- Prediction path: ' + link(row['prediction']['path']),
                  '- Verification: ' + link(row['evidence']['verification']['path']),
                  '- Completion: ' + link(row['evidence']['completion']['path']),
                  '- Last checked: ' + str(obs.get('checked_at_utc') or 'Unrecorded') + '.',
                  '- ZIP SHA256: ' + (f"`{obs['zip_sha256']}`" if obs.get('zip_sha256') else 'Unrecorded') + '.']
        if obs.get('runner_status'):
            lines.append('- Producer job: ' + obs['runner_status'] + '.')
        if row.get('parent_id'):
            lines.append('- Input variant: **' + row['parent_id'] + '**.')
        if row.get('validation'):
            lines.append('- Separate local val200 evidence: ' + link(row['validation']['report_path'], 'SideExp005 report')
                         + ' (' + row['validation'].get('status', 'unrecorded') + '); not test readiness.')
        if obs.get('error'):
            lines.append('- Observation: ' + clean(obs['error']))
        for event in row['submission_history']:
            lines.append('- Upload `' + clean(event['event_id']) + '`: ' + clean(event['submitted_name'])
                         + ', ' + event['submitted_at_utc'] + '; external ID: '
                         + str(event.get('external_id') or 'Unrecorded') + '; result references: '
                         + clean(event['result_references']) + '.')
        if row.get('notes'):
            lines.append('- Notes: ' + clean(row['notes']))
        lines.append('')
    lines += ['## Results and review', '',
              'Official leaderboard scores are kept in [leaderboard/results.json](leaderboard/results.json). '
              'See [comparison and review notes](reviews/comparison.md). Local val200 measurements are separate.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'refresh', 'render'))
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    args = parser.parse_args()
    registry_digest = sha(args.registry)
    registry = read(args.registry)
    validate(registry)
    output = args.registry.parent / 'REGISTER.md'
    if args.command == 'refresh':
        registry = refresh(registry)
        require(sha(args.registry) == registry_digest, 'registry edited during refresh; rerun to preserve edits')
        write(args.registry, registry)
        atomic_text(output, render(registry))
        conflicts = [r['id'] for r in registry['submissions'] if 'conflict' in (r['observed']['prediction_status'], r['observed']['zip_status'])]
        print(json.dumps({r['id']: r['observed'] for r in registry['submissions']}, indent=2))
        if conflicts:
            raise SystemExit('Conflicting artifact evidence: ' + ', '.join(conflicts))
    elif args.command == 'render':
        atomic_text(output, render(registry))
        print(output)
    else:
        require(output.read_text() == render(registry), 'REGISTER.md is stale; run render')
        conflicts = [r['id'] for r in registry['submissions']
                     if 'conflict' in (r.get('observed', {}).get('prediction_status'),
                                       r.get('observed', {}).get('zip_status'))]
        require(not conflicts, 'unresolved artifact conflicts: ' + ', '.join(conflicts))
        print(f"Register passed: {len(registry['submissions'])} unique variants; generated table matches.")


if __name__ == '__main__':
    main()
