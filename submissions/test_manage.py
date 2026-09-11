"""Central register identity, readiness and history regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import manage as m


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.registry = m.read(m.REGISTRY)

    def fixture(self, root, sid='c1'):
        registry = copy.deepcopy(self.registry)
        row = next(r for r in registry['submissions'] if r['id'] == sid)
        registry['families'][row['family']]['sources'] = []
        source = root / 'source.json'; m.write(source, {'source': 'fixture'})
        ref = {'path': str(source), 'sha256': m.sha(source)}
        row['postprocessing_sources']['implementation'] = ref
        row['postprocessing_sources']['routing'] = ref
        directory = root / row['prediction']['directory_name']
        row['prediction']['path'] = str(directory)
        row['observed'] = {}
        row['zip'] = {'path': None, 'sha256': None}
        row['evidence'] = {'completion': {'path': str(root / 'completion.json'), 'sha256': None},
                           'verification': {'path': str(root / 'verification.json'), 'sha256': None}}
        names = {f'train_{i}_a_1.nii.gz' for i in range(300)}
        return registry, row, directory, names

    def completed_c(self, root):
        registry, row, directory, names = self.fixture(root)
        directory.mkdir()
        files = []
        for i, name in enumerate(sorted(names)):
            (directory / name).write_bytes(b'fixture')
            f = 2 if i < 282 else 1
            files.append({'name': name, 'dtype': 'uint8', 'findings': f, 'shape': [f, 2, 3, 4]})
        m.write(row['evidence']['verification']['path'], {'cases': 300, 'findings': 582, 'files': files})
        m.write(row['evidence']['completion']['path'], {'status': 'COMPLETE',
            'checkpoint_sha256': registry['families']['c']['models'][0]['checkpoint']['sha256'],
            'test_sets': {'baseline': {'directory': str(directory),
                          'verification': row['evidence']['verification']['path']}}})
        return registry, row, directory, names

    def test_registered_identities_and_c_aliases(self):
        expected = {a + n for a in 'abcd' for n in '123'} | {'d11', 'd12'}
        if 'e' in self.registry['families']:
            expected |= {'e1', 'e2', 'e3', 'e11', 'e12'}
            self.assertEqual(len(self.registry['families']['e']['models']), 8)
            self.assertEqual(self.registry['families']['e']['recipe']['weights'], [.125]*8)
        self.assertEqual(m.validate(self.registry), expected)
        models = self.registry['families']
        self.assertEqual([len(models[f]['models']) for f in 'abcd'], [1, 9, 1, 4])
        self.assertTrue(models['a']['models'][0]['checkpoint']['sha256'].startswith('a499ad1c'))
        self.assertTrue(models['c']['models'][0]['checkpoint']['sha256'].startswith('4f36d9bb'))
        self.assertEqual(models['d']['recipe']['weights'], [0.25] * 4)
        self.assertEqual(models['b']['recipe']['kind'], 'category_routing')
        self.assertEqual([r['prediction']['directory_name'] for r in self.registry['submissions']
                          if r['family'] == 'c'], ['baseline', 'whole_lung20', 'fine20'])

    def test_duplicate_alias_id_and_postprocessing_rejected(self):
        for kind in ('id', 'path', 'policy'):
            reg = copy.deepcopy(self.registry)
            if kind == 'id':
                reg['submissions'][1]['id'] = reg['submissions'][0]['id']
            elif kind == 'path':
                reg['submissions'][1]['prediction'] = copy.deepcopy(reg['submissions'][0]['prediction'])
            else:
                reg['submissions'][1]['postprocessing'] = 'raw'
            with self.assertRaises(ValueError):
                m.validate(reg)

    def test_d_pending_has_no_false_readiness_or_upload(self):
        with tempfile.TemporaryDirectory() as d:
            reg, row, directory, names = self.fixture(Path(d), 'd1')
            state = Path(d) / 'state.json'
            m.write(state, {'job_sha256': 'expected', 'status': 'waiting_for_wave1'})
            row['evidence'].update(runner_state=str(state), job_sha256='expected')
            row['zip']['path'] = str(Path(d) / 'd1.zip')
            obs = m.observe(row, reg, names)
            self.assertEqual(obs['prediction_status'], 'pending')
            self.assertEqual(obs['zip_status'], 'pending')
            self.assertEqual(obs['runner_status'], 'waiting_for_wave1')
            self.assertIsNone(obs['findings'])
            self.assertEqual(row['submission_history'], [])
            self.assertFalse(directory.exists())

    def test_ready_missing_files_and_changed_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            reg, row, directory, names = self.completed_c(Path(d))
            obs = m.observe(row, reg, names)
            self.assertEqual(obs['prediction_status'], 'ready')
            self.assertEqual((obs['cases'], obs['findings']), (300, 582))
            p = directory / next(iter(names)); p.unlink()
            self.assertEqual(m.observe(row, reg, names)['prediction_status'], 'partial')
            p.write_bytes(b'fixture')
            row['observed'] = obs
            done = m.read(row['evidence']['completion']['path']); done['unexpected_change'] = True
            m.write(row['evidence']['completion']['path'], done)
            self.assertEqual(m.observe(row, reg, names)['prediction_status'], 'conflict')

    def test_missing_source_mount_is_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            reg, row, _, names = self.fixture(Path(d))
            Path(row['postprocessing_sources']['routing']['path']).unlink()
            self.assertEqual(m.observe(row, reg, names)['prediction_status'], 'unavailable')

    def test_zip_readiness_is_separate_and_hash_drift_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reg, row, _, names = self.completed_c(root)
            archive = root / 'c1.zip'; row['zip']['path'] = str(archive)
            with zipfile.ZipFile(archive, 'w') as z:
                for name in sorted(names): z.writestr(name, b'fixture')
            obs = m.observe(row, reg, names)
            self.assertEqual((obs['prediction_status'], obs['zip_status']), ('ready', 'ready'))
            row['observed'] = obs
            archive.write_bytes(b'wrong archive')
            obs = m.observe(row, reg, names)
            self.assertEqual((obs['prediction_status'], obs['zip_status']), ('ready', 'conflict'))

    def test_refresh_preserves_manual_history_and_notes(self):
        reg = self.registry
        event = {'event_id': 'a1-fixture-01', 'submitted_name': 'Exact external name',
                 'submitted_at_utc': '2026-09-09T12:00:00Z', 'external_id': 'external-id',
                 'zip_sha256': None, 'result_references': ['future-result']}
        reg['submissions'][0]['submission_history'] = [event]
        reg['submissions'][0]['notes'] = 'Keep this review note.'
        cases = [{'name': str(i), 'findings': {str(j): 'p' for j in range(2 if i < 282 else 1)}}
                 for i in range(300)]
        with patch.object(m, 'pinned'), patch.object(m, 'read', return_value={'test': cases}), \
             patch.object(m, 'observe', return_value={'prediction_status': 'pending'}):
            updated = m.refresh(reg)
        self.assertEqual(updated['submissions'][0]['submission_history'], [event])
        self.assertEqual(updated['submissions'][0]['notes'], 'Keep this review note.')
        self.assertIsNot(updated['submissions'][0]['submission_history'], reg['submissions'][0]['submission_history'])

    def test_deterministic_render_and_real_table_current(self):
        self.assertEqual(m.render(self.registry), m.render(copy.deepcopy(self.registry)))
        self.assertEqual((m.HERE / 'REGISTER.md').read_text(), m.render(self.registry))
        self.assertIn('Unrecorded', m.render(self.registry))
        self.assertIn('baseline', m.render(self.registry))

    def test_d11_d12_parent_and_foreign_completion_rejected(self):
        for sid in ('d11', 'd12'):
            reg = copy.deepcopy(self.registry)
            row = next(r for r in reg['submissions'] if r['id'] == sid)
            row['evidence'] = copy.deepcopy(next(r for r in reg['submissions'] if r['id'] == 'd1')['evidence'])
            with self.assertRaises(ValueError): m.validate(reg)
        reg = copy.deepcopy(self.registry)
        next(r for r in reg['submissions'] if r['id'] == 'd12')['parent_id'] = 'd1'
        with self.assertRaises(ValueError): m.validate(reg)

    def test_planned_variants_stay_pending_when_val_complete(self):
        for sid in ('d11', 'd12'):
            reg = copy.deepcopy(self.registry)
            row = next(r for r in reg['submissions'] if r['id'] == sid)
            row['evidence'] = {'kind': 'planned_postprocessing',
                               'completion': {'path': None, 'sha256': None},
                               'verification': {'path': None, 'sha256': None}}
            row['zip'] = {'path': None, 'sha256': None}
            with tempfile.TemporaryDirectory() as d:
                directory = Path(d)/sid
                row['prediction']['path'] = str(directory)
                row['zip']['path'] = str(Path(d)/(sid + '.zip'))
                row['validation']['status'] = 'complete'
                with patch.object(m, 'pinned'):
                    obs = m.observe(row, reg, set())
                    self.assertEqual(obs['prediction_status'], 'planned_pending_review')
                    self.assertEqual(obs['zip_status'], 'planned_pending_review')
                    directory.mkdir()
                    self.assertEqual(m.observe(row, reg, set())['prediction_status'], 'conflict')

    def test_authorized_test_producer_ready_without_archives(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reg, row, directory, names = self.fixture(root, 'd11')
            directory.mkdir()
            source = root/'source.json'
            ref = {'path': str(source), 'sha256': m.sha(source)}
            for key in ('frozen_policy', 'source_archive'): row['postprocessing_sources'][key] = ref
            job_path = root/'job.json'
            job = {'job_sha256': 'a'*64, 'run_id': 'r002_d1_test300_d11_d12_frozen_postprocessing',
                   'candidates': [{'checkpoint': x['checkpoint']} for x in reg['families']['d']['models']]}
            m.write(job_path, job)
            row['evidence'].update(kind='frozen_test_postprocessing', job_sha256=job['job_sha256'], run_id=job['run_id'],
                job={'path': str(job_path), 'sha256': m.sha(job_path)},
                authorization={'validation_score_requirement': 'none', 'zip_policy': 'skipped_by_request'})
            row['zip'] = {'path': None, 'sha256': None, 'policy': 'skipped_by_request'}
            m.validate(reg)
            rows = []
            for i, name in enumerate(sorted(names)):
                (directory/name).write_bytes(b'fixture')
                rows.append({'path': str(directory/name), 'dtype': 'uint8', 'shape': [2 if i < 282 else 1, 2, 3, 4]})
            verification = row['evidence']['verification']['path']
            m.write(verification, {'cases': 300, 'findings': 582, 'files': rows})
            completion = row['evidence']['completion']['path']
            done = {'status': 'complete', 'job_sha256': job['job_sha256'], 'files': 600, 'cases': 300, 'findings': 582,
                    'zip_policy': 'skipped_by_request', 'archives': {}, 'outputs': {'d11': {
                    'directory': str(directory), 'verification': verification, 'verification_sha256': m.sha(verification)}}}
            m.write(completion, done)
            obs = m.observe(row, reg, names)
            self.assertEqual((obs['prediction_status'], obs['zip_status']), ('ready', 'skipped_by_request'))
            self.assertIsNone(obs['zip_sha256'])
            done['outputs']['d11']['verification_sha256'] = 'foreign'
            m.write(completion, done)
            self.assertEqual(m.observe(row, reg, names)['prediction_status'], 'conflict')

    def test_e_family_parents_waiting_and_separate_producer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); reg = copy.deepcopy(self.registry)
            reg['submissions'] = [r for r in reg['submissions'] if r['family'] != 'e']
            family = copy.deepcopy(reg['families']['d']); family['models'] = copy.deepcopy(reg['families']['b']['models'][:8])
            family['recipe'] = {'kind': 'probability_average', 'weights': [.125]*8}; family['sources'] = []
            reg['families']['e'] = family
            job = {'job_sha256': 'a'*64, 'run_id': 'r004_top8_test300_frozen_postprocessing',
                   'candidates': [{'checkpoint': x['checkpoint']} for x in family['models']]}
            jp = root/'job.json'; m.write(jp, job); source = {'path': str(jp), 'sha256': m.sha(jp)}
            state = root/'state.json'; m.write(state, {'job_sha256': job['job_sha256'], 'status': 'waiting_for_wave2_and_val200'})
            for suffix, policy in [('1','raw'),('2','whole_lung20'),('3','fine20'),('11','semantic_v1'),('12','semantic_v2_strict')]:
                row = copy.deepcopy(next(r for r in reg['submissions'] if r['id'] == 'd12'))
                sid = 'e'+suffix
                row.update(id=sid, family='e', postprocessing=policy, parent_id=None if suffix == '1' else 'e11' if suffix == '12' else 'e1',
                           prediction={'path': str(root/sid), 'directory_name': sid}, submission_history=[], observed={})
                row['postprocessing_sources'] = {'implementation': source, 'routing': source}
                row['zip'] = {'path': None, 'sha256': None, 'policy': 'skipped_by_request'}
                row['evidence'] = {'kind': 'frozen_e_ensemble', 'run_id': job['run_id'], 'job_sha256': job['job_sha256'], 'job': source,
                    'runner_state': str(state), 'completion': {'path': str(root/'completion.json'), 'sha256': None},
                    'verification': {'path': str(root/(sid+'.json')), 'sha256': None},
                    'authorization': {'validation_score_requirement': 'none', 'zip_policy': 'skipped_by_request'}}
                reg['submissions'].append(row)
                obs = m.observe(row, reg, set())
                self.assertEqual((obs['prediction_status'], obs['zip_status']), ('waiting_for_wave2_and_val200','skipped_by_request'))
            self.assertEqual(len(m.validate(reg)), 19)
            row['parent_id'] = 'e1'
            with self.assertRaisesRegex(ValueError, 'parent'): m.validate(reg)
            row['parent_id'] = 'e11'; row['evidence']['job_sha256'] = 'foreign'
            self.assertEqual(m.observe(row, reg, set())['prediction_status'], 'conflict')


if __name__ == '__main__':
    unittest.main()
