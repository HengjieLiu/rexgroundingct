from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import resume_seeded_caruana as rs


class ResumeTests(unittest.TestCase):
    def trial(self, label, workers, seconds, status='complete'):
        return dict(label=label, workers=workers, wall_seconds=seconds,
                    status=status, correctness='passed')

    def test_repeat_baseline_is_required(self):
        trials = [self.trial('baseline_first', 5, 100), self.trial('workers20', 20, 10)]
        self.assertEqual(rs.choose_workers(trials)[0], 5)

    def test_faster_baseline_prevents_cache_warmup_false_positive(self):
        trials = [self.trial('baseline_first', 5, 100), self.trial('baseline_repeat', 5, 60),
                  self.trial('workers10', 10, 58)]
        self.assertEqual(rs.choose_workers(trials)[0], 5)

    def test_lower_concurrency_within_five_percent_wins(self):
        trials = [self.trial('baseline_first', 5, 100), self.trial('baseline_repeat', 5, 90),
                  self.trial('workers10', 10, 63), self.trial('workers20', 20, 60)]
        self.assertEqual(rs.choose_workers(trials)[0], 10)

    def test_twenty_wins_when_significantly_faster(self):
        trials = [self.trial('baseline_first', 5, 100), self.trial('baseline_repeat', 5, 90),
                  self.trial('workers10', 10, 70), self.trial('workers20', 20, 60)]
        self.assertEqual(rs.choose_workers(trials)[0], 20)

    def test_timeout_is_not_speed_evidence(self):
        trials = [self.trial('baseline_first', 5, 100), self.trial('baseline_repeat', 5, 90),
                  self.trial('workers10', 10, 1, 'timeout')]
        self.assertEqual(rs.choose_workers(trials)[0], 5)

    def test_correctness_failure_never_falls_back_silently(self):
        with self.assertRaisesRegex(RuntimeError, 'correctness'):
            rs.choose_workers([dict(correctness='failed')])

    def execution(self):
        e = dict(roster_sha256='roster', dataset_sha256='dataset', chunk_elements=42,
                 scopes=list(rs.sc.SCOPES), seed_candidate_ids={'all': ['a', 'b', 'c', 'd']},
                 code={'seeded_caruana_sha256': 'code'}, workers=5, runtime_root='/old',
                 created_at_utc='old')
        e['execution_spec_sha256'] = rs.execution_identity(e)
        return e

    def test_changed_workers_and_paths_are_explicitly_compatible(self):
        parent = self.execution()
        child = copy.deepcopy(parent)
        child.update(workers=10, runtime_root='/new', created_at_utc='new')
        child['execution_spec_sha256'] = rs.execution_identity(child)
        self.assertNotEqual(parent['execution_spec_sha256'], child['execution_spec_sha256'])
        rs.validate_import(parent, child)

    def test_rehashed_numerical_change_still_rejects_import(self):
        parent = self.execution()
        for field, value in [('dataset_sha256', 'other'), ('chunk_elements', 43),
                             ('code', {'seeded_caruana_sha256': 'other'}),
                             ('seed_candidate_ids', {'all': ['x', 'b', 'c', 'd']})]:
            child = copy.deepcopy(parent)
            child[field] = value
            child['execution_spec_sha256'] = rs.execution_identity(child)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'numerical'):
                rs.validate_import(parent, child)

    def test_import_preserves_parent_and_metric_payload(self):
        original = {'fingerprint': 'parent', 'rows': {'trial': [{'dice': 0.25}]}, 'wall_seconds': 10}
        before = copy.deepcopy(original)
        child = rs.imported_partial(original, 'parent', 'child', '/original.json', 'source-sha')
        self.assertEqual(original, before)
        self.assertEqual(child['rows'], original['rows'])
        self.assertEqual(child['reused_from']['sha256'], 'source-sha')
        self.assertEqual(child['fingerprint'], 'child')
        with self.assertRaisesRegex(RuntimeError, 'fingerprint'):
            rs.imported_partial(original, 'wrong', 'child', '/original.json', 'sha')

    def test_fingerprint_changes_with_basket_or_execution(self):
        e = self.execution()
        fp = rs.fingerprint(e, 'k09', 8, {'all': {'a': 8}})
        self.assertNotEqual(fp, rs.fingerprint(e, 'k09', 8, {'all': {'b': 8}}))
        child = copy.deepcopy(e)
        child['workers'] = 10
        child['execution_spec_sha256'] = rs.execution_identity(child)
        self.assertNotEqual(fp, rs.fingerprint(child, 'k09', 8, {'all': {'a': 8}}))

    def partial(self):
        case = {'name': 'case', 'findings': {'0': 'finding'}, 'categories': {'0': '2a'}}
        row = rs.sc._metric_row('case', 0, '2a', 2, 3, 1)
        value = {'case': 'case', 'fingerprint': 'fp', 'rows': {
            'trial:all:a': [copy.deepcopy(row)], 'trial:2a:a': [copy.deepcopy(row)]}}
        return case, value

    def test_partial_validates_coverage_and_voxel_arithmetic(self):
        case, value = self.partial()
        rs.validate_partial(value, 'fp', {'case': case}, ['a'], False)
        value['rows']['trial:all:a'][0]['dice'] = 0.9
        with self.assertRaisesRegex(RuntimeError, 'Dice'):
            rs.validate_partial(value, 'fp', {'case': case}, ['a'], False)

    def test_partial_rejects_missing_candidate_or_duplicate_finding(self):
        case, value = self.partial()
        with self.assertRaisesRegex(RuntimeError, 'trial keys'):
            rs.validate_partial(value, 'fp', {'case': case}, ['a', 'b'], False)
        value['rows']['trial:all:a'] *= 2
        with self.assertRaisesRegex(RuntimeError, 'coverage'):
            rs.validate_partial(value, 'fp', {'case': case}, ['a'], False)

    def test_cleanup_confirms_exit_and_retries_uncertain_daemon_state(self):
        supervisor = object.__new__(rs.Supervisor)
        supervisor.container = 'test-container'
        attempts = []
        def stop():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('daemon temporarily unavailable')
            supervisor.container = None
        supervisor.stop_container = stop
        with mock.patch.object(rs.time, 'sleep') as sleep:
            supervisor.cleanup()
        self.assertEqual(len(attempts), 2)
        sleep.assert_called_once_with(10)
        self.assertIsNone(supervisor.container)

    def test_benchmark_worker_recomputes_and_compares_real_case(self):
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest('container numerical dependencies required')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ['a', 'b', 'c', 'd']
            case = {'name': 'case.nii.gz', 'findings': {'0': 'finding'}, 'categories': {'0': '2a'}}
            nib.save(nib.Nifti1Image(np.array([[[[1]], [[0]]]], dtype=np.uint8), np.eye(4)), root/case['name'])
            paths = {}
            for i, candidate in enumerate(ids):
                path = root/(candidate+'.npy')
                np.save(path, np.full((1,2,1,1), 10.0 if i < 2 else -10.0, dtype=np.float16))
                paths[candidate] = str(path)
            seeds = {scope: ids for scope in rs.sc.SCOPES}
            counts = {scope: {'a': 4, 'b': 1, 'c': 1, 'd': 1} for scope in rs.sc.SCOPES}
            reference = rs.sc._evaluate_case(dict(case=case, paths=paths, candidate_ids=ids,
                pass_root=str(root/'reference'), fingerprint='reference', mode='round', seeds=seeds,
                counts=counts, basket_size=7, chunk_elements=1, segmentation_dir=str(root)))
            reference_path = rs.sc._case_file(root/'reference', case['name'])
            ticket = dict(execution={'execution_spec_sha256': 'test', 'seed_candidate_ids': seeds,
                                     'chunk_elements': 1, 'segmentation_dir': str(root)},
                roster={'candidates': [{'id': c} for c in ids]}, cases=[case],
                paths={c: {case['name']: p} for c,p in paths.items()}, workers=1,
                scratch_root=str(root/'fresh'), label='trial', counts=counts,
                reference_paths={case['name']: str(reference_path)},
                reference_hashes={case['name']: rs.pe.sha256_file(reference_path)},
                measurement_path=str(root/'measurement.json'))
            ticket_path = root/'ticket.json';rs.write(ticket_path,ticket)
            with mock.patch.object(rs.os, 'nice'):
                rs.benchmark_worker(ticket_path)
            measurement = rs.read(root/'measurement.json')
            self.assertEqual(measurement['correctness'], 'passed')
            fresh = rs.read(rs.sc._case_file(root/'fresh/passes/trial',case['name']))
            self.assertEqual(fresh['rows'],reference['rows'])


if __name__ == '__main__':
    unittest.main()
