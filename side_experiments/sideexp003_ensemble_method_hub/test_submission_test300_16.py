import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import submission_test300 as s
import submission_test300_16 as c
import handoff_submission_16 as h
import test_submission_test300 as original_tests


class ContinuationTests(unittest.TestCase):
    def test_full_sixteen_worker_pipeline_reuses_smoke(self):
        real_pool = c.futures.ProcessPoolExecutor
        def continuation(job):
            root = Path(job['runtime_root'])
            for variant in s.VARIANTS:
                (Path(job['output_root']) / variant).mkdir(exist_ok=True)
            s.write(root / 'owner.json', {'job_sha256': job['job_sha256']})
            records = s.bind_caches(job)
            for case in job['cases']:
                if case['name'] in job['smoke_cases']:
                    s.process_case(job, case, [r[case['name']] for r in records])
            hashes = {p: s.sha(p) for v in s.VARIANTS for p in (Path(job['output_root']) / v).glob('*.nii.gz')}
            execution = {'execution_sha256': 'fixture', 'reproduction': 'fixture'}
            with patch.object(c, 'validate_execution'), patch.object(c, 'available_ram', return_value=400*1024**3), \
                 patch.object(c.futures, 'ProcessPoolExecutor', wraps=real_pool) as pool:
                c.run(job, execution)
            self.assertEqual(pool.call_args.kwargs['max_workers'], 16)
            for p, value in hashes.items():
                self.assertEqual(s.sha(p), value)
        with patch.object(s, 'run', side_effect=continuation):
            original_tests.SubmissionTests('test_completed_cache_binding_and_full_runner').test_completed_cache_binding_and_full_runner()

    def test_execution_hash_and_source_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'source'; p.write_text('source')
            e = {'job_sha256': 'job', 'workers': 16, 'cpu_allowance': 16,
                 'source_files': [s.identity(p)]}
            e['execution_sha256'] = s.fc.json_sha256(e)
            with patch.object(s, 'validate_job'):
                c.validate_execution(e, {'job_sha256': 'job'})
                changed = copy.deepcopy(e); changed['workers'] = 4
                with self.assertRaisesRegex(Exception, 'execution hash'):
                    c.validate_execution(changed, {'job_sha256': 'job'})
                p.write_text('drift')
                with self.assertRaisesRegex(Exception, 'source hash'):
                    c.validate_execution(e, {'job_sha256': 'job'})

    def test_handoff_requires_both_complete_smokes_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);job = {'runtime_root': d, 'output_root': str(root/'out'),
                                'job_sha256': 'job', 'smoke_cases': ['a.nii.gz','b.nii.gz']}
            self.assertFalse(h.smoke_ready(job))
            for name in job['smoke_cases']:
                outputs = []
                for v in s.VARIANTS:
                    p = Path(job['output_root']) / v / name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'validated')
                    outputs.append({'path':str(p),'sha256':s.sha(p)})
                s.write(root/'cases'/(name+'.json'), {'job_sha256':'job','name':name,'status':'complete','outputs':outputs})
            self.assertTrue(h.smoke_ready(job))
            p.write_bytes(b'corrupt')
            with self.assertRaisesRegex(Exception,'hash drift'):
                h.smoke_ready(job)


if __name__ == '__main__':
    unittest.main()
