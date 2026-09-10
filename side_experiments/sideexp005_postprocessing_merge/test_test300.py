"""Unattended gate, method parity and publication/restart acceptance."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runner as core
import test300_postprocessing as t
import launch_test300 as launch
from test_postprocessing import PROMPTS, findings, synthetic
import test_collect

t.scientific()
np, nib = t.np, t.nib


class Test300Tests(unittest.TestCase):
    def fixture(self, root, count=1):
        cfg = {'runtime_root': str(root/'run'), 'output_root': str(root/'outputs'),
               'run_id': 'r002_d1_test300_d11_d12_frozen_postprocessing', 'workers': 4,
               'poll_seconds': .01, 'storage': {'reserve_bytes': 0, 'headroom_bytes': 0},
               'source_status': 'PASS_PENDING_MANUAL_VISUAL_REVIEW', 'inputs': {}}
        root.joinpath('inputs').mkdir(parents=True)
        cases = []
        for i in range(count):
            n = (2 if i < 282 else 1) if count == 300 else 2
            name = f'case{i:03}.nii.gz'
            affine = np.diag([-2., -3., 4., 1.])
            anatomy = np.zeros((8, 9, 10), dtype=np.uint8)
            for k in range(5): anatomy[1+k, 2:7, 2:8] = 10+k
            ct_path, anat_path, pred = [root/'inputs'/(prefix+name) for prefix in ('ct_', 'anat_', 'raw_')]
            ct = nib.Nifti1Image(np.zeros(anatomy.shape, dtype=np.uint8), affine)
            nib.save(ct, ct_path); nib.save(nib.Nifti1Image(anatomy, affine), anat_path)
            raw = np.ones((n, *anatomy.shape), dtype=np.uint8)
            nib.save(nib.Nifti1Image(raw, affine), pred)
            fs = findings([('right upper lobe nodule', '2d'), ('apical opacity', '2a')][:n])
            case = {'name': name, 'findings': fs, 'shape': list(raw.shape), 'affine': affine.tolist(),
                    'ct_path': str(ct_path), 'ct_header_sha256': t.hashlib.sha256(nib.load(ct_path).header.binaryblock).hexdigest(),
                    'anatomy': {'path': str(anat_path), 'sha256': t.sha(anat_path)},
                    'input': {'path': str(pred), 'sha256': t.sha(pred)}}
            cases.append(case)
        job = {'run_id': cfg['run_id'], 'config_sha256': t.digest(cfg), 'code': [], 'cases': cases,
               'smoke_cases': [cases[0]['name']], 'routing_counts': {}, 'candidates': []}
        job['job_sha256'] = t.job_digest(job)
        t.atomic_json(Path(cfg['runtime_root'])/'job_manifest.json', job)
        t.atomic_json(Path(cfg['runtime_root'])/'validation_gate.json', {'score_requirement': 'none'})
        t.worker_init(cfg, job)
        return cfg, job

    def gate_fixture(self, root, score):
        val_cfg = test_collect.CollectionTests().fixture(root)
        job = t.read(root/'job_manifest.json')
        t.atomic_json(root/'config.json', val_cfg)
        t.atomic_json(root/'launch.json', {'container_id': 'fixture'})
        t.atomic_json(root/'closeout_source.json', {'source': 'fixture'})
        done = t.read(root/'completion.json')
        done['metrics']['d11']['dice'] = score
        t.atomic_json(root/'completion.json', done)
        t.atomic_json(root/'reports/summary.json', done)
        t.atomic_json(root/'state.json', {'status': 'complete', 'job_sha256': job['job_sha256']})
        t.atomic_json(root/'closeout_state.json', {'status': 'complete', 'job_sha256': job['job_sha256'],
            'files_verified': 1000, 'finding_evaluations_verified': 1905,
            'report_sha256': t.sha(root/'reports/report.md'), 'summary_sha256': t.sha(root/'reports/summary.json')})
        dependency = {'runtime_root': str(root), 'job_sha256': job['job_sha256']}
        for key, filename in [('job', 'job_manifest.json'), ('config', 'config.json'),
                              ('launch', 'launch.json'), ('collector_source', 'closeout_source.json')]:
            dependency[key] = {'path': str(root/filename), 'sha256': t.sha(root/filename)}
        return {'validation_dependency': dependency}

    def test_gate_scores_irrelevant_but_integrity_required(self):
        for score in (0., 1.):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d); cfg = self.gate_fixture(root, score)
                self.assertEqual(t.validation_gate(cfg)['score_requirement'], 'none')
                close = t.read(root/'closeout_state.json')
                t.atomic_json(root/'closeout_state.json', {'status': 'waiting_for_completion'})
                self.assertIsNone(t.validation_gate(cfg))
                t.atomic_json(root/'closeout_state.json', close)
                (root/'reports/report.md').write_text('corrupt')
                with self.assertRaisesRegex(ValueError, 'report hash drift'): t.validation_gate(cfg)

    def test_failed_or_foreign_gate_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cfg = self.gate_fixture(root, .1)
            for status, identity in [('failed', cfg['validation_dependency']['job_sha256']), ('complete', 'foreign')]:
                t.atomic_json(root/'state.json', {'status': status, 'job_sha256': identity})
                with self.assertRaises(ValueError): t.validation_gate(cfg)

    def test_adapter_exact_parity_with_ours_disabled(self):
        anatomy = synthetic(); fs = findings(PROMPTS)
        raw = (np.random.default_rng(17).random((len(fs), *anatomy.shape)) > .5).astype(np.uint8)
        for affine in (np.diag([-2., -1.25, 3., 1.]), np.array([[0, -1.25, 0, 30], [2., 0, 0, -15], [0, 0, -3., 80], [0, 0, 0, 1.]])):
            expected, _ = t.methods.postprocess(raw, anatomy, affine, fs,
                         [t.methods.route_prompt(f['prompt'], f['category']) for f in fs])
            actual, routing = t.postprocess(raw, anatomy, affine, fs)
            for v in t.VARIANTS: np.testing.assert_array_equal(actual[v], expected[v])
            self.assertTrue(np.all(actual['d12'] <= actual['d11']))
            self.assertTrue(np.all(actual['d11'] <= raw))
            for route in routing:
                if not route['v2_selected']:
                    i = route['finding_idx']; np.testing.assert_array_equal(actual['d11'][i], actual['d12'][i])

    def test_case_hash_geometry_and_binary_failures(self):
        for issue in ('hash', 'geometry', 'binary'):
            with tempfile.TemporaryDirectory() as d:
                cfg, job = self.fixture(Path(d)); case = job['cases'][0]
                if issue == 'hash':
                    Path(case['anatomy']['path']).write_bytes(b'corrupt')
                elif issue == 'geometry':
                    case['ct_header_sha256'] = 'wrong'
                else:
                    img = nib.load(case['input']['path']); arr = np.asanyarray(img.dataobj).copy(); arr[0, 0, 0, 0] = 2
                    nib.save(nib.Nifti1Image(arr, img.affine), case['input']['path'])
                    case['input']['sha256'] = t.sha(case['input']['path'])
                with self.assertRaises(ValueError): t.case_worker(case)

    def test_interrupted_pair_publication_and_foreign_output(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, job = self.fixture(Path(d)); case = job['cases'][0]
            record = t.case_worker(case)
            journal = Path(cfg['runtime_root'])/'cases'/(case['name']+'.json')
            record['status'] = 'prepared'
            dest, temp = t.paths(cfg, case, 'd12'); dest.replace(temp)
            t.atomic_json(journal, record)
            self.assertEqual(t.recover_case(cfg, job, case)['status'], 'complete')
            self.assertTrue(dest.exists()); self.assertFalse(temp.exists())
            dest.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): t.recover_case(cfg, job, case)
            journal.unlink()
            with self.assertRaisesRegex(ValueError, 'unowned'): t.recover_case(cfg, job, case)

    def test_space_boundary_and_duplicate_owner(self):
        cfg = {'storage': {'reserve_bytes': 100, 'headroom_bytes': 20}}
        self.assertFalse(t.enough_space(cfg, 119)); self.assertTrue(t.enough_space(cfg, 120))
        with tempfile.TemporaryDirectory() as d:
            with t.job_lock(d):
                with self.assertRaisesRegex(ValueError, 'another live'):
                    with t.job_lock(d): pass

    def test_full_300_case_report_restart_and_no_zip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cfg, job = self.fixture(root, 300)
            availability = itertools.chain([False, False], itertools.repeat(True))
            with patch.object(t, 'validation_gate', return_value={'score_requirement': 'none'}), \
                 patch.object(t, 'enough_space', side_effect=lambda _cfg: next(availability)), \
                 patch.object(t, 'state', wraps=t.state) as observed:
                t.run(cfg)
                before = t.sha(Path(cfg['runtime_root'])/'completion.json')
                t.run(cfg)
            self.assertIn('waiting_for_space', [call.args[2] for call in observed.call_args_list])
            self.assertEqual(t.sha(Path(cfg['runtime_root'])/'completion.json'), before)
            done = t.read(Path(cfg['runtime_root'])/'completion.json')
            self.assertEqual((done['files'], done['findings'], done['archives']), (600, 582, {}))
            self.assertFalse(list(root.rglob('*.zip')))
            self.assertEqual(len(t.read(Path(cfg['runtime_root'])/'reports/findings.json')), 582)

    def test_launcher_has_no_labels_or_gpu_and_uses_exact_output_mounts(self):
        cfg = t.read(t.DEFAULT_CONFIG)
        argv = launch.command(cfg, 'run', detached=True)
        self.assertNotIn('--gpus', argv)
        self.assertFalse(any('segmentations' in x for x in argv))
        for variant in t.VARIANTS:
            path = cfg['output_root']+'/'+variant
            self.assertIn(f'type=bind,src={path},dst={path}', argv)
        self.assertIn('NUMPY_MADVISE_HUGEPAGE=0', argv)


if __name__ == '__main__':
    unittest.main()
