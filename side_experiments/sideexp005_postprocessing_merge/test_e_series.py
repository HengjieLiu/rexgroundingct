"""Eight-model computation, publication, reporting and automatic gate contracts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import e_series as coordinator
import e_series_compute as e
import runner as core

e.scientific()
np, nib, methods = e.np, e.nib, e.methods


def ref(path):
    return {'path': str(path), 'sha256': core.sha(path)}


class ESeriesTests(unittest.TestCase):
    def test_eight_probability_reference_ties_precision_and_hashes(self):
        from preliminary_ensemble import generate_probability_ensembles
        arrays = [np.array([4 if i % 4 == 0 else -1, 0, -8, 1, 5, -2, .1],
                           dtype=np.float16 if i % 2 else np.float32) for i in range(8)]
        probs = {str(i): methods.sigmoid_float32(a) for i,a in enumerate(arrays)}
        expected, _ = generate_probability_ensembles(probs, {'uniform': {str(i): 1 for i in range(8)}})['uniform']
        hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
        for chunk in (1, 3, 7, 8, 4194304):
            out, _ = e.average_arrays(arrays, hashes, chunk)
            np.testing.assert_array_equal(out, expected >= 4)
            self.assertEqual(out[1], 1)
            self.assertNotEqual(out[0], int(sum(a[0] for a in arrays) >= 0))
        with self.assertRaisesRegex(ValueError, 'source array hash'): e.average_arrays(arrays, ['0'*64]*8)
        arrays[0][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'): e.average_arrays(arrays)

    def fixture(self, root, stage, count):
        runtime = root/stage; runtime.mkdir()
        affine = np.diag([-2., 3., -4., 1.]); shape = (8, 9, 10)
        anatomy = np.zeros(shape, dtype=np.uint8)
        for i in range(5): anatomy[1+i, 2:7, 2:8] = 10+i
        ct = root/'ct.nii.gz'; mask = root/'anatomy.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine), ct)
        nib.save(nib.Nifti1Image(anatomy, affine), mask)
        examples = {}
        for n in (1, 2):
            fs = [{'finding_idx': i, 'prompt': 'subpleural nodule in the right upper lobe' if i == 0 else 'right pleural effusion',
                   'category': '2d' if i == 0 else '2e'} for i in range(n)]
            sources = []
            for i in range(8):
                a = np.full((n, *shape), 0., dtype=np.float16 if i % 2 else np.float32)
                a[..., 0, :, :] = -1
                path = root/f'logit_{n}_{i}.npy'; np.save(path, a)
                sources.append({'path': str(path), 'dtype': a.dtype.name, 'sha256': hashlib.sha256(a.tobytes()).hexdigest()})
            case = {'shape': [n, *shape], 'affine': affine.tolist(), 'ct_path': str(ct), 'anatomy': ref(mask),
                    'findings': fs, 'routes': [methods.route_prompt(f['prompt'], f['category']) for f in fs], 'sources': sources}
            if stage == 'val200':
                gt = root/f'gt{n}.nii.gz'; nib.save(nib.Nifti1Image(np.ones((n,*shape),dtype=np.uint8), affine), gt)
                case['gt'] = ref(gt)
            examples[n] = case
        cases = [{'name': f'case{i:03}.nii.gz', **copy.deepcopy(examples[2 if i < (181 if stage == 'val200' else 282) else 1])} for i in range(count)]
        cfg = {'runtime_root': str(runtime), 'output_root': str(runtime/'predictions'), 'stage': stage,
               'run_id': 'fixture_'+stage, 'workers': 4, 'chunk_elements': 137, 'heartbeat_seconds': 60,
               'reserve_bytes': 0, 'headroom_bytes': 0, 'source_status': 'fixture', 'inputs': {},
               'expected': {'cases': count, 'findings': sum(len(c['findings']) for c in cases)}}
        job = {'job_sha256': 'a'*64, 'cases': cases, 'smoke_cases': [cases[0]['name']], 'candidates': []}
        core.atomic_json(runtime/'job_manifest.json', job)
        e.worker_init(cfg, job)
        return cfg, job

    def test_case_method_parity_geometry_and_interrupted_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, job = self.fixture(Path(temp), 'test300', 1); case = job['cases'][0]
            real_replace = e.os.replace
            def interrupted(src, dst):
                if str(dst).endswith('.nii.gz'): raise InterruptedError('fixture publication interruption')
                return real_replace(src, dst)
            with patch.object(e.os, 'replace', side_effect=interrupted):
                with self.assertRaises(InterruptedError): e.case_worker(('postprocessing', case))
            rec = e.recover(cfg, job, case, 'postprocessing')
            raw = np.asanyarray(nib.load(rec['artifacts']['e1']['path']).dataobj)
            a = np.asanyarray(nib.load(case['anatomy']['path']).dataobj)
            reference, details = methods.postprocess(raw, a, np.array(case['affine']), case['findings'], case['routes'])
            for v, mask in reference.items():
                actual = np.asanyarray(nib.load(rec['artifacts']['e'+v[1:]]['path']).dataobj)
                np.testing.assert_array_equal(actual, mask)
            self.assertEqual(e.case_worker(('postprocessing', case)), rec)
            Path(rec['artifacts']['e12']['path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash drift'): e.recover(cfg, job, case, 'postprocessing')

    def test_full_val_report_uses_paired_d_evidence_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, job = self.fixture(Path(temp), 'val200', 200)
            records = [e.case_worker((phase, case)) for phase in ('baseline','postprocessing') for case in job['cases']]
            base = [r for rec in records if rec['phase'] == 'baseline' for r in rec['rows']]
            metrics = core.metric_summary(base)
            reference = Path(temp)/'reference.json'; core.atomic_json(reference, {'metrics': metrics})
            cfg['baseline'] = {'dice': metrics['dice'], 'hits': metrics['hits'], 'tolerance': 1e-12}
            cfg['inputs']['baseline_result'] = ref(reference)
            drows = [dict(row, variant='d'+row['variant'][1:]) for rec in records for row in rec['rows']]
            dr = Path(temp)/'drows.json'; core.atomic_json(dr, drows); cfg['inputs']['d_rows'] = ref(dr)
            dm = {v: core.metric_summary([r for r in drows if r['variant'] == v]) for v in methods.VARIANTS}
            dc = Path(temp)/'dcomplete.json'; core.atomic_json(dc, {'metrics': dm}); cfg['inputs']['d_completion'] = ref(dc)
            for phase in ('baseline','postprocessing'): core.atomic_json(Path(cfg['runtime_root'])/(phase+'_timings.json'), {'wall_seconds': 2})
            with patch.object(e, 'verify_job'):
                done = e.report(cfg, job)
                self.assertEqual(done, e.report(cfg, job))
            self.assertEqual(done['files'], 1000); self.assertEqual(len(done['metrics']), 10)
            self.assertEqual(len(done['comparisons']), 10)
            self.assertIsNone(done['metrics']['e12']['categories']['2f']['dice'])
            self.assertEqual(coordinator.completion(cfg, job, True), done)
            for m in done['comparisons'].values():
                self.assertEqual(sum(m['overall'][k] for k in ('improved','decreased','unchanged')), 381)
            cfg['baseline']['dice'] += .01
            with self.assertRaisesRegex(ValueError, 'anchor mismatch'): core.baseline_gate(cfg, [r for r in records if r['phase'] == 'baseline'])

    def test_full_test_folders_without_labels_or_zips(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, job = self.fixture(Path(temp), 'test300', 300)
            self.assertTrue(all('gt' not in c for c in job['cases']))
            for case in job['cases']: e.case_worker(('postprocessing', case))
            core.atomic_json(Path(cfg['runtime_root'])/'postprocessing_timings.json', {'wall_seconds': 2})
            with patch.object(e, 'verify_job'): done = e.report(cfg, job)
            self.assertEqual((done['files'], done['findings']), (1500, 582))
            self.assertIsNone(done['metrics']); self.assertEqual(done['archives'], {})
            self.assertEqual(coordinator.completion(cfg, job, True), done)
            self.assertFalse(list(Path(temp).rglob('*.zip')))

    def test_cross_family_comparison_does_not_assume_subsets(self):
        b = {'variant': 'd1', 'case': 'c', 'finding_index': 0, 'category': '2d', 'prompt': 'nodule',
             'dice': .09, 'hit': False, 'gt_voxels': 10, 'pred_voxels': 1, 'intersection_voxels': 1}
        a = dict(b, variant='e1', dice=.7, hit=True, pred_voxels=20, intersection_voxels=8)
        table, _ = e.paired_changes([a,b], 'e1', 'd1')
        self.assertEqual(table['overall']['improved'], 1); self.assertEqual(table['overall']['hit_gain'], 1)
        with self.assertRaises(ValueError): e.paired_changes([a,b], 'e1', 'd1', subset=True)

    def test_ownership_space_and_worker_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, job = self.fixture(Path(temp), 'val200', 2)
            with core.job_lock(cfg['runtime_root']):
                with self.assertRaises(ValueError):
                    with core.job_lock(cfg['runtime_root']): pass
            with patch.object(e, 'enough_space', side_effect=[False]+[True]*30), patch.object(e.time, 'sleep'):
                first = e.run_phase(cfg, job, 'baseline')
            second = e.run_phase(cfg, job, 'baseline')
            self.assertEqual(first, second)
            self.assertFalse(e.enough_space({'reserve_bytes': 20, 'headroom_bytes': 3}, free=22))
            self.assertTrue(e.enough_space({'reserve_bytes': 20, 'headroom_bytes': 3}, free=23))

    def test_wave2_ready_while_wave3_runs_and_bad_workers_block(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); master = core.read(coordinator.DEFAULT)
            wave = {'job_id': 'fixture', 'job_spec_sha256': 'b'*64, 'runtime_root': str(root)}
            path = root/'wave.json'; core.atomic_json(path, wave); master['inputs']['wave2_job'] = ref(path)
            core.atomic_json(root/'orchestrator_state.json', {'status': 'exporting', 'wave': 3})
            candidates = []
            for rank in range(1,9):
                progress = root/f'{rank}.json'; core.atomic_json(progress, {'job_spec_sha256': 'b'*64, 'status': 'strict_passed', 'cases': 300})
                candidates.append({'rank': rank, 'test_source': {'gpu': (rank-1)%4, 'progress_path': str(progress)}})
            job = {'candidates': candidates}
            info = {'Id': 'container', 'State': {'Running': False, 'ExitCode': 0}, 'Config': {'Labels': {'rex.test_job': 'b'*64}}}
            with patch.object(coordinator, 'validate_cache', side_effect=lambda cfg,j,c: {'rank': c['rank']}), patch.object(coordinator, 'docker_info', return_value=info):
                self.assertEqual(coordinator.wave2_status(master, job)['status'], 'ready')
                info['State']['Running'] = True
                self.assertEqual(coordinator.wave2_status(master, job)['status'], 'waiting')
                info['State'].update(Running=False, ExitCode=1)
                self.assertEqual(coordinator.wave2_status(master, job)['status'], 'blocked')
            with patch.object(coordinator, 'validate_cache', return_value=None), patch.object(coordinator, 'docker_info', return_value=None):
                self.assertEqual(coordinator.wave2_status(master, job)['status'], 'waiting')
            with patch.object(coordinator, 'validate_cache', side_effect=ValueError('publication hash failure')), patch.object(coordinator, 'docker_info', return_value=None):
                self.assertEqual(coordinator.wave2_status(master, job)['status'], 'blocked')

    def test_completion_gate_ignores_score_quality_and_test_mount_excludes_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            master = core.read(coordinator.DEFAULT)
            for stage in master['runs']: master['runs'][stage]['runtime_root'] = str(Path(temp)/stage)
            val = Path(master['runs']['val200']['runtime_root']); val.mkdir()
            jobs = {s: {'job_sha256': s*8} for s in master['runs']}
            snapshot = {'status': 'ready', 'bindings': [{'rank': i} for i in range(1,9)],
                        'ranks': [{'rank': i, 'container_id': str(i), 'worker_exited_successfully': True} for i in range(5,9)]}
            for score in (0., 1.):
                core.atomic_json(val/'completion.json', {'dice': score})
                binding = Path(master['runs']['test300']['runtime_root'])/'test_cache_binding.json'
                if binding.exists(): binding.unlink()
                with patch.object(coordinator, 'completion', return_value={'dice': score}): coordinator.bind_test(master, jobs, snapshot)
                self.assertEqual(core.read(binding)['validation_score_requirement'], 'none')
            with patch.object(coordinator, 'completion', return_value=None):
                with self.assertRaisesRegex(ValueError, 'unavailable'): coordinator.bind_test(master, jobs, snapshot)
            cmd = coordinator.docker_command(master, 'test300', 'run', True)
            self.assertFalse(any('segmentations' in a for a in cmd)); self.assertNotIn('--gpus', cmd)
            self.assertIn('NUMPY_MADVISE_HUGEPAGE=0', cmd)

    def test_strict_cache_publication_prompt_order_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cases = []
            for i in range(300):
                cases.append({'name': f'c{i}.nii.gz', 'shape': [2 if i < 282 else 1, 2, 3, 4], 'affine': np.eye(4).tolist(),
                              'findings': [{'finding_idx': k, 'prompt': str(k)} for k in range(2 if i < 282 else 1)]})
            source = {'cache_root': str(root), 'cache_key': 'v4_fixture', 'checkpoint': {'sha256': 'ckpt'},
                      'paired_val200': {'cache_key': 'v2_fixture'}, 'cache': {'id': 'native'}}
            candidate = {'test_source': source, 'rank': 5, 'test_job_sha256': 'job', 'id': 'model', 'checkpoint': source['checkpoint']}
            cfg = {'inputs': {'test_dataset': {'sha256': 'dataset'}}}; job = {'cases': cases}
            self.assertIsNone(coordinator.validate_cache(cfg, job, candidate))
            exported = {'status': 'strict_passed', 'job_spec_sha256': 'job', 'candidate_id': 'model', 'cache_key': 'v4_fixture',
                        'candidate_source': source, 'dataset': {'sha256': 'dataset'}, 'paired_val200': source['paired_val200'],
                        'case_count': 300, 'finding_count': 582, 'cases': []}
            for c in cases:
                exported['cases'].append({**c, 'finding_indices': [f['finding_idx'] for f in c['findings']],
                    'prompt_sha256s': [hashlib.sha256(f['prompt'].encode()).hexdigest() for f in c['findings']],
                    'dtype': 'float16', 'cache_key': 'v4_fixture', 'job_spec_sha256': 'job', 'storage_reproduction_status': 'passed',
                    'same_pass_mask_mismatch_voxels': 0, 'array_path': str(root/'cases'/(c['name']+'.npy'))})
            validation = {'status': 'passed', 'storage_reproduction_status': 'passed', 'job_spec_sha256': 'job', 'cases': 300,
                          'findings': 582, 'array_hashes_verified': True, 'same_pass_mask_mismatch_voxels': 0}
            core.atomic_json(root/'export_manifest.json', exported)
            self.assertIsNone(coordinator.validate_cache(cfg, job, candidate))
            core.atomic_json(root/'reproduction_validation.json', validation)
            self.assertEqual(coordinator.validate_cache(cfg, job, candidate)['rank'], 5)
            for kind in ('order','checkpoint','preprocessing','validation'):
                bad = copy.deepcopy(exported); check = copy.deepcopy(validation)
                if kind == 'order': bad['cases'][0]['finding_indices'] = [1,0]
                if kind == 'checkpoint': bad['candidate_source']['checkpoint']['sha256'] = 'foreign'
                if kind == 'preprocessing': bad['candidate_source']['cache']['id'] = 'foreign'
                if kind == 'validation': check['same_pass_mask_mismatch_voxels'] = 1
                core.atomic_json(root/'export_manifest.json', bad); core.atomic_json(root/'reproduction_validation.json', check)
                with self.assertRaises(ValueError): coordinator.validate_cache(cfg, job, candidate)

    def test_supervisor_dispatches_both_stages_and_records_immediate_poll(self):
        with tempfile.TemporaryDirectory() as temp:
            master = core.read(coordinator.DEFAULT)
            for stage in master['runs']:
                root = Path(temp)/stage; master['runs'][stage]['runtime_root'] = str(root)
                core.atomic_json(root/'job_manifest.json', {'job_sha256': stage, 'cases': []})
            snapshot = {'status': 'ready', 'bindings': [], 'ranks': []}
            def collect(m, stage, job): core.atomic_json(Path(m['runs'][stage]['runtime_root'])/'closeout.json', {'status': 'complete'})
            with patch.object(e, 'verify_job'), patch.object(coordinator, 'wave2_status', return_value=snapshot) as poll, \
                 patch.object(coordinator, 'launch_stage', side_effect=lambda m,s,j: {'container_id': s}) as launch, \
                 patch.object(coordinator, 'docker_info', return_value={'State': {'Running': False, 'ExitCode': 0}}), \
                 patch.object(coordinator, 'collect_stage', side_effect=collect), patch.object(coordinator, 'bind_test') as bind, \
                 patch.object(coordinator, 'update_register'):
                coordinator.supervise(master)
            self.assertEqual(poll.call_count, 1); self.assertEqual(bind.call_count, 1)
            self.assertEqual([c.args[1] for c in launch.call_args_list], ['val200','test300'])
            root = Path(master['runs']['val200']['runtime_root'])/'automation'
            self.assertEqual(core.read(root/'state.json')['status'], 'complete')
            self.assertEqual(len((root/'wave2_history.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__': unittest.main()
