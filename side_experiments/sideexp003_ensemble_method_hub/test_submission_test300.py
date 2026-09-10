"""Numerical, geometry, publication, ownership and dependency regressions."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import nibabel as nib
import numpy as np
import submission_test300 as s


class SubmissionTests(unittest.TestCase):
    def test_probability_reference_mixed_precision_chunks_and_tie(self):
        rng = np.random.default_rng(41)
        arrays = [rng.normal(size=(3, 4, 7)).astype(np.float16 if i % 2 else np.float32)
                  for i in range(4)]
        for a in arrays:
            a.reshape(-1)[0] = 0
        ref = np.zeros(arrays[0].shape, dtype=np.float32)
        for a in arrays:
            ref += s.sigmoid_float32(a)
        expected_hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
        for chunk in (1, 13, 84, 100):
            actual, _ = s.average_arrays(arrays, chunk, expected_hashes)
            np.testing.assert_array_equal(actual, ref >= 2)
            self.assertEqual(actual.reshape(-1)[0], 1)

    def test_probability_is_not_logit_average(self):
        arrays = [np.array([x], dtype=np.float32) for x in (10, -1, -1, -1)]
        actual, _ = s.average_arrays(arrays)
        self.assertEqual(actual[0], 0)
        self.assertGreater(sum(a[0] for a in arrays), 0)

    def test_source_hash_and_nonfinite_failures(self):
        arrays = [np.ones(7, dtype=np.float32) for _ in range(4)]
        with self.assertRaisesRegex(Exception, 'source array hash'):
            s.average_arrays(arrays, 3, ['bad'] * 4)
        arrays[0][0] = np.nan
        with self.assertRaisesRegex(Exception, 'invalid logits'):
            s.average_arrays(arrays)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'source'; p.write_text('original')
            rec = s.identity(p); p.write_text('drift')
            with self.assertRaisesRegex(Exception, 'source hash'):
                s.verify_identity(rec)

    def test_numeric_prompt_order(self):
        c = {'findings': {str(i): 'prompt' + str(i) for i in reversed(range(12))}}
        result = s.tc.prompt_contract(c)
        self.assertEqual(result['finding_indices'], list(range(12)))
        self.assertEqual(result['prompt_sha256s'][10], hashlib.sha256(b'prompt10').hexdigest())

    def test_postprocessing_parity_native_headers_and_support_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); name = 'train_1_a_1.nii.gz'
            ct_path = root / 'dataset/train_fixed/train_1/train_1_a' / name
            ct_path.parent.mkdir(parents=True)
            affine = np.diag([-3., -3., 8., 1.])
            shape = (23, 22, 17)
            ct = nib.Nifti1Image(np.zeros(shape, np.int16), affine)
            nib.save(ct, ct_path)
            anatomy = np.zeros(shape, np.uint8)
            anatomy[7, 8, 8] = 10; anatomy[14, 13, 8] = 12; anatomy[13, 12, 10] = 14
            mask = root / 'anatomy.nii.gz'; nib.save(nib.Nifti1Image(anatomy, affine), mask)
            prompts = [('Bilateral ground glass opacities greater on the right', '2c'),
                       ('Nodule in right upper lobe and right lower lobe', '2d'),
                       ('Pleural effusion on right', '2e'), ('Diffuse ground glass opacity', '2c')]
            routes = [s.anatomy_policy.route_prompt(p, cat) for p, cat in prompts]
            # Explicit eligible-but-empty labels must remain unchanged, as in transform_case_dir.
            routes.append({'eligible': False, 'fine_eligible': True, 'selected_labels': []})
            entry = {'name': name, 'findings': {str(i): str(i) for i in range(len(routes))}}
            raw = np.ones((len(routes), *shape), dtype=np.uint8)
            raw[:, 0, 0, 0] = 0
            src = root / 'raw'; src.mkdir()
            nib.save(nib.Nifti1Image(raw, affine), src / name)
            reference = []
            with patch.object(s.anatomy_policy, 'CT_ROOT', root):
                for policy in ('whole_lung', 'fine'):
                    dst = root / policy
                    s.anatomy_policy.transform_case_dir(src, dst,
                        {(name, i): r for i, r in enumerate(routes)}, policy,
                        {name: mask}, {'test': [entry]}, 'test')
                    reference.append(np.asanyarray(nib.load(dst / name).dataobj))
            with patch.object(s.anatomy_policy, 'physical_dilation',
                              wraps=s.anatomy_policy.physical_dilation) as dilate:
                result = s.postprocess(raw, anatomy, affine, routes)
            for a, b in zip(result, reference):
                np.testing.assert_array_equal(a, b)
            # Whole lung and bilateral fine share the same label support.
            self.assertEqual(dilate.call_count, 2)
            np.testing.assert_array_equal(result[1][-1], raw[-1])

    def test_dependency_gate_and_foreign_or_failed_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); job = {'parent_runtime': d, 'parent_job_sha256': 'parent'}
            self.assertFalse(s.gate(job)[0])
            state = {'job_spec_sha256': 'parent', 'status': 'finishing_wave1_only'}
            s.write(root / 'finish_wave1_state.json', state)
            self.assertFalse(s.gate(job)[0])
            state.update(status='stopped_after_wave1', coordinator_exited=True,
                         coordinator_pid=2147483647, held_ranks=list(range(5, 21)))
            s.write(root / 'finish_wave1_state.json', state)
            s.write(root / 'control/finish_wave1_request.json', {'coordinator': {'start_ticks': '1'}})
            self.assertTrue(s.gate(job)[0])
            state['coordinator_exited'] = False
            s.write(root / 'finish_wave1_state.json', state)
            with self.assertRaisesRegex(Exception, 'exit unconfirmed'):
                s.gate(job)
            s.write(root / 'finish_wave1_error.json', {'error': 'guard failed'})
            with self.assertRaisesRegex(Exception, 'guard failed'):
                s.gate(job)

    def test_duplicate_owner(self):
        with tempfile.TemporaryDirectory() as d:
            with s.tc.exclusive(Path(d) / 'lock'):
                with self.assertRaisesRegex(Exception, 'live owner'):
                    with s.tc.exclusive(Path(d) / 'lock'):
                        pass

    def test_interrupted_transaction_and_conflicting_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); rows = []
            for i in range(3):
                tmp = root / f'{i}.tmp'; tmp.write_bytes(bytes([i]) * 20)
                rows.append({'path': str(root / str(i)), 'temporary_path': str(tmp),
                             'sha256': s.sha(tmp)})
            tx = {'status': 'prepared', 'outputs': rows}
            s.write(root / 'tx.json', tx)
            # Simulate interruption after the first destination rename.
            Path(rows[0]['temporary_path']).rename(rows[0]['path'])
            result = s.publish_transaction(root / 'tx.json', tx)
            self.assertEqual(result['status'], 'complete')
            s.publish_transaction(root / 'tx.json', s.read(root / 'tx.json'))
            Path(rows[1]['path']).write_text('foreign')
            with self.assertRaisesRegex(Exception, 'conflicting'):
                s.publish_transaction(root / 'tx.json', tx)

    def test_zip_root_crc_hashes_and_restart(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); folder = root / 'd1'; folder.mkdir(); runtime = root / 'runtime'; runtime.mkdir()
            rows = []
            for i in range(300):
                p = folder / f'case_{i:03}.nii.gz'; p.write_bytes(b'compressed fixture' + bytes([i % 256]))
                rows.append({'path': str(p), 'sha256': s.sha(p)})
            result = s.package(root, 'd1', rows, runtime, 'job')
            with zipfile.ZipFile(result['path']) as z:
                self.assertEqual(len(z.namelist()), 300)
                self.assertTrue(all('/' not in name for name in z.namelist()))
                self.assertIsNone(z.testzip())
            self.assertEqual(s.package(root, 'd1', rows, runtime, 'job')['sha256'], result['sha256'])
            with self.assertRaisesRegex(Exception, 'foreign ZIP'):
                s.package(root, 'd1', rows, runtime, 'another')

    def test_space_reserve(self):
        job = {'output_root': '/tmp', 'minimum_shared_free_bytes': 20 * 1024**4,
               'headroom_bytes': 350 * 1024**3}
        threshold = job['minimum_shared_free_bytes'] + job['headroom_bytes']
        for free, expected in ((threshold - 1, False), (threshold, True)):
            with patch.object(s.shutil, 'disk_usage') as disk:
                disk.return_value.free = free
                self.assertEqual(s.enough_space(job), expected)

    def test_completed_cache_binding_and_full_runner(self):
        # Exercise the real four-process controller through all 900 publications
        # and ZIP closeout on a tiny 300-case/582-prompt cohort.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); runtime = root / 'runtime'; runtime.mkdir()
            out = root / 'outputs'; out.mkdir()
            affine = np.eye(4); shape = [2, 3, 4]
            ct_path = root / 'ct.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros(shape, np.int16), affine), ct_path)
            ct = nib.load(ct_path)
            mask = root / 'anatomy.nii.gz'
            nib.save(nib.Nifti1Image(np.full(shape, 10, np.uint8), affine), mask)
            cases = []
            for i in range(300):
                count = 2 if i < 282 else 1
                case = {'name': f'train_{i}_a_1.nii.gz', 'shape': shape,
                        'findings': {str(j): 'left opacity' for j in range(count)},
                        'ct_path': str(ct_path), 'affine': affine.tolist(),
                        'ct_header_sha256': hashlib.sha256(ct.header.binaryblock).hexdigest(),
                        'anatomy': s.identity(mask),
                        'routes': [s.anatomy_policy.route_prompt('left opacity', '2b')] * count}
                case.update(s.tc.prompt_contract(case)); cases.append(case)
            job = {'job_id': s.JOB_ID, 'job_sha256': 'job', 'runtime_root': str(runtime),
                   'output_root': str(out), 'parent_job_sha256': 'parent',
                   'dataset': {'cases': 300, 'findings': 582}, 'candidates': [], 'cases': cases,
                   'chunk_elements': 7, 'smoke_cases': [cases[0]['name']],
                   'anatomy_review': {'status': 'fixture'}, 'method': {'domain': 'post_sigmoid_probability'}}
            for rank in range(1, 5):
                cache = root / f'cache{rank}'; (cache / 'cases').mkdir(parents=True)
                c = {'rank': rank, 'cache_key': str(rank), 'cache_root': str(cache),
                     'id': str(rank), 'checkpoint': {'sha256': str(rank)}}
                job['candidates'].append(c)
                rows = []
                for case in cases:
                    a = np.ones((len(case['findings']), *shape), dtype=np.float16 if rank % 2 else np.float32)
                    p = cache / 'cases' / (case['name'] + '.npy'); np.save(p, a)
                    rows.append({'name': case['name'], 'array_path': str(p), 'npy_bytes': p.stat().st_size,
                                 'array_sha256': s.tc.array_sha256(a), 'dtype': str(a.dtype),
                                 'shape': list(a.shape), 'affine': affine.tolist(),
                                 'finding_indices': case['finding_indices'],
                                 'prompt_sha256s': case['prompt_sha256s'], 'cache_key': str(rank),
                                 'job_spec_sha256': 'parent', 'storage_reproduction_status': 'passed',
                                 'same_pass_mask_mismatch_voxels': 0})
                s.write(cache / 'export_manifest.json', {'status': 'strict_passed',
                        'job_spec_sha256': 'parent', 'cache_key': str(rank), 'candidate_id': str(rank),
                        'dataset': job['dataset'], 'candidate_source': c, 'cases': rows})
                s.write(cache / 'reproduction_validation.json', {'status': 'passed', 'cases': 300,
                        'findings': 582, 'job_spec_sha256': 'parent', 'array_hashes_verified': True,
                        'same_pass_mask_mismatch_voxels': 0})
            records = s.bind_caches(job)
            self.assertEqual(len(records), 4)
            vpath = root / 'cache1/reproduction_validation.json'
            original = s.read(vpath); changed = dict(original, findings=581); s.write(vpath, changed)
            with self.assertRaisesRegex(Exception, 'invalid cache validation'):
                s.bind_caches(job)
            s.write(vpath, original)
            with patch.object(s, 'validate_job'), patch.object(s, 'gate', return_value=(True, 'fixture')), \
                 patch.object(s, 'enough_space', return_value=True):
                s.run(job)
            completion = s.read(runtime / 'completion.json')
            self.assertEqual(completion['status'], 'complete')
            self.assertEqual(len(completion['archives']), 3)
            for variant in s.VARIANTS:
                self.assertEqual(len(list((out / variant).iterdir())), 300)
                self.assertEqual(s.read(runtime / (variant + '.manifest.json'))['findings'], 582)
            # A new valid but different completion manifest must not replace the binding.
            changed = dict(original, warning='different validation record'); s.write(vpath, changed)
            with self.assertRaisesRegex(Exception, 'manifest changed'):
                s.bind_caches(job)

    def test_label_free_case_export_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); runtime = root / 'runtime'; runtime.mkdir()
            for v in s.VARIANTS:
                (root / v).mkdir()
            shape = (5, 6, 7); affine = np.diag([-1., -2., 3., 1.])
            ct_path = root / 'ct.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros(shape, np.int16), affine), ct_path)
            ct = nib.load(ct_path)
            mask = root / 'anatomy.nii.gz'
            nib.save(nib.Nifti1Image(np.full(shape, 10, np.uint8), affine), mask)
            case = {'name': 'train_1_a_1.nii.gz', 'shape': list(shape), 'findings': {'0': 'left opacity'},
                    'finding_indices': [0], 'prompt_sha256s': ['prompt'], 'ct_path': str(ct_path),
                    'ct_header_sha256': hashlib.sha256(ct.header.binaryblock).hexdigest(),
                    'affine': affine.tolist(), 'anatomy': s.identity(mask),
                    'routes': [s.anatomy_policy.route_prompt('left opacity', '2b')]}
            records = []
            for i in range(4):
                a = np.ones((1, *shape), dtype=np.float16 if i % 2 else np.float32)
                p = root / f'{i}.npy'; np.save(p, a)
                records.append({'array_path': str(p), 'array_sha256': s.tc.array_sha256(a),
                                'shape': list(a.shape), 'dtype': str(a.dtype)})
            job = {'runtime_root': str(runtime), 'output_root': str(root), 'job_sha256': 'job',
                   'job_id': s.JOB_ID, 'chunk_elements': 23}
            tx = s.process_case(job, case, records)
            self.assertEqual(tx['status'], 'complete')
            for item in tx['outputs']:
                s.validate_prediction(Path(item['path']), case, np.ones((1, *shape), np.uint8))
            with patch.object(s, 'average_arrays', side_effect=AssertionError('recomputed')):
                again = s.process_case(job, case, records)
            self.assertEqual([r['sha256'] for r in again['outputs']], [r['sha256'] for r in tx['outputs']])


if __name__ == '__main__':
    unittest.main()
