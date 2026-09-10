"""CPU checks for the approved full run, complete-cache gate and live records."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from exp027_common import (ARMS, DEFAULT_CONFIG, aggregate, append_update, atomic_json, digest,
                          materialize_schedule, read_json, read_updates, reconcile_updates, sha256)
from exp027_data import (cache_contract, cache_gate, cache_producer_fingerprint, case_key,
                        compatible_contracts, require_full_cache, required_cache_keys, verify_cache_case)
from exp027_report import collect, render, report_command
from run_027_residual_refinement import worker_environment, orchestration
from test_exp027_contract import fake_prepared


def cache_fixture(root):
    config = read_json(DEFAULT_CONFIG)
    config.pop('cache', None)
    config['preprocessing']['cache_root'] = str(root / 'native')
    records = []
    for i, split in enumerate(('train', 'val', 'val')):
        name = f'train_{i+1}_a_1.nii.gz'
        records.append({'name': name, 'key': name + '::4', 'finding_id': '4', 'channel': 0,
                        'prompt': 'linear opacity', 'split': split, 'half': 'A' if i == 1 else 'B', 'voxels': 8})
    prepared = {'train': records[:1], 'val': records[1:]}
    contract = cache_contract(config, prepared)
    for record in records:
        name = record['name']
        original = root / 'native/cases' / case_key(name)
        folder = root / 'cache' / case_key(name)
        original.mkdir(parents=True); folder.mkdir(parents=True)
        image = np.ones((1, 2, 2, 2), dtype=np.float32)
        np.save(original / 'image.npy', image)
        ct = {'preprocess_id': 'crop_zscore_native_v1', 'native_cropped_shape_zyx': [2]*3,
              'resampled_shape_zyx': [2]*3, 'prompts': [record['prompt']],
              'image_sha256': hashlib.sha256(str(image.dtype).encode('ascii') +
                                             json.dumps(list(image.shape)).encode('ascii') + image.tobytes()).hexdigest()}
        atomic_json(original / 'metadata.json', ct)
        np.save(folder / 'logits.npy', np.ones((1, 2, 2, 2), dtype=np.float16))
        np.save(folder / 'targets.npy', np.ones((1, 2, 2, 2), dtype=np.uint8))
        origin = {'dtype': 'float16', 'kind': 'frozen_exp007_inference' if record['split'] == 'train' else 'strict_validation_cache',
                  'checkpoint_sha256': config['base']['checkpoint_sha256'], 'same_pass_mask_mismatch_voxels': 0,
                  'array_sha256': 'source fixture'}
        atomic_json(folder / 'metadata.json', {'contract': contract, 'name': name, 'records': [record],
                    'ct_metadata': ct, 'image_path': str(original / 'image.npy'), 'origin': origin, 'shape': [2]*3,
                    'points': {record['key']: {'gt': [[1,1,1]], 'prediction': [[1,1,1]]}},
                    'hashes': {f: sha256(folder / f) for f in ('logits.npy', 'targets.npy')}})
    return config, prepared, {r['key']: r for r in records}


class FullRunTests(unittest.TestCase):
    def test_orchestration_cache_gate_precedes_all_ten_barriers(self):
        for fail_cache in (False, True):
            with self.subTest(fail_cache=fail_cache), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = read_json(DEFAULT_CONFIG)
                config.pop('cache', None)
                config['experiment_dir'] = str(root)
                source = root / 'input.json'; atomic_json(source, config)
                prepared = fake_prepared()
                stages = []
                @contextmanager
                def reporter(*_):
                    yield lambda: None
                def workers(commands, **kwargs):
                    task = commands[0][1][2]
                    stages.append(task)
                    self.assertEqual(len(commands), 4)
                    if task == 'cache':
                        partitions = []
                        for _, command, _ in commands:
                            partitions.append(read_json(command[command.index('--keys-file')+1]))
                        self.assertEqual(sorted(k for keys in partitions for k in keys),
                                         sorted(r['key'] for r in prepared['train']+prepared['val']))
                    for _, command, _ in commands:
                        if task == 'cache': continue
                        arm = command[command.index('--arm')+1]
                        folder = root / 'full' / arm
                        if task == 'train':
                            atomic_json(folder / 'training.json', {'initial_weights_sha256':'same', 'updates':[]})
                        else:
                            update = int(command[command.index('--update')+1])
                            atomic_json(folder / f'evaluations/update_{update:07d}/summary.json',
                                        {'metrics':{'full':{'findings':len(prepared['val'])}}, 'findings':prepared['val']})
                    return 0.
                def gate(*_):
                    if fail_cache: raise ValueError('incomplete cache fixture')
                    return {'status':'verified'}
                args = SimpleNamespace(dry_run=False, root=root, allow_gpu=True, config=source, gpus=list('0123'))
                with patch('exp027_data.prepare', return_value=prepared), \
                        patch('exp027_data.cache_gate', side_effect=gate), \
                        patch('run_027_residual_refinement.live_writer', reporter), \
                        patch('run_027_residual_refinement.parallel_workers', side_effect=workers):
                    if fail_cache:
                        with self.assertRaisesRegex(ValueError, 'incomplete cache'):
                            orchestration(config, args, benchmark=False)
                        self.assertEqual(stages, ['cache'])
                        self.assertEqual(read_json(root/'status.json')['status'], 'failed')
                    else:
                        orchestration(config, args, benchmark=False)
                        self.assertEqual(stages, ['cache']+['train','evaluate']*10)
                        self.assertEqual(read_json(root/'barriers.json')['completed'], list(range(1000,10001,1000)))
                        self.assertEqual(read_json(root/'status.json')['status'], 'pending_user_review')

    def test_full_cache_selection_includes_unsampled_findings(self):
        prepared = fake_prepared()
        events = {arm: [{'key': prepared['train'][0]['key']}] for arm in ARMS}
        full = required_cache_keys(prepared, events, False)
        benchmark = required_cache_keys(prepared, events, True)
        self.assertEqual(full, sorted(r['key'] for r in prepared['train'] + prepared['val']))
        self.assertGreater(len(full), len(benchmark))

    def test_immutable_full_schedule_and_source_mixture(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'schedule.json'
            prepared = fake_prepared()
            events = materialize_schedule(path, prepared, ARMS[1], 10000, 20260909)
            original = path.read_bytes()
            self.assertEqual(len(events), 10000)
            for i in range(0, 10000, 100):
                self.assertEqual(sum(r['source'] == 'train' for r in events[i:i+100]), 50)
            self.assertEqual(materialize_schedule(path, prepared, ARMS[1], 10000, 20260909), events)
            with self.assertRaisesRegex(ValueError, 'Immutable'):
                materialize_schedule(path, prepared, ARMS[1], 1000, 20260909)
            self.assertEqual(path.read_bytes(), original)

    def test_refiner_and_export_precision_environments_are_separate(self):
        with patch.dict(os.environ, {'NVIDIA_TF32_OVERRIDE': '1'}):
            for task in ('train', 'evaluate'):
                self.assertEqual(worker_environment(task, 2)['NVIDIA_TF32_OVERRIDE'], '0')
            self.assertNotIn('NVIDIA_TF32_OVERRIDE', worker_environment('cache', 2))
        config = read_json(DEFAULT_CONFIG)
        self.assertFalse(config['training']['amp'])
        self.assertEqual(config['training']['evaluation_updates'], list(range(1000, 10001, 1000)))

    def test_cache_identity_ignores_report_and_training_config(self):
        config, prepared = read_json(DEFAULT_CONFIG), fake_prepared()
        before = cache_contract(config, prepared)
        config['training']['amp'] = True
        config['report']['interval_seconds'] = 60
        config['optimizer']['lr'] = .123
        with patch('exp027_data.code_fingerprint', return_value={'exp027_report.py': 'changed'}):
            self.assertEqual(cache_contract(config, prepared), before)
        changed = cache_producer_fingerprint()
        changed['stored_logits'] = 'changed'
        with patch('exp027_data.cache_producer_fingerprint', return_value=changed):
            self.assertNotEqual(cache_contract(config, prepared), before)
        config['base']['checkpoint_sha256'] = 'different'
        self.assertNotEqual(cache_contract(config, prepared), before)

    def test_legacy_compatibility_is_input_bound_and_hash_verified(self):
        config, prepared = read_json(DEFAULT_CONFIG), fake_prepared()
        previous_code = {'fixture_producer.py': 'original hash'}
        legacy = digest({'base':config['base'], 'preprocessing':config['preprocessing'],
                         'prepared':digest(prepared), 'storage':'native_2a_v1', 'code':previous_code})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'compatibility.json'
            atomic_json(path, {'previous_code':previous_code, 'legacy_contract':legacy,
                              'current_contract':cache_contract(config,prepared)})
            config['cache']={'compatibility_manifest':{'path':str(path),'sha256':sha256(path)}}
            self.assertIn(legacy, compatible_contracts(config,prepared))
            config['report']['interval_seconds']=77
            self.assertIn(legacy, compatible_contracts(config,prepared))
            changed=copy.deepcopy(prepared); changed['train'][0]['prompt']='different'
            self.assertNotIn(legacy, compatible_contracts(config,changed))
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                compatible_contracts(config,prepared)

    def test_cache_verification_and_gate_reject_missing_changed_corrupt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, prepared, records = cache_fixture(root)
            contract = cache_contract(config, prepared)
            rows = [verify_cache_case(root / 'cache' / case_key(r['name']), r['name'], records, config, {contract})
                    for r in records.values()]
            receipt = {'cases': rows, 'prepared_sha256': digest(prepared), 'cache_contract': contract}
            path = root / 'cache_verified_gpu0.json'
            atomic_json(path, {**receipt, 'cases': rows[:-1]})
            with self.assertRaisesRegex(ValueError, 'coverage'):
                cache_gate(root, prepared, config, list(records))
            self.assertFalse((root / 'cache_inventory.json').exists())
            atomic_json(path, receipt)
            inventory = cache_gate(root, prepared, config, list(records))
            self.assertEqual((inventory['cases'], inventory['findings']), (3,3))
            require_full_cache(root, prepared, config)
            target = root / 'cache' / case_key(prepared['train'][0]['name']) / 'targets.npy'
            target.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'changed after verification'):
                cache_gate(root, prepared, config, list(records))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verify_cache_case(target.parent, prepared['train'][0]['name'], records, config, {contract})

    def test_cache_metadata_prompt_and_origin_mismatches_rejected(self):
        for field in ('records', 'origin'):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config, prepared, records = cache_fixture(root)
                name = prepared['train'][0]['name']; path = root / 'cache' / case_key(name) / 'metadata.json'
                meta = read_json(path)
                if field == 'records': meta['records'][0]['finding_id'] = '9'
                else: meta['origin']['checkpoint_sha256'] = 'wrong'
                atomic_json(path, meta)
                with self.assertRaises(ValueError):
                    verify_cache_case(path.parent, name, records, config, {cache_contract(config, prepared)})

    def test_journal_partial_write_recovery_no_duplicate_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'training.jsonl'
            rows = [{'update': i, 'loss': 1/i} for i in range(1,4)]
            for row in rows: append_update(path, row)
            with path.open('a') as f: f.write('{"update":4')
            self.assertEqual(read_updates(path), rows)
            reconcile_updates(path, rows[:2])
            self.assertEqual(len(list(path.parent.glob('recovered_*.jsonl'))), 1)
            append_update(path, rows[2])
            self.assertEqual(read_updates(path), rows)
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                reconcile_updates(path, [{'update':1,'loss':99}])

    def test_partial_scores_appear_before_peer_and_then_become_final(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_json(root / 'status.json', {'active_phase':'full', 'status':'evaluating', 'synthetic_fixture':True})
            arm = root / 'full' / ARMS[0]
            append_update(arm / 'training.jsonl', {'update':1,'loss':.8,'patch_dice':.4,'residual_mean_abs':.1})
            row = {'key':'finding','name':'ct','half':'A','dice':.4,'base_dice':.3,'hit':1,'base_hit':1}
            metrics = aggregate([row])
            out = arm / 'evaluations/update_0001000'
            atomic_json(out / 'partial.json', {'update':1000,'metrics':metrics,'expected_counts':{'A':35,'B':34,'full':69}})
            value = render(root)
            self.assertEqual(value['arms'][0]['partial']['metrics']['A']['dice'], .4)
            self.assertIsNone(value['arms'][0]['partial']['metrics']['B']['dice'])
            self.assertIsNone(value['arms'][1]['partial'])
            self.assertEqual(value['arms'][0]['training'][0]['loss'], .8)
            self.assertIn('1/35: 0.40000', (root / 'reports/live_dashboard.md').read_text())
            atomic_json(out / 'summary.json', {'update':1000,'metrics':metrics})
            value = collect(root)
            self.assertIsNone(value['arms'][0]['partial'])
            self.assertEqual(len(value['arms'][0]['evaluations']),1)
            self.assertFalse(value['arms'][1]['evaluations'])

    def test_reporter_observes_new_finding_without_waiting_for_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); stop = root / 'stop'
            out = root / 'full' / ARMS[1] / 'evaluations/update_0001000/partial.json'
            atomic_json(root / 'status.json', {'status':'evaluating','active_phase':'full'})
            calls = []
            def sleep(_):
                if not out.exists(): atomic_json(out, {'fixture':1})
                else: stop.touch()
            with patch('exp027_report.render', side_effect=lambda _: calls.append(1)), \
                    patch('exp027_report.time.sleep', side_effect=sleep), \
                    patch('exp027_report.time.monotonic', return_value=0):
                report_command(root, True, interval=100, stop_file=stop)
            self.assertEqual(len(calls),3)  # initial, new finding, final stop


if __name__ == '__main__':
    unittest.main()
