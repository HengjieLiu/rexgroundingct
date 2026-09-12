"""CPU adapter tests for category-specific training and completed-only reporting."""
import copy
import os
import signal
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from exp027_common import REPO, atomic_json, append_update, digest, lock, read_json, read_updates, sha256
from deletion027_categories_data import (ARMS, LOSSES, DEFAULT_CONFIG, MILESTONES, validate_config,
    category_prepared, schedule, initial_state, verify_gate, verify_2a_stop)
from deletion027_categories_worker import aggregate, edit_metrics, evaluate, load_checkpoint, train
from deletion027_categories_analysis import analyze_evaluation, threshold_counts, analyze_available
from deletion027_categories_report import collect, render, plot_points
from deletion027_worker import weight_hash
from run_027_deletion_categories import verify_barrier, orchestrate
from test_deletion027_train_ablation import LossTests


def finding(key, half, category="2b", removed_tp=0, removed_fp=0):
    m = edit_metrics(10, 20, 15, removed_tp, removed_fp)
    return {"key": key, "name": key, "patient": key, "half": half, "category": category,
            "prompt": "SYNTHETIC", **m, "thresholds": {str(t): m for t in (.5, .9)}}


def dashboard_fixture(root):
    prepared = {}
    atomic_json(root/"status.json", {"status": "evaluating", "synthetic_fixture": True})
    for arm in ARMS:
        baseline = [finding(arm+"a", "A", arm), finding(arm+"b", "B", arm)]
        prepared[arm] = {"val": baseline, "baseline_findings": baseline, "train_keys": [arm+"train"], "exposure": {}}
        for update in range(1, 101):
            append_update(root/"runs"/arm/"training.jsonl", {"update": update, "key": arm+"train",
                "loss": .6+1/update, "remove_loss": 1/update, "preserve_loss": .03+.0001*update,
                "dice_loss": .6-.0001*update, "grad_norm_before_clip": 1/update})
        atomic_json(root/"runs"/arm/"status.json", {"status": "evaluating", "update": 100,
            "findings_done": 1, "findings_total": 2})
        metrics = aggregate([finding(arm+"a", "A", arm, 1, 8)])
        atomic_json(root/"runs"/arm/"evaluations/update_0000100/partial.json",
            {"status": "provisional", "update": 100, "findings_done": 1, "findings_total": 2,
             "threshold_metrics": {str(t): metrics for t in (.5, .9)}})
    atomic_json(root/"prepared.json", prepared)
    arm = ARMS[0]
    rows = [finding(arm+"a", "A", arm, 1, 8), finding(arm+"b", "B", arm, 2, 10)]
    summary = {"status": "complete", "update": 100, "findings_done": 2, "findings_total": 2,
               "threshold_metrics": {str(t): aggregate(rows) for t in (.5, .9)}}
    atomic_json(root/"runs"/arm/"evaluations/update_0000100/summary.json", summary)
    return summary


class CategoryTests(unittest.TestCase):
    def test_membership_reproducible_schedules_exclusion_and_patient_separation(self):
        records, index, halves = [], {}, {}
        for arm in ARMS:
            for j, split in enumerate(("train", "train", "val", "val")):
                key = arm+str(j)
                record = {"key": key, "name": key, "patient": key, "category": arm, "split": split, "voxels": 15}
                records.append(record)
                index[key] = {"tiles": [{"start": [0, 0, 0], "tp": 10, "fp": 20}], "tp_indices": [0],
                              "fp_indices": [0], "base_tp": 10, "base_fp": 20, "total_gt": 15}
                if split == "val": halves[key] = "A" if j == 2 else "B"
        index['2b1']['tiles'] = []
        next(r for r in records if r['key'] == '2d1')['patient'] = '2d2'
        with self.assertRaisesRegex(ValueError, 'patient overlap'):
            category_prepared(records, index, halves, [])
        prepared = category_prepared(records, index, halves, ['2d1'])
        self.assertEqual(prepared['2d']['excluded_overlap'], ['2d1'])
        self.assertEqual(prepared['2b']['train_keys'], ['2b0'])
        for arm in ARMS:
            events = schedule(prepared[arm]['train_keys'], index, 5000, 7)
            self.assertEqual(events, schedule(prepared[arm]['train_keys'], index, 5000, 7))
            self.assertEqual(events[:2000], schedule(prepared[arm]['train_keys'], index, 2000, 7))
            for start in range(0, 5000, 100):
                self.assertEqual([sum(r['branch'] == b for r in events[start:start+100]) for b in ('tp','fp','eligible')], [50,25,25])
            self.assertTrue(all(r['key'] in prepared[arm]['train_keys'] and r['source'] == 'train' for r in events))
        index['2b0']['fp_indices'] = []
        events = schedule(['2b0'], index, 100, 7)
        self.assertEqual(sum(e['fallback'] for e in events), 25)

    def test_gate_detects_modified_index_array_or_missing_stop_receipt(self):
        from exp027_multicategory_data import file_proof
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root/'array'; data.write_bytes(b'verified')
            files = file_proof([data]); index = {'a': {'total_gt': 12}}
            context = {'sha256': 'c', 'config': {'cache_inventory_sha256': 'inventory'},
                       'index_sha256': digest(index), 'input_files_sha256': digest(files)}
            manifest = {'context_sha256': 'c', 'cache_inventory_sha256': 'inventory', 'files': files,
                        'index': index, 'cases': 2730, 'findings': 5106}
            atomic_json(root/'input_manifest.json', manifest); verify_gate(root, context)
            wrong = copy.deepcopy(manifest); wrong['index']['a']['total_gt'] = 13
            atomic_json(root/'input_manifest.json', wrong)
            with self.assertRaises(ValueError): verify_gate(root, context)
            atomic_json(root/'input_manifest.json', manifest); data.write_bytes(b'changed')
            with self.assertRaises(ValueError): verify_gate(root, context)
            with self.assertRaises(FileNotFoundError): verify_2a_stop({'stop_receipt': str(root/'absent')})
            atomic_json(root/'stop.json', {'status': 'stopped_by_user', 'update': 6000, 'evaluation_count': 4, 'coordinator_exited': False})
            with self.assertRaises(ValueError): verify_2a_stop({'stop_receipt': str(root/'stop.json')})

    def test_seven_barriers_variable_cohorts_and_category_identity(self):
        self.assertEqual(MILESTONES, [100,500,1000,2000,3000,4000,5000])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = {a: {'val': [{'key': a+str(i)} for i in range(n)]} for a,n in zip(ARMS,[49,60,132,11])}
            for update in MILESTONES:
                for arm in ARMS:
                    ck = root/'runs'/arm/f'checkpoints/update_{update:07d}.pth'
                    ck.parent.mkdir(parents=True, exist_ok=True);ck.write_bytes(b'fixture')
                    atomic_json(root/'runs'/arm/f'evaluations/update_{update:07d}/summary.json',
                        {'status':'complete','arm':arm,'loss_spec':LOSSES[arm],'update':update,'context_sha256':'c',
                         'checkpoint_sha256':sha256(ck),'findings':prepared[arm]['val']})
                verify_barrier(root, update, prepared, 'c')
            path = root/'runs/2d/evaluations/update_0005000/summary.json'
            value = read_json(path);value['findings'].pop();atomic_json(path,value)
            with self.assertRaises(ValueError):verify_barrier(root,5000,prepared,'c')

    def test_complete_only_dashboard_independent_publication_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dashboard_fixture(root)
            data = collect(root)
            self.assertEqual([a['arm'] for a in data['arms']],list(ARMS))
            self.assertEqual(len(plot_points(data['arms'][0], .5)),1)
            self.assertIsNone(data['arms'][0]['pending_update'])
            self.assertEqual(plot_points(data['arms'][1], .5),[])
            self.assertEqual(data['arms'][1]['pending_update'],100)
            atomic_json(root/'runs/2e/status.json',{'status':'failed','error':'SYNTHETIC failure'})
            render(root)
            content = (root/'reports/live_dashboard.md').read_text()
            for token in ['SYNTHETIC','pending','0.50','0.90','train_2936','failure','full-validation']:
                self.assertIn(token,content)
            self.assertNotIn('provisional Dice:',content)
            with lock(root/'.report_writer.lock'):
                with self.assertRaises(RuntimeError):
                    with lock(root/'.report_writer.lock'):pass
            # Empty history / no prepared metadata must render during preparation.
            render(root/'empty')

    def test_config_pristine_initial_state_and_gpu_guards(self):
        config = read_json(DEFAULT_CONFIG);base = read_json(REPO/config['base_config'])
        validate_config(config,base)
        for field, value in [('total_updates',None),('precision','fp16'),('tf32',True)]:
            with self.assertRaises(ValueError):validate_config({**config,field:value},base)
        with patch('run_027_deletion_categories.subprocess.Popen') as popen:
            orchestrate(config,SimpleNamespace(dry_run=True));popen.assert_not_called()
            with self.assertRaisesRegex(ValueError,'opt-in'):
                orchestrate(config,SimpleNamespace(dry_run=False,allow_gpu=False))
        old = Path(config['historical_runtime'])
        historical = read_json(old/'context.json')
        context = {'sha256':'test','base_config':base,'historical_context_sha256':historical['sha256'],
                   'historical_initial_file_sha256':sha256(old/'initial.pth')}
        with tempfile.TemporaryDirectory() as directory, patch('deletion027_categories_data.check_context', return_value=context):
            root=Path(directory)
            first=initial_state(config,root);second=initial_state(config,root)
            self.assertEqual(first['weights_sha256'],config['initial_weights_sha256'])
            for key in first['model']:self.assertTrue(torch.equal(first['model'][key],second['model'][key]))

    def test_host_dry_run_never_calls_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            fake=Path(directory)/'docker';fake.write_text('#!/bin/sh\nexit 97\n');fake.chmod(0o755)
            result=subprocess.run(['bash',str(REPO/'scripts/rexgroundingct/run_027_deletion_categories_host.sh'),'orchestrate','--dry-run'],
                env={**os.environ,'START_GPU_WORK':'1','PATH':directory+':'+os.environ['PATH']},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("gpu_work_started': False",result.stdout)

class ReplayTests(unittest.TestCase):
    def test_evaluation_resume_and_dense_analysis_without_repeat_inference(self):
        from types import SimpleNamespace
        config = read_json(DEFAULT_CONFIG)
        context = {"sha256": "test", "base_config": {"experiment_dir": "unused", "model": {"patch_size": [4]*3}}}
        z = np.ones((3, 3, 3), np.float32)
        target = np.zeros_like(z, dtype=np.uint8); target[1, 1, 1] = 1
        rows = [{"key": f"finding{i}", "name": f"case{i}", "patient": f"patient{i}", "prompt": "SYNTHETIC",
                 "half": "A" if i < 2 else "B", "voxels": 1} for i in range(4)]
        config["expected_val"][ARMS[0]] = 4
        prepared = {"val": rows, "train": [], "exposure": {h: "held_out_development" for h in ("A", "B", "full")}, "baseline_findings": [
            {"key": r["key"], "dice": edit_metrics(1, 26, 1, 0, 0)["dice"]} for r in rows]}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            atomic_json(root/"prepared.json", {ARMS[0]: prepared})
            model = torch.nn.Conv3d(2, 1, 1)
            checkpoint = root/"runs"/ARMS[0]/"checkpoints/update_0000100.pth"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"model": model.state_dict(), "context_sha256": "test", "update": 100,
                        "arm": ARMS[0], "loss_spec": LOSSES[ARMS[0]]}, checkpoint)
            for name, value in (("check_context", context), ("verify_gate", {}), ("setup_device", torch.device("cpu"))):
                stack.enter_context(patch("deletion027_categories_worker."+name, return_value=value))
            stack.enter_context(patch("deletion027_categories_worker.make_editor", return_value=model))
            stack.enter_context(patch("deletion027_categories_worker.FindingStore", return_value=SimpleNamespace(
                get=lambda key: (np.zeros_like(z), z, target, {}, {"ct_metadata": {}}))))
            stack.enter_context(patch("deletion027_categories_worker.restore_crop_to_original", side_effect=lambda a, _: a))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
            stack.enter_context(patch("deletion027_categories_worker.signal.signal"))
            answer = (np.full_like(z, .5), {"tiles": 1, "active_tiles": 1})
            with patch("deletion027_categories_worker.infer_removal", side_effect=[answer, InterruptedError("synthetic interruption")]):
                with self.assertRaises(InterruptedError):
                    evaluate(config, root, ARMS[0], 100, True)
            out = root/"runs"/ARMS[0]/"evaluations/update_0000100"
            self.assertEqual(read_json(out/"partial.json")["findings_done"], 1)
            with patch("deletion027_categories_worker.infer_removal", return_value=answer) as inference:
                result = evaluate(config, root, ARMS[0], 100, True)
                self.assertEqual(inference.call_count, 3)
                evaluate(config, root, ARMS[0], 100, True)
                self.assertEqual(inference.call_count, 3)
            self.assertEqual(result["threshold_metrics"]["0.5"]["A"]["findings"], 2)
            self.assertEqual(result["threshold_metrics"]["0.5"]["B"]["findings"], 2)
            dense = analyze_evaluation(config, root, ARMS[0], out/"summary.json")
            self.assertEqual(dense["per_finding_rows"], 4*201)
            self.assertEqual(dense, analyze_evaluation(config, root, ARMS[0], out/"summary.json"))

    def test_real_training_loop_interruption_and_resume(self):
        torch.set_num_threads(2)
        config = read_json(DEFAULT_CONFIG)
        config["evaluation_updates"] = [2100, 2200]
        config["total_updates"] = 2200
        context = {"sha256": "test", "base_config": {"experiment_dir": "unused", "model": {"patch_size": [4]*3},
                   "optimizer": {"lr": 1e-4, "weight_decay": 1e-4, "betas": [.9, .999], "eps": 1e-8, "grad_clip": 1}}}
        torch.manual_seed(19)
        initial_model = torch.nn.Conv3d(2, 1, 1)
        initial = {"model": copy.deepcopy(initial_model.state_dict()), "weights_sha256": weight_hash(initial_model)}
        x = np.ones((2, 4, 4, 4), np.float32)
        y = np.zeros((1, 4, 4, 4), np.float32); y[:, :2, :2, :2] = 1
        valid = np.ones_like(y)
        index = {"a": {"base_tp": 16, "base_fp": 80, "total_gt": 24, "record": {"voxels": 24, "split": "train", "category": ARMS[2]}}}
        events = [{"update": i, "key": "a", "tile_index": 0, "tp": 8, "fp": 56, "source": "train"} for i in range(1, 2201)]
        context["schedule_hashes"] = {ARMS[2]: digest(events)}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name, value in (("check_context", context), ("verify_gate", {"index": index}),
                                ("initial_state", initial), ("setup_device", torch.device("cpu"))):
                stack.enter_context(patch("deletion027_categories_worker."+name, return_value=value))
            stack.enter_context(patch("deletion027_categories_worker.FindingStore"))
            stack.enter_context(patch("deletion027_categories_worker.make_editor", side_effect=lambda *_: torch.nn.Conv3d(2, 1, 1)))
            stack.enter_context(patch("torch.cuda.synchronize"))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
            handlers = {}
            stack.enter_context(patch("deletion027_categories_worker.signal.signal", side_effect=lambda s, h: handlers.update({s: h})))
            roots = [Path(directory)/name for name in ("continuous", "resumed")]
            for root in roots:
                root.mkdir(); (root/"initial.pth").touch()
                atomic_json(root/"prepared.json", {ARMS[2]: {"train_keys": ["a"]}})
                atomic_json(root/"runs"/ARMS[2]/"schedule.json", {"context_sha256": "test", "sha256": digest(events), "events": events})
            with patch("deletion027_categories_worker.load_event", return_value=(x, y, valid)):
                train(config, roots[0], ARMS[2], 2200, True)
                train(config, roots[1], ARMS[2], 2100, True)
            def interrupting_load(store, event, size):
                if event["update"] == 2107:
                    handlers[signal.SIGTERM](None, None)
                return x, y, valid
            with patch("deletion027_categories_worker.load_event", side_effect=interrupting_load):
                with self.assertRaises(InterruptedError):
                    train(config, roots[1], ARMS[2], 2200, True)
            folder = roots[1]/"runs"/ARMS[2]
            self.assertEqual(load_checkpoint(folder, "test", digest(events))["update"], 2107)
            with patch("deletion027_categories_worker.load_event", return_value=(x, y, valid)):
                train(config, roots[1], ARMS[2], 2200, True)
            states = [load_checkpoint(root/"runs"/ARMS[2], "test", digest(events)) for root in roots]
            self.assertEqual(states[0]["model_sha256"], states[1]["model_sha256"])
            self.assertEqual(len(read_updates(folder/"training.jsonl")), 2200)
            self.assertTrue(all(a["loss"] == b["loss"] for a, b in zip(states[0]["history"], states[1]["history"])))
            self.assertEqual(read_json(folder/"schedule.json")["events"], events)



if __name__ == "__main__":
    unittest.main()
