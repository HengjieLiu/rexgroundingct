"""CPU loss, provenance, replay, reporting and failure tests for the train-only study."""
from __future__ import annotations

import copy
import math
import signal
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from exp027_common import atomic_json, digest, lock, read_json, read_updates, sha256
from fit_027_deletion import deletion_loss, infer_removal
from deletion027_worker import weight_hash
from deletion027_train_ablation_data import (ARMS, LOSSES, DEFAULT_CONFIG, validate_config, validate_events,
                                       validate_membership, verify_gate, immutable_json)
from deletion027_ablation_loss import loss_for_arm
from deletion027_train_ablation_worker import aggregate, edit_metrics, evaluate, load_checkpoint, train
from deletion027_train_ablation_report import collect, render
from deletion027_train_ablation_analysis import analyze_evaluation, threshold_counts
from run_027_deletion_loss_ablation_train import verify_barrier


def finding(key, half, removed_tp=0, removed_fp=0):
    m = edit_metrics(10, 20, 15, removed_tp, removed_fp)
    return {"key": key, "name": key, "patient": key, "half": half, "prompt": "SYNTHETIC", **m,
            "thresholds": {str(t): m for t in (.5, .9)}}


def dashboard_fixture(root):
    rows = [finding("a", "A", 1, 8)]
    metrics = aggregate(rows)
    summary = {"update": 100, "findings": rows, "threshold_metrics": {str(t): metrics for t in (.5, .9)},
               "expected_counts": {"A": 35, "B": 34, "full": 69}}
    atomic_json(root/"status.json", {"status": "evaluating", "synthetic_fixture": True})
    atomic_json(root/"input_manifest.json", {"baseline_findings": [finding("a", "A"), finding("b", "B")]})
    from exp027_common import append_update
    for arm in ARMS:
        for update in range(1, 6):
            append_update(root/"runs"/arm/"training.jsonl",
                          {"update": update, "loss": 2/update, "remove_loss": 1/update,
                           "preserve_loss": .2*update, "dice_loss": .7-.01*update,
                           "grad_norm_before_clip": 1/update})
    out = root/"runs"/ARMS[0]/"evaluations/update_0000100"
    atomic_json(out/"partial.json", summary)
    atomic_json(root/"runs"/ARMS[2]/"status.json", {"status": "failed", "error": "SYNTHETIC failure example"})
    return summary, out


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tensors(self):
        r = torch.tensor([-.7, .2, 1.1, -.3, .9], requires_grad=True)
        z, y, valid = torch.ones(5), torch.tensor([1., 0., 0., 1., 0.]), torch.tensor([1., 1., 1., 1., 0.])
        return r, z, y, valid

    def test_four_formulas_and_historical_exact_gradient(self):
        counts = {"base_tp": 5, "base_fp": 8, "total_gt": 9}
        for arm in ARMS:
            r, z, y, v = self.tensors()
            loss, parts = loss_for_arm(r, z, y, v, counts, arm)
            w = 2 if arm.endswith("tp2") else 1
            bce, _ = deletion_loss(r, z, y, v, 1., w)
            expected = parts["dice_loss"]+.25*bce/(1+w) if arm.startswith("dice_") else bce
            self.assertTrue(torch.equal(loss, expected))
            gradient = torch.autograd.grad(loss, r, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(gradient[-1].item(), 0.)
            if arm == ARMS[0]:
                self.assertTrue(torch.equal(gradient, torch.autograd.grad(bce, r)[0]))

    def test_dice_equals_explicit_full_finding(self):
        r, z, y, v = self.tensors()
        counts = {"base_tp": 5, "base_fp": 8, "total_gt": 9}
        _, parts = loss_for_arm(r, z, y, v, counts, ARMS[2])
        # Patch has two TP/two FP; outside has three TP/six FP/four FN.
        q = torch.cat([1-r[:4].sigmoid(), torch.ones(9), torch.zeros(4)])
        gt = torch.cat([y[:4], torch.ones(3), torch.zeros(6), torch.ones(4)])
        exact = 1-(2*(q*gt).sum()+1e-6)/(q.sum()+gt.sum()+1e-6)
        torch.testing.assert_close(parts["dice_loss"], exact)
        torch.testing.assert_close(torch.autograd.grad(parts["dice_loss"], r, retain_graph=True)[0],
                                   torch.autograd.grad(exact, r)[0])

    def test_dice_gradient_signs_and_global_tp_outside_empty_patch(self):
        for y in (torch.tensor([1., 0.]), torch.zeros(2)):
            r = torch.zeros(2, requires_grad=True)
            _, parts = loss_for_arm(r, torch.ones(2), y, torch.ones(2),
                                    {"base_tp": 10, "base_fp": 20, "total_gt": 15}, ARMS[2])
            grad = torch.autograd.grad(parts["dice_loss"], r)[0]
            self.assertTrue((grad[y == 0] < 0).all())
            self.assertTrue((grad[y > 0] > 0).all())

    def test_empty_groups_fixed_normalization_and_zero_base_tp(self):
        for target in (torch.zeros(4), torch.ones(4)):
            for arm in ARMS:
                r = torch.zeros(4, requires_grad=True)
                t = int(target.sum())
                loss, parts = loss_for_arm(r, torch.ones(4), target, torch.ones(4),
                                           {"base_tp": t, "base_fp": 4-t, "total_gt": t+3}, arm)
                loss.backward()
                self.assertTrue(torch.isfinite(r.grad).all())
                self.assertEqual(parts["remove_loss" if t else "preserve_loss"].item(), 0.)
                self.assertAlmostEqual(parts["normalized_bce"].item(),
                                       (parts["remove_loss"]+(2 if arm.endswith('tp2') else 1)*parts["preserve_loss"]).item()/(3 if arm.endswith('tp2') else 2), places=6)
        with self.assertRaisesRegex(ValueError, "whole-finding"):
            loss_for_arm(torch.zeros(4), torch.ones(4), torch.ones(4), torch.ones(4),
                         {"base_tp": 2, "base_fp": 0, "total_gt": 3}, ARMS[0])

    def test_joint_deletion_improves_dice_when_fp_benefit_compensates(self):
        t, f, g = 10, 40, 20
        self.assertGreater(edit_metrics(t, f, g, 1, 20)["delta_dice"], 0)
        self.assertLess(edit_metrics(t, f, g, 5, 2)["delta_dice"], 0)

    def test_threshold_ties_and_original_metrics(self):
        labels = np.array([True, True, False, False, False])
        scores = np.array([.5, .9, .1, .5, 1.], np.float32)
        cuts = [0., .1, .5, .9, 1.]
        for base in (False, True):
            for cut, row in zip(cuts, threshold_counts(labels, scores, 4, cuts, base=base)):
                deleted = scores < cut if base else scores > cut
                self.assertEqual(row, edit_metrics(2, 3, 4, int((deleted & labels).sum()), int((deleted & ~labels).sum())))
        self.assertEqual(threshold_counts(labels, scores, 4, [1.])[0]["tp_removed"], 0)
        with self.assertRaises(ValueError):
            threshold_counts(labels, scores*np.nan, 4, cuts)

    def test_tiling_blends_probabilities_small_and_overlapping(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.full_like(x[:, :1], math.log(.3/.7))
        for shape in ((3, 4, 5), (13, 12, 11)):
            z = -np.ones(shape, np.float32); z[1, 1, 1] = 1
            scores, _ = infer_removal(Constant(), np.zeros(shape, np.float32), z, [8]*3, torch.device("cpu"))
            self.assertAlmostEqual(float(scores[1, 1, 1]), .3, places=6)
            final = (z >= 0) & ~(scores > .5)
            self.assertFalse(np.any(final & (z < 0)))


class ContractTests(unittest.TestCase):
    def test_orchestration_opt_in_and_dry_run_no_mutations(self):
        from types import SimpleNamespace
        from run_027_deletion_loss_ablation_train import orchestrate
        config = read_json(DEFAULT_CONFIG)
        with patch("run_027_deletion_loss_ablation_train.subprocess.Popen") as popen:
            orchestrate(config, SimpleNamespace(dry_run=True))
            popen.assert_not_called()
            with self.assertRaisesRegex(ValueError, "opt-in"):
                orchestrate(config, SimpleNamespace(dry_run=False, allow_gpu=False))
            popen.assert_not_called()

    def test_real_historical_membership_schedule_and_config(self):
        config = read_json(DEFAULT_CONFIG)
        old = Path(config["historical_runtime"])
        if not old.exists():
            self.skipTest("Historical data mount not present")
        prepared = read_json(old/"prepared.json")
        validate_membership(prepared)
        bad = copy.deepcopy(prepared)
        a = next(r for r in bad["val"] if r["half"] == "A")
        next(r for r in bad["val"] if r["half"] == "B")["patient"] = a["patient"]
        with self.assertRaises(ValueError):
            validate_membership(bad)
        from exp027_common import REPO
        validate_config(config, read_json(REPO/config["base_config"]))
        index = {}
        from exp027_data import case_key
        for name in {r["name"] for r in prepared["train"] + prepared["val"]}:
            index.update(read_json(old/"tile_index"/f"{case_key(name)}.json")["findings"])
        from deletion027_data import schedule
        events = schedule(prepared, index, "run1_train", 10000, config["seed"])
        self.assertEqual(events[:2000], read_json(old/"runs/run1_train/schedule.json")["events"])
        validate_events(config, prepared, index, events)
        corrupted = copy.deepcopy(events); corrupted[0]["source"] = "B"
        with self.assertRaises(ValueError):
            validate_events(config, prepared, index, corrupted)
        bad_config = {**config, "total_updates": None}
        with self.assertRaises(ValueError):
            validate_config(bad_config, read_json(REPO/config["base_config"]))

    def test_cache_gate_and_immutable_artifact_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            immutable_json(root/"x.json", {"v": 1})
            immutable_json(root/"x.json", {"v": 1})
            with self.assertRaises(ValueError):
                immutable_json(root/"x.json", {"v": 2})
            atomic_json(root/"prepared.json", {"val": []})
            atomic_json(root/"input_manifest.json", {"context_sha256": "c", "findings": 0, "cases": 0})
            with self.assertRaisesRegex(ValueError, "gate"):
                verify_gate(root, {"sha256": "c"})

    def test_full_recomposition_and_hits(self):
        rows = [finding("a", "A", 1, 8), finding("b", "B", 8, 1)]
        m = aggregate(rows)
        self.assertAlmostEqual(m["full"]["dice"], (m["A"]["dice"]+m["B"]["dice"])/2)
        self.assertEqual(m["full"]["tp_removed"], m["A"]["tp_removed"]+m["B"]["tp_removed"])
        self.assertIsNone(aggregate(rows[:1])["B"]["hits_lost"])

    def test_dashboard_partial_final_failures_and_fixed_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, out = dashboard_fixture(root)
            data = collect(root)
            self.assertEqual([a["arm"] for a in data["arms"]], list(ARMS))
            self.assertIsNotNone(data["arms"][0]["partial"])
            self.assertEqual(data["arms"][1]["evaluations"], [])
            render(root)
            text = (root/"reports/live_dashboard.md").read_text()
            for token in ("SYNTHETIC", "provisional", "pending", "0.50", "0.90", "failure example", "F + 2K"):
                self.assertIn(token, text)
            self.assertIn("10,000", text)
            self.assertIn("1,119", text)
            self.assertIn("both excluded from gradients", text)
            self.assertNotIn("Full results mix exposure", text)
            atomic_json(out/"summary.json", result)
            self.assertIsNone(collect(root)["arms"][0]["partial"])
            with lock(root/".report_writer.lock"):
                with self.assertRaises(RuntimeError):
                    with lock(root/".report_writer.lock"):
                        pass

    def test_barriers_require_every_arm_and_loss_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ARMS:
                folder = root/"runs"/arm
                ck = folder/"checkpoints/update_0000100.pth"
                ck.parent.mkdir(parents=True); ck.write_bytes(b"fixture")
                atomic_json(folder/"evaluations/update_0000100/summary.json",
                            {"status": "complete", "arm": arm, "loss_spec": LOSSES[arm], "update": 100, "context_sha256": "c",
                             "checkpoint_sha256": sha256(ck), "findings": [{"key": "a"}]})
            verify_barrier(root, 100, {"val": [{"key": "a"}]}, "c")
            (root/"runs"/ARMS[-1]/"evaluations/update_0000100/summary.json").unlink()
            with self.assertRaises(FileNotFoundError):
                verify_barrier(root, 100, {"val": [{"key": "a"}]}, "c")


class ReplayTests(unittest.TestCase):
    def test_evaluation_resume_and_dense_analysis_without_repeat_inference(self):
        from types import SimpleNamespace
        config = read_json(DEFAULT_CONFIG)
        context = {"sha256": "test", "base_config": {"experiment_dir": "unused", "model": {"patch_size": [4]*3}}}
        z = np.ones((3, 3, 3), np.float32)
        target = np.zeros_like(z, dtype=np.uint8); target[1, 1, 1] = 1
        rows = [{"key": f"finding{i}", "name": f"case{i}", "patient": f"patient{i}", "prompt": "SYNTHETIC",
                 "half": "A" if i < 35 else "B", "voxels": 1} for i in range(69)]
        prepared = {"val": rows, "train": [], "baseline_findings": [
            {"key": r["key"], "dice": edit_metrics(1, 26, 1, 0, 0)["dice"]} for r in rows]}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            atomic_json(root/"prepared.json", prepared)
            model = torch.nn.Conv3d(2, 1, 1)
            checkpoint = root/"runs"/ARMS[0]/"checkpoints/update_0000100.pth"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"model": model.state_dict(), "context_sha256": "test", "update": 100,
                        "arm": ARMS[0], "loss_spec": LOSSES[ARMS[0]]}, checkpoint)
            for name, value in (("check_context", context), ("verify_gate", {}), ("setup_device", torch.device("cpu"))):
                stack.enter_context(patch("deletion027_train_ablation_worker."+name, return_value=value))
            stack.enter_context(patch("deletion027_train_ablation_worker.make_editor", return_value=model))
            stack.enter_context(patch("deletion027_train_ablation_worker.FindingStore", return_value=SimpleNamespace(
                get=lambda key: (np.zeros_like(z), z, target, {}, {"ct_metadata": {}}))))
            stack.enter_context(patch("deletion027_train_ablation_worker.restore_crop_to_original", side_effect=lambda a, _: a))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
            stack.enter_context(patch("deletion027_train_ablation_worker.signal.signal"))
            answer = (np.full_like(z, .5), {"tiles": 1, "active_tiles": 1})
            with patch("deletion027_train_ablation_worker.infer_removal", side_effect=[answer, InterruptedError("synthetic interruption")]):
                with self.assertRaises(InterruptedError):
                    evaluate(config, root, ARMS[0], 100, True)
            out = root/"runs"/ARMS[0]/"evaluations/update_0000100"
            self.assertEqual(read_json(out/"partial.json")["findings_done"], 1)
            with patch("deletion027_train_ablation_worker.infer_removal", return_value=answer) as inference:
                result = evaluate(config, root, ARMS[0], 100, True)
                self.assertEqual(inference.call_count, 68)
                evaluate(config, root, ARMS[0], 100, True)
                self.assertEqual(inference.call_count, 68)
            self.assertEqual(result["threshold_metrics"]["0.5"]["A"]["findings"], 35)
            self.assertEqual(result["threshold_metrics"]["0.5"]["B"]["findings"], 34)
            dense = analyze_evaluation(config, root, ARMS[0], out/"summary.json")
            self.assertEqual(dense["per_finding_rows"], 69*201)
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
        index = {"a": {"base_tp": 16, "base_fp": 80, "record": {"voxels": 24, "split": "train"}}}
        events = [{"update": i, "key": "a", "tile_index": 0, "tp": 8, "fp": 56, "source": "train"} for i in range(1, 2201)]
        config["schedule_sha256"] = digest(events)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name, value in (("check_context", context), ("verify_gate", {"index": index}),
                                ("initial_state", initial), ("setup_device", torch.device("cpu"))):
                stack.enter_context(patch("deletion027_train_ablation_worker."+name, return_value=value))
            stack.enter_context(patch("deletion027_train_ablation_worker.FindingStore"))
            stack.enter_context(patch("deletion027_train_ablation_worker.make_editor", side_effect=lambda *_: torch.nn.Conv3d(2, 1, 1)))
            stack.enter_context(patch("torch.cuda.synchronize"))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
            handlers = {}
            stack.enter_context(patch("deletion027_train_ablation_worker.signal.signal", side_effect=lambda s, h: handlers.update({s: h})))
            roots = [Path(directory)/name for name in ("continuous", "resumed")]
            for root in roots:
                root.mkdir(); (root/"initial.pth").touch()
                atomic_json(root/"prepared.json", {})
                atomic_json(root/"runs"/ARMS[2]/"schedule.json", {"context_sha256": "test", "sha256": digest(events), "events": events})
            with patch("deletion027_train_ablation_worker.load_event", return_value=(x, y, valid)):
                train(config, roots[0], ARMS[2], 2200, True)
                train(config, roots[1], ARMS[2], 2100, True)
            def interrupting_load(store, event, size):
                if event["update"] == 2107:
                    handlers[signal.SIGTERM](None, None)
                return x, y, valid
            with patch("deletion027_train_ablation_worker.load_event", side_effect=interrupting_load):
                with self.assertRaises(InterruptedError):
                    train(config, roots[1], ARMS[2], 2200, True)
            folder = roots[1]/"runs"/ARMS[2]
            self.assertEqual(load_checkpoint(folder, "test", digest(events))["update"], 2107)
            with patch("deletion027_train_ablation_worker.load_event", return_value=(x, y, valid)):
                train(config, roots[1], ARMS[2], 2200, True)
            states = [load_checkpoint(root/"runs"/ARMS[2], "test", digest(events)) for root in roots]
            self.assertEqual(states[0]["model_sha256"], states[1]["model_sha256"])
            self.assertEqual(len(read_updates(folder/"training.jsonl")), 2200)
            self.assertTrue(all(a["loss"] == b["loss"] for a, b in zip(states[0]["history"], states[1]["history"])))
            self.assertEqual(read_json(folder/"schedule.json")["events"], events)


class TrainOnlyTests(unittest.TestCase):
    def test_reference_check_stops_on_weight_or_metric_discrepancy(self):
        from deletion027_train_ablation_comparison import verify_reference
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"new"; old = Path(directory)/"old"
            model = torch.nn.Conv3d(2, 1, 1)
            identity = weight_hash(model)
            current = root/"runs"/ARMS[0]/"checkpoints/update_0000100.pth"
            previous = old/"runs/run1_train/checkpoints/update_0000100.pth"
            for path in (current, previous):
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "model_sha256": identity, "context_sha256": "c"}, path)
            findings = [{"key": f"finding{i}", "thresholds": {"0.5": {"dice": .3}}} for i in range(69)]
            current_summary = root/"runs"/ARMS[0]/"evaluations/update_0000100/summary.json"
            prior_summary = old/"runs/run1_train/evaluations/update_0000100/summary.json"
            for path, checkpoint in ((current_summary, current), (prior_summary, previous)):
                atomic_json(path, {"findings": findings, "checkpoint_sha256": sha256(checkpoint)})
            atomic_json(root/"prepared.json", {"val": findings})
            context = {"sha256": "c", "base_config": {},
                       "reference_files": {str(p): sha256(p) for p in (previous, prior_summary)}}
            with patch("deletion027_train_ablation_comparison.check_context", return_value=context), \
                    patch("fit_027_deletion.make_editor", side_effect=lambda *_: torch.nn.Conv3d(2, 1, 1)):
                result = verify_reference({"historical_runtime": str(old)}, root, 100)
                self.assertTrue(result["weights_identical"] and result["metrics_identical"])
                altered = read_json(current_summary); altered["findings"][0]["thresholds"]["0.5"]["dice"] = .4
                atomic_json(current_summary, altered)
                with self.assertRaisesRegex(ValueError, "reference differs"):
                    verify_reference({"historical_runtime": str(old)}, root, 100)
                self.assertEqual(read_json(root/"reports/reference_checks.json")["status"],
                                 "reference_discrepancy_requires_investigation")

    def test_complete_cache_gate_rejects_missing_inputs_and_changed_arrays(self):
        from deletion027_train_ablation_data import EXCLUDED
        config = read_json(DEFAULT_CONFIG)
        old = Path(config["historical_runtime"])
        if not old.exists():
            self.skipTest("Historical mount absent")
        prepared = read_json(old/"prepared.json")
        records = prepared["train"]+prepared["val"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root/"synthetic_array"; payload.write_bytes(b"verified")
            stat = payload.stat()
            proof = {str(payload): {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}}
            manifest = {"context_sha256": "test", "cases": 864, "findings": 1189,
                        "bypassed": EXCLUDED, "index": {r["key"]: {} for r in records},
                        "cache_proofs": [{"name": n, "keys": [r["key"] for r in records if r["name"] == n],
                                          "files": proof} for n in sorted({r["name"] for r in records})]}
            atomic_json(root/"prepared.json", prepared)
            def save(value):
                value.pop("sha256", None); value["sha256"] = digest(value)
                atomic_json(root/"input_manifest.json", value)
            save(manifest)
            self.assertEqual(verify_gate(root, {"sha256": "test"})["findings"], 1189)
            for key in (prepared["train"][0]["key"], prepared["val"][0]["key"]):
                corrupt = copy.deepcopy(manifest); del corrupt["index"][key]; save(corrupt)
                with self.assertRaisesRegex(ValueError, "gate"):
                    verify_gate(root, {"sha256": "test"})
            corrupt = copy.deepcopy(manifest); corrupt["cache_proofs"].pop(); save(corrupt)
            with self.assertRaisesRegex(ValueError, "gate"):
                verify_gate(root, {"sha256": "test"})
            save(manifest); payload.write_bytes(b"changed payload")
            with self.assertRaisesRegex(ValueError, "Verified cache changed"):
                verify_gate(root, {"sha256": "test"})

    def test_common_historical_initialization_is_pristine_and_immutable(self):
        from deletion027_train_ablation_data import initial_state
        config = read_json(DEFAULT_CONFIG)
        old = Path(config["historical_runtime"])
        if not old.exists():
            self.skipTest("Historical mount absent")
        context = {"sha256": "synthetic_context", "base_config": read_json(old/"context.json")["base_config"],
                   "historical_context_sha256": read_json(old/"context.json")["sha256"],
                   "historical_initial_file_sha256": sha256(old/"initial.pth")}
        with tempfile.TemporaryDirectory() as directory, patch("deletion027_train_ablation_data.check_context", return_value=context):
            root = Path(directory)
            first = initial_state(config, root)
            second = initial_state(config, root)
            self.assertEqual(first["weights_sha256"], config["initial_weights_sha256"])
            for key in first["model"]:
                self.assertTrue(torch.equal(first["model"][key], second["model"][key]))
            corrupt = torch.load(root/"initial.pth", weights_only=False)
            corrupt["context_sha256"] = "wrong"
            torch.save(corrupt, root/"initial.pth")
            with self.assertRaisesRegex(ValueError, "Local initial state"):
                initial_state(config, root)

    def test_all_twelve_barriers_require_four_models(self):
        from deletion027_train_ablation_data import MILESTONES
        self.assertEqual(MILESTONES, [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = {"val": [{"key": f"finding{i}"} for i in range(69)]}
            for update in MILESTONES:
                for arm in ARMS:
                    folder = root/"runs"/arm
                    checkpoint = folder/f"checkpoints/update_{update:07d}.pth"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True); checkpoint.write_bytes(b"fixture")
                    summary = {"status": "complete", "arm": arm, "loss_spec": LOSSES[arm], "update": update,
                               "context_sha256": "c", "findings": prepared["val"], "checkpoint_sha256": sha256(checkpoint)}
                    atomic_json(folder/f"evaluations/update_{update:07d}/summary.json", summary)
                verify_barrier(root, update, prepared, "c")
            path = root/"runs"/ARMS[-1]/"evaluations/update_0010000/summary.json"
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_barrier(root, 10000, prepared, "c")

    def test_host_dry_run_never_calls_docker_even_with_opt_in(self):
        import os
        import subprocess
        from exp027_common import REPO
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory)/"docker"
            fake.write_text("#!/bin/sh\necho unexpected-docker-call >&2\nexit 97\n"); fake.chmod(0o755)
            env = {**os.environ, "PATH": directory+":"+os.environ["PATH"], "START_GPU_WORK": "1"}
            proc = subprocess.run(["bash", str(REPO/"scripts/rexgroundingct/run_027_deletion_loss_ablation_train_host.sh"),
                                   "orchestrate", "--dry-run"], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("gpu_work_started': False", proc.stdout)
            self.assertNotIn("unexpected-docker-call", proc.stderr)

    def test_full_threshold_tables_publish_pending_then_complete(self):
        from deletion027_train_ablation_comparison import threshold_comparisons
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            found = []
            for arm in ARMS:
                rows = [{"arm": arm, "update": 100, "subset": subset, "threshold": i/200,
                         "base_dice": .34, "dice": .34+.01*(1-i/200)}
                        for subset in ("A", "B", "full") for i in range(201)]
                atomic_json(root/"reports/threshold_sweep"/arm/"update_0000100/metrics.json", rows)
                found.append({"arm": arm, "update": 100, "source_sha256": arm})
            threshold_comparisons({}, root, found[:1])
            folder = root/"reports/threshold_sweep/comparison_update_0000100"
            content = (folder/"all_thresholds.md").read_text()
            self.assertIn("1/4 arms", content)
            self.assertEqual(content.count("| pending | pending | pending |"), 603)
            threshold_comparisons({}, root, found)
            content = (folder/"all_thresholds.md").read_text()
            self.assertIn("4/4 arms", content)
            self.assertNotIn("| pending |", content)
            self.assertEqual(content.count("| 1.000 |"), 3)
            before = (folder/"all_thresholds.md").stat().st_mtime_ns
            threshold_comparisons({}, root, found)
            self.assertEqual(before, (folder/"all_thresholds.md").stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
