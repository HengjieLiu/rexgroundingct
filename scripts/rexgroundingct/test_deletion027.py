"""CPU tests for the four-arm deletion sampler, recovery, barriers and live board."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from exp027_common import ARMS, append_update, atomic_json, digest, lock, read_json, read_updates, reconcile_updates, sha256
from exp027_data import extract_patch
from exp027_model import tile_starts
from deletion027_data import eligible_tiles, finish_preparation, pool_keys, schedule, verify_receipts
from deletion027_worker import aggregate, edit_metrics, load_checkpoint, restore_rng, save_checkpoint, setup_device, weight_hash
from deletion027_report import collect, render, smooth
from run_027_deletion_four import verify_barrier


def fixture():
    prepared = {"train": [{"key": "t1"}, {"key": "t0"}],
                "val": [{"key": "a1", "half": "A"}, {"key": "a2", "half": "A"}, {"key": "b1", "half": "B"}]}
    tiles = [{"starts": [0, 0, 0], "tp": 2, "fp": 0}, {"starts": [1, 0, 0], "tp": 0, "fp": 3},
             {"starts": [2, 0, 0], "tp": 1, "fp": 1}]
    index = {key: {"tiles": tiles, "tp_indices": [0, 2], "fp_indices": [1, 2]} for key in ("t1", "a1", "a2", "b1")}
    index["t0"] = {"tiles": [], "tp_indices": [], "fp_indices": []}
    return prepared, index


def finding(key, half, tp=10, fp=20, removed_tp=0, removed_fp=0):
    m = edit_metrics(tp, fp, tp+4, removed_tp, removed_fp)
    return {"key": key, "name": key, "half": half, "prompt": "SYNTHETIC", **m, "thresholds": {"0.9": m}}


class DeletionFourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_tile_counts_and_padding(self):
        rng = np.random.default_rng(4)
        for shape in ((4, 5, 6), (11, 12, 13)):
            z = rng.normal(size=shape).astype(np.float32)
            y = (rng.random(shape) < .2).astype(np.uint8)
            index = eligible_tiles(z, y, (8, 8, 8))
            expected = []
            for starts in tile_starts(shape, (8, 8, 8)):
                zz, valid = extract_patch(z, starts, (8, 8, 8), -30)
                yy, _ = extract_patch(y, starts, (8, 8, 8), 0)
                b = (zz >= 0) & (valid > 0)
                if b.any():
                    expected.append({"starts": list(starts), "tp": int((b & (yy > 0)).sum()), "fp": int((b & (yy == 0)).sum())})
            self.assertEqual(index["tiles"], expected)
        self.assertEqual(eligible_tiles(-np.ones((3, 3, 3)), np.ones((3, 3, 3)), (8, 8, 8))["tiles"], [])

    def test_arm_membership_and_exact_epoch_mixtures(self):
        prepared, index = fixture()
        expected = ({"t1"}, {"t1", "a1", "a2"}, {"a1", "a2"}, {"a1", "a2", "b1"})
        for arm, wanted in zip(ARMS, expected):
            pools = pool_keys(prepared, index, arm)
            self.assertEqual({k for rows in pools.values() for k in rows}, wanted)
            events = schedule(prepared, index, arm, 2000, 7)
            self.assertEqual(events, schedule(prepared, index, arm, 2000, 7))
            self.assertEqual([e["update"] for e in events], list(range(1, 2001)))
            for start in range(0, 2000, 100):
                rows = events[start:start+100]
                self.assertEqual([sum(e["branch"] == b for e in rows) for b in ("tp", "fp", "eligible")], [50, 25, 25])
                if arm == ARMS[1]:
                    self.assertEqual(sum(e["source"] == "train" for e in rows), 50)
                self.assertTrue(all(e["key"] in wanted for e in rows))

    def test_missing_tp_branch_falls_back_same_finding(self):
        prepared, index = fixture()
        index["t1"] = {"tiles": [{"starts": [0, 0, 0], "tp": 0, "fp": 2}], "tp_indices": [], "fp_indices": [0]}
        events = schedule(prepared, index, ARMS[0], 100, 9)
        self.assertEqual(sum(e["fallback"] for e in events), 50)
        self.assertTrue(all(e["key"] == "t1" and e["fp"] == 2 for e in events))

    def test_incomplete_index_blocks_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared, _ = fixture()
            atomic_json(root / "prepared.json", prepared)
            with patch("deletion027_data.check_context", return_value={"sha256": "test"}):
                with self.assertRaisesRegex(ValueError, "Complete-pool"):
                    finish_preparation({}, root)

    def test_verified_cache_receipt_detects_changed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            array = root / "array.npy"
            array.write_bytes(b"verified content")
            stat = array.stat()
            prepared = {"train": [], "val": []}
            row = {"name": "case", "keys": ["key"], "files": {str(array): {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}}}
            atomic_json(root / "cache_verified_gpu0.json", {"cases": [row], "prepared_sha256": digest(prepared), "cache_contract": "contract"})
            inventory = {"cases": 1, "names": ["case"], "keys": ["key"], "cache_contract": "contract"}
            with patch("deletion027_data.require_full_cache", return_value=inventory), patch("deletion027_data.cache_contract", return_value="contract"):
                self.assertEqual(verify_receipts({"experiment_dir": str(root)}, prepared)["cases"], 1)
                array.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "cache changed"):
                    verify_receipts({"experiment_dir": str(root)}, prepared)

    def test_checkpoint_optimizer_rng_replay_and_journal_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.manual_seed(10)
            model = torch.nn.Linear(3, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            save_checkpoint(root, model, optimizer, 0, [], "context", "schedule", "initial")
            self.assertEqual(load_checkpoint(root, "context", "schedule")["update"], 0)
            history = []
            for update in range(1, 4):
                optimizer.zero_grad(); model(torch.randn(2, 3)).square().mean().backward(); optimizer.step()
                row = {"update": update, "loss": .1}; history.append(row); append_update(root / "training.jsonl", row)
            save_checkpoint(root, model, optimizer, 3, history, "context", "schedule", "initial")
            optimizer.zero_grad(); model(torch.randn(2, 3)).square().mean().backward(); optimizer.step()
            expected = weight_hash(model)
            state = load_checkpoint(root, "context", "schedule")
            second = torch.nn.Linear(3, 1); second.load_state_dict(state["model"])
            opt2 = torch.optim.AdamW(second.parameters(), lr=1e-4); opt2.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
            opt2.zero_grad(); second(torch.randn(2, 3)).square().mean().backward(); opt2.step()
            self.assertEqual(weight_hash(second), expected)
            append_update(root / "training.jsonl", {"update": 4, "loss": .2})
            reconcile_updates(root / "training.jsonl", state["history"])
            self.assertEqual(len(read_updates(root / "training.jsonl")), 3)
            self.assertEqual(len(list(root.glob("recovered_*.jsonl"))), 1)
            with self.assertRaisesRegex(ValueError, "context"):
                load_checkpoint(root, "wrong_context", "schedule")

    def test_aggregate_half_recomposition_and_retention_flags(self):
        rows = [finding("a", "A", 10, 20, 7, 15), finding("b", "B", 20, 10, 0, 3), finding("b2", "B", 0, 10, 0, 2)]
        result = aggregate(rows)
        for key in ("tp", "fp", "base_tp", "base_fp", "tp_removed", "fp_removed"):
            self.assertEqual(result["full"][key], result["A"][key]+result["B"][key])
        self.assertAlmostEqual(result["full"]["dice"], (result["A"]["dice"]+2*result["B"]["dice"])/3)
        self.assertEqual(result["full"]["below_80"], 1)
        self.assertEqual(result["full"]["undefined_retention"], 1)
        self.assertIsNone(aggregate(rows[:1])["B"]["dice"])

    def test_barrier_requires_all_four_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for arm in ARMS:
                ck = root / "runs" / arm / "checkpoints/update_0000100.pth"
                ck.parent.mkdir(parents=True); ck.write_bytes(b"synthetic checkpoint")
                atomic_json(root / "runs" / arm / "evaluations/update_0000100/summary.json",
                            {"context_sha256": "x", "update": 100, "findings": [{"key": "a"}], "checkpoint_sha256": sha256(ck)})
            verify_barrier(root, 100, {"val": [{"key": "a"}]}, "x")
            (root / "runs" / ARMS[3] / "evaluations/update_0000100/summary.json").unlink()
            with self.assertRaises(FileNotFoundError):
                verify_barrier(root, 100, {"val": [{"key": "a"}]}, "x")

    def test_gpu_opt_in_and_single_writer(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            setup_device(False)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".writer.lock"
            with lock(path):
                with self.assertRaises(RuntimeError):
                    with lock(path):
                        pass

    def test_partial_arrives_before_other_arms_and_final_replaces(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_json(root / "status.json", {"status": "evaluating", "synthetic_fixture": True})
            rows = [finding("a", "A", 10, 20, 7, 15)]
            result = {"update": 100, "findings": rows, "metrics": aggregate(rows), "expected_counts": {"A": 35, "B": 34, "full": 69}}
            out = root / "runs" / ARMS[0] / "evaluations/update_0000100"
            atomic_json(out / "partial.json", result)
            data = collect(root)
            self.assertEqual([r["arm"] for r in data["arms"]], list(ARMS))
            self.assertIsNotNone(data["arms"][0]["partial"])
            self.assertEqual(data["arms"][1]["evaluations"], [])
            render(root)
            text = (root / "reports/live_dashboard.md").read_text()
            self.assertIn("provisional", text)
            self.assertIn("SEVERE", text)
            self.assertIn("SYNTHETIC", text)
            atomic_json(out / "summary.json", result)
            self.assertIsNone(collect(root)["arms"][0]["partial"])
            atomic_json(root / "runs" / ARMS[1] / "status.json", {"status": "failed", "error": "synthetic failure"})
            render(root)
            self.assertIn("synthetic failure", (root / "reports/live_dashboard.md").read_text())

    def test_smoothing_keeps_missing_pending(self):
        np.testing.assert_allclose(smooth([1, 2, 200], [1., 3., np.nan]), [1., 2., np.nan], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
