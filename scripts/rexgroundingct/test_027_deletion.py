"""CPU-only numerical and semantic checks for the bounded deletion pilot."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fit_027_deletion import (deletion_loss, infer_removal, make_editor, make_schedule,
                             metrics, report, retention_threshold, run, summarize)
from exp027_common import atomic_json


class ConstantEditor(torch.nn.Module):
    def forward(self, x):
        return torch.full_like(x[:, :1], 2.)


class DeletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_group_means_ignore_background_and_padding(self):
        z = torch.tensor([0., 0., 99., -99.], requires_grad=True)
        b = torch.tensor([1., 1., -1., 1.])
        y = torch.tensor([0., 1., 0., 0.])
        valid = torch.tensor([1., 1., 1., 0.])
        loss, _ = deletion_loss(z, b, y, valid)
        self.assertAlmostEqual(float(loss), 3 * np.log(2), places=6)
        loss.backward()
        np.testing.assert_allclose(z.grad.numpy(), [-.5, 1., 0., 0.])
        expanded, _ = deletion_loss(torch.zeros(101), torch.ones(101),
                                    torch.cat([torch.zeros(100), torch.ones(1)]), torch.ones(101))
        self.assertAlmostEqual(float(expanded), float(loss), places=6)

    def test_empty_groups_finite(self):
        for b, y in ((-1., 0.), (1., 0.), (1., 1.)):
            z = torch.zeros(1, requires_grad=True)
            loss, _ = deletion_loss(z, torch.full_like(z, b), torch.full_like(z, y), torch.ones_like(z))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(z.grad).all())

    def test_initial_keep_and_gradients(self):
        torch.manual_seed(5)
        model = make_editor({"initial_remove_probability": .05},
                            {"model": {"channels": 4, "groups": 2, "dilations": [1, 2, 4, 1]}})
        x = torch.randn(1, 2, 8, 8, 8)
        z = model(x)
        self.assertTrue(torch.allclose(z.sigmoid(), torch.full_like(z, .05), atol=1e-7))
        self.assertFalse((z.sigmoid() > .5).any())
        y = (x[:, :1] > 0).float()
        loss, _ = deletion_loss(z, x[:, 1:2], y, torch.ones_like(y))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_tiny_synthetic_fit(self):
        torch.manual_seed(1)
        model = torch.nn.Conv3d(2, 1, 1)
        x = torch.zeros(1, 2, 4, 4, 4)
        x[:, 1] = 1  # all base-positive
        x[:, 0, :2] = -1  # FP has negative CT, TP positive CT
        x[:, 0, 2:] = 1
        y = (x[:, :1] > 0).float()
        opt = torch.optim.Adam(model.parameters(), lr=.1)
        first = None
        for _ in range(80):
            opt.zero_grad()
            loss, _ = deletion_loss(model(x), x[:, 1:2], y, torch.ones_like(y))
            first = float(loss) if first is None else first
            loss.backward()
            opt.step()
        self.assertLess(float(loss), first / 10)
        np.testing.assert_array_equal((model(x).sigmoid() > .5).detach().numpy(), (y == 0).numpy())

    def test_tiling_constant_small_and_overlap(self):
        for shape in ((3, 4, 5), (11, 12, 13)):
            base = np.full(shape, -4., np.float32)
            base[tuple(n // 2 for n in shape)] = 1
            scores, stats = infer_removal(ConstantEditor(), np.zeros(shape, np.float32), base,
                                          (8, 8, 8), torch.device("cpu"))
            np.testing.assert_allclose(scores[base >= 0], torch.sigmoid(torch.tensor(2.)).item(), atol=1e-6)
            final = (base >= 0) & ~(scores > .5)
            self.assertFalse(np.any(final & (base < 0)))
            self.assertGreater(stats["active_tiles"], 0)

    def test_no_base_prediction_skips_model(self):
        class NeverCalled(torch.nn.Module):
            def forward(self, x):
                raise AssertionError("Empty tile called editor")
        scores, stats = infer_removal(NeverCalled(), np.zeros((4, 5, 6), np.float32),
                                     -np.ones((4, 5, 6), np.float32), (8, 8, 8), torch.device("cpu"))
        self.assertEqual(stats["active_tiles"], 0)
        self.assertFalse(scores.any())

    def test_metrics_correction_damage_and_empty_deletion(self):
        m = metrics([1, 1, 0, 0], [0, 1, 1, 0], 3)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 2))
        self.assertEqual(m["tp_retention"], .5)
        self.assertEqual(m["fp_removal"], .5)
        self.assertEqual(m["deletion_precision"], .5)
        self.assertIsNone(metrics([1, 0], [0, 0], 2)["deletion_precision"])
        self.assertIsNone(summarize([])["dice"])

    def test_threshold_retention_ties(self):
        for scores in (np.arange(100, dtype=np.float32), np.ones(100), np.array([.2, .2, .8])):
            for target in (1., .99, .9):
                t = retention_threshold(scores, target)
                self.assertGreaterEqual(float(np.mean(scores <= t)), target)
                t = retention_threshold(scores, target, False)
                self.assertGreaterEqual(float(np.mean(scores >= t)), target)

    def test_fixed_schedule_balance(self):
        events = make_schedule(24, 200, 20260910)
        self.assertEqual(events, make_schedule(24, 200, 20260910))
        self.assertEqual(len(events), 200)
        self.assertEqual(set(np.bincount(events)), {8, 9})

    def test_launch_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            run({}, Path("unused"), False)

    def test_pending_and_failure_dashboard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_json(root / "status.json", {"phase": "failed", "error": "synthetic fixture"})
            report(root)
            self.assertIn("synthetic fixture", (root / "report.md").read_text())
            self.assertTrue((root / "curves.png").exists())


if __name__ == "__main__":
    unittest.main()
