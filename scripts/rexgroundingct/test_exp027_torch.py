"""Run in the existing VoxTell image with no GPUs exposed."""
import copy
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
import torch

from exp027_common import ARMS, DEFAULT_CONFIG, atomic_json, read_json, segmentation_counts, lock
from exp027_data import native_to_original, original_to_native, restore_crop_to_original, cache_contract, case_key, write_case_cache
from exp027_model import ResidualRefiner, gaussian_weights, refiner_loss, refine_volume, tile_starts
from exp027_runner import (build_optimizer, checkpoint_payload, load_state, restore_checkpoint,
                           seed_all, torch_save, update_step, warmup, weight_hash)
from exp027_runner import train_worker, evaluate_worker


def tiny_config():
    config = read_json(DEFAULT_CONFIG)
    config["model"].update(channels=4, groups=2, dilations=[1], patch_size=[8, 8, 8])
    config["training"]["amp"] = False
    return config


class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cpu_only(self):
        self.assertFalse(torch.cuda.is_available(), "These tests must be run with GPUs disabled")

    def test_cache_export_preserves_all_prompts_and_only_selected_channels(self):
        from voxtell_preprocessed_cache import sha256_array
        config = tiny_config()
        name = "train_1_a_1.nii.gz"
        records = {f"{name}::{fid}": {"key": f"{name}::{fid}", "name": name, "channel": i}
                   for fid, i in (("2", 0), ("9", 2))}
        image = np.ones((1, 3, 4, 5), np.float32)
        target = np.ones((3, 3, 4, 5), np.uint8)
        logits = np.stack([np.full((3, 4, 5), v, np.float32) for v in (-40, 3, 40)])
        prompts = ["first 2a", "other category", "second 2a"]
        meta = {"preprocess_id": "crop_zscore_native_v1", "image_sha256": sha256_array(image),
                "targets_sha256": sha256_array(target), "prompts": prompts}
        predictor = object()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch("voxtell_preprocessed_cache.load_cached_case", return_value=(image, target, meta)), \
                    patch("run_voxtell_val_inference.predict_preprocessed_crop_logits", return_value=logits) as infer:
                result = write_case_cache(config, records, {name: {"findings": dict(zip(("2", "5", "9"), prompts))}},
                                          {}, folder, name, predictor, 0, "fixture")
            self.assertIs(result, predictor)
            self.assertEqual(infer.call_args.args[2], prompts)
            stored = np.load(folder / "logits.npy")
            self.assertEqual(stored.dtype, np.float16)
            self.assertEqual(stored.shape, (2, 3, 4, 5))
            np.testing.assert_array_equal(stored[:, 0, 0, 0], [-30, 30])
            self.assertEqual([r["channel"] for r in read_json(folder / "metadata.json")["records"]], [0, 2])

    def test_four_workers_wait_for_shared_context_and_agree(self):
        code = ("import sys; from pathlib import Path; "
                "from exp027_runner import session_context; "
                "from exp027_common import DEFAULT_CONFIG,read_json; "
                "root=Path(sys.argv[1]); (root/('ready_'+sys.argv[2])).write_text('ready'); "
                "print(session_context(read_json(DEFAULT_CONFIG), {}, root),flush=True)")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            children = []
            try:
                with lock(root / ".context.lock"):
                    children = [subprocess.Popen([sys.executable, "-c", code, directory, str(i)],
                                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                for i in range(4)]
                    deadline = time.monotonic() + 25
                    while len(list(root.glob("ready_*"))) < 4 and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertEqual(len(list(root.glob("ready_*"))), 4)
                    time.sleep(.2)
                    self.assertTrue(all(child.poll() is None for child in children),
                                    "Sibling workers must wait for the held metadata lock")
                outputs = []
                for child in children:
                    stdout, stderr = child.communicate(timeout=30)
                    self.assertEqual(child.returncode, 0, stderr)
                    outputs.append(stdout.strip())
                self.assertEqual(len(set(outputs)), 1)
                self.assertEqual(len(outputs[0]), 64)
                self.assertTrue((root / "context.json").is_file())
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.communicate()

    def test_zero_init_and_finite_gradients(self):
        model = ResidualRefiner(channels=4, groups=2, dilations=[1, 2])
        x = torch.randn(1, 2, 9, 10, 11)
        residual = model(x)
        self.assertEqual(torch.count_nonzero(residual), 0)
        self.assertTrue(torch.equal(x[:, 1:2] + residual, x[:, 1:2]))
        y = (torch.rand_like(residual) > .8).float()
        loss, _ = refiner_loss(x[:, 1:2], residual, y, torch.ones_like(y))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.head.weight.grad.abs().sum(), 0)

    def test_loss_ignores_padding_and_accepts_empty_gt(self):
        z = torch.zeros(1, 1, 4, 4, 4)
        delta = torch.zeros_like(z, requires_grad=True)
        y, valid = torch.zeros_like(z), torch.zeros_like(z)
        valid[:, :, 1:3, 1:3, 1:3] = 1
        before, _ = refiner_loss(z, delta, y, valid)
        changed = delta + (1 - valid) * 100
        after, _ = refiner_loss(z, changed, y, valid)
        self.assertEqual(before, after)
        before.backward()
        self.assertEqual(torch.count_nonzero(delta.grad[valid == 0]), 0)
        with self.assertRaises(ValueError):
            refiner_loss(z, delta, y, torch.zeros_like(valid))

    def test_tiny_set_can_fit(self):
        seed_all(3)
        model = ResidualRefiner(4, 2, [1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
        x = torch.zeros(1, 2, 8, 8, 8)
        y = torch.zeros(1, 1, 8, 8, 8)
        y[:, :, 2:6, 2:6, 2:6] = 1
        x[:, 0:1] = y * 2 - 1
        losses = []
        for _ in range(45):
            optimizer.zero_grad()
            loss, _ = refiner_loss(x[:, 1:2], model(x), y, torch.ones_like(y))
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0] * .4)

    def test_tiles_identity_constant_and_coverage(self):
        class Constant(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
            def forward(self, x):
                return torch.ones_like(x[:, :1]) * self.value
        for shape in ((3, 4, 5), (8, 8, 8), (13, 14, 15)):
            base = np.random.default_rng(1).normal(size=shape).astype(np.float32)
            original, zero = refine_volume(Constant(0), base, base, (8, 8, 8), torch.device("cpu"), amp=False)
            np.testing.assert_array_equal(original, base)
            np.testing.assert_array_equal(zero, 0)
            changed, residual = refine_volume(Constant(1.5), base, base, (8, 8, 8), torch.device("cpu"), amp=False)
            np.testing.assert_allclose(residual, 1.5, atol=1e-6)
            np.testing.assert_allclose(changed, base + 1.5, atol=1e-6)
        self.assertTrue(np.all(gaussian_weights((8, 8, 8)) > 0))
        self.assertEqual(len(tile_starts((8, 8, 8), (8, 8, 8))), 1)

    def test_orientation_round_trip_keeps_logits_continuous(self):
        array = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) / 7 - 3
        original = np.diag([-1., -2., 3., 1.])
        reoriented = np.diag([1., 2., 3., 1.])
        props = {"nibabel_stuff": {"original_affine": original, "reoriented_affine": reoriented}}
        native = original_to_native(array, props)
        self.assertEqual(native.shape, (5, 4, 3))
        np.testing.assert_array_equal(native_to_original(native, props), array)
        self.assertGreater(len(np.unique(native)), 2)

    def test_checkpoint_resume_is_equivalent_and_bound(self):
        config = tiny_config()
        seed_all(33)
        model = ResidualRefiner(4, 2, [1])
        opt = build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        x, y = torch.randn(1, 2, 8, 8, 8), torch.ones(1, 1, 8, 8, 8)
        valid = torch.ones_like(y)
        for _ in range(3):
            update_step(model, opt, scaler, x, y, valid, config, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pth"
            torch_save(path, checkpoint_payload(model, opt, scaler, 3, [], "fingerprint", ARMS[0], "initial"))
            for _ in range(2):
                update_step(model, opt, scaler, x, y, valid, config, torch.device("cpu"))
            expected = weight_hash(model)
            resumed = ResidualRefiner(4, 2, [1])
            resumed_opt = build_optimizer(resumed, config)
            resumed_scaler = torch.amp.GradScaler("cuda", enabled=False)
            state = load_state(path)
            restore_checkpoint(state, resumed, resumed_opt, resumed_scaler, "fingerprint", ARMS[0])
            for _ in range(2):
                update_step(resumed, resumed_opt, resumed_scaler, x, y, valid, config, torch.device("cpu"))
            self.assertEqual(weight_hash(resumed), expected)
            with self.assertRaises(ValueError):
                restore_checkpoint(state, resumed, resumed_opt, resumed_scaler, "changed", ARMS[0])

    def test_amp_overflow_retries_same_patch_and_counts_one_optimizer_step(self):
        config = tiny_config()
        seed_all(51)
        model = ResidualRefiner(4, 2, [1])
        reference = copy.deepcopy(model)
        opt, ref_opt = build_optimizer(model, config), build_optimizer(reference, config)
        # Real GradScaler on CPU exercises its inf detection and skipped step.
        scaler = torch.amp.GradScaler("cpu", init_scale=2.)
        x, y = torch.randn(1, 2, 8, 8, 8), torch.ones(1, 1, 8, 8, 8)
        valid = torch.ones_like(y)
        calls, inputs = [], []
        def overflow_once(grad):
            calls.append(1)
            return torch.full_like(grad, float("inf")) if len(calls) == 1 else grad
        def forward_rng(_module, arguments):
            inputs.append(arguments[0].data_ptr())
            torch.rand(3)
        model.head.weight.register_hook(overflow_once)
        model.register_forward_pre_hook(forward_rng)
        reference.register_forward_pre_hook(forward_rng)
        initial_rng = torch.get_rng_state().clone()
        metrics = update_step(model, opt, scaler, x, y, valid, config, torch.device("cpu"))
        final_rng = torch.get_rng_state().clone()
        self.assertEqual(metrics["amp_overflow_retries"], 1)
        self.assertEqual(metrics["amp_scale"], 1.)
        self.assertEqual(calls, [1, 1])
        self.assertEqual(inputs, [x.data_ptr()] * 2)
        self.assertEqual({float(v["step"]) for v in opt.state.values()}, {1.})
        torch.set_rng_state(initial_rng)
        update_step(reference, ref_opt, torch.amp.GradScaler("cpu", enabled=False),
                    x, y, valid, config, torch.device("cpu"))
        self.assertEqual(weight_hash(model), weight_hash(reference))
        self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))

    def test_nonfinite_loss_and_persistent_gradients_still_stop(self):
        config = tiny_config()
        x, y = torch.randn(1, 2, 8, 8, 8), torch.ones(1, 1, 8, 8, 8)
        for amp in (False, True):
            model = ResidualRefiner(4, 2, [1])
            opt = build_optimizer(model, config)
            scaler = torch.amp.GradScaler("cpu", init_scale=2., enabled=amp)
            before = weight_hash(model)
            model.head.weight.register_hook(lambda g: torch.full_like(g, float("inf")))
            with self.assertRaisesRegex(FloatingPointError, "Non-finite gradients"):
                update_step(model, opt, scaler, x, y, torch.ones_like(y), config, torch.device("cpu"))
            self.assertEqual(len(opt.state), 0)
            self.assertEqual(weight_hash(model), before)
        model = ResidualRefiner(4, 2, [1])
        opt = build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cpu", init_scale=2.)
        x.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "Non-finite loss"):
            update_step(model, opt, scaler, x, y, torch.ones_like(y), config, torch.device("cpu"))
        self.assertEqual(len(opt.state), 0)
        self.assertEqual(scaler.get_scale(), 2.)

    def test_amp_backoff_state_survives_checkpoint_resume(self):
        config = tiny_config()
        model = ResidualRefiner(4, 2, [1])
        opt = build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cpu", init_scale=4.)
        x, y = torch.randn(1, 2, 8, 8, 8), torch.ones(1, 1, 8, 8, 8)
        calls = []
        def overflow_once(grad):
            calls.append(1)
            return torch.full_like(grad, float("inf")) if len(calls) == 1 else grad
        model.head.weight.register_hook(overflow_once)
        metrics = update_step(model, opt, scaler, x, y, y, config, torch.device("cpu"))
        state = copy.deepcopy(checkpoint_payload(model, opt, scaler, 1, [metrics], "test", ARMS[0], "init"))
        update_step(model, opt, scaler, x, y, y, config, torch.device("cpu"))
        resumed = ResidualRefiner(4, 2, [1])
        resumed_opt = build_optimizer(resumed, config)
        resumed_scaler = torch.amp.GradScaler("cpu", init_scale=4.)
        restore_checkpoint(state, resumed, resumed_opt, resumed_scaler, "test", ARMS[0])
        self.assertEqual(resumed_scaler.get_scale(), 2.)
        update_step(resumed, resumed_opt, resumed_scaler, x, y, y, config, torch.device("cpu"))
        self.assertEqual(weight_hash(resumed), weight_hash(model))
        self.assertEqual(resumed_scaler.state_dict(), scaler.state_dict())
        self.assertEqual({float(v["step"]) for v in resumed_opt.state.values()}, {2.})

    def test_uncrop_orientation_preserves_uncapped_logits_and_masks(self):
        properties = {"nibabel_stuff": {"original_affine": np.diag([-1., 1., 1., 1.]).tolist(),
                                       "reoriented_affine": np.eye(4).tolist()}}
        meta = {"native_cropped_shape_zyx": [2, 3, 4], "resampled_shape_zyx": [2, 3, 4],
                "original_reoriented_shape_zyx": [4, 5, 6], "crop_bbox_zyx": [[1, 3], [1, 4], [1, 5]],
                "ct_properties": properties}
        crop = np.linspace(-90, 90, 24, dtype=np.float32).reshape(2, 3, 4)
        original = restore_crop_to_original(crop, meta, -30)
        self.assertEqual(original.shape, (6, 5, 4))
        np.testing.assert_array_equal(original_to_native(original, properties)[1:3, 1:4, 1:5], crop)
        np.testing.assert_array_equal(original >= 0, restore_crop_to_original(crop >= 0, meta))

    def test_warmup_restores_pristine_state_and_rng(self):
        config = tiny_config()
        seed_all(9)
        model = ResidualRefiner(4, 2, [1])
        opt = build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        before, rng = weight_hash(model), torch.get_rng_state().clone()
        warmup(model, opt, scaler, config, torch.device("cpu"), 5)
        self.assertEqual(before, weight_hash(model))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(opt.state), 0)

    def test_amp_warmup_restores_state_after_backoff_and_after_failure(self):
        config = tiny_config()
        for persistent in (False, True):
            seed_all(9)
            model = ResidualRefiner(4, 2, [1])
            opt = build_optimizer(model, config)
            scaler = torch.amp.GradScaler("cpu", init_scale=2.)
            calls = []
            def overflow(grad):
                calls.append(1)
                return torch.full_like(grad, float("inf")) if persistent or len(calls) == 1 else grad
            model.head.weight.register_hook(overflow)
            before, rng, scale_state = weight_hash(model), torch.get_rng_state().clone(), scaler.state_dict()
            if persistent:
                with self.assertRaisesRegex(FloatingPointError, "Non-finite gradients"):
                    warmup(model, opt, scaler, config, torch.device("cpu"), 5)
            else:
                history = warmup(model, opt, scaler, config, torch.device("cpu"), 5)
                self.assertEqual(len(history), 5)
                self.assertEqual(sum(r["amp_overflow_retries"] for r in history), 1)
            self.assertEqual(weight_hash(model), before)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(scaler.state_dict(), scale_state)
            self.assertEqual(len(opt.state), 0)

    def test_worker_failure_recovery_and_complete_evaluation_on_cpu_fixture(self):
        config = tiny_config()
        image = np.ones((1, 8, 8, 8), np.float32)
        target = np.zeros((1, 8, 8, 8), np.uint8)
        target[:, 2:6, 2:6, 2:6] = 1
        base = np.full_like(image, -1)
        base[:, 3:5, 3:5, 3:5] = 1
        records = [{"name": f"train_{n}_a_1.nii.gz", "finding_id": "4", "channel": 0,
                    "key": f"train_{n}_a_1.nii.gz::4", "voxels": 64, "entities": 1,
                    "split": "train" if n == 1 else "val", "half": "A" if n == 2 else "B"}
                   for n in (1, 2, 3)]
        baseline = segmentation_counts(base >= 0, target)["dice"]
        prepared = {"train": records[:1], "val": records[1:],
                    "baseline_findings": [{**r, "dice": baseline} for r in records[1:]]}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            for r in records:
                folder = root / "cache" / case_key(r["name"])
                folder.mkdir(parents=True)
                for name, arr in (("image", image), ("logits", base), ("targets", target)):
                    np.save(folder / f"{name}.npy", arr)
                atomic_json(folder / "metadata.json", {"contract": cache_contract(config, prepared),
                            "image_path": str(folder / "image.npy"), "records": [r],
                            "ct_metadata": {"native_cropped_shape_zyx": [8, 8, 8], "resampled_shape_zyx": [8, 8, 8],
                                            "original_reoriented_shape_zyx": [8, 8, 8], "crop_bbox_zyx": [[0, 8]] * 3,
                                            "ct_properties": {"nibabel_stuff": {"original_affine": np.eye(4).tolist(),
                                                                                "reoriented_affine": np.eye(4).tolist()}}},
                            "points": {r["key"]: {"gt": [[3, 3, 3]], "prediction": [[3, 3, 3]]}}})
            stack.enter_context(patch("exp027_runner.worker_device", return_value=torch.device("cpu")))
            for method in ("synchronize", "reset_peak_memory_stats"):
                stack.enter_context(patch(f"torch.cuda.{method}"))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
            original_step = update_step
            calls = [0]
            def failing_step(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 3:
                    raise RuntimeError("fixture interruption")
                return original_step(*args, **kwargs)
            with patch("exp027_runner.update_step", side_effect=failing_step):
                with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                    train_worker(config, prepared, root, "full", ARMS[0], 4, allow_gpu=False)
            state = read_json(root / "full" / ARMS[0] / "status.json")
            self.assertEqual((state["status"], state["update"]), ("failed", 2))
            train_worker(config, prepared, root, "full", ARMS[0], 4, allow_gpu=False)
            history = read_json(root / "full" / ARMS[0] / "training.json")["updates"]
            self.assertEqual([r["update"] for r in history], [1, 2, 3, 4])
            report = evaluate_worker(config, prepared, root, "full", ARMS[0], 4, allow_gpu=False)
            self.assertEqual(report["metrics"]["full"]["findings"], 2)
            self.assertEqual(report["metrics"]["A"]["findings"], 1)
            self.assertEqual(report["metrics"]["B"]["findings"], 1)
            self.assertEqual(report["status"], "pending_user_review")
            self.assertEqual(evaluate_worker(config, prepared, root, "full", ARMS[0], 4, allow_gpu=False), report)


if __name__ == "__main__":
    unittest.main()
