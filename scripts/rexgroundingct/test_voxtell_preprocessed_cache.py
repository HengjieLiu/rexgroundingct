#!/usr/bin/env python3
"""Focused tests for VoxTell normalization caches and padding behavior."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from run_voxtell_val_inference import (
    predict_preprocessed_crop_probabilities,
    restore_cached_native_crop,
)
from train_text_conditioned_voxtell import (
    RexVoxTellPatchSampler,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from voxtell_preprocessed_cache import (
    CLIPPED_LINEAR_ISO07_PREPROCESS_ID,
    CLIPPED_LINEAR_NATIVE_PREPROCESS_ID,
    CLIPPED_ZSCORE_NATIVE_PREPROCESS_ID,
    NATIVE_PREPROCESS_ID,
    _normalize_native_image,
    image_padding_value,
    preprocess_spec,
)


class _FakePredictor:
    patch_size = (4, 4, 4)

    def embed_text_prompts(self, prompts: list[str]) -> torch.Tensor:
        return torch.zeros((1, len(prompts), 1), dtype=torch.float32)

    def predict_sliding_window_return_logits(
        self,
        image: torch.Tensor,
        _text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(image.shape) != (1, 4, 4, 4):
            raise AssertionError(f"Unexpected padded shape: {tuple(image.shape)}")
        if not torch.any(image == -1.0):
            raise AssertionError("Expected fixed-HU padding value -1")
        return image.repeat(2, 1, 1, 1)


class VoxTellPreprocessedCacheTests(unittest.TestCase):
    def test_normalization_variants(self) -> None:
        image = np.asarray(
            [[[[ -8192.0, -1024.0], [0.0, 1024.0]]]],
            dtype=np.float32,
        )
        baseline = _normalize_native_image(image, NATIVE_PREPROCESS_ID)
        clipped_zscore = _normalize_native_image(
            image,
            CLIPPED_ZSCORE_NATIVE_PREPROCESS_ID,
        )
        linear = _normalize_native_image(
            image,
            CLIPPED_LINEAR_NATIVE_PREPROCESS_ID,
        )
        self.assertAlmostEqual(float(baseline.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(baseline.std()), 1.0, places=6)
        self.assertAlmostEqual(float(clipped_zscore.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(clipped_zscore.std()), 1.0, places=6)
        np.testing.assert_array_equal(
            linear,
            np.asarray([[[[-1.0, -1.0], [0.0, 1.0]]]], dtype=np.float32),
        )

    def test_manifest_padding_defaults(self) -> None:
        self.assertEqual(
            image_padding_value({"preprocess_id": NATIVE_PREPROCESS_ID}),
            0.0,
        )
        self.assertEqual(
            image_padding_value(
                {"preprocess_id": CLIPPED_LINEAR_NATIVE_PREPROCESS_ID}
            ),
            -1.0,
        )
        self.assertEqual(
            image_padding_value(
                {"preprocess_id": CLIPPED_LINEAR_ISO07_PREPROCESS_ID}
            ),
            -1.0,
        )
        self.assertEqual(
            image_padding_value(
                {
                    "preprocess_id": CLIPPED_LINEAR_NATIVE_PREPROCESS_ID,
                    "image_padding_value": -0.75,
                }
            ),
            -0.75,
        )

    def test_iso07_preprocess_spec(self) -> None:
        spec = preprocess_spec(CLIPPED_LINEAR_ISO07_PREPROCESS_ID)
        self.assertEqual(spec["target_spacing_zyx_mm"], [0.7, 0.7, 0.7])
        self.assertEqual(spec["image_padding_value"], -1.0)
        self.assertEqual(spec["target_padding_value"], 0)
        self.assertEqual(spec["normalization_parameters"]["clip_hu"], [-1024.0, 1024.0])
        self.assertEqual(
            spec["mask_interpolation"],
            "nearest_exact_with_foreground_center_splat_fallback",
        )

    def test_patch_sampler_uses_requested_fill_value(self) -> None:
        sampler = RexVoxTellPatchSampler.__new__(RexVoxTellPatchSampler)
        sampler.patch_size = (4, 4, 4)
        image = np.ones((1, 2, 3, 4), dtype=np.float32)
        patch = sampler.extract_patch(image, [0, 0, 0], fill_value=-1.0)
        self.assertEqual(tuple(patch.shape), (1, 4, 4, 4))
        np.testing.assert_array_equal(patch[:, :2, :3, :4], image)
        self.assertTrue(np.all(patch[:, 2:, :, :] == -1.0))
        self.assertTrue(np.all(patch[:, :, 3:, :] == -1.0))

    def test_cached_inference_restores_custom_padding(self) -> None:
        image = np.zeros((1, 2, 3, 4), dtype=np.float32)
        probabilities = predict_preprocessed_crop_probabilities(
            _FakePredictor(),
            image,
            ["a", "b"],
            padding_value=-1.0,
        )
        self.assertEqual(tuple(probabilities.shape), (2, 2, 3, 4))
        np.testing.assert_allclose(probabilities, 0.5, rtol=0, atol=1e-6)

    def test_resampled_cache_restore_interpolates_to_native_crop(self) -> None:
        prediction = np.ones((2, 2, 3, 4), dtype=np.float32)
        metadata = {
            "resampled_shape_zyx": [2, 3, 4],
            "native_cropped_shape_zyx": [4, 6, 8],
            "original_reoriented_shape_zyx": [6, 8, 10],
            "crop_bbox_zyx": [[1, 5], [1, 7], [1, 9]],
        }
        restored = restore_cached_native_crop(prediction, metadata)
        self.assertEqual(tuple(restored.shape), (2, 6, 8, 10))
        np.testing.assert_allclose(restored[:, 1:5, 1:7, 1:9], 1.0)
        np.testing.assert_allclose(restored[:, :1], 0.0)
        np.testing.assert_allclose(restored[:, 5:], 0.0)

    def test_rng_state_round_trip_for_segmented_training(self) -> None:
        original_state = capture_rng_state()
        try:
            seed_everything(20260723)
            resume_state = capture_rng_state()
            expected_python = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = torch.rand(4)

            seed_everything(9)
            self.assertTrue(restore_rng_state(resume_state))
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.random()), expected_numpy)
            torch.testing.assert_close(torch.rand(4), expected_torch)
        finally:
            restore_rng_state(original_state)


if __name__ == "__main__":
    unittest.main()
