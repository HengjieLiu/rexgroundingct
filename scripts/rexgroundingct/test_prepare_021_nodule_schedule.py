#!/usr/bin/env python3
"""Focused tests for the pure Exp021 schedule helpers."""

from __future__ import annotations

import numpy as np

from prepare_021_nodule_schedule import (
    PATCH_SIZE,
    audit_schedule,
    choose_negative_prompts,
    random_patch_starts,
)


def test_random_patch_starts_are_in_bounds() -> None:
    starts = random_patch_starts((200, 180, 192), PATCH_SIZE, np.random.default_rng(3))
    assert starts == [7, 0, 0]
    for start, dimension, patch in zip(starts, (200, 180, 192), PATCH_SIZE):
        assert 0 <= start <= max(0, dimension - patch)


def test_negative_prompt_fallback_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    assert choose_negative_prompts([], ["nodule"], 2, rng) == ["nodule", "nodule"]


def test_audit_accepts_one_positive_and_one_negative_block() -> None:
    positive = {
        "event_index": 0,
        "case_name": "positive.nii.gz",
        "sampling_source": "category_2d_positive",
        "requested_target_category": "2d",
        "requested_target_index": 0,
        "patch_starts": [0, 0, 0],
        "positive_crop_points": [{"target_index": 0, "point": [1, 2, 3]}],
        "prompt_slots": [
            {"prompt": "nodule", "target_index": 0, "is_positive": True},
            {"prompt": "other", "target_index": None, "is_positive": False},
            {"prompt": "other2", "target_index": None, "is_positive": False},
        ],
    }
    negative = {
        "event_index": 1,
        "case_name": "negative.nii.gz",
        "sampling_source": "nodule_negative_case",
        "requested_target_category": None,
        "requested_target_index": None,
        "patch_starts": [0, 0, 0],
        "prompt_slots": [
            {"prompt": "nodule", "target_index": None, "is_positive": False},
            {"prompt": "other", "target_index": None, "is_positive": False},
            {"prompt": "other2", "target_index": None, "is_positive": False},
        ],
    }
    result = audit_schedule([positive, negative], 1, 2, {"negative.nii.gz"})
    assert result["sampling_source_counts"] == {
        "category_2d_positive": 1,
        "nodule_negative_case": 1,
    }
