#!/usr/bin/env python3
"""Lightweight unit checks for Exp008 dual-branch components."""

from __future__ import annotations

import unittest

import torch

from train_text_conditioned_voxtell import asymmetric_tversky_loss_nonempty
from voxtell_dual_branch import VARIANTS, ZeroResidualAdapter


def logits(probabilities: list[float]) -> torch.Tensor:
    value = torch.tensor(probabilities, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    return torch.logit(value).reshape(1, 1, 1, 1, -1)


def target(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, -1)


class DualBranchComponentTests(unittest.TestCase):
    def test_zero_residual_adapter_starts_at_zero(self) -> None:
        adapter = ZeroResidualAdapter(8, 8, 4)
        output = adapter(torch.randn(2, 3, 8))
        self.assertEqual(torch.count_nonzero(output).item(), 0)

    def test_recall_tversky_penalizes_false_negatives_more(self) -> None:
        fn_heavy_logits = logits([0.9999, 0.0001, 0.0001, 0.9999])
        fn_heavy_target = target([1, 1, 1, 0])
        recall = asymmetric_tversky_loss_nonempty(
            fn_heavy_logits,
            fn_heavy_target,
            alpha=0.3,
            beta=0.7,
        )
        precision = asymmetric_tversky_loss_nonempty(
            fn_heavy_logits,
            fn_heavy_target,
            alpha=0.7,
            beta=0.3,
        )
        self.assertGreater(float(recall), float(precision))

    def test_precision_tversky_penalizes_false_positives_more(self) -> None:
        fp_heavy_logits = logits([0.9999, 0.0001, 0.9999, 0.9999])
        fp_heavy_target = target([1, 1, 0, 0])
        recall = asymmetric_tversky_loss_nonempty(
            fp_heavy_logits,
            fp_heavy_target,
            alpha=0.3,
            beta=0.7,
        )
        precision = asymmetric_tversky_loss_nonempty(
            fp_heavy_logits,
            fp_heavy_target,
            alpha=0.7,
            beta=0.3,
        )
        self.assertGreater(float(precision), float(recall))

    def test_empty_target_has_zero_tversky_auxiliary(self) -> None:
        value = logits([0.9, 0.1])
        loss = asymmetric_tversky_loss_nonempty(
            value,
            torch.zeros_like(value),
            alpha=0.3,
            beta=0.7,
        )
        self.assertEqual(float(loss), 0.0)

    def test_variant_contract(self) -> None:
        self.assertEqual(len(VARIANTS), 4)
        self.assertEqual(
            VARIANTS["v2_dualfusion_precision"]["final_precision_weight"],
            0.1,
        )
        self.assertFalse(
            VARIANTS["v3_dualfusion_softguide_joint"][
                "detach_proposal_guide"
            ]
        )


if __name__ == "__main__":
    unittest.main()
