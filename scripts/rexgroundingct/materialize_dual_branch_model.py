#!/usr/bin/env python3
"""Materialize an epoch-0 dual-branch model from a VoxTell source checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from voxtell.inference.predictor import VoxTellPredictor

from common import sha256_file, utc_now_iso, write_json
from voxtell_dual_branch import (
    MODEL_SPEC_FILENAME,
    VARIANTS,
    DualBranchVoxTellModel,
    write_model_spec,
)


def assert_state_equal(
    left: torch.nn.Module,
    right: torch.nn.Module,
    label: str,
) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        raise RuntimeError(f"{label}: state keys differ")
    for key in left_state:
        if not torch.equal(left_state[key].cpu(), right_state[key].cpu()):
            raise RuntimeError(f"{label}: copied tensor differs at {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output-model-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    args = parser.parse_args()

    source_checkpoint = (
        args.source_model_dir / "fold_0" / "checkpoint_final.pth"
    )
    source_plans = args.source_model_dir / "plans.json"
    if not source_checkpoint.is_file() or not source_plans.is_file():
        raise FileNotFoundError(
            f"Source model is incomplete under {args.source_model_dir}"
        )

    predictor = VoxTellPredictor(
        model_dir=str(args.source_model_dir),
        device=torch.device("cpu"),
        embedding_bank={},
        use_precomputed_embeddings=False,
    )
    source_model = predictor.network
    network = DualBranchVoxTellModel(
        source_model=source_model,
        variant=args.variant,
        deep_supervision=False,
    )

    assert_state_equal(
        network.proposal_decoder,
        source_model.decoder,
        "proposal decoder",
    )
    assert_state_equal(
        network.refinement_decoder.decoder,
        source_model.decoder,
        "refinement decoder",
    )
    if network.fusion_mode == "dual_adapter":
        assert_state_equal(
            network.proposal_projectors,
            source_model.project_to_decoder_channels,
            "proposal projectors",
        )
        assert_state_equal(
            network.refinement_projectors,
            source_model.project_to_decoder_channels,
            "refinement projectors",
        )

    zero_parameter_names = []
    for name, parameter in network.named_parameters():
        if (
            name.startswith("refinement_decoder.guide_adapters.")
            or name.endswith("fusion_adapter.layers.2.weight")
            or name.endswith("fusion_adapter.layers.2.bias")
        ):
            if torch.count_nonzero(parameter.detach()).item() != 0:
                raise RuntimeError(f"New zero-initialized parameter is nonzero: {name}")
            zero_parameter_names.append(name)

    spec = network.model_spec(source_model_dir=str(args.source_model_dir))
    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = args.output_model_dir / "fold_0"
    fold_dir.mkdir(exist_ok=True)
    shutil.copy2(source_plans, args.output_model_dir / "plans.json")
    write_model_spec(args.output_model_dir / MODEL_SPEC_FILENAME, spec)
    output_checkpoint = fold_dir / "checkpoint_final.pth"
    tmp = output_checkpoint.with_name(
        f".{output_checkpoint.name}.tmp.{os.getpid()}"
    )
    torch.save(
        {
            "epoch": 0,
            "global_update": 0,
            "network_weights": network.state_dict(),
            "experiment": "008_voxtell_dual_branch_proposal_refinement_ablation",
            "model_spec": spec,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
        },
        tmp,
    )
    os.replace(tmp, output_checkpoint)

    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in network.parameters()
        if parameter.requires_grad
    )
    manifest = {
        "created_at_utc": utc_now_iso(),
        "variant": args.variant,
        "model_spec": spec,
        "source_model_dir": str(args.source_model_dir),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_plans_sha256": sha256_file(source_plans),
        "output_model_dir": str(args.output_model_dir),
        "output_checkpoint": str(output_checkpoint),
        "output_checkpoint_sha256": sha256_file(output_checkpoint),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "zero_initialized_parameter_names": zero_parameter_names,
        "copy_audit": "passed",
    }
    write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
