#!/usr/bin/env python3
"""Evaluate an experiment 005 proposal checkpoint on a fixed dataset JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import write_json
from global_proposal import GlobalProposalUNet
from train_global_proposal import load_embeddings, validate, write_validation_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nms-radius", type=int, default=8)
    args = parser.parse_args()

    entries = json.loads(args.dataset_json.read_text())["test"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = GlobalProposalUNet(
        base_channels=int(checkpoint.get("model", {}).get("base_channels", 8))
    ).to(device)
    model.load_state_dict(checkpoint["network_weights"])
    result = validate(
        model,
        entries,
        args.cache_root,
        load_embeddings(args.embeddings),
        device,
        args.top_k,
        args.nms_radius,
    )
    result["step"] = int(checkpoint.get("step", -1))
    write_json(args.output_json, result)
    write_validation_report(args.output_md, result, int(result["step"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
